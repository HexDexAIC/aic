"""SE(3) tracker for SFP entry-mouth pose in base_link.

Assumes target is STATIC (port doesn't move in base_link). So the tracker is
about smoothing measurement noise and gracefully handling dropouts — not a
motion model.

State machine:
  TRACKING  — fresh measurement passes gates AND not an outlier → blend in
  COASTING  — fresh measurement rejected → output last good pose for ≤N frames
  LOST      — coasting exceeded N frames → no output

Smoothing:
  translation: exponential smoothing,   p_new = α·p_meas + (1-α)·p_track
  rotation:    spherical-linear interp, R_new = slerp(R_track, R_meas, α)

Outlier rejection:
  - reject if ||p_meas - p_track|| > k·σ_t  (Mahalanobis-style, k tunable)
  - reject if angle(R_meas, R_track) > k·σ_r
  - σ_t, σ_r are running estimates of measurement noise (init from Phase 2 eval)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class TrackerState(Enum):
    TRACKING = "tracking"
    COASTING = "coasting"
    LOST = "lost"


@dataclass
class TrackerConfig:
    alpha_t: float = 0.4               # smoothing factor on translation
    alpha_r: float = 0.3               # smoothing factor on rotation (slerp t)
    coast_max_frames: int = 10         # frames before LOST
    sigma_t_init_m: float = 0.005      # 5mm initial xyz σ (will be updated)
    sigma_r_init_rad: float = 0.5      # ~30° initial rotation σ (tuned: PnP rot is bimodal at distance)
    outlier_k_t: float = 4.0           # k·σ_t — strict on translation (it's reliable)
    outlier_k_r: float = 6.0           # k·σ_r — looser on rotation (PnP IPPE flips happen)
    rot_hard_reject_rad: float = 1.57  # 90° — sanity threshold; a >90° rot flip is always bogus
    sigma_decay: float = 0.95          # how fast running σ adapts (closer to 1 = slower)
    # Lazy-seed gate — wait for a high-quality measurement before first init.
    # IPPE rotation is bimodal at far distance; seeding the tracker with a
    # flipped pose corrupts the rest of the trajectory. Only seed when:
    #   z_cam < seed_z_max_m  (close enough that rotation is reliable)  OR
    #   reproj_err < seed_reproj_max_px  (very low residual = strong solution)
    seed_z_max_m: float = 0.15         # 15 cm — below this, IPPE rotation is reliable
    seed_reproj_max_px: float = 0.5    # very low residual = strong PnP solution
    seed_max_wait_frames: int = 20     # cap how long we wait for a good seed


@dataclass
class TrackerOutput:
    T_base_mouth: Optional[np.ndarray]  # (4, 4) or None if LOST
    state: TrackerState
    confidence: float
    is_tracked: bool                     # True if state == TRACKING
    coast_count: int
    last_meas_quality: str               # mirrors PortPoseEstimate.quality_flag
    sigma_t_m: float                     # current running σ on translation (m)
    sigma_r_rad: float                   # current running σ on rotation (rad)


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 R → unit quaternion (qw, qx, qy, qz)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = (tr + 1.0) ** 0.5 * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = ((1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = ((1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = ((1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz])
    return q / max(np.linalg.norm(q), 1e-9)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def _quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    dot = float(np.dot(q1, q2))
    if dot < 0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        return (q1 * (1 - t) + q2 * t) / np.linalg.norm(q1 * (1 - t) + q2 * t)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    s1 = np.sin(theta_0 - theta) / sin_theta_0
    s2 = np.sin(theta) / sin_theta_0
    out = s1 * q1 + s2 * q2
    return out / np.linalg.norm(out)


def _angle_between_R(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic distance between two rotations (rad)."""
    R_rel = R1.T @ R2
    cos_th = (np.trace(R_rel) - 1) / 2
    return float(np.arccos(np.clip(cos_th, -1, 1)))


class SE3Tracker:
    """Per-class single-target tracker. Construct one per (camera, class) or
    one per class for a fused-multi-cam tracker (Phase 5).
    """

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self._T: Optional[np.ndarray] = None
        self._sigma_t = self.cfg.sigma_t_init_m
        self._sigma_r = self.cfg.sigma_r_init_rad
        self._state = TrackerState.LOST
        self._coast = 0
        self._last_quality = "init"
        self._waiting_seed_count = 0    # frames spent waiting for a good seed

    @property
    def state(self) -> TrackerState:
        return self._state

    def update(self, T_meas: Optional[np.ndarray], quality: str = "ok",
                confidence: float = 1.0,
                z_cam_m: Optional[float] = None,
                reproj_err_px: Optional[float] = None) -> TrackerOutput:
        """Ingest a fresh measurement (or None for missing). Returns latest pose.

        quality is the PnP module's quality_flag — used to decide whether the
        measurement is even a candidate for the tracker.

        z_cam_m and reproj_err_px (optional) are used by the lazy-seed gate
        to avoid initializing with an IPPE-flipped pose at distance.
        """
        self._last_quality = quality

        # Reject the measurement outright if PnP gates failed
        if T_meas is None or quality != "ok":
            if self._T is None:
                # Still waiting for first seed; just bump counter
                self._waiting_seed_count += 1
                if self._waiting_seed_count > self.cfg.seed_max_wait_frames:
                    # Cap reached — relax: accept whatever we got next time
                    pass
            return self._coast_or_lose(confidence=0.0)

        # ---- Lazy seed gate ----
        # If first valid measurement, only seed if it's high-quality.
        if self._T is None:
            seed_ok = self._waiting_seed_count >= self.cfg.seed_max_wait_frames
            if z_cam_m is not None and z_cam_m < self.cfg.seed_z_max_m:
                seed_ok = True
            if reproj_err_px is not None and reproj_err_px < self.cfg.seed_reproj_max_px:
                seed_ok = True
            if not seed_ok:
                self._waiting_seed_count += 1
                # Stay in LOST until a high-quality seed arrives
                self._state = TrackerState.LOST
                return self._make_output(0.0)
            self._T = T_meas.copy()
            self._state = TrackerState.TRACKING
            self._coast = 0
            self._waiting_seed_count = 0
            return self._make_output(confidence)

        # Check outlier — split decision: translation must pass, rotation
        # gets relaxed treatment because PnP IPPE has bimodal rotation at distance.
        dt = float(np.linalg.norm(T_meas[:3, 3] - self._T[:3, 3]))
        dr = _angle_between_R(self._T[:3, :3], T_meas[:3, :3])

        if dt > self.cfg.outlier_k_t * self._sigma_t:
            # Translation is the reliable signal — if it's outlier, reject the whole frame
            return self._coast_or_lose(confidence=0.0)

        # Rotation outlier handling: 3 cases
        # 1. dr below k_r·σ_r → smooth normally
        # 2. dr above k_r·σ_r but below hard_reject → translation accepted, rotation kept (no slerp)
        # 3. dr above hard_reject (>90°) → reject whole frame (impossible IPPE flip)
        rot_is_outlier = dr > self.cfg.outlier_k_r * self._sigma_r
        rot_is_impossible = dr > self.cfg.rot_hard_reject_rad

        if rot_is_impossible:
            return self._coast_or_lose(confidence=0.0)

        # Smooth translation always (passed gate)
        T_smoothed = np.eye(4)
        T_smoothed[:3, 3] = (self.cfg.alpha_t * T_meas[:3, 3] +
                              (1 - self.cfg.alpha_t) * self._T[:3, 3])

        # Smooth rotation only if not outlier
        if rot_is_outlier:
            T_smoothed[:3, :3] = self._T[:3, :3]   # keep existing rotation
        else:
            q_track = _R_to_quat(self._T[:3, :3])
            q_meas  = _R_to_quat(T_meas[:3, :3])
            q_new   = _quat_slerp(q_track, q_meas, self.cfg.alpha_r)
            T_smoothed[:3, :3] = _quat_to_R(q_new)

        # Update running σ (geometric decay)
        self._sigma_t = max(self.cfg.sigma_decay * self._sigma_t +
                            (1 - self.cfg.sigma_decay) * dt,
                            1e-4)
        self._sigma_r = max(self.cfg.sigma_decay * self._sigma_r +
                            (1 - self.cfg.sigma_decay) * dr,
                            1e-4)

        self._T = T_smoothed
        self._state = TrackerState.TRACKING
        self._coast = 0
        return self._make_output(confidence)

    def _coast_or_lose(self, confidence: float) -> TrackerOutput:
        if self._T is None:
            self._state = TrackerState.LOST
            return self._make_output(0.0)

        self._coast += 1
        if self._coast > self.cfg.coast_max_frames:
            self._state = TrackerState.LOST
            self._T = None
            return self._make_output(0.0)
        self._state = TrackerState.COASTING
        return self._make_output(confidence * 0.5)  # halved during coast

    def _make_output(self, confidence: float) -> TrackerOutput:
        return TrackerOutput(
            T_base_mouth=None if self._state == TrackerState.LOST else self._T.copy(),
            state=self._state,
            confidence=float(confidence),
            is_tracked=self._state == TrackerState.TRACKING,
            coast_count=self._coast,
            last_meas_quality=self._last_quality,
            sigma_t_m=float(self._sigma_t),
            sigma_r_rad=float(self._sigma_r),
        )

    def reset(self):
        self._T = None
        self._state = TrackerState.LOST
        self._coast = 0
        self._waiting_seed_count = 0
        self._sigma_t = self.cfg.sigma_t_init_m
        self._sigma_r = self.cfg.sigma_r_init_rad
        self._last_quality = "init"
