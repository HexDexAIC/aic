"""Single-view PnP for SFP entry mouth.

Object points: SDF-locked entry-mouth corners + center
   (see aic_assets/models/NIC Card Mount/model.sdf, sfp_port_X_link_entrance)

Solver: cv2.SOLVEPNP_IPPE (planar 4-point optimal, returns 2 candidates)
Disambiguation: pick whichever solution's reprojection of the 5th keypoint
                (port-mouth center) is closest to the predicted center.

Frame-quality gates (configurable; defaults provisional pending Phase-2 eval):
  - confidence:         box.conf >= cfg.conf_min            (default 0.5)
  - on-screen:          all 4 corners ≥ cfg.edge_margin_px  (default 8)
  - bbox area:          area_px >= cfg.bbox_area_min        (default 600 — TUNABLE)
  - bbox aspect:        cfg.aspect_min <= ratio <= cfg.aspect_max (1.0..2.5)
  - PnP residual:       mean_reproj_err_px <= cfg.reproj_max  (default 3.0)
  - center residual:    center_residual_px logged separately, also gated
  - tvec sanity:        cfg.z_min <= z_cam <= cfg.z_max     (0.05..1.0 m)

Outputs PortPoseEstimate with quality_flag describing first failed gate
(or "ok") plus the full quality vector for downstream tracker decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# ----- Object points (SDF-locked entry mouth, port-mouth frame) ------------
SLOT_W_M = 0.0137  # SFP MSA spec
SLOT_H_M = 0.0085

# Canonical port-frame order: (+X+Y, +X-Y, -X-Y, -X+Y, center)
OBJECT_POINTS_4 = np.array([
    [+SLOT_W_M / 2, +SLOT_H_M / 2, 0.0],
    [+SLOT_W_M / 2, -SLOT_H_M / 2, 0.0],
    [-SLOT_W_M / 2, -SLOT_H_M / 2, 0.0],
    [-SLOT_W_M / 2, +SLOT_H_M / 2, 0.0],
], dtype=np.float64)
OBJECT_POINTS_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float64)


@dataclass
class PnPConfig:
    conf_min: float = 0.5
    edge_margin_px: int = 8
    bbox_area_min: int = 600        # provisional — tune from failure investigation
    aspect_min: float = 1.0
    aspect_max: float = 2.5
    reproj_max_px: float = 3.0
    center_residual_max_px: float = 4.0
    z_min_m: float = 0.05
    z_max_m: float = 1.0
    img_w: int = 1152
    img_h: int = 1024


@dataclass
class PortPoseEstimate:
    T_cam_mouth: np.ndarray              # (4, 4) — port mouth in camera frame
    rvec: np.ndarray                      # (3,) Rodrigues
    tvec_cam: np.ndarray                  # (3,) translation in camera frame (m)
    reprojection_err_px: float            # mean over 4 corners
    center_residual_px: float             # 5th-keypoint check (separate from gate)
    confidence: float                     # detector confidence
    n_keypoints_used: int                 # 4 (corners only — center is for disambig)
    quality_flag: str                     # "ok" / first failed gate name
    pose_class: str                       # "sfp_target" / "sfp_distractor"
    bbox_area_px: float
    K: np.ndarray
    dist_coeffs: np.ndarray
    extra: dict = field(default_factory=dict)


def _make_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert rvec + tvec to 4x4 SE(3)."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def estimate_pose(
    keypoints_uv: np.ndarray,         # (5, 2) — predicted [+XY, +X-Y, -X-Y, -X+Y, center]
    cls_id: int,                       # 0=sfp_target, 1=sfp_distractor
    bbox_xyxy: np.ndarray,            # (4,) — predicted bbox
    confidence: float,                 # detection confidence
    K: np.ndarray,                    # (3,3) camera intrinsics
    dist_coeffs: np.ndarray,          # (n,) — read from camera_info; zeros for sim
    cfg: PnPConfig,
) -> PortPoseEstimate:
    """Solve single-view PnP with quality gating.

    Always returns a PortPoseEstimate (never None) — caller checks quality_flag.
    """
    pose_class = "sfp_target" if cls_id == 0 else "sfp_distractor"
    bbox_area_px = float((bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]))

    def _make(quality_flag: str, T=None, rvec=None, tvec=None,
              reproj=float("nan"), center_resid=float("nan")) -> PortPoseEstimate:
        return PortPoseEstimate(
            T_cam_mouth=T if T is not None else np.eye(4),
            rvec=rvec if rvec is not None else np.zeros(3),
            tvec_cam=tvec if tvec is not None else np.zeros(3),
            reprojection_err_px=reproj,
            center_residual_px=center_resid,
            confidence=float(confidence),
            n_keypoints_used=4,
            quality_flag=quality_flag,
            pose_class=pose_class,
            bbox_area_px=bbox_area_px,
            K=K, dist_coeffs=dist_coeffs,
        )

    # ---- Gate 1: confidence
    if confidence < cfg.conf_min:
        return _make("low_conf")

    # ---- Gate 2: bbox area
    if bbox_area_px < cfg.bbox_area_min:
        return _make("small_bbox")

    # ---- Gate 3: bbox aspect
    bw = bbox_xyxy[2] - bbox_xyxy[0]
    bh = bbox_xyxy[3] - bbox_xyxy[1]
    if bh > 0:
        ratio = max(bw, bh) / max(min(bw, bh), 1e-6)
        if not (cfg.aspect_min <= ratio <= cfg.aspect_max):
            return _make("bad_aspect")

    # ---- Gate 4: on-screen check (4 corners + margin)
    corners_uv = keypoints_uv[:4].astype(np.float64)
    m = cfg.edge_margin_px
    on_screen = np.all((corners_uv[:, 0] >= m) & (corners_uv[:, 0] <= cfg.img_w - m) &
                        (corners_uv[:, 1] >= m) & (corners_uv[:, 1] <= cfg.img_h - m))
    if not on_screen:
        return _make("clipped")

    # ---- Solve PnP IPPE (returns up to 2 candidate solutions)
    try:
        retval, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            objectPoints=OBJECT_POINTS_4,
            imagePoints=corners_uv.reshape(-1, 1, 2),
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return _make("pnp_solver_failed")

    if retval == 0 or rvecs is None or len(rvecs) == 0:
        return _make("pnp_no_solution")

    # ---- Disambiguate IPPE candidates.
    # KEY INSIGHT: IPPE returns the two ambiguous solutions sorted by per-corner
    # reprojection error. The CORRECT solution is consistently the lower-err one;
    # the FLIPPED solution has 3-5× higher reproj error in practice.
    # Using the center keypoint for disambiguation FAILS because the center is on
    # the IPPE symmetry plane and projects identically under both solutions.
    # We log center_residual_px for diagnostics but disambiguate on IPPE reproj_err.
    center_pred = keypoints_uv[4].astype(np.float64)
    best_idx = 0     # IPPE returns solutions sorted ascending by reproj err
    best_reproj = float(reproj_errs[0].ravel()[0]) if reproj_errs is not None else float("nan")
    proj_center, _ = cv2.projectPoints(
        OBJECT_POINTS_CENTER.reshape(1, 3),
        rvecs[0], tvecs[0], K, dist_coeffs)
    best_center_resid = float(np.linalg.norm(proj_center.ravel() - center_pred))

    rvec = rvecs[best_idx].ravel()
    tvec = tvecs[best_idx].ravel()
    T = _make_T(rvec, tvec)
    z_cam = float(tvec[2])

    # ---- Gate 5: tvec sanity
    if not (cfg.z_min_m <= z_cam <= cfg.z_max_m):
        return _make("tvec_out_of_range", T=T, rvec=rvec, tvec=tvec,
                      reproj=best_reproj, center_resid=best_center_resid)

    # ---- Gate 6: PnP residual
    if best_reproj > cfg.reproj_max_px:
        return _make("high_reproj", T=T, rvec=rvec, tvec=tvec,
                      reproj=best_reproj, center_resid=best_center_resid)

    # ---- Gate 7: center residual (separate from main reproj — catches bad
    # disambiguation or bad center keypoint)
    if best_center_resid > cfg.center_residual_max_px:
        return _make("high_center_resid", T=T, rvec=rvec, tvec=tvec,
                      reproj=best_reproj, center_resid=best_center_resid)

    return _make("ok", T=T, rvec=rvec, tvec=tvec,
                  reproj=best_reproj, center_resid=best_center_resid)


def estimate_pose_in_base(
    pose_cam: PortPoseEstimate,
    T_base_cam: np.ndarray,           # (4, 4) — camera optical → base_link
) -> Optional[np.ndarray]:
    """Compose the camera-frame pose with cam→base transform.
    Returns T_base_mouth (4, 4) or None if upstream pose was rejected.
    """
    if pose_cam.quality_flag != "ok":
        return None
    return T_base_cam @ pose_cam.T_cam_mouth
