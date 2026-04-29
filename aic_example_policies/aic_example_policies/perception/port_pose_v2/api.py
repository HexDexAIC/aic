"""Controller-facing API: PortPoseSource.

Single entry point for downstream policies. Hides:
  - per-camera detection + PnP
  - best-view selection (Phase 5; v1 = single-camera-only)
  - SE(3) tracker

Usage from VisionInsert:
    src = PortPoseSource(cameras={"left": ..., "center": ..., "right": ...},
                          K_per_cam={...}, dist_per_cam={...},
                          T_tcp_optical_per_cam={...},
                          target_class="sfp_target")
    pose = src.update(images={"left": img_l, "center": img_c, "right": img_r},
                       T_base_tcp=current_tcp_pose)
    if pose.is_tracked:
        controller.go_to(pose.T_base_mouth)
    elif pose.confidence < 0.3:
        # request fresh acquisition or fallback
        ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

from .pnp import PnPConfig, PortPoseEstimate, estimate_pose
from .tracker import SE3Tracker, TrackerConfig, TrackerOutput, TrackerState


@dataclass
class ControllerPose:
    """What the controller should consume."""
    T_base_mouth: Optional[np.ndarray]   # 4x4 SE(3) in base_link, or None if LOST
    confidence: float                     # 0..1
    is_tracked: bool                      # True if current frame had a fresh accepted measurement
    state: str                            # "tracking" / "coasting" / "lost"
    source_camera: Optional[str]          # which cam produced last accepted measurement
    last_reproj_err_px: float             # mean reproj err of last accepted measurement
    coast_count: int
    last_meas_quality: str                # diagnostic: which gate failed (or "ok")


class PortPoseSource:
    def __init__(self,
                  K_per_cam: Dict[str, np.ndarray],
                  dist_per_cam: Dict[str, np.ndarray],
                  T_tcp_optical_per_cam: Dict[str, np.ndarray],
                  detector,                                    # callable: img → list of detections
                  target_class: str = "sfp_target",
                  pnp_cfg: PnPConfig | None = None,
                  tracker_cfg: TrackerConfig | None = None):
        self.K_per_cam = K_per_cam
        self.dist_per_cam = dist_per_cam
        self.T_tcp_optical_per_cam = T_tcp_optical_per_cam
        self.detector = detector
        self.target_class = target_class
        self.pnp_cfg = pnp_cfg or PnPConfig()
        self.tracker = SE3Tracker(tracker_cfg)
        self.last_source_cam: Optional[str] = None
        self.last_reproj_err_px: float = float("nan")

    def update(self, images: Dict[str, np.ndarray],
                T_base_tcp: np.ndarray) -> ControllerPose:
        """Process one frame across all available cameras and return current
        controller-facing pose estimate.
        """
        # 1. Run detector on each camera, collect candidate poses
        candidates = []
        for cam, img in images.items():
            if img is None:
                continue
            dets = self.detector(img, cam=cam)  # detector returns list of dicts
            for det in dets:
                if det["class_name"] != self.target_class:
                    continue
                K = self.K_per_cam[cam]
                dist = self.dist_per_cam.get(cam, np.zeros(5))
                pose_cam = estimate_pose(
                    keypoints_uv=det["kpts_uv"],
                    cls_id=det["cls_id"],
                    bbox_xyxy=det["bbox_xyxy"],
                    confidence=det["confidence"],
                    K=K, dist_coeffs=dist,
                    cfg=self.pnp_cfg,
                )
                if pose_cam.quality_flag != "ok":
                    continue
                # Compose into base_link
                T_base_optical = T_base_tcp @ self.T_tcp_optical_per_cam[cam]
                T_base_mouth = T_base_optical @ pose_cam.T_cam_mouth
                candidates.append({
                    "T_base_mouth": T_base_mouth,
                    "pose_cam": pose_cam,
                    "cam": cam,
                    "score": pose_cam.confidence * pose_cam.bbox_area_px,
                })

        # 2. Best-view selection (v1 simple: by score). Phase-5 will fuse.
        if candidates:
            best = max(candidates, key=lambda c: c["score"])
            self.last_source_cam = best["cam"]
            self.last_reproj_err_px = best["pose_cam"].reprojection_err_px
            tracker_out = self.tracker.update(
                T_meas=best["T_base_mouth"],
                quality="ok",
                confidence=best["pose_cam"].confidence,
                z_cam_m=float(best["pose_cam"].tvec_cam[2]),
                reproj_err_px=float(best["pose_cam"].reprojection_err_px),
            )
        else:
            tracker_out = self.tracker.update(T_meas=None, quality="all_rejected", confidence=0.0)

        return ControllerPose(
            T_base_mouth=tracker_out.T_base_mouth,
            confidence=tracker_out.confidence,
            is_tracked=tracker_out.is_tracked,
            state=tracker_out.state.value,
            source_camera=self.last_source_cam if tracker_out.is_tracked else None,
            last_reproj_err_px=self.last_reproj_err_px,
            coast_count=tracker_out.coast_count,
            last_meas_quality=tracker_out.last_meas_quality,
        )

    def reset(self):
        self.tracker.reset()
        self.last_source_cam = None
        self.last_reproj_err_px = float("nan")
