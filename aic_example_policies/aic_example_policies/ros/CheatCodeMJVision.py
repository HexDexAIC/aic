"""CheatCodeMJVision — CheatCodeMJ with vision-derived port pose.

Replaces the 3 port-frame /tf lookups in the parent's insert_cable() with
a YOLO + PnP-IPPE + SE3Tracker pipeline. Plug TF lookups remain unchanged
(still require `ground_truth:=true`); this experiment isolates whether
vision-based port localization alone is sufficient when CheatCodeMJ's
tuned motion + retry logic + plug TF stay intact.

Architecture:
  - On each `_get_port_pose(task)` call, run a fresh acquisition burst:
      → up to N attempts at 5 Hz
      → each attempt runs YOLO on left/center/right cam, picks best by
        (conf × bbox_area), runs IPPE PnP, feeds the result into a fresh
        SE3Tracker. Returns once tracker reaches TRACKING for 2+ attempts.
  - Apply +0.0037 m Z offset (in port_link's local frame) to convert from
    YOLO's "visible mouth" target back to the SDF port_link convention
    that the parent CheatCodeMJ was tuned against.

Frame conventions (matches port_pose_v2):
  - YOLO outputs T_cam_visible_mouth (port-mouth corners as defined in
    pnp.py's OBJECT_POINTS_4: SFP MSA spec 13.7×8.5 mm).
  - The visible mouth feature targeted by the YOLO labels is empirically
    ~3.7 mm above (in port_link −Z) the SDF `_link_entrance` link, which
    is itself 45.8 mm in port_link −Z below `_link`. So:
        T_visible_mouth_to_port_link[2,3] = -0.0421 m  (in mouth frame's z)
        equivalently:
        T_port_link_to_visible_mouth[2,3] = -0.0421 m  (in port_link's z)
    See aic_wiki/wiki/findings/yolo-pnp-pose-accuracy-offline.md.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from geometry_msgs.msg import Transform

from aic_example_policies.ros.CheatCodeMJ import CheatCodeMJ
from aic_example_policies.perception.port_pose_v2.pnp import (
    PnPConfig, estimate_pose,
)
from aic_example_policies.perception.port_pose_v2.tracker import (
    SE3Tracker, TrackerConfig,
)


# Empirical visible-mouth → port_link offset in port_link frame.
# We add +0.0037 m to the *visible mouth* z to land on port_link z.
# Equivalently in the mouth frame, port_link is at +0.0421 m in mouth z.
VISIBLE_MOUTH_TO_PORT_LINK_DZ = +0.0037  # meters, in port_link frame


WEIGHTS_PATH = Path(os.environ.get(
    "AIC_V1_WEIGHTS",
    str(Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"),
))


_YOLO_MODEL = None


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO(str(WEIGHTS_PATH))
        except Exception as e:
            print(f"CheatCodeMJVision: failed to load YOLO from {WEIGHTS_PATH}: {e}")
            _YOLO_MODEL = False
    return _YOLO_MODEL if _YOLO_MODEL else None


def _ros_image_to_numpy(msg) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return arr


def _quat_to_R(q):
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)


def _R_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def _detect_target_in_image(model, image_rgb, conf_min: float = 0.25):
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(img_bgr, imgsz=1280, conf=conf_min, verbose=False)
    r = results[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return None
    kpts = r.keypoints.xy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    bboxes = r.boxes.xyxy.cpu().numpy()
    target_idx = next((j for j in range(len(cls_ids)) if cls_ids[j] == 0), None)
    if target_idx is None:
        return None
    return {"kpts_uv": kpts[target_idx], "bbox_xyxy": bboxes[target_idx],
             "confidence": float(conf[target_idx]), "cls_id": 0}


class CheatCodeMJVision(CheatCodeMJ):
    """CheatCodeMJ with port pose from YOLO + PnP + SE3Tracker.

    Parent's plug TF lookups are unchanged (still requires ground_truth=true
    for plug). Only the port pose is replaced.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        # Configurable per-call acquisition budget (parameterizable later).
        self._vision_max_attempts = 20
        self._vision_attempt_sleep = 0.2
        self._pnp_cfg = PnPConfig()
        self._tracker_cfg = TrackerConfig()
        # Trigger eager YOLO load so the first _get_port_pose() doesn't pay
        # the ~3 sec model-load latency in the middle of a trial.
        _get_yolo()

    # ------------------------------------------------------------------
    # Override: get port pose from vision instead of TF
    # ------------------------------------------------------------------
    def _get_port_pose(self, task, get_observation=None) -> Transform:
        """Run a fresh YOLO+PnP+tracker acquisition burst per call.

        On success, returns a geometry_msgs/Transform (port_link in base_link)
        with the visible-mouth → port_link Z offset applied.

        On total failure (no valid pose after max_attempts), raises
        TransformException so the caller's existing error path runs.
        """
        if get_observation is None:
            raise TransformException(
                "CheatCodeMJVision: get_observation callback not provided"
            )
        model = _get_yolo()
        if model is None:
            raise TransformException(
                f"CheatCodeMJVision: YOLO model unavailable at {WEIGHTS_PATH}"
            )

        tracker = SE3Tracker(self._tracker_cfg)
        last_T_meas = None
        last_pose = None
        last_cam = None

        for attempt in range(self._vision_max_attempts):
            obs = get_observation()
            if obs is None:
                self.sleep_for(self._vision_attempt_sleep)
                continue

            best = None
            best_score = 0.0
            for cam_name, img_msg, K_msg in (
                ("left", obs.left_image, obs.left_camera_info),
                ("center", obs.center_image, obs.center_camera_info),
                ("right", obs.right_image, obs.right_camera_info),
            ):
                if not img_msg.data:
                    continue
                img_rgb = _ros_image_to_numpy(img_msg)
                det = _detect_target_in_image(model, img_rgb)
                if det is None:
                    continue
                K = np.array(list(K_msg.k)).reshape(3, 3)
                dist_coeffs = np.array(list(K_msg.d)) if K_msg.d else np.zeros(5)
                pose = estimate_pose(
                    det["kpts_uv"], det["cls_id"], det["bbox_xyxy"],
                    det["confidence"], K, dist_coeffs, self._pnp_cfg,
                )
                if pose.quality_flag != "ok":
                    continue
                cam_tf = self._lookup_cam_tf(f"{cam_name}_camera/optical")
                if cam_tf is None:
                    continue
                T_base_cam = self._tf_dict_to_T(cam_tf)
                T_base_visible = T_base_cam @ pose.T_cam_mouth
                score = float(pose.confidence * pose.bbox_area_px)
                if score > best_score:
                    best_score = score
                    best = (T_base_visible, pose, cam_name)

            if best is not None:
                T_base_visible, pose, cam_name = best
                tracker.update(
                    T_meas=T_base_visible, quality="ok",
                    confidence=float(pose.confidence),
                    z_cam_m=float(pose.tvec_cam[2]),
                    reproj_err_px=float(pose.reprojection_err_px),
                )
                last_T_meas = T_base_visible
                last_pose = pose
                last_cam = cam_name
                # Commit once tracker is TRACKING for ≥2 attempts.
                if tracker.state.value == "tracking" and attempt >= 1:
                    T_visible_locked = tracker._T  # noqa: SLF001
                    T_port_link = self._visible_mouth_to_port_link(T_visible_locked)
                    self.get_logger().info(
                        f"CheatCodeMJVision: locked port pose from {cam_name} "
                        f"after {attempt+1} attempts; reproj={pose.reprojection_err_px:.2f}px, "
                        f"port_link xyz=({T_port_link[0,3]:.3f}, "
                        f"{T_port_link[1,3]:.3f}, {T_port_link[2,3]:.3f})"
                    )
                    return self._T_to_geometry_transform(T_port_link)
            else:
                tracker.update(T_meas=None, quality="no_detection", confidence=0.0)

            self.sleep_for(self._vision_attempt_sleep)

        # Fallback: tracker has SOMETHING from at least one attempt
        if tracker._T is not None:  # noqa: SLF001
            T_visible_fallback = tracker._T  # noqa: SLF001
            T_port_link = self._visible_mouth_to_port_link(T_visible_fallback)
            self.get_logger().warn(
                f"CheatCodeMJVision: tracker never reached TRACKING after "
                f"{self._vision_max_attempts} attempts; using fallback pose."
            )
            return self._T_to_geometry_transform(T_port_link)

        raise TransformException(
            f"CheatCodeMJVision: no valid port pose after "
            f"{self._vision_max_attempts} attempts (best_cam={last_cam})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _lookup_cam_tf(self, frame_name: str):
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", frame_name, Time()
            )
            return {"x": tf.transform.translation.x,
                    "y": tf.transform.translation.y,
                    "z": tf.transform.translation.z,
                    "qw": tf.transform.rotation.w,
                    "qx": tf.transform.rotation.x,
                    "qy": tf.transform.rotation.y,
                    "qz": tf.transform.rotation.z}
        except TransformException as ex:
            self.get_logger().warn(
                f"CheatCodeMJVision: cannot read {frame_name}: {ex}"
            )
            return None

    @staticmethod
    def _tf_dict_to_T(d):
        T = np.eye(4)
        T[:3, :3] = _quat_to_R((d["qw"], d["qx"], d["qy"], d["qz"]))
        T[:3, 3] = [d["x"], d["y"], d["z"]]
        return T

    @staticmethod
    def _visible_mouth_to_port_link(T_base_visible: np.ndarray) -> np.ndarray:
        """Convert a YOLO 'visible mouth' pose to the SDF port_link pose.

        YOLO targets a feature ~3.7 mm above the SDF entrance link in port_link's
        local Z. Adding +0.0037 m in port_link's Z (which is the same direction
        as the visible-mouth's Z because the rotation is shared) recovers the
        port_link pose used by CheatCodeMJ's tunings.
        """
        T_offset = np.eye(4)
        T_offset[2, 3] = VISIBLE_MOUTH_TO_PORT_LINK_DZ
        return T_base_visible @ T_offset

    @staticmethod
    def _T_to_geometry_transform(T: np.ndarray) -> Transform:
        out = Transform()
        out.translation.x = float(T[0, 3])
        out.translation.y = float(T[1, 3])
        out.translation.z = float(T[2, 3])
        qw, qx, qy, qz = _R_to_quat(T[:3, :3])
        out.rotation.w = qw
        out.rotation.x = qx
        out.rotation.y = qy
        out.rotation.z = qz
        return out
