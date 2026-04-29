"""Vision-driven cable insertion policy — v2 (port_pose_v2 perception).

Same approach + descent logic as VisionInsert v1, but replaces the old
detector pipeline (port_detector + port_pose) with the new perception
stack (port_pose_v2: YOLO keypoints + IPPE PnP + SE(3) tracker).

This is Phase 6.robot integration — a one-shot detection at trial start
using the better perception module. Closed-loop per-iteration perception
is a future improvement (would require restructuring the descent loop).
"""

import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform

from .port_pose import PortPose6D, _quat_to_R, _R_to_quat, _tf_dict_to_T

# Phase-2 perception modules
from ..perception.port_pose_v2.pnp import PnPConfig, estimate_pose
from ..perception.port_pose_v2.tracker import SE3Tracker, TrackerConfig


# Path to trained YOLO weights inside the eval container. Override via env.
WEIGHTS_PATH = Path(os.environ.get(
    "AIC_V1_WEIGHTS",
    str(Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"),
))
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"


# Lazy module-level YOLO model (load once across trials)
_YOLO_MODEL = None


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO(str(WEIGHTS_PATH))
        except Exception as e:
            print(f"VisionInsert_v2: failed to load YOLO from {WEIGHTS_PATH}: {e}")
            _YOLO_MODEL = False
    return _YOLO_MODEL if _YOLO_MODEL else None


def _identity_offset():
    return {"x": 0.0, "y": 0.0, "z": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}


def _load_offset_for(port_type: str) -> dict:
    if not OFFSET_PATH.exists():
        return _identity_offset()
    try:
        return json.loads(OFFSET_PATH.read_text()).get(
            port_type, _identity_offset())
    except Exception:
        return _identity_offset()


def _tf_to_geometry(tf_dict) -> Transform:
    t = Transform()
    t.translation.x = float(tf_dict["x"])
    t.translation.y = float(tf_dict["y"])
    t.translation.z = float(tf_dict["z"])
    t.rotation.w = float(tf_dict["qw"])
    t.rotation.x = float(tf_dict["qx"])
    t.rotation.y = float(tf_dict["qy"])
    t.rotation.z = float(tf_dict["qz"])
    return t


def _ros_image_to_numpy(msg) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return arr  # RGB


def _T_from_tf_dict(d):
    T = np.eye(4)
    T[:3, :3] = _quat_to_R((d["qw"], d["qx"], d["qy"], d["qz"]))
    T[:3, 3] = [d["x"], d["y"], d["z"]]
    return T


def _T_to_tf_dict(T):
    qw, qx, qy, qz = _R_to_quat(T[:3, :3])
    return {
        "x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]),
        "qw": float(qw), "qx": float(qx), "qy": float(qy), "qz": float(qz),
    }


def _detect_target_in_image(model, image_rgb, conf_min: float = 0.25):
    """Run YOLO on one image; return target detection dict or None.

    Returns dict with keys: kpts_uv (5,2), bbox_xyxy (4,), confidence, cls_id (0).
    """
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(img_bgr, imgsz=1280, conf=conf_min, verbose=False)
    r = results[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return None
    kpts = r.keypoints.xy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()
    bboxes = r.boxes.xyxy.cpu().numpy()
    # sfp_target = class 0
    target_idx = next((j for j in range(len(cls_ids)) if cls_ids[j] == 0), None)
    if target_idx is None:
        return None
    return {
        "kpts_uv": kpts[target_idx],
        "bbox_xyxy": bboxes[target_idx],
        "confidence": float(conf[target_idx]),
        "cls_id": 0,
    }


class VisionInsert_v2(Policy):
    """v2: same motion plan as v1 but uses port_pose_v2 perception."""

    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        super().__init__(parent_node)

        self._dbg_dir = Path.home() / "aic_logs" / "vision_insert_v2_dbg"
        self._dbg_dir.mkdir(parents=True, exist_ok=True)

        self._port_pose: Optional[PortPose6D] = None
        self._tcp_to_plug_T = np.eye(4)

        # Pre-load YOLO model so first trial doesn't pay the cost
        _get_yolo()

    def _lookup_cam_tf(self, frame_name: str):
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", frame_name, Time())
            return {
                "x": tf.transform.translation.x,
                "y": tf.transform.translation.y,
                "z": tf.transform.translation.z,
                "qw": tf.transform.rotation.w,
                "qx": tf.transform.rotation.x,
                "qy": tf.transform.rotation.y,
                "qz": tf.transform.rotation.z,
            }
        except TransformException as ex:
            self.get_logger().warn(f"VisionInsert_v2: cannot read {frame_name}: {ex}")
            return None

    def _detect_port_pose_v2(
        self,
        get_observation: GetObservationCallback,
        port_type: str,
        max_attempts: int = 20,
    ) -> Optional[PortPose6D]:
        """Detect port using port_pose_v2 perception stack.

        Strategy: try all 3 cameras each attempt, run PnP per cam, pick
        the highest-confidence cam (largest bbox * confidence). Across
        attempts, feed into an SE(3) tracker that smooths and rejects
        outliers.
        """
        model = _get_yolo()
        if model is None:
            self.get_logger().error("VisionInsert_v2: YOLO model unavailable")
            return None

        pnp_cfg = PnPConfig()
        tracker = SE3Tracker(TrackerConfig())

        for attempt in range(max_attempts):
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.1)
                continue

            best = None  # (T_base_mouth, pose_estimate, cam_name)
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
                # Distortion coeffs: read from CameraInfo (per spec adjustment #1)
                dist_coeffs = np.array(list(K_msg.d)) if K_msg.d else np.zeros(5)

                pose = estimate_pose(
                    keypoints_uv=det["kpts_uv"],
                    cls_id=det["cls_id"],
                    bbox_xyxy=det["bbox_xyxy"],
                    confidence=det["confidence"],
                    K=K, dist_coeffs=dist_coeffs,
                    cfg=pnp_cfg,
                )
                if pose.quality_flag != "ok":
                    continue

                # Compose to base_link via cam→base lookup
                cam_tf = self._lookup_cam_tf(f"{cam_name}_camera/optical")
                if cam_tf is None:
                    continue
                T_base_cam = _T_from_tf_dict(cam_tf)
                T_base_mouth = T_base_cam @ pose.T_cam_mouth

                score = pose.confidence * pose.bbox_area_px
                if score > best_score:
                    best_score = score
                    best = (T_base_mouth, pose, cam_name)

            self.get_logger().info(
                f"VisionInsert_v2 attempt {attempt}: "
                f"{'best=' + best[2] if best else 'no detection'}"
            )

            if best is not None:
                T_base_mouth, pose_est, cam_name = best
                tracker.update(
                    T_meas=T_base_mouth,
                    quality="ok",
                    confidence=pose_est.confidence,
                    z_cam_m=float(pose_est.tvec_cam[2]),
                    reproj_err_px=float(pose_est.reprojection_err_px),
                )
                # If tracker has settled (>=2 measurements), commit.
                if tracker.state.value == "tracking" and attempt >= 1:
                    tf_dict = _T_to_tf_dict(tracker._T)
                    self.get_logger().info(
                        f"VisionInsert_v2: locked port pose from "
                        f"{cam_name}, reproj_err={pose_est.reprojection_err_px:.3f}px, "
                        f"port_xyz=({tf_dict['x']:.3f},{tf_dict['y']:.3f},{tf_dict['z']:.3f})"
                    )
                    return PortPose6D(
                        transform=tf_dict,
                        depth_m=float(pose_est.tvec_cam[2]),
                        score=pose_est.confidence,
                        method="pnp_v2_ippe",
                    )
            else:
                tracker.update(None, "no_detection", 0.0)

            self.sleep_for(0.2)

        # Fallback: if tracker has anything at all, use it
        if tracker._T is not None:
            tf_dict = _T_to_tf_dict(tracker._T)
            self.get_logger().warn(
                "VisionInsert_v2: returning tracker estimate after max attempts"
            )
            return PortPose6D(transform=tf_dict, depth_m=0.0, score=0.0,
                                method="pnp_v2_ippe_fallback")
        return None

    # --- The rest is the same as VisionInsert v1 (motion plan, descent, etc.) ---
    def _plug_pose_from_obs(self, obs):
        tp = obs.controller_state.tcp_pose
        T_tcp = np.eye(4)
        T_tcp[:3, :3] = _quat_to_R(
            (tp.orientation.w, tp.orientation.x, tp.orientation.y, tp.orientation.z))
        T_tcp[:3, 3] = [tp.position.x, tp.position.y, tp.position.z]
        T_plug = T_tcp @ self._tcp_to_plug_T
        return T_tcp, T_plug

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        latest_obs: Observation,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        q_port = (
            port_transform.rotation.w, port_transform.rotation.x,
            port_transform.rotation.y, port_transform.rotation.z,
        )
        T_tcp, T_plug = self._plug_pose_from_obs(latest_obs)
        gripper_xyz = (T_tcp[0, 3], T_tcp[1, 3], T_tcp[2, 3])
        plug_xyz = (T_plug[0, 3], T_plug[1, 3], T_plug[2, 3])

        qw_g, qx_g, qy_g, qz_g = _R_to_quat(T_tcp[:3, :3])
        qw_p, qx_p, qy_p, qz_p = _R_to_quat(T_plug[:3, :3])
        q_plug_inv = (-qw_p, qx_p, qy_p, qz_p)
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        q_gripper = (qw_g, qx_g, qy_g, qz_g)
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        port_xy = (port_transform.translation.x, port_transform.translation.y)
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]
        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = float(np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup, self._max_integrator_windup,
            ))
            self._tip_y_error_integrator = float(np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup, self._max_integrator_windup,
            ))

        i_gain = 0.15
        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(x=blend_xyz[0], y=blend_xyz[1], z=blend_xyz[2]),
            orientation=Quaternion(
                w=q_gripper_slerp[0], x=q_gripper_slerp[1],
                y=q_gripper_slerp[2], z=q_gripper_slerp[3]),
        )

    def insert_cable(
        self, task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"VisionInsert_v2_class.insert_cable() task: {task}")
        self._task = task

        offset = _load_offset_for(task.port_type)
        self._tcp_to_plug_T = _tf_dict_to_T(offset)
        self.get_logger().info(f"VisionInsert_v2: tcp_to_plug = {offset}")

        pose = self._detect_port_pose_v2(get_observation, task.port_type)
        if pose is None:
            self.get_logger().error(
                "VisionInsert_v2: failed to detect port; aborting trial.")
            return False
        self._port_pose = pose
        port_transform = _tf_to_geometry(pose.transform)
        send_feedback(f"port detected at depth {pose.depth_m:.3f} m")

        # Approach phase
        z_offset = 0.2
        for t in range(0, 100):
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.05); continue
            interp_fraction = t / 100.0
            try:
                target_pose = self.calc_gripper_pose(
                    port_transform, obs,
                    slerp_fraction=interp_fraction,
                    position_fraction=interp_fraction,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                )
                self.set_pose_target(move_robot=move_robot, pose=target_pose)
            except Exception as ex:
                self.get_logger().error(f"approach t={t}: {ex}")
            self.sleep_for(0.05)

        # Descent
        descent_step = 0
        while z_offset >= -0.015:
            z_offset -= 0.0005
            descent_step += 1
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.05); continue
            try:
                target_pose = self.calc_gripper_pose(
                    port_transform, obs, z_offset=z_offset)
                self.set_pose_target(move_robot=move_robot, pose=target_pose)
            except Exception as ex:
                self.get_logger().error(f"descent: {ex}")
            self.sleep_for(0.05)

        self.get_logger().info("VisionInsert_v2: stabilizing...")
        self.sleep_for(5.0)
        return True
