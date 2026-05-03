"""Vision-driven cable insertion policy — v3_slow (sleeps 10× longer).

Identical to VisionInsert_v3 except the per-step sim-sleep durations are
multiplied by 10. Hypothesis: under aic_eval:latest the sim runs at
real_time_factor ≈ 1.0 vs Keerti's aic_eval:fixed18_patched which appears
to run at ~1/56× real-time (per her PR #4 wall-time numbers). At full
real-time, v3's motion loops fire commands faster than the impedance
controller can physically act on them, so the plug never gets time to
compliantly seat — observed final plug-port distance 3-5 cm in our runs
vs her reported "2 cm short of full seating with partial insertion".
10× longer sleeps give the controller more wall-time per simulated step.

Motion-loop changes vs v3:
                v3            v3_slow
  approach:   30 × 0.15 s     30 × 1.50 s   (4.5 → 45 sim sec)
  descent:    50 × 0.10 s     50 × 1.00 s   (5 → 50 sim sec)
  stabilize:  5 sim sec       30 sim sec
  total:      ~3 real min     ~30 real min
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
    GetObservationCallback, MoveRobotCallback, Policy, SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform

from .port_pose import PortPose6D, _quat_to_R, _R_to_quat, _tf_dict_to_T
from ..perception.port_pose_v2.pnp import PnPConfig, estimate_pose
from ..perception.port_pose_v2.tracker import SE3Tracker, TrackerConfig

WEIGHTS_PATH = Path(os.environ.get(
    "AIC_V1_WEIGHTS",
    str(Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"),
))
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"

_YOLO_MODEL = None


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO(str(WEIGHTS_PATH))
        except Exception as e:
            print(f"VisionInsert_v3_slow: failed to load YOLO from {WEIGHTS_PATH}: {e}")
            _YOLO_MODEL = False
    return _YOLO_MODEL if _YOLO_MODEL else None


def _identity_offset():
    return {"x": 0.0, "y": 0.0, "z": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}


def _load_offset_for(port_type: str) -> dict:
    if not OFFSET_PATH.exists():
        return _identity_offset()
    try:
        return json.loads(OFFSET_PATH.read_text()).get(port_type, _identity_offset())
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
    return arr


def _T_from_tf_dict(d):
    T = np.eye(4)
    T[:3, :3] = _quat_to_R((d["qw"], d["qx"], d["qy"], d["qz"]))
    T[:3, 3] = [d["x"], d["y"], d["z"]]
    return T


def _T_to_tf_dict(T):
    qw, qx, qy, qz = _R_to_quat(T[:3, :3])
    return {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]),
            "qw": float(qw), "qx": float(qx), "qy": float(qy), "qz": float(qz)}


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


# ===== Motion-loop tunables (v3_slow: sleeps 10× v3) =====
APPROACH_ITERS = 30
APPROACH_SLEEP = 1.50        # was 0.15 in v3
DESCENT_ITERS  = 50
DESCENT_SLEEP  = 1.00        # was 0.10 in v3
DESCENT_Z_TOTAL = 0.215      # 0.2 (start z_offset) - (-0.015) (end)
DESCENT_Z_STEP = DESCENT_Z_TOTAL / DESCENT_ITERS   # ~0.0043m per step
STABILIZE_SEC  = 30.0        # was 5 in v3


class VisionInsert_v3_slow(Policy):
    """v3_slow: same perception + iters as v3, sleeps 10× longer."""

    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        super().__init__(parent_node)

        self._dbg_dir = Path.home() / "aic_logs" / "vision_insert_v3_dbg"
        self._dbg_dir.mkdir(parents=True, exist_ok=True)
        self._port_pose: Optional[PortPose6D] = None
        self._tcp_to_plug_T = np.eye(4)
        _get_yolo()

    def _lookup_cam_tf(self, frame_name: str):
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", frame_name, Time())
            return {"x": tf.transform.translation.x,
                    "y": tf.transform.translation.y,
                    "z": tf.transform.translation.z,
                    "qw": tf.transform.rotation.w,
                    "qx": tf.transform.rotation.x,
                    "qy": tf.transform.rotation.y,
                    "qz": tf.transform.rotation.z}
        except TransformException as ex:
            self.get_logger().warn(f"VisionInsert_v3_slow: cannot read {frame_name}: {ex}")
            return None

    def _detect_port_pose_v3(self, get_observation, port_type, max_attempts=20):
        model = _get_yolo()
        if model is None:
            self.get_logger().error("VisionInsert_v3_slow: YOLO model unavailable")
            return None

        pnp_cfg = PnPConfig()
        tracker = SE3Tracker(TrackerConfig())

        for attempt in range(max_attempts):
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.1); continue

            best = None; best_score = 0.0
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
                pose = estimate_pose(det["kpts_uv"], det["cls_id"],
                                       det["bbox_xyxy"], det["confidence"],
                                       K, dist_coeffs, pnp_cfg)
                if pose.quality_flag != "ok":
                    continue
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
                f"VisionInsert_v3_slow attempt {attempt}: "
                f"{'best=' + best[2] if best else 'no detection'}")

            if best is not None:
                T_base_mouth, pose_est, cam_name = best
                tracker.update(T_base_mouth, "ok", pose_est.confidence,
                                z_cam_m=float(pose_est.tvec_cam[2]),
                                reproj_err_px=float(pose_est.reprojection_err_px))
                if tracker.state.value == "tracking" and attempt >= 1:
                    tf_dict = _T_to_tf_dict(tracker._T)
                    self.get_logger().info(
                        f"VisionInsert_v3_slow: locked port pose from {cam_name}, "
                        f"reproj_err={pose_est.reprojection_err_px:.3f}px, "
                        f"port_xyz=({tf_dict['x']:.3f},{tf_dict['y']:.3f},{tf_dict['z']:.3f})")
                    return PortPose6D(transform=tf_dict,
                                       depth_m=float(pose_est.tvec_cam[2]),
                                       score=pose_est.confidence,
                                       method="pnp_v2_ippe")
            else:
                tracker.update(None, "no_detection", 0.0)
            self.sleep_for(0.2)

        if tracker._T is not None:
            tf_dict = _T_to_tf_dict(tracker._T)
            self.get_logger().warn("VisionInsert_v3_slow: returning fallback")
            return PortPose6D(transform=tf_dict, depth_m=0.0, score=0.0,
                                method="pnp_v2_ippe_fallback")
        return None

    def _plug_pose_from_obs(self, obs):
        tp = obs.controller_state.tcp_pose
        T_tcp = np.eye(4)
        T_tcp[:3, :3] = _quat_to_R(
            (tp.orientation.w, tp.orientation.x, tp.orientation.y, tp.orientation.z))
        T_tcp[:3, 3] = [tp.position.x, tp.position.y, tp.position.z]
        T_plug = T_tcp @ self._tcp_to_plug_T
        return T_tcp, T_plug

    def calc_gripper_pose(self, port_transform, latest_obs,
                           slerp_fraction=1.0, position_fraction=1.0,
                           z_offset=0.1, reset_xy_integrator=False):
        q_port = (port_transform.rotation.w, port_transform.rotation.x,
                   port_transform.rotation.y, port_transform.rotation.z)
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
        plug_tip_gripper_offset = (gripper_xyz[0] - plug_xyz[0],
                                     gripper_xyz[1] - plug_xyz[1],
                                     gripper_xyz[2] - plug_xyz[2])

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]
        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = float(np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup, self._max_integrator_windup))
            self._tip_y_error_integrator = float(np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup, self._max_integrator_windup))

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
            orientation=Quaternion(w=q_gripper_slerp[0], x=q_gripper_slerp[1],
                                     y=q_gripper_slerp[2], z=q_gripper_slerp[3]))

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self.get_logger().info(f"VisionInsert_v3_slow.insert_cable() task: {task}")
        self._task = task

        offset = _load_offset_for(task.port_type)
        self._tcp_to_plug_T = _tf_dict_to_T(offset)
        self.get_logger().info(f"VisionInsert_v3_slow: tcp_to_plug = {offset}")

        pose = self._detect_port_pose_v3(get_observation, task.port_type)
        if pose is None:
            self.get_logger().error(
                "VisionInsert_v3_slow: failed to detect port; aborting trial.")
            return False
        self._port_pose = pose
        port_transform = _tf_to_geometry(pose.transform)
        send_feedback(f"port detected at depth {pose.depth_m:.3f} m")

        self.get_logger().info(
            f"VisionInsert_v3_slow: APPROACH ({APPROACH_ITERS} iter × {APPROACH_SLEEP}s sim)")
        # Approach phase — coarser
        z_offset = 0.2
        for t in range(0, APPROACH_ITERS):
            obs = get_observation()
            if obs is None:
                self.sleep_for(APPROACH_SLEEP); continue
            interp_fraction = t / APPROACH_ITERS
            try:
                target_pose = self.calc_gripper_pose(
                    port_transform, obs,
                    slerp_fraction=interp_fraction,
                    position_fraction=interp_fraction,
                    z_offset=z_offset, reset_xy_integrator=True)
                self.set_pose_target(move_robot=move_robot, pose=target_pose)
                if t % 5 == 0:
                    self.get_logger().info(
                        f"VisionInsert_v3_slow approach t={t}/{APPROACH_ITERS}")
            except Exception as ex:
                self.get_logger().error(f"approach t={t}: {ex}")
            self.sleep_for(APPROACH_SLEEP)

        self.get_logger().info(
            f"VisionInsert_v3_slow: DESCENT ({DESCENT_ITERS} iter × {DESCENT_SLEEP}s sim, "
            f"z_step={DESCENT_Z_STEP*1000:.2f}mm)")
        # Descent — coarser
        for d in range(0, DESCENT_ITERS):
            z_offset -= DESCENT_Z_STEP
            obs = get_observation()
            if obs is None:
                self.sleep_for(DESCENT_SLEEP); continue
            try:
                target_pose = self.calc_gripper_pose(
                    port_transform, obs, z_offset=z_offset)
                self.set_pose_target(move_robot=move_robot, pose=target_pose)
                if d % 5 == 0:
                    self.get_logger().info(
                        f"VisionInsert_v3_slow descent d={d}/{DESCENT_ITERS} z_offset={z_offset:.4f}")
            except Exception as ex:
                self.get_logger().error(f"descent: {ex}")
            self.sleep_for(DESCENT_SLEEP)

        self.get_logger().info(f"VisionInsert_v3_slow: STABILIZE ({STABILIZE_SEC}s sim)")
        self.sleep_for(STABILIZE_SEC)
        self.get_logger().info("VisionInsert_v3_slow: insert_cable() done")
        return True
