#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#  Licensed under the Apache License, Version 2.0 (the "License");
#
"""Vision-driven cable insertion policy.

Same motion logic as CheatCode (line-up over the port, descend, compliant
finish) but the port pose comes from a vision pipeline on the center wrist
camera, and the plug-tip pose comes from `tcp_pose * tcp_to_plug_offset`
instead of the ground-truth TF tree. Submittable: does not require
`ground_truth:=true`.

Pipeline:
  1. Wait for first /observations message.
  2. Detect port in center_image -> 6D port pose in base_link.
  3. Use the same orientation-aligned descent as CheatCode, but read
     latest TCP pose from observations and apply the constant plug offset.

Calibration:
  - SFP / SC port physical sizes are hardcoded in port_pose.py.
  - tcp -> plug_tip offset is read from
        ~/aic_logs/tcp_to_plug_offset.json
    (per port_type, written by scripts/analyze_logs.py from a
    ground-truth run). Falls back to identity if missing.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

import cv2
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

from .port_detector import detect_port as detect_port_classical, draw_detection
from .port_detector_yolo import YoloPosePortDetector
from .port_pose import (
    PortPose6D,
    _quat_to_R,
    _R_to_quat,
    _T_to_tf_dict,
    _tf_dict_to_T,
    lift_to_base,
    lift_pnp,
    lift_triangulate,
)


# Lazy-init YOLO detector. If best.onnx is missing, this is None and we fall
# back to the classical detector.
_YOLO_DETECTOR = None


def _get_yolo():
    global _YOLO_DETECTOR
    if _YOLO_DETECTOR is None:
        try:
            _YOLO_DETECTOR = YoloPosePortDetector(conf=0.3)
        except Exception:
            _YOLO_DETECTOR = False
    return _YOLO_DETECTOR if _YOLO_DETECTOR else None


def detect_port_smart(image_rgb, port_type):
    """Prefer YOLO detector (offline benchmark with correct spec mouth
    dimensions: T1 SFP 4.4mm, T2 SFP 3.6mm, T3 SC 11.9mm). Fall back
    to classical (with rim-shrink correction) if YOLO fails.

    The classical detector's bbox extends past the spec mouth into the
    housing rim — port_detector.shrink_corners_by_rim() compensates by
    shrinking detected corners toward center by an empirically calibrated
    ratio (0.76 in W, 0.65 in H for SFP).
    """
    yolo = _get_yolo()
    if yolo is not None and yolo.available:
        yolo_det = yolo.detect(image_rgb, port_type)
        if yolo_det is not None and yolo_det.score >= 0.3:
            return yolo_det
    return detect_port_classical(image_rgb, port_type, refine=True)


OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"


def _identity_offset():
    return {"x": 0.0, "y": 0.0, "z": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}


def _load_offset_for(port_type: str) -> dict:
    if not OFFSET_PATH.exists():
        return _identity_offset()
    try:
        data = json.loads(OFFSET_PATH.read_text())
        return data.get(port_type, data.get("default", _identity_offset()))
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


class VisionInsert(Policy):
    """CheatCode without ground-truth TF: vision-derived port pose."""

    def __init__(self, parent_node):
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        super().__init__(parent_node)

        # Visualization output dir for debugging the detector.
        self._dbg_dir = Path.home() / "aic_logs" / "vision_insert_dbg"
        self._dbg_dir.mkdir(parents=True, exist_ok=True)

        self._port_pose: Optional[PortPose6D] = None  # cached from first detection
        self._tcp_to_plug_T = np.eye(4)               # filled in insert_cable

    # ------------------------------------------------------------------
    # Vision: detect on all 3 cameras and triangulate. Falls back to
    # single-camera PnP if triangulation rays don't agree.
    # ------------------------------------------------------------------
    def _lookup_cam_tf(self, frame_name: str):
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", frame_name, Time()
            )
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
            self.get_logger().warn(
                f"VisionInsert: cannot read {frame_name} TF: {ex}"
            )
            return None

    def _detect_port_pose(
        self,
        get_observation: GetObservationCallback,
        port_type: str,
        max_attempts: int = 20,
    ) -> Optional[PortPose6D]:
        for attempt in range(max_attempts):
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.1)
                continue

            cams = []
            for cam_name, img_msg, K_msg in (
                ("left", obs.left_image, obs.left_camera_info),
                ("center", obs.center_image, obs.center_camera_info),
                ("right", obs.right_image, obs.right_camera_info),
            ):
                if not img_msg.data:
                    continue
                img_rgb = _ros_image_to_numpy(img_msg)
                det = detect_port_smart(img_rgb, port_type)
                if det is None:
                    continue
                cam_tf = self._lookup_cam_tf(f"{cam_name}_camera/optical")
                if cam_tf is None:
                    continue
                K = list(K_msg.k)
                cams.append((cam_name, det, K, cam_tf, img_rgb))

            self.get_logger().info(
                f"VisionInsert attempt {attempt}: detections on "
                f"{[c[0] for c in cams]}"
            )

            if len(cams) >= 2:
                tri = lift_triangulate(
                    [(c[1], c[2], c[3]) for c in cams]
                )
                if tri is not None:
                    self.get_logger().info(
                        f"VisionInsert: TRIANGULATED port_xyz="
                        f"({tri.transform['x']:.3f},{tri.transform['y']:.3f},"
                        f"{tri.transform['z']:.3f}) from {len(cams)} cameras"
                    )
                    # Save debug frame from center
                    for (n, d, K, T, img) in cams:
                        if n == "center":
                            try:
                                bgr_dbg = draw_detection(img, d)
                                cv2.imwrite(str(self._dbg_dir / f"detect_{attempt}.jpg"), bgr_dbg)
                            except Exception:
                                pass
                            break
                    return tri
                else:
                    self.get_logger().warn(
                        "VisionInsert: triangulation rejected (rays disagree); "
                        "falling back to single-camera PnP"
                    )

            # Fallback: PnP from the center camera alone.
            for (n, d, K, T, img) in cams:
                if n != "center":
                    continue
                pose = lift_pnp(d, K, T, port_type=port_type)
                if pose is None:
                    pose = lift_to_base(d, K, T, port_type=port_type)
                if pose is not None:
                    self.get_logger().info(
                        f"VisionInsert: PnP fallback port_xyz="
                        f"({pose.transform['x']:.3f},{pose.transform['y']:.3f},"
                        f"{pose.transform['z']:.3f})"
                    )
                    return pose

            self.sleep_for(0.2)
        return None

    # ------------------------------------------------------------------
    # Plug tip pose from observation (no ground-truth TF needed).
    # ------------------------------------------------------------------
    def _plug_pose_from_obs(self, obs: Observation):
        # tcp_pose in base_link is in obs.controller_state.tcp_pose.
        tp = obs.controller_state.tcp_pose
        T_tcp = np.eye(4)
        T_tcp[:3, :3] = _quat_to_R(
            (tp.orientation.w, tp.orientation.x, tp.orientation.y, tp.orientation.z)
        )
        T_tcp[:3, 3] = [tp.position.x, tp.position.y, tp.position.z]
        T_plug = T_tcp @ self._tcp_to_plug_T
        return T_tcp, T_plug

    # ------------------------------------------------------------------
    # Mirror of CheatCode.calc_gripper_pose, but with no /tf calls.
    # ------------------------------------------------------------------
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
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        T_tcp, T_plug = self._plug_pose_from_obs(latest_obs)
        gripper_xyz = (T_tcp[0, 3], T_tcp[1, 3], T_tcp[2, 3])
        plug_xyz = (T_plug[0, 3], T_plug[1, 3], T_plug[2, 3])

        qw_g, qx_g, qy_g, qz_g = _R_to_quat(T_tcp[:3, :3])
        qw_p, qx_p, qy_p, qz_p = _R_to_quat(T_plug[:3, :3])
        q_plug = (qw_p, qx_p, qy_p, qz_p)
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
                -self._max_integrator_windup,
                self._max_integrator_windup,
            ))
            self._tip_y_error_integrator = float(np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
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
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )

    # ------------------------------------------------------------------
    # The trial entry point.
    # ------------------------------------------------------------------
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"VisionInsert.insert_cable() task: {task}")
        self._task = task

        # Load the calibrated tcp -> plug offset for this port type.
        offset = _load_offset_for(task.port_type)
        self._tcp_to_plug_T = _tf_dict_to_T(offset)
        self.get_logger().info(
            f"VisionInsert: tcp_to_plug offset = {offset}"
        )

        # Detect the port in the center wrist image.
        pose = self._detect_port_pose(get_observation, task.port_type)
        if pose is None:
            self.get_logger().error(
                "VisionInsert: failed to detect port; aborting trial."
            )
            return False
        self._port_pose = pose
        port_transform = _tf_to_geometry(pose.transform)
        send_feedback(f"port detected at depth {pose.depth_m:.3f} m")

        # Same motion plan as CheatCode.
        z_offset = 0.2
        for t in range(0, 100):
            obs = get_observation()
            if obs is None:
                if t % 10 == 0:
                    self.get_logger().info(f"VisionInsert interp t={t} obs=None")
                self.sleep_for(0.05)
                continue
            interp_fraction = t / 100.0
            try:
                target_pose = self.calc_gripper_pose(
                    port_transform,
                    obs,
                    slerp_fraction=interp_fraction,
                    position_fraction=interp_fraction,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                )
                ok = self.set_pose_target(move_robot=move_robot, pose=target_pose)
                if t % 10 == 0:
                    self.get_logger().info(
                        f"VisionInsert interp t={t} target=({target_pose.position.x:.3f},{target_pose.position.y:.3f},{target_pose.position.z:.3f}) ok={ok}"
                    )
            except Exception as ex:
                self.get_logger().error(f"VisionInsert: interp step {t} EXCEPTION: {ex}")
                import traceback
                self.get_logger().error(traceback.format_exc())
            self.sleep_for(0.05)
        self.get_logger().info("VisionInsert: interpolation done, starting descent")

        # Descent.
        descent_step = 0
        while True:
            if z_offset < -0.015:
                break
            z_offset -= 0.0005
            descent_step += 1
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.05)
                continue
            try:
                target_pose = self.calc_gripper_pose(
                    port_transform,
                    obs,
                    z_offset=z_offset,
                )
                self.set_pose_target(move_robot=move_robot, pose=target_pose)
                if descent_step % 50 == 0:
                    self.get_logger().info(
                        f"VisionInsert descent step={descent_step} z_offset={z_offset:.4f} target_z={target_pose.position.z:.3f}"
                    )
            except Exception as ex:
                self.get_logger().error(f"VisionInsert: descent EXCEPTION: {ex}")
                import traceback
                self.get_logger().error(traceback.format_exc())
            self.sleep_for(0.05)

        self.get_logger().info("VisionInsert: stabilizing...")
        self.sleep_for(5.0)
        self.get_logger().info("VisionInsert.insert_cable() exiting")
        return True
