#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
"""CheatCode + per-frame data logger.

Runs the same insertion logic as CheatCode, but additionally subscribes to
/observations and dumps every frame to disk along with the simultaneous TF
lookups (port, plug, TCP, camera optical frames). Used for:

  1. Validating CheatCode upper-bound score (Block A).
  2. Computing the constant TCP -> plug_tip offset (Block B).
  3. Capturing labeled training data for Phase 2 (Block G's seed dataset).

Output layout:

    ~/aic_logs/<timestamp>/
        trial_01_<port_type>/
            task.json
            00000.json   00000_left.jpg   00000_center.jpg   00000_right.jpg
            00001.json   ...
        trial_02_<port_type>/
            ...

Launch (replaces CheatCode):

    pixi run ros2 run aic_model aic_model --ros-args \
        -p use_sim_time:=true \
        -p policy:=aic_example_policies.ros.LoggingCheatCode

Requires the eval container to be running with ground_truth:=true so the
port and plug TF frames are published.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rclpy.time import Time
from tf2_ros import TransformException

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from .CheatCode import CheatCode


# Cap log volume per trial so a stuck/long trial does not fill the disk.
# At 20 Hz, 90 sec per trial * 3 trials * 3 cams * ~150 KB JPEG ~= 4 GB.
MAX_FRAMES_PER_TRIAL = 2000


class LoggingCheatCode(CheatCode):
    """CheatCode that logs every observation and TF lookup to disk."""

    def __init__(self, parent_node):
        super().__init__(parent_node)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = Path.home() / "aic_logs" / ts
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._trial_dir: Path | None = None
        self._frame_idx = 0
        self._trial_idx = 0
        self._task_active: Task | None = None
        self._first_obs_logged = False

        self._obs_sub = parent_node.create_subscription(
            Observation, "/observations", self._on_obs, 10
        )

        self.get_logger().info(
            f"LoggingCheatCode: dumping to {self._run_dir}"
        )

    # ------------------------------------------------------------------
    # Lifecycle: insert_cable wraps CheatCode's logic, with per-trial setup
    # ------------------------------------------------------------------
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self._trial_idx += 1
        port_tag = task.port_type or "unknown"
        self._trial_dir = (
            self._run_dir / f"trial_{self._trial_idx:02d}_{port_tag}"
        )
        self._trial_dir.mkdir(parents=True, exist_ok=True)
        self._frame_idx = 0
        self._first_obs_logged = False

        with open(self._trial_dir / "task.json", "w") as f:
            json.dump(_task_to_dict(task), f, indent=2)

        self._task_active = task
        self.get_logger().info(
            f"LoggingCheatCode trial {self._trial_idx} -> {self._trial_dir}"
        )

        try:
            return super().insert_cable(
                task, get_observation, move_robot, send_feedback
            )
        finally:
            self._task_active = None
            self.get_logger().info(
                f"LoggingCheatCode trial {self._trial_idx} done: "
                f"{self._frame_idx} frames in {self._trial_dir}"
            )

    # ------------------------------------------------------------------
    # Per-frame logger
    # ------------------------------------------------------------------
    def _on_obs(self, obs: Observation) -> None:
        if self._task_active is None or self._trial_dir is None:
            return
        if self._frame_idx >= MAX_FRAMES_PER_TRIAL:
            return

        idx = self._frame_idx
        self._frame_idx += 1

        try:
            self._write_images(obs, idx)
            self._write_record(obs, idx)
        except Exception as e:
            self.get_logger().warn(
                f"LoggingCheatCode frame {idx} dump failed: {e}"
            )

        if not self._first_obs_logged:
            self._first_obs_logged = True
            self.get_logger().info(
                f"LoggingCheatCode first frame: "
                f"{obs.center_image.width}x{obs.center_image.height} "
                f"encoding={obs.center_image.encoding}"
            )

    def _write_images(self, obs: Observation, idx: int) -> None:
        for name, msg in (
            ("left", obs.left_image),
            ("center", obs.center_image),
            ("right", obs.right_image),
        ):
            if not msg.data:
                continue
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            # cv2.imwrite expects BGR. Most AIC streams are rgb8.
            if msg.encoding == "bgr8":
                bgr = arr
            else:
                bgr = arr[:, :, ::-1]
            cv2.imwrite(
                str(self._trial_dir / f"{idx:05d}_{name}.jpg"),
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

    def _write_record(self, obs: Observation, idx: int) -> None:
        port_frame = (
            f"task_board/{self._task_active.target_module_name}/"
            f"{self._task_active.port_name}_link"
        )
        plug_frame = (
            f"{self._task_active.cable_name}/"
            f"{self._task_active.plug_name}_link"
        )

        record = {
            "frame": idx,
            "stamp_sec": int(obs.center_image.header.stamp.sec),
            "stamp_nanosec": int(obs.center_image.header.stamp.nanosec),
            "image_encoding": obs.center_image.encoding,
            "image_w": int(obs.center_image.width),
            "image_h": int(obs.center_image.height),
            "joint_position": list(obs.joint_states.position),
            "joint_velocity": list(obs.joint_states.velocity),
            "joint_name": list(obs.joint_states.name),
            "wrench": _wrench_to_dict(obs.wrist_wrench.wrench),
            "tcp_pose_obs": _pose_to_dict(obs.controller_state.tcp_pose),
            "port_tf_base": self._safe_lookup("base_link", port_frame),
            "plug_tf_base": self._safe_lookup("base_link", plug_frame),
            "tcp_tf_base": self._safe_lookup("base_link", "gripper/tcp"),
            "left_cam_optical_tf_base": self._safe_lookup(
                "base_link", "left_camera/optical"
            ),
            "center_cam_optical_tf_base": self._safe_lookup(
                "base_link", "center_camera/optical"
            ),
            "right_cam_optical_tf_base": self._safe_lookup(
                "base_link", "right_camera/optical"
            ),
            "K_center": list(obs.center_camera_info.k),
            "K_left": list(obs.left_camera_info.k),
            "K_right": list(obs.right_camera_info.k),
            "port_frame_name": port_frame,
            "plug_frame_name": plug_frame,
        }

        with open(self._trial_dir / f"{idx:05d}.json", "w") as f:
            json.dump(record, f)

    def _safe_lookup(self, target: str, source: str):
        try:
            tf = self._parent_node._tf_buffer.lookup_transform(
                target, source, Time()
            )
        except TransformException:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        return {
            "x": float(t.x),
            "y": float(t.y),
            "z": float(t.z),
            "qx": float(r.x),
            "qy": float(r.y),
            "qz": float(r.z),
            "qw": float(r.w),
        }


def _wrench_to_dict(w):
    return {
        "fx": float(w.force.x),
        "fy": float(w.force.y),
        "fz": float(w.force.z),
        "tx": float(w.torque.x),
        "ty": float(w.torque.y),
        "tz": float(w.torque.z),
    }


def _pose_to_dict(p):
    return {
        "x": float(p.position.x),
        "y": float(p.position.y),
        "z": float(p.position.z),
        "qx": float(p.orientation.x),
        "qy": float(p.orientation.y),
        "qz": float(p.orientation.z),
        "qw": float(p.orientation.w),
    }


def _task_to_dict(task: Task):
    return {
        "id": task.id,
        "cable_type": task.cable_type,
        "cable_name": task.cable_name,
        "plug_type": task.plug_type,
        "plug_name": task.plug_name,
        "port_type": task.port_type,
        "port_name": task.port_name,
        "target_module_name": task.target_module_name,
        "time_limit": int(task.time_limit),
    }
