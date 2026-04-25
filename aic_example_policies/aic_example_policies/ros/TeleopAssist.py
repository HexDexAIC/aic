#
#  Copyright (C) 2026 Hariharan Ravichandran
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""TeleopAssist — shared-autonomy policy wrapper with optional dataset recording.

Wraps any other policy class (configurable via the ``INNER_POLICY`` env var) and
optionally injects keyboard-driven cartesian deltas on top of the inner
policy's commands. Modes:

- "delta"   — inner policy's MotionUpdate goes out, teleop adds a small delta
              when keys are held. Default.
- "pause"   — inner policy is suppressed; the last commanded pose is held and
              teleop deltas accumulate on top. Useful for fine positioning.
- "stop"    — emergency: publish a hold and return False.

If ``RECORD_DATASET_PATH`` is set to a directory, every commanded MotionUpdate
is logged to a ``LeRobotDataset`` along with the latest observation. The
schema follows the design rationale in
``aic_wiki/wiki/methodology/recording-data.md`` — full observation state and
the absolute commanded pose, leaving delta/6D-rotation/compliance derivation
to a downstream training transform.

Env vars
--------
- ``INNER_POLICY``        — short class name from ``aic_example_policies.ros``,
                            e.g. ``CheatCodeMJ``, ``WaveArm``, or ``none`` for
                            pure-teleop hold mode. Default: ``WaveArm``.
- ``ENABLE_TELEOP``       — ``1``/``0``. If ``0``, runs the inner policy alone
                            (still records if ``RECORD_DATASET_PATH`` set).
                            Default: ``1``.
- ``RECORD_DATASET_PATH`` — directory under which to create the LeRobot
                            dataset. If unset, no recording.
- ``RECORD_TASK_PROMPT``  — natural-language task string written into each
                            frame. Default: derived from the InsertCable task.
- ``TELEOP_LIN_RATE``     — keyboard linear rate (m/s). Default 0.04.
- ``TELEOP_ANG_RATE``     — keyboard angular rate (rad/s). Default 0.5.
"""

from __future__ import annotations

import datetime
import importlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from aic_control_interfaces.msg import (
    MotionUpdate,
    JointMotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.node import Node
from std_msgs.msg import Header

from aic_example_policies.ros.teleop_keyboard import KeyboardTeleop


# ── Quaternion / 6D rotation helpers ──────────────────────────────────
def quat_mul(q1: tuple, q2: tuple) -> tuple:
    """Hamilton product: returns q1 * q2 in (x, y, z, w) ordering."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def small_rotation_quat(droll: float, dpitch: float, dyaw: float) -> tuple:
    """Compose a small-angle quaternion delta from rpy increments.

    Composition order is yaw → pitch → roll (extrinsic XYZ). For small
    angles (<~0.1 rad/step at 20 Hz this is comfortable), the order matters
    less than precision.
    """
    cr, sr = np.cos(droll * 0.5), np.sin(droll * 0.5)
    cp, sp = np.cos(dpitch * 0.5), np.sin(dpitch * 0.5)
    cy, sy = np.cos(dyaw * 0.5), np.sin(dyaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return (qx, qy, qz, qw)


def quat_to_rotmat_6d(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion to the first two columns of the rotation matrix.

    Returns a 6-vector — Zhou et al. 2018's continuous rotation rep that
    avoids quaternion antipodal discontinuity for learning. Caller can
    Gram-Schmidt to recover SO(3) at training time.
    """
    # Standard q → R (column-major, then take cols 0, 1)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    col0 = np.array([1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)], dtype=np.float32)
    col1 = np.array([2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)], dtype=np.float32)
    return np.concatenate([col0, col1])


# ── Observation → state vector ────────────────────────────────────────
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def observation_to_state(obs: Observation) -> np.ndarray:
    """Pack an Observation into the 27-D state vector documented in the schema.

    Layout (per ``recording-data.md``):
        [0:3]   tcp_position xyz
        [3:9]   tcp_rotation_6d (Zhou 2018)
        [9:12]  tcp_linear_velocity
        [12:15] tcp_angular_velocity
        [15:21] wrist_wrench (force xyz, torque xyz)
        [21:27] joint_positions (6 UR joints in canonical order)
    """
    cs = obs.controller_state
    p = cs.tcp_pose.position
    q = cs.tcp_pose.orientation
    lv = cs.tcp_velocity.linear
    av = cs.tcp_velocity.angular

    rot6 = quat_to_rotmat_6d(q.x, q.y, q.z, q.w)

    wf = obs.wrist_wrench.wrench.force
    wt = obs.wrist_wrench.wrench.torque

    # Joint positions — re-order to canonical UR order in case the publisher
    # uses a different ordering. Fall back to first-6 if names missing.
    name_to_pos = dict(zip(obs.joint_states.name, obs.joint_states.position))
    if all(n in name_to_pos for n in JOINT_NAMES):
        joints = np.array([name_to_pos[n] for n in JOINT_NAMES], dtype=np.float32)
    else:
        joints = np.array(list(obs.joint_states.position[:6]), dtype=np.float32)

    state = np.concatenate(
        [
            np.array([p.x, p.y, p.z], dtype=np.float32),
            rot6.astype(np.float32),
            np.array([lv.x, lv.y, lv.z], dtype=np.float32),
            np.array([av.x, av.y, av.z], dtype=np.float32),
            np.array([wf.x, wf.y, wf.z, wt.x, wt.y, wt.z], dtype=np.float32),
            joints,
        ]
    )
    assert state.shape == (27,), f"state shape mismatch: {state.shape}"
    return state


def image_msg_to_array(img_msg) -> np.ndarray:
    """sensor_msgs/Image → numpy HxWx3 uint8 (RGB)."""
    return np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3
    )


# ── LeRobot dataset writer ────────────────────────────────────────────
class _DatasetWriter:
    """Thin wrapper around LeRobotDataset.create + add_frame + save_episode.

    Keeps the lerobot import deferred until first use so importing
    ``TeleopAssist`` doesn't pay the lerobot warm-up cost when recording is
    disabled.
    """

    def __init__(self, root: Path, repo_id: str, fps: int, image_shape: tuple):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        h, w, _c = image_shape
        features = {
            "observation.images.left": {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.center": {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.right": {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (27,),
                "names": [
                    "tcp_x", "tcp_y", "tcp_z",
                    "rot6_0", "rot6_1", "rot6_2", "rot6_3", "rot6_4", "rot6_5",
                    "tcp_vx", "tcp_vy", "tcp_vz",
                    "tcp_wx", "tcp_wy", "tcp_wz",
                    "force_x", "force_y", "force_z",
                    "torque_x", "torque_y", "torque_z",
                    "q_pan", "q_lift", "q_elbow",
                    "q_wrist1", "q_wrist2", "q_wrist3",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"],
            },
            "action.stiffness_diag": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["kx", "ky", "kz", "krx", "kry", "krz"],
            },
            "action.teleop_active": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["teleop_active"],
            },
        }
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=str(root),
            use_videos=True,
        )
        self._frames_in_episode = 0

    def add_frame(self, frame: dict) -> None:
        self.dataset.add_frame(frame)
        self._frames_in_episode += 1

    def save_episode(self) -> int:
        n = self._frames_in_episode
        if n > 0:
            self.dataset.save_episode()
        else:
            # No frames recorded — nothing to save.
            self.dataset.clear_episode_buffer()
        self._frames_in_episode = 0
        return n


# ── Main policy class ─────────────────────────────────────────────────
class TeleopAssist(Policy):
    """Shared-autonomy wrapper. See module docstring."""

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        log = self.get_logger()

        # Load inner policy.
        inner_name = os.environ.get("INNER_POLICY", "WaveArm")
        self._inner_policy_name = inner_name
        if inner_name == "none":
            self.inner = None
            log.info("TeleopAssist: no inner policy (pure-teleop / hold-pose mode)")
        else:
            module_path = f"aic_example_policies.ros.{inner_name}"
            try:
                module = importlib.import_module(module_path)
            except Exception as e:
                log.fatal(f"TeleopAssist: cannot import inner policy {module_path!r}: {e}")
                raise
            inner_cls = getattr(module, inner_name, None)
            if inner_cls is None:
                raise LookupError(f"Class {inner_name} not in module {module_path}")
            log.info(f"TeleopAssist: instantiating inner policy {inner_name}")
            self.inner = inner_cls(parent_node)

        # Optional teleop.
        self._teleop_enabled = os.environ.get("ENABLE_TELEOP", "1") == "1"
        self.teleop: Optional[KeyboardTeleop] = None
        if self._teleop_enabled:
            try:
                self.teleop = KeyboardTeleop().start()
                log.info("TeleopAssist: keyboard teleop active")
            except RuntimeError as e:
                log.warn(f"TeleopAssist: keyboard teleop unavailable ({e}); inner only")

        # Optional dataset recording. Lazy-initialize on insert_cable so we
        # know fps + image shape from the first observation.
        self._dataset_root = os.environ.get("RECORD_DATASET_PATH")
        self._writer: Optional[_DatasetWriter] = None
        self._task_prompt = os.environ.get("RECORD_TASK_PROMPT", "")

        # Bookkeeping for pause-mode hold-pose.
        self._last_motion_update: Optional[MotionUpdate] = None
        self._last_publish_time: float = 0.0

    # ── Inner-policy callback wrapper ─────────────────────────────────
    def _maybe_init_writer(self, obs: Observation) -> None:
        """First-observation lazy init of the dataset writer."""
        if self._dataset_root is None or self._writer is not None or obs is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_id = f"local/aic_recording_{ts}"
        root = Path(self._dataset_root).expanduser() / f"aic_recording_{ts}"
        h, w = obs.center_image.height, obs.center_image.width
        self.get_logger().info(
            f"TeleopAssist: opening dataset at {root} (image {h}x{w}x3)"
        )
        # FPS=20 matches the engine's observation→policy call rate floor; the
        # tolerance_s parameter in LeRobotDataset accommodates per-policy jitter.
        self._writer = _DatasetWriter(root=root, repo_id=repo_id, fps=20, image_shape=(h, w, 3))

    def _apply_teleop_delta(
        self, motion_update: MotionUpdate, dt: float
    ) -> tuple[MotionUpdate, bool, str]:
        """Mutate a MotionUpdate in-place applying current teleop intent.

        Returns (motion_update, teleop_active, mode). When mode == "stop" the
        caller should publish a hold and exit.
        """
        if self.teleop is None:
            return motion_update, False, "delta"

        state = self.teleop.get_delta(dt=dt)
        if state.mode == "stop":
            return motion_update, False, "stop"

        # In pause mode, replace inner's commanded pose with our last published
        # pose (teleop becomes the only source of motion).
        if state.mode == "pause" and self._last_motion_update is not None:
            motion_update.pose = Pose(
                position=Point(
                    x=self._last_motion_update.pose.position.x,
                    y=self._last_motion_update.pose.position.y,
                    z=self._last_motion_update.pose.position.z,
                ),
                orientation=Quaternion(
                    x=self._last_motion_update.pose.orientation.x,
                    y=self._last_motion_update.pose.orientation.y,
                    z=self._last_motion_update.pose.orientation.z,
                    w=self._last_motion_update.pose.orientation.w,
                ),
            )

        # Apply linear deltas in base frame.
        if state.dx or state.dy or state.dz:
            motion_update.pose.position.x += state.dx
            motion_update.pose.position.y += state.dy
            motion_update.pose.position.z += state.dz

        # Apply rotational deltas via quaternion left-multiply.
        if state.droll or state.dpitch or state.dyaw:
            q = motion_update.pose.orientation
            dq = small_rotation_quat(state.droll, state.dpitch, state.dyaw)
            new_q = quat_mul(dq, (q.x, q.y, q.z, q.w))
            motion_update.pose.orientation = Quaternion(
                x=float(new_q[0]), y=float(new_q[1]), z=float(new_q[2]), w=float(new_q[3])
            )

        return motion_update, state.active, state.mode

    def _record(
        self,
        obs: Observation,
        motion_update: MotionUpdate,
        teleop_active: bool,
    ) -> None:
        if self._writer is None:
            return
        try:
            state_vec = observation_to_state(obs)
            p = motion_update.pose.position
            q = motion_update.pose.orientation
            action = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)
            stiffness = np.asarray(motion_update.target_stiffness, dtype=np.float32)
            stiffness_diag = (
                stiffness.reshape(6, 6).diagonal().astype(np.float32)
                if stiffness.size == 36
                else np.zeros(6, dtype=np.float32)
            )
            frame = {
                "task": self._task_prompt or "insert cable",
                "observation.images.left":   image_msg_to_array(obs.left_image),
                "observation.images.center": image_msg_to_array(obs.center_image),
                "observation.images.right":  image_msg_to_array(obs.right_image),
                "observation.state": state_vec,
                "action": action,
                "action.stiffness_diag": stiffness_diag,
                "action.teleop_active": np.array([1.0 if teleop_active else 0.0], dtype=np.float32),
            }
            self._writer.add_frame(frame)
        except Exception as e:  # pragma: no cover
            self.get_logger().error(f"TeleopAssist: record failure: {e}")

    # ── Pure-teleop self-loop (when INNER_POLICY=none) ────────────────
    def _self_loop(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        wrapped_move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Hold the current TCP pose; teleop drives. Runs ~30 s like other policies."""
        # Wait for first observation.
        log = self.get_logger()
        deadline = time.time() + 5.0
        obs = get_observation()
        while obs is None and time.time() < deadline:
            time.sleep(0.05)
            obs = get_observation()
        if obs is None:
            log.error("TeleopAssist self_loop: no observation in 5 s — aborting")
            return False

        # Seed last_motion_update with the current TCP pose from controller_state.
        cs = obs.controller_state
        seed = self.set_pose_target_via_motion_update(
            position=(cs.tcp_pose.position.x, cs.tcp_pose.position.y, cs.tcp_pose.position.z),
            orientation=(
                cs.tcp_pose.orientation.x,
                cs.tcp_pose.orientation.y,
                cs.tcp_pose.orientation.z,
                cs.tcp_pose.orientation.w,
            ),
        )
        # Publish once to establish hold pose.
        wrapped_move_robot(motion_update=seed)

        loop_dt = 0.05  # 20 Hz
        end_time = time.time() + 30.0
        while time.time() < end_time:
            time.sleep(loop_dt)
            mu = self.set_pose_target_via_motion_update(
                position=(
                    self._last_motion_update.pose.position.x,
                    self._last_motion_update.pose.position.y,
                    self._last_motion_update.pose.position.z,
                ),
                orientation=(
                    self._last_motion_update.pose.orientation.x,
                    self._last_motion_update.pose.orientation.y,
                    self._last_motion_update.pose.orientation.z,
                    self._last_motion_update.pose.orientation.w,
                ),
            )
            wrapped_move_robot(motion_update=mu)
            if self.teleop is not None and self.teleop.get_mode() == "stop":
                log.info("TeleopAssist self_loop: ESC → exiting")
                break

        send_feedback("teleop self-loop done")
        return True

    def set_pose_target_via_motion_update(
        self, position: tuple, orientation: tuple
    ) -> MotionUpdate:
        """Build a MotionUpdate with the same defaults as Policy.set_pose_target."""
        stiffness = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
        damping = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
        return MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=Pose(
                position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                orientation=Quaternion(
                    x=float(orientation[0]),
                    y=float(orientation[1]),
                    z=float(orientation[2]),
                    w=float(orientation[3]),
                ),
            ),
            target_stiffness=np.diag(stiffness).flatten(),
            target_damping=np.diag(damping).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

    # ── insert_cable: the framework entry point ───────────────────────
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        log = self.get_logger()
        log.info(
            f"TeleopAssist.insert_cable enter (inner={self._inner_policy_name}, "
            f"teleop={'on' if self.teleop else 'off'}, "
            f"record={'on' if self._dataset_root else 'off'})"
        )
        if not self._task_prompt:
            self._task_prompt = f"insert {task.plug_type} cable into {task.port_type} port"

        # Reset bookkeeping per episode.
        self._last_motion_update = None
        self._last_publish_time = time.monotonic()

        def wrapped_move_robot(
            motion_update: MotionUpdate = None,
            joint_motion_update: JointMotionUpdate = None,
        ):
            now = time.monotonic()
            dt = max(1e-3, now - self._last_publish_time)
            self._last_publish_time = now

            # Joint commands pass straight through (no teleop / recording on
            # this path — we'd need a different action representation).
            if joint_motion_update is not None:
                return move_robot(joint_motion_update=joint_motion_update)

            if motion_update is None:
                return False

            mu, teleop_active, mode = self._apply_teleop_delta(motion_update, dt)
            if mode == "stop":
                # Publish a hold using the last successfully published pose, then
                # surface the stop to the inner policy by returning False.
                if self._last_motion_update is not None:
                    move_robot(motion_update=self._last_motion_update)
                return False

            self._last_motion_update = mu

            # Record only if writer is initialized; first observation lazy-inits.
            obs = get_observation()
            if obs is not None:
                self._maybe_init_writer(obs)
                self._record(obs, mu, teleop_active=teleop_active)

            return move_robot(motion_update=mu)

        # Run inner policy or self-loop.
        try:
            if self.inner is None:
                ok = self._self_loop(task, get_observation, wrapped_move_robot, send_feedback)
            else:
                ok = self.inner.insert_cable(
                    task=task,
                    get_observation=get_observation,
                    move_robot=wrapped_move_robot,
                    send_feedback=send_feedback,
                )
        except Exception as e:
            log.error(f"TeleopAssist: inner policy raised: {e}")
            ok = False
        finally:
            # Save the episode whether or not the inner returned True.
            if self._writer is not None:
                n_frames = self._writer.save_episode()
                log.info(f"TeleopAssist: saved {n_frames} frames to dataset")
            if self.teleop is not None:
                self.teleop.stop()
                self.teleop = None

        log.info(f"TeleopAssist.insert_cable exit (ok={ok})")
        return ok
