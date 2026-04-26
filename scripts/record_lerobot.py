#!/usr/bin/env python3
#
#  Copyright (C) 2026 Hariharan Ravichandran
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#

"""Standalone LeRobotDataset recorder for AIC eval runs.

Subscribes to:
    /observations                        (obs message — published by aic_adapter)
    /aic_controller/pose_commands        (action — what the policy commanded)
    /insert_cable/_action/status         (trial bookends)

Caches the latest obs + action; a 20 Hz timer pairs them and writes a frame.
Episodes are auto-bookended by the action status:

    STATUS_EXECUTING  →  open a new episode (writer lazy-inits on first obs)
    SUCCEEDED / ABORTED / CANCELED →  save_episode() and exit

On the EXECUTING edge we drop any cached obs/action so the first written
frame is anchored to the policy's first `pose_commands`. STATUS_EXECUTING
itself fires before the policy starts driving, and `_latest.action` may
already hold an engine-side home→hover slerp command — without the reset
those frames bleed into the dataset.

Runs as its own process, so the policy thread is never blocked on dataset I/O.

Usage (typically launched by record_episode.sh alongside the policy):

    pixi run python src/aic/scripts/record_lerobot.py \
        --root ~/ws_aic/aic_data \
        --task "insert sfp cable" \
        [--vcodec h264] [--no-videos] [--multi] [--fps 20]

Run with --help for the full flag list.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import rclpy
from action_msgs.msg import GoalStatusArray
from aic_control_interfaces.msg import MotionUpdate
from aic_model_interfaces.msg import Observation
from rclpy.node import Node

# action_msgs/msg/GoalStatus enum values.
STATUS_EXECUTING = 2
TERMINAL_STATUSES = (4, 5, 6)  # SUCCEEDED, CANCELED, ABORTED


# ────────────────────── helpers (inlined, self-contained) ──────────────────
def quat_to_rotmat_6d(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion → first two columns of rotation matrix (Zhou 2018, 6-D)."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    col0 = np.array([1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)], dtype=np.float32)
    col1 = np.array([2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)], dtype=np.float32)
    return np.concatenate([col0, col1])


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def observation_to_state(obs: Observation) -> np.ndarray:
    """Pack Observation into the 27-D state vector documented in recording-data.md."""
    cs = obs.controller_state
    p, q = cs.tcp_pose.position, cs.tcp_pose.orientation
    lv, av = cs.tcp_velocity.linear, cs.tcp_velocity.angular
    rot6 = quat_to_rotmat_6d(q.x, q.y, q.z, q.w)
    wf = obs.wrist_wrench.wrench.force
    wt = obs.wrist_wrench.wrench.torque
    name_to_pos = dict(zip(obs.joint_states.name, obs.joint_states.position))
    if all(n in name_to_pos for n in JOINT_NAMES):
        joints = np.array([name_to_pos[n] for n in JOINT_NAMES], dtype=np.float32)
    else:
        joints = np.array(list(obs.joint_states.position[:6]), dtype=np.float32)
    return np.concatenate(
        [
            np.array([p.x, p.y, p.z], dtype=np.float32),
            rot6.astype(np.float32),
            np.array([lv.x, lv.y, lv.z], dtype=np.float32),
            np.array([av.x, av.y, av.z], dtype=np.float32),
            np.array([wf.x, wf.y, wf.z, wt.x, wt.y, wt.z], dtype=np.float32),
            joints,
        ]
    )


def motion_update_to_action(mu: MotionUpdate) -> np.ndarray:
    """MotionUpdate → 9-D action: TCP pos (3) + 6-D rotation rep (6).

    Quaternion is converted to the continuous 6-D rep (same convention as the
    obs's tcp rotation entry). At inference time the policy's 6-D output is
    Gram-Schmidt'd back to a rotation matrix and then to a quaternion before
    the controller sees it.
    """
    p = mu.pose.position
    q = mu.pose.orientation
    rot6 = quat_to_rotmat_6d(q.x, q.y, q.z, q.w)
    return np.concatenate(
        [
            np.array([p.x, p.y, p.z], dtype=np.float32),
            rot6.astype(np.float32),
        ]
    )


def stiffness_to_diag(mu: MotionUpdate) -> np.ndarray:
    s = np.asarray(mu.target_stiffness, dtype=np.float32)
    if s.size == 36:
        return s.reshape(6, 6).diagonal().astype(np.float32)
    return np.zeros(6, dtype=np.float32)


def image_msg_to_array(img_msg) -> np.ndarray:
    return np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
        img_msg.height, img_msg.width, 3
    )


# ────────────────────── dataset wrapper ────────────────────────────────────
class _Writer:
    def __init__(
        self,
        root: Path,
        repo_id: str,
        fps: int,
        image_shape: tuple,
        vcodec: str = "h264",
        use_videos: bool = True,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        h, w, _ = image_shape

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
                "shape": (9,),
                "names": [
                    "pos_x", "pos_y", "pos_z",
                    "rot6_0", "rot6_1", "rot6_2", "rot6_3", "rot6_4", "rot6_5",
                ],
            },
            "action.stiffness_diag": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["kx", "ky", "kz", "krx", "kry", "krz"],
            },
        }
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=str(root),
            use_videos=use_videos,
            vcodec=vcodec,
        )
        self.frames_in_episode = 0

    def add_frame(self, frame: dict) -> None:
        self.dataset.add_frame(frame)
        self.frames_in_episode += 1

    def save_episode(self) -> int:
        n = self.frames_in_episode
        if n > 0:
            self.dataset.save_episode()
        else:
            self.dataset.clear_episode_buffer()
        self.frames_in_episode = 0
        return n


# ────────────────────── recorder node ──────────────────────────────────────
@dataclass
class _Latest:
    obs: Optional[Observation] = None
    action: Optional[MotionUpdate] = None


class AICAsyncRecorder(Node):
    def __init__(
        self,
        dataset_root: Path,
        task_prompt: str,
        fps: int,
        multi_episode: bool,
        vcodec: str = "h264",
        use_videos: bool = True,
        fixed_name: Optional[str] = None,
    ):
        super().__init__("aic_async_recorder")
        self._dataset_root = dataset_root.expanduser()
        self._dataset_root.mkdir(parents=True, exist_ok=True)
        self._task_prompt = task_prompt
        self._fps = fps
        self._multi_episode = multi_episode
        self._vcodec = vcodec
        self._use_videos = use_videos
        self._fixed_name = fixed_name

        self._lock = Lock()
        self._latest = _Latest()
        self._was_executing = False
        self._writer: Optional[_Writer] = None
        self.episode_count = 0
        self.done = False  # set on terminal status when not multi-episode

        self.create_subscription(
            Observation, "/observations", self._on_obs, 10
        )
        self.create_subscription(
            MotionUpdate, "/aic_controller/pose_commands", self._on_action, 10
        )
        self.create_subscription(
            GoalStatusArray, "/insert_cable/_action/status", self._on_status, 10
        )
        self.create_timer(1.0 / self._fps, self._tick)

        self.get_logger().info(
            f"AICAsyncRecorder ready. fps={fps}, root={self._dataset_root}, "
            f"task='{task_prompt}', multi_episode={multi_episode}. "
            f"Awaiting STATUS_EXECUTING..."
        )

    # ── subscribers ────────────────────────────────────────────────
    def _on_obs(self, msg: Observation) -> None:
        with self._lock:
            self._latest.obs = msg
        # Eagerly create the dataset writer on the first observation, BEFORE
        # the policy starts driving. The LeRobotDataset.create call blocks
        # the executor for a couple of seconds; doing it here means that
        # block lands in the engine's idle window rather than in the middle
        # of the policy's slerp, where we'd lose frames. record_episode.sh
        # gates the policy launch on the WRITER_READY sentinel below.
        if self._writer is None:
            self._ensure_writer(msg)

    def _on_action(self, msg: MotionUpdate) -> None:
        with self._lock:
            self._latest.action = msg

    def _on_status(self, msg: GoalStatusArray) -> None:
        if not msg.status_list:
            return
        is_executing = any(s.status == STATUS_EXECUTING for s in msg.status_list)
        is_terminal = any(s.status in TERMINAL_STATUSES for s in msg.status_list)

        if is_executing and not self._was_executing:
            self._open_episode()
            self._was_executing = True
            with self._lock:
                # Discard any pre-policy state so the first frame is anchored
                # to the policy's first pose_commands (not a stale engine slerp).
                self._latest.obs = None
                self._latest.action = None
        elif self._was_executing and (not is_executing) and is_terminal:
            self._close_episode()
            self._was_executing = False
            if not self._multi_episode:
                self.done = True

    # ── episode lifecycle ──────────────────────────────────────────
    def _open_episode(self) -> None:
        """Mark that an episode should be recording. The writer is lazy-init'd
        on the first `_tick()` once an Observation has arrived — we cannot
        spin the executor here because we are inside an executor callback.
        """
        self.get_logger().info(
            "=== STATUS_EXECUTING — waiting for policy's first pose_commands ==="
        )

    def _ensure_writer(self, obs: Observation) -> bool:
        """Lazy-init the dataset writer the first time an observation is
        available within an active episode. Returns True if the writer is
        ready."""
        if self._writer is not None:
            return True
        if self._fixed_name:
            name = self._fixed_name
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"aic_recording_{ts}"
        repo_id = f"local/{name}"
        root = self._dataset_root / name
        h, w = obs.center_image.height, obs.center_image.width
        self.get_logger().info(
            f"=== Episode start → {root} (image {h}x{w}x3, repo_id={repo_id}) ==="
        )
        try:
            self._writer = _Writer(
                root=root,
                repo_id=repo_id,
                fps=self._fps,
                image_shape=(h, w, 3),
                vcodec=self._vcodec,
                use_videos=self._use_videos,
            )
            # Sentinel for record_episode.sh — gates the policy launch.
            print("WRITER_READY", flush=True)
            self.get_logger().info("=== Writer ready ===")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to create writer: {e}")
            self._writer = None
            return False

    def _close_episode(self) -> None:
        if self._writer is None:
            return
        n = self._writer.frames_in_episode
        self.get_logger().info(
            f"Episode end — encoding/saving {n} frames "
            f"(this can take 20–60 s for video; PNG mode is faster)..."
        )
        t0 = time.time()
        try:
            saved = self._writer.save_episode()
            self.get_logger().info(
                f"=== Episode saved: {saved} frames in {time.time() - t0:.1f}s ==="
            )
            self.episode_count += 1
        except Exception as e:
            self.get_logger().error(f"save_episode failed: {e}")
        finally:
            self._writer = None

    # ── per-tick sample ────────────────────────────────────────────
    def _tick(self) -> None:
        if not self._was_executing:
            return
        with self._lock:
            obs = self._latest.obs
            action = self._latest.action
        if obs is None or action is None:
            return
        if not self._ensure_writer(obs):
            return
        try:
            frame = {
                "task": self._task_prompt,
                "observation.images.left": image_msg_to_array(obs.left_image),
                "observation.images.center": image_msg_to_array(obs.center_image),
                "observation.images.right": image_msg_to_array(obs.right_image),
                "observation.state": observation_to_state(obs),
                "action": motion_update_to_action(action),
                "action.stiffness_diag": stiffness_to_diag(action),
            }
            self._writer.add_frame(frame)
        except Exception as e:
            self.get_logger().error(f"add_frame failed: {e}")


# ────────────────────── main ───────────────────────────────────────────────
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path("~/ws_aic/aic_data").expanduser()),
        help="Dataset root directory. A subdirectory per episode is created.",
    )
    parser.add_argument(
        "--task",
        default="insert cable",
        help="Natural-language task prompt written to each frame.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Sampling rate (Hz). Each tick writes a frame using the latest "
        "obs+action.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Stay alive after the first episode terminates and record the "
        "next STATUS_EXECUTING window as a new episode (Ctrl-C to stop).",
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        help="Video codec for LeRobotDataset video features. Default 'h264' "
        "(fast encode on consumer hardware). Use 'libsvtav1' for smaller "
        "files but ~3x slower encoding.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Store images as PNG-per-frame instead of MP4 video. Faster "
        "save (no encoder), bigger disk.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override the dataset subdirectory name. If set, the dataset is "
        "written to <root>/<name>/ instead of the default "
        "<root>/aic_recording_<TS>/. Use this to put the dataset under a "
        "run-output dir (single-tree layout). Multi-episode mode does not "
        "support --name (each episode would clash on the same path).",
    )
    args = parser.parse_args(argv)

    if args.name and args.multi:
        parser.error("--name cannot be combined with --multi (path would clash)")

    rclpy.init()
    node = AICAsyncRecorder(
        dataset_root=Path(args.root),
        task_prompt=args.task,
        fps=args.fps,
        multi_episode=args.multi,
        vcodec=args.vcodec,
        use_videos=not args.no_videos,
        fixed_name=args.name,
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # If we got Ctrl-C'd mid-episode, save what we have.
        if node._was_executing and node._writer is not None:
            node.get_logger().info("Interrupted mid-episode; flushing writer")
            node._close_episode()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
