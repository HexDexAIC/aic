#
#  Copyright (C) 2026 HexDexAIC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#

"""ACT-PR deployment policy.

Loads an ACT-PR checkpoint (port-frame residual action + auxiliary
port-pose head trained on the same image/state inputs as RunACTLocal),
then at each tick:

  1. Run the policy → get (a_residual, p_port_pred, rot6_port_pred)
  2. Un-normalize both heads using the postprocessor stats.
  3. Compose the absolute pose target:
        T_action_abs = T_port_pred · T_residual
        p_action_abs = R_port_pred · p_residual + p_port_pred
        R_action_abs = R_port_pred · R_residual
  4. Publish MotionUpdate(POSITION mode) with that absolute pose.

No detector dependency at eval — the predicted port pose comes from the
auxiliary head, which was supervised at training time with GT port pose
from the bagged TF data.

Deviates from RunACTLocal in:
  - Loads the act_pr_policy module from ../scripts/ (not in the pixi env
    package; we sys.path.insert at policy init).
  - Action composition step before MotionUpdate.
  - Optional safety guard on lateral target jumps (per ChatGPT's plan
    Phase 5: clamp moves > lateral_jump_limit_m to prevent one bad port
    prediction from slewing the arm into the board).

Configuration (ROS params):
  checkpoint_path        local pretrained_model/ dir
  repo_id                HF model fallback (default HexDexAIC/act-pr-aic-sfp-500-v1)
  control_rate_hz        25.0
  episode_timeout_s      30.0
  stiffness / damping    fixed impedance (same as RunACTLocal)
  lateral_jump_limit_m   max allowed XY jump per tick (default 0.05 = 5 cm)
                         set to 0 to disable the guard.
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import sys
import time
from pathlib import Path
from typing import Optional

from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Wrench
from std_msgs.msg import Header

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode


_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class RunACTPR(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        # Defer heavy imports into __init__ (60s on_configure budget).
        global json, torch, np, draccus
        global ACTPRPolicy, ACTPRConfig, load_file, snapshot_download, PORT_POSE_KEY
        import json
        import torch
        import numpy as np
        import draccus
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download

        # Make act_pr_policy.py importable. It lives in src/aic/scripts/
        # alongside the trainer; the deployed policy file lives in the
        # aic_example_policies package. Resolve relative to this file.
        scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from act_pr_policy import ACTPRPolicy, ACTPRConfig, PORT_POSE_KEY  # noqa

        node = parent_node
        def _p(name, default):
            node.declare_parameter(name, default)
            return node.get_parameter(name).value

        ckpt = _p("checkpoint_path", "")
        repo = _p("repo_id", "HexDexAIC/act-pr-aic-sfp-500-v1")
        self._control_period = 1.0 / float(_p("control_rate_hz", 25.0))
        self._episode_timeout = float(_p("episode_timeout_s", 30.0))
        self._stiffness = np.asarray(
            list(_p("stiffness", [90.0, 90.0, 90.0, 50.0, 50.0, 50.0])),
            dtype=np.float32,
        )
        self._damping = np.asarray(
            list(_p("damping", [50.0, 50.0, 50.0, 20.0, 20.0, 20.0])),
            dtype=np.float32,
        )
        self._lateral_jump_limit = float(_p("lateral_jump_limit_m", 0.05))

        if ckpt:
            policy_path = Path(ckpt).expanduser().resolve()
            if not policy_path.is_dir():
                raise FileNotFoundError(f"checkpoint_path not found: {policy_path}")
            self.get_logger().info(f"Loading ACT-PR from local: {policy_path}")
        else:
            policy_path = Path(snapshot_download(
                repo_id=repo,
                allow_patterns=[
                    "config.json", "model.safetensors",
                    "policy_preprocessor*", "policy_postprocessor*",
                ],
            ))
            self.get_logger().info(f"Loaded ACT-PR from HF {repo}: {policy_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build policy from saved config.
        with open(policy_path / "config.json") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("type", None)
        config = draccus.decode(ACTPRConfig, cfg_dict)
        self.policy = ACTPRPolicy(config)
        self.policy.load_state_dict(load_file(policy_path / "model.safetensors"), strict=False)
        self.policy.eval().to(self.device)

        # Stats for un-normalization (action: residual; port_pose: GT port).
        stats_pre = load_file(
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        ) if (policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors").exists() else None
        stats_post = load_file(
            policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        ) if (policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").exists() else None

        # Manual loading of stats from a sibling stats.json if pre/post stats files
        # weren't generated at save time (custom training script writes pretrained_model/
        # via lerobot's save_pretrained, but does NOT necessarily write the
        # processor safetensors). Fallback: read from train_actpr's saved stats.
        if stats_post is None:
            stats_path = policy_path.parent.parent / "stats.json"
            if not stats_path.exists():
                stats_path = policy_path / "stats.json"
            if stats_path.exists():
                stats_json = json.loads(stats_path.read_text())
                self._action_mean = torch.tensor(stats_json["action"]["mean"]).to(self.device).view(1, -1)
                self._action_std  = torch.tensor(stats_json["action"]["std"]).to(self.device).view(1, -1)
                self._port_mean   = torch.tensor(stats_json["observation.port_pose_gt"]["mean"]).to(self.device).view(1, -1)
                self._port_std    = torch.tensor(stats_json["observation.port_pose_gt"]["std"]).to(self.device).view(1, -1)
            else:
                raise FileNotFoundError(
                    f"No normalizer stats found near {policy_path}. "
                    "Expected stats.json from train_actpr or processor safetensors."
                )
        else:
            self._action_mean = stats_post["action.mean"].to(self.device).view(1, -1)
            self._action_std  = stats_post["action.std"].to(self.device).view(1, -1)
            self._port_mean   = stats_post["observation.port_pose_gt.mean"].to(self.device).view(1, -1)
            self._port_std    = stats_post["observation.port_pose_gt.std"].to(self.device).view(1, -1)

        # ImageNet normalization for vision input.
        self._img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self._img_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # State stats — load from stats.json (same source as port stats).
        try:
            stats_path = policy_path.parent.parent / "stats.json"
            if not stats_path.exists():
                stats_path = policy_path / "stats.json"
            stats_json = json.loads(stats_path.read_text())
            self._state_mean = torch.tensor(stats_json["observation.state"]["mean"]).to(self.device).view(1, -1)
            self._state_std  = torch.tensor(stats_json["observation.state"]["std"]).to(self.device).view(1, -1)
        except Exception:
            # If observation.state stats aren't in the saved stats.json,
            # use zeros / ones as a degenerate identity normalization.
            self._state_mean = torch.zeros(1, 27).to(self.device)
            self._state_std  = torch.ones(1, 27).to(self.device)
            self.get_logger().warn(
                "No observation.state stats found — using identity normalization. "
                "This will hurt accuracy."
            )

        self.get_logger().info("RunACTPR ready.")

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _quat_to_rot6(qx, qy, qz, qw):
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        col0 = np.array([1 - 2*(yy + zz), 2*(xy + wz), 2*(xz - wy)], dtype=np.float32)
        col1 = np.array([2*(xy - wz), 1 - 2*(xx + zz), 2*(yz + wx)], dtype=np.float32)
        return np.concatenate([col0, col1])

    @staticmethod
    def _rot6_to_R(rot6):
        a1, a2 = rot6[:3], rot6[3:]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        a2_proj = a2 - np.dot(b1, a2) * b1
        b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-8)
        b3 = np.cross(b1, b2)
        return np.column_stack([b1, b2, b3])

    @staticmethod
    def _R_to_quat(R):
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return float(qx), float(qy), float(qz), float(qw)

    def _build_state(self, obs: Observation):
        cs = obs.controller_state
        p, q = cs.tcp_pose.position, cs.tcp_pose.orientation
        lv, av = cs.tcp_velocity.linear, cs.tcp_velocity.angular
        rot6 = self._quat_to_rot6(q.x, q.y, q.z, q.w)
        wf = obs.wrist_wrench.wrench.force
        wt = obs.wrist_wrench.wrench.torque
        name_to_pos = dict(zip(obs.joint_states.name, obs.joint_states.position))
        if all(n in name_to_pos for n in _JOINT_NAMES):
            joints = np.array([name_to_pos[n] for n in _JOINT_NAMES], dtype=np.float32)
        else:
            joints = np.array(list(obs.joint_states.position[:6]), dtype=np.float32)
        return np.concatenate([
            np.array([p.x, p.y, p.z], dtype=np.float32),
            rot6.astype(np.float32),
            np.array([lv.x, lv.y, lv.z], dtype=np.float32),
            np.array([av.x, av.y, av.z], dtype=np.float32),
            np.array([wf.x, wf.y, wf.z, wt.x, wt.y, wt.z], dtype=np.float32),
            joints,
        ])

    def _img_to_tensor(self, raw_img):
        img = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        t = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        return (t - self._img_mean) / self._img_std

    def _prepare_obs(self, obs_msg: Observation):
        out = {}
        for view in ("left", "center", "right"):
            out[f"observation.images.{view}"] = self._img_to_tensor(
                getattr(obs_msg, f"{view}_image")
            )
        state_np = self._build_state(obs_msg)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        out["observation.state"] = (state - self._state_mean) / self._state_std
        # Aux feature: at inference, the policy uses ITS prediction, but the
        # forward pipeline still expects the key to exist (zeros are fine —
        # the head ignores it at inference, only used for training loss).
        out[PORT_POSE_KEY] = torch.zeros(1, 9, device=self.device)
        return out

    def _compose_motion_update(
        self, residual_norm, port_norm, current_tcp_xy
    ) -> MotionUpdate:
        """Un-normalize both heads, compose, build MotionUpdate. Apply
        lateral-jump guard if enabled."""
        # Un-normalize residual action (1, 9)
        a = (residual_norm * self._action_std) + self._action_mean
        a_np = a[0].detach().cpu().numpy()
        p_resid = a_np[:3]
        rot6_resid = a_np[3:9]
        R_resid = self._rot6_to_R(rot6_resid)

        # Un-normalize predicted port pose (1, 9)
        pp = (port_norm * self._port_std) + self._port_mean
        pp_np = pp[0].detach().cpu().numpy()
        p_port = pp_np[:3]
        rot6_port = pp_np[3:9]
        R_port = self._rot6_to_R(rot6_port)

        # Compose: T_abs = T_port · T_residual
        p_abs = R_port @ p_resid + p_port
        R_abs = R_port @ R_resid
        qx, qy, qz, qw = self._R_to_quat(R_abs)

        # Lateral jump guard.
        if self._lateral_jump_limit > 0.0 and current_tcp_xy is not None:
            dx, dy = p_abs[0] - current_tcp_xy[0], p_abs[1] - current_tcp_xy[1]
            d = float(np.hypot(dx, dy))
            if d > self._lateral_jump_limit:
                scale = self._lateral_jump_limit / d
                p_abs[0] = current_tcp_xy[0] + dx * scale
                p_abs[1] = current_tcp_xy[1] + dy * scale
                self.get_logger().warn(
                    f"lateral jump {d:.3f}m clamped to {self._lateral_jump_limit:.3f}m"
                )

        return MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self.get_clock().now().to_msg(),
            ),
            pose=Pose(
                position=Point(x=float(p_abs[0]), y=float(p_abs[1]), z=float(p_abs[2])),
                orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
            ),
            target_stiffness=np.diag(self._stiffness).flatten().astype(np.float32),
            target_damping=np.diag(self._damping).flatten().astype(np.float32),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

    # ── insert_cable ─────────────────────────────────────────────
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.policy.reset()
        self.get_logger().info(f"RunACTPR.insert_cable() enter. Task: {task}")

        t0 = time.time()
        n_steps = 0
        while time.time() - t0 < self._episode_timeout:
            loop_start = time.time()
            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(0.01)
                continue

            obs = self._prepare_obs(obs_msg)
            with torch.inference_mode():
                a_norm = self.policy.select_action(obs)               # (1, 9)
                p_norm = self.policy.get_last_predicted_port_pose_norm()  # (1, 9)

            current_tcp = obs_msg.controller_state.tcp_pose.position
            mu = self._compose_motion_update(
                a_norm, p_norm, (current_tcp.x, current_tcp.y)
            )
            move_robot(motion_update=mu)
            n_steps += 1
            if n_steps % 25 == 0:
                send_feedback(f"step {n_steps} t={time.time()-t0:.1f}s")

            elapsed = time.time() - loop_start
            sleep_for = self._control_period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        self.get_logger().info(
            f"RunACTPR.insert_cable() exit after {n_steps} steps, "
            f"{time.time()-t0:.2f}s wall."
        )
        return True
