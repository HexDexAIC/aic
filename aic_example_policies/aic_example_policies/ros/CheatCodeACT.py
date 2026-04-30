#
#  Copyright (C) 2026 Intrinsic Innovation LLC / HexDexAIC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#

"""Diagnostic hybrid policy: CheatCodeMJ does the approach (using GT port_tf),
then a locally-trained ACT does the descent.

Why: our analysis says the ACT baseline has *coarse* localization failure —
it doesn't know which port to aim at, so it descends at the dataset-mean
position and lands "between the ports". This policy isolates the question
"can ACT at least do precision descent if handed the correct hover pose?"
by feeding it the same hover pose CheatCodeMJ achieves with privileged TF.

  result = ACT_can_descend && CheatCodeMJ_approach_works
         ≠ deployable (still uses GT — eval-only diagnostic)

If a sweep with this policy shows insertions where pure RunACTLocal didn't,
we know the bottleneck is approach/localization. If it still fails, ACT also
can't do the descent and we have a deeper problem.

Inherits CheatCodeMJ (gets all params, TF lookups, helpers, the approach
phase loop), then in `insert_cable` we copy CheatCodeMJ's Phase A (approach)
and replace Phase B/C/D (descent + retry + release) with an ACT inference
loop.

Configuration via ROS params (set via --params-file or -p):
  All CheatCodeMJ params (approach_z_offset_*, approach_time_*, etc.)
  All RunACTLocal params (checkpoint_path, control_rate_hz, stiffness, ...)

Run with --ground-truth (engine flag) like CheatCodeMJ — TF frames are
required for the approach phase. record_episode.sh / run_scoring_loop.sh
auto-enable GT for any policy whose name starts with "CheatCode".
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import time
from pathlib import Path

from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, Wrench
from std_msgs.msg import Header
from tf2_ros import TransformException

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode

from aic_example_policies.ros.CheatCodeMJ import CheatCodeMJ, _scalar_trajectory


_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class CheatCodeACT(CheatCodeMJ):
    """CheatCodeMJ approach + locally-trained ACT descent."""

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        # Same lazy-import trick as RunACTLocal — heavy ML libs imported
        # inside __init__ so the import cost lands in on_configure (60 s
        # budget) not on the module import path (~13 s engine timeout).
        global json, torch, np, draccus
        global ACTPolicy, ACTConfig, load_file, snapshot_download
        import json
        import torch
        import numpy as np
        import draccus
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download

        node = parent_node

        def _p(name: str, default):
            node.declare_parameter(name, default)
            return node.get_parameter(name).value

        ckpt = _p("checkpoint_path", "")
        repo = _p("repo_id", "HexDexAIC/act-aic-sfp-500-v1")
        self._act_control_period = 1.0 / float(_p("act_control_rate_hz", 25.0))
        self._act_episode_timeout = float(_p("act_episode_timeout_s", 20.0))
        self._act_stiffness = np.asarray(
            list(_p("act_stiffness", [90.0, 90.0, 90.0, 50.0, 50.0, 50.0])),
            dtype=np.float32,
        )
        self._act_damping = np.asarray(
            list(_p("act_damping", [50.0, 50.0, 50.0, 20.0, 20.0, 20.0])),
            dtype=np.float32,
        )
        self._act_n_action_steps = int(_p("act_n_action_steps", 0))
        self._act_temporal_ensemble_coeff = float(_p("act_temporal_ensemble_coeff", -1.0))

        if ckpt:
            policy_path = Path(ckpt).expanduser().resolve()
            if not policy_path.is_dir():
                raise FileNotFoundError(f"checkpoint_path not found: {policy_path}")
            self.get_logger().info(f"Loading ACT from local: {policy_path}")
        else:
            policy_path = Path(
                snapshot_download(
                    repo_id=repo,
                    allow_patterns=[
                        "config.json",
                        "model.safetensors",
                        "policy_preprocessor*",
                        "policy_postprocessor*",
                        "train_config.json",
                    ],
                )
            )
            self.get_logger().info(f"Loaded ACT from HF {repo}: {policy_path}")

        self._act_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(policy_path / "config.json") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("type", None)
        if self._act_temporal_ensemble_coeff >= 0.0:
            cfg_dict["temporal_ensemble_coeff"] = self._act_temporal_ensemble_coeff
            cfg_dict["n_action_steps"] = 1

        config = draccus.decode(ACTConfig, cfg_dict)
        self._act_policy = ACTPolicy(config)
        self._act_policy.load_state_dict(
            load_file(policy_path / "model.safetensors"), strict=False
        )
        self._act_policy.eval()
        self._act_policy.to(self._act_device)

        if (self._act_temporal_ensemble_coeff < 0.0
                and self._act_n_action_steps > 0):
            old = self._act_policy.config.n_action_steps
            self._act_policy.config.n_action_steps = self._act_n_action_steps
            self.get_logger().info(
                f"act n_action_steps override: {old} → {self._act_n_action_steps}"
            )

        norm = load_file(
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )

        def _stat(key: str, shape: tuple):
            return norm[key].to(self._act_device).view(*shape)

        self._act_img_stats = {
            view: {
                "mean": _stat(f"observation.images.{view}.mean", (1, 3, 1, 1)),
                "std":  _stat(f"observation.images.{view}.std",  (1, 3, 1, 1)),
            }
            for view in ("left", "center", "right")
        }
        self._act_state_mean = _stat("observation.state.mean", (1, -1))
        self._act_state_std  = _stat("observation.state.std",  (1, -1))

        unnorm = load_file(
            policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
        self._act_action_mean = unnorm["action.mean"].to(self._act_device).view(1, -1)
        self._act_action_std  = unnorm["action.std"].to(self._act_device).view(1, -1)

        self.get_logger().info("CheatCodeACT ready (CheatCodeMJ approach + ACT descent).")

    # ──────────────── ACT-side helpers (mirror of RunACTLocal) ────────────────
    @staticmethod
    def _img_to_tensor(raw_img, device, mean, std):
        img = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )
        t = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .unsqueeze(0)
            .to(device)
        )
        return (t - mean) / std

    @staticmethod
    def _quat_to_rot6(qx, qy, qz, qw):
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        col0 = np.array([1 - 2*(yy + zz), 2*(xy + wz), 2*(xz - wy)], dtype=np.float32)
        col1 = np.array([2*(xy - wz), 1 - 2*(xx + zz), 2*(yz + wx)], dtype=np.float32)
        return np.concatenate([col0, col1])

    @staticmethod
    def _rot6_to_quat(rot6):
        a1 = rot6[:3]; a2 = rot6[3:]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        a2_proj = a2 - np.dot(b1, a2) * b1
        b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-8)
        b3 = np.cross(b1, b2)
        m = np.column_stack([b1, b2, b3]).astype(np.float64)
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            qw = 0.25 / s
            qx = (m[2, 1] - m[1, 2]) * s
            qy = (m[0, 2] - m[2, 0]) * s
            qz = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
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

    def _act_prepare_obs(self, obs_msg: Observation):
        out = {
            f"observation.images.{view}": self._img_to_tensor(
                getattr(obs_msg, f"{view}_image"),
                self._act_device,
                self._act_img_stats[view]["mean"],
                self._act_img_stats[view]["std"],
            )
            for view in ("left", "center", "right")
        }
        state_np = self._build_state(obs_msg)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self._act_device)
        out["observation.state"] = (state - self._act_state_mean) / self._act_state_std
        return out

    def _act_action_to_motion_update(self, action_norm) -> MotionUpdate:
        a_real = (action_norm * self._act_action_std) + self._act_action_mean
        a_np = a_real[0].detach().cpu().numpy()
        pos_xyz = a_np[:3]
        qx, qy, qz, qw = self._rot6_to_quat(a_np[3:9])
        return MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self.get_clock().now().to_msg(),
            ),
            pose=Pose(
                position=Point(x=float(pos_xyz[0]), y=float(pos_xyz[1]), z=float(pos_xyz[2])),
                orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
            ),
            target_stiffness=np.diag(self._act_stiffness).flatten().astype(np.float32),
            target_damping=np.diag(self._act_damping).flatten().astype(np.float32),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )

    # ──────────────── insert_cable: approach (CheatCodeMJ) → ACT ────────────────
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"CheatCodeACT.insert_cable() task: {task}")
        self._task = task

        # ── approach phase: copied from CheatCodeMJ.insert_cable ──────────
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        for frame in (port_frame, plug_frame):
            if not self._wait_for_tf("base_link", frame):
                return False

        try:
            port_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time()
            ).transform
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            return False

        # Pick hover height + timing from plug type (mirror of CheatCodeMJ).
        plug_type_lower = (task.plug_type or "").lower()
        if plug_type_lower == "sc":
            approach_z_offset = self._approach_z_offset_sc
            approach_time = self._approach_time_sc
        elif plug_type_lower == "sfp":
            approach_z_offset = self._approach_z_offset_sfp
            approach_time = self._approach_time_sfp
        else:
            approach_z_offset = self._approach_z_offset_sfp
            approach_time = self._approach_time_sc

        dt = 1.0 / self._control_rate_hz
        approach_traj = _scalar_trajectory(0.0, 1.0, approach_time)
        t0 = self.time_now()
        while True:
            elapsed = (self.time_now() - t0).nanoseconds / 1e9
            if elapsed >= approach_traj.duration + 0.5:
                break
            f, _, _ = approach_traj.get_state(elapsed)
            interp = float(f[0])
            try:
                pose = self.calc_gripper_pose(
                    port_tf,
                    slerp_fraction=interp,
                    position_fraction=interp,
                    z_offset=approach_z_offset,
                    reset_xy_integrator=True,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"Approach TF lookup failed: {ex}")
            self.sleep_for(dt)

        send_feedback("approach complete — handing off to ACT")
        self.get_logger().info("CheatCodeACT: approach phase done. Starting ACT descent.")

        # ── ACT descent phase ────────────────────────────────────────────
        self._act_policy.reset()
        t_act_start = time.time()
        n_steps = 0
        while time.time() - t_act_start < self._act_episode_timeout:
            loop_start = time.time()
            obs_msg = get_observation()
            if obs_msg is None:
                time.sleep(0.01)
                continue
            obs = self._act_prepare_obs(obs_msg)
            with torch.inference_mode():
                action_norm = self._act_policy.select_action(obs)
            move_robot(motion_update=self._act_action_to_motion_update(action_norm))
            n_steps += 1
            if n_steps % 25 == 0:
                send_feedback(f"act step {n_steps} t={time.time()-t_act_start:.1f}s")
            elapsed = time.time() - loop_start
            sleep_for = self._act_control_period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        self.get_logger().info(
            f"CheatCodeACT: ACT phase done after {n_steps} steps, "
            f"{time.time()-t_act_start:.2f}s wall."
        )
        return True
