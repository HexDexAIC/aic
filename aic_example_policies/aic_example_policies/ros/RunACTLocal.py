#
#  Copyright (C) 2026 Intrinsic Innovation LLC / HexDexAIC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#

"""ACT policy runner that loads a *locally-trained* checkpoint.

This deliberately deviates from the upstream RunACT (which loads
`grkw/aic_act_policy` from HF and uses a 7-D velocity action). Our
training pipeline (record_lerobot.py + lerobot 0.5.x) produces
checkpoints with this schema:

  observation.images.{left,center,right}  (3, 1024, 1152)
  observation.state                        (27,)  — see record_lerobot.observation_to_state
  action                                   (9,)   — TCP pos (3) + rot6 (6)
  action.stiffness_diag                    (6,)   — per-axis stiffness diag (recorded but NOT learned)

Stock lerobot ACT only fits the *primary* `action` feature: its head is
`Linear(dim_model, action_feature.shape[0])` and the loss is `l1(batch[ACTION], pred)`.
Auxiliary `action.*` features (like `action.stiffness_diag`) are normalized at
data-load time but never reach the loss, so the model never learns to predict
them. We publish a fixed default stiffness here; predicting variable stiffness
would require a multi-head ACT subclass — see [[lerobot-act-ignores-aux-action-features]].

We publish each step as a **MotionUpdate(POSITION mode)**: predicted (pos, rot6)
→ Pose, fixed stiffness/damping.

Configuration via ROS params (set via --params-file or -p):
  checkpoint_path     absolute path to a `pretrained_model/` dir on disk
  repo_id             HF repo id to download from (only used if checkpoint_path empty)
  control_rate_hz     action publish rate (default 25.0, matches CheatCodeMJ)
  episode_timeout_s   max seconds inside one insert_cable() call (default 30.0)
  stiffness           6-vec stiffness diag (default Policy.set_pose_target defaults)
  damping             6-vec damping diag (default Policy.set_pose_target defaults)
  n_action_steps      override the trained config's n_action_steps. The trained
                      ACT runs at chunk_size=100 / n_action_steps=100 — i.e.
                      *open-loop for 4 s* between inferences. Setting this to 1
                      forces re-inference every tick (true 25 Hz feedback);
                      4 → 6.25 Hz; 0 (default) → keep the trained value.
                      Closes the BC covariate-shift loop without retraining.
                      WARNING: bare n=1 throws away ACT's chunk-averaging and
                      typically scores WORSE than n=100 (the first action of
                      each chunk is the noisiest). Pair with
                      temporal_ensemble_coeff for the paper's recipe.
  temporal_ensemble_coeff
                      enables ACT's temporal ensembler. Re-infers at every
                      tick (forces n_action_steps = chunk_size = 100 internally),
                      keeps a sliding window of overlapping chunks, and
                      weight-averages them with weights w_i = exp(-coeff * i).
                      0.01 is the paper's default. Set <0 (default) to leave
                      `null` — i.e. ensembler disabled. Mutually exclusive
                      with the n_action_steps override (this wins).
"""

from __future__ import annotations

import os

# Faster HF downloads when a repo_id fallback is used.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

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


# Joint name order must match record_lerobot.JOINT_NAMES.
_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class RunACTLocal(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        # Defer heavy imports into __init__ (60 s on_configure budget) so a
        # ~16 s torch/lerobot import on the executor thread doesn't trip the
        # engine's GetState timeout. Same trick the upstream RunACT uses.
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
        self._control_period = 1.0 / float(_p("control_rate_hz", 25.0))
        self._episode_timeout = float(_p("episode_timeout_s", 30.0))
        # Defaults match Policy.set_pose_target — same impedance the
        # CheatCodeMJ training data was collected under.
        self._stiffness = np.asarray(
            list(_p("stiffness", [90.0, 90.0, 90.0, 50.0, 50.0, 50.0])),
            dtype=np.float32,
        )
        self._damping = np.asarray(
            list(_p("damping", [50.0, 50.0, 50.0, 20.0, 20.0, 20.0])),
            dtype=np.float32,
        )
        self._n_action_steps_override = int(_p("n_action_steps", 0))
        self._temporal_ensemble_coeff = float(_p("temporal_ensemble_coeff", -1.0))
        # Diagnostic: replace the normalized image tensor with zeros before
        # passing to the policy. This kills all visual information while
        # leaving proprio intact. If the resulting trial score matches the
        # natural-image baseline within noise, vision is being ignored.
        self._blank_images = bool(_p("blank_images", False))

        if ckpt:
            policy_path = Path(ckpt).expanduser().resolve()
            if not policy_path.is_dir():
                self.get_logger().warn(
                    f"checkpoint_path not found ({policy_path}) — falling back "
                    f"to HF repo_id={repo}"
                )
                ckpt = ""
            else:
                self.get_logger().info(f"Loading ACT from local: {policy_path}")
        if not ckpt:
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── policy weights ──────────────────────────────────────────────
        with open(policy_path / "config.json") as f:
            cfg_dict = json.load(f)
        cfg_dict.pop("type", None)  # draccus chokes on this hint field

        # Temporal ensembler must be set BEFORE ACTPolicy() because the policy
        # constructs the ensembler in __init__ based on this flag. It does NOT
        # affect the trained weights — it's an inference-time post-processor
        # that overlaps chunk predictions and weight-averages them.
        # ACTConfig validates n_action_steps == 1 when ensembler is on, so
        # we must set both at construction time (overriding the saved 100).
        if self._temporal_ensemble_coeff >= 0.0:
            cfg_dict["temporal_ensemble_coeff"] = self._temporal_ensemble_coeff
            cfg_dict["n_action_steps"] = 1

        config = draccus.decode(ACTConfig, cfg_dict)

        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(policy_path / "model.safetensors"), strict=False)
        self.policy.eval()
        self.policy.to(self.device)

        # n_action_steps override (only applies if temporal ensembler is OFF;
        # when ensembler is on, lerobot forces n_action_steps = chunk_size).
        if self._temporal_ensemble_coeff < 0.0 and self._n_action_steps_override > 0:
            old = self.policy.config.n_action_steps
            self.policy.config.n_action_steps = self._n_action_steps_override
            self.get_logger().info(
                f"n_action_steps override: {old} → {self._n_action_steps_override} "
                f"(re-infer every {self._n_action_steps_override / 25.0:.2f} s @ 25 Hz)"
            )
        if self._temporal_ensemble_coeff >= 0.0:
            self.get_logger().info(
                f"temporal_ensemble_coeff = {self._temporal_ensemble_coeff} "
                f"(re-infer every tick, weighted-average over chunk_size=100 chunks)"
            )
        if self._blank_images:
            self.get_logger().warn(
                "blank_images = True — feeding zero tensors instead of camera "
                "images. Diagnostic mode only."
            )

        # ── normalizer stats (input side) ───────────────────────────────
        # safetensors keys follow `<feature_name>.<mean|std>` convention.
        norm = load_file(
            policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )

        def _stat(key: str, shape: tuple):
            return norm[key].to(self.device).view(*shape)

        self._img_stats = {
            view: {
                "mean": _stat(f"observation.images.{view}.mean", (1, 3, 1, 1)),
                "std":  _stat(f"observation.images.{view}.std",  (1, 3, 1, 1)),
            }
            for view in ("left", "center", "right")
        }
        self._state_mean = _stat("observation.state.mean", (1, -1))
        self._state_std  = _stat("observation.state.std",  (1, -1))

        # ── un-normalizer stats (output side) ──────────────────────────
        # Only the primary `action` feature is predicted by ACT — see module
        # docstring. action.stiffness_diag stats exist in the safetensors but
        # the model never learned to predict them.
        unnorm = load_file(
            policy_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
        self._action_mean = unnorm["action.mean"].to(self.device).view(1, -1)
        self._action_std  = unnorm["action.std"].to(self.device).view(1, -1)

        # Sanity: action shape must be 9 to match (pos3 + rot6).
        if self._action_mean.numel() != 9:
            raise RuntimeError(
                f"expected 9-D action (pos3 + rot6), got {self._action_mean.numel()}-D — "
                "checkpoint schema mismatch"
            )

        self.get_logger().info("RunACTLocal ready.")

    # ───────────────────── helpers ────────────────────────────────────
    @staticmethod
    def _img_to_tensor(raw_img, device, mean, std):
        """ROS Image (HWC uint8) → (1, 3, H, W) normalized float on device.

        Cameras publish at native 1152×1024 which matches the trained shape
        — no resize. If a future training run scales differently, add a
        cv2.resize step here keyed off mean.shape[-2:].
        """
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
        """Quaternion → first two columns of rotation matrix (Zhou 2019, 6-D)."""
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        wx, wy, wz = qw * qx, qw * qy, qw * qz
        col0 = np.array([1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy)], dtype=np.float32)
        col1 = np.array([2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx)], dtype=np.float32)
        return np.concatenate([col0, col1])

    @staticmethod
    def _rot6_to_quat(rot6):
        """6-D rotation rep → unit quaternion (Gram-Schmidt then matrix→quat).

        rot6 = [a1, a2] where a1, a2 are the first two columns of the rotation
        matrix (concatenated). a1 is normalized; a2 is orthogonalized against
        a1 and normalized; a3 = a1 × a2.
        """
        a1 = rot6[:3]
        a2 = rot6[3:]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        a2_proj = a2 - np.dot(b1, a2) * b1
        b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-8)
        b3 = np.cross(b1, b2)
        m = np.column_stack([b1, b2, b3]).astype(np.float64)  # 3x3 rotation
        # Shepperd's method (numerically stable matrix→quaternion)
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
        """Reconstruct the 27-dim state vector — must match record_lerobot.observation_to_state."""
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

    def _prepare_obs(self, obs_msg: Observation):
        out = {
            f"observation.images.{view}": self._img_to_tensor(
                getattr(obs_msg, f"{view}_image"),
                self.device,
                self._img_stats[view]["mean"],
                self._img_stats[view]["std"],
            )
            for view in ("left", "center", "right")
        }
        if self._blank_images:
            for k in list(out.keys()):
                out[k] = torch.zeros_like(out[k])
        state_np = self._build_state(obs_msg)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        out["observation.state"] = (state - self._state_mean) / self._state_std
        return out

    def _action_to_motion_update(self, action_norm) -> MotionUpdate:
        """Un-normalize the (1, 9) normalized action tensor and pack a MotionUpdate."""
        a_real = (action_norm * self._action_std) + self._action_mean
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

    # ───────────────────── insert_cable entry point ───────────────────
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.policy.reset()  # clear ACT's internal action queue
        self.get_logger().info(f"RunACTLocal.insert_cable() enter. Task: {task}")

        # Use ROS sim time (use_sim_time:=true) so timing works correctly
        # even when sim runs slower than real-time. With wall-time, if sim
        # runs at 1/Nth real-time, publish rate is N× higher than intended,
        # producing wildly excessive jerk (138 vs 5.57 m/s³) and the arm
        # never reaches the port. Use the parent node's clock so policy
        # keeps trained inference cadence in sim seconds.
        clock = self._parent_node.get_clock()
        t0_sim = clock.now().nanoseconds / 1e9
        t0_wall = time.time()
        n_steps = 0
        last_step_sim = t0_sim

        while True:
            now_sim = clock.now().nanoseconds / 1e9
            if now_sim - t0_sim >= self._episode_timeout:
                break

            loop_start_sim = now_sim

            obs_msg = get_observation()
            if obs_msg is None:
                # observation publisher hasn't started yet — back off briefly
                time.sleep(0.01)
                continue

            obs = self._prepare_obs(obs_msg)
            with torch.inference_mode():
                action_norm = self.policy.select_action(obs)  # (1, 9)

            motion_update = self._action_to_motion_update(action_norm)
            move_robot(motion_update=motion_update)
            n_steps += 1

            if n_steps % 25 == 0:
                send_feedback(
                    f"step {n_steps} sim_t={now_sim - t0_sim:.1f}s "
                    f"wall_t={time.time() - t0_wall:.1f}s"
                )

            # Sleep until the next sim-time tick. If sim runs faster than
            # real-time, this falls through; if slower, we busy-wait
            # comparing sim clock so we stay synced to sim cadence.
            target_sim = loop_start_sim + self._control_period
            while True:
                cur_sim = clock.now().nanoseconds / 1e9
                if cur_sim >= target_sim:
                    break
                # Sleep at most 1/2 control period; recheck sim clock often.
                time.sleep(min(self._control_period * 0.5, max(0.0, target_sim - cur_sim)))

        wall_elapsed = time.time() - t0_wall
        sim_elapsed = clock.now().nanoseconds / 1e9 - t0_sim
        self.get_logger().info(
            f"RunACTLocal.insert_cable() exit after {n_steps} steps, "
            f"{sim_elapsed:.2f}s sim ({wall_elapsed:.2f}s wall)."
        )
        return True
