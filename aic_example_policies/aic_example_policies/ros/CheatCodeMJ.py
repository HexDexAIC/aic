#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


# CheatCodeMJ — CheatCode with min-jerk timing.
#
# Reuses CheatCode.calc_gripper_pose() unchanged (including its XY integrator
# and live plug/gripper TF lookups). The only difference is the two scalar
# inputs — interp_fraction (approach) and z_offset (descent) — are now driven
# by hebi min-jerk trajectories instead of discrete per-step increments.
#
# That gives:
#   - Smooth acceleration/deceleration on both scalars → lower jerk, smoother
#     motion the controller can actually track.
#   - Tunable phase durations at one place each (APPROACH_TIME, DESCENT_TIME).
#   - Preserved Tier-3 success: the XY integrator that made CheatCode work is
#     still active during descent.
#
# Requires ground-truth TF. Launch with `ground_truth:=true`.


import csv
import datetime
import os
from pathlib import Path

import numpy as np
import hebi

from aic_example_policies.ros.CheatCode import CheatCode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from tf2_ros import TransformException


# ---------------------------------------------------------------------------
# Tuning knobs — all times in seconds, all lengths in metres.
# ---------------------------------------------------------------------------
# Approach phase: smoothly interpolate from current TCP pose to "pre-hover"
# pose (above port at z_offset = APPROACH_Z_OFFSET). interp_fraction goes
# 0 → 1 on a min-jerk profile over APPROACH_TIME. Slerp tracks the same clock.
APPROACH_TIME = 4.0               # v4: 4.0 kept — extra second for SC's larger rotation & lateral distance
# Hover height depends on plug type. SFP inserts into SFP ports on tall NIC
# cards, so we need 20 cm to clear the card body. SC inserts into small rail-
# mounted ports near the board surface — no tall obstacles to clear — so a
# lower hover saves time and gives the integrator less window to wind up.
APPROACH_Z_OFFSET_SFP = 0.20
APPROACH_Z_OFFSET_SC  = 0.20  # v7 revert: v6 tried 0.10, but shorter hover reduced XY integrator settling time → T3 worse

# Descent phase: z_offset falls from APPROACH_Z_OFFSET → -INSERTION_DEPTH on a
# min-jerk profile over DESCENT_TIME. XY integrator (inside calc_gripper_pose)
# is active and corrects lateral drift as we descend.
DESCENT_TIME = 12.0               # v4: 12.0 kept — slower descent, more integrator settling time
INSERTION_DEPTH = 0.015           # v5 REVERT to 0.015 (v4 tried 0.020 — likely contributed to trial 1's 66 N impact)

SETTLE_TIME = 2.0                 # v4: 2.0 kept
CONTROL_RATE_HZ = 20.0

LOG_ENABLED = os.environ.get("CHEATCODE_MJ_LOG", "1") != "0"


def _scalar_trajectory(start: float, end: float, duration: float):
    """Min-jerk trajectory on a single scalar, endpoints pinned at rest."""
    times = np.array([0.0, duration], dtype=np.float64)
    positions = np.array([[start, end]], dtype=np.float64)
    velocities = np.array([[0.0, 0.0]], dtype=np.float64)
    return hebi.trajectory.create_trajectory(times, positions, velocity=velocities)


class CheatCodeMJ(CheatCode):
    """CheatCode with trajectory-driven timing on interp_fraction and z_offset."""

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _log_dir() -> Path:
        base = Path(os.environ.get("AIC_RESULTS_DIR", str(Path.home() / "aic_results")))
        d = base / "cheatcode_mj"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _open_logs(self, task: Task):
        if not LOG_ENABLED:
            return None, None, None
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"{stamp}_{task.id}_{task.plug_name}_to_{task.target_module_name}"
        summary_path = self._log_dir() / f"{tag}_summary.log"
        csv_path = self._log_dir() / f"{tag}_trajectory.csv"
        summary = open(summary_path, "w")
        csvf = open(csv_path, "w", newline="")
        writer = csv.writer(csvf)
        writer.writerow(
            [
                "t_elapsed_s",
                "phase",            # "approach" or "descent"
                "interp_fraction",  # 0..1 during approach, 1 during descent
                "z_offset",         # current z_offset scalar
                "plug_actual_x", "plug_actual_y", "plug_actual_z",
                "tcp_actual_x", "tcp_actual_y", "tcp_actual_z",
                "port_xy_err_x", "port_xy_err_y",  # live plug vs port xy
                "integ_x", "integ_y",
            ]
        )
        self.get_logger().info(
            f"CheatCodeMJ logging → {summary_path.name} + {csv_path.name}"
        )
        return summary, writer, csvf

    @staticmethod
    def _write_summary(summary, header: str, **data):
        if summary is None:
            return
        summary.write(f"=== {header} ===\n")
        for k, v in data.items():
            summary.write(f"{k}: {v}\n")
        summary.write("\n")
        summary.flush()

    def _log_row(self, writer, elapsed, phase, interp_fraction, z_offset, port_xyz):
        if writer is None:
            return
        try:
            plug_now = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                f"{self._task.cable_name}/{self._task.plug_name}_link",
                self._parent_node.get_clock().now().to_msg().__class__(sec=0, nanosec=0),
            ).transform if False else self._lookup(
                "base_link",
                f"{self._task.cable_name}/{self._task.plug_name}_link",
            )
            gripper_now = self._lookup("base_link", "gripper/tcp")
            writer.writerow(
                [
                    f"{elapsed:.4f}",
                    phase,
                    f"{interp_fraction:.4f}",
                    f"{z_offset:.5f}",
                    plug_now.translation.x, plug_now.translation.y, plug_now.translation.z,
                    gripper_now.translation.x, gripper_now.translation.y, gripper_now.translation.z,
                    port_xyz[0] - plug_now.translation.x,
                    port_xyz[1] - plug_now.translation.y,
                    self._tip_x_error_integrator,
                    self._tip_y_error_integrator,
                ]
            )
        except TransformException:
            pass

    def _lookup(self, target_frame, source_frame):
        from rclpy.time import Time
        return self._parent_node._tf_buffer.lookup_transform(
            target_frame, source_frame, Time()
        ).transform

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        from rclpy.time import Time  # local to avoid confusing name clashes

        self.get_logger().info(f"CheatCodeMJ.insert_cable() task: {task}")
        self._task = task

        summary, writer, csvf = self._open_logs(task)

        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        for frame in (port_frame, plug_frame):
            if not self._wait_for_tf("base_link", frame):
                if summary: summary.close()
                if csvf: csvf.close()
                return False

        try:
            port_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time()
            ).transform
        except TransformException as ex:
            self.get_logger().error(f"Could not look up port transform: {ex}")
            if summary: summary.close()
            if csvf: csvf.close()
            return False

        port_xyz = (
            port_tf.translation.x, port_tf.translation.y, port_tf.translation.z,
        )

        # Pick hover height by plug type. Default to SFP (tallest) for
        # unrecognised types so we err on the side of more clearance.
        plug_type_lower = (task.plug_type or "").lower()
        if plug_type_lower == "sc":
            approach_z_offset = APPROACH_Z_OFFSET_SC
        else:
            approach_z_offset = APPROACH_Z_OFFSET_SFP

        self._write_summary(
            summary,
            "Task",
            task_id=task.id,
            cable_name=task.cable_name,
            plug_name=task.plug_name,
            port_name=task.port_name,
            target_module_name=task.target_module_name,
            time_limit=task.time_limit,
        )
        self._write_summary(
            summary,
            "Initial TF inputs (base_link)",
            port_xyz=port_xyz,
            port_quat_wxyz=(port_tf.rotation.w, port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z),
        )
        self._write_summary(
            summary,
            "Schedule",
            approach_time_s=APPROACH_TIME,
            approach_z_offset_m=approach_z_offset,  # chosen by plug_type
            plug_type=plug_type_lower,
            descent_time_s=DESCENT_TIME,
            insertion_depth_m=INSERTION_DEPTH,
            settle_time_s=SETTLE_TIME,
            control_rate_hz=CONTROL_RATE_HZ,
        )

        dt = 1.0 / CONTROL_RATE_HZ

        # ============================================================
        # Phase A — approach: interp_fraction 0 → 1 on min-jerk.
        # calc_gripper_pose(reset_xy_integrator=True) during this phase so
        # the integrator stays at zero until we're hovering above the port.
        # ============================================================
        approach_traj = _scalar_trajectory(0.0, 1.0, APPROACH_TIME)
        t0 = self.time_now()
        while True:
            elapsed = (self.time_now() - t0).nanoseconds / 1e9
            if elapsed >= approach_traj.duration:
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
            self._log_row(writer, elapsed, "approach", interp, approach_z_offset, port_xyz)
            self.sleep_for(dt)

        send_feedback("approach complete, starting descent")

        # ============================================================
        # Phase B — descent: z_offset goes APPROACH_Z_OFFSET → -INSERTION_DEPTH
        # on min-jerk. interp_fraction is pinned at 1 (orientation already
        # aligned). XY integrator active: calc_gripper_pose applies live
        # port_xy vs plug_xy correction exactly like CheatCode.
        # ============================================================
        descent_traj = _scalar_trajectory(
            approach_z_offset, -INSERTION_DEPTH, DESCENT_TIME
        )
        t0 = self.time_now()
        last_z_offset = approach_z_offset
        while True:
            elapsed = (self.time_now() - t0).nanoseconds / 1e9
            if elapsed >= descent_traj.duration:
                break
            z, _, _ = descent_traj.get_state(elapsed)
            z_offset = float(z[0])
            last_z_offset = z_offset
            try:
                pose = self.calc_gripper_pose(
                    port_tf,
                    slerp_fraction=1.0,
                    position_fraction=1.0,
                    z_offset=z_offset,
                    reset_xy_integrator=False,
                )
                self.set_pose_target(move_robot=move_robot, pose=pose)
            except TransformException as ex:
                self.get_logger().warn(f"Descent TF lookup failed: {ex}")
            self._log_row(writer, elapsed, "descent", 1.0, z_offset, port_xyz)
            self.sleep_for(dt)

        # Settle at the final z_offset.
        self.get_logger().info("Descent complete, settling...")
        self.sleep_for(SETTLE_TIME)

        # Final snapshot
        try:
            plug_final = self._lookup("base_link", plug_frame)
            plug_port_distance = float(
                np.linalg.norm(
                    np.array([
                        plug_final.translation.x - port_xyz[0],
                        plug_final.translation.y - port_xyz[1],
                        plug_final.translation.z - port_xyz[2],
                    ])
                )
            )
            self._write_summary(
                summary,
                "Final state",
                final_plug_xyz=(plug_final.translation.x, plug_final.translation.y, plug_final.translation.z),
                plug_port_distance_m=plug_port_distance,
                final_z_offset=last_z_offset,
                final_integrator_xy=(self._tip_x_error_integrator, self._tip_y_error_integrator),
            )
            self.get_logger().info(
                f"CheatCodeMJ done. plug-port dist: {plug_port_distance:.4f}m"
            )
        except TransformException:
            pass

        if summary: summary.close()
        if csvf: csvf.close()

        return True
