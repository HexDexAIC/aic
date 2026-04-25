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
from collections import deque
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
from geometry_msgs.msg import Point, Pose, Quaternion
from tf2_ros import TransformException


# ---------------------------------------------------------------------------
# Tuning knobs — all times in seconds, all lengths in metres.
# ---------------------------------------------------------------------------
# Approach phase: smoothly interpolate from current TCP pose to "pre-hover"
# pose (above port at z_offset = APPROACH_Z_OFFSET). interp_fraction goes
# 0 → 1 on a min-jerk profile over APPROACH_TIME. Slerp tracks the same clock.
# v9: per-plug timings. SFP trials are deterministic (stdev ≤ 0.05, 6/6 full
# insertions) so we can trim without risking success rate. SC stays slow
# because its 400× variance lives in integrator settling time.
APPROACH_TIME_SFP = 3.5
APPROACH_TIME_SC  = 4.0            # v4: 4.0 kept — extra second for SC's larger rotation & lateral distance
# Hover height depends on plug type. SFP inserts into SFP ports on tall NIC
# cards, so we need 20 cm to clear the card body. SC inserts into small rail-
# mounted ports near the board surface — no tall obstacles to clear — so a
# lower hover saves time and gives the integrator less window to wind up.
APPROACH_Z_OFFSET_SFP = 0.20
APPROACH_Z_OFFSET_SC  = 0.12  # v9: lowered from 0.20. v6 tried 0.10 → T3 worse (less integrator settling); 0.12 is closer to 0.10 but above the known regression point

# Descent phase: z_offset falls from APPROACH_Z_OFFSET → -INSERTION_DEPTH on a
# min-jerk profile over DESCENT_TIME. XY integrator (inside calc_gripper_pose)
# is active and corrects lateral drift as we descend.
DESCENT_TIME_SFP = 9.0
DESCENT_TIME_SC  = 12.0            # v4: 12.0 kept — slower descent, more integrator settling time
INSERTION_DEPTH = 0.015           # v5 REVERT to 0.015 (v4 tried 0.020 — likely contributed to trial 1's 66 N impact)

SETTLE_TIME_SFP = 1.5
SETTLE_TIME_SC  = 2.0              # v4: 2.0 kept
CONTROL_RATE_HZ = 20.0

# v10: insertion success detection + retry. After the descent-then-settle
# phase, measure plug-port distance via ground-truth TF; if it's below the
# threshold the plug is considered seated. Otherwise lift back to hover,
# re-snapshot port_tf, reset the XY integrator, and re-descend — up to
# max_insertion_retries additional attempts. Tunable via ROS parameters
# declared in __init__ so we don't have to re-edit + reinstall to sweep.
DEFAULT_INSERTION_THRESHOLD_M = 0.005
DEFAULT_MAX_INSERTION_RETRIES = 2  # 1 initial + 2 retries = 3 attempts
DEFAULT_LIFT_TIME_FRAC = 0.5  # lift back to hover takes this fraction of descent_time
DEFAULT_HOVER_HOLD_BETWEEN_ATTEMPTS_S = 0.5  # brief steady-state at hover before retry descent

# Bad-offset injection defaults. Non-zero by default so the retry path is
# exercised on every run — useful while we iterate on the algorithm.
# Override with -p bad_port_offset_x:=0.0 (or `--bad-port-offset-x 0`) to
# disable. Decay <1 lets later retries see a smaller offset → exercises
# the "retry recovers" path; 1.0 means all attempts see the full offset
# (all-fail testing).
DEFAULT_BAD_PORT_OFFSET_X = 0.002
DEFAULT_BAD_PORT_OFFSET_Y = 0.0
DEFAULT_BAD_OFFSET_DECAY_PER_RETRY = 0.5

# Early-abort during descent. Sample plug-port distance every control tick;
# once we're past stuck_min_fraction of the descent, look at the recent
# stuck_window_s of samples — if the net distance change over that window
# is less than stuck_progress_m, the plug isn't getting closer to the real
# port and we bail out of this attempt early.
#
# stuck_min_fraction is REQUIRED to avoid false-triggering during the
# min-jerk ramp-up: at fraction 0.10 of descent the plug has only moved
# ~0.6mm — well below stuck_progress_m (2mm) — but it's healthy ramp-up
# motion, not stuck. The gate prevents the algorithm from looking at the
# pre-ramp window. Conservative default 0.3.
DEFAULT_STUCK_MIN_FRACTION = 0.2
DEFAULT_STUCK_WINDOW_S = 1.0
DEFAULT_STUCK_PROGRESS_M = 0.002

# v8: final "release" message before insert_cable returns.
# aic_controller holds last_tool_reference_ through entity deletion; on cable
# delete the impedance loop jolts toward that stale target. Publishing a
# hold-current-pose-with-low-stiffness MotionUpdate first makes the controller
# compliant through the engine's teardown window (delete entities → deactivate
# → reset_joints → reactivate).
RELEASE_STIFFNESS = [75.0, 75.0, 75.0, 75.0, 75.0, 75.0]  # v9.3: match yaml config default (aic_ros2_controllers.yaml:63). Pose=current_tcp so error≈0 regardless of stiffness — compliance not needed. High stiffness carries over across ctrl reactivate (impedance_params_ only reset on_configure), making trial 2+ start identical to trial 1.
RELEASE_DAMPING   = [35.0, 35.0, 35.0, 35.0, 35.0, 35.0]  # v9.3: match yaml config default (aic_ros2_controllers.yaml:64).
RELEASE_HOLD_TIME = 0.2  # s — ensure the controller ingests it before deactivate

LOG_ENABLED = os.environ.get("CHEATCODE_MJ_LOG", "1") != "0"


def _scalar_trajectory(start: float, end: float, duration: float):
    """Min-jerk trajectory on a single scalar, endpoints pinned at rest."""
    times = np.array([0.0, duration], dtype=np.float64)
    positions = np.array([[start, end]], dtype=np.float64)
    velocities = np.array([[0.0, 0.0]], dtype=np.float64)
    return hebi.trajectory.create_trajectory(times, positions, velocity=velocities)


class CheatCodeMJ(CheatCode):
    """CheatCode with trajectory-driven timing on interp_fraction and z_offset."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        # ROS parameters — set at launch time via
        #   ros2 run aic_model aic_model --ros-args \
        #       -p insertion_threshold_m:=0.003 -p max_insertion_retries:=3
        # Defaults are the values committed in the source above.
        if not parent_node.has_parameter("insertion_threshold_m"):
            parent_node.declare_parameter(
                "insertion_threshold_m", DEFAULT_INSERTION_THRESHOLD_M
            )
        if not parent_node.has_parameter("max_insertion_retries"):
            parent_node.declare_parameter(
                "max_insertion_retries", DEFAULT_MAX_INSERTION_RETRIES
            )
        self._insertion_threshold = float(
            parent_node.get_parameter("insertion_threshold_m").value
        )
        self._max_retries = int(
            parent_node.get_parameter("max_insertion_retries").value
        )
        # Deliberate-failure injection for retry testing. The XY offset is
        # added to port_tf before passing it to calc_gripper_pose, so the
        # descent commands aim at a fake target. The insertion check uses
        # the REAL port pose, so a non-zero offset reliably fails — useful
        # for verifying the retry loop fires + the lift trajectory works.
        if not parent_node.has_parameter("bad_port_offset_x"):
            parent_node.declare_parameter("bad_port_offset_x", DEFAULT_BAD_PORT_OFFSET_X)
        if not parent_node.has_parameter("bad_port_offset_y"):
            parent_node.declare_parameter("bad_port_offset_y", DEFAULT_BAD_PORT_OFFSET_Y)
        if not parent_node.has_parameter("bad_offset_decay_per_retry"):
            parent_node.declare_parameter(
                "bad_offset_decay_per_retry", DEFAULT_BAD_OFFSET_DECAY_PER_RETRY
            )
        self._bad_port_offset_x = float(
            parent_node.get_parameter("bad_port_offset_x").value
        )
        self._bad_port_offset_y = float(
            parent_node.get_parameter("bad_port_offset_y").value
        )
        # Multiplier applied to the bad offset on each retry. 1.0 = no decay
        # (default), 0.5 = halve each retry, 0.0 = retries see real port.
        # Useful for testing whether the retry path can recover from a
        # progressively-easier failure.
        self._bad_offset_decay_per_retry = float(
            parent_node.get_parameter("bad_offset_decay_per_retry").value
        )
        # Stuck-detection params (early-abort during descent).
        if not parent_node.has_parameter("stuck_min_fraction"):
            parent_node.declare_parameter("stuck_min_fraction", DEFAULT_STUCK_MIN_FRACTION)
        if not parent_node.has_parameter("stuck_window_s"):
            parent_node.declare_parameter("stuck_window_s", DEFAULT_STUCK_WINDOW_S)
        if not parent_node.has_parameter("stuck_progress_m"):
            parent_node.declare_parameter("stuck_progress_m", DEFAULT_STUCK_PROGRESS_M)
        self._stuck_min_fraction = float(
            parent_node.get_parameter("stuck_min_fraction").value
        )
        self._stuck_window_s = float(
            parent_node.get_parameter("stuck_window_s").value
        )
        self._stuck_progress_m = float(
            parent_node.get_parameter("stuck_progress_m").value
        )
        # Time-between-attempts knobs (the wait-before-retry params).
        if not parent_node.has_parameter("lift_time_frac"):
            parent_node.declare_parameter("lift_time_frac", DEFAULT_LIFT_TIME_FRAC)
        if not parent_node.has_parameter("hover_hold_s"):
            parent_node.declare_parameter("hover_hold_s", DEFAULT_HOVER_HOLD_BETWEEN_ATTEMPTS_S)
        self._lift_time_frac = float(
            parent_node.get_parameter("lift_time_frac").value
        )
        self._hover_hold_s = float(
            parent_node.get_parameter("hover_hold_s").value
        )
        self.get_logger().info(
            f"CheatCodeMJ params: insertion_threshold_m={self._insertion_threshold}, "
            f"max_insertion_retries={self._max_retries}"
        )
        if self._bad_port_offset_x != 0.0 or self._bad_port_offset_y != 0.0:
            self.get_logger().warn(
                f"⚠ BAD port offset injected: dx={self._bad_port_offset_x * 1000:.1f}mm "
                f"dy={self._bad_port_offset_y * 1000:.1f}mm — descent will aim off-target"
            )

    def _apply_bad_offset(self, port_tf, attempt: int = 0):
        """Mutate port_tf to add the configured XY offset (for retry testing).

        Default 0,0 is a no-op. The offset is applied to every refreshed
        port_tf, scaled by ``bad_offset_decay_per_retry ** attempt`` so
        retries can see a progressively-smaller offset. The insertion
        check always uses the REAL port pose (captured once at initial
        lookup), so non-zero offsets reliably fail; decay lets later
        attempts succeed if the decay factor is < 1.
        """
        if self._bad_port_offset_x == 0.0 and self._bad_port_offset_y == 0.0:
            return
        scale = self._bad_offset_decay_per_retry ** attempt
        offset_x = self._bad_port_offset_x * scale
        offset_y = self._bad_port_offset_y * scale
        port_tf.translation.x += offset_x
        port_tf.translation.y += offset_y
        if attempt > 0 and self._bad_offset_decay_per_retry != 1.0:
            self.get_logger().info(
                f"  bad-offset decayed for attempt {attempt + 1}: "
                f"({offset_x * 1000:+.2f}mm, {offset_y * 1000:+.2f}mm) "
                f"= scale {scale:.3f}"
            )

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

    def _publish_release_hold(self, move_robot: MoveRobotCallback) -> None:
        try:
            tcp = self._lookup("base_link", "gripper/tcp")
        except TransformException as ex:
            self.get_logger().warn(f"Release TF lookup failed, skipping: {ex}")
            return
        hold_pose = Pose(
            position=Point(
                x=tcp.translation.x, y=tcp.translation.y, z=tcp.translation.z
            ),
            orientation=Quaternion(
                x=tcp.rotation.x, y=tcp.rotation.y,
                z=tcp.rotation.z, w=tcp.rotation.w,
            ),
        )
        self.set_pose_target(
            move_robot=move_robot,
            pose=hold_pose,
            stiffness=RELEASE_STIFFNESS,
            damping=RELEASE_DAMPING,
        )
        self.sleep_for(RELEASE_HOLD_TIME)

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

        # Capture the REAL port pose ONCE before any offset injection — used
        # by the insertion check so a deliberately-bad offset still fails
        # honestly even though descent aims at the fake target.
        real_port_xyz = (
            port_tf.translation.x, port_tf.translation.y, port_tf.translation.z,
        )

        # Apply bad-offset injection (no-op when both offsets are 0).
        self._apply_bad_offset(port_tf)
        port_xyz = (
            port_tf.translation.x, port_tf.translation.y, port_tf.translation.z,
        )

        # Pick hover height + timings by plug type. Default to SFP's hover
        # height (tallest) for unrecognised types so we err on the side of
        # more clearance, but default *timings* to SC's (slower, safer) for
        # the same reason.
        plug_type_lower = (task.plug_type or "").lower()
        if plug_type_lower == "sc":
            approach_z_offset = APPROACH_Z_OFFSET_SC
            approach_time = APPROACH_TIME_SC
            descent_time = DESCENT_TIME_SC
            settle_time = SETTLE_TIME_SC
        elif plug_type_lower == "sfp":
            approach_z_offset = APPROACH_Z_OFFSET_SFP
            approach_time = APPROACH_TIME_SFP
            descent_time = DESCENT_TIME_SFP
            settle_time = SETTLE_TIME_SFP
        else:
            approach_z_offset = APPROACH_Z_OFFSET_SFP
            approach_time = APPROACH_TIME_SC
            descent_time = DESCENT_TIME_SC
            settle_time = SETTLE_TIME_SC

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
            approach_time_s=approach_time,
            approach_z_offset_m=approach_z_offset,  # chosen by plug_type
            plug_type=plug_type_lower,
            descent_time_s=descent_time,
            insertion_depth_m=INSERTION_DEPTH,
            settle_time_s=settle_time,
            control_rate_hz=CONTROL_RATE_HZ,
        )

        dt = 1.0 / CONTROL_RATE_HZ

        # ============================================================
        # Phase A — approach: interp_fraction 0 → 1 on min-jerk.
        # calc_gripper_pose(reset_xy_integrator=True) during this phase so
        # the integrator stays at zero until we're hovering above the port.
        # ============================================================
        approach_traj = _scalar_trajectory(0.0, 1.0, approach_time)
        t0 = self.time_now()
        while True:
            elapsed = (self.time_now() - t0).nanoseconds / 1e9
            if elapsed >= approach_traj.duration + 0.5:
                break
            # hebi.get_state clamps past-duration automatically → extra 0.5 s
            # holds interp=1.0 at hover, giving the arm a settle window.
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

        # Re-lookup port_tf after hover settles. Approach-time snapshot may be
        # stale vs current TF state; descent uses live port pose.
        try:
            port_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time()
            ).transform
            self._apply_bad_offset(port_tf)
            port_xyz = (
                port_tf.translation.x, port_tf.translation.y, port_tf.translation.z,
            )
            self._write_summary(
                summary,
                "Port TF refresh (post-approach)",
                port_xyz=port_xyz,
                port_quat_wxyz=(port_tf.rotation.w, port_tf.rotation.x, port_tf.rotation.y, port_tf.rotation.z),
            )
        except TransformException as ex:
            self.get_logger().warn(f"Post-approach port TF refresh failed, keeping snapshot: {ex}")

        # ============================================================
        # Phase B — descent + (settle, check, optionally retry)
        # z_offset goes APPROACH_Z_OFFSET → -INSERTION_DEPTH on min-jerk.
        # interp_fraction pinned at 1 (orientation already aligned). XY
        # integrator active: calc_gripper_pose applies live port_xy vs
        # plug_xy correction exactly like CheatCode.
        #
        # After descent + settle we measure plug-port distance via ground-
        # truth TF. If below self._insertion_threshold the plug is seated and
        # we exit. Otherwise we lift back to hover, re-snapshot port_tf,
        # reset the XY integrator, and re-descend — up to
        # self._max_retries additional attempts.
        # ============================================================
        inserted = False
        final_dist: float | None = None
        last_z_offset = approach_z_offset
        attempts_used = 0

        for attempt in range(self._max_retries + 1):
            attempts_used = attempt + 1

            # Lift back to hover (skipped on first attempt — we're already
            # there from approach phase).
            if attempt > 0:
                self.get_logger().info(
                    f"Lifting back to hover for retry {attempt}/{self._max_retries}"
                )
                send_feedback(f"insertion retry {attempt}/{self._max_retries}: lifting")
                # Lift from wherever the plug actually is (last_z_offset),
                # NOT from the descent target. If stuck-detection aborted
                # early, last_z_offset can still be high up — lifting from
                # -INSERTION_DEPTH would command the plug to first dash
                # DOWN to the descent target before ramping back up.
                # Duration scales proportionally with the distance to cover
                # so a small lift doesn't take the same time as a full one.
                lift_distance = approach_z_offset - last_z_offset
                full_lift_distance = approach_z_offset - (-INSERTION_DEPTH)
                lift_duration = max(
                    0.5,  # floor so very small lifts still have a sane min-jerk
                    descent_time * self._lift_time_frac
                        * (lift_distance / full_lift_distance),
                )
                self.get_logger().info(
                    f"Lift: from z={last_z_offset:.3f} to z={approach_z_offset:.3f} "
                    f"({lift_distance * 1000:.0f}mm) over {lift_duration:.2f}s"
                )
                lift_traj = _scalar_trajectory(
                    last_z_offset,
                    approach_z_offset,
                    lift_duration,
                )
                t0 = self.time_now()
                while True:
                    elapsed = (self.time_now() - t0).nanoseconds / 1e9
                    if elapsed >= lift_traj.duration:
                        break
                    z, _, _ = lift_traj.get_state(elapsed)
                    z_offset = float(z[0])
                    try:
                        pose = self.calc_gripper_pose(
                            port_tf,
                            slerp_fraction=1.0,
                            position_fraction=1.0,
                            z_offset=z_offset,
                            reset_xy_integrator=True,  # zero out integrator during lift
                        )
                        self.set_pose_target(move_robot=move_robot, pose=pose)
                    except TransformException as ex:
                        self.get_logger().warn(f"Lift TF lookup failed: {ex}")
                    self._log_row(writer, elapsed, f"lift_{attempt}", 1.0, z_offset, port_xyz)
                    self.sleep_for(dt)

                # Brief steady-state at hover so any wobble from the lift dies.
                self.sleep_for(self._hover_hold_s)

                # Re-snapshot port_tf — the live TF may have drifted between attempts.
                try:
                    port_tf = self._parent_node._tf_buffer.lookup_transform(
                        "base_link", port_frame, Time()
                    ).transform
                    # Pass attempt so bad-offset decays across retries.
                    self._apply_bad_offset(port_tf, attempt=attempt)
                    port_xyz = (
                        port_tf.translation.x, port_tf.translation.y, port_tf.translation.z,
                    )
                    self._write_summary(
                        summary, f"Port TF refresh (retry {attempt})",
                        port_xyz=port_xyz,
                    )
                except TransformException as ex:
                    self.get_logger().warn(f"Retry port TF refresh failed, keeping snapshot: {ex}")

                # Reset XY integrator — explicit even though the lift's
                # reset_xy_integrator=True already did this. Belt-and-suspenders.
                self._tip_x_error_integrator = 0.0
                self._tip_y_error_integrator = 0.0

            # Descent
            descent_traj = _scalar_trajectory(
                approach_z_offset, -INSERTION_DEPTH, descent_time
            )
            t0 = self.time_now()
            stuck_distances: deque = deque()  # (elapsed, dist_to_real_port)
            stuck_detected = False
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
                phase_label = "descent" if attempt == 0 else f"descent_retry_{attempt}"
                self._log_row(writer, elapsed, phase_label, 1.0, z_offset, port_xyz)
                self.sleep_for(dt)

                # ── Early-abort: distance to REAL port not decreasing ──
                # Sample plug-port distance and accumulate a window. Once
                # we're past the min-fraction guard and the window is full,
                # check net progress; if below threshold, the plug isn't
                # advancing toward the real port — bail out and let the
                # retry path lift + try again.
                fraction = elapsed / descent_traj.duration
                if fraction > self._stuck_min_fraction:
                    try:
                        plug_now = self._lookup("base_link", plug_frame)
                        d = float(
                            np.linalg.norm(
                                np.array([
                                    plug_now.translation.x - real_port_xyz[0],
                                    plug_now.translation.y - real_port_xyz[1],
                                    plug_now.translation.z - real_port_xyz[2],
                                ])
                            )
                        )
                        stuck_distances.append((elapsed, d))
                        cutoff = elapsed - self._stuck_window_s
                        while stuck_distances and stuck_distances[0][0] < cutoff:
                            stuck_distances.popleft()
                        window_span = stuck_distances[-1][0] - stuck_distances[0][0]
                        if window_span >= self._stuck_window_s * 0.9:
                            net_progress = (
                                stuck_distances[0][1] - stuck_distances[-1][1]
                            )
                            current_dist = stuck_distances[-1][1]
                            already_seated = current_dist <= self._insertion_threshold
                            if (not already_seated
                                    and net_progress < self._stuck_progress_m):
                                self.get_logger().warn(
                                    f"⚠ Stuck detected at t={elapsed:.2f}s "
                                    f"(fraction={fraction:.2f}, dist={current_dist * 1000:.2f}mm): "
                                    f"net progress over last {self._stuck_window_s:.1f}s = "
                                    f"{net_progress * 1000:+.2f}mm "
                                    f"(threshold={self._stuck_progress_m * 1000:.2f}mm) — "
                                    f"aborting descent"
                                )
                                stuck_detected = True
                                break
                    except TransformException:
                        pass

            # Settle — skipped on a stuck-aborted descent (we know it failed,
            # no point waiting another settle_time for the obvious).
            if stuck_detected:
                self.get_logger().info(
                    f"Descent attempt {attempt + 1} aborted (stuck); skipping settle."
                )
            else:
                self.get_logger().info(
                    f"Descent attempt {attempt + 1} complete, settling..."
                )
                self.sleep_for(settle_time)

            # Insertion check via plug-port distance — use the REAL port pose
            # captured before any bad-offset injection, so a deliberately-bad
            # offset still fails the check honestly.
            try:
                plug_final = self._lookup("base_link", plug_frame)
                final_dist = float(
                    np.linalg.norm(
                        np.array([
                            plug_final.translation.x - real_port_xyz[0],
                            plug_final.translation.y - real_port_xyz[1],
                            plug_final.translation.z - real_port_xyz[2],
                        ])
                    )
                )
            except TransformException as ex:
                self.get_logger().warn(f"Final plug TF lookup failed: {ex}")
                final_dist = None

            if final_dist is not None and final_dist < self._insertion_threshold:
                inserted = True
                self.get_logger().info(
                    f"✓ Insertion confirmed on attempt {attempt + 1}: "
                    f"dist={final_dist * 1000:.2f}mm < threshold={self._insertion_threshold * 1000:.2f}mm"
                )
                self._write_summary(
                    summary, f"Attempt {attempt + 1} result",
                    inserted=True,
                    plug_port_distance_m=final_dist,
                    threshold_m=self._insertion_threshold,
                )
                break

            dist_str = f"{final_dist * 1000:.2f}mm" if final_dist is not None else "unknown"
            self.get_logger().info(
                f"✗ Attempt {attempt + 1} not inserted: "
                f"dist={dist_str} (threshold={self._insertion_threshold * 1000:.2f}mm)"
            )
            self._write_summary(
                summary, f"Attempt {attempt + 1} result",
                inserted=False,
                plug_port_distance_m=final_dist,
                threshold_m=self._insertion_threshold,
            )

        # Final summary + log line.
        self._write_summary(
            summary,
            "Final state",
            inserted=inserted,
            attempts_used=attempts_used,
            max_attempts=self._max_retries + 1,
            plug_port_distance_m=final_dist,
            insertion_threshold_m=self._insertion_threshold,
            final_z_offset=last_z_offset,
            final_integrator_xy=(self._tip_x_error_integrator, self._tip_y_error_integrator),
        )
        if final_dist is not None:
            self.get_logger().info(
                f"CheatCodeMJ done. inserted={inserted}, "
                f"plug-port dist: {final_dist:.4f}m, attempts={attempts_used}"
            )
        else:
            self.get_logger().info(
                f"CheatCodeMJ done. inserted={inserted}, "
                f"plug-port dist: unknown, attempts={attempts_used}"
            )

        self._publish_release_hold(move_robot)

        if summary: summary.close()
        if csvf: csvf.close()

        return inserted
