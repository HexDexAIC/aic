#!/usr/bin/env python3
"""Compare per-trial initial scene poses across two continuous sweep bags.

Adapts extract_initial_poses.py to continuous bags (whole-sweep mcap rather than
the per-trial bag_trial_N/*.mcap files the engine writes when bagging is on).

Trial boundaries are derived from /insert_cable/_action/status — each trial
starts at its own STATUS_EXECUTING transition. For each trial we collect
first-seen /tf_static, /tf, and /scoring/tf transforms from the start of that
trial and walk the resulting forest to extract:
  - task_board pose (world)
  - target port pose (world) — uses sample_config.yaml to map trial_N → port name
  - plug tip pose (world)
  - cable root pose (world)
  - TCP pose (world)
  - initial joint positions

Then it diffs trial-by-trial across the two bags and reports any pose that
disagrees by more than EPS (default 1 mm linear / 1e-3 rad orientation).

Use this to answer: "does the engine spawn the same scene given the same
deterministic YAML, or does Gazebo introduce run-to-run jitter?"

Usage:
    src/aic/scripts/.venv/bin/python \
        src/aic/scripts/compare_initial_poses_continuous.py \
        <bag_dir_a> <bag_dir_b>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# Reuse algebra + extraction helpers from the existing per-trial extractor.
sys.path.insert(0, str(Path(__file__).parent))
from extract_initial_poses import (  # noqa: E402
    BagSnapshot,
    Transform,
    chain_transforms,
    load_trial_task_map,
)

# Action status enum (action_msgs/GoalStatus).
STATUS_EXECUTING = 2

# Pose comparison tolerances.
EPS_LINEAR = 1e-3      # 1 mm
EPS_ANGULAR = 1e-3     # 1 mrad on quaternion components


# ---------------------------------------------------------------------------
# Trial-boundary detection
# ---------------------------------------------------------------------------
def find_trial_starts(mcap_path: Path) -> list[int]:
    """Return list of log_time_ns at which each trial enters STATUS_EXECUTING."""
    starts: list[int] = []
    last_was_executing = False
    with mcap_path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=["/insert_cable/_action/status"],
        ):
            for status in ros_msg.status_list:
                is_exec = status.status == STATUS_EXECUTING
                if is_exec and not last_was_executing:
                    starts.append(message.log_time)
                last_was_executing = is_exec
    return starts


# ---------------------------------------------------------------------------
# Per-trial snapshot from a continuous bag
# ---------------------------------------------------------------------------
def snapshot_at_time(mcap_path: Path, t_start_ns: int, window_s: float = 3.0) -> BagSnapshot:
    """Collect first-seen transforms + first joint_states in [t_start, t_start+window].

    A 3-second window comfortably covers the engine's spawn-and-publish-statics
    burst that follows every STATUS_EXECUTING transition.
    """
    snap = BagSnapshot()
    end_ns = t_start_ns + int(window_s * 1e9)
    interesting = ["/tf", "/tf_static", "/scoring/tf", "/joint_states"]
    with mcap_path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=interesting,
            start_time=t_start_ns,
            end_time=end_ns,
        ):
            topic = channel.topic
            if topic in ("/tf", "/tf_static", "/scoring/tf"):
                target = {
                    "/tf": snap.first_tf,
                    "/tf_static": snap.first_tf_static,
                    "/scoring/tf": snap.first_scoring_tf,
                }[topic]
                for tr in ros_msg.transforms:
                    key = (tr.header.frame_id, tr.child_frame_id)
                    if key not in target:
                        target[key] = Transform.from_ros(tr.transform)
            elif topic == "/joint_states" and snap.joint_state is None:
                snap.joint_state = dict(zip(ros_msg.name, ros_msg.position))
                snap.joint_state_velocity = dict(zip(ros_msg.name, ros_msg.velocity))
    return snap


def extract_trial_poses(mcap_path: Path, t_start_ns: int, task: dict) -> dict:
    snap = snapshot_at_time(mcap_path, t_start_ns)
    edges = snap.all_edges()
    out: dict[str, Any] = {"task": task}

    cable = task.get("cable_name") or "cable_0"
    plug = task.get("plug_name") or ""
    port = task.get("port_name") or ""
    module = task.get("module") or ""

    board = chain_transforms(edges, "aic_world", "task_board")
    if board:
        out["task_board_world"] = board.as_dict()

    if module:
        m_world = chain_transforms(edges, "aic_world", f"task_board/{module}")
        if m_world:
            out["module_world"] = m_world.as_dict()

    if module and port:
        port_frame = f"task_board/{module}/{port}_link"
        p_world = chain_transforms(edges, "aic_world", port_frame)
        if p_world is None:
            for parent, child in edges:
                if child.endswith(f"/{port}_link") and module in parent:
                    p_world = chain_transforms(edges, "aic_world", child)
                    if p_world:
                        break
        if p_world:
            out["port_world"] = p_world.as_dict()

    cable_world = chain_transforms(edges, "aic_world", cable)
    if cable_world:
        out["cable_world"] = cable_world.as_dict()
    if plug:
        plug_world = chain_transforms(edges, "aic_world", f"{cable}/{plug}_link")
        if plug_world:
            out["plug_world"] = plug_world.as_dict()

    base_world = chain_transforms(edges, "aic_world", "base_link")
    if base_world:
        out["base_link_world"] = base_world.as_dict()
    tcp_world = chain_transforms(edges, "aic_world", "gripper/tcp")
    if tcp_world:
        out["tcp_world"] = tcp_world.as_dict()

    if out.get("plug_world") and out.get("port_world"):
        p = out["plug_world"]
        q = out["port_world"]
        out["plug_to_port_world"] = {
            "dx": q["x"] - p["x"],
            "dy": q["y"] - p["y"],
            "dz": q["z"] - p["z"],
            "distance": (
                (q["x"] - p["x"]) ** 2
                + (q["y"] - p["y"]) ** 2
                + (q["z"] - p["z"]) ** 2
            ) ** 0.5,
        }

    if snap.joint_state:
        out["joint_positions"] = snap.joint_state
    return out


# ---------------------------------------------------------------------------
# Diff pretty-printer
# ---------------------------------------------------------------------------
POSE_KEYS = (
    "task_board_world",
    "module_world",
    "port_world",
    "cable_world",
    "plug_world",
    "tcp_world",
    "base_link_world",
)


def diff_pose(a: dict, b: dict) -> dict[str, float]:
    """Return component-wise abs differences for a 7-D pose dict."""
    return {k: abs(a[k] - b[k]) for k in ("x", "y", "z", "qw", "qx", "qy", "qz")}


def fmt_pose(p: dict | None) -> str:
    if p is None:
        return "                     —                     "
    return (
        f"x={p['x']:+.5f} y={p['y']:+.5f} z={p['z']:+.5f}  "
        f"qw={p['qw']:+.4f} qx={p['qx']:+.4f} qy={p['qy']:+.4f} qz={p['qz']:+.4f}"
    )


def report(bag_a: Path, bag_b: Path, trials_a: list[dict], trials_b: list[dict]) -> None:
    print()
    print(f"BAG A: {bag_a.name}")
    print(f"BAG B: {bag_b.name}")
    print()

    n = min(len(trials_a), len(trials_b))
    for i in range(n):
        a = trials_a[i]
        b = trials_b[i]
        task = a.get("task") or b.get("task") or {}
        print(
            f"════ Trial {i + 1}  "
            f"(plug={task.get('plug_name')}, port={task.get('port_name')}, "
            f"module={task.get('module')}) ════"
        )
        for key in POSE_KEYS:
            pa = a.get(key)
            pb = b.get(key)
            if pa is None and pb is None:
                continue
            print(f"  {key}")
            print(f"    A: {fmt_pose(pa)}")
            print(f"    B: {fmt_pose(pb)}")
            if pa and pb:
                d = diff_pose(pa, pb)
                lin = max(d["x"], d["y"], d["z"])
                ang = max(d["qw"], d["qx"], d["qy"], d["qz"])
                flag = ""
                if lin > EPS_LINEAR:
                    flag += f" [Δlin={lin*1000:.2f}mm]"
                if ang > EPS_ANGULAR:
                    flag += f" [Δang={ang:.4f}]"
                if not flag:
                    flag = " (≤1mm / ≤1mrad — match)"
                print(f"    Δ:  Δx={d['x']*1000:+.2f}mm Δy={d['y']*1000:+.2f}mm "
                      f"Δz={d['z']*1000:+.2f}mm  Δqw={d['qw']:+.4f}{flag}")

        # Joint diff
        ja = a.get("joint_positions") or {}
        jb = b.get("joint_positions") or {}
        common = sorted(set(ja) & set(jb))
        if common:
            qdiffs = {j: ja[j] - jb[j] for j in common}
            mx = max(abs(v) for v in qdiffs.values())
            print(f"  joint_positions  max|Δ| = {mx:.5f} rad ({mx * 180 / 3.14159:.3f} deg)")
            if mx > 1e-3:
                for j, dv in qdiffs.items():
                    if abs(dv) > 1e-4:
                        print(f"    Δ{j} = {dv:+.5f}")

        # Plug→port distance comparison
        ptp_a = a.get("plug_to_port_world")
        ptp_b = b.get("plug_to_port_world")
        if ptp_a and ptp_b:
            print(
                f"  plug→port  A: dx={ptp_a['dx']*1000:+.2f}mm dy={ptp_a['dy']*1000:+.2f}mm "
                f"dz={ptp_a['dz']*1000:+.2f}mm  |  "
                f"B: dx={ptp_b['dx']*1000:+.2f}mm dy={ptp_b['dy']*1000:+.2f}mm "
                f"dz={ptp_b['dz']*1000:+.2f}mm"
            )
            print(
                f"             dist A={ptp_a['distance']*1000:.2f}mm  "
                f"B={ptp_b['distance']*1000:.2f}mm  "
                f"Δdist={(ptp_a['distance'] - ptp_b['distance'])*1000:+.2f}mm"
            )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process_bag(bag_dir: Path, trial_task_map: dict) -> list[dict]:
    mcap = next(bag_dir.glob("*.mcap"))
    starts = find_trial_starts(mcap)
    if not starts:
        raise RuntimeError(f"No trial starts found in {bag_dir.name}")
    trials: list[dict] = []
    for i, t in enumerate(starts):
        trial_id = f"trial_{i + 1}"
        task = trial_task_map.get(trial_id, {})
        trial = extract_trial_poses(mcap, t, task)
        trial["trial_id"] = trial_id
        trial["t_start_ns"] = t
        trials.append(trial)
    return trials


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    bag_a = Path(sys.argv[1]).resolve()
    bag_b = Path(sys.argv[2]).resolve()
    if not bag_a.is_dir() or not bag_b.is_dir():
        print("Both arguments must be bag directories", file=sys.stderr)
        return 2

    trial_task_map = load_trial_task_map()
    trials_a = process_bag(bag_a, trial_task_map)
    trials_b = process_bag(bag_b, trial_task_map)

    report(bag_a, bag_b, trials_a, trials_b)

    # Also dump JSON next to each bag for later inspection.
    for bag, trials in ((bag_a, trials_a), (bag_b, trials_b)):
        out = bag / "initial_poses.json"
        with out.open("w") as f:
            json.dump(trials, f, indent=2)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
