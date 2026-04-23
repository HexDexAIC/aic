#!/usr/bin/env python3
"""Post-process a sweep directory to extract starting scene poses per trial.

Reads the `bag_trial_N/*.mcap` files that the engine records and pulls out:
  - task_board pose (world frame)
  - target port pose (world frame)
  - cable root pose + plug tip pose (world frame)
  - initial joint positions
  - initial TCP pose (world frame)

Outputs:
  <sweep_dir>/run_NNN/initial_poses.json       — one file per trial, full detail
  <sweep_dir>/initial_poses.csv                — one flat row per trial, aggregated

Per-trial task/plug/port names come from aic_engine/config/sample_config.yaml
(the trial schema is fixed for the qualification phase).

Usage:
    scripts/extract_initial_poses.py <sweep_dir>

Must be run with the scripts/.venv python (has mcap + mcap-ros2-support installed):
    scripts/.venv/bin/python scripts/extract_initial_poses.py <sweep_dir>
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# ---------------------------------------------------------------------------
# Sample config — maps trial_N to the specific cable/plug/port/module names
# that the engine spawns and scores against.
# ---------------------------------------------------------------------------
SAMPLE_CONFIG = Path(
    "/home/hariharan/ws_aic/src/aic/aic_engine/config/sample_config.yaml"
)


def load_trial_task_map(config_path: Path = SAMPLE_CONFIG) -> dict[str, dict]:
    """Extract per-trial {cable, plug, port, module} from the engine config.

    Returns a dict like:
        {"trial_1": {"cable_name": "cable_0", "plug_name": "sfp_tip",
                     "port_name": "sfp_port_0", "module": "nic_card_mount_0"}}
    """
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    out: dict[str, dict] = {}
    for trial_id, trial in cfg.get("trials", {}).items():
        tasks = trial.get("tasks", {})
        if not tasks:
            continue
        # Every shipped config has a single task per trial.
        task = next(iter(tasks.values()))
        out[trial_id] = {
            "cable_name": task.get("cable_name"),
            "plug_name": task.get("plug_name"),
            "port_name": task.get("port_name"),
            "module": task.get("target_module_name"),
        }
    return out


# ---------------------------------------------------------------------------
# Transform algebra: compose TF transforms so we can chain
# aic_world -> task_board -> module -> port_link into aic_world -> port_link.
# ---------------------------------------------------------------------------
@dataclass
class Transform:
    tx: float
    ty: float
    tz: float
    qw: float
    qx: float
    qy: float
    qz: float

    @classmethod
    def from_ros(cls, t) -> "Transform":
        return cls(
            tx=t.translation.x,
            ty=t.translation.y,
            tz=t.translation.z,
            qw=t.rotation.w,
            qx=t.rotation.x,
            qy=t.rotation.y,
            qz=t.rotation.z,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.tx, "y": self.ty, "z": self.tz,
            "qw": self.qw, "qx": self.qx, "qy": self.qy, "qz": self.qz,
        }


def q_mul(a: Transform, b: Transform) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = a.qw, a.qx, a.qy, a.qz
    w2, x2, y2, z2 = b.qw, b.qx, b.qy, b.qz
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def q_rotate_vec(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    qw, qx, qy, qz = q
    vx, vy, vz = v
    # v' = q * (0, v) * q_conj
    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def compose(parent: Transform, child: Transform) -> Transform:
    """child pose (child_frame_id → parent_frame_id) composed with parent:
    result is the child_frame pose in parent's parent frame.
    """
    rx, ry, rz = q_rotate_vec(
        (parent.qw, parent.qx, parent.qy, parent.qz),
        (child.tx, child.ty, child.tz),
    )
    qw, qx, qy, qz = q_mul(parent, child)
    return Transform(
        tx=parent.tx + rx,
        ty=parent.ty + ry,
        tz=parent.tz + rz,
        qw=qw, qx=qx, qy=qy, qz=qz,
    )


def chain_transforms(edges: dict[tuple[str, str], Transform],
                     root: str, leaf: str) -> Transform | None:
    """BFS through edges to chain transforms from root to leaf."""
    # Build adjacency: parent_frame -> list of (child_frame, transform)
    adj: dict[str, list[tuple[str, Transform]]] = {}
    for (parent, child), tr in edges.items():
        adj.setdefault(parent, []).append((child, tr))

    # BFS from root
    queue = [(root, None)]
    parents: dict[str, tuple[str, Transform] | None] = {root: None}
    while queue:
        node, _ = queue.pop(0)
        if node == leaf:
            break
        for child, tr in adj.get(node, []):
            if child not in parents:
                parents[child] = (node, tr)
                queue.append((child, tr))

    if leaf not in parents:
        return None

    # Walk back and compose
    path: list[Transform] = []
    n = leaf
    while parents[n] is not None:
        parent_frame, tr = parents[n]
        path.append(tr)
        n = parent_frame
    path.reverse()
    result = Transform(0, 0, 0, 1, 0, 0, 0)
    for tr in path:
        result = compose(result, tr)
    return result


# ---------------------------------------------------------------------------
# Bag reader — collect first-seen transforms + first joint_states message.
# ---------------------------------------------------------------------------
@dataclass
class BagSnapshot:
    first_tf: dict[tuple[str, str], Transform] = field(default_factory=dict)
    first_tf_static: dict[tuple[str, str], Transform] = field(default_factory=dict)
    first_scoring_tf: dict[tuple[str, str], Transform] = field(default_factory=dict)
    joint_state: dict[str, float] | None = None
    joint_state_velocity: dict[str, float] | None = None

    def all_edges(self) -> dict[tuple[str, str], Transform]:
        # scoring/tf and tf_static are the stable references; live tf fills any
        # gaps (robot chain frames, camera frames).
        merged: dict[tuple[str, str], Transform] = {}
        merged.update(self.first_tf)
        merged.update(self.first_tf_static)  # static wins over live for the robot/camera rigid frames
        merged.update(self.first_scoring_tf)  # scoring wins over tf for task_board/cable (ground-truth source)
        return merged


def read_bag_snapshot(mcap_path: Path, max_seconds: float = 2.0) -> BagSnapshot:
    """Read the bag's first `max_seconds` worth of data.

    Per-trial bags contain ~73k /tf + 9k /scoring/tf messages over ~30 s of
    simulation. We only care about *first-seen* transforms and the first
    joint_states, both of which are published within the first second.
    Limiting the time window drops processing from ~1 min/trial to <1 s/trial.
    """
    snap = BagSnapshot()
    start_time: int | None = None
    end_time_ns: int | None = None
    interesting_topics = {"/tf", "/tf_static", "/scoring/tf", "/joint_states"}
    with mcap_path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=list(interesting_topics),
        ):
            if start_time is None:
                start_time = message.log_time
                end_time_ns = start_time + int(max_seconds * 1e9)
            if message.log_time > end_time_ns:
                break
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


# ---------------------------------------------------------------------------
# Per-trial extraction
# ---------------------------------------------------------------------------
def extract_trial(bag_dir: Path, trial_id: str, task: dict) -> dict:
    """Given a bag_trial_N dir, extract the initial scene poses."""
    mcap_files = list(bag_dir.glob("*.mcap"))
    if not mcap_files:
        return {"error": f"no .mcap file in {bag_dir}"}
    snap = read_bag_snapshot(mcap_files[0])
    edges = snap.all_edges()

    cable = task["cable_name"] or "cable_0"
    plug = task["plug_name"] or ""
    port = task["port_name"] or ""
    module = task["module"] or ""

    out: dict[str, Any] = {
        "trial_id": trial_id,
        "task": task,
    }

    # Task board pose in world
    board = chain_transforms(edges, "aic_world", "task_board")
    if board:
        out["task_board_world"] = board.as_dict()

    # Module pose in world (parent of target port)
    if module:
        module_frame = f"task_board/{module}"
        module_world = chain_transforms(edges, "aic_world", module_frame)
        if module_world:
            out["module_world"] = module_world.as_dict()

    # Target port pose in world.
    # Port frame is: task_board/<module>/<port>_link
    if module and port:
        port_frame = f"task_board/{module}/{port}_link"
        port_world = chain_transforms(edges, "aic_world", port_frame)
        if port_world:
            out["port_world"] = port_world.as_dict()
        else:
            # Trial 3 has a flatter structure: task_board/sc_port_N/<port>_link
            # Try an alternate pattern:
            for parent, child in edges:
                if child == f"task_board/{module}/{port}_link" or \
                   (child.endswith(f"/{port}_link") and parent.startswith(f"task_board/{module}")):
                    pw = chain_transforms(edges, "aic_world", child)
                    if pw:
                        out["port_world"] = pw.as_dict()
                        break

    # Cable root in world
    cable_world = chain_transforms(edges, "aic_world", cable)
    if cable_world:
        out["cable_world"] = cable_world.as_dict()

    # Plug tip pose in world (starting grasp pose of the tip that must insert)
    if plug:
        plug_frame = f"{cable}/{plug}_link"
        plug_world = chain_transforms(edges, "aic_world", plug_frame)
        if plug_world:
            out["plug_world"] = plug_world.as_dict()

    # Robot base in world (static, should be ~constant)
    base_world = chain_transforms(edges, "aic_world", "base_link")
    if base_world:
        out["base_link_world"] = base_world.as_dict()

    # TCP in world
    tcp_world = chain_transforms(edges, "aic_world", "gripper/tcp")
    if tcp_world:
        out["tcp_world"] = tcp_world.as_dict()

    # Plug-to-port vector (most useful starting condition for insertion)
    if out.get("plug_world") and out.get("port_world"):
        p = out["plug_world"]
        q = out["port_world"]
        out["plug_to_port_world"] = {
            "dx": q["x"] - p["x"],
            "dy": q["y"] - p["y"],
            "dz": q["z"] - p["z"],
            "distance": ((q["x"] - p["x"]) ** 2 + (q["y"] - p["y"]) ** 2 + (q["z"] - p["z"]) ** 2) ** 0.5,
        }

    # Initial joint state
    if snap.joint_state:
        out["joint_positions"] = snap.joint_state

    return out


# ---------------------------------------------------------------------------
# Sweep-level aggregation
# ---------------------------------------------------------------------------
BAG_DIR_RE = re.compile(r"^bag_trial_(\d+)_")


def process_run(run_dir: Path, trial_task_map: dict) -> list[dict]:
    """Return a list of per-trial dicts for this run."""
    trials: list[dict] = []
    for bag_dir in sorted(run_dir.glob("bag_trial_*")):
        if not bag_dir.is_dir():
            continue
        m = BAG_DIR_RE.match(bag_dir.name)
        if not m:
            continue
        trial_num = m.group(1)
        trial_id = f"trial_{trial_num}"
        task = trial_task_map.get(trial_id, {})
        result = extract_trial(bag_dir, trial_id, task)
        result["run"] = run_dir.name
        trials.append(result)

    # Write per-run JSON
    out_path = run_dir / "initial_poses.json"
    with out_path.open("w") as f:
        json.dump(trials, f, indent=2)
    return trials


def flatten_for_csv(trial: dict) -> dict:
    """Flatten nested trial dict into CSV-compatible columns."""
    row = {
        "run": trial.get("run"),
        "trial": trial.get("trial_id"),
        "cable": trial.get("task", {}).get("cable_name"),
        "plug": trial.get("task", {}).get("plug_name"),
        "port": trial.get("task", {}).get("port_name"),
        "module": trial.get("task", {}).get("module"),
    }
    for key in ("task_board_world", "port_world", "plug_world",
                "cable_world", "tcp_world", "base_link_world"):
        pose = trial.get(key)
        if pose:
            for axis in ("x", "y", "z", "qw", "qx", "qy", "qz"):
                row[f"{key}.{axis}"] = pose[axis]

    ptp = trial.get("plug_to_port_world")
    if ptp:
        for k, v in ptp.items():
            row[f"plug_to_port.{k}"] = v

    jp = trial.get("joint_positions", {})
    for jname, jval in jp.items():
        row[f"q.{jname}"] = jval
    return row


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    sweep_dir = Path(sys.argv[1]).resolve()
    if not sweep_dir.is_dir():
        print(f"Not a directory: {sweep_dir}", file=sys.stderr)
        return 2

    trial_task_map = load_trial_task_map()
    run_dirs = sorted(p for p in sweep_dir.iterdir()
                      if p.is_dir() and p.name.startswith("run_"))
    if not run_dirs:
        print(f"No run_* subdirectories in {sweep_dir}", file=sys.stderr)
        return 1

    all_trials: list[dict] = []
    for rd in run_dirs:
        try:
            trials = process_run(rd, trial_task_map)
            all_trials.extend(trials)
            print(f"  {rd.name}: extracted {len(trials)} trial(s)")
        except Exception as ex:
            print(f"  {rd.name}: FAILED — {ex}", file=sys.stderr)

    # Aggregate CSV
    csv_path = sweep_dir / "initial_poses.csv"
    rows = [flatten_for_csv(t) for t in all_trials]
    if rows:
        all_keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nAggregated CSV: {csv_path}  ({len(rows)} rows, {len(all_keys)} columns)")
    else:
        print("\nNo trials extracted — no CSV written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
