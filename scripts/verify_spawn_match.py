#!/usr/bin/env python3
"""Post-process a spawn-sweep dir: compare each spawned scene to its YAML.

For each seed in <sweep_dir>/seeds/seed_NN/, opens the bag at
seeds/seed_NN/bag_trial_1*/, reads /scoring/tf and /tf_static for the
aic_world → task_board transform, and compares against the YAML at
<sweep_dir>/configs/seed_NN.yaml.

Writes <sweep_dir>/spawn_verification.json with per-seed deltas.

Run with the .venv python that has mcap + mcap-ros2-support installed:
  src/aic/scripts/.venv/bin/python src/aic/scripts/verify_spawn_match.py <sweep_dir>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """ZYX intrinsic yaw (rotation around world Z), assuming small roll/pitch."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_to_rpy(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """ZYX intrinsic Euler angles (roll, pitch, yaw)."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def read_first_transforms(bag_dir: Path, topics: list[str]) -> dict:
    """Return {child_frame_id: transform_dict} taking the FIRST sighting per
    child across the given topics. /scoring/tf is the ground-truth source for
    spawned task_board / cable / module poses; /tf_static is the static frames
    (robot, sensors)."""
    mcaps = list(bag_dir.glob("*.mcap"))
    if not mcaps:
        return {}
    out: dict[str, dict] = {}
    with open(mcaps[0], "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for schema, channel, msg, ros_msg in reader.iter_decoded_messages(
            topics=topics
        ):
            for t in ros_msg.transforms:
                child = t.child_frame_id
                if child in out:
                    continue
                tx = t.transform.translation
                rt = t.transform.rotation
                out[child] = {
                    "parent": t.header.frame_id,
                    "x": tx.x, "y": tx.y, "z": tx.z,
                    "qx": rt.x, "qy": rt.y, "qz": rt.z, "qw": rt.w,
                }
    return out


def normalize_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def angle_delta(actual: float, expected: float) -> float:
    """Smallest signed difference, accounting for wrap."""
    return normalize_angle(actual - expected)


def verify_seed(seed_dir: Path, config_path: Path) -> dict:
    """Compare spawned scene to YAML for one seed.

    seed_dir is the per-seed run dir (single-tree layout) containing the
    bag_trial_1*/, terminal logs, scoring.yaml, and dataset/.
    """
    if not seed_dir.exists():
        return {"checked": False, "reason": f"seed_dir gone: {seed_dir}"}

    bag_candidates = list(seed_dir.glob("bag_trial_1*"))
    if not bag_candidates:
        return {"checked": False, "reason": "no bag_trial_1*"}
    bag_dir = bag_candidates[0]

    tf = read_first_transforms(bag_dir, topics=["/scoring/tf", "/tf_static"])
    if not tf:
        return {"checked": False, "reason": "no transforms"}

    cfg = yaml.safe_load(config_path.read_text())
    expected_tb = cfg["trials"]["trial_1"]["scene"]["task_board"]["pose"]

    # Find the task_board transform. Some bags use child_frame "task_board",
    # others a model-namespaced frame. Match either.
    tb_tf = None
    for child, t in tf.items():
        if child == "task_board" or child.endswith("/task_board"):
            tb_tf = t
            break
    if tb_tf is None:
        return {
            "checked": False,
            "reason": "no task_board in /tf_static",
            "frames_seen": sorted(tf.keys())[:20],
        }

    actual_rpy = quat_to_rpy(tb_tf["qx"], tb_tf["qy"], tb_tf["qz"], tb_tf["qw"])

    deltas = {
        "x": {
            "expected": float(expected_tb["x"]),
            "actual": tb_tf["x"],
            "delta": tb_tf["x"] - float(expected_tb["x"]),
        },
        "y": {
            "expected": float(expected_tb["y"]),
            "actual": tb_tf["y"],
            "delta": tb_tf["y"] - float(expected_tb["y"]),
        },
        "z": {
            "expected": float(expected_tb["z"]),
            "actual": tb_tf["z"],
            "delta": tb_tf["z"] - float(expected_tb["z"]),
        },
        "yaw": {
            "expected": float(expected_tb["yaw"]),
            "actual": actual_rpy[2],
            "delta": angle_delta(actual_rpy[2], float(expected_tb["yaw"])),
        },
    }

    # ≤1 mm and ≤1 mrad tolerance — engine should be exact.
    matched = (
        abs(deltas["x"]["delta"]) < 1e-3
        and abs(deltas["y"]["delta"]) < 1e-3
        and abs(deltas["z"]["delta"]) < 1e-3
        and abs(deltas["yaw"]["delta"]) < 1e-3
    )

    # Confirm the chosen nic_card_mount_<i> actually spawned. That proves
    # the discrete rail-choice took effect. Any nic_card_mount_<j> with
    # j != chosen would mean we spawned the wrong rail.
    import re as _re
    expected_module = cfg["trials"]["trial_1"]["tasks"]["task_1"]["target_module_name"]
    chosen_idx = int(expected_module.rsplit("_", 1)[1])
    mount_indices_seen: set[int] = set()
    for child in tf:
        m = _re.search(r"nic_card_mount_(\d+)", child)
        if m:
            mount_indices_seen.add(int(m.group(1)))
    module_present = chosen_idx in mount_indices_seen
    extra_mounts = sorted(mount_indices_seen - {chosen_idx})
    if not module_present or extra_mounts:
        matched = False

    return {
        "checked": True,
        "matched": matched,
        "deltas": deltas,
        "expected_module": expected_module,
        "module_present": module_present,
        "extra_mounts": extra_mounts,
        "tb_tf_parent": tb_tf["parent"],
        "n_static_frames": len(tf),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_spawn_match.py <sweep_dir>", file=sys.stderr)
        return 2

    sweep_dir = Path(sys.argv[1])
    if not sweep_dir.exists():
        print(f"missing: {sweep_dir}", file=sys.stderr)
        return 1

    runs = sorted((sweep_dir / "seeds").glob("seed_*"))
    results = []
    matched_count = 0
    checked_count = 0

    for seed_dir in runs:
        seed = int(seed_dir.name.split("_")[1])
        cfg_path = sweep_dir / "configs" / f"seed_{seed:02d}.yaml"
        info = verify_seed(seed_dir, cfg_path) if cfg_path.exists() else {
            "checked": False, "reason": "no config",
        }
        info["seed"] = seed
        results.append(info)
        if info.get("checked"):
            checked_count += 1
            if info.get("matched"):
                matched_count += 1

    out = {
        "n_total": len(runs),
        "n_checked": checked_count,
        "n_matched": matched_count,
        "results": results,
    }
    (sweep_dir / "spawn_verification.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )

    print(f"verified {matched_count}/{checked_count} (of {len(runs)} seeds)")
    for r in results:
        if not r.get("checked"):
            print(f"  seed {r['seed']:02d}: SKIPPED — {r.get('reason')}")
        elif not r.get("matched"):
            d = r["deltas"]
            note = ""
            if not r.get("module_present"):
                note = f"  (module {r.get('expected_module')!r} missing)"
            if r.get("extra_mounts"):
                note += f"  (extra nic_card_mount indices: {r['extra_mounts']})"
            print(f"  seed {r['seed']:02d}: MISMATCH "
                  f"dx={d['x']['delta']*1000:+.2f}mm "
                  f"dy={d['y']['delta']*1000:+.2f}mm "
                  f"dyaw={d['yaw']['delta']*1000:+.2f}mrad{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
