#!/usr/bin/env python3
"""Sparse SFP spawn sweep: sample, template, run, validate, summarize.

Generates N spawn-config samples from the ranges documented in
[[sfp-spawn-parameters]], writes one YAML per sample (templated from
single_trial_sfp.yaml), runs record_episode.sh against each, then
validates each recorded LeRobot dataset and aggregates insertion success.

All artefacts go under <results>/spawn_sweep_<TS>/:
  configs/seed_NN.yaml
  runs/seed_NN/{driver.log, output_dir.txt}
  datasets/seed_NN/aic_recording_<TS>/
  samples.json
  summary.json

Usage:
  pixi run python src/aic/scripts/spawn_sweep_sfp.py [--n 20] [--seed 0]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ────────────────────────────────────────────────────────────────────────────
# Sample-space definition (from wiki/methodology/sfp-spawn-parameters.md).
# ────────────────────────────────────────────────────────────────────────────

# Continuous knobs: (name, low, high). Sampled by stratified-uniform.
# Each row picks one value per stratum (n_samples strata) then shuffles.
CONTINUOUS = [
    # Task board pose
    ("board_x",          0.150 - 0.005,    0.150 + 0.005),    # ±5 mm   (Isaac DR)
    ("board_y",         -0.200 - 0.005,   -0.200 + 0.005),    # ±5 mm
    ("board_yaw",        3.1415 - 0.15,    3.1415 + 0.15),    # ±0.15 rad (suggested)

    # Target NIC rail (within-rail translation + yaw)
    ("nic_translation", -0.0215,           0.0234),           # engine clamp
    ("nic_yaw",         -0.05,             0.05),             # ±0.05 rad (suggested)

    # Cable grasp
    ("grip_x",           0.0   - 0.002,    0.0   + 0.002),    # ±2 mm
    ("grip_y",           0.015385 - 0.002, 0.015385 + 0.002), # ±2 mm
    ("grip_z",           0.04045,          0.04545),          # empirical SFP range
]

# Discrete: nic_rail choice, distributed roughly evenly.
NIC_RAIL_CHOICES = [0, 1, 2, 3, 4]


def stratified_uniform(low: float, high: float, n: int, rng: random.Random) -> list[float]:
    """One sample per stratum, jittered within the stratum, shuffled."""
    width = (high - low) / n
    samples = [low + (i + rng.random()) * width for i in range(n)]
    rng.shuffle(samples)
    return samples


def sample_specs(n: int, seed: int) -> list[dict]:
    """Build n spawn-spec dicts, one per episode."""
    rng = random.Random(seed)

    cols: dict[str, list[float]] = {}
    for name, lo, hi in CONTINUOUS:
        cols[name] = stratified_uniform(lo, hi, n, rng)

    # Discrete rail: cycle through 0..4 then top-up from rng to total n.
    rails = (NIC_RAIL_CHOICES * (n // len(NIC_RAIL_CHOICES) + 1))[:n]
    rng.shuffle(rails)

    specs: list[dict] = []
    for i in range(n):
        specs.append({
            "seed": i,
            "nic_rail":        rails[i],
            "board_x":         cols["board_x"][i],
            "board_y":         cols["board_y"][i],
            "board_yaw":       cols["board_yaw"][i],
            "nic_translation": cols["nic_translation"][i],
            "nic_yaw":         cols["nic_yaw"][i],
            "grip_x":          cols["grip_x"][i],
            "grip_y":          cols["grip_y"][i],
            "grip_z":          cols["grip_z"][i],
        })
    return specs


# ────────────────────────────────────────────────────────────────────────────
# YAML templating.
# ────────────────────────────────────────────────────────────────────────────

def templated_config(base_yaml: dict, spec: dict) -> dict:
    """Return a deep-copied YAML dict with spawn fields replaced from spec."""
    cfg = json.loads(json.dumps(base_yaml))  # cheap deep-copy of plain dict

    trial = cfg["trials"]["trial_1"]
    scene = trial["scene"]

    # --- Task board pose -------------------------------------------------
    scene["task_board"]["pose"]["x"]   = float(spec["board_x"])
    scene["task_board"]["pose"]["y"]   = float(spec["board_y"])
    scene["task_board"]["pose"]["yaw"] = float(spec["board_yaw"])

    # --- Target NIC rail -------------------------------------------------
    chosen = int(spec["nic_rail"])
    for i in range(5):
        rail_key = f"nic_rail_{i}"
        if i == chosen:
            scene["task_board"][rail_key] = {
                "entity_present": True,
                "entity_name":    f"nic_card_{i}",
                "entity_pose": {
                    "translation": float(spec["nic_translation"]),
                    "roll":  0.0,
                    "pitch": 0.0,
                    "yaw":   float(spec["nic_yaw"]),
                },
            }
        else:
            scene["task_board"][rail_key] = {"entity_present": False}

    # --- Cable grasp -----------------------------------------------------
    cable = scene["cables"]["cable_0"]["pose"]
    cable["gripper_offset"]["x"] = float(spec["grip_x"])
    cable["gripper_offset"]["y"] = float(spec["grip_y"])
    cable["gripper_offset"]["z"] = float(spec["grip_z"])
    # roll / pitch / yaw kept at defaults (no DR source).

    # --- Task wiring -----------------------------------------------------
    trial["tasks"]["task_1"]["target_module_name"] = f"nic_card_mount_{chosen}"

    return cfg


# ────────────────────────────────────────────────────────────────────────────
# Run driver.
# ────────────────────────────────────────────────────────────────────────────

def run_one_episode(
    *,
    record_script: Path,
    config_path: Path,
    dataset_root: Path,
    log_dir: Path,
    timeout_s: int,
) -> dict:
    """Run one episode. Returns a dict with timing + paths + exit info."""
    log_dir.mkdir(parents=True, exist_ok=True)
    driver_log = log_dir / "driver.log"

    cmd = [
        str(record_script), "sfp",
        "--config", str(config_path),
        "--dataset-root", str(dataset_root),
        "--timeout", str(timeout_s),
    ]

    start = time.time()
    with open(driver_log, "wb") as fh:
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(record_script.parents[2]),
            check=False,
        )
    elapsed = time.time() - start

    # Extract record_episode.sh's OUTPUT_DIR (stamps logs/scoring) from driver
    # log. The shell prints "output:        <OUTPUT_DIR>".
    output_dir = ""
    try:
        for line in driver_log.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("output:"):
                output_dir = stripped.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    (log_dir / "output_dir.txt").write_text(output_dir + "\n")

    return {
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "output_dir": output_dir,
        "driver_log": str(driver_log),
    }


# ────────────────────────────────────────────────────────────────────────────
# Per-episode validation + verification.
# ────────────────────────────────────────────────────────────────────────────

def parse_policy_log(output_dir: str) -> dict:
    """Pull insertion-status fields out of terminal2_policy.log."""
    if not output_dir:
        return {"found": False}
    log = Path(output_dir) / "terminal2_policy.log"
    if not log.exists():
        return {"found": False}
    text = log.read_text(errors="ignore")
    inserted = None
    attempts = None
    final_dist = None
    # Final summary line: "CheatCodeMJ done. inserted=True, plug-port dist: 0.0011m, attempts=1"
    for line in reversed(text.splitlines()):
        if "CheatCodeMJ done" in line:
            try:
                inserted = "inserted=True" in line
                if "attempts=" in line:
                    attempts = int(line.rsplit("attempts=", 1)[1].split()[0].rstrip(","))
                if "plug-port dist:" in line:
                    seg = line.split("plug-port dist:", 1)[1]
                    final_dist = float(seg.strip().split("m")[0])
            except Exception:
                pass
            break
    return {
        "found": inserted is not None,
        "inserted": inserted,
        "attempts": attempts,
        "final_dist_m": final_dist,
    }


def find_dataset(dataset_root: Path) -> Path | None:
    """Return the single aic_recording_* dir under dataset_root, if any."""
    if not dataset_root.exists():
        return None
    candidates = sorted(p for p in dataset_root.iterdir()
                        if p.is_dir() and p.name.startswith("aic_recording_"))
    return candidates[-1] if candidates else None


def validate_dataset(ds_dir: Path) -> dict:
    """Fast schema checks against meta/info.json + meta/episodes/."""
    info_path = ds_dir / "meta" / "info.json"
    if not info_path.exists():
        return {"ok": False, "reason": "no meta/info.json"}
    info = json.loads(info_path.read_text())
    feats = info.get("features", {})
    expected_state = feats.get("observation.state", {}).get("shape", [])
    expected_action = feats.get("action", {}).get("shape", [])

    return {
        "ok": (
            info.get("total_episodes", 0) >= 1
            and info.get("total_frames", 0) > 0
            and expected_state == [27]
            and expected_action == [9]
        ),
        "frames": info.get("total_frames"),
        "episodes": info.get("total_episodes"),
        "fps": info.get("fps"),
        "state_shape": expected_state,
        "action_shape": expected_action,
        "video_keys": [k for k in feats if k.startswith("observation.images.")],
    }


def first_frame_state(ds_dir: Path) -> list | None:
    """Read the first row of observation.state from the parquet file."""
    try:
        import pyarrow.parquet as pq  # type: ignore
        chunks = sorted((ds_dir / "data").glob("chunk-*/file-*.parquet"))
        if not chunks:
            return None
        tbl = pq.read_table(chunks[0], columns=["observation.state"])
        if tbl.num_rows == 0:
            return None
        first = tbl.column("observation.state")[0].as_py()
        return list(first)
    except Exception as exc:
        return [f"err:{exc!r}"]


# ────────────────────────────────────────────────────────────────────────────
# Spawn-vs-config verification.
# ────────────────────────────────────────────────────────────────────────────

def verify_spawn_matches_config(spec: dict, output_dir: str) -> dict:
    """Spawn-vs-config verification deferred to a post-pass over the bag.

    The engine doesn't echo spawn args into terminal logs in a parseable
    form — the authoritative source is the per-trial rosbag's /tf_static.
    A separate post-processing pass (verify_spawn_match.py) reads each bag
    and compares to the YAML.
    """
    bag_dir = ""
    if output_dir:
        for p in Path(output_dir).glob("bag_trial_1*/"):
            bag_dir = str(p)
            break
    return {
        "checked": False,
        "deferred": True,
        "bag_dir": bag_dir,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main.
# ────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ws", type=Path, default=Path("/home/hariharan/ws_aic"))
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument("--start-from", type=int, default=0,
                   help="Skip seeds 0..start_from-1 (resume support).")
    p.add_argument("--sweep-dir", type=Path, default=None,
                   help="Reuse an existing sweep_dir (for resume). Default: new TS dir.")
    args = p.parse_args()

    ws = args.ws
    src_aic = ws / "src" / "aic"
    record_script = src_aic / "scripts" / "record_episode.sh"
    base_yaml_path = src_aic / "aic_engine" / "config" / "single_trial_sfp.yaml"

    if not record_script.exists():
        print(f"missing: {record_script}", file=sys.stderr)
        return 1
    if not base_yaml_path.exists():
        print(f"missing: {base_yaml_path}", file=sys.stderr)
        return 1

    base_yaml = yaml.safe_load(base_yaml_path.read_text())

    if args.sweep_dir:
        sweep_dir = args.sweep_dir
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = ws / "aic_results" / f"spawn_sweep_{ts}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "configs").mkdir(exist_ok=True)
    (sweep_dir / "runs").mkdir(exist_ok=True)
    (sweep_dir / "datasets").mkdir(exist_ok=True)

    print(f"sweep dir: {sweep_dir}", flush=True)

    # 1) Sample.
    specs = sample_specs(args.n, args.seed)
    (sweep_dir / "samples.json").write_text(json.dumps(specs, indent=2) + "\n")
    print(f"sampled {len(specs)} specs (seed={args.seed})", flush=True)

    # 2) Template + run.
    sweep_start = time.time()
    results = []

    for spec in specs:
        seed = spec["seed"]
        if seed < args.start_from:
            continue
        cfg = templated_config(base_yaml, spec)
        cfg_path = sweep_dir / "configs" / f"seed_{seed:02d}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        ds_root = sweep_dir / "datasets" / f"seed_{seed:02d}"
        log_dir = sweep_dir / "runs" / f"seed_{seed:02d}"

        print(f"[seed {seed:02d}] rail={spec['nic_rail']} "
              f"board=({spec['board_x']:+.4f},{spec['board_y']:+.4f},"
              f"yaw={spec['board_yaw']:.4f}) "
              f"nic=({spec['nic_translation']:+.4f},yaw={spec['nic_yaw']:+.3f}) "
              f"grip=({spec['grip_x']:+.4f},{spec['grip_y']:+.4f},"
              f"{spec['grip_z']:.4f})", flush=True)

        run_info = run_one_episode(
            record_script=record_script,
            config_path=cfg_path,
            dataset_root=ds_root,
            log_dir=log_dir,
            timeout_s=args.timeout_s,
        )
        print(f"  → exit={run_info['exit_code']} "
              f"elapsed={run_info['elapsed_s']}s", flush=True)

        # Per-episode summary: dataset, policy log, spawn match.
        ds_dir = find_dataset(ds_root)
        ds_check = validate_dataset(ds_dir) if ds_dir else {"ok": False, "reason": "no dataset"}
        policy = parse_policy_log(run_info["output_dir"])
        spawn = verify_spawn_matches_config(spec, run_info["output_dir"])

        results.append({
            "seed": seed,
            "spec": spec,
            "run":  run_info,
            "policy": policy,
            "dataset": {
                "path": str(ds_dir) if ds_dir else None,
                **ds_check,
            },
            "spawn_match": spawn,
        })

        # Persist after each episode so a crash mid-sweep keeps progress.
        (sweep_dir / "summary.json").write_text(json.dumps({
            "n": args.n, "seed": args.seed,
            "started_at": dt.datetime.fromtimestamp(sweep_start).isoformat(timespec="seconds"),
            "elapsed_s": round(time.time() - sweep_start, 1),
            "results": results,
        }, indent=2) + "\n")

    total_elapsed = time.time() - sweep_start

    # 3) Summary print.
    inserted_count = sum(1 for r in results if r["policy"].get("inserted") is True)
    print()
    print(f"=== sweep complete in {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f} min) ===")
    print(f"  episodes ran:    {len(results)}/{args.n}")
    print(f"  inserted=True:   {inserted_count}/{len(results)}")
    print(f"  sweep dir:       {sweep_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
