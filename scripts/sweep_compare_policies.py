"""Compare CheatCodeMJVision vs VisionInsert_v3 across N spawn seeds.

Generates N unique single-trial SFP configs (reusing
spawn_sweep_sfp.sample_specs/templated_config), then runs each config
through both policies via record_episode.sh. Aggregates scoring.yaml
into a single CSV.

Usage:
    pixi run --as-is python scripts/sweep_compare_policies.py \\
        --n 20 --sweep-dir ~/aic_results/policy_compare_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Reuse spawn_sweep_sfp's spec generator + templating.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import spawn_sweep_sfp  # type: ignore

POLICIES = ["CheatCodeMJVision", "VisionInsert_v3"]


def parse_scoring(scoring_path: Path) -> dict:
    if not scoring_path.exists():
        return {"total": None, "tier_3_msg": "MISSING", "final_dist_m": None}
    try:
        data = yaml.safe_load(scoring_path.read_text())
    except Exception as ex:
        return {"total": None, "tier_3_msg": f"PARSE_ERR: {ex}", "final_dist_m": None}
    total = float(data.get("total", 0.0))
    trial = data.get("trial_1") or {}
    tier_3 = trial.get("tier_3", {}) or {}
    tier_3_msg = tier_3.get("message", "")
    tier_3_score = float(tier_3.get("score", 0.0))
    tier_2 = trial.get("tier_2", {}) or {}
    tier_2_score = float(tier_2.get("score", 0.0))
    # Pull final dist from message if present.
    final_dist = None
    for token in tier_3_msg.replace(":", " ").replace("m.", "m").split():
        if token.endswith("m") and "." in token[:-1]:
            try:
                final_dist = float(token.rstrip("m"))
                break
            except ValueError:
                pass
    return {
        "total": total,
        "tier_2_score": tier_2_score,
        "tier_3_score": tier_3_score,
        "tier_3_msg": tier_3_msg,
        "final_dist_m": final_dist,
    }


def run_one(record_script: Path, config_path: Path, output_dir: Path,
            policy: str, timeout_s: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(record_script), "sfp",
        "--policy", policy,
        "--config", str(config_path),
        "--output-dir", str(output_dir),
        "--no-record", "--headless",
        "--timeout", str(timeout_s),
    ]
    driver_log = output_dir / "driver.log"
    # Ensure record_episode.sh uses `docker exec` (no sudo / no distrobox).
    # Without this, the script tries distrobox + sudo and dies in <5s
    # waiting on a TTY password prompt.
    env = {
        **os.environ,
        "AIC_USE_DOCKER_EXEC": "1",
        "AIC_V1_WEIGHTS": os.environ.get(
            "AIC_V1_WEIGHTS",
            str(Path.home() / "aic_runs/v1_h100_results/best.pt"),
        ),
    }
    start = time.time()
    with open(driver_log, "wb") as fh:
        proc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT,
            cwd=str(record_script.parents[2]), check=False, env=env,
        )
    elapsed = time.time() - start
    return {
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        **parse_scoring(output_dir / "scoring.yaml"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0,
                   help="Spec sampling seed (deterministic across machines)")
    p.add_argument("--ws", type=Path, default=Path.home() / "ws_aic")
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument("--sweep-dir", type=Path, default=None)
    args = p.parse_args()

    src_aic = args.ws / "src" / "aic"
    record_script = src_aic / "scripts" / "record_episode.sh"
    base_yaml_path = src_aic / "aic_engine" / "config" / "single_trial_sfp.yaml"
    if not record_script.exists():
        print(f"missing: {record_script}", file=sys.stderr); return 1
    if not base_yaml_path.exists():
        print(f"missing: {base_yaml_path}", file=sys.stderr); return 1

    base_yaml = yaml.safe_load(base_yaml_path.read_text())

    if args.sweep_dir:
        sweep_dir = args.sweep_dir.expanduser().resolve()
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = (args.ws / "aic_results" / f"policy_compare_{ts}").resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "configs").mkdir(exist_ok=True)
    print(f"sweep dir: {sweep_dir}")

    specs = spawn_sweep_sfp.sample_specs(args.n, args.seed)
    (sweep_dir / "samples.json").write_text(json.dumps(specs, indent=2) + "\n")
    print(f"sampled {len(specs)} specs (seed={args.seed})")

    summary_path = sweep_dir / "summary.csv"
    summary_path.write_text(
        "seed,policy,exit_code,elapsed_s,total,tier_2,tier_3,final_dist_m,tier_3_msg\n"
    )

    sweep_start = time.time()
    for i, spec in enumerate(specs):
        seed = spec["seed"]
        cfg = spawn_sweep_sfp.templated_config(base_yaml, spec)
        cfg_path = sweep_dir / "configs" / f"seed_{seed}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        for policy in POLICIES:
            out_dir = sweep_dir / "runs" / f"seed_{seed}" / policy
            print(f"\n[{i+1}/{len(specs)}] seed {seed}  policy {policy}  →  {out_dir}")
            result = run_one(record_script, cfg_path, out_dir, policy, args.timeout_s)
            tier_3_msg_clean = (result.get("tier_3_msg") or "").replace(",", ";").replace("\n", " ")
            with open(summary_path, "a") as fh:
                fh.write(
                    f"{seed},{policy},{result['exit_code']},{result['elapsed_s']},"
                    f"{result.get('total')},{result.get('tier_2_score')},"
                    f"{result.get('tier_3_score')},{result.get('final_dist_m')},"
                    f"{tier_3_msg_clean}\n"
                )
            print(
                f"  exit={result['exit_code']}  elapsed={result['elapsed_s']}s  "
                f"total={result.get('total')}  tier_3={result.get('tier_3_score')}  "
                f"dist={result.get('final_dist_m')}m"
            )

    elapsed_total = time.time() - sweep_start
    print(f"\nsweep complete in {elapsed_total/60:.1f} min")
    print(f"summary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
