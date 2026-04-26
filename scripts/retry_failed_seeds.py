#!/usr/bin/env python3
"""Retry seeds whose policy log is missing or empty.

Walks <sweep_dir>/seeds/seed_NN/, classifies each as ok / failed based on
whether the policy log has a 'CheatCodeMJ done.' line, and re-runs the
failed seeds via record_episode.sh --output-dir <seed_dir>. Wipes the
seed's previous tree before retrying so old artifacts don't leak through.
Updates summary.json in place.

Usage:
  AIC_USE_DOCKER_EXEC=1 \
    src/aic/scripts/retry_failed_seeds.py <sweep_dir> [--max-retries 2]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def policy_succeeded(output_dir: str) -> bool:
    if not output_dir:
        return False
    log = Path(output_dir) / "terminal2_policy.log"
    if not log.exists():
        return False
    try:
        for line in reversed(log.read_text(errors="ignore").splitlines()):
            if "CheatCodeMJ done" in line:
                return True
    except Exception:
        return False
    return False


def find_dataset(seed_dir: Path) -> Path | None:
    """Return <seed_dir>/dataset/ if present."""
    ds = seed_dir / "dataset"
    return ds if ds.is_dir() else None


def run_one(record_script: Path, cfg_path: Path, seed_dir: Path,
            timeout_s: int) -> dict:
    """Re-run one seed, single-tree layout under seed_dir."""
    import shutil
    if seed_dir.exists():
        # Wipe the previous attempt; bag/dataset/etc. would otherwise
        # collide. Keep the parent dirs.
        for child in seed_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except Exception:
                    pass
    seed_dir.mkdir(parents=True, exist_ok=True)
    driver_log = seed_dir / "driver.log"
    cmd = [
        str(record_script), "sfp",
        "--config", str(cfg_path),
        "--output-dir", str(seed_dir),
        "--timeout", str(timeout_s),
    ]
    start = time.time()
    with open(driver_log, "wb") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(record_script.parents[2]), check=False)
    elapsed = time.time() - start
    return {"exit_code": proc.returncode, "elapsed_s": round(elapsed, 1),
            "output_dir": str(seed_dir)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir", type=Path)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--timeout-s", type=int, default=300)
    args = p.parse_args()

    sweep = args.sweep_dir
    summary_path = sweep / "summary.json"
    summary = json.loads(summary_path.read_text())
    record_script = Path("/home/hariharan/ws_aic/src/aic/scripts/record_episode.sh")

    failed_seeds = []
    for r in summary["results"]:
        if not r["policy"].get("found"):
            failed_seeds.append(r["seed"])

    print(f"Found {len(failed_seeds)} failed seeds: {failed_seeds}")
    if not failed_seeds:
        return 0

    for seed in failed_seeds:
        cfg_path = sweep / "configs" / f"seed_{seed:02d}.yaml"
        seed_dir = sweep / "seeds" / f"seed_{seed:02d}"
        if not cfg_path.exists():
            print(f"seed {seed:02d}: skipping — no config")
            continue

        for attempt in range(1, args.max_retries + 1):
            print(f"seed {seed:02d}: retry attempt {attempt}", flush=True)
            run_info = run_one(record_script, cfg_path, seed_dir,
                               args.timeout_s)
            if policy_succeeded(run_info["output_dir"]):
                # Update summary.json in place.
                for r in summary["results"]:
                    if r["seed"] == seed:
                        # Re-parse the new policy log
                        text = Path(run_info["output_dir"], "terminal2_policy.log").read_text()
                        for line in reversed(text.splitlines()):
                            if "CheatCodeMJ done" in line:
                                inserted = "inserted=True" in line
                                attempts = None
                                final_dist = None
                                if "attempts=" in line:
                                    try:
                                        attempts = int(line.rsplit("attempts=", 1)[1].split()[0].rstrip(","))
                                    except Exception:
                                        pass
                                if "plug-port dist:" in line:
                                    try:
                                        seg = line.split("plug-port dist:", 1)[1]
                                        final_dist = float(seg.strip().split("m")[0])
                                    except Exception:
                                        pass
                                r["policy"] = {
                                    "found": True,
                                    "inserted": inserted,
                                    "attempts": attempts,
                                    "final_dist_m": final_dist,
                                }
                                break
                        r["run"] = {**r.get("run", {}), **run_info,
                                     "retry_attempt": attempt}
                        ds = find_dataset(seed_dir)
                        if ds:
                            r["dataset"]["path"] = str(ds)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                print(f"  → recovered on attempt {attempt}")
                break
            else:
                print(f"  → still failed (elapsed {run_info['elapsed_s']}s)")
        else:
            print(f"seed {seed:02d}: gave up after {args.max_retries} retries")

    return 0


if __name__ == "__main__":
    sys.exit(main())
