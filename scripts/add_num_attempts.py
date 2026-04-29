#!/usr/bin/env python3
"""Add a per-frame `num_attempts` column to a LeRobotDataset.

Reads <sweep_dir>/summary.json for each seed's `policy.attempts`
(CheatCodeMJ's per-episode retry count: 1 = first-try insert,
2-3 = retried, capped at max_insertion_retries+1), then for each
parquet at <dataset>/data/chunk-NNN/file-NNN.parquet appends a
`num_attempts` column (constant across the episode's frames) and
registers the feature in meta/info.json + meta/stats.json.

Why: vanilla BC training contaminates near-port behavior when an
episode contains lift-and-redescend cycles — the same (state, near-port-z)
gets both "lift up" and "keep descending" labels. Filter on
`episode_success == 1 AND num_attempts == 1` for the cleanest subset.

Idempotent: re-running on a dataset that already has the column
rewrites it from current summary.json. Safe to run before pushing to HF.

Strict post-pass requirement (per [[lerobot-schema-mid-sweep-mismatch]]):
do NOT run this while a sweep is in flight — only after the sweep is
guaranteed done.

Usage (pixi env — has pyarrow + lerobot):
    pixi run --manifest-path src/aic/pixi.toml \\
        python src/aic/scripts/add_num_attempts.py <sweep_dir>

Optional --dataset PATH overrides the default <sweep_dir>/dataset/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="Override the dataset path "
                         "(default: <sweep_dir>/dataset/).")
    args = ap.parse_args()

    sweep = args.sweep_dir
    dataset = args.dataset or (sweep / "dataset")
    summary_path = sweep / "summary.json"
    info_path = dataset / "meta" / "info.json"

    if not summary_path.exists():
        print(f"missing: {summary_path}", file=sys.stderr)
        return 1
    if not info_path.exists():
        print(f"missing: {info_path}", file=sys.stderr)
        return 1

    summary = json.loads(summary_path.read_text())
    attempts_by_seed: dict[int, int] = {}
    for r in summary["results"]:
        p = r.get("policy", {})
        n = p.get("attempts")
        if n is None:
            continue
        attempts_by_seed[int(r["seed"])] = int(n)

    info = json.loads(info_path.read_text())
    feats = info.setdefault("features", {})
    if "num_attempts" not in feats:
        feats["num_attempts"] = {
            "dtype": "int32",
            "shape": [1],
            "names": ["attempts"],
        }
        info_path.write_text(json.dumps(info, indent=4) + "\n")
        print(f"  registered feature 'num_attempts' in {info_path}")
    else:
        print(f"  feature 'num_attempts' already in info.json")

    pq_files = sorted((dataset / "data").glob("chunk-*/file-*.parquet"))
    if not pq_files:
        print(f"no parquets under {dataset}/data/", file=sys.stderr)
        return 1

    n_updated = 0
    n_unknown_eps = 0
    attempt_hist: dict[int, int] = {}
    for p in pq_files:
        tbl = pq.read_table(p)
        if "num_attempts" in tbl.column_names:
            tbl = tbl.drop_columns(["num_attempts"])
        eps = tbl.column("episode_index").to_pylist()
        unique_eps = set(eps)
        if len(unique_eps) != 1:
            print(f"WARN: {p.name} has {len(unique_eps)} episodes (expected 1); "
                  f"using first sample's attempt count", file=sys.stderr)
        ep = next(iter(unique_eps))
        if ep not in attempts_by_seed:
            n = 0
            n_unknown_eps += 1
        else:
            n = attempts_by_seed[ep]
        attempt_hist[n] = attempt_hist.get(n, 0) + 1
        # Match the parquet dtype convention used by other shape-[1] scalar
        # features (frame_index is int64, episode_success is float32). int32
        # is the right fit for a small bounded count.
        col = pa.array([int(n)] * tbl.num_rows, type=pa.int32())
        tbl = tbl.append_column("num_attempts", col)
        pq.write_table(tbl, p)
        n_updated += 1

    print(f"  parquets updated: {n_updated}")
    print(f"  unknown episodes: {n_unknown_eps}")
    print(f"  episode-attempts histogram:")
    for k in sorted(attempt_hist):
        print(f"    {k} attempts: {attempt_hist[k]} episodes")

    # Compute + write stats over the full dataset so LeRobot's loader has
    # normalisation stats for the new column.
    total_frames = 0
    sum_n = 0
    sum_sq = 0
    min_n = None
    max_n = None
    counts: list[int] = []
    for p in pq_files:
        tbl = pq.read_table(p, columns=["num_attempts"])
        for v in tbl.column("num_attempts").to_pylist():
            if v is None:
                continue
            total_frames += 1
            sum_n += v
            sum_sq += v * v
            counts.append(v)
            if min_n is None or v < min_n:
                min_n = v
            if max_n is None or v > max_n:
                max_n = v

    if total_frames == 0:
        print("ERROR: no frames found across data parquets", file=sys.stderr)
        return 1

    mean = sum_n / total_frames
    var = sum_sq / total_frames - mean * mean
    std = var ** 0.5 if var > 0 else 0.0
    counts.sort()

    def q(pct: float) -> float:
        if not counts:
            return 0.0
        i = int(round((pct / 100.0) * (len(counts) - 1)))
        return float(counts[max(0, min(i, len(counts) - 1))])

    stats_path = dataset / "meta" / "stats.json"
    if stats_path.exists():
        all_stats = json.loads(stats_path.read_text())
    else:
        all_stats = {}
    all_stats["num_attempts"] = {
        "min":   [float(min_n)],
        "max":   [float(max_n)],
        "mean":  [float(mean)],
        "std":   [float(std)],
        "count": [total_frames],
        "q01":   [q(1)],
        "q10":   [q(10)],
        "q50":   [q(50)],
        "q90":   [q(90)],
        "q99":   [q(99)],
    }
    stats_path.write_text(json.dumps(all_stats, indent=4) + "\n")
    print(f"  wrote stats: count={total_frames}, "
          f"mean={mean:.4f}, std={std:.4f}, "
          f"min={min_n}, max={max_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
