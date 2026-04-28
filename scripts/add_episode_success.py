#!/usr/bin/env python3
"""Add a per-frame `episode_success` flag to a LeRobotDataset.

Reads <sweep_dir>/summary.json for the per-seed insertion outcome,
then for each parquet at <dataset>/data/chunk-NNN/file-NNN.parquet
appends an `episode_success` column (1.0 if inserted else 0.0,
constant across the episode's frames) and registers the feature
in meta/info.json.

Idempotent: re-running on a dataset that already has the column
is a no-op per parquet. Safe to run before pushing to HF.

Usage (pixi env — has pyarrow + lerobot):
    pixi run --manifest-path src/aic/pixi.toml \\
        python src/aic/scripts/add_episode_success.py <sweep_dir>

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
    # Episode index in the shared dataset == seed number
    # (one episode per seed, appended in seed order).
    success_by_seed = {
        int(r["seed"]): bool(r["policy"].get("inserted"))
        for r in summary["results"]
    }

    info = json.loads(info_path.read_text())
    feats = info.setdefault("features", {})
    if "episode_success" not in feats:
        feats["episode_success"] = {
            "dtype": "float32",
            "shape": [1],
            "names": ["success"],
        }
        info_path.write_text(json.dumps(info, indent=4) + "\n")
        print(f"  registered feature 'episode_success' in {info_path}")
    else:
        print(f"  feature 'episode_success' already in info.json")

    pq_files = sorted((dataset / "data").glob("chunk-*/file-*.parquet"))
    if not pq_files:
        print(f"no parquets under {dataset}/data/", file=sys.stderr)
        return 1

    n_updated = n_inserted_eps = n_failed_eps = n_unknown_eps = 0
    for p in pq_files:
        # LeRobot v3.0 names parquets file-EEEEEE.parquet where EEEEEE
        # is the episode_index. Pull it from the file rather than the
        # filename so we don't depend on the naming scheme.
        tbl = pq.read_table(p)
        # If episode_success already present (re-run), drop it so we can
        # rewrite with the correct scalar dtype.
        if "episode_success" in tbl.column_names:
            tbl = tbl.drop_columns(["episode_success"])
        eps = tbl.column("episode_index").to_pylist()
        unique_eps = set(eps)
        if len(unique_eps) != 1:
            print(f"WARN: {p.name} has {len(unique_eps)} episodes (expected 1); "
                  f"using first sample's success", file=sys.stderr)
        ep = next(iter(unique_eps))
        if ep not in success_by_seed:
            success = 0.0
            n_unknown_eps += 1
        elif success_by_seed[ep]:
            success = 1.0
            n_inserted_eps += 1
        else:
            success = 0.0
            n_failed_eps += 1
        # Match the parquet dtype convention this dataset uses for shape-[1]
        # scalar-like features (timestamp, frame_index): plain scalar column,
        # NOT a fixed-size list. Mismatched types fail HF datasets schema check.
        col = pa.array([float(success)] * tbl.num_rows, type=pa.float32())
        tbl = tbl.append_column("episode_success", col)
        pq.write_table(tbl, p)
        n_updated += 1

    print(f"  parquets updated: {n_updated}")
    print(f"  inserted episodes: {n_inserted_eps}")
    print(f"  failed   episodes: {n_failed_eps}")
    print(f"  unknown  episodes: {n_unknown_eps}")

    # Compute + write stats over the full dataset so LeRobot's loader doesn't
    # complain about a feature without normalisation stats.
    total_frames = 0
    n_one = 0
    for p in pq_files:
        tbl = pq.read_table(p, columns=["episode_success"])
        for v in tbl.column("episode_success").to_pylist():
            total_frames += 1
            if v is not None and v >= 0.5:
                n_one += 1
    p_success = n_one / total_frames if total_frames else 0.0
    # std for Bernoulli with prob p_success.
    std = (p_success * (1.0 - p_success)) ** 0.5
    # Percentiles for binary data: q[X] = 0 if X% <= (1-p_success), else 1.
    def q(pct: float) -> float:
        return 0.0 if (pct / 100.0) <= (1.0 - p_success) else 1.0

    stats_path = dataset / "meta" / "stats.json"
    if stats_path.exists():
        all_stats = json.loads(stats_path.read_text())
    else:
        all_stats = {}
    all_stats["episode_success"] = {
        "min":   [0.0],
        "max":   [1.0],
        "mean":  [p_success],
        "std":   [std],
        "count": [total_frames],
        "q01":   [q(1)],
        "q10":   [q(10)],
        "q50":   [q(50)],
        "q90":   [q(90)],
        "q99":   [q(99)],
    }
    stats_path.write_text(json.dumps(all_stats, indent=4) + "\n")
    print(f"  wrote stats: count={total_frames}, mean={p_success:.4f}, std={std:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
