#!/usr/bin/env python3
"""Build the strict-clean episode list for the AIC SFP dataset.

Reads each episode's first row from the dataset's data parquets to
extract `episode_success` and `num_attempts`, filters to the strict
training subset (success=1 AND attempts=1), and emits a comma-separated
list to stdout (or to a file with --output).

Why this exists: lerobot 0.5.1 silently drops int32 features from
LeRobotDataset.__getitem__, so we can't filter via lerobot's API
(see [[lerobot-int32-feature-filter]] in the wiki). Direct pyarrow
reads work fine.

Cost: ~30 MB total downloaded across the per-episode parquets (videos
not pulled). ~1 min on a fresh cache.

Usage (default reads HexDexAIC/aic-sfp-500):
    python build_clean_eps.py > clean_eps.txt
    # or
    python build_clean_eps.py --output ~/clean_eps.txt --n 500

Then pass to lerobot-train:
    lerobot-train --dataset.repo_id=HexDexAIC/aic-sfp-500 \\
        --dataset.episodes="[$(cat clean_eps.txt)]" ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo-id", default="HexDexAIC/aic-sfp-500",
                    help="HF dataset repo id (default: HexDexAIC/aic-sfp-500).")
    ap.add_argument("--n", type=int, default=500,
                    help="Total number of episodes in the dataset (default 500).")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Write the comma-separated list here. Default: stdout.")
    ap.add_argument("--lenient", action="store_true",
                    help="Use lenient filter (success=1 only) instead of strict "
                         "(success=1 AND attempts=1). Lenient includes the 2 "
                         "retry-success episodes.")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-episode progress to stderr.")
    args = ap.parse_args()

    clean: list[int] = []
    n_failed = n_retry = n_clean = 0

    for ep in range(args.n):
        if not args.quiet and ep % 50 == 0:
            print(f"  fetching ep {ep}/{args.n}...", file=sys.stderr, flush=True)
        path = hf_hub_download(
            args.repo_id,
            f"data/chunk-000/file-{ep:03d}.parquet",
            repo_type="dataset",
        )
        tbl = pq.read_table(path, columns=["episode_success", "num_attempts"])
        es = tbl.column("episode_success")[0].as_py()
        na = tbl.column("num_attempts")[0].as_py()

        if es < 0.5:
            n_failed += 1
            continue
        if na > 1:
            n_retry += 1
            if args.lenient:
                clean.append(ep)
            continue
        clean.append(ep)
        n_clean += 1

    out = ",".join(str(e) for e in clean)
    if args.output is not None:
        args.output.write_text(out + "\n")
        print(f"wrote {len(clean)} episode ids to {args.output}", file=sys.stderr)
    else:
        print(out)

    print(f"  strict-clean (success=1, attempts=1): {n_clean}", file=sys.stderr)
    print(f"  retry-success (success=1, attempts>1): {n_retry}", file=sys.stderr)
    print(f"  failed:                              {n_failed}", file=sys.stderr)
    print(f"  total emitted:                       {len(clean)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
