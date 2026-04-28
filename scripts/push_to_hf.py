#!/usr/bin/env python3
"""Push an existing LeRobotDataset to the Hugging Face Hub — private by default.

Use this for sweeps that already write to a shared dataset
(<sweep>/dataset/) — no consolidation needed. For old sweeps with
per-seed datasets, use consolidate_and_push.py instead.

Auth (one-time, via the pixi env's HF CLI):
    pixi run --manifest-path src/aic/pixi.toml hf auth login
    # paste a token with write scope from
    # https://huggingface.co/settings/tokens

Usage:
    pixi run --manifest-path src/aic/pixi.toml \\
        python src/aic/scripts/push_to_hf.py <dataset_dir> \\
        --repo-id YOUR_USER/aic-sfp-1000 [--public]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dataset_dir", type=Path,
                    help="Path to the LeRobotDataset directory "
                         "(e.g. <sweep>/dataset/).")
    ap.add_argument("--repo-id", required=True,
                    help="HF repo id, e.g. YOUR_USER/aic-sfp-1000")
    ap.add_argument("--public", action="store_true",
                    help="Push as a PUBLIC HF repo (default: private).")
    ap.add_argument("--license", default="apache-2.0")
    args = ap.parse_args()

    if not (args.dataset_dir / "meta" / "info.json").exists():
        print(f"no LeRobotDataset at {args.dataset_dir} "
              f"(missing meta/info.json)", file=sys.stderr)
        return 1

    private = not args.public
    print(f"opening {args.dataset_dir} ...")
    ds = LeRobotDataset(repo_id=args.repo_id, root=str(args.dataset_dir))
    print(f"  {len(ds)} frames across "
          f"{ds.meta.total_episodes} episodes, fps={ds.fps}")

    print(f"pushing to {args.repo_id} (private={private}) ...")
    ds.push_to_hub(private=private, license=args.license, push_videos=True)
    visibility = "private" if private else "public"
    print(f"pushed: https://huggingface.co/datasets/{args.repo_id} ({visibility})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
