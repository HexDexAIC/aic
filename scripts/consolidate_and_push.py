#!/usr/bin/env python3
"""Consolidate per-seed datasets from a spawn-sweep into one
LeRobotDataset and (optionally) push to Hugging Face — private by default.

Reads <sweep_dir>/seeds/seed_NN/dataset/ for every seed, creates a fresh
combined LeRobotDataset under <sweep_dir>/combined/ (or --out), then
pushes it to HF via LeRobotDataset.push_to_hub.

The combined dataset has one episode per source seed; episodes are
appended in seed order. Videos are re-encoded as part of save_episode
(LeRobot v3.0 stores one MP4 per chunk-file, not per-episode, so a
file-level merge is more pain than a clean re-write).

Auth (one-time, via the pixi env's HF CLI):
    pixi run --manifest-path src/aic/pixi.toml hf auth login

Usage (pixi env — has lerobot):
    pixi run --manifest-path src/aic/pixi.toml \\
        python src/aic/scripts/consolidate_and_push.py <sweep_dir> \\
            --repo-id YOUR_USER/aic-sfp-1000 [--public] [--no-push]

Examples:
    # Consolidate only, skip the push (smoke-test the merge first):
    ... <sweep_dir> --repo-id local/aic-sfp-test --no-push

    # Consolidate + push as a private repo (default):
    ... <sweep_dir> --repo-id YOUR_USER/aic-sfp-1000

    # Push public:
    ... <sweep_dir> --repo-id YOUR_USER/aic-sfp-public --public
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def chw_float_to_hwc_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a CHW float32 [0,1] image tensor to HWC uint8 numpy.

    LeRobotDataset.add_frame expects HWC uint8 for video features (the
    same shape image_msg_to_array produces in record_lerobot.py).
    """
    arr = (t.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return np.ascontiguousarray(arr)


def consolidate(sweep_dir: Path, out_dir: Path, repo_id: str) -> LeRobotDataset:
    seeds = sorted((sweep_dir / "seeds").glob("seed_*"))
    src_paths = [s / "dataset" for s in seeds
                 if (s / "dataset" / "meta" / "info.json").exists()]
    if not src_paths:
        raise SystemExit(f"no datasets found under {sweep_dir}/seeds/")
    print(f"merging {len(src_paths)} per-seed datasets → {out_dir}")

    info = json.loads((src_paths[0] / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    raw_features = info["features"]
    # LeRobot adds these columns automatically; passing them in the schema
    # makes add_frame complain that the frame is missing them.
    INTERNAL = {"episode_index", "frame_index", "index", "task_index",
                "timestamp"}
    features = {k: v for k, v in raw_features.items() if k not in INTERNAL}
    # JSON deserializes shape tuples as lists; LeRobot's frame validator does
    # strict equality so we restore them to tuples to match the per-frame
    # numpy arrays we'll feed in.
    for fname, feat in features.items():
        if isinstance(feat.get("shape"), list):
            feat["shape"] = tuple(feat["shape"])
    image_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    if out_dir.exists():
        raise SystemExit(
            f"output dir already exists: {out_dir}\n"
            f"delete it or pass --out to write somewhere else."
        )

    combined = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=str(out_dir),
        use_videos=True,
    )

    t0 = time.time()
    for i, ds_path in enumerate(src_paths):
        seed_num = int(ds_path.parent.name.split("_")[1])
        src = LeRobotDataset(repo_id=f"local/seed_{seed_num:02d}",
                             root=str(ds_path))
        n = len(src)
        ep_t0 = time.time()
        for fi in range(n):
            sample = src[fi]
            frame = {"task": sample.get("task", "insert sfp cable")}
            for key, feat in features.items():
                if key not in sample:
                    continue
                if feat.get("dtype") == "video":
                    frame[key] = chw_float_to_hwc_uint8(sample[key])
                else:
                    val = sample[key]
                    if isinstance(val, torch.Tensor):
                        val = val.numpy() if val.dim() > 0 else val.item()
                    frame[key] = val
            combined.add_frame(frame)
        combined.save_episode()
        elapsed = time.time() - ep_t0
        total = time.time() - t0
        remaining = (len(src_paths) - i - 1) * (total / (i + 1))
        print(f"  [{i+1}/{len(src_paths)}] seed_{seed_num:02d}: "
              f"{n} frames in {elapsed:.1f}s "
              f"(eta {remaining/60:.1f} min)", flush=True)

    print(f"done. consolidated {len(src_paths)} episodes in {(time.time()-t0)/60:.1f} min")
    return combined


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--repo-id", required=True,
                    help="HF repo id, e.g. YOUR_USER/aic-sfp-1000")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output dir (default: <sweep>/combined).")
    ap.add_argument("--public", action="store_true",
                    help="Push as a PUBLIC HF repo (default: private).")
    ap.add_argument("--no-push", action="store_true",
                    help="Just consolidate locally, skip the HF push.")
    ap.add_argument("--license", default="apache-2.0",
                    help="HF dataset card license (default: apache-2.0).")
    args = ap.parse_args()

    sweep_dir: Path = args.sweep_dir
    out_dir: Path = args.out or (sweep_dir / "combined")

    combined = consolidate(sweep_dir, out_dir, args.repo_id)

    if args.no_push:
        print(f"\n--no-push set; skipping HF upload. Combined dataset at {out_dir}")
        return 0

    private = not args.public
    print(f"\npushing {args.repo_id} (private={private}) ...")
    combined.push_to_hub(private=private, license=args.license, push_videos=True)
    visibility = "private" if private else "public"
    print(f"pushed: https://huggingface.co/datasets/{args.repo_id} ({visibility})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
