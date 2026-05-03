#!/usr/bin/env python3
"""Derive ground-truth port pose per episode from action commands.

Idea: CheatCodeMJ reads port pose from /scoring/tf and commands TCP toward
port + plug_offset. At successful insertion (last frames), the TCP target
pose CONVERGES to port_pose + R(port) @ plug_offset.

For each successful episode:
  1. Take the last N frames where action.pos_z is nearly constant (settled).
  2. Median those poses → settled TCP pose.
  3. Subtract the known TCP-to-plug offset → port pose in base_link.

This gives us GT port pose (xyz + 6D rot) for ~286 episodes for free,
without needing the raw bags.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.home() / "aic_hexdex_sfp300"


def rot6_to_R(rot6):
    """Recover 3x3 rotation from 6-D continuous rep (first 2 cols, Gram-Schmidt)."""
    a1 = rot6[..., 0:3]
    a2 = rot6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    dot = (b1 * a2).sum(axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def main():
    pqs = sorted((ROOT / "data" / "chunk-000").glob("*.parquet"))
    print(f"Found {len(pqs)} parquet files")

    all_rows = []
    for pf in pqs:
        tbl = pq.read_table(pf, columns=[
            "episode_index", "frame_index", "timestamp",
            "action", "observation.state", "episode_success"
        ])
        df = tbl.to_pandas()
        all_rows.append(df)
    df = pd.concat(all_rows, ignore_index=True)
    print(f"Total frames: {len(df)}")
    print(f"Episodes: {df['episode_index'].nunique()}")

    out = {}
    by_ep = df.groupby("episode_index")
    for ep, g in by_ep:
        g = g.sort_values("frame_index")
        success = float(g["episode_success"].iloc[-1])
        if success < 0.5:
            continue
        n = len(g)
        # Find deepest-insertion frame (min z in TCP state). At that moment
        # the plug is fully seated in the port → TCP pose ≈ port pose + plug_offset.
        states = np.stack(g["observation.state"].values)  # (N, 27)
        actions = np.stack(g["action"].values)            # (N, 9)
        z_min_idx = int(np.argmin(states[:, 2]))
        # Average a small window around the deepest moment for noise resilience.
        lo = max(0, z_min_idx - 2)
        hi = min(n, z_min_idx + 3)
        actual_xyz = np.median(states[lo:hi, 0:3], axis=0)
        actual_rot6 = np.median(states[lo:hi, 3:9], axis=0)
        Ra = rot6_to_R(actual_rot6)
        tcp_xyz = np.median(actions[lo:hi, 0:3], axis=0)
        tcp_rot6 = np.median(actions[lo:hi, 3:9], axis=0)
        R = rot6_to_R(tcp_rot6)
        # Stability around the seated moment:
        action_std = actions[lo:hi, 0:3].std(axis=0)
        tail_n = hi - lo
        out[int(ep)] = {
            "n_frames": int(n),
            "tail_n": int(tail_n),
            "action_target_xyz": tcp_xyz.tolist(),
            "action_target_R": R.tolist(),
            "actual_tcp_xyz": actual_xyz.tolist(),
            "actual_tcp_R": Ra.tolist(),
            "settle_std_xyz_m": action_std.tolist(),
        }

    print(f"\nDerived from {len(out)} successful episodes")
    # Sanity check spread
    xs = np.array([v["actual_tcp_xyz"] for v in out.values()])
    print(f"Settled TCP xyz range:")
    print(f"  x: {xs[:,0].min():.4f} .. {xs[:,0].max():.4f}  (mean {xs[:,0].mean():.4f})")
    print(f"  y: {xs[:,1].min():.4f} .. {xs[:,1].max():.4f}  (mean {xs[:,1].mean():.4f})")
    print(f"  z: {xs[:,2].min():.4f} .. {xs[:,2].max():.4f}  (mean {xs[:,2].mean():.4f})")

    settle_stds = np.array([v["settle_std_xyz_m"] for v in out.values()])
    print(f"\nAction-target settle stds (m):")
    print(f"  median: {np.median(settle_stds, axis=0)}")
    print(f"  max:    {np.max(settle_stds, axis=0)}")

    # Inter-episode variation = port-pose variation across DR knobs
    # (board_x, board_y, board_yaw, nic_rail, etc.)
    out_path = Path.home() / "aic_gt_port_poses.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved per-episode settled TCP poses to: {out_path}")
    print("Subtract known TCP→plug offset to get port pose in base_link.")


if __name__ == "__main__":
    main()
