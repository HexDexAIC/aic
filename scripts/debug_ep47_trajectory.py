#!/usr/bin/env python3
"""Look at ep47's full TCP trajectory to find the right 'settled' moment."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.home() / "aic_hexdex_sfp300"

for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
    tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "action", "observation.state"])
    df = tbl.to_pandas()
    e47 = df[df["episode_index"] == 47].sort_values("frame_index")
    if len(e47) == 0:
        continue

    n = len(e47)
    states = np.stack(e47["observation.state"].values)  # (N, 27)
    actions = np.stack(e47["action"].values)            # (N, 9)
    frames = e47["frame_index"].to_numpy()

    print(f"ep47 has {n} frames in {pf.name}")
    print(f"\nState xyz min/max:")
    print(f"  x: {states[:,0].min():.4f} .. {states[:,0].max():.4f}")
    print(f"  y: {states[:,1].min():.4f} .. {states[:,1].max():.4f}")
    print(f"  z: {states[:,2].min():.4f} .. {states[:,2].max():.4f}")
    print(f"\nAction target xyz min/max:")
    print(f"  x: {actions[:,0].min():.4f} .. {actions[:,0].max():.4f}")
    print(f"  y: {actions[:,1].min():.4f} .. {actions[:,1].max():.4f}")
    print(f"  z: {actions[:,2].min():.4f} .. {actions[:,2].max():.4f}")

    # Find argmin(state z)
    z_min_idx = int(np.argmin(states[:, 2]))
    print(f"\nargmin(state z) = local idx {z_min_idx}, frame {frames[z_min_idx]}")
    print(f"  state at min-z: x={states[z_min_idx,0]:.4f} y={states[z_min_idx,1]:.4f} z={states[z_min_idx,2]:.4f}")
    print(f"  action at min-z: x={actions[z_min_idx,0]:.4f} y={actions[z_min_idx,1]:.4f} z={actions[z_min_idx,2]:.4f}")

    # Look at trajectory near the end
    print(f"\nLast 20 frames (trajectory tail):")
    print(f"  {'idx':>4} {'fr':>4} {'sx':>8} {'sy':>8} {'sz':>8} {'ax':>8} {'ay':>8} {'az':>8}")
    for i in range(max(0, n - 20), n):
        print(f"  {i:>4} {frames[i]:>4} {states[i,0]:+.4f} {states[i,1]:+.4f} {states[i,2]:+.4f} "
              f"{actions[i,0]:+.4f} {actions[i,1]:+.4f} {actions[i,2]:+.4f}")

    # Find a "stable" window near insertion: where xy of action is constant for >=5 frames
    # and z is at minimum
    z_thresh = states[:, 2].min() + 0.005  # within 5mm of minimum
    insertion_mask = states[:, 2] <= z_thresh
    insertion_idx = np.where(insertion_mask)[0]
    print(f"\nFrames within 5mm of min-z: {len(insertion_idx)}")
    if len(insertion_idx) > 0:
        print(f"  range: idx {insertion_idx[0]}..{insertion_idx[-1]} = frames {frames[insertion_idx[0]]}..{frames[insertion_idx[-1]]}")
        # state xyz over insertion period
        insert_states = states[insertion_idx]
        insert_actions = actions[insertion_idx]
        print(f"  state xyz mean: {insert_states[:, 0:3].mean(axis=0)}")
        print(f"  state xyz std:  {insert_states[:, 0:3].std(axis=0)}")
        print(f"  action xyz mean: {insert_actions[:, 0:3].mean(axis=0)}")

    break
