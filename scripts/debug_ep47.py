#!/usr/bin/env python3
"""Debug GT projection for ep47 vs user annotations."""
import json
from pathlib import Path

import numpy as np

gt_pose = json.loads((Path.home() / "aic_gt_port_poses.json").read_text())
gt_2d = json.loads((Path.home() / "aic_gt_port_2d.json").read_text())
ann = json.loads((Path.home() / "aic_audit_annotations.json").read_text())

# Episode 47 in pose file?
e47 = gt_pose.get("47")
print(f"ep47 in pose file: {e47 is not None}")
if e47:
    print(f"  n_frames: {e47['n_frames']}")
    print(f"  actual_tcp_xyz: {e47['actual_tcp_xyz']}")
    print(f"  settle_std_xyz_m: {e47['settle_std_xyz_m']}")

# Episode 47 fr 0 projections
e47_dict = gt_2d.get("ep_47", {})
print(f"\nep47 has {len(e47_dict)} frames in projection")
for k in ["fr_0", "fr_50", "fr_100", "fr_150", "fr_200"]:
    v = e47_dict.get(k)
    print(f"  {k}: {v}")

# User annotations for ep47
print("\nUser annotations for ep47:")
for k, v in ann.items():
    if v.get("episode") == 47:
        for b in v.get("boxes", []):
            label = b.get("label")
            box = b.get("bbox_xyxy")
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            print(f"  {k} {label}: cx={cx:.0f} cy={cy:.0f} bbox={box}")

# Compare success status
import pyarrow.parquet as pq
ROOT = Path.home() / "aic_hexdex_sfp300"
for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
    tbl = pq.read_table(pf, columns=["episode_index", "episode_success", "frame_index"])
    df = tbl.to_pandas()
    e47_df = df[df["episode_index"] == 47]
    if len(e47_df):
        print(f"\nep47 success: {float(e47_df['episode_success'].iloc[0])}")
        print(f"  total frames: {len(e47_df)}")
        break
