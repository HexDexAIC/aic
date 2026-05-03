#!/usr/bin/env python3
"""For each successful episode, compare action_target vs actual_state at
deepest insertion. The difference reveals control lag / partial-seating.

If action and state are consistently close (<2 mm), we can use either.
If they diverge, action_target is the more reliable port-pose source
(it's what the policy COMMANDED, with integrator-based XY correction
already baked in — i.e., the policy's best belief of port pose).
"""
import json
from pathlib import Path

import numpy as np

gt = json.loads((Path.home() / "aic_gt_port_poses.json").read_text())

diffs_xyz = []
for ep_str, info in gt.items():
    a = np.array(info["action_target_xyz"])
    s = np.array(info["actual_tcp_xyz"])
    diffs_xyz.append((int(ep_str), s - a))

diffs_xyz.sort()
arr = np.array([d for _, d in diffs_xyz])  # (N, 3)
print(f"State - Action diff (mm) over {len(arr)} successful eps:")
print(f"  x: mean {arr[:, 0].mean()*1000:+.1f}  std {arr[:, 0].std()*1000:.1f}  max {np.abs(arr[:, 0]).max()*1000:.1f}")
print(f"  y: mean {arr[:, 1].mean()*1000:+.1f}  std {arr[:, 1].std()*1000:.1f}  max {np.abs(arr[:, 1]).max()*1000:.1f}")
print(f"  z: mean {arr[:, 2].mean()*1000:+.1f}  std {arr[:, 2].std()*1000:.1f}  max {np.abs(arr[:, 2]).max()*1000:.1f}")

print(f"\nWorst 10 episodes by |diff|_max:")
worst = sorted(diffs_xyz, key=lambda x: np.abs(x[1]).max(), reverse=True)[:10]
for ep, d in worst:
    print(f"  ep{ep:3d}: dx={d[0]*1000:+6.1f}mm dy={d[1]*1000:+6.1f}mm dz={d[2]*1000:+6.1f}mm")

# Check ep47 specifically
ep47_d = next((d for ep, d in diffs_xyz if ep == 47), None)
if ep47_d is not None:
    print(f"\nep47 specifically: dx={ep47_d[0]*1000:+.2f}mm dy={ep47_d[1]*1000:+.2f}mm dz={ep47_d[2]*1000:+.2f}mm")
