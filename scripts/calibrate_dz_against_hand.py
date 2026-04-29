#!/usr/bin/env python3
"""Calibrate dz (and optional in-plane offsets) against the 22 hand-label pairs.

For each candidate (dz, dx, dy) along port axes, compute pixel error vs the
filtered hand-annotation set. Find the offset that minimizes median |Δ|.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import (
    K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R,
)

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
ANN_PATH = Path.home() / "aic_audit_annotations.json"
SLOT_W, SLOT_H = 0.0137, 0.0085
KEEP_THRESH_PIX = 80.0


def project_with_offset(T_base_port_canon, T_base_tcp, K, T_tcp_opt, w, h, dx_m, dy_m, dz_m):
    """Apply (dx, dy, dz) offset in PORT FRAME along port axes."""
    T_co = np.eye(4)
    T_co[0, 3] = dx_m; T_co[1, 3] = dy_m; T_co[2, 3] = dz_m
    T_base_mouth = T_base_port_canon @ T_co
    corners = np.array([
        [+w/2, +h/2, 0, 1], [+w/2, -h/2, 0, 1],
        [-w/2, -h/2, 0, 1], [-w/2, +h/2, 0, 1],
        [0, 0, 0, 1],
    ]).T
    T_base_opt = T_base_tcp @ T_tcp_opt
    pts = (np.linalg.inv(T_base_opt) @ T_base_mouth) @ corners
    Z = pts[2]
    if (Z <= 0).any():
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx_p, cy_p = K[0, 2], K[1, 2]
    u = fx * pts[0] / Z + cx_p
    v = fy * pts[1] / Z + cy_p
    return np.stack([u, v], axis=1)


def main():
    ann = json.loads(ANN_PATH.read_text())
    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset_cal = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset_cal["qw"], offset_cal["qx"], offset_cal["qy"], offset_cal["qz"])
    T_TCP_plug[:3, 3] = [offset_cal["x"], offset_cal["y"], offset_cal["z"]]

    needed_eps = set()
    for k, v in ann.items():
        if any(b.get("label") == "target" for b in v.get("boxes", [])):
            needed_eps.add(int(v["episode"]))

    ep_to_data = {}
    for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep_val in df["episode_index"].unique():
            ep_int = int(ep_val)
            if ep_int in needed_eps and ep_int not in ep_to_data:
                file_idx = int(pf.stem.replace("file-", ""))
                eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
                ep_to_data[ep_int] = (file_idx, eg)

    # Build the hand-label evaluation set (filtered for likely-correct-target)
    hand_pairs = []  # list of dicts with raw inputs needed for re-projection
    for k, v in ann.items():
        ep = int(v["episode"]); fr = int(v["frame"]); cam = v["camera"]
        targets = [b for b in v.get("boxes", []) if b.get("label") == "target"]
        if not targets or str(ep) not in gt_pose or ep not in ep_to_data:
            continue
        bbox = targets[0]["bbox_xyxy"]
        user_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

        file_idx, eg = ep_to_data[ep]
        states = np.stack(eg["observation.state"].values)
        frames = eg["frame_index"].to_numpy()
        m = (frames == fr)
        if not m.any():
            continue
        idx = int(np.where(m)[0][0])

        T_settled = np.eye(4)
        T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
        T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
        T_base_port = T_settled @ T_TCP_plug
        T_base_tcp = state_to_T(states[idx])
        K = K_PER_CAM[cam]
        T_tcp_opt = T_TCP_OPT[cam]

        # Filter: drop wrong-port outliers using canonical projection vs user
        canon_proj = project_with_offset(T_base_port, T_base_tcp, K, T_tcp_opt, SLOT_W, SLOT_H, 0, 0, 0)
        if canon_proj is None:
            continue
        canon_c = canon_proj[:4].mean(axis=0)
        if np.linalg.norm(user_c - canon_c) > KEEP_THRESH_PIX:
            continue  # likely wrong-port click

        hand_pairs.append({
            "ep": ep, "fr": fr, "cam": cam,
            "T_base_port": T_base_port, "T_base_tcp": T_base_tcp,
            "K": K, "T_tcp_opt": T_tcp_opt,
            "user_c": user_c,
        })

    print(f"Loaded {len(hand_pairs)} accepted hand-label pairs")

    # Sweep dz only first (along port +z)
    print("\n=== dz-only sweep (port-axis depth) ===")
    print(f"{'dz_mm':>8} {'med|d|':>8} {'med Δcx':>9} {'med Δcy':>9} {'mean|d|':>9}")
    dz_results = []
    for dz_mm in range(-15, 16):
        dz = dz_mm / 1000.0
        deltas = []
        for hp in hand_pairs:
            proj = project_with_offset(hp["T_base_port"], hp["T_base_tcp"],
                                          hp["K"], hp["T_tcp_opt"],
                                          SLOT_W, SLOT_H, 0, 0, dz)
            if proj is None:
                continue
            proj_c = proj[:4].mean(axis=0)
            deltas.append(hp["user_c"] - proj_c)
        deltas = np.array(deltas)
        norms = np.linalg.norm(deltas, axis=1)
        dz_results.append((dz_mm, float(np.median(norms)),
                            float(np.median(deltas[:, 0])),
                            float(np.median(deltas[:, 1])),
                            float(norms.mean())))
        print(f"  {dz_mm:>+5d}mm {dz_results[-1][1]:>7.2f} {dz_results[-1][2]:>+8.2f} "
              f"{dz_results[-1][3]:>+8.2f} {dz_results[-1][4]:>8.2f}")
    best_dz = min(dz_results, key=lambda r: r[1])
    print(f"\n  best dz: {best_dz[0]:+d}mm (median |Δ|={best_dz[1]:.2f}px)")

    # Now full 3-axis search around best dz
    best_dz_m = best_dz[0]
    print(f"\n=== 3-axis sweep around dz={best_dz_m:+d}mm ===")
    print(f"{'dx_mm':>6} {'dy_mm':>6} {'dz_mm':>6} {'med|d|':>8} {'med Δcx':>9} {'med Δcy':>9}")
    grid_results = []
    for dx_mm in range(-6, 7, 2):
        for dy_mm in range(-6, 7, 2):
            for dz_mm in range(best_dz_m - 4, best_dz_m + 5, 2):
                dx = dx_mm / 1000.0; dy = dy_mm / 1000.0; dz = dz_mm / 1000.0
                deltas = []
                for hp in hand_pairs:
                    proj = project_with_offset(hp["T_base_port"], hp["T_base_tcp"],
                                                  hp["K"], hp["T_tcp_opt"],
                                                  SLOT_W, SLOT_H, dx, dy, dz)
                    if proj is None:
                        continue
                    proj_c = proj[:4].mean(axis=0)
                    deltas.append(hp["user_c"] - proj_c)
                deltas = np.array(deltas)
                norms = np.linalg.norm(deltas, axis=1)
                grid_results.append((dx_mm, dy_mm, dz_mm,
                                      float(np.median(norms)),
                                      float(np.median(deltas[:, 0])),
                                      float(np.median(deltas[:, 1]))))
    grid_results.sort(key=lambda r: r[3])
    print(f"\nTop-10 from full 3-axis sweep (sorted by median |Δ|):")
    for i, r in enumerate(grid_results[:10]):
        marker = "★" if i == 0 else " "
        print(f"  {i+1}{marker} dx={r[0]:+3d} dy={r[1]:+3d} dz={r[2]:+3d}mm  "
              f"med|d|={r[3]:6.2f}  Δcx={r[4]:+6.2f}  Δcy={r[5]:+6.2f}")

    best = grid_results[0]
    print(f"\n=== RECOMMENDATION ===")
    print(f"  T_canonical_to_visible_mouth: dx={best[0]}mm dy={best[1]}mm dz={best[2]}mm")
    print(f"  Median pixel error: {best[3]:.2f}px")
    print(f"  Residual cx bias: {best[4]:+.2f}px")
    print(f"  Residual cy bias: {best[5]:+.2f}px")
    print(f"  → from baseline (canonical, identity): {dz_results[15][1]:.2f}px median error")
    print(f"  → after calibration: {best[3]:.2f}px median error")
    improvement = dz_results[15][1] / best[3] if best[3] > 0 else float('inf')
    print(f"  → {improvement:.1f}x improvement")


if __name__ == "__main__":
    main()
