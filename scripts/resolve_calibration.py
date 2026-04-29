#!/usr/bin/env python3
"""Re-solve calibration from saved user clicks with proper bounds.

The previous solver let w_scale/h_scale go to 0, collapsing the projected
rectangle to a point. This version uses bounded optimization and also
reports a fixed-size variant (translation only) for comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import (
    K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R,
)

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
CLICKS_PATH = Path.home() / "aic_calib_clicks.json"
OUT_PATH = Path.home() / "aic_visible_mouth_calib.json"

SLOT_W, SLOT_H = 0.0137, 0.0085


def project_rect(T_base_port, T_co, T_base_tcp, K, T_tcp_opt, w, h):
    T_base_mouth = T_base_port @ T_co
    corners = np.array([
        [+w/2, +h/2, 0, 1], [+w/2, -h/2, 0, 1],
        [-w/2, -h/2, 0, 1], [-w/2, +h/2, 0, 1],
    ]).T
    T_opt = np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T_base_mouth
    pts = T_opt @ corners
    Z = pts[2]
    if (Z <= 0).any():
        return None
    return np.stack([K[0, 0] * pts[0] / Z + K[0, 2],
                     K[1, 1] * pts[1] / Z + K[1, 2]], axis=1)


def matched_corner_err(proj, user_clicks):
    """Order-invariant per-corner error via bipartite matching.
    user_clicks: (N, 2) array of N ≤ 4 visible corner clicks.
    Returns N-element array of distances (matched proj→user pairing).
    """
    from scipy.optimize import linear_sum_assignment
    user = np.atleast_2d(user_clicks)
    cost = np.linalg.norm(proj[:, None, :] - user[None, :, :], axis=2)  # (4, N)
    rr, cc = linear_sum_assignment(cost)
    return cost[rr, cc]


def main():
    clicks_data = json.loads(CLICKS_PATH.read_text())
    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    needed_eps = set(c["ep"] for c in clicks_data)
    ep_to_pq = {}
    for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep_val in df["episode_index"].unique():
            ep_int = int(ep_val)
            if ep_int in needed_eps and ep_int not in ep_to_pq:
                file_idx = int(pf.stem.replace("file-", ""))
                eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
                ep_to_pq[ep_int] = (file_idx, eg)

    pairs = []
    for cd in clicks_data:
        ep = cd["ep"]; fr = cd["fr"]
        # Accept partial clicks (≥2 visible corners). Fewer than 2 is ambiguous.
        visible_clicks = [c for c in cd["clicks"] if c is not None]
        if len(visible_clicks) < 2:
            continue
        clicks = np.array(visible_clicks)  # (N, 2) where 2 ≤ N ≤ 4
        if ep not in ep_to_pq:
            continue
        _, eg = ep_to_pq[ep]
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
        pairs.append({
            "ep": ep, "fr": fr,
            "T_base_port": T_base_port, "T_base_tcp": T_base_tcp,
            "K": K_PER_CAM["center"], "T_tcp_opt": T_TCP_OPT["center"],
            "user_corners": clicks,  # (4, 2)
        })

    print(f"Loaded {len(pairs)} fully-clicked frames")

    # === Variant 1: fixed size (spec 13.7×8.5), solve translation only ===
    print("\n=== Variant 1: fixed size 13.7×8.5mm, optimize (dx, dy, dz) only ===")

    def loss_fixed(params):
        dx, dy, dz = params
        T_co = np.eye(4); T_co[:3, 3] = [dx, dy, dz]
        total = 0.0
        for p in pairs:
            proj = project_rect(p["T_base_port"], T_co, p["T_base_tcp"],
                                  p["K"], p["T_tcp_opt"], SLOT_W, SLOT_H)
            if proj is None:
                return 1e6
            err_per_corner = matched_corner_err(proj, p["user_corners"])
            total += float(np.sum(err_per_corner ** 2))
        return total

    res1 = minimize(loss_fixed, [0, 0, 0], method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-3, "maxiter": 5000})
    dx1, dy1, dz1 = res1.x

    # Per-frame errors
    T_co1 = np.eye(4); T_co1[:3, 3] = [dx1, dy1, dz1]
    per_frame_v1 = []
    for p in pairs:
        proj = project_rect(p["T_base_port"], T_co1, p["T_base_tcp"],
                              p["K"], p["T_tcp_opt"], SLOT_W, SLOT_H)
        err = matched_corner_err(proj, p["user_corners"])
        per_frame_v1.append({"ep": p["ep"], "med": float(np.median(err)),
                              "max": float(err.max()), "mean": float(err.mean())})
    overall_v1 = float(np.median([f["med"] for f in per_frame_v1]))
    print(f"  Optimal translation: dx={dx1*1000:+.2f} dy={dy1*1000:+.2f} dz={dz1*1000:+.2f} mm")
    print(f"  Per-frame median corner errors:")
    for f in per_frame_v1:
        print(f"    ep{f['ep']:3d}: median={f['med']:.2f}px  max={f['max']:.2f}  mean={f['mean']:.2f}")
    print(f"  Overall median: {overall_v1:.2f}px")

    # === Variant 2: bounded scales ===
    print("\n=== Variant 2: optimize (dx, dy, dz, w_scale, h_scale) with bounds ===")

    def loss_full(params):
        dx, dy, dz, ws, hs = params
        if ws < 0.4 or ws > 2.0 or hs < 0.4 or hs > 2.0:
            return 1e6
        T_co = np.eye(4); T_co[:3, 3] = [dx, dy, dz]
        total = 0.0
        for p in pairs:
            proj = project_rect(p["T_base_port"], T_co, p["T_base_tcp"],
                                  p["K"], p["T_tcp_opt"], SLOT_W*ws, SLOT_H*hs)
            if proj is None:
                return 1e6
            err_per_corner = matched_corner_err(proj, p["user_corners"])
            total += float(np.sum(err_per_corner ** 2))
        return total

    from scipy.optimize import differential_evolution
    bounds = [(-0.03, 0.03), (-0.03, 0.03), (-0.03, 0.03), (0.25, 2.0), (0.25, 2.0)]
    res2 = differential_evolution(loss_full, bounds, seed=0, maxiter=400, tol=1e-8)
    dx2, dy2, dz2, ws2, hs2 = res2.x
    final_w = SLOT_W * ws2; final_h = SLOT_H * hs2

    T_co2 = np.eye(4); T_co2[:3, 3] = [dx2, dy2, dz2]
    per_frame_v2 = []
    for p in pairs:
        proj = project_rect(p["T_base_port"], T_co2, p["T_base_tcp"],
                              p["K"], p["T_tcp_opt"], final_w, final_h)
        err = matched_corner_err(proj, p["user_corners"])
        per_frame_v2.append({"ep": p["ep"], "med": float(np.median(err)),
                              "max": float(err.max()), "mean": float(err.mean())})
    overall_v2 = float(np.median([f["med"] for f in per_frame_v2]))
    print(f"  Optimal: dx={dx2*1000:+.2f} dy={dy2*1000:+.2f} dz={dz2*1000:+.2f} mm")
    print(f"  Final rectangle: {final_w*1000:.2f}mm × {final_h*1000:.2f}mm "
           f"(scales: {ws2:.3f}, {hs2:.3f})")
    print(f"  Per-frame median corner errors:")
    for f in per_frame_v2:
        print(f"    ep{f['ep']:3d}: median={f['med']:.2f}px  max={f['max']:.2f}  mean={f['mean']:.2f}")
    print(f"  Overall median: {overall_v2:.2f}px")

    # === Pick winner: lower error, with reasonable size ===
    print(f"\n=== Comparison ===")
    print(f"  V1 (fixed 13.7×8.5):  overall median {overall_v1:.2f}px")
    print(f"  V2 (free w,h):        overall median {overall_v2:.2f}px  size {final_w*1000:.2f}×{final_h*1000:.2f}mm")

    # Recommend V2 if size is reasonable AND it's at least 1px better
    if 0.6 < ws2 < 1.5 and 0.6 < hs2 < 1.5 and overall_v2 < overall_v1 - 0.5:
        winner = "V2"
        result = {
            "n_frames": len(pairs),
            "T_canonical_to_visible_mouth": {
                "dx_mm": float(dx2 * 1000), "dy_mm": float(dy2 * 1000), "dz_mm": float(dz2 * 1000),
            },
            "rectangle": {"width_mm": float(final_w * 1000), "height_mm": float(final_h * 1000)},
            "median_corner_err_px": overall_v2,
            "per_frame": per_frame_v2,
            "variant": "free w,h (bounded)",
        }
    else:
        winner = "V1"
        result = {
            "n_frames": len(pairs),
            "T_canonical_to_visible_mouth": {
                "dx_mm": float(dx1 * 1000), "dy_mm": float(dy1 * 1000), "dz_mm": float(dz1 * 1000),
            },
            "rectangle": {"width_mm": float(SLOT_W * 1000), "height_mm": float(SLOT_H * 1000)},
            "median_corner_err_px": overall_v1,
            "per_frame": per_frame_v1,
            "variant": "fixed spec size 13.7×8.5",
        }
    print(f"\n  → Winner: {winner}")
    print(f"  Saving: {OUT_PATH}")
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
