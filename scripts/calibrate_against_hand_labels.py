#!/usr/bin/env python3
"""Calibrate T_canonical_to_visible_mouth against the user's 21 hand annotations.

The user already manually clicked target bboxes on 21 cam-frames. Use those as
ground truth for visible-mouth pixel location, since they reflect human judgment
about where the actual port is.

Method:
  1. For each annotation, project canonical port pose at same (ep, fr, cam).
  2. Compare canonical-projection center to user-click center.
  3. Filter out outliers (likely wrong-port clicks like ep47 distractor confusion):
     keep only annotations where canonical center is within 80px of user click
     (i.e., on the same visible region — same port).
  4. From the filtered set, compute systematic pixel offset.
  5. Report: if median offset < 5px, canonical ≈ mouth (T_co_mouth = identity).
     Otherwise compute the corresponding world-space offset.

This is FAR more reliable than auto-detection because the labels are human-quality.
"""
import json
import sys
from pathlib import Path

import cv2
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


def project(T_base_port, T_base_tcp, K, T_tcp_opt, w, h, dz=0.0):
    T_co = np.eye(4); T_co[2, 3] = dz
    T_base_mouth = T_base_port @ T_co
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
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    # Index parquet files for ep_idx -> (file_idx, df)
    needed_eps = set()
    for k, v in ann.items():
        if any(b.get("label") == "target" for b in v.get("boxes", [])):
            needed_eps.add(int(v["episode"]))
    print(f"Episodes referenced in target annotations: {sorted(needed_eps)}")

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

    pairs = []  # list of (ep, fr, cam, user_center_xy, canon_center_xy, dist)
    for k, v in ann.items():
        ep = int(v["episode"]); fr = int(v["frame"]); cam = v["camera"]
        targets = [b for b in v.get("boxes", []) if b.get("label") == "target"]
        if not targets or str(ep) not in gt_pose or ep not in ep_to_data:
            continue
        # Use first target click
        bbox = targets[0]["bbox_xyxy"]
        user_cx = (bbox[0] + bbox[2]) / 2
        user_cy = (bbox[1] + bbox[3]) / 2
        user_w = bbox[2] - bbox[0]
        user_h = bbox[3] - bbox[1]

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
        proj = project(T_base_port, T_base_tcp, K, T_tcp_opt, SLOT_W, SLOT_H, dz=0.0)
        if proj is None:
            continue
        canon_cx, canon_cy = proj[:4].mean(axis=0)
        canon_w = float(np.linalg.norm(proj[1] - proj[0]))
        canon_h = float(np.linalg.norm(proj[2] - proj[1]))

        dist_to_user = float(np.hypot(user_cx - canon_cx, user_cy - canon_cy))
        tcp_to_port = float(np.linalg.norm(states[idx, 0:3] - T_base_port[:3, 3]))

        pairs.append({
            "ep": ep, "fr": fr, "cam": cam, "tcp_to_port": tcp_to_port,
            "user": (user_cx, user_cy, user_w, user_h),
            "canon": (canon_cx, canon_cy, canon_w, canon_h),
            "dist": dist_to_user,
        })

    pairs.sort(key=lambda p: p["dist"])
    print(f"\nLoaded {len(pairs)} target annotations with valid GT projection\n")
    print(f"{'ep':>4} {'fr':>4} {'cam':>6} {'tcp-port':>9} {'user_cx,cy':>14} "
          f"{'canon_cx,cy':>14} {'Δcx':>5} {'Δcy':>5} {'|d|':>5} {'flag':>10}")
    KEEP_THRESH = 80.0
    accepted = []
    for p in pairs:
        ucx, ucy, uw, uh = p["user"]
        ccx, ccy, cw, ch = p["canon"]
        dcx = ucx - ccx; dcy = ucy - ccy
        flag = "OK" if p["dist"] < KEEP_THRESH else "WRONG-PORT?"
        if p["dist"] < KEEP_THRESH:
            accepted.append((dcx, dcy, ucx, ucy, ccx, ccy, p))
        print(f"{p['ep']:>4} {p['fr']:>4} {p['cam']:>6} {p['tcp_to_port']*100:>7.1f}cm "
              f"{ucx:>6.0f},{ucy:>5.0f} {ccx:>6.0f},{ccy:>5.0f} "
              f"{dcx:>+5.0f} {dcy:>+5.0f} {p['dist']:>5.1f} {flag:>10}")

    if not accepted:
        print("\nNo annotations within keep threshold!")
        return

    print(f"\nAccepted (likely target, not distractor): {len(accepted)} / {len(pairs)}")
    arr = np.array([(a[0], a[1]) for a in accepted])
    print(f"\nPixel offset (user - canonical), accepted set:")
    print(f"  Δcx: median {np.median(arr[:,0]):+.2f}px  mean {arr[:,0].mean():+.2f}  std {arr[:,0].std():.2f}")
    print(f"  Δcy: median {np.median(arr[:,1]):+.2f}px  mean {arr[:,1].mean():+.2f}  std {arr[:,1].std():.2f}")
    print(f"  |Δ|: median {np.median(np.linalg.norm(arr, axis=1)):.2f}px  "
          f"mean {np.linalg.norm(arr, axis=1).mean():.2f}px")

    print(f"\nInterpretation:")
    median_offset = float(np.linalg.norm(np.median(arr, axis=0)))
    if median_offset < 3:
        print(f"  Median offset {median_offset:.2f}px is within click noise.")
        print(f"  → Canonical IS visible-mouth plane. T_canonical_to_visible_mouth = identity.")
    elif median_offset < 8:
        print(f"  Small systematic offset ({median_offset:.2f}px).")
        print(f"  → Canonical is close but slightly biased. Could refine, or accept.")
    else:
        print(f"  LARGE offset ({median_offset:.2f}px).")
        print(f"  → Canonical is NOT at visible mouth. Calibration needed.")


if __name__ == "__main__":
    main()
