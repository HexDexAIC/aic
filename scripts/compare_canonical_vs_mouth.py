#!/usr/bin/env python3
"""Visualize CANONICAL GT projection vs MOUTH-plane projection side-by-side.

The canonical port frame (T_settled_TCP @ T_TCP_plug) places the origin at
the plug-tip-at-deepest-insertion — which may sit a few mm INSIDE the cage,
not on the visible mouth plane.

This tool renders 4 candidate `T_canonical_to_mouth` translations along the
port +z axis (out of the port mouth toward the camera) so we can pick the
offset that lands the rectangle on the visible mouth feature.

Output: 4 columns per row, each row = one (ep, distance) frame:
  col 0: +0 mm  (canonical = current behavior)   YELLOW
  col 1: +3 mm  (slight forward offset)          MAGENTA
  col 2: +6 mm  (medium forward offset)          CYAN
  col 3: +9 mm  (strong forward offset)          GREEN
"""
from __future__ import annotations

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
OUT_DIR = Path("/tmp/aic_canon_vs_mouth")
OUT_DIR.mkdir(parents=True, exist_ok=True)
GRID_OUT = Path("/mnt/c/Users/Dell/aic_canon_vs_mouth.jpg")

SLOT_W, SLOT_H = 0.0137, 0.0085

# 4 candidate canonical->mouth translations along port +z (out of port toward cam)
# Each: (label, dz_meters, BGR color)
OFFSETS = [
    ("+0mm canonical",  0.000, (0, 255, 255)),    # YELLOW — current
    ("+3mm",            0.003, (255, 0, 255)),    # MAGENTA
    ("+6mm",            0.006, (255, 255, 0)),    # CYAN
    ("+9mm",            0.009, (0, 255, 0)),      # GREEN
]

# Pick representative frames spanning rails and the train/val/test split
TEST_EPS = [3, 78, 160, 235]   # train / train / val / test
TARGET_DISTS = [0.20, 0.10]    # one mid-distance, one close-to-filter


def project_with_offset(T_base_port_canonical, dz_m, T_base_tcp, K, T_tcp_opt):
    """Project SFP slot rectangle, applying canonical→mouth offset along port +z."""
    T_canonical_to_mouth = np.eye(4)
    T_canonical_to_mouth[2, 3] = dz_m
    T_base_mouth = T_base_port_canonical @ T_canonical_to_mouth
    corners_local = np.array([
        [+SLOT_W/2, +SLOT_H/2, 0, 1],
        [+SLOT_W/2, -SLOT_H/2, 0, 1],
        [-SLOT_W/2, -SLOT_H/2, 0, 1],
        [-SLOT_W/2, +SLOT_H/2, 0, 1],
        [0, 0, 0, 1],
    ]).T
    T_base_opt = T_base_tcp @ T_tcp_opt
    T_opt_mouth = np.linalg.inv(T_base_opt) @ T_base_mouth
    pts = T_opt_mouth @ corners_local
    Z = pts[2]
    if (Z <= 0).any():
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx_p, cy_p = K[0, 2], K[1, 2]
    u = fx * pts[0] / Z + cx_p
    v = fy * pts[1] / Z + cy_p
    return np.stack([u, v], axis=1)


def find_frame_at_dist(states, port_xyz, target_d):
    dists = np.linalg.norm(states[:, 0:3] - port_xyz, axis=1)
    valid = dists >= 0.06
    if not valid.any():
        return None
    err = np.where(valid, np.abs(dists - target_d), np.inf)
    return int(np.argmin(err))


def main():
    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    K_center = K_PER_CAM["center"]
    T_tcp_opt = T_TCP_OPT["center"]

    # Index parquet files
    ep_to_pq = {}
    for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep_val in df["episode_index"].unique():
            ep_int = int(ep_val)
            if ep_int in TEST_EPS and ep_int not in ep_to_pq:
                file_idx = int(pf.stem.replace("file-", ""))
                eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
                ep_to_pq[ep_int] = (file_idx, eg)

    rows = []  # rows of 4-cell side-by-side strips

    for ep in TEST_EPS:
        if ep not in ep_to_pq or str(ep) not in gt_pose:
            continue
        file_idx, eg = ep_to_pq[ep]
        states = np.stack(eg["observation.state"].values)
        frames = eg["frame_index"].to_numpy()

        T_settled = np.eye(4)
        T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
        T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
        T_base_port_canonical = T_settled @ T_TCP_plug
        port_xyz = T_base_port_canonical[:3, 3]

        tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                             columns=["episode_index", "frame_index"])
        df_full = tbl.to_pandas()
        video_path = ROOT / "videos" / "observation.images.center" / "chunk-000" / f"file-{file_idx:03d}.mp4"

        for target_d in TARGET_DISTS:
            idx = find_frame_at_dist(states, port_xyz, target_d)
            if idx is None:
                continue
            fr_idx = int(frames[idx])
            actual_d = float(np.linalg.norm(states[idx, 0:3] - port_xyz))

            row_in_file = df_full[(df_full["episode_index"] == ep) &
                                    (df_full["frame_index"] == fr_idx)].index[0]
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row_in_file))
            ok, img = cap.read()
            cap.release()
            if not ok:
                continue

            T_base_tcp = state_to_T(states[idx])

            # Project all 4 candidates
            projs = []
            for label, dz, color in OFFSETS:
                pts = project_with_offset(T_base_port_canonical, dz, T_base_tcp, K_center, T_tcp_opt)
                projs.append((label, color, pts))

            # Common crop window: tight around the canonical projection
            ref_pts = projs[0][2]
            if ref_pts is None:
                continue
            cx_p, cy_p = ref_pts[:4].mean(axis=0)
            port_pix_w = float(np.linalg.norm(ref_pts[1] - ref_pts[0]))
            half = max(80, int(port_pix_w * 3.0))
            x0 = max(0, int(cx_p - half)); x1 = min(img.shape[1], int(cx_p + half))
            y0 = max(0, int(cy_p - half)); y1 = min(img.shape[0], int(cy_p + half))
            if x1 - x0 < 60 or y1 - y0 < 60:
                continue

            ZOOM = 5
            CANVAS = 480

            # Render one panel per offset, all from the same crop
            panels = []
            for label, color, pts in projs:
                if pts is None:
                    continue
                crop = img[y0:y1, x0:x1].copy()
                crop_big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
                pts_local = (pts - np.array([[x0, y0]])) * ZOOM
                poly = pts_local[:4].astype(np.int32)
                cv2.polylines(crop_big, [poly], True, color, 3)
                ctr = tuple(pts_local[4].astype(int))
                cv2.circle(crop_big, ctr, 6, color, -1)
                # Letterbox to CANVAS
                h_b, w_b = crop_big.shape[:2]
                s = CANVAS / max(h_b, w_b)
                nh, nw = int(h_b * s), int(w_b * s)
                resized = cv2.resize(crop_big, (nw, nh))
                canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
                yo = (CANVAS - nh) // 2; xo = (CANVAS - nw) // 2
                canvas[yo:yo+nh, xo:xo+nw] = resized
                cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.putText(canvas, f"ep{ep:03d} fr{fr_idx:04d} d={actual_d*100:.0f}cm",
                             (10, CANVAS - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panels.append(canvas)

            row_strip = np.hstack(panels)
            rows.append(row_strip)
            row_path = OUT_DIR / f"ep{ep:03d}_d{int(actual_d*100):02d}cm.jpg"
            cv2.imwrite(str(row_path), row_strip)
            user_path = Path(f"/mnt/c/Users/Dell/aic_canonvsmouth_ep{ep:03d}_d{int(actual_d*100):02d}.jpg")
            cv2.imwrite(str(user_path), row_strip)

    if not rows:
        print("No rows produced.")
        return
    grid = np.vstack(rows)
    cv2.imwrite(str(GRID_OUT), grid)
    print(f"Wrote {len(rows)} rows.")
    print(f"Per-frame strips: /mnt/c/Users/Dell/aic_canonvsmouth_*.jpg")
    print(f"Full grid: {GRID_OUT}")


if __name__ == "__main__":
    main()
