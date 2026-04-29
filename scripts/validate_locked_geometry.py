#!/usr/bin/env python3
"""Extended validation of the locked SFP geometry (13.7 x 8.5 mm spec slot).

Renders the projected port mouth on N sampled pre-insertion frames spanning
all 5 rails (by sampling 8 episodes evenly), at 3 distances each (far/mid/near).

Before bulk-exporting 17k labels, we check:
  - does the yellow rectangle land on the visible port mouth feature?
  - does it stay aligned across rails (board_yaw and nic_yaw variation)?
  - does it stay aligned across distances?
  - is the canonical port frame == visible mouth plane (no extra translation needed)?

Output:
  /tmp/aic_validate/ep{E}_fr{F}_d{D}.jpg  — individual high-zoom crops
  /mnt/c/Users/Dell/aic_validate_grid.jpg — 8-row x 3-col stitched grid
  Individual files copied to /mnt/c/Users/Dell/aic_validate_*.jpg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import (
    K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R,
)

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
OUT_DIR = Path("/tmp/aic_validate")
OUT_DIR.mkdir(parents=True, exist_ok=True)
GRID_OUT = Path("/mnt/c/Users/Dell/aic_validate_grid.jpg")

# Locked SFP slot geometry (port-frame xy, metres)
SLOT_W, SLOT_H = 0.0137, 0.0085

# Sample episodes spanning 0-299 to hit all 5 rails
TEST_EPS = [3, 41, 78, 122, 160, 197, 235, 272]
# 3 distances per episode: far (~20cm) / mid (~15cm) / near (~10cm, just inside the 6cm filter)
TARGET_DISTS = [0.20, 0.15, 0.10]


def project_slot(T_base_port, T_base_tcp, K, T_tcp_opt, w, h):
    corners_local = np.array([
        [+w/2, +h/2, 0, 1],   # +X+Y  (canonical order from locked plan)
        [+w/2, -h/2, 0, 1],   # +X-Y
        [-w/2, -h/2, 0, 1],   # -X-Y
        [-w/2, +h/2, 0, 1],   # -X+Y
        [0, 0, 0, 1],         # center
    ]).T
    T_base_opt = T_base_tcp @ T_tcp_opt
    T_opt_port = np.linalg.inv(T_base_opt) @ T_base_port
    pts = T_opt_port @ corners_local
    Z = pts[2]
    if (Z <= 0).any():
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx_p, cy_p = K[0, 2], K[1, 2]
    u = fx * pts[0] / Z + cx_p
    v = fy * pts[1] / Z + cy_p
    return np.stack([u, v], axis=1)  # (5, 2): 4 corners + center


def find_frame_at_dist(states, port_xyz, target_d):
    """Find the frame whose TCP-to-port distance is closest to target_d
    AND TCP-to-port distance is at least 6cm (insertion-phase filter)."""
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

    # Index episodes -> (file_idx, dataframe rows)
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

    crops_by_ep = {}  # ep -> [crop_far, crop_mid, crop_near]

    for ep in TEST_EPS:
        if ep not in ep_to_pq or str(ep) not in gt_pose:
            print(f"ep{ep}: no GT or no parquet match, skipping")
            continue
        file_idx, eg = ep_to_pq[ep]
        states = np.stack(eg["observation.state"].values)
        frames = eg["frame_index"].to_numpy()
        n = len(frames)

        # Recover T_base_port for this ep
        T_settled = np.eye(4)
        T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
        T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
        T_base_port = T_settled @ T_TCP_plug
        port_xyz = T_base_port[:3, 3]

        # Re-load this file's parquet to get full row indexing
        tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                             columns=["episode_index", "frame_index"])
        df_full = tbl.to_pandas()

        ep_crops = []
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
            pts = project_slot(T_base_port, T_base_tcp, K_center, T_tcp_opt, SLOT_W, SLOT_H)
            if pts is None:
                continue

            # Crop tightly around projected port — 4x port width
            corners = pts[:4]
            cx_p, cy_p = corners.mean(axis=0)
            port_pix_w = float(np.linalg.norm(corners[1] - corners[0]))
            half = max(80, int(port_pix_w * 3.0))
            x0 = max(0, int(cx_p - half)); x1 = min(img.shape[1], int(cx_p + half))
            y0 = max(0, int(cy_p - half)); y1 = min(img.shape[0], int(cy_p + half))
            if x1 - x0 < 60 or y1 - y0 < 60:
                continue

            crop = img[y0:y1, x0:x1].copy()

            # Upscale 5x with NEAREST so 1-px alignment is visible
            ZOOM = 5
            crop_big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

            # Draw the projected polygon + center + 4 corners as numbered dots
            pts_local = (pts - np.array([[x0, y0]])) * ZOOM
            poly = pts_local[:4].astype(np.int32)
            cv2.polylines(crop_big, [poly], True, (0, 255, 255), 2)  # yellow rectangle
            for i in range(4):
                p = tuple(pts_local[i].astype(int))
                cv2.circle(crop_big, p, 6, (0, 255, 255), -1)
                cv2.putText(crop_big, str(i), (p[0] + 8, p[1] - 8),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            ctr = tuple(pts_local[4].astype(int))
            cv2.circle(crop_big, ctr, 6, (0, 255, 0), -1)

            # Pad to fixed canvas (640x640) for clean grid
            CANVAS = 640
            h_b, w_b = crop_big.shape[:2]
            scale = CANVAS / max(h_b, w_b)
            new_h, new_w = int(h_b * scale), int(w_b * scale)
            crop_resized = cv2.resize(crop_big, (new_w, new_h))
            canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
            yo = (CANVAS - new_h) // 2; xo = (CANVAS - new_w) // 2
            canvas[yo:yo+new_h, xo:xo+new_w] = crop_resized
            crop = canvas
            cv2.putText(crop, f"ep{ep:03d} fr{fr_idx:04d} d={actual_d*100:.1f}cm",
                         (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            ind_path = OUT_DIR / f"ep{ep:03d}_fr{fr_idx:04d}_d{int(actual_d*100):02d}cm.jpg"
            cv2.imwrite(str(ind_path), crop)
            ep_crops.append(crop)

        if ep_crops:
            crops_by_ep[ep] = ep_crops
            # Also copy this ep's row to user-visible path
            row_img = np.hstack(ep_crops) if len(ep_crops) > 1 else ep_crops[0]
            user_path = Path(f"/mnt/c/Users/Dell/aic_validate_ep{ep:03d}.jpg")
            cv2.imwrite(str(user_path), row_img)

    # Stitched grid
    if not crops_by_ep:
        print("No crops produced.")
        return
    rows = []
    for ep in TEST_EPS:
        if ep not in crops_by_ep:
            continue
        row = np.hstack(crops_by_ep[ep])
        # Pad short rows to common width
        rows.append(row)
    max_w = max(r.shape[1] for r in rows)
    rows_pad = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        rows_pad.append(r)
    grid = np.vstack(rows_pad)
    legend = np.zeros((50, max_w, 3), dtype=np.uint8)
    cv2.putText(legend, "YELLOW=13.7x8.5 spec slot   GREEN=center landmark   numbered=corner indices",
                 (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    grid_out = np.vstack([legend, grid])
    cv2.imwrite(str(GRID_OUT), grid_out)
    print(f"Wrote {sum(len(v) for v in crops_by_ep.values())} crops covering {len(crops_by_ep)} episodes")
    print(f"Per-ep stitched: /mnt/c/Users/Dell/aic_validate_ep*.jpg")
    print(f"Full grid: {GRID_OUT}")


if __name__ == "__main__":
    main()
