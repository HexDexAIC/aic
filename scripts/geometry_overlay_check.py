#!/usr/bin/env python3
"""Compare 3 SFP-port geometry candidates by overlaying each on real frames.

Renders, for each (episode, frame) sample, a zoomed crop showing all three
candidate rectangles in different colors:
  yellow  = 13.7 x 8.5 mm  (SFP spec slot — port mouth)
  magenta = 11.0 x 11.0 mm (cage opening, square)
  cyan    = 13.4 x 13.4 mm (outer cage / recessed cutout — estimated)

Sampling strategy:
  - 10 episodes spanning the 5 rails (so we see diverse port poses)
  - per episode, frames at 3 distances:
      far    (frame 0  — port small in image, ~20cm)
      medium (~25% in  — descent phase)
      near   (~60% in  — just before insertion-phase filter would kick in)

Output:
  /tmp/aic_geom_check/ep{E}_fr{F}_dist{D}.jpg   (cropped & annotated)
  /mnt/c/Users/Dell/aic_geom_check_grid.jpg     (stitched grid for review)
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
OUT_DIR = Path("/tmp/aic_geom_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)
GRID_OUT = Path("/mnt/c/Users/Dell/aic_geom_check_grid.jpg")

# SFP port geometry candidates (width × height in port frame, metres).
# Listed lightest → heaviest so the heaviest is drawn last and on top.
CANDIDATES = [
    ("13.7x8.5_slot",       0.0137, 0.0085, (0, 255, 255)),   # YELLOW  (BGR)
    ("11x11_cage",          0.0110, 0.0110, (255, 0, 255)),   # MAGENTA
    ("13.4x13.4_outer",     0.0134, 0.0134, (255, 255, 0)),   # CYAN
    ("17.9x13.0_visible",   0.0179, 0.0130, (0, 255, 0)),     # GREEN — empirical from real-eval rim-shrink calibration
]


def quat_R(qw, qx, qy, qz):
    return quat_to_R(qw, qx, qy, qz)


def project_corners(T_base_port, T_base_tcp, K, T_tcp_opt, w_m, h_m):
    """Project a w×h rect in port +z=0 plane → 4 image points (TL,TR,BR,BL)."""
    corners_local = np.array([
        [-w_m/2, -h_m/2, 0, 1],
        [+w_m/2, -h_m/2, 0, 1],
        [+w_m/2, +h_m/2, 0, 1],
        [-w_m/2, +h_m/2, 0, 1],
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
    return np.stack([u, v], axis=1)


def main():
    # 4 episodes × 1 close-up frame each. At "close" the geometry diff is most visible.
    test_eps = [0, 47, 96, 178]

    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    K_center = K_PER_CAM["center"]
    T_tcp_opt = T_TCP_OPT["center"]

    # Index parquet files: (ep) -> (file_idx, df)
    ep_to_pq = {}
    for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep in df["episode_index"].unique():
            ep = int(ep)
            if ep in test_eps and ep not in ep_to_pq:
                file_idx = int(pf.stem.replace("file-", ""))
                ep_to_pq[ep] = (file_idx, df[df["episode_index"] == ep].sort_values("frame_index"))

    crops = []  # list of (ep, fr, dist_label, crop)

    for ep in test_eps:
        if ep not in ep_to_pq or str(ep) not in gt_pose:
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

        # Pick a frame at the boundary just before insertion phase: ~9 cm. At this
        # distance the port mouth is large in image (≈80–100 px), making 1-2 px
        # geometry differences visually obvious.
        port_xyz = T_base_port[:3, 3]
        dists = np.linalg.norm(states[:, 0:3] - port_xyz, axis=1)
        close_idx = int(np.argmax(dists < 0.09))  # first frame inside 9cm
        if close_idx == 0:
            close_idx = n // 3
        sample_frames = [("close9cm", close_idx)]
        for label, idx in sample_frames:
            fr_idx = int(frames[idx])
            file_in = idx  # row offset within this ep == file_in_file iff ep maps to one file

            # Load frame
            video_path = ROOT / "videos" / "observation.images.center" / "chunk-000" / f"file-{file_idx:03d}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, eg.index.values[idx] - eg.index.values[0])
            # Note: file_in_file is the row offset within the parquet (== frame offset within mp4
            # for v3 layout), but only when this ep is the FIRST one in this file. Let's recompute.
            # The audit-server stores GLOBAL_FRAME_INDEX[(ep, fr)]=(file_idx, frame_in_file).
            # Easier: rebuild that locally.
            cap.release()

            # Re-load with correct frame_in_file. Build it now.
            tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                                columns=["episode_index", "frame_index"])
            df_full = tbl.to_pandas()
            row_in_file = df_full[(df_full["episode_index"] == ep) &
                                   (df_full["frame_index"] == fr_idx)].index[0]

            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row_in_file))
            ok, img = cap.read()
            cap.release()
            if not ok:
                continue

            T_base_tcp = state_to_T(states[idx])

            # Compute distance from TCP to port (for filter awareness)
            tcp_to_port_m = float(np.linalg.norm(states[idx, 0:3] - T_base_port[:3, 3]))

            # Project all 3 candidates
            polys = []
            for name, w, h, color in CANDIDATES:
                pts = project_corners(T_base_port, T_base_tcp, K_center, T_tcp_opt, w, h)
                polys.append((name, color, pts))

            # Crop tightly around the port — use ~2.5x port width as half-extent.
            all_pts = np.concatenate([p[2] for p in polys if p[2] is not None], axis=0)
            if len(all_pts) == 0:
                continue
            cx, cy = all_pts.mean(axis=0)
            port_pix_w = float(np.linalg.norm(all_pts[1] - all_pts[0]))
            half = max(60, int(port_pix_w * 2.0))
            x0 = max(0, int(cx - half)); x1 = min(img.shape[1], int(cx + half))
            y0 = max(0, int(cy - half)); y1 = min(img.shape[0], int(cy + half))
            if x1 - x0 < 40 or y1 - y0 < 40:
                continue

            crop = img[y0:y1, x0:x1].copy()

            # Upscale ~6x with NEAREST (so 1-px geometry differences stay sharp)
            ZOOM = 6
            crop_big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

            for name, color, pts in polys:
                if pts is None:
                    continue
                pts_local = (pts - np.array([[x0, y0]])) * ZOOM
                pts_int = pts_local.astype(np.int32)
                cv2.polylines(crop_big, [pts_int], True, color, 2)
                cmean = pts_local.mean(axis=0).astype(int)
                cv2.circle(crop_big, tuple(cmean), 5, color, -1)

            # No further downscale — keep the high zoom visible.
            crop = crop_big
            cv2.putText(crop, f"ep{ep} fr{fr_idx} d={tcp_to_port_m*100:.1f}cm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save individual at native zoom + a copy to /mnt/c for direct viewing
            ind_path = OUT_DIR / f"ep{ep:03d}_fr{fr_idx:04d}_{label}.jpg"
            cv2.imwrite(str(ind_path), crop)
            user_path = Path("/mnt/c/Users/Dell") / f"aic_geom_ep{ep:03d}_{label}.jpg"
            cv2.imwrite(str(user_path), crop)
            crops.append(crop)

    if not crops:
        print("No crops produced.")
        return

    # Pad all to a common size so they tile cleanly
    max_h = max(c.shape[0] for c in crops)
    max_w = max(c.shape[1] for c in crops)
    crops_pad = []
    for c in crops:
        pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        pad[:c.shape[0], :c.shape[1]] = c
        crops_pad.append(pad)

    cols = 2
    rows = (len(crops_pad) + cols - 1) // cols
    grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)
    for i, c in enumerate(crops_pad):
        r, cc = i // cols, i % cols
        grid[r*max_h:(r+1)*max_h, cc*max_w:(cc+1)*max_w] = c

    legend = np.zeros((60, cols * max_w, 3), dtype=np.uint8)
    legend_lines = [
        "YELLOW=13.7x8.5 spec slot     MAGENTA=11x11 cage opening",
        "CYAN=13.4x13.4 outer cage     GREEN=17.9x13.0 calibrated visible",
    ]
    for i, ln in enumerate(legend_lines):
        cv2.putText(legend, ln, (10, 22 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    grid_out = np.vstack([legend, grid])
    cv2.imwrite(str(GRID_OUT), grid_out)
    print(f"Wrote {len(crops)} crops to {OUT_DIR}")
    print(f"Stitched grid: {GRID_OUT}")


if __name__ == "__main__":
    main()
