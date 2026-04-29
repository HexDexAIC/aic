#!/usr/bin/env python3
"""Sanity-check the GT projection on one episode.

For each tested episode:
  - find the deepest-insertion frame
  - project port corners onto each cam at that frame
  - draw onto frame, save to /tmp/aic_sanity_ep{e}_fr{f}_{cam}.jpg
  - also draw at frame_idx=0 (start of approach) → port should be far,
    visible across the frame
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import (
    K_PER_CAM, T_TCP_OPT, W, H, state_to_T, quat_to_R,
)

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
OUT_DIR = Path("/tmp/aic_sanity")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_frame(ep, fr, cam, file_idx, frame_in_file):
    chunk_idx = 0
    p = ROOT / "videos" / f"observation.images.{cam}" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    if not p.exists():
        return None
    cap = cv2.VideoCapture(str(p))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_file)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main():
    gt = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    # Build (ep, fr) -> (file_idx, frame_in_file) for the test eps
    test_eps = [0, 5, 30, 60, 120, 180]
    pq_files = sorted((ROOT / "data" / "chunk-000").glob("*.parquet"))

    for pf in pq_files:
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep in test_eps:
            g = df[df["episode_index"] == ep]
            if len(g) == 0:
                continue
            file_idx_int = int(pf.stem.replace("file-", ""))
            states = np.stack(g["observation.state"].values)
            frames = g["frame_index"].to_numpy()

            # episode-local frame_in_file = row index within this file's slice for this ep
            file_offsets = g.index.values - g.index.values[0]

            # Find deepest-insertion frame (min z)
            min_z_local = int(np.argmin(states[:, 2]))
            frames_to_check = [
                ("start", 0),
                ("midway", len(frames) // 2),
                ("deepest", min_z_local),
            ]

            # Recover port pose
            ep_str = str(ep)
            if ep_str not in gt:
                continue
            T_settled = np.eye(4)
            T_settled[:3, :3] = np.array(gt[ep_str]["actual_tcp_R"])
            T_settled[:3, 3] = gt[ep_str]["actual_tcp_xyz"]
            T_base_port = T_settled @ T_TCP_plug

            pw, ph = 0.0137, 0.0085
            corners_local = np.array([
                [-pw/2, -ph/2, 0, 1],
                [+pw/2, -ph/2, 0, 1],
                [+pw/2, +ph/2, 0, 1],
                [-pw/2, +ph/2, 0, 1],
            ]).T

            for tag, idx in frames_to_check:
                fr_idx = int(frames[idx])
                file_in_idx = int(file_offsets[idx])
                T_base_tcp = state_to_T(states[idx])
                for cam, T_tcp_opt in T_TCP_OPT.items():
                    img = load_frame(ep, fr_idx, cam, file_idx_int, file_in_idx)
                    if img is None:
                        continue
                    K_cam = K_PER_CAM[cam]
                    fx, fy = K_cam[0, 0], K_cam[1, 1]
                    cx_p, cy_p = K_cam[0, 2], K_cam[1, 2]
                    T_base_opt = T_base_tcp @ T_tcp_opt
                    T_opt_port = np.linalg.inv(T_base_opt) @ T_base_port
                    pts = T_opt_port @ corners_local
                    Z = pts[2]
                    in_front = (Z > 0).all()
                    if not in_front:
                        text = f"ep{ep} fr{fr_idx} {cam} {tag}: BEHIND CAM (z={Z})"
                        print(text)
                        cv2.putText(img, "BEHIND CAM", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        u = fx * pts[0] / Z + cx_p
                        v = fy * pts[1] / Z + cy_p
                        cx = float(u.mean()); cy = float(v.mean())
                        text = f"ep{ep} fr{fr_idx} {cam} {tag}: cx={cx:.0f} cy={cy:.0f} z={float(Z.mean()):.3f}"
                        print(text)
                        pts2d = np.stack([u, v], axis=1).astype(np.int32)
                        cv2.polylines(img, [pts2d], True, (0, 255, 255), 3)
                        cv2.putText(img, f"GT {tag}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    out_path = OUT_DIR / f"ep{ep}_fr{fr_idx}_{cam}_{tag}.jpg"
                    cv2.imwrite(str(out_path), img)
        # Test eps appearing in this file are now done
        # Don't loop more files for them; but move on to find other eps in other files
    print(f"\nSaved samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
