#!/usr/bin/env python3
"""Project the per-episode GT port pose into every (episode, frame, camera).

For each frame:
  T_base_port = port pose recovered at deepest-insertion (constant per ep)
  T_base_TCP  = observation.state[0:9] at this frame (live)
  T_base_cam  = T_base_TCP @ T_TCP_tool0 @ T_tool0_camlink @ T_camlink_optical
  pixels      = K @ (T_base_cam^-1 @ T_base_port @ port_corners_local)

Assumes TCP frame in observation.state == tool0 (UR canonical) — verify
by visual sanity on the deepest-insertion frame.

Output: ~/aic_gt_port_2d.json
  {
    "ep_5": {
      "fr_0":   { "left": [[u,v]*4], "center": [...], "right": [...] },
      ...
    },
    ...
  }
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
OUT_PATH = Path.home() / "aic_gt_port_2d.json"

# --- Camera intrinsics + T_TCP_optical pulled from a logged trial frame ---
# (cameras are rigidly mounted on the wrist, so these are constant)
CAM_OFFSETS = json.loads((Path.home() / "aic_cam_tcp_offsets.json").read_text())
W, H = 1152, 1024
T_TCP_OPT = {cam: np.array(d["T_tcp_optical"]) for cam, d in CAM_OFFSETS.items()}
K_PER_CAM = {cam: np.array(d["K"]).reshape(3, 3) for cam, d in CAM_OFFSETS.items()}


def rot6_to_R(rot6):
    a1 = rot6[..., 0:3]; a2 = rot6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-9)
    dot = (b1 * a2).sum(axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def quat_to_R(qw, qx, qy, qz):
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def state_to_T(state_row):
    """observation.state[0:3]=xyz, [3:9]=rot6 → 4x4 T."""
    T = np.eye(4)
    T[:3, :3] = rot6_to_R(state_row[3:9])
    T[:3, 3] = state_row[0:3]
    return T


def main():
    gt_poses = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]
    print(f"K_center (intrinsics):\n{K_PER_CAM['center']}")
    print(f"\nT_TCP_plug:\n{T_TCP_plug}")
    print("\nT_TCP_optical translations:")
    for cam, T in T_TCP_OPT.items():
        print(f"  {cam}: ({T[0, 3]:+.4f}, {T[1, 3]:+.4f}, {T[2, 3]:+.4f})")

    # SFP port mouth dimensions (in port-local frame, port at z=0):
    # x = ±13.7/2 mm, y = ±8.5/2 mm. Port plane = port +z out of mouth.
    pw, ph = 0.0137, 0.0085
    port_corners_local = np.array([
        [-pw/2, -ph/2, 0, 1],
        [+pw/2, -ph/2, 0, 1],
        [+pw/2, +ph/2, 0, 1],
        [-pw/2, +ph/2, 0, 1],
    ]).T  # (4, 4)

    # Recover port pose per ep:
    # at deepest-insertion frame, T_base_plug ≈ T_base_port (plug seated)
    # so T_base_port = T_base_TCP_settled @ T_TCP_plug
    port_pose_by_ep = {}
    for ep_str, info in gt_poses.items():
        ep = int(ep_str)
        T_settled = np.eye(4)
        T_settled[:3, :3] = np.array(info["actual_tcp_R"])
        T_settled[:3, 3] = info["actual_tcp_xyz"]
        T_base_port = T_settled @ T_TCP_plug
        port_pose_by_ep[ep] = T_base_port
    print(f"\nRecovered port pose for {len(port_pose_by_ep)} episodes")

    # Walk all parquet files, project per frame.
    pqs = sorted((ROOT / "data" / "chunk-000").glob("*.parquet"))
    out: dict = {}

    n_frames_done = 0
    n_frames_total = 0
    for pf in pqs:
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep, g in df.groupby("episode_index"):
            ep = int(ep)
            if ep not in port_pose_by_ep:
                continue
            T_base_port = port_pose_by_ep[ep]
            ep_key = f"ep_{ep}"
            ep_dict = out.setdefault(ep_key, {})
            states = np.stack(g["observation.state"].values)
            frames = g["frame_index"].to_numpy()
            for i, fr in enumerate(frames):
                fr = int(fr)
                T_base_tcp = state_to_T(states[i])  # observation TCP frame == tcp_tf_base (verified)
                fr_dict = {}
                for cam, T_tcp_opt in T_TCP_OPT.items():
                    K_cam = K_PER_CAM[cam]
                    fx, fy = K_cam[0, 0], K_cam[1, 1]
                    cx_p, cy_p = K_cam[0, 2], K_cam[1, 2]
                    T_base_opt = T_base_tcp @ T_tcp_opt
                    T_opt_base = np.linalg.inv(T_base_opt)
                    T_opt_port = T_opt_base @ T_base_port
                    pts_cam = T_opt_port @ port_corners_local  # (4, 4)
                    Z = pts_cam[2]
                    if (Z <= 0).any():
                        # Behind the camera → not visible
                        fr_dict[cam] = None
                        continue
                    u = fx * pts_cam[0] / Z + cx_p
                    v = fy * pts_cam[1] / Z + cy_p
                    # Off-screen filter (allow some margin to keep boxes that
                    # poke out the edge — these are still informative).
                    margin = 80
                    if (u < -margin).all() or (u > W + margin).all() or \
                       (v < -margin).all() or (v > H + margin).all():
                        fr_dict[cam] = None
                        continue
                    pass  # keep going
                    fr_dict[cam] = [[float(u[k]), float(v[k])] for k in range(4)]
                ep_dict[f"fr_{fr}"] = fr_dict
                n_frames_done += 1
            n_frames_total += len(frames)
        print(f"  {pf.name}: {n_frames_done}/{n_frames_total} done")

    OUT_PATH.write_text(json.dumps(out))
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"Frames with GT projection: {n_frames_done}/{n_frames_total}")


if __name__ == "__main__":
    main()
