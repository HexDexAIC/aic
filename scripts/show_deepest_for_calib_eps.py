"""Render deepest-insertion frames for the calibration episodes so we can
see which physical feature is the actual SFP target port (where plug seats).
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R

ROOT = Path.home() / "aic_hexdex_sfp300"
gt_pose = json.loads((Path.home() / "aic_gt_port_poses.json").read_text())
offset = json.loads((Path.home() / "aic_logs/tcp_to_plug_offset.json").read_text())["sfp"]
T_TCP_plug = np.eye(4)
T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

EPS = [3, 41, 78, 235]

ep_data = {}
for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
    tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
    df = tbl.to_pandas()
    for ep_val in df["episode_index"].unique():
        ep_int = int(ep_val)
        if ep_int in EPS and ep_int not in ep_data:
            file_idx = int(pf.stem.replace("file-", ""))
            eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
            ep_data[ep_int] = (file_idx, eg)

for ep in EPS:
    if ep not in ep_data: continue
    file_idx, eg = ep_data[ep]
    states = np.stack(eg["observation.state"].values)
    frames = eg["frame_index"].to_numpy()

    T_settled = np.eye(4)
    T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
    T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
    T_base_port = T_settled @ T_TCP_plug

    # Find deepest insertion frame (min state z)
    z_min_idx = int(np.argmin(states[:, 2]))
    fr_idx = int(frames[z_min_idx])

    tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                        columns=["episode_index", "frame_index"])
    df_full = tbl.to_pandas()
    row_in_file = df_full[(df_full["episode_index"] == ep) & (df_full["frame_index"] == fr_idx)].index[0]
    cap = cv2.VideoCapture(str(ROOT / "videos" / "observation.images.center" / "chunk-000" / f"file-{file_idx:03d}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row_in_file))
    ok, img = cap.read(); cap.release()
    if not ok: continue

    T_base_tcp = state_to_T(states[z_min_idx])

    # Project canonical port (where /scoring/tf says it is)
    K = K_PER_CAM["center"]; T_tcp_opt = T_TCP_OPT["center"]
    SLOT_W, SLOT_H = 0.0137, 0.0085
    corners = np.array([[+SLOT_W/2, +SLOT_H/2, 0, 1], [+SLOT_W/2, -SLOT_H/2, 0, 1],
                         [-SLOT_W/2, -SLOT_H/2, 0, 1], [-SLOT_W/2, +SLOT_H/2, 0, 1]]).T
    pts = (np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T_base_port) @ corners
    Z = pts[2]
    canon = np.stack([K[0,0]*pts[0]/Z + K[0,2], K[1,1]*pts[1]/Z + K[1,2]], axis=1)

    # Crop around plug-tip (visible in image)
    cx, cy = canon[:4].mean(axis=0)
    half = 250
    x0 = max(0, int(cx - half)); x1 = min(img.shape[1], int(cx + half))
    y0 = max(0, int(cy - half)); y1 = min(img.shape[0], int(cy + half))
    crop = img[y0:y1, x0:x1].copy()

    # Draw canonical (yellow) at deepest insertion — this is where the plug IS
    cl = canon - np.array([x0, y0])
    cv2.polylines(crop, [cl.astype(np.int32)], True, (0, 255, 255), 2)
    cv2.putText(crop, "CANONICAL (where plug seats)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
    cv2.putText(crop, f"ep{ep:03d} fr{fr_idx:04d} DEEPEST INSERTION", (10, crop.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    out = Path(f"/mnt/c/Users/Dell/aic_deepest_ep{ep:03d}.jpg")
    cv2.imwrite(str(out), crop)
    print(f"saved {out}")
