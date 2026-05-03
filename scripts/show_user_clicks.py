"""Render user clicks on calibration frames so we can see WHAT was clicked."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R

ROOT = Path.home() / "aic_hexdex_sfp300"
clicks_data = json.loads((Path.home() / "aic_calib_clicks.json").read_text())
gt_pose = json.loads((Path.home() / "aic_gt_port_poses.json").read_text())
offset = json.loads((Path.home() / "aic_logs/tcp_to_plug_offset.json").read_text())["sfp"]
T_TCP_plug = np.eye(4)
T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

needed = set(c["ep"] for c in clicks_data)
ep_data = {}
for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
    tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
    df = tbl.to_pandas()
    for ep_val in df["episode_index"].unique():
        ep_int = int(ep_val)
        if ep_int in needed and ep_int not in ep_data:
            file_idx = int(pf.stem.replace("file-", ""))
            eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
            ep_data[ep_int] = (file_idx, eg)

# Render first 4 calibration frames
for cd in clicks_data[:6]:
    ep, fr = cd["ep"], cd["fr"]
    if ep not in ep_data: continue
    file_idx, eg = ep_data[ep]
    states = np.stack(eg["observation.state"].values)
    frames = eg["frame_index"].to_numpy()
    m = (frames == fr)
    if not m.any(): continue
    idx = int(np.where(m)[0][0])

    T_settled = np.eye(4)
    T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
    T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
    T_base_port = T_settled @ T_TCP_plug
    T_base_tcp = state_to_T(states[idx])

    tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                        columns=["episode_index", "frame_index"])
    df_full = tbl.to_pandas()
    row_in_file = df_full[(df_full["episode_index"] == ep) & (df_full["frame_index"] == fr)].index[0]
    cap = cv2.VideoCapture(str(ROOT / "videos" / "observation.images.center" / "chunk-000" / f"file-{file_idx:03d}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row_in_file))
    ok, img = cap.read(); cap.release()
    if not ok: continue

    # Project canonical
    K = K_PER_CAM["center"]; T_tcp_opt = T_TCP_OPT["center"]
    SLOT_W, SLOT_H = 0.0137, 0.0085
    corners = np.array([[+SLOT_W/2, +SLOT_H/2, 0, 1], [+SLOT_W/2, -SLOT_H/2, 0, 1],
                         [-SLOT_W/2, -SLOT_H/2, 0, 1], [-SLOT_W/2, +SLOT_H/2, 0, 1]]).T
    pts = (np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T_base_port) @ corners
    Z = pts[2]
    canon = np.stack([K[0,0]*pts[0]/Z + K[0,2], K[1,1]*pts[1]/Z + K[1,2]], axis=1)

    # Crop center
    valid_clicks = [c for c in cd["clicks"] if c is not None]
    all_pts = np.array(valid_clicks + canon.tolist())
    cx_c, cy_c = all_pts.mean(axis=0)
    half = max(120, int(np.linalg.norm(canon[1] - canon[0]) * 3.0))
    x0 = max(0, int(cx_c - half)); x1 = min(img.shape[1], int(cx_c + half))
    y0 = max(0, int(cy_c - half)); y1 = min(img.shape[0], int(cy_c + half))
    crop = img[y0:y1, x0:x1].copy()
    ZOOM = 5
    big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

    # Canonical (yellow)
    cl = (canon - np.array([x0, y0])) * ZOOM
    cv2.polylines(big, [cl.astype(np.int32)], True, (0, 255, 255), 2)
    cv2.putText(big, "CANONICAL (yellow)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # User clicks (orange numbered)
    for i, click in enumerate(cd["clicks"]):
        if click is None: continue
        x, y = (click[0] - x0) * ZOOM, (click[1] - y0) * ZOOM
        cv2.circle(big, (int(x), int(y)), 12, (0, 165, 255), -1)
        cv2.circle(big, (int(x), int(y)), 12, (255, 255, 255), 2)
        cv2.putText(big, str(i+1), (int(x)-7, int(y)+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    cv2.putText(big, f"USER CLICKS (orange)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    cv2.putText(big, f"ep{ep:03d} fr{fr:04d}", (10, big.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    out = Path(f"/mnt/c/Users/Dell/aic_clicks_ep{ep:03d}_fr{fr:04d}.jpg")
    cv2.imwrite(str(out), big)
    print(f"saved {out}")
