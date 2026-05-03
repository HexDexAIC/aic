"""For each calibration episode, render a side-by-side:
  LEFT  — your calibration frame (~18cm) with USER clicks (orange) + CANONICAL (yellow)
  RIGHT — the SAME episode at deepest insertion (where plug physically seats), with CANONICAL (yellow)

This shows where the plug actually GOES (right panel) vs where you clicked (left panel orange).
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
clicks_data = json.loads((Path.home() / "aic_calib_clicks.json").read_text())
gt_pose = json.loads((Path.home() / "aic_gt_port_poses.json").read_text())
offset = json.loads((Path.home() / "aic_logs/tcp_to_plug_offset.json").read_text())["sfp"]
T_TCP_plug = np.eye(4)
T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

# Pick the original 6 calibration eps
EPS_TO_SHOW = [3, 41, 78, 235]

# Load frames per episode
ep_data = {}
for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
    tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
    df = tbl.to_pandas()
    for ep_val in df["episode_index"].unique():
        ep_int = int(ep_val)
        if ep_int in EPS_TO_SHOW and ep_int not in ep_data:
            file_idx = int(pf.stem.replace("file-", ""))
            eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
            ep_data[ep_int] = (file_idx, eg)


def project_canon(T_base_port, T_base_tcp):
    K = K_PER_CAM["center"]; T_tcp_opt = T_TCP_OPT["center"]
    SLOT_W, SLOT_H = 0.0137, 0.0085
    corners = np.array([[+SLOT_W/2, +SLOT_H/2, 0, 1], [+SLOT_W/2, -SLOT_H/2, 0, 1],
                         [-SLOT_W/2, -SLOT_H/2, 0, 1], [-SLOT_W/2, +SLOT_H/2, 0, 1]]).T
    pts = (np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T_base_port) @ corners
    Z = pts[2]
    return np.stack([K[0,0]*pts[0]/Z + K[0,2], K[1,1]*pts[1]/Z + K[1,2]], axis=1)


def load_frame(ep, fr):
    file_idx, eg = ep_data[ep]
    states = np.stack(eg["observation.state"].values)
    frames = eg["frame_index"].to_numpy()
    m = (frames == fr)
    if not m.any(): return None, None
    idx = int(np.where(m)[0][0])
    tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                        columns=["episode_index", "frame_index"])
    df_full = tbl.to_pandas()
    row = df_full[(df_full["episode_index"] == ep) & (df_full["frame_index"] == fr)].index[0]
    cap = cv2.VideoCapture(str(ROOT / "videos" / "observation.images.center" / "chunk-000" / f"file-{file_idx:03d}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row))
    ok, img = cap.read(); cap.release()
    return (img, states[idx]) if ok else (None, None)


def render_panel(img, T_base_port, state, user_clicks=None, label="", subtitle=""):
    T_base_tcp = state_to_T(state)
    canon = project_canon(T_base_port, T_base_tcp)
    cx, cy = canon[:4].mean(axis=0)
    half = 220
    x0 = max(0, int(cx - half)); x1 = min(img.shape[1], int(cx + half))
    y0 = max(0, int(cy - half)); y1 = min(img.shape[0], int(cy + half))
    crop = img[y0:y1, x0:x1].copy()
    ZOOM = 3
    big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    cl = (canon - np.array([x0, y0])) * ZOOM
    cv2.polylines(big, [cl.astype(np.int32)], True, (0, 255, 255), 3)
    cv2.putText(big, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(big, subtitle, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    if user_clicks:
        for i, c in enumerate(user_clicks):
            if c is None: continue
            xx, yy = (c[0] - x0) * ZOOM, (c[1] - y0) * ZOOM
            cv2.circle(big, (int(xx), int(yy)), 14, (0, 165, 255), -1)
            cv2.circle(big, (int(xx), int(yy)), 14, (255, 255, 255), 2)
            cv2.putText(big, str(i+1), (int(xx)-8, int(yy)+7),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
        cv2.putText(big, "USER CLICKS (orange)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,165,255), 2)
    # Letterbox to 700x700
    CANVAS = 700
    h_b, w_b = big.shape[:2]
    s = CANVAS / max(h_b, w_b)
    nh, nw = int(h_b * s), int(w_b * s)
    res = cv2.resize(big, (nw, nh))
    canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    yo = (CANVAS - nh) // 2; xo = (CANVAS - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = res
    return canvas


for ep in EPS_TO_SHOW:
    if ep not in ep_data: continue
    cd = next((c for c in clicks_data if c["ep"] == ep and c["fr"] < 100), None)
    if cd is None:
        cd = next((c for c in clicks_data if c["ep"] == ep), None)
    if cd is None: continue
    file_idx, eg = ep_data[ep]
    states_all = np.stack(eg["observation.state"].values)

    T_settled = np.eye(4)
    T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
    T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
    T_base_port = T_settled @ T_TCP_plug
    port_xyz = T_base_port[:3, 3]
    dists = np.linalg.norm(states_all[:, 0:3] - port_xyz, axis=1)

    # LEFT panel: calibration frame (the one user clicked)
    img_l, state_l = load_frame(ep, cd["fr"])
    if img_l is None: continue
    actual_d_l = float(dists[np.where(eg["frame_index"].to_numpy() == cd["fr"])[0][0]])

    # RIGHT panel: deepest insertion frame
    z_min_idx = int(np.argmin(states_all[:, 2]))
    fr_deep = int(eg["frame_index"].to_numpy()[z_min_idx])
    img_r, state_r = load_frame(ep, fr_deep)
    if img_r is None: continue
    actual_d_r = float(dists[z_min_idx])

    left = render_panel(img_l, T_base_port, state_l, user_clicks=cd["clicks"],
                          label=f"Calibration view (you clicked here)",
                          subtitle=f"ep{ep:03d} fr{cd['fr']:04d}  d={actual_d_l*100:.0f}cm")
    right = render_panel(img_r, T_base_port, state_r, user_clicks=None,
                           label=f"Deepest insertion (where plug seats)",
                           subtitle=f"ep{ep:03d} fr{fr_deep:04d}  d={actual_d_r*100:.0f}cm")

    # Add a vertical separator
    sep = np.full((700, 6, 3), 80, dtype=np.uint8)
    side = np.hstack([left, sep, right])

    # Add big label at top
    label_strip = np.zeros((50, side.shape[1], 3), dtype=np.uint8)
    cv2.putText(label_strip, f"ep{ep:03d}: same NIC card, two distances",
                 (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    final = np.vstack([label_strip, side])
    out = Path(f"/mnt/c/Users/Dell/aic_evidence_ep{ep:03d}.jpg")
    cv2.imwrite(str(out), final)
    print(f"saved {out}")
