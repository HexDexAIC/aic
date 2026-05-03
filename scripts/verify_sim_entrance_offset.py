"""Verify the -45.8 mm port_entrance offset from sim SDF matches user clicks.

For each user-clicked calibration frame:
  - Project sfp_port_0_link (canonical) — yellow
  - Project sfp_port_0_link_entrance (canonical + dz=-45.8mm) — green
  - User clicks — orange
Compute pixel error: green vs user clicks.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R

ROOT = Path.home() / "aic_hexdex_sfp300"
clicks_data = json.loads((Path.home() / "aic_calib_clicks.json").read_text())
gt_pose = json.loads((Path.home() / "aic_gt_port_poses.json").read_text())
offset_cal = json.loads((Path.home() / "aic_logs/tcp_to_plug_offset.json").read_text())["sfp"]
T_TCP_plug = np.eye(4)
T_TCP_plug[:3, :3] = quat_to_R(offset_cal["qw"], offset_cal["qx"], offset_cal["qy"], offset_cal["qz"])
T_TCP_plug[:3, 3] = [offset_cal["x"], offset_cal["y"], offset_cal["z"]]

# THE DEFINITIVE VALUE FROM SIM SDF
ENTRANCE_DZ = -0.0458  # m in port +z
SLOT_W, SLOT_H = 0.0137, 0.0085


def project(T_base_port, dz, T_base_tcp, K, T_tcp_opt, w=SLOT_W, h=SLOT_H):
    T_co = np.eye(4); T_co[2, 3] = dz
    T_base_target = T_base_port @ T_co
    corners = np.array([[+w/2, +h/2, 0, 1], [+w/2, -h/2, 0, 1],
                         [-w/2, -h/2, 0, 1], [-w/2, +h/2, 0, 1]]).T
    pts = (np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T_base_target) @ corners
    Z = pts[2]
    if (Z <= 0).any(): return None
    return np.stack([K[0,0]*pts[0]/Z + K[0,2], K[1,1]*pts[1]/Z + K[1,2]], axis=1)


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


print(f"{'ep':>4} {'fr':>5} {'d_cm':>5} {'mean canon-user':>16} {'mean entrance-user':>20}")
canon_errs = []; ent_errs = []
for cd in clicks_data:
    ep = cd["ep"]; fr = cd["fr"]
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
    K = K_PER_CAM["center"]; T_tcp_opt = T_TCP_OPT["center"]

    canon_proj = project(T_base_port, 0.0, T_base_tcp, K, T_tcp_opt)
    ent_proj   = project(T_base_port, ENTRANCE_DZ, T_base_tcp, K, T_tcp_opt)
    if canon_proj is None or ent_proj is None: continue

    user = np.array([c for c in cd["clicks"] if c is not None])
    if len(user) < 2: continue

    # Bipartite-matched per-corner error
    cost_c = np.linalg.norm(canon_proj[:, None, :] - user[None, :, :], axis=2)
    rr, cc = linear_sum_assignment(cost_c)
    err_c = float(np.mean(cost_c[rr, cc]))
    cost_e = np.linalg.norm(ent_proj[:, None, :] - user[None, :, :], axis=2)
    rr, cc = linear_sum_assignment(cost_e)
    err_e = float(np.mean(cost_e[rr, cc]))

    port_xyz = T_base_port[:3, 3]
    d = float(np.linalg.norm(states[idx, 0:3] - port_xyz))
    print(f"{ep:>4} {fr:>5} {d*100:>5.1f} {err_c:>16.2f} {err_e:>20.2f}")
    canon_errs.append(err_c); ent_errs.append(err_e)

print(f"\nMedian per-corner error:")
print(f"  canonical (dz=0):       {np.median(canon_errs):.2f} px")
print(f"  port_entrance (dz=-45.8mm): {np.median(ent_errs):.2f} px")
print(f"\nMean per-corner error:")
print(f"  canonical:              {np.mean(canon_errs):.2f} px")
print(f"  port_entrance:          {np.mean(ent_errs):.2f} px")
