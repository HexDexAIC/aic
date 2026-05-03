"""Quick sanity check: at a known-good frame, what is the port +Z axis direction
in the camera frame? Confirms whether my disambiguation sign is right.
"""
import json
import numpy as np
import sys
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import quat_to_R, state_to_T

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE = json.loads((Path.home() / "aic_gt_port_poses.json").read_text())
plug_off = json.loads((Path.home() / "aic_logs" / "tcp_to_plug_offset.json").read_text())["sfp"]
T_TCP_plug = np.eye(4)
T_TCP_plug[:3, :3] = quat_to_R(plug_off["qw"], plug_off["qx"], plug_off["qy"], plug_off["qz"])
T_TCP_plug[:3, 3] = [plug_off["x"], plug_off["y"], plug_off["z"]]
cam_offs = json.loads((Path.home() / "aic_cam_tcp_offsets.json").read_text())
T_tcp_opt = {c: np.array(v["T_tcp_optical"]) for c, v in cam_offs.items()}
calib = json.loads((Path.home() / "aic_visible_mouth_calib.json").read_text())
co = calib["T_canonical_to_visible_mouth"]
T_co_mouth = np.eye(4); T_co_mouth[:3, 3] = [co["dx_mm"]/1000, co["dy_mm"]/1000, co["dz_mm"]/1000]

# Take ep203 (passes), some early frame
ep = 203
state_lookup = {}
for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
    tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
    df = tbl.to_pandas()
    if not (df["episode_index"] == ep).any():
        continue
    for _, row in df[df["episode_index"] == ep].iterrows():
        state_lookup[int(row["frame_index"])] = np.asarray(row["observation.state"])

T_settled = np.eye(4)
T_settled[:3, :3] = np.array(GT_POSE[str(ep)]["actual_tcp_R"])
T_settled[:3, 3] = GT_POSE[str(ep)]["actual_tcp_xyz"]
T_base_target_canon = T_settled @ T_TCP_plug
T_base_mouth_GT = T_base_target_canon @ T_co_mouth

# Pick a frame where TCP is at moderate distance
frs = sorted(state_lookup.keys())
for fr in frs[::10]:  # every 10th frame
    s = state_lookup[fr]
    T_base_tcp = state_to_T(s)
    d_cm = float(np.linalg.norm(s[:3] - T_base_target_canon[:3, 3]) * 100)
    if d_cm < 18 or d_cm > 22:  # only mid-distance
        continue
    for cam in ("center", "right", "left"):
        T_base_opt = T_base_tcp @ T_tcp_opt[cam]
        T_opt_mouth_GT = np.linalg.inv(T_base_opt) @ T_base_mouth_GT
        port_z_in_cam = T_opt_mouth_GT[:3, :3][:, 2]
        print(f"ep{ep} fr{fr} {cam} d={d_cm:.1f}cm  "
              f"port_+Z in cam frame: ({port_z_in_cam[0]:+.3f}, {port_z_in_cam[1]:+.3f}, {port_z_in_cam[2]:+.3f})  "
              f"z-component: {port_z_in_cam[2]:+.3f}  → "
              f"{'TOWARD CAM (expect <0)' if port_z_in_cam[2] < 0 else 'AWAY FROM CAM (expect >0)'}")
    break
