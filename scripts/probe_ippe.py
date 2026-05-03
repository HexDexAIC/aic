"""Probe what cv2.solvePnPGeneric IPPE actually returns. Are we getting 2 solutions
or 1? What do their rotations look like?
"""
import json
import numpy as np
import sys
from pathlib import Path
import cv2
import pyarrow.parquet as pq
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import quat_to_R, state_to_T

WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
ROOT = Path.home() / "aic_hexdex_sfp300"
TEST_IMAGES = Path.home() / "aic_yolo_v1" / "images" / "test"
GT_POSE = json.loads((Path.home() / "aic_gt_port_poses.json").read_text())
plug_off = json.loads((Path.home() / "aic_logs" / "tcp_to_plug_offset.json").read_text())["sfp"]
T_TCP_plug = np.eye(4)
T_TCP_plug[:3, :3] = quat_to_R(plug_off["qw"], plug_off["qx"], plug_off["qy"], plug_off["qz"])
T_TCP_plug[:3, 3] = [plug_off["x"], plug_off["y"], plug_off["z"]]
cam_offs = json.loads((Path.home() / "aic_cam_tcp_offsets.json").read_text())
K_per_cam = {c: np.array(v["K"]).reshape(3, 3) for c, v in cam_offs.items()}
T_tcp_opt = {c: np.array(v["T_tcp_optical"]) for c, v in cam_offs.items()}
calib = json.loads((Path.home() / "aic_visible_mouth_calib.json").read_text())
co = calib["T_canonical_to_visible_mouth"]
T_co_mouth = np.eye(4); T_co_mouth[:3, 3] = [co["dx_mm"]/1000, co["dy_mm"]/1000, co["dz_mm"]/1000]

OBJECT_POINTS_4 = np.array([
    [+0.0137/2, +0.0085/2, 0.0],
    [+0.0137/2, -0.0085/2, 0.0],
    [-0.0137/2, -0.0085/2, 0.0],
    [-0.0137/2, +0.0085/2, 0.0],
], dtype=np.float64)

# Pick a test frame at far distance (where IPPE flips most)
ep = 203
state_lookup = {}
for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
    tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
    df = tbl.to_pandas()
    if not (df["episode_index"] == ep).any(): continue
    for _, row in df[df["episode_index"] == ep].iterrows():
        state_lookup[int(row["frame_index"])] = np.asarray(row["observation.state"])

T_settled = np.eye(4)
T_settled[:3, :3] = np.array(GT_POSE[str(ep)]["actual_tcp_R"])
T_settled[:3, 3] = GT_POSE[str(ep)]["actual_tcp_xyz"]
T_base_target_canon = T_settled @ T_TCP_plug
T_base_mouth_GT = T_base_target_canon @ T_co_mouth

model = YOLO(str(WEIGHTS))

for fr_idx in (0, 30, 50, 80, 110, 140):
    s = state_lookup.get(fr_idx)
    if s is None: continue
    T_base_tcp = state_to_T(s)
    d_cm = float(np.linalg.norm(s[:3] - T_base_target_canon[:3, 3]) * 100)

    img_path = TEST_IMAGES / f"ep{ep:03d}_fr{fr_idx:05d}_center.jpg"
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    res = model.predict(img, imgsz=1280, conf=0.25, device=0, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0: continue
    cls_all = res.boxes.cls.cpu().numpy().astype(int)
    target_idx = next((j for j in range(len(cls_all)) if cls_all[j] == 0), None)
    if target_idx is None: continue
    kpts = res.keypoints.xy.cpu().numpy()[target_idx][:4].astype(np.float64)
    K = K_per_cam["center"]

    retval, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
        objectPoints=OBJECT_POINTS_4,
        imagePoints=kpts.reshape(-1, 1, 2),
        cameraMatrix=K,
        distCoeffs=np.zeros(5),
        flags=cv2.SOLVEPNP_IPPE,
    )
    n_sols = len(rvecs) if rvecs is not None else 0

    # GT in cam frame
    T_base_opt = T_base_tcp @ T_tcp_opt["center"]
    T_opt_mouth_GT = np.linalg.inv(T_base_opt) @ T_base_mouth_GT
    gt_z = T_opt_mouth_GT[:3, :3][:, 2]

    print(f"\n=== ep{ep} fr{fr_idx} center d={d_cm:.1f}cm  IPPE returned {n_sols} solutions ===")
    print(f"  GT  port_+Z in cam = ({gt_z[0]:+.3f}, {gt_z[1]:+.3f}, {gt_z[2]:+.3f})  (z-comp = {gt_z[2]:+.3f})")
    for i in range(n_sols):
        R_i, _ = cv2.Rodrigues(rvecs[i])
        port_z = R_i[:, 2]
        rerr = float(reproj_errs[i].ravel()[0]) if reproj_errs is not None else float("nan")
        # Rotation error vs GT
        R_rel = R_i.T @ T_opt_mouth_GT[:3, :3]
        cos_th = (np.trace(R_rel) - 1) / 2
        rot_err_deg = float(np.degrees(np.arccos(np.clip(cos_th, -1, 1))))
        passes_my_filter = port_z[2] > 0
        print(f"    sol {i}: port_+Z = ({port_z[0]:+.3f}, {port_z[1]:+.3f}, {port_z[2]:+.3f})  "
              f"reproj_err = {rerr:.3f}  rot_err_vs_GT = {rot_err_deg:.1f}°  "
              f"my_filter_pass={passes_my_filter}")
