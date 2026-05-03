#!/usr/bin/env python3
"""Estimate the calibration offset between SDF port_link and the
'visible mouth' that the YOLO model is actually trained to detect.

Hypothesis: the GT in observation.port_pose_gt is the port_link frame
(from CheatCodeMJ's TF lookup of `task_board/.../<port>_link`). The SDF
says the entry is at port_link + (0,0,-0.0458). But the YOLO model was
trained on the VISIBLE slot opening, which Keerti calibrated separately
(her phase2_pnp_eval.py loads `~/aic_visible_mouth_calib.json` with a
T_canonical_to_visible_mouth offset).

For each detection where YOLO+PnP succeeded:
    T_world_mouth_visible = T_world_cam @ pose.T_cam_mouth
    T_port_to_visible     = inv(T_world_port_link) @ T_world_mouth_visible
    → record the per-axis translation + rotation

If T_port_to_visible is CONSISTENT across episodes/frames, that's the
calibration offset. The CONSTANT part = the offset to apply; the
SCATTER = the actual perception noise.

Output: print per-axis median + std + IQR.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "aic_example_policies"))
from aic_example_policies.perception.port_pose_v2.pnp import PnPConfig, estimate_pose

DATASET_ROOT = Path("/home/hariharan/aic_results/aic-sfp-500-pr")
WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
CAM_CALIB = Path.home() / "aic_cam_tcp_offsets.json"


def rot6_to_R(a1, a2):
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-9)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def state_to_T(state):
    T = np.eye(4); T[:3, :3] = rot6_to_R(state[3:6], state[6:9]); T[:3, 3] = state[0:3]
    return T


def port_gt_to_T(port):
    T = np.eye(4); T[:3, :3] = rot6_to_R(port[3:6], port[6:9]); T[:3, 3] = port[0:3]
    return T


def angle_between_R(R1, R2):
    R_rel = R1.T @ R2
    cos_th = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0))))


def main():
    ds = LeRobotDataset(repo_id="HexDexAIC/aic-sfp-500-pr",
                         root=str(DATASET_ROOT), revision="main")
    cam_offs = json.loads(CAM_CALIB.read_text())
    K_per_cam = {c: np.array(v["K"]).reshape(3, 3) for c, v in cam_offs.items()}
    T_tcp_opt_per_cam = {c: np.array(v["T_tcp_optical"]) for c, v in cam_offs.items()}
    model = YOLO(str(WEIGHTS))
    pnp_cfg = PnPConfig()

    # Pull samples spanning episodes and phases — focus on frames where
    # the port is large in image (close range) so PnP is reliable.
    test_eps = list(range(0, 250, 25))  # 10 episodes spread across 0-249

    rows = []
    n_attempts = 0
    n_ok = 0
    for ep in test_eps:
        try:
            start = int(ds.meta.episodes[ep]["dataset_from_index"])
            L = int(ds.meta.episodes[ep]["length"])
        except (KeyError, IndexError):
            continue
        # Sample mostly close-range frames where PnP is most reliable
        frame_pcts = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for pct in frame_pcts:
            fr = int(round(pct * (L - 1)))
            n_attempts += 1
            sample = ds[start + fr]
            state = sample["observation.state"].numpy()
            T_base_tcp = state_to_T(state)
            T_base_port_link = port_gt_to_T(sample["observation.port_pose_gt"].numpy())

            best = None
            best_score = 0.0
            for cam in ("left", "center", "right"):
                img_chw = sample[f"observation.images.{cam}"]
                img_rgb = (img_chw.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                results = model.predict(img_bgr, imgsz=1280, conf=0.25, verbose=False)
                r = results[0]
                if r.keypoints is None or len(r.keypoints) == 0:
                    continue
                cls_all = r.boxes.cls.cpu().numpy().astype(int)
                target = np.where(cls_all == 0)[0]
                if len(target) == 0: continue
                j = target[r.boxes.conf.cpu().numpy()[target].argmax()]
                kp = r.keypoints.xy.cpu().numpy()[j]
                bbox = r.boxes.xyxy.cpu().numpy()[j]
                conf = float(r.boxes.conf.cpu().numpy()[j])

                K = K_per_cam[cam]
                pose = estimate_pose(
                    keypoints_uv=kp, cls_id=0, bbox_xyxy=bbox, confidence=conf,
                    K=K, dist_coeffs=np.zeros(5), cfg=pnp_cfg,
                )
                if pose.quality_flag != "ok":
                    continue
                T_base_opt = T_base_tcp @ T_tcp_opt_per_cam[cam]
                T_base_visible = T_base_opt @ pose.T_cam_mouth
                score = float(pose.confidence * pose.bbox_area_px)
                if score > best_score:
                    best_score = score
                    best = (T_base_visible, pose, cam)

            if best is None:
                continue
            n_ok += 1
            T_base_visible, pose, cam = best

            # Compute T_port_link_to_visible_mouth = inv(T_base_port_link) @ T_base_visible
            T_pl_to_visible = np.linalg.inv(T_base_port_link) @ T_base_visible
            t_local = T_pl_to_visible[:3, 3]   # in port_link frame
            r_local = T_pl_to_visible[:3, :3]
            # Rotation as Rodrigues for averaging
            rvec, _ = cv2.Rodrigues(r_local)
            rows.append({
                "ep": ep, "fr": fr, "cam": cam,
                "z_cam_m": float(pose.tvec_cam[2]),
                "reproj_err_px": float(pose.reprojection_err_px),
                "tx_mm": float(t_local[0] * 1000),
                "ty_mm": float(t_local[1] * 1000),
                "tz_mm": float(t_local[2] * 1000),
                "rvec_deg": np.degrees(rvec.ravel()).tolist(),
            })

    print(f"\nAggregated {n_ok}/{n_attempts} successful PnP estimates.\n")
    if not rows:
        print("No detections — cannot estimate calibration.")
        return

    print(f"=== T_port_link → visible_mouth (per-axis, in port_link frame) ===")
    for axis in ("tx_mm", "ty_mm", "tz_mm"):
        a = np.array([r[axis] for r in rows])
        print(f"  {axis:>6}: median={np.median(a):>+8.2f} mm  "
              f"mean={np.mean(a):>+8.2f}  std={np.std(a):>6.2f}  "
              f"IQR=[{np.percentile(a, 25):>+7.2f}, {np.percentile(a, 75):>+7.2f}]")

    # Rotation: per-axis Rodrigues components
    print(f"\n=== Rotation Rodrigues (in port_link frame, degrees) ===")
    for i, axis in enumerate(("rx", "ry", "rz")):
        a = np.array([r["rvec_deg"][i] for r in rows])
        print(f"  {axis:>3}: median={np.median(a):>+8.2f}°  "
              f"mean={np.mean(a):>+8.2f}  std={np.std(a):>6.2f}")

    # Filtering: only frames with low reproj error and close range (more reliable)
    rows_close = [r for r in rows if r["z_cam_m"] < 0.20 and r["reproj_err_px"] < 1.0]
    print(f"\n=== Same, but only close-range (z<0.20m) low-reproj frames "
          f"(n={len(rows_close)}) ===")
    if rows_close:
        for axis in ("tx_mm", "ty_mm", "tz_mm"):
            a = np.array([r[axis] for r in rows_close])
            print(f"  {axis:>6}: median={np.median(a):>+8.2f} mm  "
                  f"mean={np.mean(a):>+8.2f}  std={np.std(a):>6.2f}  "
                  f"IQR=[{np.percentile(a, 25):>+7.2f}, {np.percentile(a, 75):>+7.2f}]")
        print(f"\nIf the medians are consistent and std is small, this IS the "
              f"calibration\noffset between port_link (GT) and visible_mouth (YOLO target).\n"
              f"If std is large, there's still real perception noise on top.")

    # Also report by cam — the visible-mouth calibration should be cam-independent
    print(f"\n=== By camera (close-range filter) ===")
    for cam in ("left", "center", "right"):
        rs = [r for r in rows_close if r["cam"] == cam]
        if not rs: continue
        ts = np.array([[r["tx_mm"], r["ty_mm"], r["tz_mm"]] for r in rs])
        med = np.median(ts, axis=0)
        std = np.std(ts, axis=0)
        print(f"  {cam:<7} (n={len(rs):>3}): median t = "
              f"({med[0]:>+6.2f}, {med[1]:>+6.2f}, {med[2]:>+6.2f}) mm  "
              f"std = ({std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f})")


if __name__ == "__main__":
    main()
