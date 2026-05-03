#!/usr/bin/env python3
"""Phase-2 PnP eval against GT pose on test split.

For every test frame:
  1. Run detector → keypoints + bbox + confidence
  2. Run PnP module → T_cam_mouth (or rejected with quality_flag)
  3. Compute T_base_mouth via per-cam T_tcp_optical and per-frame TCP
  4. Compute T_base_mouth_GT from SDF pipeline (canonical + entrance offset)
  5. Report:
       translation_err_mm
       rotation_err_deg
       reprojection_err_px
       per (camera × distance × visibility × class) bin

Output: ~/aic_runs/v1_h100_results/test_eval/phase2_pnp_eval.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "aic_example_policies"))
from aic_example_policies.perception.port_pose_v2.pnp import (
    PnPConfig, estimate_pose,
)

DATA_ROOT = Path.home() / "aic_hexdex_sfp300"
WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
TEST_LABELS = Path.home() / "aic_yolo_v1" / "labels" / "test"
TEST_IMAGES = Path.home() / "aic_yolo_v1" / "images" / "test"
GT_POSE = Path.home() / "aic_gt_port_poses.json"
TCP_PLUG_OFFSET = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
CAM_OFFSETS = Path.home() / "aic_cam_tcp_offsets.json"
CALIB = Path.home() / "aic_visible_mouth_calib.json"

OUT = Path.home() / "aic_runs" / "v1_h100_results" / "test_eval" / "phase2_pnp_eval.json"

IMG_W, IMG_H = 1152, 1024


def quat_to_R(qw, qx, qy, qz):
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def state_to_T(s):
    """observation.state[0:9] → 4x4 T_base_tcp (rot6 representation)."""
    T = np.eye(4)
    a1 = s[3:6]; a2 = s[6:9]
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-9)
    b3 = np.cross(b1, b2)
    T[:3, :3] = np.stack([b1, b2, b3], axis=-1)
    T[:3, 3] = s[0:3]
    return T


def visibility_label(kpts, w, h, margin=8):
    on = sum(1 for x, y in kpts[:4] if margin <= x < w - margin and margin <= y < h - margin)
    if on == 4: return "fully_visible"
    elif on >= 2: return "partially_clipped"
    else: return "heavily_clipped"


def dist_band_cm(d_cm):
    if d_cm < 10: return "<10cm"
    elif d_cm < 15: return "10-15cm"
    elif d_cm < 20: return "15-20cm"
    else: return ">20cm"


def main():
    print(f"Loading {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    # Load all GT and calibration
    gt_pose = json.loads(GT_POSE.read_text())
    plug_off = json.loads(TCP_PLUG_OFFSET.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(plug_off["qw"], plug_off["qx"], plug_off["qy"], plug_off["qz"])
    T_TCP_plug[:3, 3] = [plug_off["x"], plug_off["y"], plug_off["z"]]

    cam_offs = json.loads(CAM_OFFSETS.read_text())
    K_per_cam = {c: np.array(v["K"]).reshape(3, 3) for c, v in cam_offs.items()}
    T_tcp_opt_per_cam = {c: np.array(v["T_tcp_optical"]) for c, v in cam_offs.items()}
    dist_per_cam = {c: np.zeros(5) for c in cam_offs}  # zeros for sim

    calib = json.loads(CALIB.read_text())
    co = calib["T_canonical_to_visible_mouth"]
    T_co_mouth = np.eye(4)
    T_co_mouth[:3, 3] = [co["dx_mm"]/1000, co["dy_mm"]/1000, co["dz_mm"]/1000]
    td = calib["T_target_to_distractor"]
    T_target_to_distractor = np.eye(4)
    T_target_to_distractor[:3, 3] = [td["dx_mm"]/1000, td["dy_mm"]/1000, td["dz_mm"]/1000]

    # Index test parquet for TCP state
    print("Indexing parquet (test eps only)...")
    state_by_ef = {}
    for pf in sorted((DATA_ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        eps = df["episode_index"].unique()
        if eps.min() < 200 and eps.max() < 200:  # skip files with no test eps
            continue
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            if ep < 200:
                continue
            fr = int(row["frame_index"])
            state_by_ef[(ep, fr)] = np.asarray(row["observation.state"])

    pnp_cfg = PnPConfig()
    label_files = sorted(TEST_LABELS.glob("*.txt"))
    print(f"Test frames: {len(label_files)}")

    # Bin: (camera, distance_band, visibility, class) -> list of error records
    buckets = defaultdict(list)
    quality_flag_counts = defaultdict(int)

    for i, lbl_path in enumerate(label_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(label_files)}")
        m = re.match(r"ep(\d+)_fr(\d+)_(left|center|right)", lbl_path.stem)
        if not m: continue
        ep = int(m.group(1)); fr = int(m.group(2)); cam = m.group(3)

        img_path = TEST_IMAGES / (lbl_path.stem + ".jpg")
        if not img_path.exists(): continue
        if (ep, fr) not in state_by_ef or str(ep) not in gt_pose: continue

        state = state_by_ef[(ep, fr)]
        T_base_tcp = state_to_T(state)

        # GT mouth poses in base_link (target + distractor)
        T_settled = np.eye(4)
        T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
        T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
        T_base_target_canon = T_settled @ T_TCP_plug
        T_base_target_mouth_GT = T_base_target_canon @ T_co_mouth
        T_base_distractor_mouth_GT = T_base_target_canon @ T_target_to_distractor @ T_co_mouth

        d_cm = float(np.linalg.norm(state[:3] - T_base_target_canon[:3, 3]) * 100)
        d_band = dist_band_cm(d_cm)

        # Predict
        res = model.predict(str(img_path), imgsz=1280, conf=0.25, device=0, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            quality_flag_counts["no_detection"] += 1
            continue

        K = K_per_cam[cam]
        dist_coeffs = dist_per_cam[cam]
        T_tcp_opt = T_tcp_opt_per_cam[cam]

        kpts_all = res.keypoints.xy.cpu().numpy()
        cls_all = res.boxes.cls.cpu().numpy().astype(int)
        conf_all = res.boxes.conf.cpu().numpy()
        bbox_all = res.boxes.xyxy.cpu().numpy()

        for j in range(len(kpts_all)):
            kpts = kpts_all[j]
            cls_id = int(cls_all[j])
            class_name = "sfp_target" if cls_id == 0 else "sfp_distractor"

            # Visibility from PREDICTED corners (gates use these)
            vis = visibility_label(kpts, IMG_W, IMG_H)

            pose = estimate_pose(
                keypoints_uv=kpts,
                cls_id=cls_id,
                bbox_xyxy=bbox_all[j],
                confidence=float(conf_all[j]),
                K=K, dist_coeffs=dist_coeffs,
                cfg=pnp_cfg,
            )
            quality_flag_counts[pose.quality_flag] += 1

            if pose.quality_flag != "ok":
                # Record rejection in buckets so we can compute acceptance rate
                buckets[(cam, d_band, vis, class_name)].append({
                    "rejected": True, "quality_flag": pose.quality_flag,
                })
                continue

            # Compose to base_link
            T_base_opt = T_base_tcp @ T_tcp_opt
            T_base_mouth_pred = T_base_opt @ pose.T_cam_mouth

            # GT pose for this class
            T_gt = T_base_target_mouth_GT if cls_id == 0 else T_base_distractor_mouth_GT

            # Translation error (mm)
            t_err_mm = float(np.linalg.norm(T_base_mouth_pred[:3, 3] - T_gt[:3, 3]) * 1000)

            # Rotation error (deg)
            R_rel = T_base_mouth_pred[:3, :3].T @ T_gt[:3, :3]
            cos_th = (np.trace(R_rel) - 1) / 2
            r_err_deg = float(np.degrees(np.arccos(np.clip(cos_th, -1, 1))))

            buckets[(cam, d_band, vis, class_name)].append({
                "rejected": False,
                "translation_err_mm": t_err_mm,
                "rotation_err_deg": r_err_deg,
                "reproj_err_px": float(pose.reprojection_err_px),
                "center_resid_px": float(pose.center_residual_px),
                "z_cam_m": float(pose.tvec_cam[2]),
                "bbox_area": float(pose.bbox_area_px),
                "conf": float(pose.confidence),
                "d_cm": d_cm,
            })

    # Aggregate per bucket
    summary = {"buckets": {}, "quality_flag_counts": dict(quality_flag_counts)}
    for k, vals in sorted(buckets.items()):
        cam, d_band, vis, class_name = k
        n_total = len(vals)
        accepted = [v for v in vals if not v["rejected"]]
        rejected = [v for v in vals if v["rejected"]]
        if not accepted:
            summary["buckets"][f"{cam}/{d_band}/{vis}/{class_name}"] = {
                "n_total": n_total, "n_accepted": 0,
                "acceptance_rate": 0.0, "median_t_mm": None, "median_r_deg": None,
            }
            continue
        ts = np.array([v["translation_err_mm"] for v in accepted])
        rs = np.array([v["rotation_err_deg"] for v in accepted])
        rps = np.array([v["reproj_err_px"] for v in accepted])
        summary["buckets"][f"{cam}/{d_band}/{vis}/{class_name}"] = {
            "n_total": n_total, "n_accepted": len(accepted),
            "acceptance_rate": len(accepted) / n_total,
            "median_t_mm": float(np.median(ts)), "p90_t_mm": float(np.percentile(ts, 90)),
            "median_r_deg": float(np.median(rs)), "p90_r_deg": float(np.percentile(rs, 90)),
            "median_reproj_px": float(np.median(rps)), "p90_reproj_px": float(np.percentile(rps, 90)),
        }

    # Aggregate overall
    all_accepted = [v for vs in buckets.values() for v in vs if not v["rejected"]]
    if all_accepted:
        ts = np.array([v["translation_err_mm"] for v in all_accepted])
        rs = np.array([v["rotation_err_deg"] for v in all_accepted])
        summary["overall"] = {
            "n_accepted": len(all_accepted),
            "acceptance_rate": len(all_accepted) / sum(len(vs) for vs in buckets.values()),
            "median_t_mm": float(np.median(ts)),
            "p90_t_mm": float(np.percentile(ts, 90)),
            "p99_t_mm": float(np.percentile(ts, 99)),
            "median_r_deg": float(np.median(rs)),
            "p90_r_deg": float(np.percentile(rs, 90)),
            "p99_r_deg": float(np.percentile(rs, 99)),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nResults: {OUT}")
    print(json.dumps(summary.get("overall", {}), indent=2))
    print("\nQuality flag counts:")
    for k, v in sorted(quality_flag_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
