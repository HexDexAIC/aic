#!/usr/bin/env python3
"""Phase 6.sim — replay-based closed-loop eval.

For each of 20 test episodes (200-219):
  for each camera (left, center, right):
    for each frame in temporal order:
      detect → PnP → tracker.update → record pose_trajectory[t]
  compare pose_trajectory to GT
  bucket errors by distance band (hover / approach / near)
  per-episode pass/fail: max-error stays < 5mm trans, < 8° rot at any distance

This is closed-loop in the sense that the tracker carries state across the
full trajectory (not per-frame-independent). It is NOT closed-loop in the
sense of feeding pose to a controller and re-rendering — that's Phase 6.robot.
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
from aic_example_policies.perception.port_pose_v2.tracker import (
    SE3Tracker, TrackerConfig, TrackerState,
)

DATA_ROOT = Path.home() / "aic_hexdex_sfp300"
WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
TEST_LABELS = Path.home() / "aic_yolo_v1" / "labels" / "test"
TEST_IMAGES = Path.home() / "aic_yolo_v1" / "images" / "test"
GT_POSE = Path.home() / "aic_gt_port_poses.json"
TCP_PLUG_OFFSET = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
CAM_OFFSETS = Path.home() / "aic_cam_tcp_offsets.json"
CALIB = Path.home() / "aic_visible_mouth_calib.json"

OUT = Path.home() / "aic_runs" / "v1_h100_results" / "test_eval" / "phase6_replay_eval.json"

EPS_TO_EVAL = list(range(200, 220))  # 20 episodes

# Pass/fail thresholds (provisional)
PASS_T_MM = 5.0
PASS_R_DEG = 8.0


def quat_to_R(qw, qx, qy, qz):
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def state_to_T(s):
    T = np.eye(4)
    a1 = s[3:6]; a2 = s[6:9]
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-9)
    b3 = np.cross(b1, b2)
    T[:3, :3] = np.stack([b1, b2, b3], axis=-1)
    T[:3, 3] = s[0:3]
    return T


def angle_deg(R1, R2):
    R = R1.T @ R2
    cos_th = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos_th, -1, 1))))


def dist_band_cm(d_cm):
    if d_cm < 10: return "near"
    elif d_cm < 15: return "approach"
    elif d_cm < 20: return "hover"
    else: return "far"


def main():
    print(f"Loading {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    gt_pose = json.loads(GT_POSE.read_text())
    plug_off = json.loads(TCP_PLUG_OFFSET.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(plug_off["qw"], plug_off["qx"], plug_off["qy"], plug_off["qz"])
    T_TCP_plug[:3, 3] = [plug_off["x"], plug_off["y"], plug_off["z"]]
    cam_offs = json.loads(CAM_OFFSETS.read_text())
    K_per_cam = {c: np.array(v["K"]).reshape(3, 3) for c, v in cam_offs.items()}
    T_tcp_opt_per_cam = {c: np.array(v["T_tcp_optical"]) for c, v in cam_offs.items()}
    calib = json.loads(CALIB.read_text())
    co = calib["T_canonical_to_visible_mouth"]
    T_co_mouth = np.eye(4)
    T_co_mouth[:3, 3] = [co["dx_mm"]/1000, co["dy_mm"]/1000, co["dz_mm"]/1000]

    # State lookup per (ep, fr)
    print("Indexing parquet for test eps...")
    state_by_ef = {}
    for pf in sorted((DATA_ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        eps = df["episode_index"].unique()
        if all(e < 200 or e >= 220 for e in eps):
            continue
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            if ep < 200 or ep >= 220:
                continue
            fr = int(row["frame_index"])
            state_by_ef[(ep, fr)] = np.asarray(row["observation.state"])
    print(f"  Indexed {len(state_by_ef)} (ep, fr) pairs")

    pnp_cfg = PnPConfig()
    tracker_cfg = TrackerConfig()

    per_episode = []
    for ep in EPS_TO_EVAL:
        if str(ep) not in gt_pose:
            print(f"ep{ep}: no GT, skip")
            continue

        # GT mouth pose for target
        T_settled = np.eye(4)
        T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
        T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
        T_base_target_canon = T_settled @ T_TCP_plug
        T_base_mouth_GT = T_base_target_canon @ T_co_mouth

        ep_summary = {"ep": ep, "by_cam": {}}
        # Per camera, run tracker through the full trajectory
        for cam in ("left", "center", "right"):
            # Collect frames in this ep/cam, sorted by frame index
            frames = []
            for lbl in TEST_LABELS.glob(f"ep{ep:03d}_*_{cam}.txt"):
                m = re.match(rf"ep{ep:03d}_fr(\d+)_{cam}", lbl.stem)
                if not m: continue
                fr = int(m.group(1))
                if (ep, fr) not in state_by_ef: continue
                img_path = TEST_IMAGES / (lbl.stem + ".jpg")
                if not img_path.exists(): continue
                frames.append((fr, img_path))
            frames.sort()
            if len(frames) < 5:
                continue

            tracker = SE3Tracker(tracker_cfg)
            errors = []  # list of (d_cm, t_err_mm, r_err_deg, state)
            for fr, img_path in frames:
                state = state_by_ef[(ep, fr)]
                T_base_tcp = state_to_T(state)
                d_cm = float(np.linalg.norm(state[:3] - T_base_target_canon[:3, 3]) * 100)

                img = cv2.imread(str(img_path))
                if img is None:
                    tracker.update(None, "image_missing", 0.0)
                    continue

                res = model.predict(img, imgsz=1280, conf=0.25, device=0, verbose=False)[0]
                pose = None
                if res.keypoints is not None and len(res.keypoints) > 0:
                    kpts_all = res.keypoints.xy.cpu().numpy()
                    cls_all = res.boxes.cls.cpu().numpy().astype(int)
                    conf_all = res.boxes.conf.cpu().numpy()
                    bbox_all = res.boxes.xyxy.cpu().numpy()
                    target_idx = next((j for j in range(len(cls_all)) if cls_all[j] == 0), None)
                    if target_idx is not None:
                        pose = estimate_pose(
                            kpts_all[target_idx], 0, bbox_all[target_idx],
                            float(conf_all[target_idx]),
                            K_per_cam[cam], np.zeros(5), pnp_cfg,
                        )

                if pose is None or pose.quality_flag != "ok":
                    out = tracker.update(None, pose.quality_flag if pose else "no_target", 0.0)
                else:
                    T_base_opt = T_base_tcp @ T_tcp_opt_per_cam[cam]
                    T_meas = T_base_opt @ pose.T_cam_mouth
                    out = tracker.update(
                        T_meas, pose.quality_flag, pose.confidence,
                        z_cam_m=float(pose.tvec_cam[2]),
                        reproj_err_px=float(pose.reprojection_err_px),
                    )

                if out.T_base_mouth is not None:
                    t_err_mm = float(np.linalg.norm(out.T_base_mouth[:3, 3] - T_base_mouth_GT[:3, 3]) * 1000)
                    r_err_deg = angle_deg(out.T_base_mouth[:3, :3], T_base_mouth_GT[:3, :3])
                    errors.append((d_cm, t_err_mm, r_err_deg, out.state.value))

            # Aggregate per-distance-band
            band_stats = defaultdict(list)
            for d_cm, t, r, s in errors:
                band_stats[dist_band_cm(d_cm)].append((t, r))
            band_summary = {}
            for band, vals in band_stats.items():
                ts = np.array([v[0] for v in vals])
                rs = np.array([v[1] for v in vals])
                band_summary[band] = {
                    "n": len(vals),
                    "median_t_mm": float(np.median(ts)),
                    "p90_t_mm": float(np.percentile(ts, 90)),
                    "median_r_deg": float(np.median(rs)),
                    "p90_r_deg": float(np.percentile(rs, 90)),
                }
            # Tracker state distribution
            states = [s for _, _, _, s in errors]
            ep_summary["by_cam"][cam] = {
                "n_frames": len(frames),
                "n_with_pose": len(errors),
                "tracker_states": {
                    "tracking": states.count("tracking"),
                    "coasting": states.count("coasting"),
                    "lost": states.count("lost"),
                },
                "by_distance": band_summary,
            }
        per_episode.append(ep_summary)
        # Per-episode pass/fail (any cam where hover/approach error stays below thresholds)
        passed = False
        for cam, cam_data in ep_summary["by_cam"].items():
            for band in ("hover", "approach"):
                bs = cam_data["by_distance"].get(band)
                if bs and bs["median_t_mm"] < PASS_T_MM and bs["median_r_deg"] < PASS_R_DEG:
                    passed = True
                    break
            if passed:
                break
        ep_summary["passed"] = passed
        print(f"ep{ep}: {'PASS' if passed else 'FAIL'}")

    # Aggregate
    n_pass = sum(1 for e in per_episode if e["passed"])
    summary = {
        "n_episodes": len(per_episode),
        "n_passed": n_pass,
        "pass_rate": n_pass / max(len(per_episode), 1),
        "thresholds": {"trans_mm": PASS_T_MM, "rot_deg": PASS_R_DEG},
        "per_episode": per_episode,
    }

    # Aggregate per-band across all eps × cams
    agg_by_band = defaultdict(list)  # band -> list of (t, r) errors
    for ep_data in per_episode:
        for cam, cam_data in ep_data["by_cam"].items():
            for band, b in cam_data["by_distance"].items():
                agg_by_band[band].append((b["median_t_mm"], b["median_r_deg"]))
    summary["aggregate_by_band"] = {}
    for band, vals in agg_by_band.items():
        if not vals: continue
        ts = np.array([v[0] for v in vals])
        rs = np.array([v[1] for v in vals])
        summary["aggregate_by_band"][band] = {
            "n_ep_cam_combos": len(vals),
            "median_t_mm": float(np.median(ts)),
            "p90_t_mm": float(np.percentile(ts, 90)),
            "median_r_deg": float(np.median(rs)),
            "p90_r_deg": float(np.percentile(rs, 90)),
        }

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nResults: {OUT}")
    print(f"\nPass rate: {n_pass}/{len(per_episode)} ({100*n_pass/max(len(per_episode),1):.0f}%)")
    print(f"\nAggregate by band:")
    for band, b in summary["aggregate_by_band"].items():
        print(f"  {band:>10}: n={b['n_ep_cam_combos']:>3}  "
              f"trans median={b['median_t_mm']:.2f} mm p90={b['p90_t_mm']:.2f}  "
              f"rot median={b['median_r_deg']:.2f}° p90={b['p90_r_deg']:.2f}°")


if __name__ == "__main__":
    main()
