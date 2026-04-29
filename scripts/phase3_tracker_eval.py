#!/usr/bin/env python3
"""Phase-3 tracker eval — three tests:
  1. Smoothness: tracked-pose jitter on a real episode trajectory
  2. Coasting recovery: drop random frames, check tracker maintains pose
  3. Outlier rejection: inject worst-case PnP outputs into the stream

For all three, GT pose is constant within an episode (port is static), so:
  smoothness = std deviation of tracked pose over time
  recovery = does state return to TRACKING within coast_max_frames?
  outlier reject = does corrupted measurement get refused?
"""
from __future__ import annotations

import json
import re
import sys
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
TEST_IMAGES = Path.home() / "aic_yolo_v1" / "images" / "test"
TEST_LABELS = Path.home() / "aic_yolo_v1" / "labels" / "test"
GT_POSE = Path.home() / "aic_gt_port_poses.json"
TCP_PLUG_OFFSET = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
CAM_OFFSETS = Path.home() / "aic_cam_tcp_offsets.json"
CALIB = Path.home() / "aic_visible_mouth_calib.json"

OUT = Path.home() / "aic_runs" / "v1_h100_results" / "test_eval" / "phase3_tracker_eval.json"


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


def load_episode_frames(ep, cam):
    """Load all (frame_idx, image, T_base_tcp) for a test episode/camera."""
    pat = TEST_LABELS.glob(f"ep{ep:03d}_*_{cam}.txt")
    frames = []
    state_lookup = {}
    for pf in sorted((DATA_ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for _, row in df.iterrows():
            if int(row["episode_index"]) != ep:
                continue
            state_lookup[int(row["frame_index"])] = np.asarray(row["observation.state"])

    for lbl in sorted(pat):
        m = re.match(rf"ep{ep:03d}_fr(\d+)_{cam}", lbl.stem)
        if not m: continue
        fr = int(m.group(1))
        if fr not in state_lookup: continue
        img_path = TEST_IMAGES / (lbl.stem + ".jpg")
        if not img_path.exists(): continue
        frames.append({
            "fr": fr, "img_path": str(img_path),
            "T_base_tcp": state_to_T(state_lookup[fr]),
        })
    return frames


def angle_deg(R1, R2):
    R = R1.T @ R2
    cos_th = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos_th, -1, 1))))


def main():
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

    pnp_cfg = PnPConfig()
    tracker_cfg = TrackerConfig()

    # === TEST 1: Smoothness on real episode trajectory ===
    # Pick 3 episodes from test split, run through tracker, measure jitter
    print("=== Test 1: Smoothness ===")
    smoothness_results = {}
    for ep in (210, 240, 285):
        for cam in ("center", "right", "left"):
            frames = load_episode_frames(ep, cam)
            if len(frames) < 5:
                continue

            # GT mouth pose (constant per ep — port is static)
            T_settled = np.eye(4)
            T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
            T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
            T_base_target_canon = T_settled @ T_TCP_plug
            T_base_mouth_GT = T_base_target_canon @ T_co_mouth

            tracker = SE3Tracker(tracker_cfg)
            raw_translations = []   # PnP measurements
            tracked_translations = []  # smoothed
            for f in frames:
                img = cv2.imread(f["img_path"])
                if img is None: continue
                res = model.predict(img, imgsz=1280, conf=0.25, device=0, verbose=False)[0]
                if res.keypoints is None or len(res.keypoints) == 0:
                    tracker.update(None, "no_detection")
                    continue
                # Take first sfp_target detection
                kpts_all = res.keypoints.xy.cpu().numpy()
                cls_all = res.boxes.cls.cpu().numpy().astype(int)
                conf_all = res.boxes.conf.cpu().numpy()
                bbox_all = res.boxes.xyxy.cpu().numpy()
                target_idx = next((j for j in range(len(cls_all)) if cls_all[j] == 0), None)
                if target_idx is None:
                    tracker.update(None, "no_target_class")
                    continue
                pose = estimate_pose(kpts_all[target_idx], 0, bbox_all[target_idx],
                                       float(conf_all[target_idx]),
                                       K_per_cam[cam], np.zeros(5), pnp_cfg)
                if pose.quality_flag != "ok":
                    tracker.update(None, pose.quality_flag)
                    continue
                T_base_opt = f["T_base_tcp"] @ T_tcp_opt_per_cam[cam]
                T_meas = T_base_opt @ pose.T_cam_mouth
                raw_translations.append(T_meas[:3, 3])
                out = tracker.update(T_meas, pose.quality_flag, pose.confidence)
                if out.T_base_mouth is not None:
                    tracked_translations.append(out.T_base_mouth[:3, 3])

            if len(raw_translations) < 5:
                continue
            raw = np.array(raw_translations)
            trk = np.array(tracked_translations)
            # Jitter = std deviation of position over the trajectory (port is static)
            raw_jitter = float(np.linalg.norm(raw.std(axis=0)) * 1000)  # mm
            trk_jitter = float(np.linalg.norm(trk.std(axis=0)) * 1000)
            # Bias (vs GT)
            raw_bias = float(np.linalg.norm(raw.mean(axis=0) - T_base_mouth_GT[:3, 3]) * 1000)
            trk_bias = float(np.linalg.norm(trk.mean(axis=0) - T_base_mouth_GT[:3, 3]) * 1000)
            smoothness_results[f"ep{ep}/{cam}"] = {
                "n_frames": len(frames), "n_accepted": len(raw),
                "raw_jitter_mm": raw_jitter, "tracked_jitter_mm": trk_jitter,
                "raw_bias_mm": raw_bias, "tracked_bias_mm": trk_bias,
                "jitter_reduction": (raw_jitter - trk_jitter) / max(raw_jitter, 1e-6),
            }
            print(f"  ep{ep}/{cam}: raw_jit={raw_jitter:.2f}mm  trk_jit={trk_jitter:.2f}mm  "
                   f"raw_bias={raw_bias:.2f}mm  trk_bias={trk_bias:.2f}mm")

    # === TEST 2: Forced-dropout coasting recovery ===
    print("\n=== Test 2: Coasting recovery ===")
    coast_results = {}
    ep, cam = 210, "center"
    frames = load_episode_frames(ep, cam)
    if len(frames) >= 10:
        # Run through with ALL frames first to seed tracker
        tracker = SE3Tracker(tracker_cfg)
        states_seen = []
        np.random.seed(42)
        # 30% random dropouts
        drop_mask = np.random.random(len(frames)) < 0.3
        for i, f in enumerate(frames):
            if drop_mask[i]:
                out = tracker.update(None, "forced_dropout", 0.0)
            else:
                img = cv2.imread(f["img_path"])
                res = model.predict(img, imgsz=1280, conf=0.25, device=0, verbose=False)[0]
                if res.keypoints is None or len(res.keypoints) == 0:
                    out = tracker.update(None, "no_detection", 0.0)
                else:
                    kpts_all = res.keypoints.xy.cpu().numpy()
                    cls_all = res.boxes.cls.cpu().numpy().astype(int)
                    conf_all = res.boxes.conf.cpu().numpy()
                    bbox_all = res.boxes.xyxy.cpu().numpy()
                    target_idx = next((j for j in range(len(cls_all)) if cls_all[j] == 0), None)
                    if target_idx is None:
                        out = tracker.update(None, "no_target", 0.0)
                    else:
                        pose = estimate_pose(kpts_all[target_idx], 0, bbox_all[target_idx],
                                              float(conf_all[target_idx]),
                                              K_per_cam[cam], np.zeros(5), pnp_cfg)
                        if pose.quality_flag != "ok":
                            out = tracker.update(None, pose.quality_flag, 0.0)
                        else:
                            T_base_opt = f["T_base_tcp"] @ T_tcp_opt_per_cam[cam]
                            T_meas = T_base_opt @ pose.T_cam_mouth
                            out = tracker.update(T_meas, "ok", pose.confidence)
            states_seen.append(out.state.value)

        # Tally state transitions
        n_tracking = states_seen.count("tracking")
        n_coasting = states_seen.count("coasting")
        n_lost = states_seen.count("lost")
        coast_results = {
            "ep": ep, "cam": cam,
            "n_frames": len(frames), "n_dropouts": int(drop_mask.sum()),
            "tracking_frames": n_tracking,
            "coasting_frames": n_coasting,
            "lost_frames": n_lost,
            "lost_rate": n_lost / len(frames),
        }
        print(f"  dropout 30%: TRACKING={n_tracking} COASTING={n_coasting} LOST={n_lost}")

    # === TEST 3: Outlier rejection ===
    print("\n=== Test 3: Outlier rejection ===")
    tracker = SE3Tracker(tracker_cfg)
    # Seed with valid pose
    T_seed = np.eye(4); T_seed[:3, 3] = [1.0, 1.0, 1.0]
    for _ in range(5):
        tracker.update(T_seed, "ok", 0.9)
    # Inject corrupted poses 50cm off
    rejected = 0
    for delta in (0.5, -0.4, 0.3, -0.5, 0.4):
        T_bad = np.eye(4)
        T_bad[:3, 3] = T_seed[:3, 3] + np.array([delta, 0, 0])
        out = tracker.update(T_bad, "ok", 0.9)
        if out.state == TrackerState.COASTING:
            rejected += 1
        elif out.state == TrackerState.TRACKING:
            tracker_drift = np.linalg.norm(out.T_base_mouth[:3, 3] - T_seed[:3, 3])
            print(f"  WARN: outlier accepted; drift = {tracker_drift:.3f}m")
    print(f"  outlier reject rate: {rejected}/5")

    # === Summary
    summary = {
        "smoothness": smoothness_results,
        "coasting_recovery": coast_results,
        "outlier_rejection": {"n_outliers_injected": 5, "n_rejected": rejected},
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nResults: {OUT}")


if __name__ == "__main__":
    main()
