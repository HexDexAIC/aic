#!/usr/bin/env python3
"""Tracked YOLO-pose accuracy eval — walk full episodes with SE3Tracker.

Run with the pixi env:
    /home/hariharan/ws_aic/src/aic/.pixi/envs/default/bin/python \\
        run_tracked_eval.py [--n-episodes 10] [--phases 0,0.1,0.25,0.5,0.7,0.85,0.95,0.99]

Difference vs run_pose_eval.py
------------------------------
The single-frame harness invokes YOLO+PnP independently per (ep, fr, cam).
This one walks each episode frame-by-frame from t=0, mirroring how
VisionInsert_v3 uses the SE3Tracker:

  fresh tracker per episode
  for each frame:
      best = argmax over cams of (conf × bbox_area_px) of pose_v2.estimate_pose
      tracker.update(T_meas=best.T_base_mouth, quality, conf, z_cam, reproj)
      record tracker state + smoothed T_base_mouth
  at each requested phase-pct, sample tracker output and compare to GT

This captures:
  - lazy-seed gating: tracker stays LOST until z_cam < 0.15 m or reproj < 0.5 px
  - outlier rejection: bimodal IPPE rotations get hard-rejected (>90°) or
    rotation-frozen (>k_r σ_r)
  - smoothing: alpha_t=0.4 EMA, alpha_r=0.3 slerp
  - coasting: up to 10 frames of dropped detections before LOST

The scene set is identical to run_pose_eval.py so results are directly
comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    import rerun as rr
    _RR_AVAILABLE = True
except ImportError:
    _RR_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "aic_example_policies"))
from aic_example_policies.perception.port_pose_v2.pnp import (  # noqa: E402
    PnPConfig, estimate_pose,
)
from aic_example_policies.perception.port_pose_v2.tracker import (  # noqa: E402
    SE3Tracker, TrackerConfig, TrackerState,
)

DATASET_ROOT = Path("/home/hariharan/aic_results/aic-sfp-500-pr")
WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
CAM_CALIB = Path.home() / "aic_cam_tcp_offsets.json"

IMG_W, IMG_H = 1152, 1024
TARGET_CLS_ID = 0


# ---------- math helpers (matches run_pose_eval.py) ----------

def rot6_to_R(a1, a2):
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def state_to_T(state):
    T = np.eye(4)
    T[:3, :3] = rot6_to_R(state[3:6], state[6:9])
    T[:3, 3] = state[0:3]
    return T


def port_gt_to_T(port):
    T = np.eye(4)
    T[:3, :3] = rot6_to_R(port[3:6], port[6:9])
    T[:3, 3] = port[0:3]
    return T


PORT_LINK_TO_MOUTH_DZ = -0.0458


def gt_link_to_mouth(T_base_port_link):
    T_offset = np.eye(4)
    T_offset[2, 3] = PORT_LINK_TO_MOUTH_DZ
    return T_base_port_link @ T_offset


def angle_between_R(R1, R2):
    R_rel = R1.T @ R2
    cos_th = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0))))


# ---------- YOLO + PnP per cam ----------

def run_yolo(model, img_chw_float, conf_min=0.25):
    img_rgb = (img_chw_float.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(img_bgr, imgsz=1280, conf=conf_min, verbose=False)
    r = results[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return None
    kpts_all = r.keypoints.xy.cpu().numpy()
    cls_all = r.boxes.cls.cpu().numpy().astype(int)
    conf_all = r.boxes.conf.cpu().numpy()
    bbox_all = r.boxes.xyxy.cpu().numpy()
    target_mask = (cls_all == TARGET_CLS_ID)
    if not target_mask.any():
        return None
    target_idx = np.where(target_mask)[0]
    j = target_idx[conf_all[target_idx].argmax()]
    return {
        "kpts_uv": kpts_all[j],
        "bbox_xyxy": bbox_all[j],
        "confidence": float(conf_all[j]),
        "cls_id": int(cls_all[j]),
    }


# For visualizing GT corners projected back into the image — same canonical
# order as pnp.py's OBJECT_POINTS_4 (port-mouth frame, z=0):
SLOT_W_M = 0.0137
SLOT_H_M = 0.0085
MOUTH_CORNERS = np.array([
    [+SLOT_W_M / 2, +SLOT_H_M / 2, 0.0],
    [+SLOT_W_M / 2, -SLOT_H_M / 2, 0.0],
    [-SLOT_W_M / 2, -SLOT_H_M / 2, 0.0],
    [-SLOT_W_M / 2, +SLOT_H_M / 2, 0.0],
    [0.0, 0.0, 0.0],          # center
], dtype=np.float64)


def project_points(K, T_cam_world, points_world):
    """Project Nx3 world points into image plane via T_cam_world (4x4) and K."""
    n = points_world.shape[0]
    p_h = np.hstack([points_world, np.ones((n, 1))])
    p_cam = (T_cam_world @ p_h.T).T[:, :3]
    z = np.maximum(p_cam[:, 2:3], 1e-6)
    p_norm = p_cam[:, :2] / z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.stack([fx * p_norm[:, 0] + cx, fy * p_norm[:, 1] + cy], axis=1)


def detect_best_cam(model, sample, K_per_cam, T_tcp_opt_per_cam, T_base_tcp, pnp_cfg,
                     return_per_cam_dets=False):
    """Run all 3 cams, return best (T_base_mouth_meas, pose_estimate, cam) or None.

    If `return_per_cam_dets` is True, also returns the raw YOLO detection
    per cam (kpts_uv, bbox_xyxy, confidence, cls_id, pose) for visualization.
    """
    best = None
    best_score = 0.0
    per_cam_diag = {}
    per_cam_dets = {}
    for cam in ("left", "center", "right"):
        img = sample[f"observation.images.{cam}"]
        det = run_yolo(model, img)
        if det is None:
            per_cam_diag[cam] = {"detected": False}
            continue
        K = K_per_cam[cam]
        pose = estimate_pose(
            keypoints_uv=det["kpts_uv"],
            cls_id=det["cls_id"],
            bbox_xyxy=det["bbox_xyxy"],
            confidence=det["confidence"],
            K=K, dist_coeffs=np.zeros(5), cfg=pnp_cfg,
        )
        per_cam_diag[cam] = {
            "detected": True,
            "quality_flag": pose.quality_flag,
            "conf": float(pose.confidence),
            "z_cam_m": float(pose.tvec_cam[2]),
            "reproj_err_px": float(pose.reprojection_err_px),
        }
        if return_per_cam_dets:
            per_cam_dets[cam] = {"det": det, "pose": pose}
        if pose.quality_flag != "ok":
            continue
        T_base_opt = T_base_tcp @ T_tcp_opt_per_cam[cam]
        T_base_mouth = T_base_opt @ pose.T_cam_mouth
        score = float(pose.confidence * pose.bbox_area_px)
        if score > best_score:
            best_score = score
            best = (T_base_mouth, pose, cam)
    if return_per_cam_dets:
        return best, per_cam_diag, per_cam_dets
    return best, per_cam_diag


# ---------- per-episode walk ----------

def walk_episode(model, ds, ep_idx, ep_meta, K_per_cam, T_tcp_opt_per_cam,
                  pnp_cfg, tracker_cfg, rerun_log=False, rerun_stride=1):
    """Walk one episode start-to-end. Returns list of per-frame records.

    If rerun_log=True, also emits rerun spans for each frame (transforms,
    images, keypoints, scalars). rerun_stride controls which frames get
    image+keypoint logs (scalars are always logged).
    """
    start = int(ep_meta[ep_idx]["dataset_from_index"])
    L = int(ep_meta[ep_idx]["length"])

    tracker = SE3Tracker(tracker_cfg)
    records = []

    if rerun_log:
        # One-shot static logs per episode: pinhole intrinsics + view conventions.
        for cam in ("left", "center", "right"):
            K = K_per_cam[cam]
            rr.log(f"world/tcp/cam_{cam}",
                    rr.Pinhole(image_from_camera=K, resolution=[IMG_W, IMG_H]),
                    static=True)
            T_tcp_opt = T_tcp_opt_per_cam[cam]
            rr.log(f"world/tcp/cam_{cam}",
                    rr.Transform3D(translation=T_tcp_opt[:3, 3],
                                    mat3x3=T_tcp_opt[:3, :3]),
                    static=True)
        # Episode-constant: GT port mouth (computed below per-frame, but really
        # constant — log it once for clarity)
        first_sample = ds[start]
        port_gt_link0 = port_gt_to_T(first_sample["observation.port_pose_gt"].numpy())
        T_gt_mouth0 = gt_link_to_mouth(port_gt_link0)
        rr.log("world/port_gt",
                rr.Transform3D(translation=T_gt_mouth0[:3, 3],
                                mat3x3=T_gt_mouth0[:3, :3]),
                static=True)
        rr.log("world/port_gt/mouth",
                rr.Boxes3D(half_sizes=[(SLOT_W_M / 2, SLOT_H_M / 2, 0.001)],
                            colors=[(0, 200, 0)]),
                static=True)

    for fr in range(L):
        sample = ds[start + fr]
        state = sample["observation.state"].numpy()
        port_gt_link = port_gt_to_T(sample["observation.port_pose_gt"].numpy())
        T_gt_mouth = gt_link_to_mouth(port_gt_link)
        T_base_tcp = state_to_T(state)
        d_to_port_m = float(np.linalg.norm(state[:3] - T_gt_mouth[:3, 3]))

        if rerun_log:
            rr.set_time("frame", sequence=fr)

        if rerun_log:
            best, _, per_cam_dets = detect_best_cam(
                model, sample, K_per_cam, T_tcp_opt_per_cam,
                T_base_tcp, pnp_cfg, return_per_cam_dets=True)
        else:
            best, _ = detect_best_cam(model, sample, K_per_cam, T_tcp_opt_per_cam,
                                        T_base_tcp, pnp_cfg)

        if best is None:
            tout = tracker.update(T_meas=None, quality="all_rejected", confidence=0.0)
            cam_used = None
            best_z = None
            best_reproj = None
        else:
            T_meas, pose_est, cam_used = best
            best_z = float(pose_est.tvec_cam[2])
            best_reproj = float(pose_est.reprojection_err_px)
            tout = tracker.update(
                T_meas=T_meas, quality="ok",
                confidence=float(pose_est.confidence),
                z_cam_m=best_z,
                reproj_err_px=best_reproj,
            )

        rec = {
            "ep": ep_idx, "fr": fr,
            "state_xyz": state[:3].tolist(),
            "gt_xyz": T_gt_mouth[:3, 3].tolist(),
            "d_to_port_m": d_to_port_m,
            "best_cam": cam_used,
            "best_z_cam_m": best_z,
            "best_reproj_px": best_reproj,
            "tracker_state": tout.state.value,
            "tracker_is_tracked": tout.is_tracked,
            "tracker_coast": tout.coast_count,
            "tracker_sigma_t_m": tout.sigma_t_m,
            "tracker_sigma_r_rad": tout.sigma_r_rad,
            "T_pred": tout.T_base_mouth.tolist() if tout.T_base_mouth is not None else None,
            "T_gt": T_gt_mouth.tolist(),
        }

        if tout.T_base_mouth is not None:
            t_err_mm = float(np.linalg.norm(tout.T_base_mouth[:3, 3] - T_gt_mouth[:3, 3]) * 1000)
            r_err_deg = angle_between_R(tout.T_base_mouth[:3, :3], T_gt_mouth[:3, :3])
            ax_err_mm = (tout.T_base_mouth[:3, 3] - T_gt_mouth[:3, 3]) * 1000
            rec.update({
                "t_err_mm": t_err_mm,
                "r_err_deg": r_err_deg,
                "t_err_x_mm": float(ax_err_mm[0]),
                "t_err_y_mm": float(ax_err_mm[1]),
                "t_err_z_mm": float(ax_err_mm[2]),
            })

        records.append(rec)

        if rerun_log:
            log_frame_to_rerun(
                fr=fr, sample=sample, T_base_tcp=T_base_tcp, T_gt_mouth=T_gt_mouth,
                tracker_out=tout, K_per_cam=K_per_cam,
                T_tcp_opt_per_cam=T_tcp_opt_per_cam,
                per_cam_dets=per_cam_dets, rec=rec,
                stride=rerun_stride,
            )

    return records


# ---------- rerun logging ----------

def log_frame_to_rerun(*, fr, sample, T_base_tcp, T_gt_mouth, tracker_out,
                        K_per_cam, T_tcp_opt_per_cam, per_cam_dets, rec, stride):
    """Log one frame's worth of state to rerun. Scalars always; images on stride."""
    # ----- transforms -----
    rr.log("world/tcp",
            rr.Transform3D(translation=T_base_tcp[:3, 3],
                            mat3x3=T_base_tcp[:3, :3]))

    if tracker_out.T_base_mouth is not None:
        T_pred = tracker_out.T_base_mouth
        rr.log("world/port_pred",
                rr.Transform3D(translation=T_pred[:3, 3],
                                mat3x3=T_pred[:3, :3]))
        rr.log("world/port_pred/mouth",
                rr.Boxes3D(half_sizes=[(SLOT_W_M / 2, SLOT_H_M / 2, 0.001)],
                            colors=[(220, 100, 0)]))
        # Connecting line from pred to GT — visualizes the residual
        rr.log("world/err_line",
                rr.LineStrips3D([np.stack([T_pred[:3, 3], T_gt_mouth[:3, 3]])],
                                 colors=[(255, 0, 0)]))
    else:
        rr.log("world/port_pred", rr.Clear(recursive=True))
        rr.log("world/err_line", rr.Clear(recursive=True))

    # ----- per-cam image + detections -----
    if fr % stride == 0:
        for cam in ("left", "center", "right"):
            img_chw = sample[f"observation.images.{cam}"]
            img_rgb = (img_chw.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            rr.log(f"world/tcp/cam_{cam}/image", rr.Image(img_rgb))

            # YOLO predicted keypoints (5×2) and bbox
            det_info = per_cam_dets.get(cam)
            if det_info is not None:
                kpts = det_info["det"]["kpts_uv"]
                bbox = det_info["det"]["bbox_xyxy"]
                pose = det_info["pose"]
                conf = det_info["det"]["confidence"]
                rr.log(f"world/tcp/cam_{cam}/yolo_kpts",
                        rr.Points2D(kpts, colors=[(0, 200, 255)] * len(kpts), radii=4))
                rr.log(f"world/tcp/cam_{cam}/yolo_bbox",
                        rr.Boxes2D(array=np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]]),
                                    array_format=rr.Box2DFormat.XYXY,
                                    labels=[f"{conf:.2f}/{pose.quality_flag}"],
                                    colors=[(0, 200, 255)]))
            else:
                rr.log(f"world/tcp/cam_{cam}/yolo_kpts", rr.Clear(recursive=False))
                rr.log(f"world/tcp/cam_{cam}/yolo_bbox", rr.Clear(recursive=False))

            # GT mouth corners projected back into image — green dots
            T_world_optical = T_base_tcp @ T_tcp_opt_per_cam[cam]
            T_optical_world = np.linalg.inv(T_world_optical)
            mouth_pts_world = (T_gt_mouth @ np.hstack(
                [MOUTH_CORNERS, np.ones((5, 1))]).T).T[:, :3]
            uv_gt = project_points(K_per_cam[cam], T_optical_world, mouth_pts_world)
            rr.log(f"world/tcp/cam_{cam}/gt_corners_proj",
                    rr.Points2D(uv_gt, colors=[(0, 255, 0)] * 5, radii=3))

    # ----- scalars -----
    rr.log("metrics/d_to_port_m", rr.Scalars(rec["d_to_port_m"]))
    if rec.get("t_err_mm") is not None:
        rr.log("metrics/t_err_mm", rr.Scalars(rec["t_err_mm"]))
        rr.log("metrics/r_err_deg", rr.Scalars(rec["r_err_deg"]))
        rr.log("metrics/t_err_x_mm", rr.Scalars(rec["t_err_x_mm"]))
        rr.log("metrics/t_err_y_mm", rr.Scalars(rec["t_err_y_mm"]))
        rr.log("metrics/t_err_z_mm", rr.Scalars(rec["t_err_z_mm"]))
    if rec.get("best_z_cam_m") is not None:
        rr.log("metrics/best_z_cam_m", rr.Scalars(rec["best_z_cam_m"]))
    if rec.get("best_reproj_px") is not None:
        rr.log("metrics/best_reproj_px", rr.Scalars(rec["best_reproj_px"]))

    # tracker state encoded numerically: tracking=2, coasting=1, lost=0
    state_num = {"tracking": 2, "coasting": 1, "lost": 0}.get(tracker_out.state.value, -1)
    rr.log("tracker/state", rr.Scalars(state_num))
    rr.log("tracker/coast", rr.Scalars(tracker_out.coast_count))
    rr.log("tracker/sigma_t_mm", rr.Scalars(tracker_out.sigma_t_m * 1000))
    rr.log("tracker/sigma_r_deg", rr.Scalars(np.degrees(tracker_out.sigma_r_rad)))


def phase_label(d_to_port_m: float) -> str:
    if d_to_port_m > 0.18: return "init"
    if d_to_port_m > 0.12: return "approach"
    if d_to_port_m > 0.05: return "descent"
    return "insertion"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--phases", default="0,0.1,0.25,0.5,0.7,0.85,0.95,0.99")
    ap.add_argument("--out", default=str(Path.home() / "aic_runs" / "yolo_tracked_eval.json"))
    ap.add_argument("--seed", type=int, default=42, help="rng seed for episode picks")
    ap.add_argument("--rerun-out", default=None,
                     help="path to .rrd output (e.g. ~/aic_runs/yolo_tracked.rrd). "
                          "If unset, no rerun logging.")
    ap.add_argument("--rerun-spawn", action="store_true",
                     help="spawn a live rerun viewer instead of saving to .rrd")
    ap.add_argument("--rerun-stride", type=int, default=1,
                     help="log image+keypoints every Nth frame (scalars always); "
                          "default 1 = every frame")
    ap.add_argument("--rerun-eps", type=int, default=None,
                     help="if set, only the first N selected episodes get rerun logging "
                          "(file-size cap). Defaults to all selected episodes.")
    args = ap.parse_args()

    phases = [float(x) for x in args.phases.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    pnp_cfg = PnPConfig()
    tracker_cfg = TrackerConfig()
    print(f"Tracker config: alpha_t={tracker_cfg.alpha_t}, alpha_r={tracker_cfg.alpha_r}, "
          f"seed_z_max_m={tracker_cfg.seed_z_max_m}, seed_reproj_max_px={tracker_cfg.seed_reproj_max_px}, "
          f"coast_max={tracker_cfg.coast_max_frames}")

    print(f"Loading cam calib {CAM_CALIB}")
    cam_offs = json.loads(CAM_CALIB.read_text())
    K_per_cam = {c: np.array(v["K"]).reshape(3, 3) for c, v in cam_offs.items()}
    T_tcp_opt_per_cam = {c: np.array(v["T_tcp_optical"]) for c, v in cam_offs.items()}

    print(f"Loading dataset {DATASET_ROOT}")
    ds = LeRobotDataset(repo_id="HexDexAIC/aic-sfp-500-pr",
                         root=str(DATASET_ROOT), revision="main")
    ep_meta = ds.meta.episodes
    print(f"  total ep: {ds.num_episodes}, total frames: {ds.num_frames}")

    # Match the seed/episode selection used by run_pose_eval.py for direct comparison
    rng = np.random.default_rng(args.seed)
    ep_indices = rng.choice(ds.num_episodes, size=args.n_episodes, replace=False)
    ep_indices = sorted(ep_indices.tolist())
    print(f"Sampled episodes: {ep_indices}")
    print(f"Phases: {phases}")

    # Rerun setup
    rerun_active = (args.rerun_out is not None) or args.rerun_spawn
    if rerun_active and not _RR_AVAILABLE:
        print("WARNING: rerun requested but `rerun-sdk` not importable; disabling.")
        rerun_active = False
    rerun_eps_set = set(ep_indices[:args.rerun_eps]) if args.rerun_eps else set(ep_indices)
    if rerun_active:
        rr.init("yolo_pose_eval", spawn=args.rerun_spawn)
        if args.rerun_out and not args.rerun_spawn:
            out_rrd = Path(args.rerun_out).expanduser()
            out_rrd.parent.mkdir(parents=True, exist_ok=True)
            rr.save(str(out_rrd))
            print(f"Rerun: saving to {out_rrd}")
        elif args.rerun_spawn:
            print("Rerun: live viewer spawned")
        # rerun blueprint: image-per-cam + 3D + scalar timeseries
        # (lightweight blueprint - rerun viewer can be reorganized at view time)

    all_records = {}   # ep -> list of frame records
    t0 = time.time()
    for k, ep in enumerate(ep_indices):
        L = int(ep_meta[ep]["length"])
        do_rerun = rerun_active and ep in rerun_eps_set
        print(f"\n[{k+1}/{len(ep_indices)}] ep{ep} ({L} frames){' [rerun]' if do_rerun else ''}...")
        if do_rerun:
            # Group rerun timeline by episode for easy navigation in viewer
            rr.log(f"episode_marker", rr.TextLog(f"BEGIN ep{ep}"), static=False)
            rr.set_time("episode", sequence=ep)
        recs = walk_episode(model, ds, ep, ep_meta, K_per_cam, T_tcp_opt_per_cam,
                              pnp_cfg, tracker_cfg,
                              rerun_log=do_rerun, rerun_stride=args.rerun_stride)
        all_records[ep] = recs
        # Quick per-episode summary
        seeded_at = next((r["fr"] for r in recs if r["tracker_is_tracked"]), None)
        n_tracked = sum(r["tracker_is_tracked"] for r in recs)
        n_coast = sum(r["tracker_state"] == "coasting" for r in recs)
        n_lost = sum(r["tracker_state"] == "lost" for r in recs)
        elapsed = time.time() - t0
        print(f"  seed@fr={seeded_at}, tracked={n_tracked}, coast={n_coast}, lost={n_lost}"
              f"  ({elapsed:.1f}s elapsed)")

    # ----- Phase-aligned sampling -----
    phase_samples = []
    for ep, recs in all_records.items():
        L = len(recs)
        for ph in phases:
            fr = int(round(ph * (L - 1)))
            r = recs[fr]
            r2 = dict(r)
            r2["phase_pct"] = float(ph)
            r2["phase"] = phase_label(r["d_to_port_m"])
            phase_samples.append(r2)

    out_payload = {
        "weights": str(WEIGHTS),
        "cam_calib": str(CAM_CALIB),
        "n_episodes": args.n_episodes,
        "phases": phases,
        "tracker_cfg": {
            "alpha_t": tracker_cfg.alpha_t, "alpha_r": tracker_cfg.alpha_r,
            "coast_max_frames": tracker_cfg.coast_max_frames,
            "seed_z_max_m": tracker_cfg.seed_z_max_m,
            "seed_reproj_max_px": tracker_cfg.seed_reproj_max_px,
            "outlier_k_t": tracker_cfg.outlier_k_t,
            "outlier_k_r": tracker_cfg.outlier_k_r,
            "rot_hard_reject_rad": tracker_cfg.rot_hard_reject_rad,
        },
        "phase_samples": phase_samples,
        "all_records": {str(ep): recs for ep, recs in all_records.items()},
    }
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(f"\nSaved {sum(len(v) for v in all_records.values())} frame records "
          f"({len(phase_samples)} phase-aligned) to {out_path}")

    # ----- Phase-aligned summary -----
    print("\n=== Tracked phase-aligned summary ===")
    print(f"{'phase':<12} {'n':>4} {'tracked%':>9} "
          f"{'med_t':>7} {'p90_t':>7} {'med_r':>7} {'p90_r':>7} "
          f"{'med_z':>7} {'med_d':>6}")
    by_phase = defaultdict(list)
    for r in phase_samples:
        by_phase[r["phase"]].append(r)
    for phase in ("init", "approach", "descent", "insertion"):
        rs = by_phase.get(phase, [])
        if not rs:
            continue
        n = len(rs)
        tracked = [r for r in rs if r["tracker_is_tracked"]]
        if tracked:
            ts = np.array([r["t_err_mm"] for r in tracked])
            rrs = np.array([r["r_err_deg"] for r in tracked])
            zs = np.array([abs(r["t_err_z_mm"]) for r in tracked])
            dds = np.array([r["d_to_port_m"] for r in tracked])
            print(f"{phase:<12} {n:>4d} {100*len(tracked)/n:>8.0f}% "
                  f"{np.median(ts):>7.2f} {np.percentile(ts, 90):>7.2f} "
                  f"{np.median(rrs):>7.2f} {np.percentile(rrs, 90):>7.2f} "
                  f"{np.median(zs):>7.2f} {np.median(dds)*100:>5.1f}cm")
        else:
            print(f"{phase:<12} {n:>4d}     0%      --      --      --      --      --      --")

    # ----- Time-to-first-tracked (frames) per episode -----
    print("\n=== Time-to-first-tracked-pose per episode ===")
    for ep in ep_indices:
        recs = all_records[ep]
        seeded_at = next((r["fr"] for r in recs if r["tracker_is_tracked"]), None)
        d_at_seed = recs[seeded_at]["d_to_port_m"] if seeded_at is not None else None
        if seeded_at is None:
            print(f"  ep{ep}: NEVER tracked")
        else:
            print(f"  ep{ep}: fr{seeded_at} (d_to_port={d_at_seed*100:.1f}cm)")


if __name__ == "__main__":
    main()
