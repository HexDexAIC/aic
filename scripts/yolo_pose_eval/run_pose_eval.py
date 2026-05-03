#!/usr/bin/env python3
"""Offline YOLO-pose accuracy harness — port-mouth 6D pose vs GT.

Run with the pixi env:
    /home/hariharan/ws_aic/src/aic/.pixi/envs/default/bin/python \\
        run_pose_eval.py [--n-episodes 8] [--phases 0,0.1,0.3,0.6,0.9]

What it does
------------
For each chosen (episode, frame_idx_pct) sample:
  1. Read 3 wrist-cam images + observation.state + observation.port_pose_gt
  2. Run YOLO (target class only) on each image
  3. PnP-IPPE on the predicted 4 corners → T_cam_mouth
  4. Compose T_base_mouth_pred = T_base_tcp @ T_tcp_optical[cam] @ T_cam_mouth
  5. Compute translation_err_mm and rotation_err_deg vs the GT in the dataset
  6. Bin by phase (init/approach/descent/insertion) × camera

The dataset is HexDexAIC/aic-sfp-500-pr (LeRobotDataset with port_pose_gt).
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

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "aic_example_policies"))
from aic_example_policies.perception.port_pose_v2.pnp import (  # noqa: E402
    PnPConfig, estimate_pose,
)

DATASET_ROOT = Path("/home/hariharan/aic_results/aic-sfp-500-pr")
WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
CAM_CALIB = Path.home() / "aic_cam_tcp_offsets.json"

IMG_W, IMG_H = 1152, 1024
TARGET_CLS_ID = 0  # 0=sfp_target


# ---------- math helpers ----------

def rot6_to_R(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Zhou-2019 6D rotation -> R (column stacking matches phase2_pnp_eval)."""
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def state_to_T(state: np.ndarray) -> np.ndarray:
    """observation.state[0:9] -> 4x4 T_base_tcp."""
    T = np.eye(4)
    T[:3, :3] = rot6_to_R(state[3:6], state[6:9])
    T[:3, 3] = state[0:3]
    return T


def port_gt_to_T(port: np.ndarray) -> np.ndarray:
    """observation.port_pose_gt[0:9] -> 4x4 T_base_port_link_GT.

    NOTE: this is the `sfp_port_X_link` frame (port BODY), not the mouth.
    Use port_link_to_mouth() to convert to the entry-mouth frame the PnP
    module reports.
    """
    T = np.eye(4)
    T[:3, :3] = rot6_to_R(port[3:6], port[6:9])
    T[:3, 3] = port[0:3]
    return T


# EMPIRICAL: estimated by inverting (T_world_visible_mouth from YOLO+PnP)
# against the GT port_link, across 55 detections from 10 episodes.
# Per-axis median: tx=-0.78mm, ty=+0.78mm, tz=-42.08mm (std 4.2mm).
# The SDF claims -45.8mm to sfp_port_X_link_entrance, but the YOLO model
# targets a different "visible mouth" definition ~3.7mm above that.
# See estimate_mouth_calib.py for the derivation.
PORT_LINK_TO_MOUTH_DZ = -0.04208
PORT_LINK_TO_MOUTH_DX = -0.00078
PORT_LINK_TO_MOUTH_DY = +0.00078


def gt_link_to_mouth(T_base_port_link: np.ndarray) -> np.ndarray:
    """Apply the port_link → visible-mouth offset (empirical, see comment)."""
    T_offset = np.eye(4)
    T_offset[0, 3] = PORT_LINK_TO_MOUTH_DX
    T_offset[1, 3] = PORT_LINK_TO_MOUTH_DY
    T_offset[2, 3] = PORT_LINK_TO_MOUTH_DZ
    return T_base_port_link @ T_offset


def angle_between_R(R1: np.ndarray, R2: np.ndarray) -> float:
    R_rel = R1.T @ R2
    cos_th = (np.trace(R_rel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0))))


# ---------- YOLO + PnP ----------

def run_yolo(model, img_chw_float: torch.Tensor, conf_min: float = 0.25):
    """img_chw_float in [0,1], shape (3, H, W). Returns dict or None."""
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
    # Pick highest-confidence target
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


# ---------- main eval ----------

def evaluate_one(model, sample, K_per_cam, T_tcp_opt_per_cam, pnp_cfg):
    """Return per-cam result dicts for one sample."""
    state = sample["observation.state"].numpy()
    port_gt = sample["observation.port_pose_gt"].numpy()

    T_base_tcp = state_to_T(state)
    T_gt_port_link = port_gt_to_T(port_gt)
    T_gt = gt_link_to_mouth(T_gt_port_link)        # port_link -> entrance/mouth

    d_to_port_m = float(np.linalg.norm(state[:3] - T_gt[:3, 3]))
    out = {
        "ep": int(sample["episode_index"].item()),
        "fr": int(sample["frame_index"].item()),
        "tcp_xyz": state[:3].tolist(),
        "port_gt_xyz": port_gt[:3].tolist(),
        "d_to_port_m": d_to_port_m,
        "per_cam": {},
    }

    for cam in ("left", "center", "right"):
        img = sample[f"observation.images.{cam}"]
        det = run_yolo(model, img)
        cam_res = {"detected": det is not None}
        if det is None:
            out["per_cam"][cam] = cam_res
            continue

        K = K_per_cam[cam]
        dist_coeffs = np.zeros(5)
        pose = estimate_pose(
            keypoints_uv=det["kpts_uv"],
            cls_id=det["cls_id"],
            bbox_xyxy=det["bbox_xyxy"],
            confidence=det["confidence"],
            K=K, dist_coeffs=dist_coeffs, cfg=pnp_cfg,
        )
        cam_res.update({
            "conf": det["confidence"],
            "bbox_area_px": float(pose.bbox_area_px),
            "reproj_err_px": float(pose.reprojection_err_px),
            "center_resid_px": float(pose.center_residual_px),
            "z_cam_m": float(pose.tvec_cam[2]),
            "quality_flag": pose.quality_flag,
        })
        if pose.quality_flag != "ok":
            out["per_cam"][cam] = cam_res
            continue

        # Compose to base_link
        T_base_opt = T_base_tcp @ T_tcp_opt_per_cam[cam]
        T_pred = T_base_opt @ pose.T_cam_mouth

        t_err_mm = float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]) * 1000)
        r_err_deg = angle_between_R(T_pred[:3, :3], T_gt[:3, :3])
        # Per-axis translation err (in BASE frame)
        ax_err_mm = (T_pred[:3, 3] - T_gt[:3, 3]) * 1000
        cam_res.update({
            "t_err_mm": t_err_mm,
            "r_err_deg": r_err_deg,
            "t_err_x_mm": float(ax_err_mm[0]),
            "t_err_y_mm": float(ax_err_mm[1]),
            "t_err_z_mm": float(ax_err_mm[2]),
            "pred_xyz": T_pred[:3, 3].tolist(),
        })
        out["per_cam"][cam] = cam_res

    return out


def phase_label(d_to_port_m: float) -> str:
    if d_to_port_m > 0.18:
        return "init"            # >18cm
    if d_to_port_m > 0.12:
        return "approach"        # 12-18cm
    if d_to_port_m > 0.05:
        return "descent"         # 5-12cm
    return "insertion"           # <5cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=8)
    ap.add_argument("--phases", default="0,0.1,0.3,0.5,0.7,0.9,0.99",
                    help="comma-separated frame fractions per episode")
    ap.add_argument("--out", default=str(Path.home() / "aic_runs" / "yolo_pose_eval.json"))
    args = ap.parse_args()

    phases = [float(x) for x in args.phases.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    pnp_cfg = PnPConfig()

    print(f"Loading cam calib {CAM_CALIB}")
    cam_offs = json.loads(CAM_CALIB.read_text())
    K_per_cam = {c: np.array(v["K"]).reshape(3, 3) for c, v in cam_offs.items()}
    T_tcp_opt_per_cam = {c: np.array(v["T_tcp_optical"]) for c, v in cam_offs.items()}

    print(f"Loading dataset {DATASET_ROOT}")
    ds = LeRobotDataset(repo_id="HexDexAIC/aic-sfp-500-pr",
                         root=str(DATASET_ROOT), revision="main")
    print(f"  total ep: {ds.num_episodes}, total frames: {ds.num_frames}")

    # Pick episodes evenly across 0..n-1; bias toward succeeded if available
    rng = np.random.default_rng(42)
    ep_indices = rng.choice(ds.num_episodes, size=args.n_episodes, replace=False)
    ep_indices = sorted(ep_indices.tolist())
    print(f"\nSampled episodes: {ep_indices}")
    print(f"Phases: {phases}")

    # Per-episode frame ranges from meta
    ep_frame_starts = ds.meta.episodes  # dict: ep_idx -> {"length": N}

    samples_to_eval = []
    for ep in ep_indices:
        ep_info = ep_frame_starts[ep]
        L = int(ep_info["length"])
        for ph in phases:
            fr = int(round(ph * (L - 1)))
            samples_to_eval.append((ep, fr))

    print(f"\nTotal samples to evaluate: {len(samples_to_eval)}")

    # Build a global dataset index: (ep, fr) -> dataset row idx
    # episodes[ep]["dataset_from_index"] gives starting global index for episode ep
    global_idx = {}
    for ep in ep_indices:
        start = int(ep_frame_starts[ep]["dataset_from_index"])
        ep_len = int(ep_frame_starts[ep]["length"])
        for fr in range(ep_len):
            global_idx[(ep, fr)] = start + fr

    results = []
    t0 = time.time()
    for k, (ep, fr) in enumerate(samples_to_eval):
        if (ep, fr) not in global_idx:
            print(f"  skip ({ep},{fr}): not found")
            continue
        sample = ds[global_idx[(ep, fr)]]
        res = evaluate_one(model, sample, K_per_cam, T_tcp_opt_per_cam, pnp_cfg)
        res["phase_pct"] = float(fr / max(int(ep_frame_starts[ep]["length"]) - 1, 1))
        res["phase"] = phase_label(res["d_to_port_m"])
        results.append(res)
        if (k+1) % 5 == 0 or k == len(samples_to_eval) - 1:
            elapsed = time.time() - t0
            print(f"  [{k+1}/{len(samples_to_eval)}] ep{ep} fr{fr} "
                  f"d={res['d_to_port_m']*100:.1f}cm phase={res['phase']} "
                  f"({elapsed:.1f}s)")

    out_path.write_text(json.dumps({
        "weights": str(WEIGHTS),
        "cam_calib": str(CAM_CALIB),
        "n_episodes": args.n_episodes,
        "phases": phases,
        "results": results,
    }, indent=2))
    print(f"\nSaved {len(results)} results to {out_path}")

    # ----- Aggregate summary -----
    print("\n=== Aggregate summary ===")
    by_phase_cam = defaultdict(list)
    for r in results:
        for cam, c in r["per_cam"].items():
            by_phase_cam[(r["phase"], cam)].append(c)

    print(f"{'phase':<12} {'cam':<7} {'n':>4} {'det%':>5} "
          f"{'ok%':>5} {'med_t':>7} {'p90_t':>7} "
          f"{'med_r':>7} {'p90_r':>7} {'med_d':>6}")
    for (phase, cam), cs in sorted(by_phase_cam.items()):
        n = len(cs)
        det = sum(c["detected"] for c in cs)
        ok = [c for c in cs if c.get("quality_flag") == "ok"]
        if ok:
            ts = np.array([c["t_err_mm"] for c in ok])
            rs = np.array([c["r_err_deg"] for c in ok])
            ds_ = np.array([c["z_cam_m"] for c in ok])
            print(f"{phase:<12} {cam:<7} {n:>4d} {100*det/n:>4.0f}% "
                  f"{100*len(ok)/n:>4.0f}% {np.median(ts):>7.2f} "
                  f"{np.percentile(ts, 90):>7.2f} {np.median(rs):>7.2f} "
                  f"{np.percentile(rs, 90):>7.2f} {np.median(ds_):>5.2f}m")
        else:
            print(f"{phase:<12} {cam:<7} {n:>4d} {100*det/n:>4.0f}% "
                  f"  0%   --      --      --      --      --")


if __name__ == "__main__":
    main()
