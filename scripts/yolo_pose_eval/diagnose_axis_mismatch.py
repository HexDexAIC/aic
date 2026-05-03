#!/usr/bin/env python3
"""Diagnose the GT-projection vs YOLO-keypoint pixel mismatch.

We saw ~12px median misalignment between:
  - GT mouth corners projected through K + TF (using pnp.py's OBJECT_POINTS_4)
  - YOLO predicted keypoints

That's bigger than expected (~1-2 px). Possible causes:
  H1. port_link X/Y axes swapped vs pnp.py's (W,H) → SLOT axes mismatch
  H2. SLOT_W_M / SLOT_H_M values are wrong
  H3. port_link → mouth offset direction wrong (sign or axis)
  H4. The YOLO-trained corners are not the SDF entrance corners — they may
      be the visible housing rim (~larger), or different keypoint definitions

Strategy: for each (ep, frame) where YOLO has a confident detection,
  - Compare: YOLO predicted 4 corner keypoints (in image)
  - vs:      4 corner predictions for SEVERAL hypotheses:
              A. canonical pnp.py + offset -0.0458Z  (current)
              B. swap X/Y in pnp.py corners
              C. swap SLOT_W and SLOT_H values
              D. flip mouth offset sign (+0.0458)
              E. set offset to 0
              F. larger outer dims (try 0.020 × 0.012, ~rim)
  - Pick the hypothesis with smallest pixel error.

This isolates whether the issue is rotational (B), dimensional (C/F), or
positional (D/E) — and tells us what to actually use.
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


def project(K, T_optical_world, points_world):
    n = points_world.shape[0]
    p_h = np.hstack([points_world, np.ones((n, 1))])
    p_cam = (T_optical_world @ p_h.T).T[:, :3]
    z = np.maximum(p_cam[:, 2:3], 1e-6)
    fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
    return np.stack([fx * p_cam[:, 0] / z[:, 0] + cx,
                       fy * p_cam[:, 1] / z[:, 0] + cy], axis=1)


def make_corners(W, H, swap_xy=False):
    """Return 4 corners in mouth frame, ordered as pnp.py does."""
    if swap_xy:
        W, H = H, W  # swap dimensions
    return np.array([
        [+W / 2, +H / 2, 0.0],
        [+W / 2, -H / 2, 0.0],
        [-W / 2, -H / 2, 0.0],
        [-W / 2, +H / 2, 0.0],
    ], dtype=np.float64)


def make_offset_T(dx=0, dy=0, dz=0):
    T = np.eye(4); T[:3, 3] = [dx, dy, dz]; return T


# Hypotheses
HYPS = {
    "A. baseline (W=13.7, H=8.5, dz=-0.0458)": dict(
        corners=make_corners(0.0137, 0.0085), T_off=make_offset_T(dz=-0.0458)),
    "B. swap_xy in corners":                  dict(
        corners=make_corners(0.0137, 0.0085, swap_xy=True), T_off=make_offset_T(dz=-0.0458)),
    "C. SLOT_W=8.5, SLOT_H=13.7 (swap dims)": dict(
        corners=make_corners(0.0085, 0.0137), T_off=make_offset_T(dz=-0.0458)),
    "D. flip offset sign (+0.0458)":          dict(
        corners=make_corners(0.0137, 0.0085), T_off=make_offset_T(dz=+0.0458)),
    "E. zero mouth offset":                   dict(
        corners=make_corners(0.0137, 0.0085), T_off=np.eye(4)),
    "F. larger 0.020 x 0.012 (housing rim)":  dict(
        corners=make_corners(0.020, 0.012), T_off=make_offset_T(dz=-0.0458)),
    "G. SC port spec (5x5)":                  dict(
        corners=make_corners(0.005, 0.005), T_off=make_offset_T(dz=-0.0458)),
    "H. swap_xy AND zero offset":             dict(
        corners=make_corners(0.0137, 0.0085, swap_xy=True), T_off=np.eye(4)),
    "I. swap_xy + flip Z offset":             dict(
        corners=make_corners(0.0137, 0.0085, swap_xy=True), T_off=make_offset_T(dz=+0.0458)),
}


def kp_match_err(kp_yolo, uv_pred):
    """Brute-force match best permutation: try all 4 cyclic + flipped perms."""
    perms = [
        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
        [0, 3, 2, 1], [1, 0, 3, 2], [2, 1, 0, 3], [3, 2, 1, 0],
    ]
    best = float("inf")
    for p in perms:
        d = np.linalg.norm(uv_pred - kp_yolo[list(p)], axis=1)
        m = float(np.median(d))
        if m < best: best = m
    return best


def main():
    ds = LeRobotDataset(repo_id="HexDexAIC/aic-sfp-500-pr",
                         root=str(DATASET_ROOT), revision="main")
    cam_offs = json.loads(CAM_CALIB.read_text())
    K_per_cam = {c: np.array(v["K"]).reshape(3, 3) for c, v in cam_offs.items()}
    T_tcp_opt_per_cam = {c: np.array(v["T_tcp_optical"]) for c, v in cam_offs.items()}
    model = YOLO(str(WEIGHTS))

    # Walk a few frames where YOLO will succeed (close range)
    test_eps = [42, 100, 214, 322]
    samples = []
    for ep in test_eps:
        start = int(ds.meta.episodes[ep]["dataset_from_index"])
        L = int(ds.meta.episodes[ep]["length"])
        # Pick descent frames where YOLO is reliable
        for fr in [int(L * 0.55), int(L * 0.7), int(L * 0.85)]:
            samples.append((ep, fr, start + fr))

    print(f"Testing {len(samples)} (ep, fr) samples × {len(HYPS)} hypotheses\n")

    by_hyp = {h: [] for h in HYPS}
    n_dets = 0
    for ep, fr, gid in samples:
        sample = ds[gid]
        state = sample["observation.state"].numpy()
        port_gt_link = port_gt_to_T(sample["observation.port_pose_gt"].numpy())
        T_base_tcp = state_to_T(state)

        # YOLO on center cam
        img_chw = sample["observation.images.center"]
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
        kp_yolo = r.keypoints.xy.cpu().numpy()[j][:4]
        n_dets += 1

        K = K_per_cam["center"]
        T_world_optical = T_base_tcp @ T_tcp_opt_per_cam["center"]
        T_optical_world = np.linalg.inv(T_world_optical)

        for hname, h in HYPS.items():
            T_base_mouth = port_gt_link @ h["T_off"]
            corners_world = (T_base_mouth @ np.hstack(
                [h["corners"], np.ones((4, 1))]).T).T[:, :3]
            uv_pred = project(K, T_optical_world, corners_world)
            err = kp_match_err(kp_yolo, uv_pred)
            by_hyp[hname].append(err)

    print(f"=== Median per-corner error (px) across {n_dets} detections ===")
    print(f"{'hypothesis':<55} {'med':>7} {'p90':>7} {'max':>7}")
    sorted_hyps = sorted(by_hyp.items(), key=lambda x: np.median(x[1]) if x[1] else 999)
    for h, errs in sorted_hyps:
        if not errs: continue
        a = np.array(errs)
        marker = "  ← BEST" if h == sorted_hyps[0][0] else ""
        print(f"{h:<55} {np.median(a):>7.2f} {np.percentile(a, 90):>7.2f} {a.max():>7.2f}{marker}")


if __name__ == "__main__":
    main()
