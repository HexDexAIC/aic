#!/usr/bin/env python3
"""Auto-label captured frames using ground-truth TF.

Block G of the AIC plan. Given a LoggingCheatCode dump (JSON + JPG triplets
per frame, with port_tf_base + camera intrinsics + camera_optical_tf_base)
projects the known 3D port mouth corners and forward-axis 'nail' point to
2D pixel coordinates in each camera. Writes YOLOv8-pose-format labels.

Output layout (YOLOv8 expects):
  <out>/
    images/
      <frame>_center.jpg  <frame>_left.jpg  <frame>_right.jpg
    labels/
      <frame>_center.txt  ...
    dataset.yaml

Each .txt has one line per port instance:
  class x_center y_center w h kp1x kp1y kp1v kp2x kp2y kp2v kp3x kp3y kp3v kp4x kp4y kp4v kp5x kp5y kp5v

All coordinates normalized [0,1]. kpV = visibility (2 = visible).

Classes:
  0 = sfp_port
  1 = sc_port

Keypoints (in port-link frame, i.e. with z=0 at the port mouth plane):
  kp0: corner (-W/2, -H/2, 0)  top-left of the mouth
  kp1: corner (+W/2, -H/2, 0)  top-right
  kp2: corner (+W/2, +H/2, 0)  bottom-right
  kp3: corner (-W/2, +H/2, 0)  bottom-left
  kp4: depth nail at (0, 0, +DEPTH) -- inside the port, helps disambiguate

Usage:
  pixi run python scripts/auto_label.py [<RUN_DIR>] [--out <OUT>]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


# Spec dimensions (meters). Use spec, not visible region — for YOLO training
# we want the network to learn the actual mouth geometry so PnP works.
SFP_MOUTH = (0.0137, 0.0085)  # (W, H)
SFP_DEPTH = 0.012             # nail depth into port
SC_MOUTH = (0.005, 0.005)     # ~5mm diameter -> use as square for keypoints
SC_DEPTH = 0.008

CLASS_IDS = {"sfp": 0, "sc": 1}


def keypoints_for(port_type: str):
    if port_type == "sfp":
        W, H = SFP_MOUTH
        D = SFP_DEPTH
    elif port_type == "sc":
        W, H = SC_MOUTH
        D = SC_DEPTH
    else:
        raise ValueError(port_type)
    return np.array([
        [-W / 2, -H / 2, 0],
        [+W / 2, -H / 2, 0],
        [+W / 2, +H / 2, 0],
        [-W / 2, +H / 2, 0],
        [0, 0, D],
    ], dtype=np.float64)


def quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def tf_dict_to_T(d):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(d["qw"], d["qx"], d["qy"], d["qz"])
    T[:3, 3] = [d["x"], d["y"], d["z"]]
    return T


def project(point_cam, K_flat):
    """Project camera-frame 3D point to 2D pixel."""
    if point_cam[2] <= 1e-3:
        return None
    fx = K_flat[0]; fy = K_flat[4]; cx = K_flat[2]; cy = K_flat[5]
    u = fx * point_cam[0] / point_cam[2] + cx
    v = fy * point_cam[1] / point_cam[2] + cy
    return float(u), float(v)


def label_one_view(cam_name, cam_tf_base, K_flat, port_tf_base, port_type, img_w, img_h):
    """Return YOLOv8-pose-format label string for one camera view, or None.

    None = port is not visible in this view (out of frame or behind camera).
    """
    if cam_tf_base is None or port_tf_base is None:
        return None
    T_base_cam = tf_dict_to_T(cam_tf_base)
    T_base_port = tf_dict_to_T(port_tf_base)
    T_cam_port = np.linalg.inv(T_base_cam) @ T_base_port

    kps_port = keypoints_for(port_type)
    kps_cam = (T_cam_port @ np.hstack([kps_port, np.ones((kps_port.shape[0], 1))]).T).T[:, :3]

    pixels = []
    for p in kps_cam:
        uv = project(p, K_flat)
        if uv is None:
            return None
        pixels.append(uv)
    pixels = np.array(pixels)

    # Reject if any keypoint is way out of frame.
    if (pixels[:, 0] < -img_w * 0.1).any() or (pixels[:, 0] > img_w * 1.1).any():
        return None
    if (pixels[:, 1] < -img_h * 0.1).any() or (pixels[:, 1] > img_h * 1.1).any():
        return None

    xs = pixels[:4, 0]
    ys = pixels[:4, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = 0.5 * (x_min + x_max) / img_w
    cy = 0.5 * (y_min + y_max) / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h
    if bw <= 0 or bh <= 0:
        return None

    cls = CLASS_IDS[port_type]
    parts = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for u, v in pixels:
        kx = u / img_w
        ky = v / img_h
        # visibility: 2 if inside frame, 1 if marginal, 0 if outside.
        vis = 2 if (0 <= kx <= 1 and 0 <= ky <= 1) else 1
        parts.append(f"{kx:.6f}")
        parts.append(f"{ky:.6f}")
        parts.append(str(vis))
    return " ".join(parts)


def process_run(run_dir: Path, out_dir: Path):
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_kept = 0
    for trial in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")):
        task_p = trial / "task.json"
        if not task_p.exists():
            continue
        task = json.loads(task_p.read_text())
        port_type = task["port_type"]
        if port_type not in CLASS_IDS:
            continue

        for json_p in sorted(trial.glob("[0-9]*.json")):
            rec = json.loads(json_p.read_text())
            n_total += 1
            for cam in ("center", "left", "right"):
                jpg = json_p.with_name(json_p.stem + f"_{cam}.jpg")
                if not jpg.exists():
                    continue
                cam_tf = rec.get(f"{cam}_cam_optical_tf_base")
                K = rec.get(f"K_{cam}")
                if cam_tf is None or K is None:
                    continue
                line = label_one_view(
                    cam, cam_tf, K, rec.get("port_tf_base"),
                    port_type, rec["image_w"], rec["image_h"],
                )
                if line is None:
                    continue
                # Copy image and write label.
                stem = f"{trial.name}_{json_p.stem}_{cam}"
                shutil.copy(jpg, images_dir / f"{stem}.jpg")
                (labels_dir / f"{stem}.txt").write_text(line + "\n")
                n_kept += 1

    # Write a YOLOv8 dataset.yaml.
    yaml_text = f"""# YOLOv8 pose dataset for AIC port detection
path: {out_dir.resolve()}
train: images
val: images
names:
  0: sfp_port
  1: sc_port
nc: 2
kpt_shape: [5, 3]
"""
    (out_dir / "dataset.yaml").write_text(yaml_text)

    print(f"Processed {n_total} frames, kept {n_kept} labeled views in {out_dir}")
    return n_kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--out", default=str(Path.home() / "aic_dataset"))
    args = ap.parse_args()

    base = Path.home() / "aic_logs"
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
    else:
        runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
        if not runs:
            sys.exit(f"No runs under {base}")
        run_dir = runs[-1]

    out_dir = Path(args.out).expanduser()
    print(f"Run: {run_dir}")
    print(f"Out: {out_dir}")
    process_run(run_dir, out_dir)


if __name__ == "__main__":
    main()
