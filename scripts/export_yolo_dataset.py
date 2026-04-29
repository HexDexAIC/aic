#!/usr/bin/env python3
"""Export YOLO keypoint dataset for v1 SFP entry-mouth detection.

For every (episode, frame, camera) tuple satisfying the v1 plan filters:
  - Project sfp_target entry mouth (canonical + dz=-45.8mm) → 5 landmarks + bbox
  - Project sfp_distractor entry mouth (target + (-23.2mm, 0, 0) + dz=-45.8mm)
  - Apply visibility flags (2 in-frame / 1 projectable-off-screen)
  - Skip frame if target center is off-screen
  - Apply 6cm insertion-phase filter
  - Apply 5x frame subsample
  - Save JPEG image + YOLO-format label

Output:
  $OUT_DIR/
    images/{train,val,test}/ep{E}_fr{F}_{cam}.jpg
    labels/{train,val,test}/ep{E}_fr{F}_{cam}.txt
    data.yaml  (Ultralytics config)

Run:
  pixi run python scripts/export_yolo_dataset.py [--limit-eps 5]
  pixi run python scripts/export_yolo_dataset.py            # full export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import (
    K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R,
)

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
CALIB_PATH = Path.home() / "aic_visible_mouth_calib.json"
DEFAULT_OUT = Path.home() / "aic_yolo_v1"

# v1 plan parameters
SUBSAMPLE_STRIDE = 5             # every 5th frame
INSERTION_FILTER_M = 0.06        # drop frames where ||TCP - port|| < 6cm
EVAL_TRAIN_END = 160             # train: 0..159
EVAL_VAL_END = 200               # val: 160..199
                                  # test: 200..299

CAMS = ["left", "center", "right"]
CLASSES = {"sfp_target": 0, "sfp_distractor": 1}

# Image resolution for export (native)
IMG_W, IMG_H = 1152, 1024


def split_for_ep(ep):
    if ep < EVAL_TRAIN_END:
        return "train"
    elif ep < EVAL_VAL_END:
        return "val"
    else:
        return "test"


def project_landmarks(T_base_mouth, T_base_tcp, K, T_tcp_opt, w, h):
    """Returns (5, 2) array of [u, v] for 4 corners + center, or None if behind cam."""
    pts_local = np.array([
        [+w/2, +h/2, 0, 1],
        [+w/2, -h/2, 0, 1],
        [-w/2, -h/2, 0, 1],
        [-w/2, +h/2, 0, 1],
        [0, 0, 0, 1],
    ]).T
    T_opt = np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T_base_mouth
    pts3 = T_opt @ pts_local
    Z = pts3[2]
    if (Z <= 0).any():
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx_p, cy_p = K[0, 2], K[1, 2]
    u = fx * pts3[0] / Z + cx_p
    v = fy * pts3[1] / Z + cy_p
    return np.stack([u, v], axis=1)


def visibility(uv, w_img, h_img):
    """2 if in-frame, 1 if off-screen but projectable."""
    u, v = float(uv[0]), float(uv[1])
    return 2 if (0 <= u < w_img and 0 <= v < h_img) else 1


def yolo_label_line(class_id, landmarks, w_img, h_img):
    """Format one YOLO keypoint label line:
       class cx cy w h kpt0_x kpt0_y kpt0_v ... kpt4_x kpt4_y kpt4_v
    All normalized to [0, 1].
    """
    corners = landmarks[:4]
    bx_min = corners[:, 0].min(); bx_max = corners[:, 0].max()
    by_min = corners[:, 1].min(); by_max = corners[:, 1].max()
    cx = (bx_min + bx_max) / 2 / w_img
    cy = (by_min + by_max) / 2 / h_img
    bw = (bx_max - bx_min) / w_img
    bh = (by_max - by_min) / h_img
    parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for i in range(5):
        u, v = float(landmarks[i, 0]), float(landmarks[i, 1])
        vis = visibility((u, v), w_img, h_img)
        parts += [f"{u/w_img:.6f}", f"{v/h_img:.6f}", str(vis)]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit-eps", type=int, default=0,
                     help="if >0, only export this many episodes (sanity check)")
    args = ap.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    calib = json.loads(CALIB_PATH.read_text())
    co = calib["T_canonical_to_visible_mouth"]
    td = calib["T_target_to_distractor"]
    rect = calib["rectangle_at_mouth"]
    SLOT_W = rect["width_mm"] / 1000
    SLOT_H = rect["height_mm"] / 1000
    T_co_mouth = np.eye(4); T_co_mouth[:3, 3] = [co["dx_mm"]/1000, co["dy_mm"]/1000, co["dz_mm"]/1000]
    T_target_to_distractor = np.eye(4); T_target_to_distractor[:3, 3] = [
        td["dx_mm"]/1000, td["dy_mm"]/1000, td["dz_mm"]/1000]
    print(f"Calibration loaded: T_co_mouth dz={co['dz_mm']}mm, T_target_to_distractor dx={td['dx_mm']}mm")

    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    successful_eps = sorted(int(ep) for ep in gt_pose.keys())
    if args.limit_eps:
        successful_eps = successful_eps[:args.limit_eps]
    print(f"Exporting {len(successful_eps)} successful episodes")

    # Stats counters
    counts = {"train": 0, "val": 0, "test": 0}
    counts_by_cam = {c: 0 for c in CAMS}
    skipped_filter = 0
    skipped_offscreen = 0
    skipped_behind = 0

    # Process each parquet file once; iterate frames from there
    pq_files = sorted((ROOT / "data" / "chunk-000").glob("*.parquet"))

    for pf in pq_files:
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        file_idx = int(pf.stem.replace("file-", ""))

        # Pre-open per-cam video capture for this file
        caps = {}
        for cam in CAMS:
            video_path = ROOT / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{file_idx:03d}.mp4"
            if not video_path.exists():
                continue
            caps[cam] = cv2.VideoCapture(str(video_path))

        # Iterate frames row-by-row in original parquet order (matches mp4 frame order)
        for row_idx in range(len(df)):
            ep = int(df.iloc[row_idx]["episode_index"])
            fr = int(df.iloc[row_idx]["frame_index"])
            if ep not in successful_eps:
                continue
            if fr % SUBSAMPLE_STRIDE != 0:
                continue

            # Compute target port pose for this episode
            T_settled = np.eye(4)
            T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
            T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
            T_base_target_canon = T_settled @ T_TCP_plug
            T_base_target_mouth = T_base_target_canon @ T_co_mouth
            T_base_distractor_mouth = T_base_target_canon @ T_target_to_distractor @ T_co_mouth

            # State for this frame
            state = df.iloc[row_idx]["observation.state"]
            T_base_tcp = state_to_T(np.asarray(state))

            # Insertion-phase filter
            tcp_to_port = float(np.linalg.norm(np.asarray(state)[0:3] - T_base_target_canon[:3, 3]))
            if tcp_to_port < INSERTION_FILTER_M:
                skipped_filter += 1
                continue

            split = split_for_ep(ep)

            for cam in CAMS:
                if cam not in caps:
                    continue
                K = K_PER_CAM[cam]
                T_tcp_opt = T_TCP_OPT[cam]

                target_pts = project_landmarks(T_base_target_mouth, T_base_tcp, K, T_tcp_opt, SLOT_W, SLOT_H)
                distractor_pts = project_landmarks(T_base_distractor_mouth, T_base_tcp, K, T_tcp_opt, SLOT_W, SLOT_H)

                if target_pts is None and distractor_pts is None:
                    skipped_behind += 1
                    continue

                # Drop frame if target's center is off-screen
                target_drop = (target_pts is None) or visibility(target_pts[4], IMG_W, IMG_H) != 2
                if target_drop and (distractor_pts is None or visibility(distractor_pts[4], IMG_W, IMG_H) != 2):
                    skipped_offscreen += 1
                    continue

                # Read video frame
                caps[cam].set(cv2.CAP_PROP_POS_FRAMES, row_idx)
                ok, img = caps[cam].read()
                if not ok or img is None:
                    continue

                # Build label
                lines = []
                if target_pts is not None and visibility(target_pts[4], IMG_W, IMG_H) == 2:
                    lines.append(yolo_label_line(CLASSES["sfp_target"], target_pts, IMG_W, IMG_H))
                if distractor_pts is not None and visibility(distractor_pts[4], IMG_W, IMG_H) == 2:
                    lines.append(yolo_label_line(CLASSES["sfp_distractor"], distractor_pts, IMG_W, IMG_H))
                if not lines:
                    skipped_offscreen += 1
                    continue

                stem = f"ep{ep:03d}_fr{fr:05d}_{cam}"
                img_path = OUT / "images" / split / f"{stem}.jpg"
                lbl_path = OUT / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                lbl_path.write_text("\n".join(lines))
                counts[split] += 1
                counts_by_cam[cam] += 1

        for cap in caps.values():
            cap.release()
        # Progress
        n_done = sum(counts.values())
        print(f"  {pf.name}: cumulative {n_done} labels exported "
              f"(train={counts['train']} val={counts['val']} test={counts['test']})")

    print(f"\nDone. Total labels: {sum(counts.values())}")
    print(f"  train: {counts['train']}")
    print(f"  val:   {counts['val']}")
    print(f"  test:  {counts['test']}")
    print(f"  by cam: {counts_by_cam}")
    print(f"\nFiltered out:")
    print(f"  insertion-phase (<{INSERTION_FILTER_M*100:.0f}cm): {skipped_filter}")
    print(f"  off-screen center: {skipped_offscreen}")
    print(f"  behind camera: {skipped_behind}")

    # Write Ultralytics data.yaml
    data_yaml = (
        f"path: {OUT.absolute()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"kpt_shape: [5, 3]   # 5 landmarks (4 corners + center), each (x, y, visibility)\n"
        f"flip_idx: [1, 0, 3, 2, 4]   # horizontal flip permutes corners 0<->1, 2<->3\n"
        f"\n"
        f"names:\n"
    )
    for name, cls_id in sorted(CLASSES.items(), key=lambda x: x[1]):
        data_yaml += f"  {cls_id}: {name}\n"
    (OUT / "data.yaml").write_text(data_yaml)
    print(f"\nWrote {OUT / 'data.yaml'}")
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
