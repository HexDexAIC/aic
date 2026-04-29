#!/usr/bin/env python3
"""Generate representative landmark prediction overlays across quality bands:
  BEST    — frames where predicted corners almost exactly match GT
  GOOD    — typical performance (median range)
  WORST   — frames already in failure_overlays — copy a few here too

For each, render: image + GT corners (green) + predicted corners (red)
+ class label + per-corner pixel error annotation.
"""
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
TEST_LABELS_DIR = Path.home() / "aic_yolo_v1" / "labels" / "test"
TEST_IMAGES_DIR = Path.home() / "aic_yolo_v1" / "images" / "test"
OUT = Path("/mnt/c/Users/Dell/aic_v1_pred_samples")
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*.jpg"):
    f.unlink()

# Limit to a manageable subset for speed
random.seed(42)
all_labels = sorted(TEST_LABELS_DIR.glob("*.txt"))
sample = random.sample(all_labels, 800)  # eval 800, then bucket

model = YOLO(str(WEIGHTS))


def parse_gt(lbl_path, w_img, h_img):
    objects = []
    for line in lbl_path.read_text().strip().split("\n"):
        if not line.strip(): continue
        parts = line.split()
        cls = int(parts[0])
        kpts = []
        for k in range(5):
            kx = float(parts[5 + k*3]) * w_img
            ky = float(parts[5 + k*3 + 1]) * h_img
            kpts.append((kx, ky))
        objects.append({"cls": cls, "kpts": np.array(kpts, dtype=np.float32)})
    return objects


per_frame_quality = []
for lbl_path in sample:
    img_path = TEST_IMAGES_DIR / (lbl_path.stem + ".jpg")
    if not img_path.exists(): continue
    img = cv2.imread(str(img_path))
    if img is None: continue
    h_img, w_img = img.shape[:2]

    gt_objs = parse_gt(lbl_path, w_img, h_img)
    res = model.predict(str(img_path), imgsz=1280, conf=0.25, device=0, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0: continue

    pred_kpts = res.keypoints.xy.cpu().numpy()
    pred_cls = res.boxes.cls.cpu().numpy().astype(int)
    pred_objs = [{"cls": int(pred_cls[i]), "kpts": pred_kpts[i]} for i in range(len(pred_kpts))]

    # Match per-class
    all_errs = []
    matches = []  # (gt_obj, pred_obj)
    for cls in (0, 1):
        gts = [g for g in gt_objs if g["cls"] == cls]
        preds = [p for p in pred_objs if p["cls"] == cls]
        if not gts or not preds: continue
        gt_c = np.array([g["kpts"][4] for g in gts])
        pred_c = np.array([p["kpts"][4] for p in preds])
        cost = np.linalg.norm(gt_c[:, None, :] - pred_c[None, :, :], axis=2)
        rr, cc = linear_sum_assignment(cost)
        for ri, ci in zip(rr, cc):
            if cost[ri, ci] > 100: continue
            errs = np.linalg.norm(gts[ri]["kpts"] - preds[ci]["kpts"], axis=1)
            all_errs.extend(errs.tolist())
            matches.append((gts[ri], preds[ci]))

    if not all_errs: continue
    median_err = float(np.median(all_errs))
    max_err = float(max(all_errs))
    per_frame_quality.append({
        "stem": lbl_path.stem, "img_path": str(img_path),
        "median_err": median_err, "max_err": max_err,
        "matches": matches,
    })

per_frame_quality.sort(key=lambda x: x["median_err"])
print(f"Evaluated {len(per_frame_quality)} frames")


def draw_overlay(frame_q, label_prefix):
    img = cv2.imread(frame_q["img_path"])
    for gt, pred in frame_q["matches"]:
        # GT in GREEN
        for j in range(5):
            p = tuple(gt["kpts"][j].astype(int))
            cv2.circle(img, p, 7, (0, 255, 0), -1)
            cv2.putText(img, str(j), (p[0]+8, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        # PRED in RED
        for j in range(5):
            p = tuple(pred["kpts"][j].astype(int))
            cv2.circle(img, p, 7, (0, 0, 255), -1)
        # Lines connecting GT→pred
        for j in range(5):
            cv2.line(img, tuple(gt["kpts"][j].astype(int)),
                      tuple(pred["kpts"][j].astype(int)), (0, 165, 255), 2)
        # Class label
        cls_name = "TARGET" if gt["cls"] == 0 else "DISTR"
        cv2.putText(img, cls_name, tuple(gt["kpts"][0].astype(int) + [0, -15]),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    # Crop around port
    pts = np.vstack([m[0]["kpts"] for m in frame_q["matches"]] +
                     [m[1]["kpts"] for m in frame_q["matches"]])
    cx, cy = pts.mean(axis=0)
    half = 200
    x0 = max(0, int(cx - half)); x1 = min(img.shape[1], int(cx + half))
    y0 = max(0, int(cy - half)); y1 = min(img.shape[0], int(cy + half))
    crop = img[y0:y1, x0:x1]
    big = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_NEAREST)
    cv2.putText(big, f"{label_prefix}: {frame_q['stem']}", (10, 25),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(big, f"median_err={frame_q['median_err']:.2f}px max={frame_q['max_err']:.1f}px",
                 (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
    cv2.putText(big, "GREEN=GT  RED=pred  ORANGE=offset",
                 (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)
    return big


# 5 best, 5 worst (within scanned), 5 median
n = len(per_frame_quality)
best5 = per_frame_quality[:5]
worst5 = per_frame_quality[-5:]
median5 = per_frame_quality[n//2 - 2:n//2 + 3]

for k, fq in enumerate(best5):
    cv2.imwrite(str(OUT / f"best_{k+1}_{fq['stem']}_med{int(fq['median_err']*10):03d}.jpg"),
                 draw_overlay(fq, f"BEST #{k+1}"))
for k, fq in enumerate(median5):
    cv2.imwrite(str(OUT / f"median_{k+1}_{fq['stem']}_med{int(fq['median_err']*10):03d}.jpg"),
                 draw_overlay(fq, f"MEDIAN #{k+1}"))
for k, fq in enumerate(worst5):
    cv2.imwrite(str(OUT / f"worst_{k+1}_{fq['stem']}_med{int(fq['median_err']*10):03d}.jpg"),
                 draw_overlay(fq, f"WORST #{k+1}"))

print(f"Saved 15 samples to {OUT}")
print("\nbest medians (px):", [round(f["median_err"], 2) for f in best5])
print("median medians (px):", [round(f["median_err"], 2) for f in median5])
print("worst medians (px):", [round(f["median_err"], 2) for f in worst5])
