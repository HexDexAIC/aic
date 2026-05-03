#!/usr/bin/env python3
"""Evaluate v1 trained YOLO keypoint model on held-out test split (eps 200-299).

Two evaluations:
  1. Ultralytics built-in: bbox/pose mAP, precision, recall
  2. Custom landmark pixel-error metric (the v1 success criterion):
       median, mean per-corner pixel error
       % landmarks within 1 / 2 / 4 px
       per-class breakdown
       worst N failures saved as overlays
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
DATA_YAML = Path.home() / "aic_yolo_v1" / "data.yaml"
TEST_LABELS_DIR = Path.home() / "aic_yolo_v1" / "labels" / "test"
TEST_IMAGES_DIR = Path.home() / "aic_yolo_v1" / "images" / "test"
OUT_DIR = Path.home() / "aic_runs" / "v1_h100_results" / "test_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FAILURE_DIR = OUT_DIR / "failures"
FAILURE_DIR.mkdir(exist_ok=True)


def main():
    print(f"Loading {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    # 1. Built-in test eval
    print("\n=== Built-in test-split eval ===")
    metrics = model.val(data=str(DATA_YAML), split="test",
                        imgsz=1280, batch=16, device=0,
                        save_json=False, plots=True,
                        project=str(OUT_DIR.parent), name="test_eval",
                        exist_ok=True, verbose=True)
    summary = {
        "test_mAP50_box":     float(metrics.box.map50),
        "test_mAP50_95_box":  float(metrics.box.map),
        "test_precision_box": float(metrics.box.mp),
        "test_recall_box":    float(metrics.box.mr),
        "test_mAP50_pose":     float(metrics.pose.map50),
        "test_mAP50_95_pose":  float(metrics.pose.map),
    }
    print("\nBuilt-in summary:", json.dumps(summary, indent=2))

    # 2. Custom landmark pixel error
    print("\n=== Custom landmark pixel-error metric ===")
    label_files = sorted(TEST_LABELS_DIR.glob("*.txt"))
    print(f"Test labels: {len(label_files)}")

    # Per-class accumulators
    err_by_class = {0: [], 1: []}  # class_id -> list of per-corner-distance arrays
    matched_count = {0: 0, 1: 0}
    gt_count = {0: 0, 1: 0}
    n_no_pred = 0
    n_extra_pred = 0

    # Load YOLO model once for inference
    model.predict  # warm load

    failure_candidates = []  # (max_corner_err, image_path, gt_dict, pred_dict)

    for i, lbl_path in enumerate(label_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(label_files)}")
        img_path = TEST_IMAGES_DIR / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        # Parse GT
        gt_objects = []  # list of {cls, kpts: (5, 2)}
        for line in lbl_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            kpts = []
            for k in range(5):
                kx = float(parts[5 + k*3]) * w_img
                ky = float(parts[5 + k*3 + 1]) * h_img
                vis = int(parts[5 + k*3 + 2])
                kpts.append((kx, ky, vis))
            gt_objects.append({"cls": cls, "kpts": np.array(kpts, dtype=np.float32)})
            gt_count[cls] += 1

        # Predict
        results = model.predict(str(img_path), imgsz=1280, conf=0.25,
                                  device=0, verbose=False)
        r = results[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            n_no_pred += len(gt_objects)
            continue

        # Pred objects
        pred_objects = []
        if r.keypoints is not None:
            kpts_xy = r.keypoints.xy.cpu().numpy() if torch.is_tensor(r.keypoints.xy) else r.keypoints.xy
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for j in range(len(kpts_xy)):
                pred_objects.append({"cls": int(cls_ids[j]),
                                       "kpts": kpts_xy[j]})

        # Match GT to pred per class via centroid bipartite matching
        for cls in (0, 1):
            gt_list = [g for g in gt_objects if g["cls"] == cls]
            pred_list = [p for p in pred_objects if p["cls"] == cls]
            if not gt_list or not pred_list:
                if gt_list:
                    n_no_pred += len(gt_list)
                if pred_list:
                    n_extra_pred += len(pred_list)
                continue
            gt_centers = np.array([g["kpts"][4, :2] for g in gt_list])
            pred_centers = np.array([p["kpts"][4, :2] for p in pred_list])
            cost = np.linalg.norm(gt_centers[:, None, :] - pred_centers[None, :, :], axis=2)
            rr, cc = linear_sum_assignment(cost)
            for ri, ci in zip(rr, cc):
                if cost[ri, ci] > 100:  # too far apart, not a match
                    n_no_pred += 1
                    continue
                gt_kpts = gt_list[ri]["kpts"][:, :2]
                pred_kpts = pred_list[ci]["kpts"]
                per_corner = np.linalg.norm(gt_kpts - pred_kpts, axis=1)
                err_by_class[cls].append(per_corner)
                matched_count[cls] += 1
                # Track failure candidates
                max_err = float(per_corner.max())
                if max_err > 5.0:
                    failure_candidates.append({
                        "max_err": max_err, "img_path": str(img_path),
                        "gt": gt_kpts.tolist(), "pred": pred_kpts.tolist(),
                        "cls": cls, "stem": lbl_path.stem,
                    })

    # Aggregate
    print(f"\n  Total matched: target={matched_count[0]} distractor={matched_count[1]}")
    print(f"  Total GT:       target={gt_count[0]} distractor={gt_count[1]}")
    print(f"  No-pred:        {n_no_pred}")
    print(f"  Extra-pred:     {n_extra_pred}")

    detection_recall = {
        "sfp_target":     matched_count[0] / max(gt_count[0], 1),
        "sfp_distractor": matched_count[1] / max(gt_count[1], 1),
    }

    landmark_err = {}
    for cls, name in [(0, "sfp_target"), (1, "sfp_distractor")]:
        if not err_by_class[cls]:
            continue
        errs = np.concatenate(err_by_class[cls])  # all per-corner distances
        landmark_err[name] = {
            "n_corners": int(len(errs)),
            "median_px": float(np.median(errs)),
            "mean_px":   float(errs.mean()),
            "p90_px":    float(np.percentile(errs, 90)),
            "p95_px":    float(np.percentile(errs, 95)),
            "p99_px":    float(np.percentile(errs, 99)),
            "frac_within_1px":  float((errs < 1).mean()),
            "frac_within_2px":  float((errs < 2).mean()),
            "frac_within_4px":  float((errs < 4).mean()),
            "frac_within_8px":  float((errs < 8).mean()),
        }

    overall = {
        "detection_recall": detection_recall,
        "landmark_pixel_error": landmark_err,
        "builtin_metrics": summary,
    }
    out_path = OUT_DIR / "v1_test_eval.json"
    out_path.write_text(json.dumps(overall, indent=2))
    print(f"\nResults saved: {out_path}")
    print(json.dumps(overall, indent=2))

    # Save top-12 failures
    failure_candidates.sort(key=lambda x: -x["max_err"])
    print(f"\n=== Top failures (worst max-corner-err) ===")
    for k, fc in enumerate(failure_candidates[:12]):
        img = cv2.imread(fc["img_path"])
        gt = np.array(fc["gt"])
        pred = np.array(fc["pred"])
        for j in range(5):
            cv2.circle(img, tuple(gt[j].astype(int)), 6, (0, 255, 0), -1)
            cv2.circle(img, tuple(pred[j].astype(int)), 6, (0, 0, 255), -1)
            cv2.line(img, tuple(gt[j].astype(int)), tuple(pred[j].astype(int)),
                      (0, 165, 255), 2)
        # Crop around GT center
        cx, cy = gt[4]
        half = 100
        x0 = max(0, int(cx - half)); x1 = min(img.shape[1], int(cx + half))
        y0 = max(0, int(cy - half)); y1 = min(img.shape[0], int(cy + half))
        crop = img[y0:y1, x0:x1]
        crop_big = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop_big, f"max_err={fc['max_err']:.1f}px {fc['stem']}",
                     (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        out = FAILURE_DIR / f"failure_{k:02d}_{int(fc['max_err'])}px_{fc['stem']}.jpg"
        cv2.imwrite(str(out), crop_big)
        print(f"  {k+1:2d}. {fc['stem']} max_err={fc['max_err']:.1f}px")

    print(f"\nFailure overlays saved to: {FAILURE_DIR}")


if __name__ == "__main__":
    main()
