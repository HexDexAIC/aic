#!/usr/bin/env python3
"""Read ~/aic_user_annotations.json and compare against current detectors.

For each annotated frame:
  1. Run YOLO and classical detectors on the same image.
  2. Compute IoU between detector output and each user-labeled box.
  3. Determine which user-box (target / distractor / other) the
     detector matched best.
  4. Aggregate stats by trial + camera.

Outputs a structured report to stdout and writes a JSON summary to
~/aic_annotation_analysis.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))

from aic_example_policies.ros.port_detector import detect_port as detect_classical
from aic_example_policies.ros.port_detector_yolo import YoloPosePortDetector


ANNOTATIONS_FILE = Path.home() / "aic_user_annotations.json"
OUTPUT_FILE = Path.home() / "aic_annotation_analysis.json"


def iou(a, b):
    """xyxy IoU."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def det_to_xyxy(det):
    if det is None:
        return None
    if det.corners_xy is not None:
        xs = det.corners_xy[:, 0]
        ys = det.corners_xy[:, 1]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return [
        float(det.cx - det.width / 2),
        float(det.cy - det.height / 2),
        float(det.cx + det.width / 2),
        float(det.cy + det.height / 2),
    ]


def best_match(det_box, user_boxes):
    """Return (best_label, best_iou, best_idx)."""
    if det_box is None:
        return None, 0.0, -1
    best = (None, 0.0, -1)
    for i, b in enumerate(user_boxes):
        score = iou(det_box, b["bbox_xyxy"])
        if score > best[1]:
            best = (b.get("label"), score, i)
    return best


def main():
    if not ANNOTATIONS_FILE.exists():
        sys.exit(f"No annotations at {ANNOTATIONS_FILE}")
    annotations = json.loads(ANNOTATIONS_FILE.read_text())
    if not annotations:
        sys.exit("Annotations file is empty.")
    yolo = YoloPosePortDetector(conf=0.25)

    summary = {
        "total_frames": 0,
        "total_boxes": 0,
        "by_label": defaultdict(int),
        "by_trial_camera": defaultdict(lambda: {
            "frames": 0,
            "yolo_targets": 0,        # YOLO matched user 'target'
            "yolo_distractors": 0,    # YOLO matched user 'distractor'
            "yolo_other": 0,
            "yolo_no_match": 0,
            "yolo_no_det": 0,
            "yolo_target_iou": [],
            "classical_targets": 0,
            "classical_distractors": 0,
            "classical_other": 0,
            "classical_no_match": 0,
            "classical_no_det": 0,
            "classical_target_iou": [],
        }),
        "frames": [],
    }

    print(f"Reading {len(annotations)} annotated frames\n")

    for img_path, rec in sorted(annotations.items()):
        boxes = rec.get("boxes")
        if not boxes and rec.get("user_bbox_xyxy"):
            boxes = [{"label": "target", "bbox_xyxy": rec["user_bbox_xyxy"],
                     "comment": rec.get("comment", "")}]
        if not boxes:
            continue

        port_type = rec.get("port_type", "sfp")
        trial = rec.get("trial", "?")
        cam = rec.get("camera", "?")
        key = f"{trial}/{cam}"

        if not Path(img_path).exists():
            continue
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        yolo_det = yolo.detect(img_rgb, port_type) if (yolo and yolo.available) else None
        cls_det = detect_classical(img_rgb, port_type, refine=True)

        yolo_box = det_to_xyxy(yolo_det)
        cls_box = det_to_xyxy(cls_det)

        y_label, y_iou, y_idx = best_match(yolo_box, boxes)
        c_label, c_iou, c_idx = best_match(cls_box, boxes)

        def all_label_ious(det_box):
            r = {}
            if det_box is None:
                return r
            for b in boxes:
                lab = b.get("label", "?")
                r[lab] = max(r.get(lab, 0), iou(det_box, b["bbox_xyxy"]))
            return r

        def all_label_dists(det_box):
            """Center distance from det to each user box centroid (in pixels)."""
            r = {}
            if det_box is None:
                return r
            dx = (det_box[0] + det_box[2]) / 2
            dy = (det_box[1] + det_box[3]) / 2
            for b in boxes:
                lab = b.get("label", "?")
                bb = b["bbox_xyxy"]
                bcx = (bb[0] + bb[2]) / 2
                bcy = (bb[1] + bb[3]) / 2
                d = ((dx - bcx) ** 2 + (dy - bcy) ** 2) ** 0.5
                if lab not in r or d < r[lab]:
                    r[lab] = float(d)
            return r

        y_ious = all_label_ious(yolo_box)
        c_ious = all_label_ious(cls_box)
        y_dists = all_label_dists(yolo_box)
        c_dists = all_label_dists(cls_box)

        # Per-frame log
        print(f"{trial} / {cam} / frame {rec.get('frame_idx','?')}")
        print(f"  user boxes: {[b['label'] for b in boxes]}")
        if yolo_det is None:
            print("  YOLO: NO DET")
        else:
            iou_str = ", ".join(f"{lab}={v:.2f}" for lab, v in y_ious.items())
            dist_str = ", ".join(f"{lab}={d:.0f}px" for lab, d in y_dists.items())
            print(f"  YOLO IoU vs user: {iou_str};   ctr-dist: {dist_str}")
        if cls_det is None:
            print("  classical: NO DET")
        else:
            iou_str = ", ".join(f"{lab}={v:.2f}" for lab, v in c_ious.items())
            dist_str = ", ".join(f"{lab}={d:.0f}px" for lab, d in c_dists.items())
            print(f"  classical IoU vs user: {iou_str};   ctr-dist: {dist_str}")
        print()

        # Aggregate
        summary["total_frames"] += 1
        for b in boxes:
            summary["by_label"][b.get("label", "?")] += 1
        summary["total_boxes"] += len(boxes)

        s = summary["by_trial_camera"][key]
        s["frames"] += 1
        if yolo_det is None:
            s["yolo_no_det"] += 1
        elif y_iou < 0.05:
            s["yolo_no_match"] += 1
        else:
            s[f"yolo_{y_label}s" if y_label != "other" else "yolo_other"] += 1
            if y_label == "target":
                s["yolo_target_iou"].append(y_iou)
        if cls_det is None:
            s["classical_no_det"] += 1
        elif c_iou < 0.05:
            s["classical_no_match"] += 1
        else:
            s[f"classical_{c_label}s" if c_label != "other" else "classical_other"] += 1
            if c_label == "target":
                s["classical_target_iou"].append(c_iou)

        summary["frames"].append({
            "image_path": img_path,
            "trial": trial,
            "camera": cam,
            "user_boxes": boxes,
            "yolo": {"box_xyxy": yolo_box, "matched_label": y_label, "iou": y_iou} if yolo_det else None,
            "classical": {"box_xyxy": cls_box, "matched_label": c_label, "iou": c_iou} if cls_det else None,
        })

    # Print summary
    print("=" * 70)
    print(f"SUMMARY: {summary['total_frames']} frames, {summary['total_boxes']} boxes total")
    print(f"By label: {dict(summary['by_label'])}")
    print()
    print(f"{'trial/cam':<24}  {'YOLO target/distract/none/nodet':<40}  {'classical target/distract/none/nodet'}")
    for key, s in sorted(summary["by_trial_camera"].items()):
        y_iou_med = float(np.median(s["yolo_target_iou"])) if s["yolo_target_iou"] else 0.0
        c_iou_med = float(np.median(s["classical_target_iou"])) if s["classical_target_iou"] else 0.0
        print(f"  {key:<22}  Y: T={s['yolo_targets']} D={s['yolo_distractors']} O={s['yolo_other']} NoMatch={s['yolo_no_match']} NoDet={s['yolo_no_det']} (T-IoU med={y_iou_med:.2f})")
        print(f"  {' ':<22}  C: T={s['classical_targets']} D={s['classical_distractors']} O={s['classical_other']} NoMatch={s['classical_no_match']} NoDet={s['classical_no_det']} (T-IoU med={c_iou_med:.2f})")

    # Convert defaultdicts to plain dicts for JSON
    summary["by_label"] = dict(summary["by_label"])
    summary["by_trial_camera"] = {k: dict(v) for k, v in summary["by_trial_camera"].items()}
    OUTPUT_FILE.write_text(json.dumps(summary, indent=2, default=list))
    print(f"\nFull report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
