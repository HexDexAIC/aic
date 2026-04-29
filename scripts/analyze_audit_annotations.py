#!/usr/bin/env python3
"""Analyze ~/aic_audit_annotations.json (LeRobot-dataset audit format)
against current YOLO + classical detector outputs.

Output: per-camera IoU & centroid-distance vs user target/distractor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))

from aic_example_policies.ros.port_detector import detect_port as detect_classical
from aic_example_policies.ros.port_detector_yolo import YoloPosePortDetector


ANN_PATH = Path.home() / "aic_audit_annotations.json"
DATASET = Path.home() / "aic_hexdex_sfp300"


def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    aa = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    bb = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def det_to_xyxy(det):
    if det is None:
        return None
    if det.corners_xy is not None:
        xs = det.corners_xy[:, 0]; ys = det.corners_xy[:, 1]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return [float(det.cx - det.width/2), float(det.cy - det.height/2),
            float(det.cx + det.width/2), float(det.cy + det.height/2)]


def center(b):
    return ((b[0]+b[2])/2, (b[1]+b[3])/2)


_INDEX = None


def _build_index():
    """One-time scan: (ep,fr) -> (file_idx, frame_in_file)."""
    import pyarrow.parquet as pq
    import re
    idx = {}
    for pq_path in sorted((DATASET / "data" / "chunk-000").glob("*.parquet")):
        m = re.match(r"file-(\d+)", pq_path.stem)
        if not m:
            continue
        file_idx = int(m.group(1))
        tbl = pq.read_table(pq_path, columns=["episode_index", "frame_index"])
        ep_arr = tbl["episode_index"].to_numpy()
        fr_arr = tbl["frame_index"].to_numpy()
        for i in range(len(ep_arr)):
            idx[(int(ep_arr[i]), int(fr_arr[i]))] = (file_idx, i)
    return idx


def get_frame(ep, fr, cam):
    global _INDEX
    if _INDEX is None:
        _INDEX = _build_index()
    entry = _INDEX.get((ep, fr))
    if entry is None:
        return None
    file_idx, frame_in_file = entry
    chunk_idx = 0
    video_path = DATASET / "videos" / f"observation.images.{cam}" / \
                 f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_file)
    ok, frame_bgr = cap.read()
    cap.release()
    return frame_bgr if ok else None


def main():
    if not ANN_PATH.exists():
        sys.exit("no annotations")
    annotations = json.loads(ANN_PATH.read_text())
    yolo = YoloPosePortDetector(conf=0.25)
    print(f"Loaded {len(annotations)} annotated cam-frames\n")

    for k, v in sorted(annotations.items()):
        ep, fr, cam = v["episode"], v["frame"], v["camera"]
        boxes = v["boxes"]
        targets = [b for b in boxes if b.get("label") == "target"]
        distractors = [b for b in boxes if b.get("label") == "distractor"]

        frame_bgr = get_frame(ep, fr, cam)
        if frame_bgr is None:
            print(f"{k}: NO FRAME")
            continue
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        yolo_det = yolo.detect(img_rgb, "sfp") if yolo.available else None
        cls_det = detect_classical(img_rgb, "sfp", refine=True)
        yolo_box = det_to_xyxy(yolo_det)
        cls_box = det_to_xyxy(cls_det)

        print(f"=== {k} (ep{ep} fr{fr} {cam}) ===")
        print(f"  user: {len(targets)} target(s), {len(distractors)} distractor(s)")

        for det_name, det_box in (("YOLO", yolo_box), ("classical", cls_box)):
            if det_box is None:
                print(f"  {det_name}: NO DET")
                continue
            dcx, dcy = center(det_box)
            best_target_iou = 0.0; best_target_dist = float("inf")
            for t in targets:
                tb = t["bbox_xyxy"]
                ti = iou(det_box, tb)
                tcx, tcy = center(tb)
                td = ((dcx-tcx)**2 + (dcy-tcy)**2)**0.5
                best_target_iou = max(best_target_iou, ti)
                best_target_dist = min(best_target_dist, td)
            best_distr_iou = 0.0; best_distr_dist = float("inf")
            for d in distractors:
                db = d["bbox_xyxy"]
                di = iou(det_box, db)
                dcxd, dcyd = center(db)
                dd = ((dcx-dcxd)**2 + (dcy-dcyd)**2)**0.5
                best_distr_iou = max(best_distr_iou, di)
                best_distr_dist = min(best_distr_dist, dd)
            ts = f"target IoU={best_target_iou:.2f} dist={best_target_dist:.0f}px" if targets else ""
            ds = f"distr IoU={best_distr_iou:.2f} dist={best_distr_dist:.0f}px" if distractors else ""
            verdict = ""
            if targets and distractors:
                if best_target_iou > best_distr_iou and best_target_dist < best_distr_dist:
                    verdict = "[CORRECT — closer to target]"
                elif best_distr_iou > best_target_iou or best_distr_dist < best_target_dist:
                    verdict = "[WRONG — closer to distractor]"
            elif targets:
                verdict = "[on-target]" if best_target_iou > 0.1 else "[far from target]"
            print(f"  {det_name:10s}: {ts}  {ds}  {verdict}")
        print()


if __name__ == "__main__":
    main()
