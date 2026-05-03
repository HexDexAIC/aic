#!/usr/bin/env python3
"""For each annotated frame, draw user boxes + YOLO + classical side by side
so the user can see exactly what each detector picked vs what they marked."""
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


def main():
    ann_path = Path.home() / "aic_user_annotations.json"
    if not ann_path.exists():
        sys.exit("no annotations")
    annotations = json.loads(ann_path.read_text())
    out_dir = Path.home() / "aic_user_vs_dets"
    out_dir.mkdir(exist_ok=True)
    yolo = YoloPosePortDetector(conf=0.25)

    for img_path, rec in sorted(annotations.items()):
        if not Path(img_path).exists():
            continue
        boxes = rec.get("boxes", [])
        port_type = rec.get("port_type", "sfp")
        trial = rec.get("trial", "?")
        cam = rec.get("camera", "?")
        frame_idx = rec.get("frame_idx", 0)

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out = img_bgr.copy()

        # User boxes
        for b in boxes:
            color = {"target": (0, 200, 255),
                     "distractor": (255, 0, 200),
                     "other": (180, 180, 180)}.get(b.get("label"), (200, 200, 200))
            x0, y0, x1, y1 = [int(round(v)) for v in b["bbox_xyxy"]]
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
            cv2.putText(out, "USER " + b.get("label", "?"), (x0, max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # YOLO
        yolo_det = yolo.detect(img_rgb, port_type) if (yolo and yolo.available) else None
        if yolo_det is not None and yolo_det.corners_xy is not None:
            box = yolo_det.corners_xy.astype(np.int32)
            cv2.polylines(out, [box], True, (0, 255, 0), 2)
            cv2.putText(out, f"YOLO {yolo_det.score:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Classical
        cls_det = detect_classical(img_rgb, port_type, refine=True)
        if cls_det is not None and cls_det.corners_xy is not None:
            box = cls_det.corners_xy.astype(np.int32)
            cv2.polylines(out, [box], True, (0, 0, 255), 2)
            cv2.putText(out, "classical", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        title = f"{trial} {cam} fr{frame_idx}"
        cv2.putText(out, title, (10, out.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        outp = out_dir / f"{trial}_{cam}_{frame_idx:05d}.jpg"
        cv2.imwrite(str(outp), out)
        print(f"  {outp}")


if __name__ == "__main__":
    main()
