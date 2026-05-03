#!/usr/bin/env python3
"""Visualize YOLO predictions vs auto-labels on the same frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))

from aic_example_policies.ros.port_detector_yolo import YoloPosePortDetector


def parse_label(line, w, h):
    parts = line.strip().split()
    cls = int(parts[0])
    cx = float(parts[1]) * w
    cy = float(parts[2]) * h
    bw = float(parts[3]) * w
    bh = float(parts[4]) * h
    rest = parts[5:]
    kps = []
    for i in range(0, len(rest), 3):
        kx = float(rest[i]) * w
        ky = float(rest[i + 1]) * h
        kps.append((kx, ky))
    return cls, (cx, cy, bw, bh), kps


def main():
    yolo = YoloPosePortDetector(conf=0.25)
    if not yolo.available:
        print("yolo unavailable")
        return

    base = Path.home() / "aic_logs"
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
    run_dir = runs[-1]
    dataset = Path.home() / "aic_dataset_full"

    out_dir = Path.home() / "aic_yolo_viz"
    out_dir.mkdir(exist_ok=True)

    for trial in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")):
        task = json.loads((trial / "task.json").read_text())
        port_type = task["port_type"]
        center_jpgs = sorted(trial.glob("*_center.jpg"))[:3]
        for jpg in center_jpgs:
            img_bgr = cv2.imread(str(jpg))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_bgr.shape[:2]

            det = yolo.detect(img_rgb, port_type)

            # Find label for this frame
            label_stem = f"{trial.name}_{jpg.stem.replace('_center', '')}_center"
            lbl_path = dataset / "labels" / f"{label_stem}.txt"

            out = img_bgr.copy()
            # Draw label (green = ground truth)
            if lbl_path.exists():
                line = lbl_path.read_text().strip().splitlines()[0]
                cls, (cx, cy, bw, bh), kps = parse_label(line, w, h)
                x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
                x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
                cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 0), 2)
                for i, (kx, ky) in enumerate(kps):
                    col = (0, 255, 0) if i < 4 else (0, 200, 0)
                    cv2.circle(out, (int(kx), int(ky)), 4, col, -1)
                    cv2.putText(out, f"L{i}", (int(kx) + 4, int(ky)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

            # Draw YOLO predictions (red)
            if det is not None:
                x0 = int(det.cx - det.width / 2)
                y0 = int(det.cy - det.height / 2)
                x1 = int(det.cx + det.width / 2)
                y1 = int(det.cy + det.height / 2)
                cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
                if det.corners_xy is not None:
                    for i, (kx, ky) in enumerate(det.corners_xy):
                        cv2.circle(out, (int(kx), int(ky)), 4, (0, 0, 255), -1)
                        cv2.putText(out, f"Y{i}", (int(kx) + 4, int(ky) + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(out, f"YOLO conf={det.score:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(out, "YOLO: NO DET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            outp = out_dir / f"{trial.name}_{jpg.stem}.jpg"
            cv2.imwrite(str(outp), out)
            print(f"wrote {outp}")


if __name__ == "__main__":
    main()
