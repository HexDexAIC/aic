#!/usr/bin/env python3
"""Visualize SC detection to diagnose what's being detected."""
import json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))
from aic_example_policies.ros.port_detector import detect_port

base = Path.home() / "aic_logs"
runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
trial = sorted(p for p in runs[-1].iterdir() if p.is_dir() and p.name == "trial_03_sc")[0]
out_dir = Path.home() / "aic_sc_viz"
out_dir.mkdir(exist_ok=True)

for jpg in sorted(trial.glob("*_center.jpg"))[:5]:
    img_bgr = cv2.imread(str(jpg))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    det = detect_port(img_rgb, "sc")
    out = img_bgr.copy()
    if det is not None:
        x0 = int(det.cx - det.width / 2)
        y0 = int(det.cy - det.height / 2)
        x1 = int(det.cx + det.width / 2)
        y1 = int(det.cy + det.height / 2)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(out, (int(det.cx), int(det.cy)), 3, (0, 0, 255), -1)
        cv2.putText(out, f"sc w={det.width:.0f} h={det.height:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if det.corners_xy is not None:
            for i, (kx, ky) in enumerate(det.corners_xy):
                cv2.circle(out, (int(kx), int(ky)), 4, (255, 0, 0), -1)
    else:
        cv2.putText(out, "NO DET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    op = out_dir / jpg.name
    cv2.imwrite(str(op), out)
    print(f"{jpg.name}: det={det}")
