#!/usr/bin/env python3
"""Test YOLO ONNX inference on a real captured frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))

from aic_example_policies.ros.port_detector_yolo import YoloPosePortDetector
from aic_example_policies.ros.port_pose import lift_pnp


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else \
        str(Path.home() / "aic_runs" / "yolo_smoke" / "weights" / "best.onnx")
    print(f"Weights: {weights}")
    det = YoloPosePortDetector(weights, imgsz=640, conf=0.25)
    if not det.available:
        print("Model not loaded")
        return

    base = Path.home() / "aic_logs"
    runs = sorted(p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit())
    run_dir = runs[-1]
    trials = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_"))
    for trial in trials:
        task = json.loads((trial / "task.json").read_text())
        port_type = task["port_type"]
        center_jpgs = sorted(trial.glob("*_center.jpg"))
        if not center_jpgs:
            continue
        print(f"\n=== {trial.name} port_type={port_type} ===")
        for i in [0, len(center_jpgs) // 2, len(center_jpgs) - 1]:
            jpg = center_jpgs[i]
            json_p = jpg.with_name(jpg.name.replace("_center.jpg", ".json"))
            img_bgr = cv2.imread(str(jpg))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            d = det.detect(img_rgb, port_type)
            if d is None:
                print(f"  frame {i}: NO YOLO DETECTION")
                continue
            print(
                f"  frame {i}: yolo conf={d.score:.3f} cx,cy=({d.cx:.0f},{d.cy:.0f}) wh=({d.width:.0f},{d.height:.0f})"
            )
            if json_p.exists():
                rec = json.loads(json_p.read_text())
                pnp = lift_pnp(d, rec["K_center"], rec["center_cam_optical_tf_base"], port_type=port_type)
                if pnp is not None and rec.get("port_tf_base") is not None:
                    gt = rec["port_tf_base"]
                    dx = pnp.transform["x"] - gt["x"]
                    dy = pnp.transform["y"] - gt["y"]
                    dz = pnp.transform["z"] - gt["z"]
                    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                    print(
                        f"           pnp: depth={pnp.depth_m:.3f}m err={dist*1000:.1f}mm "
                        f"(dx={dx*1000:+.1f} dy={dy*1000:+.1f} dz={dz*1000:+.1f})"
                    )


if __name__ == "__main__":
    main()
