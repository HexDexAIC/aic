#!/usr/bin/env python3
"""Run the CORRECTED detector (YOLO + classical w/ rim shrink) on saved
frames from all 3 trials and save annotated images for review.

Generates:
    ~/aic_corrected_viz/
        trial_01_sfp/   trial_02_sfp/   trial_03_sc/
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
from aic_example_policies.ros.port_pose import lift_pnp


def quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def project_gt(rec):
    gt = rec.get("port_tf_base")
    cam = rec.get("center_cam_optical_tf_base")
    if gt is None or cam is None:
        return None
    K = rec["K_center"]
    Tcam = np.eye(4)
    Tcam[:3, :3] = quat_to_R(cam["qw"], cam["qx"], cam["qy"], cam["qz"])
    Tcam[:3, 3] = [cam["x"], cam["y"], cam["z"]]
    p_b = np.array([gt["x"], gt["y"], gt["z"], 1.0])
    p_c = np.linalg.inv(Tcam) @ p_b
    if p_c[2] <= 0:
        return None
    return (K[0] * p_c[0] / p_c[2] + K[2], K[4] * p_c[1] / p_c[2] + K[5]), p_c[2]


def annotate(img_rgb, port_type, rec, yolo):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = img_bgr.shape[:2]

    # YOLO (green)
    yolo_det = yolo.detect(img_rgb, port_type) if (yolo and yolo.available) else None
    if yolo_det is not None:
        if yolo_det.corners_xy is not None:
            box = yolo_det.corners_xy.astype(np.int32)
            cv2.polylines(img_bgr, [box], True, (0, 255, 0), 2)
            for (x, y) in yolo_det.corners_xy:
                cv2.circle(img_bgr, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(img_bgr, f"YOLO conf={yolo_det.score:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Compute PnP and error
        K = rec["K_center"]
        cam_tf = rec["center_cam_optical_tf_base"]
        pnp = lift_pnp(yolo_det, K, cam_tf, port_type=port_type)
        gt = rec.get("port_tf_base")
        if pnp is not None and gt is not None:
            dx = pnp.transform["x"] - gt["x"]
            dy = pnp.transform["y"] - gt["y"]
            dz = pnp.transform["z"] - gt["z"]
            err = ((dx * dx + dy * dy + dz * dz) ** 0.5) * 1000
            cv2.putText(img_bgr, f"YOLO err={err:.1f}mm",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Classical with rim shrink (red)
    cls_det = detect_classical(img_rgb, port_type, refine=True)
    if cls_det is not None:
        if cls_det.corners_xy is not None:
            box = cls_det.corners_xy.astype(np.int32)
            cv2.polylines(img_bgr, [box], True, (0, 0, 255), 2)
            for (x, y) in cls_det.corners_xy:
                cv2.circle(img_bgr, (int(x), int(y)), 4, (0, 0, 255), -1)
        cv2.putText(img_bgr, "classical (red)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Ground truth port projected (blue cross)
    proj = project_gt(rec)
    if proj is not None:
        (uu, vv), depth = proj
        cv2.drawMarker(img_bgr, (int(uu), int(vv)), (255, 200, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(img_bgr, f"GT (blue X) depth={depth:.3f}m",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    return img_bgr


def main():
    yolo = YoloPosePortDetector(conf=0.25)
    base = Path.home() / "aic_logs"
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
    run_dir = runs[-1]
    out_dir = Path.home() / "aic_corrected_viz"
    out_dir.mkdir(exist_ok=True)
    print(f"Run: {run_dir}\nOutput: {out_dir}")

    for trial in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")):
        td = out_dir / trial.name
        td.mkdir(exist_ok=True)
        task = json.loads((trial / "task.json").read_text())
        port_type = task["port_type"]
        # Sample 8 frames spread through the trial
        center_jpgs = sorted(trial.glob("*_center.jpg"))
        sample_indices = np.linspace(0, len(center_jpgs) - 1, 8).astype(int)
        for idx in sample_indices:
            jpg = center_jpgs[idx]
            json_p = jpg.with_name(jpg.name.replace("_center.jpg", ".json"))
            if not json_p.exists():
                continue
            rec = json.loads(json_p.read_text())
            img_bgr = cv2.imread(str(jpg))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            annotated = annotate(img_rgb, port_type, rec, yolo)
            outp = td / f"{idx:05d}.jpg"
            cv2.imwrite(str(outp), annotated)
            print(f"  {trial.name}/{idx:05d}.jpg")


if __name__ == "__main__":
    main()
