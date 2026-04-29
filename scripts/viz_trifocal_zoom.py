#!/usr/bin/env python3
"""Trifocal viz with ZOOMED crops around the detected/GT port location.

For each frame and each camera, crop a 200x200 region around the
ground-truth port projection and show classical / YOLO detections
clearly inside that region.
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
from aic_example_policies.ros.port_pose import lift_triangulate


def quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def project(p_base, K, cam_tf):
    if cam_tf is None: return None
    Tcam = np.eye(4)
    Tcam[:3, :3] = quat_to_R(cam_tf["qw"], cam_tf["qx"], cam_tf["qy"], cam_tf["qz"])
    Tcam[:3, 3] = [cam_tf["x"], cam_tf["y"], cam_tf["z"]]
    p = np.array([p_base[0], p_base[1], p_base[2], 1.0])
    p_c = np.linalg.inv(Tcam) @ p
    if p_c[2] <= 0: return None
    u = K[0] * p_c[0] / p_c[2] + K[2]
    v = K[4] * p_c[1] / p_c[2] + K[5]
    return (float(u), float(v))


def make_zoom_panel(img_rgb, port_type, K, cam_tf, gt_base, tri_base, yolo, cam_name, crop_size=240):
    """Return an annotated, zoomed-in panel centered on GT port projection."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    yolo_det = yolo.detect(img_rgb, port_type) if (yolo and yolo.available) else None
    cls_det = detect_classical(img_rgb, port_type, refine=True)

    # Compute crop center (use GT projection if available, else image center)
    proj_gt = project([gt_base["x"], gt_base["y"], gt_base["z"]], K, cam_tf) if gt_base else None
    if proj_gt is not None:
        cx_zoom, cy_zoom = int(proj_gt[0]), int(proj_gt[1])
    elif yolo_det is not None:
        cx_zoom, cy_zoom = int(yolo_det.cx), int(yolo_det.cy)
    elif cls_det is not None:
        cx_zoom, cy_zoom = int(cls_det.cx), int(cls_det.cy)
    else:
        cx_zoom, cy_zoom = w // 2, h // 2

    half = crop_size // 2
    x0 = max(0, cx_zoom - half)
    y0 = max(0, cy_zoom - half)
    x1 = min(w, x0 + crop_size)
    y1 = min(h, y0 + crop_size)
    x0 = max(0, x1 - crop_size)
    y0 = max(0, y1 - crop_size)
    crop = img_bgr[y0:y1, x0:x1].copy()
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    # Draw detections in crop coords
    def to_local(u, v):
        return (int(u - x0), int(v - y0))

    if yolo_det is not None and yolo_det.corners_xy is not None:
        pts = np.array([to_local(x, y) for (x, y) in yolo_det.corners_xy], dtype=np.int32)
        cv2.polylines(crop, [pts], True, (0, 255, 0), 2)
        for (x, y) in yolo_det.corners_xy:
            cv2.circle(crop, to_local(x, y), 5, (0, 255, 0), -1)

    if cls_det is not None and cls_det.corners_xy is not None:
        pts = np.array([to_local(x, y) for (x, y) in cls_det.corners_xy], dtype=np.int32)
        cv2.polylines(crop, [pts], True, (0, 0, 255), 2)
        for (x, y) in cls_det.corners_xy:
            cv2.circle(crop, to_local(x, y), 4, (0, 0, 255), -1)

    if proj_gt is not None:
        cv2.drawMarker(crop, to_local(proj_gt[0], proj_gt[1]), (0, 255, 255),
                       cv2.MARKER_CROSS, 24, 3)

    if tri_base is not None:
        proj_tri = project([tri_base[0], tri_base[1], tri_base[2]], K, cam_tf)
        if proj_tri is not None:
            cv2.circle(crop, to_local(proj_tri[0], proj_tri[1]), 12, (255, 200, 0), 2)

    # Upscale 2x and label
    crop_big = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop_big, cam_name, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(crop_big, cam_name, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return crop_big


def main():
    yolo = YoloPosePortDetector(conf=0.25)
    base = Path.home() / "aic_logs"
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
    run_dir = runs[-1]
    out_dir = Path.home() / "aic_trifocal_zoom"
    out_dir.mkdir(exist_ok=True)
    print(f"Run: {run_dir}\nOutput: {out_dir}")

    for trial in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")):
        td = out_dir / trial.name
        td.mkdir(exist_ok=True)
        task = json.loads((trial / "task.json").read_text())
        port_type = task["port_type"]
        center_jpgs = sorted(trial.glob("*_center.jpg"))
        sample_idx = np.linspace(0, len(center_jpgs) - 1, 8).astype(int)

        for idx in sample_idx:
            center_jpg = center_jpgs[idx]
            json_p = center_jpg.with_name(center_jpg.name.replace("_center.jpg", ".json"))
            if not json_p.exists():
                continue
            rec = json.loads(json_p.read_text())
            stem = center_jpg.name.replace("_center.jpg", "")
            triplet = {}
            for cam in ("left", "center", "right"):
                jpg = center_jpg.with_name(f"{stem}_{cam}.jpg")
                if jpg.exists():
                    triplet[cam] = cv2.imread(str(jpg))

            # Detect on each camera and triangulate
            detections = []
            for cam in ("left", "center", "right"):
                if cam not in triplet:
                    continue
                img_rgb = cv2.cvtColor(triplet[cam], cv2.COLOR_BGR2RGB)
                K = rec.get(f"K_{cam}")
                cam_tf = rec.get(f"{cam}_cam_optical_tf_base")
                if yolo.available:
                    det = yolo.detect(img_rgb, port_type)
                    if det is None:
                        det = detect_classical(img_rgb, port_type, refine=True)
                else:
                    det = detect_classical(img_rgb, port_type, refine=True)
                if det is not None and K and cam_tf:
                    detections.append((det, K, cam_tf, cam))
            tri_pose = lift_triangulate([(d, K, T) for (d, K, T, _) in detections])
            tri_xyz = None
            tri_err = None
            gt = rec.get("port_tf_base")
            if tri_pose is not None:
                tri_xyz = (tri_pose.transform["x"], tri_pose.transform["y"], tri_pose.transform["z"])
                if gt is not None:
                    tri_err = ((tri_xyz[0] - gt["x"]) ** 2 + (tri_xyz[1] - gt["y"]) ** 2 + (tri_xyz[2] - gt["z"]) ** 2) ** 0.5 * 1000

            panels = []
            for cam in ("left", "center", "right"):
                if cam not in triplet:
                    continue
                img_rgb = cv2.cvtColor(triplet[cam], cv2.COLOR_BGR2RGB)
                K = rec.get(f"K_{cam}")
                cam_tf = rec.get(f"{cam}_cam_optical_tf_base")
                panel = make_zoom_panel(img_rgb, port_type, K, cam_tf, gt, tri_xyz, yolo, cam)
                panels.append(panel)

            min_h = min(p.shape[0] for p in panels)
            panels = [p[:min_h] for p in panels]
            grid = np.hstack(panels)

            footer_h = 130
            footer = np.zeros((footer_h, grid.shape[1], 3), dtype=np.uint8)
            cv2.putText(footer, "ZOOM 2x around GT port projection",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(footer, "GREEN=YOLO, RED=classical, YELLOW X=GT, CYAN circle=triangulated",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tri_str = f"Triangulation 3D err: {tri_err:.2f} mm" if tri_err is not None else "Triangulation: REJECTED"
            cv2.putText(footer, tri_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            full = np.vstack([grid, footer])

            outp = td / f"{idx:05d}.jpg"
            cv2.imwrite(str(outp), full)
            print(f"  {trial.name}/{idx:05d}.jpg  tri_err={tri_err}")


if __name__ == "__main__":
    main()
