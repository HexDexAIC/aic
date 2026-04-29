#!/usr/bin/env python3
"""Trifocal port-detection visualization: shows ALL 3 cameras side-by-side
with per-camera detections and the triangulated 3D position projected
back into each view.

Output: ~/aic_trifocal_viz/trial_NN_<port>/<frame>.jpg
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
from aic_example_policies.ros.port_pose import lift_pnp, lift_triangulate


def quat_to_R(qw, qx, qy, qz):
    return np.array([
        [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def project_into_cam(p_base, K, cam_tf):
    """Project a 3D base-frame point to camera image coordinates."""
    if cam_tf is None:
        return None
    Tcam = np.eye(4)
    Tcam[:3, :3] = quat_to_R(cam_tf["qw"], cam_tf["qx"], cam_tf["qy"], cam_tf["qz"])
    Tcam[:3, 3] = [cam_tf["x"], cam_tf["y"], cam_tf["z"]]
    p = np.array([p_base[0], p_base[1], p_base[2], 1.0])
    p_c = np.linalg.inv(Tcam) @ p
    if p_c[2] <= 0:
        return None
    u = K[0] * p_c[0] / p_c[2] + K[2]
    v = K[4] * p_c[1] / p_c[2] + K[5]
    return (float(u), float(v))


def annotate_one(img_rgb, port_type, K, cam_tf, gt_base, tri_base, yolo):
    """Draw YOLO + classical + projected GT and projected triangulation onto a single cam image."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

    yolo_det = yolo.detect(img_rgb, port_type) if (yolo and yolo.available) else None
    cls_det = detect_classical(img_rgb, port_type, refine=True)

    if yolo_det is not None and yolo_det.corners_xy is not None:
        box = yolo_det.corners_xy.astype(np.int32)
        cv2.polylines(img_bgr, [box], True, (0, 255, 0), 2)

    if cls_det is not None and cls_det.corners_xy is not None:
        box = cls_det.corners_xy.astype(np.int32)
        cv2.polylines(img_bgr, [box], True, (0, 0, 255), 2)

    # Project GT (yellow) and triangulated (cyan)
    if gt_base is not None:
        proj = project_into_cam([gt_base["x"], gt_base["y"], gt_base["z"]], K, cam_tf)
        if proj is not None:
            cv2.drawMarker(img_bgr, (int(proj[0]), int(proj[1])), (0, 255, 255),
                           cv2.MARKER_CROSS, 24, 3)

    if tri_base is not None:
        proj = project_into_cam([tri_base[0], tri_base[1], tri_base[2]], K, cam_tf)
        if proj is not None:
            cv2.circle(img_bgr, (int(proj[0]), int(proj[1])), 8, (255, 200, 0), 3)

    return img_bgr, yolo_det, cls_det


def detect_in(img_rgb, port_type, yolo):
    """Return primary detection (YOLO if available, else classical refined)."""
    if yolo and yolo.available:
        d = yolo.detect(img_rgb, port_type)
        if d is not None:
            return d
    return detect_classical(img_rgb, port_type, refine=True)


def main():
    yolo = YoloPosePortDetector(conf=0.25)
    base = Path.home() / "aic_logs"
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
    run_dir = runs[-1]
    out_dir = Path.home() / "aic_trifocal_viz"
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
                if not jpg.exists():
                    continue
                triplet[cam] = cv2.imread(str(jpg))
            if "center" not in triplet:
                continue

            # Detect on each camera; attempt triangulation
            detections = []
            for cam in ("left", "center", "right"):
                if cam not in triplet:
                    continue
                img_rgb = cv2.cvtColor(triplet[cam], cv2.COLOR_BGR2RGB)
                K = rec.get(f"K_{cam}")
                cam_tf = rec.get(f"{cam}_cam_optical_tf_base")
                det = detect_in(img_rgb, port_type, yolo)
                detections.append((det, K, cam_tf, cam))

            tri_pose = lift_triangulate(
                [(d, K, T) for (d, K, T, _) in detections]
            )

            # PnP from center alone, for comparison.
            pnp_pose = None
            for (det, K, cam_tf, cam) in detections:
                if cam == "center" and det is not None:
                    pnp_pose = lift_pnp(det, K, cam_tf, port_type=port_type)
                    break

            gt = rec.get("port_tf_base")
            tri_xyz = None
            tri_err = None
            if tri_pose is not None:
                tri_xyz = (tri_pose.transform["x"], tri_pose.transform["y"], tri_pose.transform["z"])
                if gt is not None:
                    tri_err = ((tri_xyz[0] - gt["x"]) ** 2 + (tri_xyz[1] - gt["y"]) ** 2 + (tri_xyz[2] - gt["z"]) ** 2) ** 0.5 * 1000
            pnp_err = None
            if pnp_pose is not None and gt is not None:
                pnp_err = ((pnp_pose.transform["x"] - gt["x"]) ** 2 + (pnp_pose.transform["y"] - gt["y"]) ** 2 + (pnp_pose.transform["z"] - gt["z"]) ** 2) ** 0.5 * 1000

            # Annotate each cam image
            panels = []
            for cam in ("left", "center", "right"):
                if cam not in triplet:
                    continue
                img_rgb = cv2.cvtColor(triplet[cam], cv2.COLOR_BGR2RGB)
                K = rec.get(f"K_{cam}")
                cam_tf = rec.get(f"{cam}_cam_optical_tf_base")
                annotated, _, _ = annotate_one(img_rgb, port_type, K, cam_tf, gt, tri_xyz, yolo)
                # add label
                cv2.putText(annotated, f"{cam}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                cv2.putText(annotated, f"{cam}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                # downscale for the side-by-side
                h, w = annotated.shape[:2]
                small = cv2.resize(annotated, (w // 2, h // 2))
                panels.append(small)

            # Stitch horizontally
            min_h = min(p.shape[0] for p in panels)
            panels = [p[:min_h] for p in panels]
            grid = np.hstack(panels)

            # Append a footer with the numeric results
            footer = np.zeros((100, grid.shape[1], 3), dtype=np.uint8)
            txt = f"GT (yellow X)  TRI (cyan circle)"
            cv2.putText(footer, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            txt2 = f"PnP (center) err: {pnp_err:.1f} mm" if pnp_err is not None else "PnP: N/A"
            cv2.putText(footer, txt2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            txt3 = f"Triangulation err: {tri_err:.1f} mm" if tri_err is not None else "Triangulation: rejected (rays disagree)"
            cv2.putText(footer, txt3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            full = np.vstack([grid, footer])

            outp = td / f"{idx:05d}.jpg"
            cv2.imwrite(str(outp), full)
            print(f"  {trial.name}/{idx:05d}.jpg  pnp={pnp_err}  tri={tri_err}")


if __name__ == "__main__":
    main()
