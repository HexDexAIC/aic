#!/usr/bin/env python3
"""Offline benchmark: detector accuracy on saved frames from all 3 trials.

For each trial, evaluates each detector strategy on the first N frames
(simulating trial-start detection). Reports per-trial median port-pose
error in mm. The lowest median error is what would maximize the score.

Strategies compared:
  1. classical_pnp        — single-frame classical detector + PnP
  2. classical_known_size — single-frame classical + known-size depth
  3. classical_multiframe — average over N start frames + PnP
  4. yolo_pnp             — single-frame YOLO + PnP
  5. classical_ransac     — RANSAC over multi-frame classical detections
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))

from aic_example_policies.ros.port_detector import detect_port
from aic_example_policies.ros.port_pose import lift_pnp, lift_to_base, lift_triangulate
try:
    from aic_example_policies.ros.port_detector_yolo import YoloPosePortDetector
    YOLO = YoloPosePortDetector(conf=0.25)
except Exception as e:
    print(f"yolo unavailable: {e}")
    YOLO = None


def err_mm(pose, gt):
    if pose is None or gt is None:
        return None
    dx = pose["x"] - gt["x"]
    dy = pose["y"] - gt["y"]
    dz = pose["z"] - gt["z"]
    return ((dx * dx + dy * dy + dz * dz) ** 0.5) * 1000


def estimate_one(jpg_path: Path, json_path: Path, port_type: str, method: str):
    rec = json.loads(json_path.read_text())
    img_bgr = cv2.imread(str(jpg_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    K = rec["K_center"]
    cam_tf = rec["center_cam_optical_tf_base"]

    expected_uv = None
    if "filtered" in method:
        # Use ground-truth port location projected to image — this is the
        # ORACLE prior (only available offline); production VisionInsert
        # would compute this from TCP forward direction.
        gt = rec.get("port_tf_base")
        if gt is not None:
            import numpy as np
            def quat_to_R(qw, qx, qy, qz):
                return np.array([
                    [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
                ])
            Tcam = np.eye(4)
            Tcam[:3, :3] = quat_to_R(cam_tf["qw"], cam_tf["qx"], cam_tf["qy"], cam_tf["qz"])
            Tcam[:3, 3] = [cam_tf["x"], cam_tf["y"], cam_tf["z"]]
            p_b = np.array([gt["x"], gt["y"], gt["z"], 1.0])
            p_c = np.linalg.inv(Tcam) @ p_b
            if p_c[2] > 0:
                u = K[0] * p_c[0] / p_c[2] + K[2]
                v = K[4] * p_c[1] / p_c[2] + K[5]
                expected_uv = (float(u), float(v))

    # Triangulation methods checked FIRST (before generic classical_pnp branch)
    if "triangulate" in method:
        triplet = []
        for cam in ("left", "center", "right"):
            jpg_other = jpg_path.with_name(jpg_path.name.replace("_center.jpg", f"_{cam}.jpg"))
            if not jpg_other.exists():
                continue
            other_bgr = cv2.imread(str(jpg_other))
            other_rgb = cv2.cvtColor(other_bgr, cv2.COLOR_BGR2RGB)
            K_other = rec.get(f"K_{cam}")
            cam_tf_other = rec.get(f"{cam}_cam_optical_tf_base")
            if K_other is None or cam_tf_other is None:
                continue
            if method.startswith("yolo"):
                if YOLO is None or not YOLO.available:
                    continue
                det = YOLO.detect(other_rgb, port_type)
            else:
                det = detect_port(other_rgb, port_type, refine=True)
            if det is None:
                continue
            triplet.append((det, K_other, cam_tf_other))
        if len(triplet) < 2:
            return None
        return lift_triangulate(triplet)

    if method.startswith("classical"):
        from aic_example_policies.ros.port_detector import detect_port_filtered
        refine = "refined" in method
        if "filtered" in method:
            det = detect_port_filtered(img_rgb, port_type, expected_uv=expected_uv,
                                       expected_radius=150)
        else:
            det = detect_port(img_rgb, port_type, refine=refine)
        if det is None:
            return None
        if "pnp" in method:
            return lift_pnp(det, K, cam_tf, port_type=port_type)
        if "known" in method:
            return lift_to_base(det, K, cam_tf, port_type=port_type)
    elif method == "yolo_pnp":
        if YOLO is None or not YOLO.available:
            return None
        det = YOLO.detect(img_rgb, port_type)
        if det is None:
            return None
        return lift_pnp(det, K, cam_tf, port_type=port_type)
    return None


def estimate_multiframe(jpg_paths, json_paths, port_type: str, method="classical_pnp"):
    """Run single-frame estimate on each frame, return median (xyz-wise) of valid ones."""
    poses = []
    for jpg, jp in zip(jpg_paths, json_paths):
        p = estimate_one(jpg, jp, port_type, method)
        if p is not None:
            poses.append(p.transform)
    if not poses:
        return None
    xs = sorted([p["x"] for p in poses])
    ys = sorted([p["y"] for p in poses])
    zs = sorted([p["z"] for p in poses])
    n = len(poses) // 2
    return {
        "x": xs[n], "y": ys[n], "z": zs[n],
        "qw": poses[0]["qw"], "qx": poses[0]["qx"], "qy": poses[0]["qy"], "qz": poses[0]["qz"],
    }


def estimate_ransac(jpg_paths, json_paths, port_type: str, threshold_m=0.005):
    """RANSAC: pick the pose whose inlier set (other poses within threshold_m) is biggest."""
    candidates = []
    for jpg, jp in zip(jpg_paths, json_paths):
        p = estimate_one(jpg, jp, port_type, "classical_pnp")
        if p is not None:
            candidates.append((np.array([p.transform["x"], p.transform["y"], p.transform["z"]]), p.transform))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][1]

    best_inliers = []
    for ref_pos, ref_tf in candidates:
        inliers = [tf for pos, tf in candidates if np.linalg.norm(pos - ref_pos) < threshold_m]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    if not best_inliers:
        return None
    xs = sorted([t["x"] for t in best_inliers])
    ys = sorted([t["y"] for t in best_inliers])
    zs = sorted([t["z"] for t in best_inliers])
    n = len(best_inliers) // 2
    return {
        "x": xs[n], "y": ys[n], "z": zs[n],
        "qw": best_inliers[0]["qw"], "qx": best_inliers[0]["qx"],
        "qy": best_inliers[0]["qy"], "qz": best_inliers[0]["qz"],
    }


def main():
    base = Path.home() / "aic_logs"
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
    if not runs:
        sys.exit("no runs")
    run_dir = runs[-1]
    print(f"Run: {run_dir}\n")

    methods = [
        "classical_refined_pnp",
        "classical_triangulate",
        "yolo_pnp",
        "yolo_triangulate",
    ]
    multi_methods = ["classical_pnp_multiframe_5", "classical_pnp_multiframe_20", "classical_ransac_20"]

    trials = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_"))
    summary = {}
    for trial in trials:
        task = json.loads((trial / "task.json").read_text())
        port_type = task["port_type"]
        center_jpgs = sorted(trial.glob("*_center.jpg"))[:50]  # first 50 frames
        json_files = [j.with_name(j.name.replace("_center.jpg", ".json")) for j in center_jpgs]

        # Get ground truth port pose (use frame 0)
        if not json_files or not json_files[0].exists():
            continue
        gt = json.loads(json_files[0].read_text())["port_tf_base"]

        print(f"=== {trial.name} (port_type={port_type}, gt_z={gt['z']:.3f}) ===")
        per_trial = {}
        # Single-frame methods
        for m in methods:
            errs = []
            misses = 0
            for jpg, jp in zip(center_jpgs, json_files):
                p = estimate_one(jpg, jp, port_type, m)
                if p is None:
                    misses += 1
                    continue
                e = err_mm(p.transform, gt)
                if e is not None:
                    errs.append(e)
            if errs:
                arr = np.array(errs)
                print(f"  {m:<32s}: n={len(errs):3d} miss={misses:2d} median={np.median(arr):.1f}mm  p25={np.percentile(arr, 25):.1f}  p75={np.percentile(arr, 75):.1f}")
                per_trial[m] = float(np.median(arr))

        # Multi-frame methods (single estimate from N frames)
        for n_frames in [5, 20]:
            sub_jpg = center_jpgs[:n_frames]
            sub_jp = json_files[:n_frames]
            mf = estimate_multiframe(sub_jpg, sub_jp, port_type)
            if mf is not None:
                e = err_mm(mf, gt)
                print(f"  classical_multiframe_n{n_frames:<5d}        : {e:.1f}mm")
                per_trial[f"classical_multiframe_{n_frames}"] = e

        # RANSAC
        ransac = estimate_ransac(center_jpgs[:20], json_files[:20], port_type)
        if ransac is not None:
            e = err_mm(ransac, gt)
            print(f"  classical_ransac_20             : {e:.1f}mm")
            per_trial["classical_ransac_20"] = e

        summary[trial.name] = per_trial
        print()

    # Summary table
    print("=== SUMMARY (median error mm) ===")
    all_methods = set()
    for d in summary.values():
        all_methods.update(d.keys())
    print(f"{'Method':<40s}  {'  '.join(t.replace('trial_', 'T') for t in summary.keys())}")
    for m in sorted(all_methods):
        row = []
        for t in summary.keys():
            v = summary[t].get(m)
            row.append(f"{v:6.1f}" if v is not None else "   N/A")
        print(f"{m:<40s}  {'  '.join(row)}")


if __name__ == "__main__":
    main()
