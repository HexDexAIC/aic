#!/usr/bin/env python3
"""Offline simulation of VisionInsert: detect port + lift + compare with GT.

Doesn't run any motion — just exercises the perception pipeline end-to-end
on logged frames, returning the per-frame error in mm relative to the
ground-truth port_tf_base. Useful for validating that detector + PnP +
TF composition are all consistent before kicking off a live eval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))

from aic_example_policies.ros.port_detector import detect_port
from aic_example_policies.ros.port_pose import lift_pnp, lift_to_base


def main():
    base = Path.home() / "aic_logs"
    runs = sorted(p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit())
    run_dir = runs[-1]
    print(f"Run: {run_dir}")
    trials = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_"))

    overall_results = []
    for trial in trials:
        task = json.loads((trial / "task.json").read_text())
        port_type = task["port_type"]
        center_jpgs = sorted(trial.glob("*_center.jpg"))
        if not center_jpgs:
            continue
        print(f"\n=== {trial.name}  port_type={port_type}  {len(center_jpgs)} frames ===")
        ok = 0
        bad = 0
        errs = []
        for jpg in center_jpgs[:50]:  # check first 50 frames
            json_p = jpg.with_name(jpg.name.replace("_center.jpg", ".json"))
            if not json_p.exists():
                continue
            rec = json.loads(json_p.read_text())
            gt = rec.get("port_tf_base")
            if gt is None:
                continue
            img_bgr = cv2.imread(str(jpg))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            det = detect_port(img_rgb, port_type)
            if det is None:
                bad += 1
                continue
            pnp = lift_pnp(det, rec["K_center"], rec["center_cam_optical_tf_base"], port_type=port_type)
            ks = lift_to_base(det, rec["K_center"], rec["center_cam_optical_tf_base"], port_type=port_type)
            for name, lifted in (("pnp", pnp), ("known", ks)):
                if lifted is None:
                    continue
                dx = lifted.transform["x"] - gt["x"]
                dy = lifted.transform["y"] - gt["y"]
                dz = lifted.transform["z"] - gt["z"]
                d = (dx * dx + dy * dy + dz * dz) ** 0.5
                errs.append((name, d * 1000))
            ok += 1
        for method in ("pnp", "known"):
            ms = [e[1] for e in errs if e[0] == method]
            if ms:
                arr = np.array(ms)
                print(f"  {method}: ok_frames={ok} bad={bad} err_mm mean={arr.mean():.1f} median={np.median(arr):.1f} min={arr.min():.1f} max={arr.max():.1f}")
        overall_results.append((trial.name, ok, bad, errs))

    print("\n=== Summary ===")
    for name, ok, bad, errs in overall_results:
        med_pnp = np.median([e[1] for e in errs if e[0] == "pnp"]) if errs else float("nan")
        med_kn = np.median([e[1] for e in errs if e[0] == "known"]) if errs else float("nan")
        print(f"  {name}: ok={ok} bad={bad} med_pnp={med_pnp:.1f}mm med_known={med_kn:.1f}mm")


if __name__ == "__main__":
    main()
