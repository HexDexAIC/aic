#!/usr/bin/env python3
"""Quick offline test of the classical port detector on captured frames.

Reads center JPGs from the latest aic_logs run, detects the port matching
each trial's port_type, draws an overlay, and writes annotated JPGs to
the same dir for visual inspection. Also reports detection success rate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add the policy package to path so we can import port_detector + port_pose.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))

from aic_example_policies.ros.port_detector import detect_port, draw_detection
from aic_example_policies.ros.port_pose import lift_to_base, lift_stereo, lift_pnp


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    base = Path.home() / "aic_logs"
    if arg:
        run_dir = Path(arg).expanduser()
    else:
        runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
        if not runs:
            sys.exit(f"No runs under {base}")
        run_dir = runs[-1]
    print(f"Run dir: {run_dir}")

    trials = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_"))
    for trial in trials:
        task = json.loads((trial / "task.json").read_text())
        port_type = task["port_type"]
        print(f"\n=== {trial.name}  port_type={port_type} ===")
        center_jpgs = sorted(trial.glob("*_center.jpg"))
        if not center_jpgs:
            print("  no center frames yet")
            continue
        # Test on first, middle, and a sample frame near each pfrac=0.5
        sample_idx = [0, len(center_jpgs) // 2, len(center_jpgs) - 1]
        for i in sample_idx:
            jpg = center_jpgs[i]
            json_p = jpg.with_name(jpg.name.replace("_center.jpg", ".json"))
            img_bgr = cv2.imread(str(jpg))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            det = detect_port(img_rgb, port_type)
            if det is None:
                print(f"  frame {i:>4}: NO DETECTION")
                continue
            print(
                f"  frame {i:>4}: {port_type} det score={det.score:.3f} "
                f"cx,cy=({det.cx:.0f},{det.cy:.0f}) wh=({det.width:.0f},{det.height:.0f})"
            )
            # Lift to base if we have the supporting data
            if json_p.exists():
                rec = json.loads(json_p.read_text())
                lifted = lift_to_base(
                    det,
                    rec["K_center"],
                    rec["center_cam_optical_tf_base"],
                    port_type=port_type,
                )
                if lifted is not None:
                    gt = rec.get("port_tf_base")
                    if gt is not None:
                        dx = lifted.transform["x"] - gt["x"]
                        dy = lifted.transform["y"] - gt["y"]
                        dz = lifted.transform["z"] - gt["z"]
                        d = (dx * dx + dy * dy + dz * dz) ** 0.5
                        print(
                            f"           known-size: depth={lifted.depth_m:.3f}m "
                            f"err={d * 1000:.1f}mm  (dx={dx*1000:+.1f} dy={dy*1000:+.1f} dz={dz*1000:+.1f})"
                        )
                # PnP from the 4 detected corners.
                pnp = lift_pnp(det, rec["K_center"], rec["center_cam_optical_tf_base"], port_type=port_type)
                if pnp is not None and rec.get("port_tf_base") is not None:
                    gt = rec["port_tf_base"]
                    dx = pnp.transform["x"] - gt["x"]
                    dy = pnp.transform["y"] - gt["y"]
                    dz = pnp.transform["z"] - gt["z"]
                    d = (dx * dx + dy * dy + dz * dz) ** 0.5
                    print(
                        f"           pnp:        depth={pnp.depth_m:.3f}m "
                        f"err={d * 1000:.1f}mm  (dx={dx*1000:+.1f} dy={dy*1000:+.1f} dz={dz*1000:+.1f})"
                    )
                # Stereo: detect on right cam too and triangulate.
                right_jpg = jpg.with_name(jpg.name.replace("_center.jpg", "_right.jpg"))
                if right_jpg.exists():
                    img_r_bgr = cv2.imread(str(right_jpg))
                    img_r_rgb = cv2.cvtColor(img_r_bgr, cv2.COLOR_BGR2RGB)
                    det_r = detect_port(img_r_rgb, port_type)
                    if det_r is not None:
                        st = lift_stereo(
                            det,
                            det_r,
                            rec["K_center"],
                            rec.get("K_right", rec["K_center"]),
                            rec["center_cam_optical_tf_base"],
                            rec["right_cam_optical_tf_base"],
                        )
                        if st is not None and rec.get("port_tf_base") is not None:
                            gt = rec["port_tf_base"]
                            dx = st.transform["x"] - gt["x"]
                            dy = st.transform["y"] - gt["y"]
                            dz = st.transform["z"] - gt["z"]
                            d = (dx * dx + dy * dy + dz * dz) ** 0.5
                            print(
                                f"           stereo:     depth={st.depth_m:.3f}m "
                                f"err={d * 1000:.1f}mm  (dx={dx*1000:+.1f} dy={dy*1000:+.1f} dz={dz*1000:+.1f})"
                            )
            # Save overlay
            overlay = draw_detection(img_rgb, det)
            out_path = jpg.with_name(jpg.stem + ".det.jpg")
            cv2.imwrite(str(out_path), overlay)


if __name__ == "__main__":
    main()
