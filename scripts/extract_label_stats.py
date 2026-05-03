#!/usr/bin/env python3
"""Compute target / distractor pixel-location and size distributions
per camera from user audit annotations. Used to derive better priors
and ROI gates for the classical detector.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = Path.home() / "aic_audit_annotations.json"
    d = json.loads(p.read_text())

    by_cam_label = defaultdict(list)
    for k, v in d.items():
        cam = v["camera"]
        for b in v["boxes"]:
            label = b.get("label", "?")
            x0, y0, x1, y1 = b["bbox_xyxy"]
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            w = x1 - x0
            h = y1 - y0
            by_cam_label[(cam, label)].append({
                "cx": cx, "cy": cy, "w": w, "h": h,
                "bbox": (x0, y0, x1, y1),
                "ep": v["episode"], "fr": v["frame"],
            })

    print(f"{'camera':>8s}  {'label':>11s}  {'n':>3s}  cx [mn,mx,mean]  cy [mn,mx,mean]  w [mn,mx]  h [mn,mx]")
    for (cam, label), entries in sorted(by_cam_label.items()):
        if not entries:
            continue
        cxs = [e["cx"] for e in entries]
        cys = [e["cy"] for e in entries]
        ws  = [e["w"] for e in entries]
        hs  = [e["h"] for e in entries]
        print(f"{cam:>8s}  {label:>11s}  {len(entries):>3d}  "
              f"cx [{min(cxs):.0f},{max(cxs):.0f},{np.mean(cxs):.0f}]  "
              f"cy [{min(cys):.0f},{max(cys):.0f},{np.mean(cys):.0f}]  "
              f"w [{min(ws):.0f},{max(ws):.0f}]  h [{min(hs):.0f},{max(hs):.0f}]")

    # Also dump as JSON for the detector to consume
    out = {}
    for (cam, label), entries in by_cam_label.items():
        if label != "target":
            continue
        cxs = np.array([e["cx"] for e in entries])
        cys = np.array([e["cy"] for e in entries])
        ws = np.array([e["w"] for e in entries])
        hs = np.array([e["h"] for e in entries])
        out[cam] = {
            "n": int(len(entries)),
            "cx_mean": float(cxs.mean()), "cx_std": float(cxs.std()),
            "cy_mean": float(cys.mean()), "cy_std": float(cys.std()),
            "w_min": float(ws.min()), "w_max": float(ws.max()), "w_mean": float(ws.mean()),
            "h_min": float(hs.min()), "h_max": float(hs.max()), "h_mean": float(hs.mean()),
            "area_min": float((ws * hs).min()), "area_max": float((ws * hs).max()),
        }
    out_path = Path.home() / "aic_target_priors.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved per-cam target priors to: {out_path}")
    print("(Use cx_mean, cy_mean as search center; ~3*std as radius;")
    print(" area_min/max for size gate.)")


if __name__ == "__main__":
    main()
