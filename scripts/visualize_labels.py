#!/usr/bin/env python3
"""Visualize a few YOLO-format labels overlaid on their images.

Sanity check that auto_label.py is producing geometrically correct keypoints.
Writes annotated copies to <dataset>/viz/.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


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
        v = int(rest[i + 2])
        kps.append((kx, ky, v))
    return cls, (cx, cy, bw, bh), kps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "aic_dataset"))
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    data = Path(args.data)
    images_dir = data / "images"
    labels_dir = data / "labels"
    viz_dir = data / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        print(f"No labels under {labels_dir}")
        return
    sample = random.sample(label_files, min(args.n, len(label_files)))

    for lp in sample:
        ip = images_dir / (lp.stem + ".jpg")
        if not ip.exists():
            continue
        img = cv2.imread(str(ip))
        h, w = img.shape[:2]
        for line in lp.read_text().splitlines():
            if not line:
                continue
            cls, (cx, cy, bw, bh), kps = parse_label(line, w, h)
            x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
            x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
            color = (0, 255, 0) if cls == 0 else (0, 200, 255)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            for i, (kx, ky, v) in enumerate(kps):
                col = (0, 0, 255) if i < 4 else (255, 0, 0)
                cv2.circle(img, (int(kx), int(ky)), 4, col, -1)
                cv2.putText(img, str(i), (int(kx) + 5, int(ky)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        out = viz_dir / lp.with_suffix(".jpg").name
        cv2.imwrite(str(out), img)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
