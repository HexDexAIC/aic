#!/usr/bin/env python3
"""Sample HSV values from a known image region to calibrate detector thresholds."""
import sys
from pathlib import Path
import cv2
import numpy as np
import json

base = Path.home() / "aic_logs"
runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
trial = sorted(p for p in runs[-1].iterdir() if p.is_dir() and p.name == "trial_03_sc")[0]

img_bgr = cv2.imread(str(trial / "00000_center.jpg"))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Sample from a few approximate locations on the blue connector visible in image
# (eyeballed from earlier viz; SC port at upper-left of frame)
coords = [(245, 215), (250, 220), (240, 220), (245, 230), (250, 215)]
for cx, cy in coords:
    if cx < img_bgr.shape[1] and cy < img_bgr.shape[0]:
        bgr = img_bgr[cy, cx]
        rgb = img_rgb[cy, cx]
        h, s, v = hsv[cy, cx]
        print(f"({cx:>4},{cy:>4}): BGR={tuple(int(x) for x in bgr)}  HSV=({h},{s},{v})")

# Also sample center of detected (286,275) to see what's there
print("\nDetected (286,275):")
h, s, v = hsv[275, 286]
print(f"  HSV=({h},{s},{v}) BGR={tuple(int(x) for x in img_bgr[275, 286])}")

# Sample over a 300x300 ROI to find blue-ish pixels
print("\nBlue regions in upper-left quadrant:")
hsv_sub = hsv[:300, :400]
for hue_low, hue_high in [(85, 135), (95, 130), (100, 125)]:
    mask = cv2.inRange(hsv_sub, (hue_low, 50, 50), (hue_high, 255, 255))
    n = (mask > 0).sum()
    print(f"  H[{hue_low},{hue_high}], S>=50, V>=50: {n} pixels")
