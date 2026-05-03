#!/usr/bin/env python3
"""Visualize SFP detector intermediates on Trial 2 frame 0."""
from pathlib import Path
import cv2
import numpy as np

p = Path.home() / "aic_logs/20260425_232925/trial_02_sfp/00000_center.jpg"
img_bgr = cv2.imread(str(p))
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Same masks as detect_sfp_port
not_green = cv2.bitwise_not(cv2.inRange(hsv, (35, 50, 30), (90, 255, 255)))
dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 70))
cand = cv2.bitwise_and(not_green, dark)

# Save intermediate masks
cv2.imwrite("/mnt/c/Users/Dell/aic/scripts/_t2_dark.jpg", dark)
cv2.imwrite("/mnt/c/Users/Dell/aic/scripts/_t2_cand.jpg", cand)

# How many candidate dark contours?
contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"candidate contours: {len(contours)}")
for i, c in enumerate(contours[:20]):
    area = cv2.contourArea(c)
    if area < 50:
        continue
    x, y, w, h = cv2.boundingRect(c)
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), ang = rect
    long_side = max(rw, rh)
    short_side = min(rw, rh)
    if short_side < 1: continue
    ar = long_side / short_side
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull) + 1e-6
    sol = area / hull_area
    print(f"  c{i}: area={area:.0f} bbox={x},{y} {w}x{h} cx,cy=({cx:.0f},{cy:.0f}) ar={ar:.2f} sol={sol:.2f}")

# Render top contours over image
out = img_bgr.copy()
out[cand > 0] = [0, 255, 255]
cv2.imwrite("/mnt/c/Users/Dell/aic/scripts/_t2_cand_overlay.jpg", out)
