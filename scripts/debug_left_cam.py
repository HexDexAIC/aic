#!/usr/bin/env python3
"""Show what dark contours exist on a left-cam frame and why none pass classical's gates."""
import sys
from pathlib import Path
import cv2
import numpy as np

p = Path.home() / "aic_logs/20260425_232925/trial_01_sfp/00006_left.jpg"
img_bgr = cv2.imread(str(p))
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

not_green = cv2.bitwise_not(cv2.inRange(hsv, (35, 50, 30), (90, 255, 255)))
dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 70))
cand = cv2.bitwise_and(not_green, dark)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Total contours: {len(contours)}")
h, w = img_bgr.shape[:2]
y_max = h * 0.60
print(f"Image size {w}x{h}, y_max={y_max:.0f}")

# Print all contours that have any area, sorted
keepers = []
for c in contours:
    area = cv2.contourArea(c)
    if area < 50:
        continue
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), ang = rect
    if rw < 1 or rh < 1: continue
    long_side = max(rw, rh)
    short_side = min(rw, rh)
    ar = long_side / short_side
    hull = cv2.convexHull(c)
    sol = area / (cv2.contourArea(hull) + 1e-6)
    in_roi = "Y" if cy <= y_max else "N (gripper)"
    aspect_ok = "Y" if 1.2 <= ar <= 2.5 else "N"
    area_ok = "Y" if 800 <= area <= 200_000 else "N"
    sol_ok = "Y" if sol >= 0.6 else "N"
    keepers.append((area, cx, cy, rw, rh, ar, sol, in_roi, aspect_ok, area_ok, sol_ok))

keepers.sort(reverse=True)
print(f"\n{'area':>7} {'cx':>4} {'cy':>4} {'w':>4} {'h':>4} {'ar':>5} {'sol':>5} ROI ar area sol")
for k in keepers[:15]:
    area, cx, cy, rw, rh, ar, sol, in_roi, aspect_ok, area_ok, sol_ok = k
    print(f"{area:>7.0f} {cx:>4.0f} {cy:>4.0f} {rw:>4.0f} {rh:>4.0f} {ar:>5.2f} {sol:>5.2f}  {in_roi:<13} {aspect_ok}  {area_ok}  {sol_ok}")

# Save the candidate mask + originalfor inspection
overlay = img_bgr.copy()
overlay[cand > 0] = (0, 255, 255)
cv2.line(overlay, (0, int(y_max)), (w, int(y_max)), (255, 0, 0), 2)
cv2.imwrite("/mnt/c/Users/Dell/aic/scripts/_left_cand.jpg", overlay)
print("\nSaved candidate mask viz")
