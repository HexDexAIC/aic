#!/usr/bin/env python3
"""Find where the blue connector actually is in the image."""
import sys
from pathlib import Path
import cv2
import numpy as np

base = Path.home() / "aic_logs"
runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
trial = sorted(p for p in runs[-1].iterdir() if p.is_dir() and p.name == "trial_03_sc")[0]
img_bgr = cv2.imread(str(trial / "00000_center.jpg"))
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, (90, 80, 60), (130, 255, 255))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(f"Found {len(contours)} blue contours")
for i, c in enumerate(contours[:10]):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    print(f"  blob{i}: area={area:.0f} bbox=({x},{y}) {w}x{h}, center=({x+w//2},{y+h//2})")

# Save mask for visualization
out = img_bgr.copy()
out[mask > 0] = [0, 255, 0]
cv2.imwrite("/mnt/c/Users/Dell/aic/scripts/_blue_mask.jpg", out)

# Also save the largest contour bbox
if contours:
    x, y, w, h = cv2.boundingRect(contours[0])
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.imwrite("/mnt/c/Users/Dell/aic/scripts/_blue_mask.jpg", out)
print("\nSaved mask viz")
