"""Read YOLO labels and overlay them on images to sanity-check the export."""
import sys
from pathlib import Path
import cv2
import numpy as np

OUT = Path.home() / "aic_yolo_v1"
SAMPLE_STEMS = [
    "ep000_fr00000_left", "ep000_fr00000_center", "ep000_fr00000_right",
    "ep000_fr00045_center", "ep001_fr00050_center",
]

for stem in SAMPLE_STEMS:
    img_path = OUT / "images" / "train" / f"{stem}.jpg"
    lbl_path = OUT / "labels" / "train" / f"{stem}.txt"
    if not img_path.exists() or not lbl_path.exists():
        # try other distances
        candidates = list((OUT / "images" / "train").glob(f"{stem.split('_fr')[0]}_*"))
        if not candidates:
            continue
        img_path = candidates[0]
        lbl_path = OUT / "labels" / "train" / (img_path.stem + ".txt")
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h_img, w_img = img.shape[:2]

    for line in lbl_path.read_text().strip().split("\n"):
        parts = line.split()
        cls = int(parts[0])
        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
        kpts = [float(x) for x in parts[5:]]
        # Decode bbox
        x0 = int((cx - bw / 2) * w_img); x1 = int((cx + bw / 2) * w_img)
        y0 = int((cy - bh / 2) * h_img); y1 = int((cy + bh / 2) * h_img)
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        label = "sfp_target" if cls == 0 else "sfp_distractor"
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, label, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Decode 5 keypoints
        for k in range(5):
            kx = kpts[k*3] * w_img
            ky = kpts[k*3 + 1] * h_img
            v = int(kpts[k*3 + 2])
            cv2.circle(img, (int(kx), int(ky)), 5, color, -1)
            cv2.putText(img, str(k), (int(kx)+6, int(ky)-6),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out_path = Path(f"/mnt/c/Users/Dell/aic_yolo_check_{stem}.jpg")
    cv2.imwrite(str(out_path), img)
    print(f"saved {out_path}")
