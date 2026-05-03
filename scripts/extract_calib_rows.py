"""Extract individual rows from calibration verification grid for clear viewing."""
import cv2
from pathlib import Path
img = cv2.imread("/mnt/c/Users/Dell/aic_calib_top3_verification.jpg")
print(f"Grid shape: {img.shape}")
ROW_H = 480
n_rows = img.shape[0] // ROW_H
for i in range(min(n_rows, 6)):
    row = img[i*ROW_H:(i+1)*ROW_H]
    out = Path(f"/mnt/c/Users/Dell/aic_calib_row{i:02d}.jpg")
    cv2.imwrite(str(out), row)
    print(f"  saved {out}")
