"""Pick ~24 random samples from the exported YOLO dataset spanning splits,
episodes, and cameras. Render label overlay onto each. Save to a Windows-
accessible folder for easy browsing.
"""
import random
import re
from pathlib import Path
import cv2
import numpy as np

OUT_DIR = Path.home() / "aic_yolo_v1"
BROWSE_DIR = Path("/mnt/c/Users/Dell/aic_yolo_browse")
BROWSE_DIR.mkdir(parents=True, exist_ok=True)
for f in BROWSE_DIR.glob("*.jpg"):
    f.unlink()

random.seed(42)

# Pick: 8 train + 4 val + 8 test, varying eps and cams
def sample_split(split, n):
    files = list((OUT_DIR / "labels" / split).glob("*.txt"))
    if not files:
        return []
    return random.sample(files, min(n, len(files)))


def render_one(label_path: Path, dest: Path):
    img_path = OUT_DIR / "images" / label_path.parts[-2] / (label_path.stem + ".jpg")
    if not img_path.exists():
        return False
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h_img, w_img = img.shape[:2]

    for line in label_path.read_text().strip().split("\n"):
        if not line.strip(): continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
        kpts = [float(x) for x in parts[5:]]
        x0 = int((cx - bw/2) * w_img); x1 = int((cx + bw/2) * w_img)
        y0 = int((cy - bh/2) * h_img); y1 = int((cy + bh/2) * h_img)
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        label = "sfp_target" if cls == 0 else "sfp_distractor"
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, label, (x0, max(20, y0 - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for k in range(5):
            kx = kpts[k*3] * w_img
            ky = kpts[k*3 + 1] * h_img
            cv2.circle(img, (int(kx), int(ky)), 6, color, -1)
            cv2.putText(img, str(k), (int(kx)+8, int(ky)-6),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Add filename overlay
    m = re.match(r"ep(\d+)_fr(\d+)_(left|center|right)", label_path.stem)
    if m:
        ep, fr, cam = int(m.group(1)), int(m.group(2)), m.group(3)
        cv2.putText(img, f"ep{ep:03d} fr{fr:04d} {cam} [{label_path.parts[-2].upper()}]",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(dest), img)
    return True


for split, n in [("train", 8), ("val", 4), ("test", 8)]:
    samples = sample_split(split, n)
    for lbl in samples:
        out = BROWSE_DIR / f"{split}_{lbl.stem}.jpg"
        ok = render_one(lbl, out)
        if ok:
            print(f"saved {out.name}")

# Also save 4 close-distance test samples (likely 10-12cm) for harder cases
test_close = [f for f in (OUT_DIR / "labels" / "test").glob("ep*.txt")
               if int(re.match(r"ep(\d+)", f.stem).group(1)) in (210, 240, 260, 285)][:4]
for lbl in test_close:
    out = BROWSE_DIR / f"test_close_{lbl.stem}.jpg"
    ok = render_one(lbl, out)
    if ok:
        print(f"saved {out.name}")

print(f"\nAll samples in: {BROWSE_DIR}")
print("Open any of these with Photos / Chrome / IrfanView to inspect.")
