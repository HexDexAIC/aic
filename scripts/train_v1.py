#!/usr/bin/env python3
"""Train YOLOv8s-pose for AIC v1 SFP entry-mouth detection.

Settings tuned for RTX 3080 Ti Laptop (16 GB VRAM):
  - imgsz=1280 (preserves small SFP slot detail)
  - batch=16
  - cache="ram" (1.9 GB dataset fits comfortably)
  - amp=True (mixed precision)
  - workers=8
  - augmentations: light photometric only (no mosaic/flip/rotate)
  - patience=20 (early stop)
"""
import os
import sys
from pathlib import Path

# wandb auth via env var (set by launcher; never written to disk by us)
if "WANDB_API_KEY" in os.environ:
    os.environ["WANDB_PROJECT"] = "aic-v1-sfp-pose"
    os.environ["WANDB_RUN_GROUP"] = "v1"
    # Tell Ultralytics to use wandb logger
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")  # avoid weird WSL git lookups

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from ultralytics import YOLO
import ultralytics.utils as ul_utils

# Ensure wandb integration is on
try:
    from ultralytics.utils.callbacks import wb as wb_cb  # noqa
    print("[ok] ultralytics wandb callback available")
except Exception as e:
    print(f"[warn] wandb integration may not be active: {e}")

DATA_YAML = Path.home() / "aic_yolo_v1" / "data.yaml"
PROJECT = Path.home() / "aic_runs"
PROJECT.mkdir(exist_ok=True)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Data: {DATA_YAML}")

    model = YOLO("yolov8s-pose.pt")  # pretrained on COCO keypoints

    results = model.train(
        data=str(DATA_YAML),
        imgsz=1280,
        batch=16,
        epochs=100,
        patience=20,
        workers=8,
        device=0,
        cache="ram",
        amp=True,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        close_mosaic=0,
        warmup_epochs=3.0,

        # Augmentations: keep keypoint geometry intact
        mosaic=0.0,
        mixup=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.15,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        project=str(PROJECT),
        name="aic_v1_sfp",
        exist_ok=False,
        plots=True,
        save=True,
        save_period=5,
        val=True,

        # Resume-friendly
        verbose=True,
    )
    print("Training complete.")
    print(f"Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
