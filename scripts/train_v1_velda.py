#!/usr/bin/env python3
"""Train YOLOv8s-pose on Velda instance for AIC v1 SFP entry-mouth detection.

Settings tuned for H100 80GB:
  - imgsz=1280
  - batch=32 (bigger than local — H100 has more VRAM)
  - workers=12
  - cache="ram"
  - amp=True
  - patience=15 (early stop sooner; H100 epochs are fast)
"""
import os
from pathlib import Path

if "WANDB_API_KEY" in os.environ:
    os.environ["WANDB_PROJECT"] = "aic-v1-sfp-pose"
    os.environ["WANDB_RUN_GROUP"] = "v1-velda-h100"
    os.environ.setdefault("WANDB_DISABLE_GIT", "true")

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from ultralytics import YOLO

DATA_YAML = Path("/home/user/aic_yolo_v1/data.yaml")
PROJECT = Path("/home/user/aic_runs")
PROJECT.mkdir(exist_ok=True)


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Data: {DATA_YAML}")

    model = YOLO("yolov8s-pose.pt")

    results = model.train(
        data=str(DATA_YAML),
        imgsz=1280,
        batch=32,           # H100 can handle bigger batch
        epochs=100,
        patience=15,
        workers=12,
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
        name="aic_v1_sfp_h100",
        exist_ok=False,
        plots=True,
        save=True,
        save_period=5,
        val=True,
        verbose=True,
    )
    print("Training complete.")
    print(f"Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
