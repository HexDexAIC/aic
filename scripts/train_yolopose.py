#!/usr/bin/env python3
"""Train YOLOv8n-pose on the AIC port keypoint dataset.

Usage:
  pixi run python scripts/train_yolopose.py [--data <yaml>] [--epochs N] [--imgsz N]

Defaults to ~/aic_dataset/dataset.yaml. The Ultralytics CLI handles model
download (yolov8n-pose.pt) on first run.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "aic_dataset" / "dataset.yaml"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")  # 0 = first GPU; 'cpu' for CPU
    ap.add_argument("--project", default=str(Path.home() / "aic_runs"))
    ap.add_argument("--name", default="yolopose_aic")
    ap.add_argument("--pretrained", default="yolov8n-pose.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.pretrained)
    print(f"Loaded base model: {args.pretrained}")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
        # Augmentation conservative — synthetic data already varied through
        # ground-truth runs, no need for aggressive distortions.
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
        flipud=0.0, fliplr=0.0,  # ports are orientation-meaningful
    )
    print(f"Training done. Best weights at: {results.best if hasattr(results, 'best') else 'see project dir'}")

    # Export ONNX for fast policy-side inference (no torch dependency).
    weights_dir = Path(args.project) / args.name / "weights"
    best = weights_dir / "best.pt"
    if best.exists():
        m = YOLO(str(best))
        m.export(format="onnx", imgsz=args.imgsz, simplify=True)
        print(f"Exported ONNX next to {best}")
    else:
        print(f"No best.pt at {best}; skipping ONNX export.")


if __name__ == "__main__":
    main()
