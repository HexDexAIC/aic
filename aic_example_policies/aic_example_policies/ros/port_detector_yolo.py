"""YOLOv8-pose-based port detector.

Loads an ONNX export of a YOLOv8n-pose model trained on
~/aic_dataset/dataset.yaml. Runs inference on a single RGB image,
returns a PortDetection2D compatible with port_pose.lift_pnp.

Falls back gracefully when the model file is missing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from .port_detector import PortDetection2D


CLASS_NAMES = ["sfp", "sc"]


class YoloPosePortDetector:
    def __init__(self, weights_path: Optional[str] = None, imgsz: int = 640, conf: float = 0.3):
        self._imgsz = imgsz
        self._conf = conf
        if weights_path is None:
            # Default: prefer the full-data run, fall back to whatever exists.
            candidates = [
                Path.home() / "aic_runs" / "yolopose_full" / "weights" / "best.onnx",
                Path.home() / "aic_runs" / "yolopose_aic" / "weights" / "best.onnx",
                Path.home() / "aic_runs" / "yolo_smoke" / "weights" / "best.onnx",
            ]
            weights_path = ""
            for c in candidates:
                if c.exists():
                    weights_path = str(c)
                    break
        self._weights_path = weights_path

        if not os.path.exists(weights_path):
            self._sess = None
            return

        import onnxruntime as ort
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self._sess = ort.InferenceSession(weights_path, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._input_shape = self._sess.get_inputs()[0].shape
        # Standard YOLOv8 export: [1, 3, H, W]

    @property
    def available(self) -> bool:
        return self._sess is not None

    def _letterbox(self, img: np.ndarray):
        h, w = img.shape[:2]
        s = self._imgsz / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._imgsz, self._imgsz, 3), 114, dtype=np.uint8)
        ph, pw = (self._imgsz - nh) // 2, (self._imgsz - nw) // 2
        canvas[ph:ph + nh, pw:pw + nw] = resized
        return canvas, s, ph, pw

    def detect(self, image_rgb: np.ndarray, port_type: str) -> Optional[PortDetection2D]:
        if self._sess is None:
            return None

        h, w = image_rgb.shape[:2]
        canvas, s, ph, pw = self._letterbox(image_rgb)

        x = canvas.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]  # (1,3,H,W)

        out = self._sess.run(None, {self._input_name: x})[0]
        # YOLOv8-pose ONNX raw output: (1, 4 + nc + 3*kpts, num_anchors).
        # We prefer to filter ourselves. For 2 classes, 5 keypoints:
        # rows: bx, by, bw, bh, cls0, cls1, kp0_x, kp0_y, kp0_v, kp1_x, ...
        out = out[0]  # (channels, anchors)
        out = out.T   # (anchors, channels)
        nc = len(CLASS_NAMES)
        cls_scores = out[:, 4:4 + nc]
        cls_id = cls_scores.argmax(axis=1)
        conf = cls_scores.max(axis=1)

        target_cls = CLASS_NAMES.index(port_type) if port_type in CLASS_NAMES else None
        mask = conf >= self._conf
        if target_cls is not None:
            mask &= (cls_id == target_cls)
        if not mask.any():
            return None

        idx = np.where(mask)[0]
        # Pick highest confidence.
        best = idx[conf[idx].argmax()]
        bx, by, bw, bh = out[best, 0:4]
        kp_block = out[best, 4 + nc:]
        # Reshape into (5, 3): x, y, vis
        kps = kp_block.reshape(-1, 3)
        if kps.shape[0] < 4:
            return None

        # Undo letterbox: convert from canvas space back to original image.
        def unbox(u, v):
            return ((u - pw) / s, (v - ph) / s)

        cx, cy = unbox(bx, by)
        rw, rh = bw / s, bh / s

        corners_uv = np.array([unbox(kp[0], kp[1]) for kp in kps[:4]], dtype=np.float32)

        return PortDetection2D(
            port_type=port_type,
            cx=float(cx), cy=float(cy),
            width=float(rw), height=float(rh),
            angle_deg=0.0,
            score=float(conf[best]),
            corners_xy=corners_uv,
        )
