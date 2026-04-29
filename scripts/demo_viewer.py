#!/usr/bin/env python3
"""Live port detection demo viewer.

Subscribes to the 3 wrist cameras + observations from a running aic_eval
container, runs the smart port detector on each frame, and serves an
auto-refreshing HTML page showing the annotated image.

Usage:
    pixi run python scripts/demo_viewer.py [--port 8765] [--type sfp|sc]

Then open http://localhost:8765/ in your browser.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from aic_model_interfaces.msg import Observation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))
from aic_example_policies.ros.port_detector import detect_port as detect_port_classical, draw_detection
from aic_example_policies.ros.port_detector_yolo import YoloPosePortDetector
from aic_example_policies.ros.port_pose import lift_pnp


YOLO = None
LATEST_FRAME = {"jpg": None, "info": "(waiting for first frame)"}


def annotate(img_rgb, port_type):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    yolo_det = YOLO.detect(img_rgb, port_type) if (YOLO and YOLO.available) else None
    classical_det = detect_port_classical(img_rgb, port_type, refine=True)

    info_lines = [f"port_type={port_type}"]
    # Draw classical (red)
    if classical_det is not None:
        if classical_det.corners_xy is not None:
            for (x, y) in classical_det.corners_xy:
                cv2.circle(img_bgr, (int(x), int(y)), 4, (0, 0, 255), -1)
            box = classical_det.corners_xy.astype(np.int32)
            cv2.polylines(img_bgr, [box], True, (0, 0, 255), 2)
        info_lines.append(f"classical: cx={classical_det.cx:.0f},cy={classical_det.cy:.0f} score={classical_det.score:.2f}")
    else:
        info_lines.append("classical: NO DET")

    # Draw YOLO (green)
    if yolo_det is not None:
        if yolo_det.corners_xy is not None:
            for (x, y) in yolo_det.corners_xy:
                cv2.circle(img_bgr, (int(x), int(y)), 5, (0, 255, 0), -1)
            box = yolo_det.corners_xy.astype(np.int32)
            cv2.polylines(img_bgr, [box], True, (0, 255, 0), 2)
        info_lines.append(f"yolo: cx={yolo_det.cx:.0f},cy={yolo_det.cy:.0f} conf={yolo_det.score:.2f}")
    else:
        info_lines.append("yolo: NO DET")

    # Overlay text
    for i, line in enumerate(info_lines):
        cv2.putText(img_bgr, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(img_bgr, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 1)
    legend = "RED=classical, GREEN=yolo"
    cv2.putText(img_bgr, legend, (10, img_bgr.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img_bgr, "\n".join(info_lines)


class DetNode(Node):
    def __init__(self, port_type):
        super().__init__("port_detection_demo")
        self.port_type = port_type
        # Either /center_camera/image directly, or via /observations
        self.create_subscription(Observation, "/observations", self._on_obs, 1)
        self.create_subscription(Image, "/center_camera/image", self._on_img, 1)
        self.get_logger().info(f"Demo viewer started; port_type={port_type}")

    def _process(self, img_msg):
        try:
            arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            )
            if img_msg.encoding == "bgr8":
                rgb = arr[:, :, ::-1]
            else:
                rgb = arr
            annotated, info = annotate(rgb.copy(), self.port_type)
            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                LATEST_FRAME["jpg"] = jpg.tobytes()
                LATEST_FRAME["info"] = info
        except Exception as e:
            self.get_logger().warn(f"process error: {e}")

    def _on_obs(self, msg):
        self._process(msg.center_image)

    def _on_img(self, msg):
        self._process(msg)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            html = ("<!doctype html><html><body style='background:#222;color:#eee;font-family:monospace'>"
                    "<h2>AIC Port Detection - Live</h2>"
                    "<p>RED=classical, GREEN=yolo. Auto-refreshes every 500ms.</p>"
                    "<img id='frame' src='/frame.jpg' style='max-width:90%' />"
                    "<pre id='info'></pre>"
                    "<script>"
                    "setInterval(async () => {"
                    "  const ts = Date.now();"
                    "  document.getElementById('frame').src = '/frame.jpg?t=' + ts;"
                    "  const r = await fetch('/info?t=' + ts);"
                    "  document.getElementById('info').textContent = await r.text();"
                    "}, 500);"
                    "</script></body></html>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path.startswith("/frame.jpg"):
            jpg = LATEST_FRAME.get("jpg")
            if jpg is None:
                placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
                cv2.putText(placeholder, "waiting for frame...", (40, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                _, jpgb = cv2.imencode(".jpg", placeholder)
                jpg = jpgb.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(jpg)
        elif self.path.startswith("/info"):
            info = LATEST_FRAME.get("info", "")
            data = info.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # silence access logs


def main():
    global YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--type", default="sfp", choices=["sfp", "sc"])
    args = ap.parse_args()

    YOLO = YoloPosePortDetector(conf=0.3)
    if not YOLO.available:
        print("WARN: YOLO model not available; classical only.")

    rclpy.init()
    node = DetNode(args.type)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    print(f"Demo viewer listening on http://localhost:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
