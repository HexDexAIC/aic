#!/usr/bin/env python3
import json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))
from aic_example_policies.ros.port_detector import detect_port

base = Path.home() / "aic_logs"
runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
trial = sorted(p for p in runs[-1].iterdir() if p.is_dir() and p.name == "trial_02_sfp")[0]
out_dir = Path.home() / "aic_t2_viz"
out_dir.mkdir(exist_ok=True)

for jpg in sorted(trial.glob("*_center.jpg"))[:3]:
    img_bgr = cv2.imread(str(jpg))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    json_p = jpg.with_name(jpg.name.replace("_center.jpg", ".json"))
    rec = json.loads(json_p.read_text())
    gt = rec["port_tf_base"]

    det = detect_port(img_rgb, "sfp")
    out = img_bgr.copy()
    if det is not None:
        x0 = int(det.cx - det.width / 2)
        y0 = int(det.cy - det.height / 2)
        x1 = int(det.cx + det.width / 2)
        y1 = int(det.cy + det.height / 2)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(out, f"det ({det.cx:.0f},{det.cy:.0f}) w={det.width:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(out, "NO DET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Project ground truth to image to see where it SHOULD be
    cam_tf = rec["center_cam_optical_tf_base"]
    K = rec["K_center"]
    fx, fy, cx_k, cy_k = K[0], K[4], K[2], K[5]
    # Compute T_cam_port = inv(T_base_cam) @ T_base_port
    def quat_to_R(qw, qx, qy, qz):
        return np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
    Tcam = np.eye(4); Tcam[:3, :3] = quat_to_R(cam_tf["qw"], cam_tf["qx"], cam_tf["qy"], cam_tf["qz"])
    Tcam[:3, 3] = [cam_tf["x"], cam_tf["y"], cam_tf["z"]]
    Tport = np.eye(4); Tport[:3, :3] = quat_to_R(gt["qw"], gt["qx"], gt["qy"], gt["qz"])
    Tport[:3, 3] = [gt["x"], gt["y"], gt["z"]]
    Tcam_port = np.linalg.inv(Tcam) @ Tport
    p = Tcam_port[:3, 3]
    if p[2] > 0:
        u = fx * p[0] / p[2] + cx_k
        v = fy * p[1] / p[2] + cy_k
        cv2.circle(out, (int(u), int(v)), 8, (0, 255, 0), 2)
        cv2.putText(out, f"GT ({u:.0f},{v:.0f})", (int(u) + 10, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(str(out_dir / jpg.name), out)
    print(jpg.name, det)
