#!/usr/bin/env python3
"""From a logged trial frame, extract the fixed T_TCP_optical for each camera.

The cameras are rigidly mounted on the wrist, so T_TCP_optical is constant
across all frames and across all runs. Pull it from one logged frame and
write to ~/aic_cam_tcp_offsets.json — used for projecting GT into LeRobot
dataset frames.

Also emits the K matrix, taken from the same log.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LOG = Path.home() / "aic_logs/20260425_232925/trial_01_sfp/00000.json"
OUT = Path.home() / "aic_cam_tcp_offsets.json"


def quat_to_R(qw, qx, qy, qz):
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def tf_to_T(d):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(d["qw"], d["qx"], d["qy"], d["qz"])
    T[:3, 3] = [d["x"], d["y"], d["z"]]
    return T


def T_to_tf(T):
    R = T[:3, :3]
    # Convert R to quat (manual)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = (tr + 1.0) ** 0.5 * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = ((1.0 + R[0, 0] - R[1, 1] - R[2, 2]) ** 0.5) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = ((1.0 + R[1, 1] - R[0, 0] - R[2, 2]) ** 0.5) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = ((1.0 + R[2, 2] - R[0, 0] - R[1, 1]) ** 0.5) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]),
            "qw": float(qw), "qx": float(qx), "qy": float(qy), "qz": float(qz)}


def main():
    rec = json.loads(LOG.read_text())
    T_base_tcp = tf_to_T(rec["tcp_tf_base"])
    out = {}
    for cam in ("left", "center", "right"):
        T_base_opt = tf_to_T(rec[f"{cam}_cam_optical_tf_base"])
        T_tcp_opt = np.linalg.inv(T_base_tcp) @ T_base_opt
        out[cam] = {
            "T_tcp_optical": T_tcp_opt.tolist(),
            "K": rec[f"K_{cam}"],
        }
        # Sanity print
        d = T_tcp_opt[:3, 3]
        print(f"{cam}: T_TCP_optical translation = ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}) m")

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
