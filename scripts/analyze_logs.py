#!/usr/bin/env python3
"""Quick analyzer for LoggingCheatCode dumps.

Reports:
  - Per-trial frame count + image encoding + intrinsics
  - Whether port_tf_base / plug_tf_base / tcp_tf_base are populated
  - Stability of (tcp -> plug_tip) offset across frames (Block B)
  - One sample frame's full record for sanity

Usage:  pixi run python scripts/analyze_logs.py [/home/dell/aic_logs/<TS>]
        (defaults to most-recent dir under ~/aic_logs)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_inv(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def tf_to_mat(d):
    if d is None:
        return None
    qw, qx, qy, qz = d["qw"], d["qx"], d["qy"], d["qz"]
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [d["x"], d["y"], d["z"]]
    return T


def mat_to_xyzq(T):
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(x), float(y), float(z), float(qx), float(qy), float(qz), float(qw)


def find_run_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    base = Path.home() / "aic_logs"
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        sys.exit(f"No runs under {base}")
    return runs[-1]


def report_trial(trial_dir: Path):
    print(f"\n=== {trial_dir.name} ===")
    task_path = trial_dir / "task.json"
    if task_path.exists():
        task = json.loads(task_path.read_text())
        print(f"  task: {task['cable_name']} -> {task['target_module_name']}/{task['port_name']} ({task['port_type']})")
    json_files = sorted(trial_dir.glob("[0-9]*.json"))
    if not json_files:
        print("  NO FRAMES")
        return
    print(f"  frames: {len(json_files)}")

    first = json.loads(json_files[0].read_text())
    print(f"  image: {first['image_w']}x{first['image_h']} encoding={first['image_encoding']}")
    print(f"  K_center: {first['K_center'][:3]}, {first['K_center'][3:6]}, {first['K_center'][6:]}")

    have_port = first["port_tf_base"] is not None
    have_plug = first["plug_tf_base"] is not None
    have_tcp = first["tcp_tf_base"] is not None
    print(f"  port_tf_base: {'OK' if have_port else 'NULL'}    plug_tf_base: {'OK' if have_plug else 'NULL'}    tcp_tf_base: {'OK' if have_tcp else 'NULL'}")
    if have_port:
        p = first["port_tf_base"]
        print(f"    port pose: x={p['x']:+.3f} y={p['y']:+.3f} z={p['z']:+.3f}  q=({p['qw']:+.3f},{p['qx']:+.3f},{p['qy']:+.3f},{p['qz']:+.3f})")

    # Block B: stability of TCP -> plug offset
    if have_tcp and have_plug:
        offsets = []
        for jf in json_files[::5]:  # subsample
            r = json.loads(jf.read_text())
            t = r["tcp_tf_base"]
            p = r["plug_tf_base"]
            if t is None or p is None:
                continue
            T_tcp = tf_to_mat(t)
            T_plug = tf_to_mat(p)
            T_off = np.linalg.inv(T_tcp) @ T_plug  # plug expressed in TCP frame
            offsets.append(T_off)
        if offsets:
            arr = np.stack([T[:3, 3] for T in offsets])
            mean_xyz = arr.mean(0)
            std_xyz = arr.std(0)
            print(f"  TCP->plug offset (TCP frame, mean): x={mean_xyz[0]:+.4f} y={mean_xyz[1]:+.4f} z={mean_xyz[2]:+.4f}")
            print(f"  TCP->plug offset (TCP frame, std):  x={std_xyz[0]:.4f}  y={std_xyz[1]:.4f}  z={std_xyz[2]:.4f}    [<1mm = stable]")
            # Print the median orientation as quaternion of an averaged rotation
            mid = offsets[len(offsets) // 2]
            x, y, z, qx, qy, qz, qw = mat_to_xyzq(mid)
            print(f"  TCP->plug offset orientation (median): q=({qw:+.4f},{qx:+.4f},{qy:+.4f},{qz:+.4f})")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_dir = find_run_dir(arg)
    print(f"Analyzing {run_dir}")
    trials = sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_"))
    if not trials:
        sys.exit(f"No trial_* dirs under {run_dir}")
    for t in trials:
        report_trial(t)

    # Scoring summary
    scoring = Path.home() / "aic_results" / "scoring.yaml"
    print("\n=== SCORING ===")
    if scoring.exists():
        text = scoring.read_text()
        # print first 80 lines max
        for i, line in enumerate(text.splitlines()):
            if i > 120:
                print("... (truncated)")
                break
            print(line)
    else:
        print(f"  {scoring} does not exist")


if __name__ == "__main__":
    main()
