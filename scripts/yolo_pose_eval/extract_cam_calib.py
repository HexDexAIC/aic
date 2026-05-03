#!/usr/bin/env python3
"""Extract per-camera (K, T_tcp_optical) from a sweep bag.

Run with the side-venv (~/ws_aic/src/aic/scripts/.venv) which has mcap +
mcap-ros2-support installed:
    ~/ws_aic/src/aic/scripts/.venv/bin/python extract_cam_calib.py \\
        <bag.mcap> <out.json>

Strategy:
  1. Read all /tf_static once into a dict {(parent, child): T_4x4}.
  2. Read /tf and /aic_controller/controller_state in time order until we
     have a fully populated dynamic TF chain plus a controller_state.
  3. Compose base_link -> {left,center,right}_camera/optical to get T_base_opt.
  4. Read T_base_tcp from controller_state.tcp_pose.
  5. T_tcp_optical = inv(T_base_tcp) @ T_base_optical (constant — wrist mount).
  6. K computed from URDF horizontal_fov:
        fx = fy = (W/2) / tan(hfov/2),  cx = (W-1)/2, cy = (H-1)/2
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


# Per Basler Camera macro (aic_assets/models/Basler Camera/basler_camera_macro.xacro)
IMG_W, IMG_H = 1152, 1024
HFOV_RAD = 0.8718


def K_from_fov(W: int, H: int, hfov: float) -> np.ndarray:
    fx = (W / 2.0) / math.tan(hfov / 2.0)
    fy = fx
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]])


def quat_to_R(qw, qx, qy, qz):
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


def tf_msg_to_T(tf):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(
        tf.transform.rotation.w, tf.transform.rotation.x,
        tf.transform.rotation.y, tf.transform.rotation.z)
    T[:3, 3] = [tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z]
    return T


def chain_lookup(static_tf, dynamic_tf, target_frame: str, base: str = "base_link"):
    """Walk parent -> child chain from `base` to `target_frame`.

    static_tf and dynamic_tf are dicts {(parent, child): T}; we union them
    (dynamic wins on collision). Returns T_base_target or None if no chain.
    """
    edges = dict(static_tf)
    edges.update(dynamic_tf)

    # Build reverse adjacency: child -> parent
    parents = {child: parent for (parent, child) in edges}

    # Walk back from target_frame to base
    chain = [target_frame]
    cur = target_frame
    while cur != base:
        if cur not in parents:
            return None
        cur = parents[cur]
        chain.append(cur)
    chain.reverse()  # base, ..., target

    # Compose
    T = np.eye(4)
    for parent, child in zip(chain[:-1], chain[1:]):
        T = T @ edges[(parent, child)]
    return T


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <bag.mcap> <out.json>", file=sys.stderr)
        sys.exit(2)
    bag_path = sys.argv[1]
    out_path = Path(sys.argv[2])

    static_tf = {}        # {(parent, child): T}
    dynamic_tf = {}       # latest /tf snapshot
    tcp_pose = None
    saw_controller_state = False
    n_dyn_msgs = 0

    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        # Pass 1: collect static TFs
        for _, channel, _, dec_msg in reader.iter_decoded_messages(topics=["/tf_static"]):
            for tf in dec_msg.transforms:
                static_tf[(tf.header.frame_id, tf.child_frame_id)] = tf_msg_to_T(tf)
        print(f"Collected {len(static_tf)} static TF edges.")

    # Pass 2: get dynamic TFs and a controller_state
    with open(bag_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, msg, dec_msg in reader.iter_decoded_messages(
                topics=["/tf", "/aic_controller/controller_state"]):
            if channel.topic == "/tf":
                for tf in dec_msg.transforms:
                    dynamic_tf[(tf.header.frame_id, tf.child_frame_id)] = tf_msg_to_T(tf)
                n_dyn_msgs += 1
            elif channel.topic == "/aic_controller/controller_state":
                if not saw_controller_state:
                    tp = dec_msg.tcp_pose
                    T_base_tcp = np.eye(4)
                    T_base_tcp[:3, :3] = quat_to_R(
                        tp.orientation.w, tp.orientation.x,
                        tp.orientation.y, tp.orientation.z)
                    T_base_tcp[:3, 3] = [tp.position.x, tp.position.y, tp.position.z]
                    tcp_pose = T_base_tcp
                    saw_controller_state = True
            # Stop once we have controller_state and at least 200 dynamic msgs
            # (enough to settle dynamic chain)
            if saw_controller_state and n_dyn_msgs > 500:
                break

    if not saw_controller_state:
        print("ERROR: no controller_state in bag", file=sys.stderr); sys.exit(1)
    print(f"Collected {len(dynamic_tf)} dynamic TF edges (after {n_dyn_msgs} /tf msgs).")
    print(f"T_base_tcp = {tcp_pose[:3, 3]}")

    K = K_from_fov(IMG_W, IMG_H, HFOV_RAD)
    print(f"K (from URDF FOV):\n{K}")

    out = {}
    for cam in ("left", "center", "right"):
        opt_frame = f"{cam}_camera/optical"
        T_base_opt = chain_lookup(static_tf, dynamic_tf, opt_frame, base="base_link")
        if T_base_opt is None:
            print(f"  WARN: no TF chain to {opt_frame}", file=sys.stderr)
            continue
        T_tcp_opt = np.linalg.inv(tcp_pose) @ T_base_opt
        d = T_tcp_opt[:3, 3]
        print(f"  {cam}: T_TCP_optical translation = "
              f"({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}) m")
        out[cam] = {
            "T_tcp_optical": T_tcp_opt.tolist(),
            "K": K.flatten().tolist(),
            "img_w": IMG_W,
            "img_h": IMG_H,
        }

    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
