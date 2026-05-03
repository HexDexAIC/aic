#!/usr/bin/env python3
"""Test calc_gripper_pose computation against ground truth, using saved data.

Simulates what VisionInsert would compute in its descent loop, and prints
the target pose at each pfrac value. If targets look sensible (i.e. arm
should move toward port), the math is OK. If they look static or NaN,
the bug is in calc_gripper_pose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))

# Mock the rclpy + transforms3d so we can import VisionInsert-related code.
import types
sys.modules["rclpy"] = types.SimpleNamespace()
sys.modules["rclpy.duration"] = types.SimpleNamespace(Duration=lambda **kw: kw)
sys.modules["tf2_ros"] = types.SimpleNamespace(TransformException=Exception)
sys.modules["transforms3d"] = types.SimpleNamespace()
sys.modules["transforms3d._gohlketransforms"] = types.SimpleNamespace(
    quaternion_multiply=lambda a, b: tuple(np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ])),
    quaternion_slerp=lambda a, b, frac: tuple(
        a[i] * (1 - frac) + b[i] * frac for i in range(4)
    ),
)


from aic_example_policies.ros.port_pose import _quat_to_R, _R_to_quat, _tf_dict_to_T, _T_to_tf_dict


def calc_target_pose(port_tf, tcp_pose_obs, plug_offset, slerp_fraction=1.0,
                     position_fraction=1.0, z_offset=0.2):
    """Reproduce VisionInsert.calc_gripper_pose without ROS dependencies."""
    # T_tcp from observation
    T_tcp = np.eye(4)
    T_tcp[:3, :3] = _quat_to_R((tcp_pose_obs["qw"], tcp_pose_obs["qx"], tcp_pose_obs["qy"], tcp_pose_obs["qz"]))
    T_tcp[:3, 3] = [tcp_pose_obs["x"], tcp_pose_obs["y"], tcp_pose_obs["z"]]
    # T_plug = T_tcp @ offset
    T_offset = _tf_dict_to_T(plug_offset)
    T_plug = T_tcp @ T_offset

    gripper_xyz = (T_tcp[0, 3], T_tcp[1, 3], T_tcp[2, 3])
    plug_xyz = (T_plug[0, 3], T_plug[1, 3], T_plug[2, 3])

    qw_g, qx_g, qy_g, qz_g = _R_to_quat(T_tcp[:3, :3])
    qw_p, qx_p, qy_p, qz_p = _R_to_quat(T_plug[:3, :3])

    plug_tip_gripper_offset = (
        gripper_xyz[0] - plug_xyz[0],
        gripper_xyz[1] - plug_xyz[1],
        gripper_xyz[2] - plug_xyz[2],
    )

    port_xy = (port_tf["x"], port_tf["y"])
    target_x = port_xy[0]
    target_y = port_xy[1]
    target_z = port_tf["z"] + z_offset - plug_tip_gripper_offset[2]

    blend = (
        position_fraction * target_x + (1 - position_fraction) * gripper_xyz[0],
        position_fraction * target_y + (1 - position_fraction) * gripper_xyz[1],
        position_fraction * target_z + (1 - position_fraction) * gripper_xyz[2],
    )
    return blend, plug_xyz, gripper_xyz


def main():
    rec_p = Path.home() / "aic_logs/20260425_232925/trial_01_sfp/00000.json"
    rec = json.loads(rec_p.read_text())
    port_gt = rec["port_tf_base"]
    tcp_obs = rec["tcp_pose_obs"]

    offset = json.loads(Path.home().joinpath("aic_logs/tcp_to_plug_offset.json").read_text())["sfp"]

    print(f"Port (GT): x={port_gt['x']:.3f} y={port_gt['y']:.3f} z={port_gt['z']:.3f}")
    print(f"TCP (obs): x={tcp_obs['x']:.3f} y={tcp_obs['y']:.3f} z={tcp_obs['z']:.3f}")
    print(f"Offset: y={offset['y']:.3f} z={offset['z']:.3f}")
    print()

    for pfrac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        blend, plug, grip = calc_target_pose(port_gt, tcp_obs, offset,
                                             slerp_fraction=pfrac, position_fraction=pfrac, z_offset=0.2)
        print(f"  pfrac={pfrac}: target=({blend[0]:+.3f}, {blend[1]:+.3f}, {blend[2]:+.3f})  "
              f"plug=({plug[0]:+.3f}, {plug[1]:+.3f}, {plug[2]:+.3f})  "
              f"grip=({grip[0]:+.3f}, {grip[1]:+.3f}, {grip[2]:+.3f})")

    # Descent
    print("\n  --- DESCENT ---")
    for z_offset in [0.2, 0.1, 0.05, 0.0, -0.015]:
        blend, plug, grip = calc_target_pose(port_gt, tcp_obs, offset,
                                             z_offset=z_offset)
        print(f"  z_offset={z_offset:+.3f}: target_z={blend[2]:+.3f}")


if __name__ == "__main__":
    main()
