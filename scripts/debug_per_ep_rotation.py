#!/usr/bin/env python3
"""Compare TCP rotation at settle across eps + verify projection.

Theory: plug_offset is calibrated from a specific run. If the per-ep TCP
rotation at insertion differs (board_yaw + nic_yaw DR), the offset's
effect in BASE frame rotates with it. Applying the same T_TCP_plug should
still yield correct port pose, but only IF the TCP frame at settle is
defined consistently — which we should verify.

Also: re-derive port pose using ACTION target (not state actual) and see
if that better matches the user's annotations.
"""
import json
from pathlib import Path

import numpy as np

gt = json.loads((Path.home() / "aic_gt_port_poses.json").read_text())

# Compare ep0 vs ep47 settled TCP rotation
for ep in [0, 5, 30, 47, 60, 120, 180]:
    info = gt[str(ep)]
    R_state = np.array(info["actual_tcp_R"])
    R_action = np.array(info["action_target_R"])
    # Convert to Euler angles for inspection
    sy = np.sqrt(R_state[0,0]**2 + R_state[1,0]**2)
    if sy > 1e-6:
        roll = np.degrees(np.arctan2(R_state[2,1], R_state[2,2]))
        pitch = np.degrees(np.arctan2(-R_state[2,0], sy))
        yaw = np.degrees(np.arctan2(R_state[1,0], R_state[0,0]))
    else:
        roll = pitch = yaw = 0
    xyz = info["actual_tcp_xyz"]
    print(f"ep{ep:3d}: state TCP xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) "
          f"rpy=({roll:+6.1f}, {pitch:+6.1f}, {yaw:+6.1f}) deg")

# Now apply the CALIBRATED plug offset to ep47 settled TCP and project to all 3 cams
offset = json.loads((Path.home() / "aic_logs/tcp_to_plug_offset.json").read_text())["sfp"]


def quat_to_R(qw, qx, qy, qz):
    n = (qw*qw + qx*qx + qy*qy + qz*qz) ** 0.5
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])


T_TCP_plug = np.eye(4)
T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]
print(f"\nCalibrated T_TCP_plug:")
print(T_TCP_plug)

# For ep47, compute port pose from STATE and from ACTION
for tag, key_xyz, key_R in [("STATE", "actual_tcp_xyz", "actual_tcp_R"),
                              ("ACTION", "action_target_xyz", "action_target_R")]:
    T = np.eye(4)
    T[:3, :3] = np.array(gt["47"][key_R])
    T[:3, 3] = gt["47"][key_xyz]
    T_base_port = T @ T_TCP_plug
    print(f"\nep47 {tag}-derived port pose:")
    print(f"  xyz: {T_base_port[:3, 3]}")

# How much does ep47 settle TCP rotation differ from ep0?
R0 = np.array(gt["0"]["actual_tcp_R"])
R47 = np.array(gt["47"]["actual_tcp_R"])
R_rel = R0.T @ R47
ang = np.arccos((np.trace(R_rel) - 1) / 2)
print(f"\nep47 vs ep0 TCP rotation diff: {np.degrees(ang):.2f} deg")
