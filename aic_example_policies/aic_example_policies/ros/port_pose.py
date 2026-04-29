"""Lift 2D port detection -> 6D pose in base_link.

Block D of the AIC perception plan. Given:
  - a 2D PortDetection2D in the *center* wrist camera
  - the camera intrinsics K
  - the camera->base TF (e.g. base_link <- center_camera/optical)
  - knowledge of the port's physical size

we output a geometry_msgs/Transform compatible dict with port pose in
base_link. The orientation is approximate: we reuse the camera optical
forward axis as the port-normal direction (good enough for a downward
descent under impedance control; refined later via PnP if needed).

This module is dependency-light: no rclpy / ROS imports, just numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .port_detector import PortDetection2D


# Physical port-mouth dimensions per the SDF/CAD spec. The detector's coarse
# contour will extend past these into the housing rim — see
# shrink_corners_by_rim() in port_detector.py which compensates by shrinking
# the detected bbox by an empirically calibrated ratio toward center.
SFP_PORT_MOUTH_M = (0.0137, 0.0085)   # spec mouth (W, H)
SC_PORT_MOUTH_M = (0.005, 0.005)      # SC spec mouth, matches auto_label SC_MOUTH


def _intrinsics_from_K(K9):
    fx = K9[0]; fy = K9[4]; cx = K9[2]; cy = K9[5]
    return float(fx), float(fy), float(cx), float(cy)


def _quat_to_R(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _R_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def _tf_dict_to_T(d):
    if d is None:
        return None
    R = _quat_to_R((d["qw"], d["qx"], d["qy"], d["qz"]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [d["x"], d["y"], d["z"]]
    return T


def _T_to_tf_dict(T):
    qw, qx, qy, qz = _R_to_quat(T[:3, :3])
    return {
        "x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]),
        "qw": qw, "qx": qx, "qy": qy, "qz": qz,
    }


@dataclass
class PortPose6D:
    """6D port pose in base_link plus diagnostics."""
    transform: dict        # {x,y,z,qx,qy,qz,qw} in base_link
    depth_m: float         # estimated camera-frame z
    score: float           # 2D detection score
    method: str            # 'known_size' or 'stereo'


def _ray_in_base(det_uv, K, cam_optical_tf_base):
    """Return (origin_base, dir_base) for a ray from the camera through pixel (u,v)."""
    fx, fy, cx_k, cy_k = _intrinsics_from_K(K)
    u, v = det_uv
    d_cam = np.array([(u - cx_k) / fx, (v - cy_k) / fy, 1.0])
    d_cam = d_cam / np.linalg.norm(d_cam)
    T = _tf_dict_to_T(cam_optical_tf_base)
    R = T[:3, :3]
    o = T[:3, 3]
    d = R @ d_cam
    return o, d


def _closest_point_two_rays(o1, d1, o2, d2):
    """Midpoint of the closest approach between two rays.

    Returns (midpoint, dist) where dist is the minimum distance
    between the rays (gives a sense of triangulation quality).
    """
    d1d1 = float(d1 @ d1)
    d2d2 = float(d2 @ d2)
    d1d2 = float(d1 @ d2)
    do = o2 - o1
    A = np.array([[d1d1, -d1d2], [d1d2, -d2d2]])
    b = np.array([float(d1 @ do), float(d2 @ do)])
    try:
        ts = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, np.inf
    t, s = float(ts[0]), float(ts[1])
    p1 = o1 + t * d1
    p2 = o2 + s * d2
    mid = 0.5 * (p1 + p2)
    return mid, float(np.linalg.norm(p1 - p2))


def lift_pnp(
    det,
    K_flat,
    cam_optical_tf_base,
    port_type: Optional[str] = None,
) -> Optional["PortPose6D"]:
    """Use PnP from the detected 4 corners to recover full 6D port pose.

    Falls back to known-size if corners unavailable. The 3D corners are
    in port-link frame (z=0 mouth plane). The result is the port_link
    pose in base_link.
    """
    import cv2
    if det is None or det.corners_xy is None or cam_optical_tf_base is None:
        return None
    pt = (port_type or det.port_type).lower()
    if pt == "sfp":
        W, H = SFP_PORT_MOUTH_M
    elif pt == "sc":
        W, H = SC_PORT_MOUTH_M
    else:
        return None

    obj_pts = np.array([
        [-W / 2, -H / 2, 0],
        [+W / 2, -H / 2, 0],
        [+W / 2, +H / 2, 0],
        [-W / 2, +H / 2, 0],
    ], dtype=np.float32)
    img_pts = det.corners_xy.astype(np.float32)

    K_mat = np.array(K_flat, dtype=np.float32).reshape(3, 3)
    dist = np.zeros((5, 1), dtype=np.float32)

    ok, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K_mat, dist,
        flags=cv2.SOLVEPNP_IPPE,  # rectangle-friendly
    )
    if not ok:
        return None

    R_cam_port, _ = cv2.Rodrigues(rvec)
    T_cam_port = np.eye(4)
    T_cam_port[:3, :3] = R_cam_port
    T_cam_port[:3, 3] = tvec.ravel()

    T_base_cam = _tf_dict_to_T(cam_optical_tf_base)
    T_base_port = T_base_cam @ T_cam_port

    return PortPose6D(
        transform=_T_to_tf_dict(T_base_port),
        depth_m=float(tvec[2]),
        score=float(det.score),
        method="pnp",
    )


def lift_triangulate(
    detections,  # list of (det, K_flat, cam_optical_tf_base) tuples
    max_residual_m: float = 0.020,
) -> Optional["PortPose6D"]:
    """N-ray least-squares triangulation.

    For each ray (origin O_i, direction d_i normalized), the constraint is
        (I - d_i d_i^T) * P = (I - d_i d_i^T) * O_i
    Stacking all rays and solving in least-squares gives the point closest
    to all rays simultaneously. Returns None if the residual is too large
    (rays don't agree on a 3D point — e.g. detector picked different ports).
    """
    valid = [(d, K, T) for (d, K, T) in detections if d is not None and T is not None]
    if len(valid) < 2:
        return None

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for (det, K, cam_tf) in valid:
        o, d = _ray_in_base((det.cx, det.cy), K, cam_tf)
        d = d / np.linalg.norm(d)
        # Projection-onto-ray-perpendicular matrix: I - d d^T
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    try:
        P = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    # Compute mean perpendicular distance (residual) per ray; if rays don't
    # converge to a common point, reject.
    residuals = []
    for (det, K, cam_tf) in valid:
        o, d = _ray_in_base((det.cx, det.cy), K, cam_tf)
        d = d / np.linalg.norm(d)
        # perpendicular distance from P to ray = |(P - o) - ((P - o)·d) d|
        v = P - o
        v_perp = v - np.dot(v, d) * d
        residuals.append(np.linalg.norm(v_perp))
    if max(residuals) > max_residual_m:
        return None

    # Use the FIRST camera's optical orientation as the port orientation
    # (we don't get orientation from triangulation alone). Future improvement
    # would also do orientation from PnP averaging.
    T_base_cam0 = _tf_dict_to_T(valid[0][2])
    T_port = np.eye(4)
    T_port[:3, :3] = T_base_cam0[:3, :3]
    T_port[:3, 3] = P

    # Estimated depth in first cam's frame.
    p_in_cam = (np.linalg.inv(T_base_cam0) @ np.append(P, 1.0))[:3]

    # Score = mean detection score
    mean_score = float(np.mean([d.score for (d, _, _) in valid]))
    return PortPose6D(
        transform=_T_to_tf_dict(T_port),
        depth_m=float(p_in_cam[2]),
        score=mean_score,
        method=f"triangulate_{len(valid)}cam",
    )


def lift_stereo(
    det_center,
    det_right,
    K_center,
    K_right,
    cam_center_tf_base,
    cam_right_tf_base,
) -> Optional["PortPose6D"]:
    """Triangulate port position from center+right camera detections.

    Robust to known-size errors. Orientation is reused from the center
    camera optical frame, same as lift_to_base.
    """
    if det_center is None or det_right is None:
        return None
    if cam_center_tf_base is None or cam_right_tf_base is None:
        return None

    o1, d1 = _ray_in_base((det_center.cx, det_center.cy), K_center, cam_center_tf_base)
    o2, d2 = _ray_in_base((det_right.cx, det_right.cy), K_right, cam_right_tf_base)
    mid, dist = _closest_point_two_rays(o1, d1, o2, d2)
    if mid is None or dist > 0.05:
        return None  # rays don't agree -> reject
    # Build the 6D pose with center camera's orientation.
    T_base_cam = _tf_dict_to_T(cam_center_tf_base)
    T_port = np.eye(4)
    T_port[:3, :3] = T_base_cam[:3, :3]
    T_port[:3, 3] = mid

    # Estimated depth in center cam frame.
    p_in_cam = (np.linalg.inv(T_base_cam) @ np.append(mid, 1.0))[:3]
    depth = float(p_in_cam[2])

    return PortPose6D(
        transform=_T_to_tf_dict(T_port),
        depth_m=depth,
        score=0.5 * (det_center.score + det_right.score),
        method="stereo",
    )


def lift_to_base(
    det: PortDetection2D,
    K_center,
    cam_optical_tf_base,   # dict from JSON record (base_link <- center_camera/optical)
    port_type: Optional[str] = None,
) -> Optional[PortPose6D]:
    """Lift a 2D center-camera detection to a 6D base_link pose."""
    if det is None or cam_optical_tf_base is None:
        return None

    pt = (port_type or det.port_type).lower()
    if pt == "sfp":
        known_w_m, known_h_m = SFP_PORT_MOUTH_M
    elif pt == "sc":
        known_w_m, known_h_m = SC_PORT_MOUTH_M
    else:
        raise ValueError(f"Unknown port type {pt!r}")

    fx, fy, cx_k, cy_k = _intrinsics_from_K(K_center)

    # Known-size depth recovery. Use the longer side of the detection vs the
    # longer side of the physical port for stability.
    px_long = max(det.width, det.height)
    px_short = min(det.width, det.height)
    m_long = max(known_w_m, known_h_m)
    m_short = min(known_w_m, known_h_m)
    if px_long < 5:
        return None
    z_long = fx * m_long / px_long
    z_short = fx * m_short / max(px_short, 1)
    depth = 0.5 * (z_long + z_short)
    if depth <= 0 or depth > 2.0:
        return None

    # Unproject pixel center into camera optical frame.
    u, v = det.cx, det.cy
    x_c = (u - cx_k) * depth / fx
    y_c = (v - cy_k) * depth / fy
    z_c = depth
    p_cam = np.array([x_c, y_c, z_c, 1.0])

    # Camera optical frame -> base_link (we already have this TF in the log).
    T_base_cam = _tf_dict_to_T(cam_optical_tf_base)
    p_base = T_base_cam @ p_cam

    # Orientation: reuse camera-to-base orientation as a starting estimate.
    # The port normal in base_link approximates the camera optical-z direction.
    # Downstream impedance/insertion can refine via the PI controller already
    # in CheatCode.
    R_base_cam = T_base_cam[:3, :3]
    T_port = np.eye(4)
    T_port[:3, :3] = R_base_cam
    T_port[:3, 3] = p_base[:3]

    return PortPose6D(
        transform=_T_to_tf_dict(T_port),
        depth_m=float(depth),
        score=float(det.score),
        method="known_size",
    )
