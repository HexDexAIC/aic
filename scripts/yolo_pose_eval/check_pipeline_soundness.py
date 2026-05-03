#!/usr/bin/env python3
"""Soundness checks for the YOLO+PnP pipeline.

Independent of the metrics. Verifies geometric consistency end-to-end:

  Check 1: GT mouth corners projected back into image align with YOLO
           predicted keypoints (within a few pixels). If not, K, T_tcp_optical,
           port-link-to-mouth offset, or rot6 decoding has a sign/order error.

  Check 2: Z-axis of the port mouth in base_link is roughly antiparallel
           to base_link Z (port faces upward). Catches a flipped sign in
           the link-to-mouth offset (if I used +0.0458 instead of -0.0458,
           the mouth would be on the wrong side of port_link).

  Check 3: Each camera optical Z-axis in base_link is roughly along
           base_link -Z (cameras pointing down at task-board) when TCP
           is at hover pose. Catches T_tcp_optical sign errors.

  Check 4: For a known-success episode, the FINAL TCP position should
           be very close to (port_xyz + tcp_to_plug_offset_z + plug_seat_depth).
           Catches gross frame errors in the dataset itself.

  Check 5: PnP rvec/tvec on a synthetic perfect-corners projection
           recovers the ground truth. Tests the PnP module in isolation.

  Check 6: Rendered overlays for one episode at multiple phases —
           saves PNGs to /tmp/soundness/ for visual inspection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "aic_example_policies"))
from aic_example_policies.perception.port_pose_v2.pnp import (  # noqa: E402
    PnPConfig, estimate_pose, OBJECT_POINTS_4, OBJECT_POINTS_CENTER,
)

DATASET_ROOT = Path("/home/hariharan/aic_results/aic-sfp-500-pr")
WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
CAM_CALIB = Path.home() / "aic_cam_tcp_offsets.json"
OUT_DIR = Path("/tmp/soundness")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rot6_to_R(a1, a2):
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / max(np.linalg.norm(b2), 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def state_to_T(state):
    T = np.eye(4)
    T[:3, :3] = rot6_to_R(state[3:6], state[6:9])
    T[:3, 3] = state[0:3]
    return T


def port_gt_to_T(port):
    T = np.eye(4)
    T[:3, :3] = rot6_to_R(port[3:6], port[6:9])
    T[:3, 3] = port[0:3]
    return T


PORT_LINK_TO_MOUTH_DZ = -0.0458


def gt_link_to_mouth(T_base_port_link):
    T_offset = np.eye(4)
    T_offset[2, 3] = PORT_LINK_TO_MOUTH_DZ
    return T_base_port_link @ T_offset


# Same as the canonical OBJECT_POINTS in pnp.py — port-mouth frame, z=0
MOUTH_CORNERS_5 = np.vstack([OBJECT_POINTS_4, OBJECT_POINTS_CENTER.reshape(1, 3)])


def project_world_to_image(K, T_optical_world, points_world):
    """Project Nx3 world points → Nx2 image points via T_optical_world (4x4) and K."""
    n = points_world.shape[0]
    p_h = np.hstack([points_world, np.ones((n, 1))])
    p_cam = (T_optical_world @ p_h.T).T[:, :3]
    z = np.maximum(p_cam[:, 2:3], 1e-6)
    p_norm = p_cam[:, :2] / z
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.stack([fx * p_norm[:, 0] + cx, fy * p_norm[:, 1] + cy], axis=1)


# ---------- Check 2 + 3 + 4 (no YOLO needed) ----------

def check_axes_and_geometry(ds, ep, T_tcp_opt_per_cam):
    """Check that port frame Z and camera Z point sensibly."""
    print("\n=== Check 2-4: axes and geometry ===")
    start = int(ds.meta.episodes[ep]["dataset_from_index"])
    L = int(ds.meta.episodes[ep]["length"])
    s0 = ds[start]
    s_last = ds[start + L - 1]

    state0 = s0["observation.state"].numpy()
    state_last = s_last["observation.state"].numpy()
    port_gt = s0["observation.port_pose_gt"].numpy()

    T_base_tcp_0 = state_to_T(state0)
    T_base_tcp_end = state_to_T(state_last)
    T_base_port_link = port_gt_to_T(port_gt)
    T_base_mouth = gt_link_to_mouth(T_base_port_link)

    # --- Check 2: port mouth Z axis in base_link (hopefully ~ +Z = up)
    R_mouth = T_base_mouth[:3, :3]
    z_axis_world = R_mouth[:, 2]
    print(f"  Port-mouth Z-axis in base_link: ({z_axis_world[0]:+.3f}, "
          f"{z_axis_world[1]:+.3f}, {z_axis_world[2]:+.3f})")
    print(f"    → {'✓ points UP' if z_axis_world[2] > 0.9 else '✗ NOT pointing up'} "
          f"(expected +Z)")

    # --- Check 3: camera optical Z in base_link (hopefully -Z = down at hover)
    for cam in ("left", "center", "right"):
        T_base_opt = T_base_tcp_0 @ T_tcp_opt_per_cam[cam]
        z_cam_world = T_base_opt[:3, 2]
        print(f"  Cam {cam} Z-axis in base_link at hover: ({z_cam_world[0]:+.3f}, "
              f"{z_cam_world[1]:+.3f}, {z_cam_world[2]:+.3f})")
        print(f"    → {'✓ pointing DOWN' if z_cam_world[2] < -0.9 else '✗ NOT pointing down'}")

    # --- Check 4: at last frame TCP should be near the port mouth
    final_tcp_xyz = T_base_tcp_end[:3, 3]
    mouth_xyz = T_base_mouth[:3, 3]
    delta = final_tcp_xyz - mouth_xyz
    print(f"  Final TCP pos:    ({final_tcp_xyz[0]:+.3f}, {final_tcp_xyz[1]:+.3f}, {final_tcp_xyz[2]:+.3f})")
    print(f"  Port mouth pos:   ({mouth_xyz[0]:+.3f}, {mouth_xyz[1]:+.3f}, {mouth_xyz[2]:+.3f})")
    print(f"  Δ (TCP - mouth):  ({delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f}) m")
    xy_off = np.linalg.norm(delta[:2])
    print(f"    XY offset {xy_off*1000:.1f} mm, Z offset {delta[2]*1000:+.1f} mm")
    print(f"    → {'✓ TCP at port (success)' if xy_off < 0.02 else '✗ TCP NOT at port'} "
          f"({'inserted past mouth' if delta[2] < 0 else 'above mouth'})")


# ---------- Check 1 + 6: project GT corners and overlay on image ----------

def project_and_overlay(ds, ep_indices, K_per_cam, T_tcp_opt_per_cam, model):
    """For each episode, save 4 frames (start/mid/late/last) with overlays:
       - YOLO predicted keypoints (cyan)
       - GT mouth corners projected through K (green)
       - GT mouth center (yellow)
       - bbox in red
       - measure pixel-distance between cyan and green corners
    """
    print(f"\n=== Check 1+6: GT projection vs YOLO predictions ===")
    print("Soundness: GT projected corners should overlap YOLO predicted corners")
    print(f"Saving overlays to {OUT_DIR}/")
    pnp_cfg = PnPConfig()

    summary = []
    for ep in ep_indices:
        start = int(ds.meta.episodes[ep]["dataset_from_index"])
        L = int(ds.meta.episodes[ep]["length"])
        for label, fr in [("init", 0), ("mid", L // 2),
                            ("late", int(L * 0.85)), ("last", L - 1)]:
            sample = ds[start + fr]
            state = sample["observation.state"].numpy()
            port_gt = sample["observation.port_pose_gt"].numpy()
            T_base_tcp = state_to_T(state)
            T_base_mouth = gt_link_to_mouth(port_gt_to_T(port_gt))

            for cam in ("center",):  # only center cam for this overlay
                img_chw = sample[f"observation.images.{cam}"]
                img_rgb = (img_chw.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                # GT projection
                T_world_optical = T_base_tcp @ T_tcp_opt_per_cam[cam]
                T_optical_world = np.linalg.inv(T_world_optical)
                mouth_pts_world = (T_base_mouth @ np.hstack(
                    [MOUTH_CORNERS_5, np.ones((5, 1))]).T).T[:, :3]
                uv_gt = project_world_to_image(K_per_cam[cam], T_optical_world,
                                                 mouth_pts_world)

                # YOLO prediction
                results = model.predict(img_bgr, imgsz=1280, conf=0.25, verbose=False)
                r = results[0]
                cyan_pts = None
                bbox = None
                if r.keypoints is not None and len(r.keypoints) > 0:
                    cls_all = r.boxes.cls.cpu().numpy().astype(int)
                    target = np.where(cls_all == 0)[0]
                    if len(target) > 0:
                        j = target[r.boxes.conf.cpu().numpy()[target].argmax()]
                        cyan_pts = r.keypoints.xy.cpu().numpy()[j]
                        bbox = r.boxes.xyxy.cpu().numpy()[j]

                # Per-corner pixel distance
                if cyan_pts is not None:
                    deltas = np.linalg.norm(uv_gt - cyan_pts, axis=1)
                    summary.append({
                        "ep": ep, "fr": fr, "phase": label, "cam": cam,
                        "kp_pixel_err": deltas.tolist(),
                        "median_kp_err_px": float(np.median(deltas)),
                    })

                # Render overlay
                out = img_bgr.copy()
                for x, y in uv_gt[:4]:
                    cv2.circle(out, (int(x), int(y)), 6, (0, 200, 0), 2)  # green
                cx_, cy_ = uv_gt[4]
                cv2.circle(out, (int(cx_), int(cy_)), 4, (0, 220, 220), -1)  # yellow
                if cyan_pts is not None:
                    for x, y in cyan_pts[:4]:
                        cv2.circle(out, (int(x), int(y)), 4, (255, 200, 0), -1)  # cyan
                if bbox is not None:
                    cv2.rectangle(out, (int(bbox[0]), int(bbox[1])),
                                    (int(bbox[2]), int(bbox[3])), (0, 0, 220), 2)

                # Crop a 250×250 window around GT center, scale 4x for inspection
                w = 200
                x1, y1 = int(cx_ - w), int(cy_ - w)
                x2, y2 = int(cx_ + w), int(cy_ + w)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)
                crop = out[y1:y2, x1:x2]
                crop = cv2.resize(crop, None, fx=2.5, fy=2.5,
                                    interpolation=cv2.INTER_NEAREST)
                cv2.putText(crop, f"ep{ep} {label} fr={fr} cam={cam}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                fname = OUT_DIR / f"ep{ep:03d}_{label}_{cam}_crop.png"
                cv2.imwrite(str(fname), crop)
                # Also save the full frame
                fname_full = OUT_DIR / f"ep{ep:03d}_{label}_{cam}_full.png"
                cv2.imwrite(str(fname_full), out)

    # Aggregate kp pixel error
    if summary:
        all_deltas = np.array([d for s in summary for d in s["kp_pixel_err"]])
        print(f"\n  Per-keypoint GT-vs-YOLO pixel distance ({len(all_deltas)} keypoints):")
        print(f"    median: {np.median(all_deltas):.2f} px")
        print(f"    p90:    {np.percentile(all_deltas, 90):.2f} px")
        print(f"    max:    {all_deltas.max():.2f} px")
        if np.median(all_deltas) < 5:
            print("    → ✓ GT projections align with YOLO keypoints (geometry sound)")
        elif np.median(all_deltas) < 20:
            print("    → ~ alignment is OK but worse than expected")
        else:
            print("    → ✗ MISALIGNMENT: K, TF, mouth offset, or rot6 decoding has an error")
    return summary


# ---------- Check 5: synthetic PnP recovery ----------

def check_synthetic_pnp(K_per_cam):
    print("\n=== Check 5: PnP IPPE round-trip on synthetic data ===")
    K = K_per_cam["center"]
    pnp_cfg = PnPConfig()

    # Place a port at z=0.25m in front of camera, slight rotation
    R_true = cv2.Rodrigues(np.array([0.05, -0.1, 0.2]))[0]
    t_true = np.array([0.01, -0.005, 0.25])
    T_true = np.eye(4); T_true[:3, :3] = R_true; T_true[:3, 3] = t_true

    pts_world = MOUTH_CORNERS_5
    rvec_true, _ = cv2.Rodrigues(R_true)
    proj, _ = cv2.projectPoints(pts_world, rvec_true, t_true, K, np.zeros(5))
    uv = proj.reshape(-1, 2)

    bbox = np.array([uv[:, 0].min(), uv[:, 1].min(),
                       uv[:, 0].max(), uv[:, 1].max()])
    pose = estimate_pose(
        keypoints_uv=uv, cls_id=0, bbox_xyxy=bbox, confidence=0.95,
        K=K, dist_coeffs=np.zeros(5), cfg=pnp_cfg,
    )
    print(f"  quality_flag: {pose.quality_flag}")
    if pose.quality_flag == "ok":
        t_err_mm = np.linalg.norm(pose.tvec_cam - t_true) * 1000
        R_rel = pose.T_cam_mouth[:3, :3].T @ R_true
        cos_th = (np.trace(R_rel) - 1) / 2
        r_err_deg = float(np.degrees(np.arccos(np.clip(cos_th, -1, 1))))
        print(f"  Recovered translation err: {t_err_mm:.4f} mm")
        print(f"  Recovered rotation err:    {r_err_deg:.4f}°")
        if t_err_mm < 0.1 and r_err_deg < 0.1:
            print("    → ✓ PnP module recovers ground truth (sub-mm, sub-degree)")
        else:
            print("    → ✗ PnP recovery has unexpected error")


def main():
    print("Loading dataset, model, calibration...")
    ds = LeRobotDataset(repo_id="HexDexAIC/aic-sfp-500-pr",
                         root=str(DATASET_ROOT), revision="main")
    cam_offs = json.loads(CAM_CALIB.read_text())
    K_per_cam = {c: np.array(v["K"]).reshape(3, 3) for c, v in cam_offs.items()}
    T_tcp_opt_per_cam = {c: np.array(v["T_tcp_optical"]) for c, v in cam_offs.items()}
    model = YOLO(str(WEIGHTS))

    # Pick a known episode
    test_eps = [42, 100]
    check_axes_and_geometry(ds, test_eps[0], T_tcp_opt_per_cam)
    check_synthetic_pnp(K_per_cam)
    project_and_overlay(ds, test_eps, K_per_cam, T_tcp_opt_per_cam, model)


if __name__ == "__main__":
    main()
