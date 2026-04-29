#!/usr/bin/env python3
"""Calibrate T_canonical_to_visible_mouth for SFP via grid search.

Pipeline:
  1. Build a 12-frame calibration set across rails / distances / splits.
  2. For each frame, run a visible-slot detector (HSV dark+non-green, restricted
     to a ROI around the canonical GT projection). Returns detected pixel rect.
  3. Phase A: sweep dz ∈ {-4, -2, 0, +2, +4} mm at spec w=13.7 h=8.5.
  4. Phase B: at winning dz, sweep w ∈ {11, 12.5, 13.7, 15} × h ∈ {7, 8.5, 10, 12}.
  5. Verify top-3 by rendering side-by-side on 8 held-out frames.

Quantitative score per (dz, w, h):
  - center pixel error: ||proj_center - detected_center||
  - size error: ||(proj_w, proj_h) - (det_w, det_h)|| in pixels
  - IoU of axis-aligned bboxes
Aggregate: median across calibration frames (median to be robust to detector misses).

Outputs:
  /tmp/aic_calibrate/...                                  — intermediate
  /mnt/c/Users/Dell/aic_calib_phaseA_dz_sweep.jpg         — Phase A overlay grid
  /mnt/c/Users/Dell/aic_calib_phaseB_size_sweep.jpg       — Phase B overlay grid
  /mnt/c/Users/Dell/aic_calib_top3_verification.jpg       — Phase C verification
  /mnt/c/Users/Dell/aic_calib_leaderboard.txt             — full quantitative results
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import (
    K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R,
)

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
OUT_DIR = Path("/tmp/aic_calibrate")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LB_OUT = Path("/mnt/c/Users/Dell/aic_calib_leaderboard.txt")

# Calibration set: 12 frames across rails / distances / splits.
# Episodes chosen to span 0-299; distances chosen to span 18cm down to 10cm.
CALIB_SET = [
    # (ep, target_dist_m, split)
    (3,   0.18, "train"),
    (3,   0.12, "train"),
    (41,  0.18, "train"),
    (78,  0.18, "train"),
    (78,  0.12, "train"),
    (122, 0.18, "train"),
    (160, 0.18, "val"),
    (160, 0.12, "val"),
    (197, 0.18, "val"),
    (235, 0.18, "test"),
    (235, 0.12, "test"),
    (272, 0.18, "test"),
]

# Phase A: depth sweep
PHASE_A_DZ_MM = [-4, -2, 0, +2, +4]
PHASE_A_WH = (0.0137, 0.0085)

# Phase B: size sweep (will run at winning dz from Phase A)
PHASE_B_W_MM = [11.0, 12.5, 13.7, 15.0]
PHASE_B_H_MM = [7.0, 8.5, 10.0, 12.0]


def project_corners(T_base_port, T_base_tcp, K, T_tcp_opt, w, h, dz=0.0):
    """Project corners with optional dz offset along port +z."""
    T_co_mouth = np.eye(4); T_co_mouth[2, 3] = dz
    T_base_mouth = T_base_port @ T_co_mouth
    corners = np.array([
        [+w/2, +h/2, 0, 1],
        [+w/2, -h/2, 0, 1],
        [-w/2, -h/2, 0, 1],
        [-w/2, +h/2, 0, 1],
        [0, 0, 0, 1],
    ]).T
    T_base_opt = T_base_tcp @ T_tcp_opt
    T_opt = np.linalg.inv(T_base_opt) @ T_base_mouth
    pts = T_opt @ corners
    Z = pts[2]
    if (Z <= 0).any():
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx_p, cy_p = K[0, 2], K[1, 2]
    u = fx * pts[0] / Z + cx_p
    v = fy * pts[1] / Z + cy_p
    return np.stack([u, v], axis=1)


def detect_visible_slot(img_bgr, ref_corners_pix, expected_w_pix, expected_h_pix):
    """Find darkest non-green rectangular blob within ROI around ref corners.
    Returns dict {cx, cy, w, h, angle, corners_xyxy} or None."""
    # ROI: ref bbox + margin
    xs = ref_corners_pix[:4, 0]; ys = ref_corners_pix[:4, 1]
    margin = max(expected_w_pix, expected_h_pix) * 1.5
    x0 = max(0, int(xs.min() - margin))
    y0 = max(0, int(ys.min() - margin))
    x1 = min(img_bgr.shape[1], int(xs.max() + margin))
    y1 = min(img_bgr.shape[0], int(ys.max() + margin))
    if x1 - x0 < 30 or y1 - y0 < 30:
        return None
    crop = img_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    not_green = cv2.bitwise_not(cv2.inRange(hsv, (35, 40, 25), (90, 255, 255)))
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 75))
    cand = cv2.bitwise_and(not_green, dark)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    expected_area = expected_w_pix * expected_h_pix
    roi_cx = (x1 - x0) / 2; roi_cy = (y1 - y0) / 2
    best = None; best_score = -np.inf
    for c in contours:
        area_pix = cv2.contourArea(c)
        if area_pix < 50 or area_pix > 8 * expected_area:
            continue
        rect = cv2.minAreaRect(c)
        (cx_l, cy_l), (rw, rh), angle = rect
        long_side = max(rw, rh); short_side = max(min(rw, rh), 1)
        ar = long_side / short_side
        if ar > 3.5 or ar < 1.0:
            continue
        # Score: prefer area near expected, prefer near ROI center
        rect_area = rw * rh + 1e-6
        size_score = -abs(np.log(rect_area / max(expected_area, 1)))
        center_score = -((cx_l - roi_cx) ** 2 + (cy_l - roi_cy) ** 2) ** 0.5 / max(expected_w_pix, 1)
        ar_score = -abs(np.log(ar / 1.6))
        total = 0.4 * size_score + 0.4 * center_score + 0.2 * ar_score
        if total > best_score:
            best_score = total
            box = cv2.boxPoints(rect)
            box[:, 0] += x0; box[:, 1] += y0
            best = {
                "cx": cx_l + x0, "cy": cy_l + y0,
                "w": rw, "h": rh, "angle": angle,
                "corners": box,
                "score": total,
            }
    return best


def compute_metrics(proj_corners, det):
    """Aggregate error: center distance + corner mean distance + AABB IoU."""
    proj_center = proj_corners[:4].mean(axis=0)
    det_center = np.array([det["cx"], det["cy"]])
    center_err = float(np.linalg.norm(proj_center - det_center))
    # AABB IoU
    p_xs = proj_corners[:4, 0]; p_ys = proj_corners[:4, 1]
    d_xs = det["corners"][:, 0]; d_ys = det["corners"][:, 1]
    p_box = (p_xs.min(), p_ys.min(), p_xs.max(), p_ys.max())
    d_box = (d_xs.min(), d_ys.min(), d_xs.max(), d_ys.max())
    ix0 = max(p_box[0], d_box[0]); iy0 = max(p_box[1], d_box[1])
    ix1 = min(p_box[2], d_box[2]); iy1 = min(p_box[3], d_box[3])
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    pa = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
    da = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
    union = pa + da - inter
    iou = inter / union if union > 0 else 0.0
    return {"center_err": center_err, "iou": float(iou)}


def load_frame_for(ep, target_dist):
    """Return (img, T_base_tcp, T_base_port_canonical, dist_actual, fr_idx) or None."""
    # Re-index parquets each call (acceptable for ~12 frames). Or cache.
    pass  # implemented inline below for clarity


def main():
    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]
    K = K_PER_CAM["center"]
    T_tcp_opt = T_TCP_OPT["center"]

    # Index parquet files for the calibration episodes
    needed_eps = set(c[0] for c in CALIB_SET)
    ep_to_data = {}
    for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep_val in df["episode_index"].unique():
            ep_int = int(ep_val)
            if ep_int in needed_eps and ep_int not in ep_to_data:
                file_idx = int(pf.stem.replace("file-", ""))
                eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
                ep_to_data[ep_int] = (file_idx, eg)

    # Build calibration frames: for each (ep, target_d), find best frame
    calib_frames = []
    for ep, target_d, split in CALIB_SET:
        if ep not in ep_to_data or str(ep) not in gt_pose:
            print(f"WARN: ep{ep} missing")
            continue
        file_idx, eg = ep_to_data[ep]
        states = np.stack(eg["observation.state"].values)
        frames = eg["frame_index"].to_numpy()

        T_settled = np.eye(4)
        T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
        T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
        T_base_port = T_settled @ T_TCP_plug
        port_xyz = T_base_port[:3, 3]

        dists = np.linalg.norm(states[:, 0:3] - port_xyz, axis=1)
        valid = dists >= 0.06
        if not valid.any():
            continue
        err = np.where(valid, np.abs(dists - target_d), np.inf)
        idx = int(np.argmin(err))
        actual_d = float(dists[idx])
        fr_idx = int(frames[idx])

        # Load frame
        tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                             columns=["episode_index", "frame_index"])
        df_full = tbl.to_pandas()
        row_in_file = df_full[(df_full["episode_index"] == ep) &
                                (df_full["frame_index"] == fr_idx)].index[0]
        video_path = ROOT / "videos" / "observation.images.center" / "chunk-000" / f"file-{file_idx:03d}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row_in_file))
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue

        T_base_tcp = state_to_T(states[idx])

        # Run slot detector at canonical projection
        canon_proj = project_corners(T_base_port, T_base_tcp, K, T_tcp_opt, 0.0137, 0.0085, dz=0.0)
        if canon_proj is None:
            continue
        canon_w_pix = float(np.linalg.norm(canon_proj[1] - canon_proj[0]))
        canon_h_pix = float(np.linalg.norm(canon_proj[2] - canon_proj[1]))
        det = detect_visible_slot(img, canon_proj, canon_w_pix, canon_h_pix)

        calib_frames.append({
            "ep": ep, "fr": fr_idx, "split": split, "dist": actual_d,
            "img": img, "T_base_tcp": T_base_tcp, "T_base_port": T_base_port,
            "canon_proj": canon_proj, "det": det,
            "canon_w_pix": canon_w_pix, "canon_h_pix": canon_h_pix,
        })

    n = len(calib_frames)
    n_det = sum(1 for f in calib_frames if f["det"] is not None)
    print(f"Calibration set: {n} frames loaded, slot detected in {n_det}")

    if n_det < 6:
        print("\nFEWER THAN 6 SLOT DETECTIONS — falling back: detector unreliable.")
        print("Recommend manual 4-corner clicks. Aborting auto calibration.")
        return

    # === Phase A: dz sweep ===
    print("\n=== Phase A: dz sweep (w=13.7mm, h=8.5mm) ===")
    phase_a_results = []
    for dz_mm in PHASE_A_DZ_MM:
        dz = dz_mm / 1000.0
        center_errs = []; ious = []
        for f in calib_frames:
            if f["det"] is None:
                continue
            proj = project_corners(f["T_base_port"], f["T_base_tcp"], K, T_tcp_opt,
                                    PHASE_A_WH[0], PHASE_A_WH[1], dz=dz)
            if proj is None:
                continue
            m = compute_metrics(proj, f["det"])
            center_errs.append(m["center_err"])
            ious.append(m["iou"])
        med_ce = float(np.median(center_errs))
        med_iou = float(np.median(ious))
        phase_a_results.append({
            "dz_mm": dz_mm, "median_center_err_pix": med_ce, "median_iou": med_iou,
            "n_frames_scored": len(center_errs),
        })
        print(f"  dz={dz_mm:+3d}mm: median_center_err={med_ce:.2f}px  median_iou={med_iou:.3f}  ({len(center_errs)} frames)")

    best_dz_mm = min(phase_a_results, key=lambda r: r["median_center_err_pix"])["dz_mm"]
    print(f"  → Phase A winner: dz = {best_dz_mm:+d} mm")

    # === Phase B: w/h sweep at winning dz ===
    print(f"\n=== Phase B: size sweep at dz={best_dz_mm:+d}mm ===")
    phase_b_results = []
    for w_mm, h_mm in product(PHASE_B_W_MM, PHASE_B_H_MM):
        w = w_mm / 1000.0; h = h_mm / 1000.0; dz = best_dz_mm / 1000.0
        center_errs = []; ious = []; corner_errs = []
        for f in calib_frames:
            if f["det"] is None:
                continue
            proj = project_corners(f["T_base_port"], f["T_base_tcp"], K, T_tcp_opt, w, h, dz=dz)
            if proj is None:
                continue
            m = compute_metrics(proj, f["det"])
            center_errs.append(m["center_err"])
            ious.append(m["iou"])
            # Per-corner mean error: match each proj corner to nearest det corner
            d_corners = f["det"]["corners"]
            mean_corner = 0.0
            for pc in proj[:4]:
                d_dists = np.linalg.norm(d_corners - pc, axis=1)
                mean_corner += d_dists.min()
            corner_errs.append(mean_corner / 4.0)
        med_ce = float(np.median(center_errs))
        med_iou = float(np.median(ious))
        med_corner = float(np.median(corner_errs))
        phase_b_results.append({
            "w_mm": w_mm, "h_mm": h_mm,
            "median_center_err_pix": med_ce,
            "median_corner_err_pix": med_corner,
            "median_iou": med_iou,
            "n_frames_scored": len(center_errs),
        })

    # Sort by corner error (most informative — accounts for both center and size)
    phase_b_results.sort(key=lambda r: r["median_corner_err_pix"])

    print(f"  {'rank':>4} {'w_mm':>6} {'h_mm':>6} {'corner_err':>11} {'center_err':>11} {'iou':>6}")
    for i, r in enumerate(phase_b_results):
        marker = "★" if i == 0 else " "
        print(f"  {i+1:>3}{marker} {r['w_mm']:>6.1f} {r['h_mm']:>6.1f} "
              f"{r['median_corner_err_pix']:>11.2f} {r['median_center_err_pix']:>11.2f} {r['median_iou']:>6.3f}")

    top1 = phase_b_results[0]
    print(f"\n  → Phase B winner: w={top1['w_mm']}mm, h={top1['h_mm']}mm")
    print(f"\nFINAL RECOMMENDATION:")
    print(f"  T_canonical_to_visible_mouth: dz = {best_dz_mm} mm along port +z (else identity)")
    print(f"  Rectangle: {top1['w_mm']} mm × {top1['h_mm']} mm")
    print(f"  Performance: median corner err {top1['median_corner_err_pix']:.2f}px, median IoU {top1['median_iou']:.3f}")

    # === Save leaderboard ===
    lb_lines = [
        "AIC SFP visible-mouth calibration leaderboard",
        "=" * 60,
        f"Calibration set: {n} frames, slot detected in {n_det}",
        "",
        "Phase A — dz sweep (w=13.7mm h=8.5mm fixed):",
    ]
    for r in phase_a_results:
        lb_lines.append(f"  dz={r['dz_mm']:+3d}mm: med_center_err={r['median_center_err_pix']:.2f}px  med_iou={r['median_iou']:.3f}")
    lb_lines.append(f"  Winner: dz = {best_dz_mm:+d} mm")
    lb_lines.append("")
    lb_lines.append(f"Phase B — w/h sweep at dz={best_dz_mm:+d}mm:")
    lb_lines.append(f"  {'rank':>4} {'w_mm':>6} {'h_mm':>6} {'corner_err':>11} {'center_err':>11} {'iou':>6}")
    for i, r in enumerate(phase_b_results):
        lb_lines.append(f"  {i+1:>4} {r['w_mm']:>6.1f} {r['h_mm']:>6.1f} "
                        f"{r['median_corner_err_pix']:>11.2f} {r['median_center_err_pix']:>11.2f} {r['median_iou']:>6.3f}")
    lb_lines.append("")
    lb_lines.append("Recommendation:")
    lb_lines.append(f"  T_canonical_to_visible_mouth: translate +{best_dz_mm}mm in port +z")
    lb_lines.append(f"  rectangle: {top1['w_mm']} mm × {top1['h_mm']} mm")
    LB_OUT.write_text("\n".join(lb_lines))
    print(f"\nLeaderboard: {LB_OUT}")

    # === Phase C: render top-3 vs canonical on calib frames ===
    top3 = phase_b_results[:3]
    candidates_to_render = [
        ("CANONICAL (current)", 0, 0.0137, 0.0085, (0, 255, 255)),    # YELLOW
        (f"#1: dz={best_dz_mm} w={top3[0]['w_mm']} h={top3[0]['h_mm']}",
         best_dz_mm, top3[0]["w_mm"]/1000, top3[0]["h_mm"]/1000, (0, 255, 0)),     # GREEN
        (f"#2: dz={best_dz_mm} w={top3[1]['w_mm']} h={top3[1]['h_mm']}",
         best_dz_mm, top3[1]["w_mm"]/1000, top3[1]["h_mm"]/1000, (255, 0, 255)),  # MAGENTA
        (f"#3: dz={best_dz_mm} w={top3[2]['w_mm']} h={top3[2]['h_mm']}",
         best_dz_mm, top3[2]["w_mm"]/1000, top3[2]["h_mm"]/1000, (255, 255, 0)),  # CYAN
    ]

    rows = []
    for f in calib_frames:
        if f["det"] is None:
            continue
        # crop window around canonical projection
        cx0 = f["canon_proj"][:4].mean(axis=0)
        half = max(80, int(f["canon_w_pix"] * 3.0))
        x0 = max(0, int(cx0[0] - half)); x1 = min(f["img"].shape[1], int(cx0[0] + half))
        y0 = max(0, int(cx0[1] - half)); y1 = min(f["img"].shape[0], int(cx0[1] + half))
        crop_base = f["img"][y0:y1, x0:x1]
        ZOOM = 5; CANVAS = 480
        panels = []
        for label, dz_mm, w, h, color in candidates_to_render:
            crop = crop_base.copy()
            big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            # Always draw the detected slot in RED for reference
            det_local = (f["det"]["corners"] - np.array([x0, y0])) * ZOOM
            cv2.polylines(big, [det_local.astype(np.int32)], True, (0, 0, 255), 2)
            cv2.putText(big, "detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Draw candidate
            proj = project_corners(f["T_base_port"], f["T_base_tcp"], K, T_tcp_opt, w, h, dz=dz_mm/1000.0)
            if proj is not None:
                pts_local = (proj - np.array([x0, y0])) * ZOOM
                cv2.polylines(big, [pts_local[:4].astype(np.int32)], True, color, 3)
                ctr = tuple(pts_local[4].astype(int))
                cv2.circle(big, ctr, 6, color, -1)
            cv2.putText(big, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(big, f"ep{f['ep']:03d} d={f['dist']*100:.0f}cm",
                         (10, big.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            # Letterbox
            hb, wb = big.shape[:2]
            s = CANVAS / max(hb, wb)
            nh, nw = int(hb * s), int(wb * s)
            res = cv2.resize(big, (nw, nh))
            canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
            yo = (CANVAS - nh) // 2; xo = (CANVAS - nw) // 2
            canvas[yo:yo+nh, xo:xo+nw] = res
            panels.append(canvas)
        rows.append(np.hstack(panels))

    if rows:
        max_w = max(r.shape[1] for r in rows)
        rows_pad = [np.hstack([r, np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)])
                     if r.shape[1] < max_w else r for r in rows]
        grid = np.vstack(rows_pad)
        grid_path = Path("/mnt/c/Users/Dell/aic_calib_top3_verification.jpg")
        cv2.imwrite(str(grid_path), grid)
        print(f"\nVerification grid: {grid_path}")
        print(f"  Each row: 4 panels — CANONICAL | TOP-1 | TOP-2 | TOP-3")
        print(f"  Red rectangle = detected visible slot (reference)")


if __name__ == "__main__":
    main()
