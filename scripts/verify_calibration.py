#!/usr/bin/env python3
"""Render the calibrated visible-mouth transform on HELD-OUT frames the user
did NOT click during calibration. Confirms the transform generalizes.

Output: side-by-side panels per frame:
  CANONICAL (yellow) | CALIBRATED (green) | + user bbox if hand-annotated (orange)
"""
import json
import sys
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
ANN_PATH = Path.home() / "aic_audit_annotations.json"
CALIB_PATH = Path.home() / "aic_visible_mouth_calib.json"

SLOT_W, SLOT_H = 0.0137, 0.0085


def project(T_base_port, T_co, T_base_tcp, K, T_tcp_opt, w, h):
    T = T_base_port @ T_co
    corners = np.array([
        [+w/2, +h/2, 0, 1], [+w/2, -h/2, 0, 1],
        [-w/2, -h/2, 0, 1], [-w/2, +h/2, 0, 1],
    ]).T
    pts = (np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T) @ corners
    Z = pts[2]
    if (Z <= 0).any():
        return None
    return np.stack([K[0, 0] * pts[0] / Z + K[0, 2],
                     K[1, 1] * pts[1] / Z + K[1, 2]], axis=1)


def main():
    calib = json.loads(CALIB_PATH.read_text())
    t = calib["T_canonical_to_visible_mouth"]
    T_co = np.eye(4)
    T_co[:3, 3] = [t["dx_mm"]/1000, t["dy_mm"]/1000, t["dz_mm"]/1000]
    cal_w = calib["rectangle"]["width_mm"] / 1000
    cal_h = calib["rectangle"]["height_mm"] / 1000
    print(f"Calibrated transform: dx={t['dx_mm']:+.2f} dy={t['dy_mm']:+.2f} dz={t['dz_mm']:+.2f} mm")
    print(f"Rectangle: {cal_w*1000:.2f} × {cal_h*1000:.2f} mm")

    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    # Held-out: episodes NOT in calibration set
    CALIB_EPS_USED = {3, 17, 41, 60, 78, 96, 122, 138, 175, 197, 235, 272}
    HELDOUT = [(0, 0.18), (5, 0.18), (30, 0.18), (90, 0.18),    # train splits (heldout from calib)
               (180, 0.18), (250, 0.18), (220, 0.18),            # one val & two test
               (5, 0.10), (30, 0.10)]                            # close distance held-out

    # Load user hand annotations (for orange overlay where available)
    ann = json.loads(ANN_PATH.read_text())
    ann_by_efc = {}
    for k, v in ann.items():
        ep = int(v["episode"]); fr = int(v["frame"]); cam = v["camera"]
        for b in v.get("boxes", []):
            if b.get("label") == "target":
                ann_by_efc[(ep, fr, cam)] = b["bbox_xyxy"]
                break

    needed = set(c[0] for c in HELDOUT)
    ep_to_data = {}
    for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep_val in df["episode_index"].unique():
            ep_int = int(ep_val)
            if ep_int in needed and ep_int not in ep_to_data:
                file_idx = int(pf.stem.replace("file-", ""))
                eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
                ep_to_data[ep_int] = (file_idx, eg)

    rows = []
    for ep, target_d in HELDOUT:
        if ep not in ep_to_data or str(ep) not in gt_pose:
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
        err = np.where(valid, np.abs(dists - target_d), np.inf)
        idx = int(np.argmin(err))
        actual_d = float(dists[idx])
        fr_idx = int(frames[idx])

        tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                             columns=["episode_index", "frame_index"])
        df_full = tbl.to_pandas()
        row_in_file = df_full[(df_full["episode_index"] == ep) &
                                (df_full["frame_index"] == fr_idx)].index[0]
        cap = cv2.VideoCapture(str(ROOT / "videos" / "observation.images.center" / "chunk-000" / f"file-{file_idx:03d}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row_in_file))
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue

        T_base_tcp = state_to_T(states[idx])
        K = K_PER_CAM["center"]; T_tcp_opt = T_TCP_OPT["center"]

        # Canonical (no offset) projection at spec size
        canon = project(T_base_port, np.eye(4), T_base_tcp, K, T_tcp_opt, SLOT_W, SLOT_H)
        # Calibrated projection
        calibrated = project(T_base_port, T_co, T_base_tcp, K, T_tcp_opt, cal_w, cal_h)
        if canon is None or calibrated is None:
            continue

        # Crop around midpoint
        mid = canon[:4].mean(axis=0)
        port_pix_w = float(np.linalg.norm(canon[1] - canon[0]))
        half = max(120, int(port_pix_w * 3.0))
        x0 = max(0, int(mid[0] - half)); x1 = min(img.shape[1], int(mid[0] + half))
        y0 = max(0, int(mid[1] - half)); y1 = min(img.shape[0], int(mid[1] + half))
        crop = img[y0:y1, x0:x1].copy()

        ZOOM = 4
        big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

        # Draw canonical (yellow)
        canon_local = (canon - np.array([x0, y0])) * ZOOM
        cv2.polylines(big, [canon_local.astype(np.int32)], True, (0, 255, 255), 2)
        cv2.putText(big, "CANONICAL", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw calibrated (green)
        cal_local = (calibrated - np.array([x0, y0])) * ZOOM
        cv2.polylines(big, [cal_local.astype(np.int32)], True, (0, 255, 0), 3)
        cv2.putText(big, "CALIBRATED", (10, 60),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # If hand-annotated, draw user bbox (orange)
        ub = ann_by_efc.get((ep, fr_idx, "center"))
        if ub:
            ub_l = [(ub[0] - x0) * ZOOM, (ub[1] - y0) * ZOOM,
                     (ub[2] - x0) * ZOOM, (ub[3] - y0) * ZOOM]
            cv2.rectangle(big, (int(ub_l[0]), int(ub_l[1])),
                            (int(ub_l[2]), int(ub_l[3])), (0, 165, 255), 2)
            cv2.putText(big, "USER (hand)", (10, 90),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        cv2.putText(big, f"ep{ep:03d} fr{fr_idx:04d} d={actual_d*100:.1f}cm (HELD-OUT)",
                     (10, big.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Letterbox to 600
        CANVAS = 600
        h_b, w_b = big.shape[:2]
        s = CANVAS / max(h_b, w_b)
        nh, nw = int(h_b * s), int(w_b * s)
        res = cv2.resize(big, (nw, nh))
        canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
        yo = (CANVAS - nh) // 2; xo = (CANVAS - nw) // 2
        canvas[yo:yo+nh, xo:xo+nw] = res

        out_path = Path(f"/mnt/c/Users/Dell/aic_verify_ep{ep:03d}_d{int(actual_d*100):02d}.jpg")
        cv2.imwrite(str(out_path), canvas)
        rows.append(canvas)
        print(f"  ep{ep:03d} fr{fr_idx:04d} d={actual_d*100:.0f}cm: saved {out_path.name}")

    if rows:
        # Stitch grid (3 cols)
        cols = 3
        nrows = (len(rows) + cols - 1) // cols
        H = rows[0].shape[0]; W = rows[0].shape[1]
        grid = np.zeros((nrows * H, cols * W, 3), dtype=np.uint8)
        for i, r in enumerate(rows):
            row_idx = i // cols; col_idx = i % cols
            grid[row_idx*H:(row_idx+1)*H, col_idx*W:(col_idx+1)*W] = r
        cv2.imwrite("/mnt/c/Users/Dell/aic_verify_grid.jpg", grid)
        print(f"\nStitched grid: /mnt/c/Users/Dell/aic_verify_grid.jpg")


if __name__ == "__main__":
    main()
