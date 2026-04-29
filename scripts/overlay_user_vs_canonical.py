#!/usr/bin/env python3
"""Render user hand-clicks (orange) and canonical projection (yellow) on same
frames so the user can visually inspect the offset.

For each of the 22 accepted hand-label pairs, render a zoomed crop with:
  - YELLOW: canonical 13.7×8.5 spec projection
  - ORANGE: user's hand-drawn target bbox
  - Δ between centers shown as an arrow + label

Output one row per (ep, fr, cam) sorted by camera. Saved to
/mnt/c/Users/Dell/aic_user_vs_canonical_*.jpg
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
SLOT_W, SLOT_H = 0.0137, 0.0085


def project(T_base_port, T_base_tcp, K, T_tcp_opt, w, h, dz=0):
    T_co = np.eye(4); T_co[2, 3] = dz
    T = T_base_port @ T_co
    corners = np.array([
        [+w/2, +h/2, 0, 1], [+w/2, -h/2, 0, 1],
        [-w/2, -h/2, 0, 1], [-w/2, +h/2, 0, 1],
        [0, 0, 0, 1],
    ]).T
    T_opt = np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T
    pts = T_opt @ corners
    Z = pts[2]
    if (Z <= 0).any():
        return None
    fx, fy = K[0, 0], K[1, 1]; cx_p, cy_p = K[0, 2], K[1, 2]
    return np.stack([fx * pts[0] / Z + cx_p, fy * pts[1] / Z + cy_p], axis=1)


def main():
    ann = json.loads(ANN_PATH.read_text())
    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    needed_eps = set()
    for k, v in ann.items():
        if any(b.get("label") == "target" for b in v.get("boxes", [])):
            needed_eps.add(int(v["episode"]))

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

    rows_by_cam = {"left": [], "center": [], "right": []}
    for k, v in ann.items():
        ep = int(v["episode"]); fr = int(v["frame"]); cam = v["camera"]
        targets = [b for b in v.get("boxes", []) if b.get("label") == "target"]
        if not targets or str(ep) not in gt_pose or ep not in ep_to_data:
            continue
        bbox = targets[0]["bbox_xyxy"]
        ucx = (bbox[0] + bbox[2]) / 2; ucy = (bbox[1] + bbox[3]) / 2

        file_idx, eg = ep_to_data[ep]
        states = np.stack(eg["observation.state"].values)
        frames = eg["frame_index"].to_numpy()
        m = (frames == fr)
        if not m.any():
            continue
        idx = int(np.where(m)[0][0])
        T_settled = np.eye(4)
        T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
        T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
        T_base_port = T_settled @ T_TCP_plug
        T_base_tcp = state_to_T(states[idx])
        K = K_PER_CAM[cam]; T_tcp_opt = T_TCP_OPT[cam]
        proj = project(T_base_port, T_base_tcp, K, T_tcp_opt, SLOT_W, SLOT_H, dz=0)
        if proj is None:
            continue
        ccx, ccy = proj[:4].mean(axis=0)
        if np.hypot(ucx - ccx, ucy - ccy) > 80:  # filter wrong-port
            continue

        # Load video frame
        tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                             columns=["episode_index", "frame_index"])
        df_full = tbl.to_pandas()
        row_in_file = df_full[(df_full["episode_index"] == ep) &
                                (df_full["frame_index"] == fr)].index[0]
        cap = cv2.VideoCapture(str(ROOT / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{file_idx:03d}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row_in_file))
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue

        # Crop around midpoint of user/canon centers
        mid = ((ucx + ccx) / 2, (ucy + ccy) / 2)
        diag = np.hypot(ucx - ccx, ucy - ccy)
        canon_w_pix = float(np.linalg.norm(proj[1] - proj[0]))
        half = max(120, int(diag * 1.5), int(canon_w_pix * 2.5))
        x0 = max(0, int(mid[0] - half)); x1 = min(img.shape[1], int(mid[0] + half))
        y0 = max(0, int(mid[1] - half)); y1 = min(img.shape[0], int(mid[1] + half))
        crop = img[y0:y1, x0:x1].copy()
        ZOOM = 4
        crop_big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

        # Draw user bbox (orange)
        ub = [(bbox[0] - x0) * ZOOM, (bbox[1] - y0) * ZOOM,
               (bbox[2] - x0) * ZOOM, (bbox[3] - y0) * ZOOM]
        cv2.rectangle(crop_big, (int(ub[0]), int(ub[1])), (int(ub[2]), int(ub[3])), (0, 165, 255), 3)
        cv2.putText(crop_big, "USER", (int(ub[0]), int(ub[1]) - 6),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Draw canonical (yellow polygon)
        canon_local = (proj[:4] - np.array([x0, y0])) * ZOOM
        cv2.polylines(crop_big, [canon_local.astype(np.int32)], True, (0, 255, 255), 3)
        ctr = ((ccx - x0) * ZOOM, (ccy - y0) * ZOOM)
        cv2.circle(crop_big, (int(ctr[0]), int(ctr[1])), 6, (0, 255, 255), -1)
        cv2.putText(crop_big, "CANONICAL", (int(canon_local[0, 0]), int(canon_local[0, 1]) - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Draw arrow from canonical to user center
        u_ctr = ((ucx - x0) * ZOOM, (ucy - y0) * ZOOM)
        cv2.arrowedLine(crop_big, (int(ctr[0]), int(ctr[1])), (int(u_ctr[0]), int(u_ctr[1])),
                         (255, 255, 255), 2, tipLength=0.15)
        dx = ucx - ccx; dy = ucy - ccy; d = np.hypot(dx, dy)
        cv2.putText(crop_big, f"d={d:.0f}px (dx={dx:+.0f}, dy={dy:+.0f})",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(crop_big, f"ep{ep:03d} fr{fr:04d} {cam}",
                     (10, crop_big.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Letterbox to 480
        CANVAS = 540
        h_b, w_b = crop_big.shape[:2]
        s = CANVAS / max(h_b, w_b)
        nh, nw = int(h_b * s), int(w_b * s)
        res = cv2.resize(crop_big, (nw, nh))
        canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
        yo = (CANVAS - nh) // 2; xo = (CANVAS - nw) // 2
        canvas[yo:yo+nh, xo:xo+nw] = res
        rows_by_cam[cam].append((ep, fr, canvas, d))

    # Build one stitched image per camera
    for cam, rows in rows_by_cam.items():
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])  # sort by episode
        cells = [r[2] for r in rows]
        # Lay out in 3 columns
        N_COLS = 3
        N_ROWS = (len(cells) + N_COLS - 1) // N_COLS
        H = cells[0].shape[0]; W = cells[0].shape[1]
        grid = np.zeros((N_ROWS * H, N_COLS * W, 3), dtype=np.uint8)
        for i, c in enumerate(cells):
            r = i // N_COLS; col = i % N_COLS
            grid[r*H:(r+1)*H, col*W:(col+1)*W] = c
        out = Path(f"/mnt/c/Users/Dell/aic_user_vs_canonical_{cam}.jpg")
        cv2.imwrite(str(out), grid)
        print(f"Wrote {len(rows)} {cam}-cam comparisons -> {out}")


if __name__ == "__main__":
    main()
