#!/usr/bin/env python3
"""Preview the v1.2 auto annotator on diverse sample frames before bulk export.

For each sampled frame, projects + draws:
  GREEN  — target SFP port mouth (sfp_port_0_link_entrance)
           5 landmarks + bbox
  RED    — distractor SFP port mouth (sfp_port_1_link_entrance)
           5 landmarks + bbox
Saves individual high-zoom crops + a stitched grid.

Selection criteria (matches v1 plan):
  - Frames where ||TCP-port|| ≥ 6cm (insertion-phase filter)
  - Diverse rails / distances / split (train/val/test)
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
CALIB_PATH = Path.home() / "aic_visible_mouth_calib.json"

SLOT_W, SLOT_H = 0.0137, 0.0085


def project_landmarks(T_base_target_mouth, T_base_tcp, K, T_tcp_opt, w, h):
    """Project 5 landmarks (4 corners + center) at the target mouth plane.
    Corner order: (+X+Y, +X-Y, -X-Y, -X+Y, center) — port-frame canonical.
    """
    pts_local = np.array([
        [+w/2, +h/2, 0, 1],
        [+w/2, -h/2, 0, 1],
        [-w/2, -h/2, 0, 1],
        [-w/2, +h/2, 0, 1],
        [0, 0, 0, 1],
    ]).T
    T_opt = np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T_base_target_mouth
    pts3 = T_opt @ pts_local
    Z = pts3[2]
    if (Z <= 0).any():
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx_p, cy_p = K[0, 2], K[1, 2]
    return np.stack([fx * pts3[0] / Z + cx_p, fy * pts3[1] / Z + cy_p], axis=1)


def annotate_one(img, T_base_port_canon, T_base_tcp, calib, K, T_tcp_opt):
    """Run v1.2 auto-annotator logic and return (target_pts, distractor_pts, target_bbox, distractor_bbox)."""
    co = calib["T_canonical_to_visible_mouth"]
    td = calib["T_target_to_distractor"]
    rect = calib["rectangle_at_mouth"]
    w = rect["width_mm"] / 1000
    h = rect["height_mm"] / 1000

    # Target mouth = canonical @ T_co_mouth
    T_mouth = np.eye(4); T_mouth[:3, 3] = [co["dx_mm"]/1000, co["dy_mm"]/1000, co["dz_mm"]/1000]
    T_base_target_mouth = T_base_port_canon @ T_mouth

    # Distractor mouth = canonical @ T_target_to_distractor @ T_co_mouth
    T_dist = np.eye(4); T_dist[:3, 3] = [td["dx_mm"]/1000, td["dy_mm"]/1000, td["dz_mm"]/1000]
    T_base_distractor_mouth = T_base_port_canon @ T_dist @ T_mouth

    target_pts = project_landmarks(T_base_target_mouth, T_base_tcp, K, T_tcp_opt, w, h)
    distractor_pts = project_landmarks(T_base_distractor_mouth, T_base_tcp, K, T_tcp_opt, w, h)
    return target_pts, distractor_pts


def visibility_flag(uv, w_img, h_img):
    """v1 visibility: 2 if in-frame, 1 if off-screen (still projectable)."""
    u, v = uv
    return 2 if (0 <= u < w_img and 0 <= v < h_img) else 1


def main():
    calib = json.loads(CALIB_PATH.read_text())
    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_plug = np.eye(4)
    T_TCP_plug[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_plug[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    # Sample 12 frames spanning train/val/test × multiple distances × different rails
    SAMPLES = [
        # train
        (0,  0.18, "train"), (0,  0.10, "train"),
        (35, 0.16, "train"), (60, 0.12, "train"),
        (90, 0.18, "train"), (140, 0.14, "train"),
        # val
        (172, 0.16, "val"),  (190, 0.10, "val"),
        # test
        (210, 0.18, "test"), (240, 0.12, "test"),
        (260, 0.14, "test"), (285, 0.10, "test"),
    ]

    needed = set(s[0] for s in SAMPLES)
    ep_data = {}
    for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep_val in df["episode_index"].unique():
            ep_int = int(ep_val)
            if ep_int in needed and ep_int not in ep_data:
                file_idx = int(pf.stem.replace("file-", ""))
                eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
                ep_data[ep_int] = (file_idx, eg)

    K = K_PER_CAM["center"]; T_tcp_opt = T_TCP_OPT["center"]
    crops = []
    for ep, target_d, split in SAMPLES:
        if ep not in ep_data or str(ep) not in gt_pose:
            print(f"skip ep{ep}")
            continue
        file_idx, eg = ep_data[ep]
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

        # Load video frame
        tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                             columns=["episode_index", "frame_index"])
        df_full = tbl.to_pandas()
        row = df_full[(df_full["episode_index"] == ep) & (df_full["frame_index"] == fr_idx)].index[0]
        cap = cv2.VideoCapture(str(ROOT / "videos" / "observation.images.center" / "chunk-000" / f"file-{file_idx:03d}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row))
        ok, img = cap.read(); cap.release()
        if not ok:
            continue

        T_base_tcp = state_to_T(states[idx])
        target_pts, distractor_pts = annotate_one(img, T_base_port, T_base_tcp, calib, K, T_tcp_opt)
        if target_pts is None:
            continue

        h_img, w_img = img.shape[:2]

        # Compute bboxes & visibility
        def bbox_from_corners(corners):
            x = corners[:4, 0]; y = corners[:4, 1]
            return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]

        target_bbox = bbox_from_corners(target_pts)
        distractor_bbox = bbox_from_corners(distractor_pts) if distractor_pts is not None else None

        # Crop centered on the union of both ports
        all_pts = target_pts.copy() if distractor_pts is None else np.vstack([target_pts, distractor_pts])
        cx, cy = all_pts.mean(axis=0)
        spread = max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1]))
        half = max(180, int(spread * 1.6))
        x0 = max(0, int(cx - half)); x1 = min(w_img, int(cx + half))
        y0 = max(0, int(cy - half)); y1 = min(h_img, int(cy + half))
        crop = img[y0:y1, x0:x1].copy()
        ZOOM = 3
        big = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

        def draw(corners, color, label):
            if corners is None: return
            local = (corners - np.array([x0, y0])) * ZOOM
            poly = local[:4].astype(np.int32)
            cv2.polylines(big, [poly], True, color, 3)
            for i in range(4):
                p = tuple(local[i].astype(int))
                cv2.circle(big, p, 6, color, -1)
                cv2.putText(big, str(i), (p[0]+8, p[1]-6),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.circle(big, tuple(local[4].astype(int)), 6, (255, 255, 255), -1)
            # bbox
            bx = local[:4]
            bb_min = bx.min(axis=0).astype(int); bb_max = bx.max(axis=0).astype(int)
            cv2.rectangle(big, tuple(bb_min), tuple(bb_max), color, 1)
            cv2.putText(big, label, (bb_min[0], bb_min[1] - 6),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        draw(target_pts, (0, 255, 0), "sfp_target")
        draw(distractor_pts, (0, 0, 255), "sfp_distractor")

        cv2.putText(big, f"ep{ep:03d} fr{fr_idx:04d} d={actual_d*100:.0f}cm  [{split.upper()}]",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Letterbox to 720x720
        CANVAS = 720
        h_b, w_b = big.shape[:2]
        s = CANVAS / max(h_b, w_b)
        nh, nw = int(h_b * s), int(w_b * s)
        res = cv2.resize(big, (nw, nh))
        canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
        yo = (CANVAS - nh) // 2; xo = (CANVAS - nw) // 2
        canvas[yo:yo+nh, xo:xo+nw] = res

        out_path = Path(f"/mnt/c/Users/Dell/aic_v12_preview_ep{ep:03d}_d{int(actual_d*100):02d}_{split}.jpg")
        cv2.imwrite(str(out_path), canvas)
        crops.append(canvas)
        print(f"  ep{ep:03d} fr{fr_idx:04d} d={actual_d*100:.0f}cm [{split}] saved")

    if crops:
        # Stitched grid (3 cols)
        cols = 3
        nrows = (len(crops) + cols - 1) // cols
        H, W = crops[0].shape[:2]
        grid = np.zeros((nrows * H, cols * W, 3), dtype=np.uint8)
        for i, c in enumerate(crops):
            r = i // cols; col = i % cols
            grid[r*H:(r+1)*H, col*W:(col+1)*W] = c
        cv2.imwrite("/mnt/c/Users/Dell/aic_v12_preview_grid.jpg", grid)
        print(f"\nGrid: /mnt/c/Users/Dell/aic_v12_preview_grid.jpg")


if __name__ == "__main__":
    main()
