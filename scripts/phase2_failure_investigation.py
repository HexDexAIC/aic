#!/usr/bin/env python3
"""Phase-2 prereq: failure-pattern investigation on test split.

For every test frame, compute prediction vs GT and bucket by:
  - camera (left/center/right)
  - distance band (<10 / 10-15 / 15-20 / >20 cm) — using GT TCP-port distance
  - visibility (fully-on-screen vs partially-clipped vs heavily-clipped GT corners)
  - class (sfp_target / sfp_distractor)
  - role mismatch (predicted-target lands closer to GT-distractor than GT-target)

Output:
  ~/aic_runs/v1_h100_results/test_eval/failure_investigation.csv
  ~/aic_runs/v1_h100_results/test_eval/failure_investigation_summary.json
  /mnt/c/Users/Dell/aic_ep221_trajectory.jpg   — full approach for ep221
  /mnt/c/Users/Dell/aic_failure_summary.txt    — 1-page text summary
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

ROOT = Path.home() / "aic_hexdex_sfp300"
WEIGHTS = Path.home() / "aic_runs" / "v1_h100_results" / "best.pt"
TEST_LABELS = Path.home() / "aic_yolo_v1" / "labels" / "test"
TEST_IMAGES = Path.home() / "aic_yolo_v1" / "images" / "test"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
OUT_DIR = Path.home() / "aic_runs" / "v1_h100_results" / "test_eval"
OUT_DIR.mkdir(exist_ok=True)

CSV_OUT = OUT_DIR / "failure_investigation.csv"
SUMMARY_JSON = OUT_DIR / "failure_investigation_summary.json"
SUMMARY_TXT = Path("/mnt/c/Users/Dell/aic_failure_summary.txt")
TRAJECTORY_OUT = Path("/mnt/c/Users/Dell/aic_ep221_trajectory.jpg")

IMG_W, IMG_H = 1152, 1024


def parse_label(path, w_img, h_img):
    """Parse YOLO label → list of {cls, kpts (5x2), bbox(xyxy)}"""
    objs = []
    if not path.exists():
        return objs
    for line in path.read_text().strip().split("\n"):
        if not line.strip(): continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        bbox = [(cx - bw/2) * w_img, (cy - bh/2) * h_img,
                (cx + bw/2) * w_img, (cy + bh/2) * h_img]
        kpts = []
        for k in range(5):
            kx = float(parts[5 + k*3]) * w_img
            ky = float(parts[5 + k*3 + 1]) * h_img
            kpts.append((kx, ky))
        objs.append({"cls": cls, "bbox": bbox, "kpts": np.array(kpts, dtype=np.float32)})
    return objs


def visibility_label(kpts, w, h, margin=8):
    """Classify GT corner visibility."""
    on = sum(1 for x, y in kpts[:4] if margin <= x < w - margin and margin <= y < h - margin)
    if on == 4:
        return "fully_visible"
    elif on >= 2:
        return "partially_clipped"
    else:
        return "heavily_clipped"


def main():
    print(f"Loading {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    gt_pose = json.loads(GT_POSE_PATH.read_text())

    # Index test parquet for TCP state per frame (for distance computation)
    print("Indexing parquet files for TCP state lookup...")
    state_by_ef = {}  # (ep, fr) -> state[27]
    for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for _, row in df.iterrows():
            ep = int(row["episode_index"])
            if ep < 200:  # only test split
                continue
            fr = int(row["frame_index"])
            state_by_ef[(ep, fr)] = np.asarray(row["observation.state"])

    label_files = sorted(TEST_LABELS.glob("*.txt"))
    print(f"Test labels: {len(label_files)}")

    # Header for CSV
    csv_lines = ["ep,fr,cam,d_cm,visibility,n_gt,n_pred,n_matched,role_mismatch,"
                 "max_corner_err_target,max_corner_err_distr,"
                 "median_corner_err,bbox_area_target,bbox_area_distr,"
                 "max_pred_conf,gate_clipped,gate_small_bbox,gate_low_conf"]

    buckets = defaultdict(list)  # key -> list of records
    role_mismatches = []
    ep221_records = []  # for trajectory rendering

    for i, lbl_path in enumerate(label_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(label_files)}")

        m = re.match(r"ep(\d+)_fr(\d+)_(left|center|right)", lbl_path.stem)
        if not m: continue
        ep = int(m.group(1)); fr = int(m.group(2)); cam = m.group(3)

        img_path = TEST_IMAGES / (lbl_path.stem + ".jpg")
        if not img_path.exists(): continue
        gts = parse_label(lbl_path, IMG_W, IMG_H)
        if not gts: continue

        # GT TCP→port distance
        d_cm = -1.0
        if (ep, fr) in state_by_ef and str(ep) in gt_pose:
            tcp_xyz = state_by_ef[(ep, fr)][:3]
            port_settled = np.array(gt_pose[str(ep)]["actual_tcp_xyz"])
            # Note: port_settled is the TCP-at-deepest, ≈ port location
            d_cm = float(np.linalg.norm(tcp_xyz - port_settled) * 100)

        # Visibility from GT
        vis = visibility_label(gts[0]["kpts"], IMG_W, IMG_H)

        # Predict
        res = model.predict(str(img_path), imgsz=1280, conf=0.25, device=0, verbose=False)[0]
        pred_kpts = []
        pred_cls = []
        pred_conf = []
        pred_bbox_area = []
        if res.keypoints is not None and len(res.keypoints) > 0:
            pred_kpts = res.keypoints.xy.cpu().numpy()
            pred_cls = res.boxes.cls.cpu().numpy().astype(int)
            pred_conf = res.boxes.conf.cpu().numpy()
            xyxy = res.boxes.xyxy.cpu().numpy()
            pred_bbox_area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

        # Match per class
        per_cls_results = {0: {"matches": [], "max_err": np.nan, "bbox_area": 0.0},
                          1: {"matches": [], "max_err": np.nan, "bbox_area": 0.0}}
        all_corner_errs = []
        for cls in (0, 1):
            gts_c = [g for g in gts if g["cls"] == cls]
            preds_c_idx = [j for j in range(len(pred_kpts)) if pred_cls[j] == cls]
            if not gts_c or not preds_c_idx:
                continue
            gt_c = np.array([g["kpts"][4] for g in gts_c])
            pred_c = np.array([pred_kpts[j][4] for j in preds_c_idx])
            cost = np.linalg.norm(gt_c[:, None, :] - pred_c[None, :, :], axis=2)
            rr, cc = linear_sum_assignment(cost)
            for ri, ci in zip(rr, cc):
                if cost[ri, ci] > 100:
                    continue
                jp = preds_c_idx[ci]
                errs = np.linalg.norm(gts_c[ri]["kpts"] - pred_kpts[jp], axis=1)
                per_cls_results[cls]["matches"].append({
                    "gt_kpts": gts_c[ri]["kpts"], "pred_kpts": pred_kpts[jp],
                    "errs": errs, "conf": float(pred_conf[jp]),
                    "bbox_area": float(pred_bbox_area[jp]),
                })
                per_cls_results[cls]["max_err"] = max(
                    per_cls_results[cls].get("max_err", 0) if per_cls_results[cls]["max_err"] is not np.nan else 0,
                    float(errs.max()))
                per_cls_results[cls]["bbox_area"] = float(pred_bbox_area[jp])
                all_corner_errs.extend(errs.tolist())

        # Detect role mismatch: any pred labeled "target" actually closest to GT distractor center?
        role_mismatch = False
        if 0 in [g["cls"] for g in gts] and 1 in [g["cls"] for g in gts]:
            gt_target_c = next(g["kpts"][4] for g in gts if g["cls"] == 0)
            gt_distr_c  = next(g["kpts"][4] for g in gts if g["cls"] == 1)
            for j in range(len(pred_kpts)):
                pc = pred_kpts[j][4]
                d_to_target = np.linalg.norm(pc - gt_target_c)
                d_to_distr = np.linalg.norm(pc - gt_distr_c)
                if pred_cls[j] == 0 and d_to_distr < d_to_target:
                    role_mismatch = True
                if pred_cls[j] == 1 and d_to_target < d_to_distr:
                    role_mismatch = True
            if role_mismatch:
                role_mismatches.append({
                    "ep": ep, "fr": fr, "cam": cam, "stem": lbl_path.stem,
                    "img_path": str(img_path),
                })

        max_t = per_cls_results[0]["max_err"]
        max_d = per_cls_results[1]["max_err"]
        bbox_t = per_cls_results[0]["bbox_area"]
        bbox_d = per_cls_results[1]["bbox_area"]
        med_err = float(np.median(all_corner_errs)) if all_corner_errs else float("nan")
        max_conf = float(max(pred_conf)) if len(pred_conf) > 0 else 0.0
        gate_clip = vis != "fully_visible"
        gate_small = (bbox_t > 0 and bbox_t < 600) or (bbox_d > 0 and bbox_d < 600)
        gate_lowconf = max_conf < 0.5

        n_matched = len(per_cls_results[0]["matches"]) + len(per_cls_results[1]["matches"])
        csv_lines.append(
            f"{ep},{fr},{cam},{d_cm:.1f},{vis},{len(gts)},{len(pred_kpts)},{n_matched},"
            f"{int(role_mismatch)},{max_t},{max_d},{med_err:.3f},"
            f"{bbox_t:.0f},{bbox_d:.0f},{max_conf:.3f},"
            f"{int(gate_clip)},{int(gate_small)},{int(gate_lowconf)}"
        )

        bkey = (cam, vis)
        buckets[bkey].append(med_err if not np.isnan(med_err) else float("inf"))

        if ep == 221:
            ep221_records.append({
                "fr": fr, "cam": cam, "img_path": str(img_path),
                "gts": gts, "pred_kpts": pred_kpts, "pred_cls": pred_cls,
                "med_err": med_err, "max_t": max_t, "max_d": max_d,
            })

    # Save CSV
    CSV_OUT.write_text("\n".join(csv_lines))
    print(f"\nCSV saved: {CSV_OUT}")

    # Summary stats
    summary = {
        "total_frames_analyzed": len(label_files),
        "role_mismatch_count": len(role_mismatches),
        "role_mismatch_examples": [r["stem"] for r in role_mismatches[:10]],
        "buckets_median_err_px": {},
    }
    for k, vals in sorted(buckets.items()):
        cam, vis = k
        clean = [v for v in vals if v != float("inf")]
        summary["buckets_median_err_px"][f"{cam}/{vis}"] = {
            "n": len(vals), "median": float(np.median(clean)) if clean else None,
            "p90": float(np.percentile(clean, 90)) if clean else None,
            "p95": float(np.percentile(clean, 95)) if clean else None,
        }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    # Render ep221 trajectory grid
    if ep221_records:
        ep221_records.sort(key=lambda r: (r["cam"], r["fr"]))
        n = min(len(ep221_records), 18)
        records = ep221_records[:n]
        crops = []
        for rec in records:
            img = cv2.imread(rec["img_path"])
            for g in rec["gts"]:
                col = (0, 255, 0) if g["cls"] == 0 else (255, 200, 0)
                for j in range(5):
                    p = tuple(g["kpts"][j].astype(int))
                    cv2.circle(img, p, 6, col, -1)
            for j in range(len(rec["pred_kpts"])):
                col = (0, 0, 255) if rec["pred_cls"][j] == 0 else (255, 0, 200)
                for k in range(5):
                    p = tuple(rec["pred_kpts"][j][k].astype(int))
                    cv2.circle(img, p, 6, col, 2)
            # Crop around mean of GT centers
            gt_centers = np.array([g["kpts"][4] for g in rec["gts"]])
            cx, cy = gt_centers.mean(axis=0)
            half = 200
            x0 = max(0, int(cx - half)); x1 = min(img.shape[1], int(cx + half))
            y0 = max(0, int(cy - half)); y1 = min(img.shape[0], int(cy + half))
            crop = cv2.resize(img[y0:y1, x0:x1], (320, 320), interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, f"{rec['cam']} fr{rec['fr']:04d} med={rec['med_err']:.1f}px",
                         (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            crops.append(crop)

        cols = 6
        rows = (len(crops) + cols - 1) // cols
        grid = np.zeros((rows * 320, cols * 320, 3), dtype=np.uint8)
        for i, c in enumerate(crops):
            r, cc = i // cols, i % cols
            grid[r*320:(r+1)*320, cc*320:(cc+1)*320] = c
        cv2.imwrite(str(TRAJECTORY_OUT), grid)
        print(f"ep221 trajectory: {TRAJECTORY_OUT}")

    # Text summary
    lines = [
        "Phase-2 failure investigation summary",
        "=" * 60,
        f"Total test frames analyzed: {summary['total_frames_analyzed']}",
        f"Role mismatch count: {summary['role_mismatch_count']}  ({100*summary['role_mismatch_count']/max(summary['total_frames_analyzed'],1):.2f}%)",
        "",
        "Per-(camera × visibility) bucket median pixel error:",
        f"  {'bucket':<28} {'n':>6} {'median':>10} {'p90':>8} {'p95':>8}",
    ]
    for k, b in summary["buckets_median_err_px"].items():
        med = f"{b['median']:.2f}" if b["median"] is not None else "n/a"
        p90 = f"{b['p90']:.2f}" if b["p90"] is not None else "n/a"
        p95 = f"{b['p95']:.2f}" if b["p95"] is not None else "n/a"
        lines.append(f"  {k:<28} {b['n']:>6} {med:>10} {p90:>8} {p95:>8}")
    lines.append("")
    lines.append(f"ep221 frames analyzed: {len(ep221_records)}")
    if ep221_records:
        ep221_med = np.nanmedian([r['med_err'] for r in ep221_records])
        lines.append(f"  ep221 median per-frame median err: {ep221_med:.2f} px")
    lines.append("")
    SUMMARY_TXT.write_text("\n".join(lines))
    print(f"\nText summary: {SUMMARY_TXT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
