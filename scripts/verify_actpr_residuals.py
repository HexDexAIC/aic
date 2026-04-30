#!/usr/bin/env python3
"""Phase 0 full: derive port-frame residual actions for ALL 488 strict-clean
episodes. Verify recomposition + generate per-channel residual statistics
with the std-floor pathology check from ChatGPT's plan.

Does NOT write a dataset on disk yet — that's the next step. This script
verifies the math at full scale and produces the Phase 0 report.

Outputs:
  /tmp/phase0_report/recomposition_errors.json
  /tmp/phase0_report/residual_stats.json    (per-channel mean/std/min/max/q01/q99)
  /tmp/phase0_report/residual_histograms.png
  /tmp/phase0_report/std_floor_check.json   (any pathological dims)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "HexDexAIC/aic-sfp-500"
N_TOTAL = 500
SWEEP_DIRS = [
    Path("/home/hariharan/ws_aic/aic_results/spawn_sweep_20260428_160056"),
    Path("/home/hariharan/ws_aic/aic_results/spawn_sweep_20260427_080402"),
]
OUT_DIR = Path("/tmp/phase0_report")

STD_FLOORS = {
    "p_resid": 1e-3,       # 1 mm
    "rot6_resid": 1e-2,
    "port_xyz": 1e-3,
    "port_rot6": 1e-2,
}


# ── geometry helpers ──────────────────────────────────────────────────
def quat_wxyz_to_R(qw, qx, qy, qz):
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz), s*(qx*qy - qz*qw),     s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),     1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),     s*(qy*qz + qx*qw),     1 - s*(qx*qx + qy*qy)],
    ])


def R_to_quat_wxyz(R):
    """Shepperd's method, R → quat (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


def quat_to_rot6(qw, qx, qy, qz):
    """Same convention as record_lerobot.quat_to_rotmat_6d: first two
    columns of the rotation matrix concatenated."""
    R = quat_wxyz_to_R(qw, qx, qy, qz)
    return np.concatenate([R[:, 0], R[:, 1]])


def rot6_to_R(rot6):
    a1, a2 = rot6[:3], rot6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def R_to_rot6(R):
    return np.concatenate([R[:, 0], R[:, 1]])


def geodesic_R(R1, R2):
    M = R1.T @ R2
    cos = (np.trace(M) - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    return float(np.arccos(cos))


# ── episode discovery ────────────────────────────────────────────────
_PORT_XYZ_RE = re.compile(r"port_xyz:\s*\(([^)]+)\)")
_PORT_QUAT_RE = re.compile(r"port_quat_wxyz:\s*\(([^)]+)\)")


def find_summary_log(seed: int):
    for sweep in SWEEP_DIRS:
        seed_dir = sweep / "seeds" / f"seed_{seed:02d}"
        if not seed_dir.is_dir():
            continue
        cm_dir = seed_dir / "cheatcode_mj"
        if not cm_dir.is_dir():
            continue
        candidates = list(cm_dir.glob("*_summary.log"))
        if candidates:
            return candidates[0]
    return None


def parse_port_pose(summary_path: Path):
    text = summary_path.read_text()
    xyz = _PORT_XYZ_RE.findall(text)
    quat = _PORT_QUAT_RE.findall(text)
    if not xyz or not quat:
        raise ValueError(f"no port pose in {summary_path.name}")
    parse = lambda s: tuple(float(x.strip()) for x in s.split(","))
    return np.asarray(parse(xyz[0]), dtype=np.float64), parse(quat[0])


# ── main ─────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-episode aggregates
    per_ep = []
    p_resid_all, rot6_resid_all = [], []
    port_xyz_all, port_rot6_all = [], []
    err_p_max_all, err_R_max_all = [], []

    n_strict, n_failed_filter, n_no_log, n_skip_other = 0, 0, 0, 0

    print(f"Phase 0 full: scanning {N_TOTAL} episodes...", flush=True)
    for ep in range(N_TOTAL):
        if ep % 50 == 0 and ep > 0:
            print(f"  ep {ep}/{N_TOTAL} (strict={n_strict}, "
                  f"failed_filter={n_failed_filter}, no_log={n_no_log})",
                  flush=True)

        try:
            parquet_path = hf_hub_download(
                REPO, f"data/chunk-000/file-{ep:03d}.parquet",
                repo_type="dataset",
            )
        except Exception as e:
            n_skip_other += 1
            continue

        t = pq.read_table(parquet_path,
                          columns=["action", "episode_success", "num_attempts"])
        es = t.column("episode_success")[0].as_py()
        na = t.column("num_attempts")[0].as_py()
        if not (es >= 0.5 and na == 1):
            n_failed_filter += 1
            continue

        log_path = find_summary_log(ep)
        if log_path is None:
            n_no_log += 1
            continue

        p_port, q_port = parse_port_pose(log_path)
        R_port = quat_wxyz_to_R(*q_port)
        port_rot6 = quat_to_rot6(*q_port)

        action_abs = np.stack(t.column("action").to_pylist()).astype(np.float64)
        n = action_abs.shape[0]

        p_resid = np.empty((n, 3))
        rot6_resid = np.empty((n, 6))
        err_p, err_R = np.empty(n), np.empty(n)

        for i in range(n):
            p_act, rot6_act = action_abs[i, :3], action_abs[i, 3:9]
            R_act = rot6_to_R(rot6_act)
            R_resid = R_port.T @ R_act
            p_r = R_port.T @ (p_act - p_port)
            p_resid[i] = p_r
            rot6_resid[i] = R_to_rot6(R_resid)
            R_recomp = R_port @ R_resid
            p_recomp = R_port @ p_r + p_port
            err_p[i] = np.linalg.norm(p_recomp - p_act)
            err_R[i] = geodesic_R(R_recomp, R_act)

        p_resid_all.append(p_resid)
        rot6_resid_all.append(rot6_resid)
        port_xyz_all.append(p_port)
        port_rot6_all.append(port_rot6)
        err_p_max_all.append(float(err_p.max()))
        err_R_max_all.append(float(err_R.max()))
        per_ep.append({
            "ep": ep,
            "n_frames": int(n),
            "err_p_max": float(err_p.max()),
            "err_p_mean": float(err_p.mean()),
            "err_R_max": float(err_R.max()),
            "err_R_mean": float(err_R.mean()),
        })
        n_strict += 1

    print(f"\nDONE: {n_strict} strict-clean episodes processed, "
          f"{n_failed_filter} filtered (success/attempts), "
          f"{n_no_log} no_log, {n_skip_other} other_skip", flush=True)

    if n_strict == 0:
        print("ERROR: no strict-clean episodes processed.")
        sys.exit(1)

    # ── aggregate stats ──────────────────────────────────────────
    p_resid_cat = np.concatenate(p_resid_all)        # (sum_frames, 3)
    rot6_resid_cat = np.concatenate(rot6_resid_all)  # (sum_frames, 6)
    port_xyz_arr = np.stack(port_xyz_all)            # (n_ep, 3)
    port_rot6_arr = np.stack(port_rot6_all)          # (n_ep, 6)

    def per_dim_stats(arr, label):
        return {
            "label": label,
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
            "q01":  np.quantile(arr, 0.01, axis=0).tolist(),
            "q99":  np.quantile(arr, 0.99, axis=0).tolist(),
            "n":    int(arr.shape[0]),
        }

    stats = {
        "p_resid":    per_dim_stats(p_resid_cat,    "action.p_resid (m)"),
        "rot6_resid": per_dim_stats(rot6_resid_cat, "action.rot6_resid"),
        "port_xyz":   per_dim_stats(port_xyz_arr,   "observation.port_pose_gt.xyz (m, per-episode)"),
        "port_rot6":  per_dim_stats(port_rot6_arr,  "observation.port_pose_gt.rot6 (per-episode)"),
    }
    (OUT_DIR / "residual_stats.json").write_text(json.dumps(stats, indent=2))

    # ── std-floor pathology check ────────────────────────────────
    floored = []
    for k, st in stats.items():
        floor = STD_FLOORS[k]
        for i, s in enumerate(st["std"]):
            if s < floor:
                floored.append({
                    "feature": k, "dim_idx": i, "std": float(s), "floor": floor,
                })
    (OUT_DIR / "std_floor_check.json").write_text(
        json.dumps({"floors": STD_FLOORS, "violations": floored}, indent=2)
    )

    # ── recomposition error summary ──────────────────────────────
    recomp = {
        "n_strict_clean": n_strict,
        "err_p_max_overall": float(np.max(err_p_max_all)),
        "err_p_max_p99":     float(np.quantile(err_p_max_all, 0.99)),
        "err_R_max_overall": float(np.max(err_R_max_all)),
        "err_R_max_p99":     float(np.quantile(err_R_max_all, 0.99)),
        "exit_signals": {
            "err_p_max < 1e-5 m":  bool(np.max(err_p_max_all) < 1e-5),
            "err_R_max < 1e-3 rad": bool(np.max(err_R_max_all) < 1e-3),
        },
        "per_episode": per_ep,
    }
    (OUT_DIR / "recomposition_errors.json").write_text(json.dumps(recomp, indent=2))

    # ── histograms ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(13, 9))
        # Top row: action.p_resid (3 dims)
        for i, ax in enumerate(axes[0]):
            ax.hist(p_resid_cat[:, i], bins=80)
            ax.set_title(f"action.p_resid[{i}] (m) — std={stats['p_resid']['std'][i]:.4f}")
            ax.grid(True, alpha=0.3)
        # Middle row: rot6_resid (first 3 dims for space)
        for i, ax in enumerate(axes[1]):
            ax.hist(rot6_resid_cat[:, i], bins=80, color="tab:orange")
            ax.set_title(f"action.rot6_resid[{i}] — std={stats['rot6_resid']['std'][i]:.4f}")
            ax.grid(True, alpha=0.3)
        # Bottom row: port_xyz (3 dims, per-episode)
        for i, ax in enumerate(axes[2]):
            ax.hist(port_xyz_arr[:, i], bins=40, color="tab:green")
            ax.set_title(f"port_xyz[{i}] (m) — std={stats['port_xyz']['std'][i]:.4f}")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Phase 0 residual distributions  ({n_strict} strict-clean episodes, "
                     f"{p_resid_cat.shape[0]} frames)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "residual_histograms.png", dpi=110)
        print(f"wrote {OUT_DIR/'residual_histograms.png'}")
    except ImportError as e:
        print(f"(skipping histograms: {e})")

    # ── final summary ───────────────────────────────────────────
    print("\n=== EXIT SIGNALS ===")
    for k, v in recomp["exit_signals"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"\n  err_p_max overall: {recomp['err_p_max_overall']:.2e} m  "
          f"(p99: {recomp['err_p_max_p99']:.2e})")
    print(f"  err_R_max overall: {recomp['err_R_max_overall']:.2e} rad  "
          f"(p99: {recomp['err_R_max_p99']:.2e})")

    if floored:
        print(f"\n=== STD-FLOOR VIOLATIONS ({len(floored)}) ===")
        for f in floored:
            print(f"  {f['feature']}[{f['dim_idx']}]: std={f['std']:.2e}  "
                  f"(floor={f['floor']:.2e})")
    else:
        print("\nNo std-floor violations.")

    print(f"\nReports written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
