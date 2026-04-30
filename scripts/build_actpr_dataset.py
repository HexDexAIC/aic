#!/usr/bin/env python3
"""Phase 0: build the derived ACT-PR LeRobotDataset.

Output structure (mirrors lerobot v3.0 conventions, except video files are
SYMLINKED from the source HF cache to avoid re-encoding 5.9 GB of MP4):

  ~/aic_results/aic-sfp-500-pr/
    data/chunk-000/file-XXX.parquet     ← rebuilt: residual action + port_pose_gt
    videos/observation.images.{view}/chunk-000/file-XXX.mp4  ← symlink to source
    meta/info.json                      ← schema updated
    meta/stats.json                     ← recomputed for new action + port_pose_gt
    meta/tasks.parquet                  ← copy from source
    meta/episodes/chunk-000/file-XXX.parquet  ← copy from source

The transform per row:
  T_action_abs = (p_target_t in base_link, R_target_t)
  T_residual   = T_port⁻¹ · T_action_abs
  observation.port_pose_gt = (port_xyz, port_rot6)   ← constant per episode

For non-strict-clean episodes (success=0 or attempts>1, 12 of them):
the residual is still well-defined since port_pose is from CheatCodeMJ's
target. We keep them in the dataset; the strict-clean filter is applied at
training time via an episodes list (clean_eps.txt-style).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC_SNAP = Path("/home/hariharan/.cache/huggingface/hub/datasets--HexDexAIC--aic-sfp-500/snapshots/a84f50d51f4849b118f3da1f8eff190102b00a5d")
OUT = Path("/home/hariharan/aic_results/aic-sfp-500-pr")
SWEEP_DIRS = [
    Path("/home/hariharan/ws_aic/aic_results/spawn_sweep_20260428_160056"),
    Path("/home/hariharan/ws_aic/aic_results/spawn_sweep_20260427_080402"),
]
N_TOTAL = 500
STD_FLOORS = {"p_resid": 1e-3, "rot6_resid": 1e-2, "port_pose_gt": 1e-3}


# ── geometry (same as phase0_full_residuals.py) ────────────────────────
def quat_wxyz_to_R(qw, qx, qy, qz):
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-12: return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(qy*qy + qz*qz), s*(qx*qy - qz*qw),     s*(qx*qz + qy*qw)],
        [s*(qx*qy + qz*qw),     1 - s*(qx*qx + qz*qz), s*(qy*qz - qx*qw)],
        [s*(qx*qz - qy*qw),     s*(qy*qz + qx*qw),     1 - s*(qx*qx + qy*qy)],
    ])


def quat_to_rot6(qw, qx, qy, qz):
    R = quat_wxyz_to_R(qw, qx, qy, qz)
    return np.concatenate([R[:, 0], R[:, 1]])


def rot6_to_R(rot6):
    a1, a2 = rot6[:3], rot6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    a2_proj = a2 - np.dot(b1, a2) * b1
    b2 = a2_proj / (np.linalg.norm(a2_proj) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def R_to_rot6(R): return np.concatenate([R[:, 0], R[:, 1]])


_PORT_XYZ_RE = re.compile(r"port_xyz:\s*\(([^)]+)\)")
_PORT_QUAT_RE = re.compile(r"port_quat_wxyz:\s*\(([^)]+)\)")


def find_summary_log(seed: int):
    for sweep in SWEEP_DIRS:
        sd = sweep / "seeds" / f"seed_{seed:02d}" / "cheatcode_mj"
        if sd.is_dir():
            cands = list(sd.glob("*_summary.log"))
            if cands: return cands[0]
    return None


def parse_port_pose(p: Path):
    txt = p.read_text()
    xyz = _PORT_XYZ_RE.findall(txt); quat = _PORT_QUAT_RE.findall(txt)
    parse = lambda s: tuple(float(x.strip()) for x in s.split(","))
    return np.asarray(parse(xyz[0]), dtype=np.float64), parse(quat[0])


# ── per-episode transform ─────────────────────────────────────────────
def transform_episode(ep: int):
    """Return (new_table, port_pose_gt 9-vec, n_frames)
    or (None, None, 0) if no port pose available."""
    src_pq = SRC_SNAP / "data" / "chunk-000" / f"file-{ep:03d}.parquet"
    if not src_pq.exists():
        return None, None, 0

    log_path = find_summary_log(ep)
    if log_path is None:
        return None, None, 0  # skip episodes with no log

    p_port, q_port = parse_port_pose(log_path)
    R_port = quat_wxyz_to_R(*q_port)
    R_port_T = R_port.T
    port_rot6 = quat_to_rot6(*q_port)
    port_pose_gt = np.concatenate([p_port, port_rot6]).astype(np.float32)  # (9,)

    t = pq.read_table(src_pq)
    action_abs = np.stack(t.column("action").to_pylist()).astype(np.float64)
    n = action_abs.shape[0]

    new_action = np.empty((n, 9), dtype=np.float32)
    for i in range(n):
        p_act, rot6_act = action_abs[i, :3], action_abs[i, 3:9]
        R_act = rot6_to_R(rot6_act)
        p_resid = R_port_T @ (p_act - p_port)
        rot6_resid = R_to_rot6(R_port_T @ R_act)
        new_action[i, :3] = p_resid.astype(np.float32)
        new_action[i, 3:9] = rot6_resid.astype(np.float32)

    # Build the new table: replace `action`, add `observation.port_pose_gt`.
    ep_port_pose = np.tile(port_pose_gt, (n, 1))  # broadcast to (n, 9)

    cols = {}
    for name in t.column_names:
        if name == "action":
            cols[name] = pa.array(
                new_action.tolist(),
                type=pa.list_(pa.float32(), 9),
            )
        else:
            cols[name] = t.column(name)
    cols["observation.port_pose_gt"] = pa.array(
        ep_port_pose.tolist(), type=pa.list_(pa.float32(), 9)
    )

    schema_fields = []
    for name, col in cols.items():
        schema_fields.append(pa.field(name, col.type))
    new_table = pa.Table.from_arrays(list(cols.values()), schema=pa.schema(schema_fields))
    return new_table, port_pose_gt, n


# ── driver ────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (OUT / "meta").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)

    # ── per-episode parquet rewriting ─────────────────────────────
    print("Building per-episode parquets...", flush=True)
    n_processed = 0
    n_skip = 0
    p_resid_all, rot6_resid_all = [], []
    port_xyz_all = []
    total_frames = 0
    episodes_in_dataset = []  # episode indices that made it in

    for ep in range(N_TOTAL):
        new_table, port_pose, n = transform_episode(ep)
        if new_table is None:
            n_skip += 1
            continue
        out_path = OUT / "data" / "chunk-000" / f"file-{ep:03d}.parquet"
        pq.write_table(new_table, out_path)

        action_arr = np.stack(new_table.column("action").to_pylist())
        p_resid_all.append(action_arr[:, :3])
        rot6_resid_all.append(action_arr[:, 3:9])
        port_xyz_all.append(port_pose[:3])
        episodes_in_dataset.append(ep)
        total_frames += n
        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  {n_processed} eps processed, {total_frames} frames", flush=True)

    print(f"\nDone: {n_processed} episodes, {n_skip} skipped, {total_frames} total frames\n",
          flush=True)

    # ── symlink videos from source ────────────────────────────────
    print("Symlinking videos from source HF cache...", flush=True)
    for view in ("left", "center", "right"):
        src_dir = SRC_SNAP / "videos" / f"observation.images.{view}" / "chunk-000"
        dst_dir = OUT / "videos" / f"observation.images.{view}" / "chunk-000"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for ep in episodes_in_dataset:
            src = src_dir / f"file-{ep:03d}.mp4"
            if src.exists():
                dst = dst_dir / f"file-{ep:03d}.mp4"
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
    print("  symlinks done", flush=True)

    # ── copy meta/episodes/, meta/tasks.parquet ──────────────────
    src_episodes = SRC_SNAP / "meta" / "episodes"
    dst_episodes = OUT / "meta" / "episodes"
    if dst_episodes.exists():
        shutil.rmtree(dst_episodes)
    shutil.copytree(src_episodes, dst_episodes)
    shutil.copy2(SRC_SNAP / "meta" / "tasks.parquet", OUT / "meta" / "tasks.parquet")
    print("  copied meta/episodes/ and meta/tasks.parquet", flush=True)

    # ── build new info.json ──────────────────────────────────────
    src_info = json.loads((SRC_SNAP / "meta" / "info.json").read_text())
    new_info = dict(src_info)  # shallow copy
    new_info["total_episodes"] = n_processed
    new_info["total_frames"] = total_frames
    new_info["splits"] = {"train": f"0:{total_frames}"}

    # Update action feature names to reflect the new semantics; shape unchanged.
    feats = dict(src_info["features"])
    feats["action"] = {
        "dtype": "float32",
        "shape": [9],
        "names": [
            "p_residual_x_in_port_frame",
            "p_residual_y_in_port_frame",
            "p_residual_z_in_port_frame",
            "rot6_residual_0",
            "rot6_residual_1",
            "rot6_residual_2",
            "rot6_residual_3",
            "rot6_residual_4",
            "rot6_residual_5",
        ],
    }
    feats["observation.port_pose_gt"] = {
        "dtype": "float32",
        "shape": [9],
        "names": [
            "port_x", "port_y", "port_z",
            "port_rot6_0", "port_rot6_1", "port_rot6_2",
            "port_rot6_3", "port_rot6_4", "port_rot6_5",
        ],
    }
    new_info["features"] = feats
    (OUT / "meta" / "info.json").write_text(json.dumps(new_info, indent=2))
    print("  wrote meta/info.json", flush=True)

    # ── stats.json ─────────────────────────────────────────────
    p_resid_cat = np.concatenate(p_resid_all)
    rot6_resid_cat = np.concatenate(rot6_resid_all)
    port_xyz_arr = np.stack(port_xyz_all)

    def floored_std(arr, floor):
        s = arr.std(axis=0)
        return np.maximum(s, floor)

    stats = {
        "action": {
            "mean": np.concatenate([p_resid_cat.mean(0), rot6_resid_cat.mean(0)]).tolist(),
            "std":  np.concatenate([
                floored_std(p_resid_cat, STD_FLOORS["p_resid"]),
                floored_std(rot6_resid_cat, STD_FLOORS["rot6_resid"]),
            ]).tolist(),
            "min":  np.concatenate([p_resid_cat.min(0), rot6_resid_cat.min(0)]).tolist(),
            "max":  np.concatenate([p_resid_cat.max(0), rot6_resid_cat.max(0)]).tolist(),
        },
        "observation.port_pose_gt": {
            "mean": np.concatenate([
                port_xyz_arr.mean(0),
                # port_rot6 is aggregated per-frame from the new table — but
                # the value is constant per episode, so the per-episode array
                # is sufficient. We use the per-episode stats rounded to fit
                # the per-frame distribution (since port pose is broadcast
                # constant within an episode, per-frame mean = per-episode
                # weighted by episode length).
                np.zeros(6).tolist(),  # filled below
            ]).tolist(),
        },
    }

    # Rebuild port_pose stats correctly using per-frame broadcasting.
    port_pose_per_frame = []
    for ep, p_xyz in zip(episodes_in_dataset, port_xyz_all):
        # need to pull port_rot6 too; recomputing from the parquet
        pq_path = OUT / "data" / "chunk-000" / f"file-{ep:03d}.parquet"
        pp = np.stack(pq.read_table(pq_path, columns=["observation.port_pose_gt"])
                       .column("observation.port_pose_gt").to_pylist())
        port_pose_per_frame.append(pp)
    port_pose_cat = np.concatenate(port_pose_per_frame)

    stats["observation.port_pose_gt"] = {
        "mean": port_pose_cat.mean(axis=0).tolist(),
        "std":  np.maximum(port_pose_cat.std(axis=0),
                            STD_FLOORS["port_pose_gt"]).tolist(),
        "min":  port_pose_cat.min(axis=0).tolist(),
        "max":  port_pose_cat.max(axis=0).tolist(),
    }
    (OUT / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
    print("  wrote meta/stats.json", flush=True)

    print(f"\n=== DONE ===")
    print(f"Output: {OUT}")
    print(f"Episodes: {n_processed}/{N_TOTAL}  (skipped: {n_skip})")
    print(f"Frames:   {total_frames}")
    print(f"Disk usage:")
    import subprocess
    print("  ", subprocess.check_output(["du", "-sh", str(OUT)]).decode().strip())


if __name__ == "__main__":
    main()
