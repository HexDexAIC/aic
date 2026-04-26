#!/usr/bin/env python3
"""Comprehensive per-episode inspection of a spawn-sweep.

For each seed, prints:
  - sampled spawn spec
  - actually-spawned task_board pose (from spawn_verification.json)
  - episode outcome (frames, attempts, inserted, final-dist)
  - dataset stats: TCP travel range, wrench peak, action.z descent depth
  - validation flags

Run with the pixi env (needs pyarrow):
  pixi run python src/aic/scripts/inspect_sweep.py <sweep_dir>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq


def find_dataset(seed_dataset_dir: Path) -> Path | None:
    if not seed_dataset_dir.exists():
        return None
    cands = sorted(p for p in seed_dataset_dir.iterdir()
                   if p.is_dir() and p.name.startswith("aic_recording_"))
    return cands[-1] if cands else None


def episode_stats(ds: Path) -> dict:
    """Pull per-frame stats from the parquet."""
    pq_files = sorted((ds / "data").glob("chunk-*/file-*.parquet"))
    if not pq_files:
        return {}
    tbl = pq.read_table(pq_files[0],
                        columns=["observation.state", "action"])
    n = tbl.num_rows
    states = tbl.column("observation.state").to_pylist()
    actions = tbl.column("action").to_pylist()

    # observation.state layout: 0:3 = pos, 3:9 = rot6, 9:12 = lin vel,
    # 12:15 = ang vel, 15:18 = F (N), 18:21 = tau (N·m), 21:27 = joints.
    pos = [(s[0], s[1], s[2]) for s in states]
    forces = [(s[15], s[16], s[17]) for s in states]
    torques = [(s[18], s[19], s[20]) for s in states]

    pos_x = [p[0] for p in pos]; pos_y = [p[1] for p in pos]; pos_z = [p[2] for p in pos]

    fmag = [math.sqrt(f[0]**2 + f[1]**2 + f[2]**2) for f in forces]
    tmag = [math.sqrt(t[0]**2 + t[1]**2 + t[2]**2) for t in torques]

    a_z = [a[2] for a in actions]

    return {
        "frames": n,
        "tcp_x_range_mm": (max(pos_x) - min(pos_x)) * 1000,
        "tcp_y_range_mm": (max(pos_y) - min(pos_y)) * 1000,
        "tcp_z_range_mm": (max(pos_z) - min(pos_z)) * 1000,
        "tcp_z_min_m": min(pos_z),
        "tcp_z_max_m": max(pos_z),
        "force_max_N": max(fmag),
        "force_baseline_N": fmag[0] if fmag else None,
        "torque_max_Nm": max(tmag),
        "action_z_min_m": min(a_z),
        "action_z_max_m": max(a_z),
        "action_z_descent_mm": (max(a_z) - min(a_z)) * 1000,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: inspect_sweep.py <sweep_dir>", file=sys.stderr)
        return 2
    sweep = Path(sys.argv[1])

    samples = json.loads((sweep / "samples.json").read_text())
    summary = json.loads((sweep / "summary.json").read_text())
    spawn = json.loads((sweep / "spawn_verification.json").read_text())
    val = json.loads((sweep / "dataset_validation.json").read_text())

    by_seed_summary = {r["seed"]: r for r in summary["results"]}
    by_seed_spawn = {r["seed"]: r for r in spawn["results"]}
    by_seed_val = {r["seed"]: r for r in val["results"]}

    rows = []
    for spec in samples:
        seed = spec["seed"]
        sm = by_seed_summary[seed]
        sp = by_seed_spawn.get(seed, {})
        vl = by_seed_val.get(seed, {})

        ds_path = sweep / "datasets" / f"seed_{seed:02d}"
        ds_dir = find_dataset(ds_path)
        stats = episode_stats(ds_dir) if ds_dir else {}

        rows.append({
            "seed": seed,
            "spec": spec,
            "summary": sm,
            "spawn": sp,
            "val": vl,
            "stats": stats,
        })

    # Print compact table.
    print(f"{'seed':>4} {'rail':>4} {'bd_x':>8} {'bd_y':>8} {'bd_yaw':>8} "
          f"{'nic_t':>8} {'nic_y':>8} {'gx':>8} {'gy':>8} {'gz':>8}  "
          f"{'spawn':>5} {'val':>4} {'frm':>5} {'ins':>3} {'#':>3} "
          f"{'dist':>7} {'fmax':>5} {'taumax':>6} {'desc':>6}")
    print("-" * 168)
    for r in rows:
        s = r["spec"]
        sm = r["summary"]
        st = r["stats"]
        sp = r["spawn"]
        vl = r["val"]

        spawn_ok = "✓" if sp.get("matched") else ("?" if not sp.get("checked") else "✗")
        val_ok = "✓" if vl.get("ok") else "✗"
        ins = sm["policy"].get("inserted")
        ins_str = "✓" if ins is True else ("✗" if ins is False else "?")
        att = sm["policy"].get("attempts") or "—"
        dist = sm["policy"].get("final_dist_m")
        dist_str = f"{dist*1000:5.1f}mm" if dist is not None else "    —"

        print(f"{r['seed']:>4d} {s['nic_rail']:>4d} "
              f"{s['board_x']:>+8.4f} {s['board_y']:>+8.4f} {s['board_yaw']:>+8.4f} "
              f"{s['nic_translation']:>+8.4f} {s['nic_yaw']:>+8.4f} "
              f"{s['grip_x']:>+8.4f} {s['grip_y']:>+8.4f} {s['grip_z']:>+8.4f}  "
              f"{spawn_ok:>5} {val_ok:>4} "
              f"{st.get('frames','—'):>5} {ins_str:>3} {att:>3} "
              f"{dist_str:>7} "
              f"{st.get('force_max_N',0):>5.1f} {st.get('torque_max_Nm',0):>6.2f} "
              f"{st.get('action_z_descent_mm',0):>5.1f}")

    # Aggregate stats.
    print()
    print("=== Aggregates ===")
    print(f"  episodes:              {len(rows)}")
    print(f"  spawn-matched config:  {sum(1 for r in rows if r['spawn'].get('matched'))}/{len(rows)}")
    print(f"  dataset valid:         {sum(1 for r in rows if r['val'].get('ok'))}/{len(rows)}")
    print(f"  inserted:              {sum(1 for r in rows if r['summary']['policy'].get('inserted') is True)}/{len(rows)}")
    print()
    print("  range across episodes:")
    bd_x = [r["spec"]["board_x"] for r in rows]
    bd_y = [r["spec"]["board_y"] for r in rows]
    bd_yaw = [r["spec"]["board_yaw"] for r in rows]
    print(f"    board_x:    [{min(bd_x):+.4f}, {max(bd_x):+.4f}] m  (span {(max(bd_x)-min(bd_x))*1000:.1f} mm)")
    print(f"    board_y:    [{min(bd_y):+.4f}, {max(bd_y):+.4f}] m  (span {(max(bd_y)-min(bd_y))*1000:.1f} mm)")
    print(f"    board_yaw:  [{min(bd_yaw):+.4f}, {max(bd_yaw):+.4f}] rad  (span {math.degrees(max(bd_yaw)-min(bd_yaw)):.1f}°)")

    rail_count = {}
    for r in rows:
        rail_count[r["spec"]["nic_rail"]] = rail_count.get(r["spec"]["nic_rail"], 0) + 1
    print(f"    nic_rail dist: {dict(sorted(rail_count.items()))}")

    fmaxes = [r["stats"].get("force_max_N", 0) for r in rows if r["stats"]]
    descents = [r["stats"].get("action_z_descent_mm", 0) for r in rows if r["stats"]]
    if fmaxes:
        print()
        print("  per-episode peaks:")
        print(f"    force_max:           [{min(fmaxes):.1f}, {max(fmaxes):.1f}] N")
        print(f"    descent depth:       [{min(descents):.1f}, {max(descents):.1f}] mm")

    return 0


if __name__ == "__main__":
    sys.exit(main())
