#!/usr/bin/env python3
"""Render the morning briefing for a spawn sweep.

Reads <sweep_dir>/{samples.json, summary.json, spawn_verification.json,
dataset_validation.json} and writes <sweep_dir>/briefing.md.

Usage:
  python src/aic/scripts/build_briefing.py <sweep_dir> [--total-elapsed-s SECONDS]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(sec):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(sec):02d}s"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir", type=Path)
    p.add_argument("--total-elapsed-s", type=float, default=None,
                   help="Override the total wall-clock elapsed (otherwise from summary.json).")
    p.add_argument("--task-started-at", default=None,
                   help="ISO timestamp when the human kicked off the task (for total-time accounting).")
    p.add_argument("--task-ended-at", default=None,
                   help="ISO timestamp when post-processing finished.")
    args = p.parse_args()

    sweep_dir: Path = args.sweep_dir
    samples = json.loads((sweep_dir / "samples.json").read_text())
    summary = json.loads((sweep_dir / "summary.json").read_text())

    spawn_path = sweep_dir / "spawn_verification.json"
    spawn = json.loads(spawn_path.read_text()) if spawn_path.exists() else None

    val_path = sweep_dir / "dataset_validation.json"
    val = json.loads(val_path.read_text()) if val_path.exists() else None

    n = summary.get("n", len(samples))
    results = summary.get("results", [])
    elapsed_s = args.total_elapsed_s if args.total_elapsed_s is not None else summary.get("elapsed_s", 0.0)

    # Aggregates
    inserted = sum(1 for r in results if r["policy"].get("inserted") is True)
    not_inserted = sum(1 for r in results if r["policy"].get("inserted") is False)
    no_log = sum(1 for r in results if r["policy"].get("found") is not True)
    attempts_dist: dict[int, int] = {}
    for r in results:
        a = r["policy"].get("attempts")
        if a is not None:
            attempts_dist[a] = attempts_dist.get(a, 0) + 1

    rail_dist: dict[int, list[bool]] = {}
    for r in results:
        rail = r["spec"]["nic_rail"]
        rail_dist.setdefault(rail, []).append(r["policy"].get("inserted") is True)

    spawn_matched = spawn["n_matched"] if spawn else 0
    spawn_checked = spawn["n_checked"] if spawn else 0
    val_ok = val["n_ok"] if val else 0
    val_total = val["n_total"] if val else 0

    # Per-seed rows
    rows = []
    for r in results:
        seed = r["seed"]
        spec = r["spec"]
        pol = r["policy"]
        ds = r["dataset"]
        spawn_row = None
        if spawn:
            for sr in spawn["results"]:
                if sr["seed"] == seed:
                    spawn_row = sr
                    break
        val_row = None
        if val:
            for vr in val["results"]:
                if vr["seed"] == seed:
                    val_row = vr
                    break
        ins = pol.get("inserted")
        ins_str = "✓" if ins is True else "✗" if ins is False else "?"
        atts = pol.get("attempts")
        dist = pol.get("final_dist_m")
        spawn_str = (
            "✓" if spawn_row and spawn_row.get("matched")
            else "—" if spawn_row is None
            else "✗"
        )
        val_str = (
            "✓" if val_row and val_row.get("ok")
            else "—" if val_row is None
            else "✗"
        )
        rows.append({
            "seed": seed,
            "rail": spec["nic_rail"],
            "elapsed_s": r["run"]["elapsed_s"],
            "ins_str": ins_str,
            "attempts": atts,
            "dist_mm": (dist * 1000.0) if dist is not None else None,
            "spawn_str": spawn_str,
            "val_str": val_str,
            "frames": ds.get("frames"),
        })

    md = []
    md.append(f"# Morning Briefing — Spawn Sweep")
    md.append("")
    md.append(f"Sweep dir: `{sweep_dir}`")
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"- **Episodes ran**: {len(results)}/{n}")
    md.append(f"- **CheatCodeMJ inserted**: {inserted}/{len(results)} "
              f"({inserted/max(1,len(results)):.0%})")
    md.append(f"- **Spawn matches config (≤1 mm / ≤1 mrad)**: "
              f"{spawn_matched}/{spawn_checked}" + (" (post-pass)" if spawn else " — not run"))
    md.append(f"- **Datasets valid (schema + rot unit-norm + descent)**: "
              f"{val_ok}/{val_total}" + (" (post-pass)" if val else " — not run"))
    md.append(f"- **Total wall-clock (sweep only)**: {fmt_dur(elapsed_s)}")
    if args.task_started_at and args.task_ended_at:
        from datetime import datetime
        st = datetime.fromisoformat(args.task_started_at)
        et = datetime.fromisoformat(args.task_ended_at)
        md.append(f"- **Total wall-clock (task end-to-end)**: "
                  f"{fmt_dur((et - st).total_seconds())} "
                  f"(from {st.strftime('%H:%M:%S')} to {et.strftime('%H:%M:%S')})")
    md.append("")

    md.append("## Time breakdown")
    md.append("")
    if args.task_started_at and args.task_ended_at:
        from datetime import datetime
        st = datetime.fromisoformat(args.task_started_at)
        et = datetime.fromisoformat(args.task_ended_at)
        total = (et - st).total_seconds()
        md.append(f"- Task kickoff → first episode: ~10 min (build sampler, "
                  f"templater, driver, verifier, validator, briefing scripts; "
                  f"plus one false-start fixing the headless `docker exec` "
                  f"path because `distrobox enter -r` needs sudo from a TTY)")
        md.append(f"- 20 episodes (sequential, ~50 s each): {fmt_dur(elapsed_s)}")
        md.append(f"- Retry pass + spawn-match verification + dataset "
                  f"validation + briefing render: "
                  f"{fmt_dur(total - elapsed_s - 600)}")
        md.append(f"- **Total**: {fmt_dur(total)}")
    else:
        md.append(f"- 20 episodes (sequential): {fmt_dur(elapsed_s)}")
    md.append("")

    md.append("## What ran")
    md.append("")
    md.append("- Sampler: stratified-uniform over 8 continuous knobs + 1 discrete (1-of-5 NIC rail). "
              "Distribution: 4 episodes per NIC rail (0..4).")
    md.append("- Templater: copies `single_trial_sfp.yaml`, fills task-board pose / target rail / "
              "cable-grasp from each spec. Other rails forced absent.")
    md.append("- Driver: per seed → write YAML → run `record_episode.sh sfp --config ...` "
              "(under `AIC_USE_DOCKER_EXEC=1`) → CheatCodeMJ policy → recorder writes "
              "LeRobotDataset → driver captures policy log + bag dir.")
    md.append("- Spawn-vs-config check: `verify_spawn_match.py` reads `/scoring/tf` from each "
              "trial bag, extracts the spawned task_board pose, compares against the YAML.")
    md.append("- Dataset validation: `validate_sweep_datasets.py` checks meta/info.json schema "
              "(27-D state, 9-D action, 6-D stiffness, ≥1 video stream), parquet frame count "
              "matches meta, and 6-D rotation columns are unit-norm at first frame.")
    md.append("")

    md.append("## Per-seed table")
    md.append("")
    md.append("| seed | rail | elapsed | inserted | attempts | final dist | spawn-match | dataset | frames |")
    md.append("|---:|---:|---:|:---:|---:|---:|:---:|:---:|---:|")
    for r in rows:
        dist = f"{r['dist_mm']:.2f} mm" if r["dist_mm"] is not None else "—"
        att = str(r["attempts"]) if r["attempts"] is not None else "—"
        elapsed = f"{r['elapsed_s']:.0f}s"
        frames = str(r["frames"]) if r["frames"] is not None else "—"
        md.append(f"| {r['seed']:02d} | {r['rail']} | {elapsed} | {r['ins_str']} | {att} | {dist} | {r['spawn_str']} | {r['val_str']} | {frames} |")
    md.append("")

    md.append("## Insertion success breakdown")
    md.append("")
    md.append(f"- Inserted: **{inserted}/{len(results)}**")
    md.append(f"- Not inserted: {not_inserted}/{len(results)}")
    if no_log:
        md.append(f"- No policy-log finding: {no_log}/{len(results)}")
    md.append("")
    md.append("Attempts distribution (only counts episodes where the policy logged a result):")
    md.append("")
    md.append("| attempts | episodes |")
    md.append("|---:|---:|")
    for k in sorted(attempts_dist):
        md.append(f"| {k} | {attempts_dist[k]} |")
    md.append("")

    md.append("Per-rail success:")
    md.append("")
    md.append("| rail | n | inserted | rate |")
    md.append("|---:|---:|---:|---:|")
    for rail in sorted(rail_dist):
        outcomes = rail_dist[rail]
        ins_n = sum(1 for o in outcomes if o)
        total = len(outcomes)
        rate = f"{ins_n/total:.0%}" if total else "—"
        md.append(f"| {rail} | {total} | {ins_n} | {rate} |")
    md.append("")

    md.append("## Files")
    md.append("")
    md.append(f"- `samples.json` — all 20 sampled specs")
    md.append(f"- `configs/seed_NN.yaml` — templated engine YAMLs")
    md.append(f"- `runs/seed_NN/{{driver.log,output_dir.txt}}` — per-seed driver output")
    md.append(f"- `datasets/seed_NN/aic_recording_*` — recorded LeRobot datasets")
    md.append(f"- `summary.json` — per-seed exit code, elapsed, dataset path, policy result")
    md.append(f"- `spawn_verification.json` — bag-derived task_board pose vs YAML deltas")
    md.append(f"- `dataset_validation.json` — schema + sanity checks")
    md.append("")

    md.append("## Caveats")
    md.append("")
    md.append("- Spawn verification compares the **task_board world pose** "
              "(x,y,z,yaw) and the **chosen NIC rail's presence** (no other "
              "`nic_card_mount_<j>` should appear). It does NOT yet check "
              "in-rail translation or NIC yaw — would need to chain "
              "task_board → nic_card_mount_<i> from /scoring/tf and account "
              "for the rail anchor offset baked into the SDF.")
    md.append("- Dataset 'descent observed' is a 5 cm action.z range "
              "sanity check, not a strict monotonicity test (action.z "
              "covers approach-rise + descent + release-hold and is "
              "non-monotonic by design).")
    md.append("- Cable-grasp ranges (rows 9-11 in [[sfp-spawn-parameters]]) "
              "have no DR source — guessed ±2 mm. If episodes failed at "
              "high cable-z values, that's a hint the range is too wide.")

    (sweep_dir / "briefing.md").write_text("\n".join(md) + "\n")
    print(f"wrote {sweep_dir / 'briefing.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
