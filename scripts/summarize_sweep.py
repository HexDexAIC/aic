#!/usr/bin/env python3
"""Aggregate scoring.yaml files across a sweep directory.

Usage:
    scripts/summarize_sweep.py SWEEP_DIR

Emits markdown to stdout and writes SWEEP_DIR/summary.csv alongside.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def find_key(d: Any, keys: tuple[str, ...]) -> Any:
    """DFS for the first matching key in nested dicts."""
    if isinstance(d, dict):
        for k in keys:
            if k in d:
                return d[k]
        for v in d.values():
            r = find_key(v, keys)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = find_key(v, keys)
            if r is not None:
                return r
    return None


def extract_run(scoring_path: Path) -> dict:
    """Parse a single scoring.yaml and pull out the salient numbers."""
    with scoring_path.open() as f:
        data = yaml.safe_load(f)

    total = find_key(data, ("total_score", "total", "overall_score"))

    # The engine writes trial_1, trial_2, trial_3 as top-level keys, each
    # containing tier_1 / tier_2 / tier_3. No "trials" wrapper key.
    per_trial: dict[str, float] = {}
    per_trial_tiers: dict[str, dict[str, float]] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if not (isinstance(key, str) and key.startswith("trial_") and isinstance(value, dict)):
                continue
            tiers = {}
            subtotal = 0.0
            for tkey in ("tier_1", "tier_2", "tier_3"):
                tscore = value.get(tkey, {}).get("score") if isinstance(value.get(tkey), dict) else None
                if isinstance(tscore, (int, float)):
                    tiers[tkey] = float(tscore)
                    subtotal += float(tscore)
            per_trial[key] = subtotal if tiers else None
            per_trial_tiers[key] = tiers

    return {
        "total": total,
        "per_trial": per_trial,
        "per_trial_tiers": per_trial_tiers,
        "raw": data,
    }


def format_num(x: Any, width: int = 7) -> str:
    if x is None:
        return "-".rjust(width)
    if isinstance(x, (int, float)) and not math.isnan(x):
        return f"{x:>{width}.2f}"
    return str(x).rjust(width)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    sweep_dir = Path(sys.argv[1]).resolve()
    if not sweep_dir.is_dir():
        print(f"Not a directory: {sweep_dir}", file=sys.stderr)
        return 2

    run_dirs = sorted(p for p in sweep_dir.iterdir() if p.is_dir() and p.name.startswith("run_"))
    if not run_dirs:
        print(f"No run_* subdirectories in {sweep_dir}", file=sys.stderr)
        return 1

    rows = []
    all_trial_names: list[str] = []
    for rd in run_dirs:
        yml = rd / "scoring.yaml"
        if not yml.exists():
            rows.append({
                "run": rd.name,
                "total": None,
                "per_trial": {},
                "status": "NO_YAML",
            })
            continue
        try:
            info = extract_run(yml)
            for t in info["per_trial"]:
                if t not in all_trial_names:
                    all_trial_names.append(t)
            rows.append({
                "run": rd.name,
                "total": info["total"],
                "per_trial": info["per_trial"],
                "per_trial_tiers": info["per_trial_tiers"],
                "status": "OK" if info["total"] is not None else "PARSE_PARTIAL",
            })
        except Exception as ex:
            rows.append({
                "run": rd.name,
                "total": None,
                "per_trial": {},
                "per_trial_tiers": {},
                "status": f"PARSE_ERR: {ex}",
            })

    # CSV
    csv_path = sweep_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["run", "status", "total"] + all_trial_names
        w.writerow(header)
        for r in rows:
            row = [r["run"], r["status"], r["total"]]
            for t in all_trial_names:
                row.append(r["per_trial"].get(t))
            w.writerow(row)

    # Markdown
    md: list[str] = []
    md.append(f"# Sweep summary — {sweep_dir.name}")
    md.append("")
    md.append(f"- Runs: {len(rows)}")
    ok_rows = [r for r in rows if r["total"] is not None]
    md.append(f"- Completed with score: {len(ok_rows)}")
    md.append(f"- Failed (no/invalid scoring.yaml): {len(rows) - len(ok_rows)}")
    md.append("")

    if ok_rows:
        totals = [r["total"] for r in ok_rows]
        md.append("## Totals")
        md.append("")
        md.append(f"- mean: **{statistics.mean(totals):.2f}**")
        md.append(f"- stdev: **{statistics.pstdev(totals):.2f}**" if len(totals) > 1 else "- stdev: n/a (1 run)")
        md.append(f"- min:  **{min(totals):.2f}**")
        md.append(f"- max:  **{max(totals):.2f}**")
        md.append(f"- range: **{max(totals) - min(totals):.2f}**")
        md.append("")

        # Per-trial stats
        if all_trial_names:
            md.append("## Per-trial stats (trial total = T1 + T2 + T3)")
            md.append("")
            md.append("| trial | mean | stdev | min | max |")
            md.append("|---|---:|---:|---:|---:|")
            for t in all_trial_names:
                vals = [r["per_trial"].get(t) for r in ok_rows]
                vals = [v for v in vals if isinstance(v, (int, float))]
                if not vals:
                    continue
                md.append(
                    f"| {t} | {statistics.mean(vals):.2f} | "
                    f"{(statistics.pstdev(vals) if len(vals) > 1 else 0):.2f} | "
                    f"{min(vals):.2f} | {max(vals):.2f} |"
                )
            md.append("")

            # Tier-3 (insertion) success breakdown: most strategy-relevant.
            md.append("## Tier 3 (insertion outcome) per trial")
            md.append("")
            md.append("| trial | full (75) | partial/other | rate |")
            md.append("|---|---:|---:|---:|")
            for t in all_trial_names:
                t3_vals = [r["per_trial_tiers"].get(t, {}).get("tier_3") for r in ok_rows]
                t3_vals = [v for v in t3_vals if isinstance(v, (int, float))]
                if not t3_vals:
                    continue
                full = sum(1 for v in t3_vals if v >= 74.9)
                other = len(t3_vals) - full
                rate = f"{full}/{len(t3_vals)}"
                md.append(f"| {t} | {full} | {other} | {rate} |")
            md.append("")

    # Per-run table
    md.append("## Per-run")
    md.append("")
    hdr = ["run", "status", "total"] + all_trial_names
    md.append("| " + " | ".join(hdr) + " |")
    md.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        cells = [r["run"], r["status"], format_num(r["total"])]
        for t in all_trial_names:
            cells.append(format_num(r["per_trial"].get(t)))
        md.append("| " + " | ".join(str(c).strip() for c in cells) + " |")
    md.append("")

    md.append(f"CSV written to `{csv_path}`")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
