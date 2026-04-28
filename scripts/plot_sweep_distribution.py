#!/usr/bin/env python3
"""Plot the sample distribution from a spawn-sweep over its DR ranges.

For each of the 8 continuous knobs sampled by `spawn_sweep_sfp.py`, draws
a horizontal range bar with every seed's value as a tick, colored by
CheatCodeMJ insertion outcome (green=inserted, red=missed, gray=unknown).
The discrete `nic_rail` choice is rendered as a stacked bar chart.

Inputs read from <sweep_dir>:
    samples.json   — list of per-seed spawn specs
    summary.json   — per-seed run results (policy.inserted)

Output:
    <sweep_dir>/samples_distribution.png

Run with the side-venv (matplotlib lives there; pixi env doesn't have it):
    src/aic/scripts/.venv/bin/python \
        src/aic/scripts/plot_sweep_distribution.py <sweep_dir>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Must match CONTINUOUS in spawn_sweep_sfp.py. Kept duplicated rather
# than imported so this script stays runnable without the pixi env.
CONTINUOUS = [
    ("board_x",          0.150 - 0.005,    0.150 + 0.005,    "m"),
    ("board_y",         -0.200 - 0.005,   -0.200 + 0.005,    "m"),
    ("board_yaw",        3.1415 - 0.15,    3.1415 + 0.15,    "rad"),
    ("nic_translation", -0.0215,           0.0234,           "m"),
    ("nic_yaw",         -0.05,             0.05,             "rad"),
    ("grip_x",           0.0   - 0.002,    0.0   + 0.002,    "m"),
    ("grip_y",           0.015385 - 0.002, 0.015385 + 0.002, "m"),
    ("grip_z",           0.04045,          0.04545,          "m"),
]

NIC_RAIL_CHOICES = [0, 1, 2, 3, 4]


def outcome_color(inserted) -> str:
    if inserted is True:
        return "#2e7d32"   # green
    if inserted is False:
        return "#c62828"   # red
    return "#9e9e9e"        # gray (unknown / no policy log)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: plot_sweep_distribution.py <sweep_dir>", file=sys.stderr)
        return 2
    sweep = Path(sys.argv[1])
    samples = json.loads((sweep / "samples.json").read_text())
    summary = json.loads((sweep / "summary.json").read_text())
    by_seed = {r["seed"]: r for r in summary["results"]}

    # Build (seed, spec, outcome_color) tuples.
    rows = []
    for spec in samples:
        seed = spec["seed"]
        result = by_seed.get(seed, {})
        inserted = (result.get("policy") or {}).get("inserted")
        rows.append((seed, spec, inserted))

    n_inserted = sum(1 for _, _, ins in rows if ins is True)
    n_missed   = sum(1 for _, _, ins in rows if ins is False)
    n_unknown  = sum(1 for _, _, ins in rows if ins is None)

    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    axes = axes.flatten()

    # 8 continuous knobs.
    import random as _r
    rng = _r.Random(0)  # stable jitter
    for ax, (name, lo, hi, unit) in zip(axes[:8], CONTINUOUS):
        span = hi - lo
        # Range bar — fills most of the panel vertically.
        ax.add_patch(Rectangle(
            (lo, -0.6), span, 1.2,
            facecolor="#e3f2fd", edgecolor="#90caf9", linewidth=0.8,
        ))

        # Per-sample dots with small y-jitter for disambiguation.
        for seed, spec, inserted in rows:
            v = spec[name]
            y = rng.uniform(-0.45, 0.45)
            ax.plot(v, y, marker="o", markersize=5,
                    markerfacecolor=outcome_color(inserted),
                    markeredgecolor="white", markeredgewidth=0.6,
                    linestyle="none", alpha=0.9)

        scale = 1000 if unit == "m" else 1
        units_label = "mm" if unit == "m" else unit
        ax.set_xlim(lo - 0.08 * span, hi + 0.08 * span)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([])
        ax.set_xlabel(name, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        # Min/max bookends shown above the bar with units.
        ax.text(lo, 0.75, f"{lo*scale:+.2f}", fontsize=7, color="#1565c0",
                ha="center", va="bottom")
        ax.text(hi, 0.75, f"{hi*scale:+.2f}", fontsize=7, color="#1565c0",
                ha="center", va="bottom")
        ax.text((lo+hi)/2, -0.95, f"({units_label})", fontsize=7, color="#666",
                ha="center", va="top", style="italic")

    # 9th panel: discrete nic_rail bar chart, stacked by outcome.
    ax = axes[8]
    rail_outcomes = {r: {"ins": 0, "miss": 0, "unk": 0} for r in NIC_RAIL_CHOICES}
    for seed, spec, inserted in rows:
        r = spec["nic_rail"]
        key = "ins" if inserted is True else "miss" if inserted is False else "unk"
        rail_outcomes[r][key] += 1
    rails = NIC_RAIL_CHOICES
    ins  = [rail_outcomes[r]["ins"] for r in rails]
    mis  = [rail_outcomes[r]["miss"] for r in rails]
    unk  = [rail_outcomes[r]["unk"] for r in rails]
    ax.bar(rails, ins, color="#2e7d32", label="inserted")
    ax.bar(rails, mis, bottom=ins, color="#c62828", label="missed")
    ax.bar(rails, unk, bottom=[i+m for i, m in zip(ins, mis)],
           color="#9e9e9e", label="unknown")
    ax.set_xticks(rails)
    ax.set_xlabel("nic_rail (discrete: 0..4)", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Figure-level title with summary.
    success_pct = 100.0 * n_inserted / max(len(rows), 1)
    title = (
        f"Spawn-sweep sample distribution — {sweep.name}\n"
        f"n={len(rows)}   inserted={n_inserted}   missed={n_missed}   "
        f"unknown={n_unknown}   ({success_pct:.0f}% success)"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = sweep / "samples_distribution.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
