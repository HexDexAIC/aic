#!/usr/bin/env python3
"""Split /fts_broadcaster/wrench by trial and plot per-trial force traces.

Trial boundaries come from /insert_cable/_action/status (ACCEPTED / EXECUTING
/ SUCCEEDED / ABORTED transitions). Each trial gets its own subplot with
||F||, Fx, Fy, Fz over time.

A summary table is printed: max / p95 / mean / sustained-over-threshold
durations for each trial, useful for picking a contact-detection threshold.

Saves `<bag_dir>/wrench_per_trial.png`.

Run:
    scripts/.venv/bin/python scripts/plot_wrench_per_trial.py <bag_dir>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

GOAL_STATUS = {0: "UNKNOWN", 1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING",
               4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}
TRIAL_END_STATES = {"SUCCEEDED", "ABORTED", "CANCELED"}
THRESHOLDS_N = [5.0, 10.0, 15.0, 20.0]


@dataclass
class Trial:
    idx: int
    start_t: float   # bag-relative seconds when EXECUTING
    end_t: float     # bag-relative seconds when SUCCEEDED/ABORTED
    end_state: str


def read_bag(bag_dir: Path):
    mcap_path = next(bag_dir.glob("*.mcap"))
    wrench_t = []
    wrench_f = []   # (fx, fy, fz)
    wrench_tau = []
    seen_statuses: set[tuple[str, int]] = set()
    # goal_uuid -> {state_label: t}
    per_goal: dict[str, dict[str, float]] = {}

    t0_ns = None
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=["/fts_broadcaster/wrench", "/insert_cable/_action/status"]
        ):
            if t0_ns is None:
                t0_ns = message.log_time
            t = (message.log_time - t0_ns) / 1e9
            if channel.topic == "/fts_broadcaster/wrench":
                w = ros_msg.wrench
                wrench_t.append(t)
                wrench_f.append((w.force.x, w.force.y, w.force.z))
                wrench_tau.append((w.torque.x, w.torque.y, w.torque.z))
            elif channel.topic == "/insert_cable/_action/status":
                for s in ros_msg.status_list:
                    uuid = bytes(s.goal_info.goal_id.uuid).hex()
                    label = GOAL_STATUS.get(s.status, str(s.status))
                    key = (uuid, s.status)
                    if key in seen_statuses:
                        continue
                    seen_statuses.add(key)
                    per_goal.setdefault(uuid, {})[label] = t

    # Build one Trial per goal that reached a terminal state.
    # Semantics: trial N spans from its own "start" to trial N+1's "start"
    # (so engine setup / teardown windows are attributed to the adjacent trial).
    # Trial 1 starts at the bag start; the last trial ends at bag end.
    wt_all = np.array(wrench_t, dtype=float)
    bag_end = float(wt_all[-1]) if len(wt_all) else 0.0

    goal_starts: list[tuple[float, str]] = []  # (start_t, end_label)
    for uuid, states in per_goal.items():
        end_label = next((lbl for lbl in ("SUCCEEDED", "ABORTED", "CANCELED")
                          if lbl in states), "UNKNOWN")
        start_t = states.get("EXECUTING", states.get("ACCEPTED", 0.0))
        goal_starts.append((start_t, end_label))
    goal_starts.sort(key=lambda x: x[0])

    trials: list[Trial] = []
    for i, (start_t, end_label) in enumerate(goal_starts):
        if i == 0:
            split_start = 0.0
        else:
            split_start = start_t
        if i + 1 < len(goal_starts):
            split_end = goal_starts[i + 1][0]
        else:
            split_end = bag_end
        trials.append(Trial(i + 1, split_start, split_end, end_label))

    wf = np.array(wrench_f, dtype=float) if wrench_f else np.zeros((0, 3))
    wtau = np.array(wrench_tau, dtype=float) if wrench_tau else np.zeros((0, 3))
    return wt_all, wf, wtau, trials


def slice_trial(wt, wf, wtau, trial: Trial):
    mask = (wt >= trial.start_t) & (wt <= trial.end_t)
    return wt[mask] - trial.start_t, wf[mask], wtau[mask]


def summarize(t, f, tau, threshold_dts):
    """Return summary dict for one trial's wrench slice."""
    if len(t) == 0:
        return None
    mag = np.linalg.norm(f, axis=1)
    out = {
        "n_samples": len(t),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "max_N":  float(mag.max()),
        "p95_N":  float(np.quantile(mag, 0.95)),
        "p50_N":  float(np.quantile(mag, 0.50)),
        "mean_N": float(mag.mean()),
        "baseline_N": float(mag[:20].mean()) if len(mag) >= 20 else float(mag[0]),
    }
    # sustained-above-threshold durations
    if len(t) > 1:
        dt = np.diff(t, prepend=t[0])
    else:
        dt = np.array([0.0])
    for thr in THRESHOLDS_N:
        above = mag >= thr
        out[f"time_above_{int(thr)}N_s"] = float(dt[above].sum())
    return out, mag


def print_summary_table(summaries):
    if not summaries:
        print("No trials found.")
        return
    headers = ["trial", "dur", "baseline", "max", "p95", "mean"] + \
              [f">{int(t)}N(s)" for t in THRESHOLDS_N]
    fmt = "{:>6}  {:>6}  {:>9}  {:>7}  {:>7}  {:>7}" + "  {:>8}" * len(THRESHOLDS_N)
    print("\n" + fmt.format(*headers))
    print("-" * (sum(len(h) + 2 for h in headers) + 6))
    for s in summaries:
        print(fmt.format(
            f"{s['label']}",
            f"{s['duration_s']:.1f}",
            f"{s['baseline_N']:.2f}",
            f"{s['max_N']:.2f}",
            f"{s['p95_N']:.2f}",
            f"{s['mean_N']:.2f}",
            *[f"{s[f'time_above_{int(th)}N_s']:.2f}" for th in THRESHOLDS_N],
        ))


def plot_per_trial(trials_data, out_path: Path):
    n = len(trials_data)
    if n == 0:
        print("Nothing to plot.")
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, td in zip(axes, trials_data):
        t, f, mag, label = td["t"], td["f"], td["mag"], td["label"]
        ax.plot(t, mag, color="black", lw=1.2, label="||F||")
        ax.plot(t, f[:, 0], color="tab:red",   lw=0.6, alpha=0.7, label="Fx")
        ax.plot(t, f[:, 1], color="tab:green", lw=0.6, alpha=0.7, label="Fy")
        ax.plot(t, f[:, 2], color="tab:blue",  lw=0.6, alpha=0.7, label="Fz")
        for thr, color in [(10.0, "#aa7722"), (20.0, "#aa2222")]:
            ax.axhline(thr, color=color, ls="--", lw=0.6, alpha=0.5)
            ax.text(t[-1] * 0.995, thr, f"{int(thr)}N", va="bottom",
                    ha="right", fontsize=7, color=color)
        ax.set_title(f"{label}  (dur {t[-1]:.1f}s, max ||F|| {mag.max():.2f} N)")
        ax.set_xlabel("t from trial start (s)")
        ax.set_ylabel("N")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nPlot saved: {out_path}")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(2)
    bag_dir = Path(argv[1]).expanduser().resolve()
    if not bag_dir.is_dir() or not list(bag_dir.glob("*.mcap")):
        print(f"Not a bag dir (no .mcap): {bag_dir}")
        sys.exit(2)

    print(f"Reading {bag_dir} ...")
    wt, wf, wtau, trials = read_bag(bag_dir)
    print(f"  /fts_broadcaster/wrench: {len(wt)} samples")
    print(f"  trials from /insert_cable/_action/status: {len(trials)}")
    for tr in trials:
        print(f"    trial {tr.idx}: {tr.start_t:.1f} -> {tr.end_t:.1f} s  ({tr.end_state})")

    trials_data = []
    summaries = []
    for tr in trials:
        t, f, tau = slice_trial(wt, wf, wtau, tr)
        res = summarize(t, f, tau, THRESHOLDS_N)
        if res is None:
            continue
        s, mag = res
        label = f"trial {tr.idx} ({tr.end_state})"
        s["label"] = f"t{tr.idx}"
        summaries.append(s)
        trials_data.append({"t": t, "f": f, "mag": mag, "label": label})

    print_summary_table(summaries)
    plot_per_trial(trials_data, bag_dir / "wrench_per_trial.png")


if __name__ == "__main__":
    main(sys.argv)
