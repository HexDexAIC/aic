#!/usr/bin/env python3
"""Detect wrench spikes per trial and pull the matching center-camera frame.

For each trial in <bag_dir>:
  - locate force / torque peaks in /fts_broadcaster/wrench (residual above
    baseline, prominence-based; see PEAK_* constants)
  - find the closest /center_camera/image message
  - save a PNG annotated with bag / trial / spike-relative time, ||F||, ||tau||

Output: <bag_dir>/spikes/  with names sortable by trial+timestamp:
  spike_t<trial>_<TT.TTs>_<F|T>_F<XX.XX>N_T<X.XXX>Nm.png
where the third token marks which channel triggered the detection (F = force,
T = torque, B = both at the same instant).

Run:
    scripts/.venv/bin/python scripts/extract_spike_images.py <bag_dir> [<bag_dir2> ...]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks

GOAL_STATUS = {0: "UNKNOWN", 1: "ACCEPTED", 2: "EXECUTING", 3: "CANCELING",
               4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}

# Peak detection thresholds (residual above per-trial baseline).
PEAK_FORCE_PROMINENCE_N = 3.0
PEAK_TORQUE_PROMINENCE_NM = 0.5
PEAK_MIN_DISTANCE_S = 0.3
DEDUPE_WINDOW_S = 0.25  # peaks in F and tau within this window are merged


@dataclass
class Trial:
    idx: int
    start_t: float
    end_t: float
    end_state: str


@dataclass
class Spike:
    trial_idx: int
    t_bag: float       # bag-relative seconds
    t_trial: float     # trial-relative seconds
    log_time_ns: int   # original ns timestamp on the wrench message
    kind: str          # "F", "T", or "B"
    force_N: float     # ||F|| at this instant
    torque_Nm: float   # ||tau|| at this instant


def read_wrench_and_trials(mcap_path: Path):
    wrench_t, wrench_f, wrench_tau, wrench_ns = [], [], [], []
    seen_statuses: set[tuple[str, int]] = set()
    per_goal: dict[str, dict[str, float]] = {}
    image_ns: list[int] = []

    t0_ns = None
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=["/fts_broadcaster/wrench", "/insert_cable/_action/status",
                    "/center_camera/image"]
        ):
            if t0_ns is None:
                t0_ns = message.log_time
            t = (message.log_time - t0_ns) / 1e9
            if channel.topic == "/fts_broadcaster/wrench":
                w = ros_msg.wrench
                wrench_t.append(t)
                wrench_f.append((w.force.x, w.force.y, w.force.z))
                wrench_tau.append((w.torque.x, w.torque.y, w.torque.z))
                wrench_ns.append(message.log_time)
            elif channel.topic == "/insert_cable/_action/status":
                for s in ros_msg.status_list:
                    uuid = bytes(s.goal_info.goal_id.uuid).hex()
                    label = GOAL_STATUS.get(s.status, str(s.status))
                    key = (uuid, s.status)
                    if key in seen_statuses:
                        continue
                    seen_statuses.add(key)
                    per_goal.setdefault(uuid, {})[label] = t
            elif channel.topic == "/center_camera/image":
                image_ns.append(message.log_time)

    wt = np.array(wrench_t, dtype=float)
    wf = np.array(wrench_f, dtype=float) if wrench_f else np.zeros((0, 3))
    wtau = np.array(wrench_tau, dtype=float) if wrench_tau else np.zeros((0, 3))
    wns = np.array(wrench_ns, dtype=np.int64)
    img_ns = np.array(image_ns, dtype=np.int64)
    bag_end = float(wt[-1]) if len(wt) else 0.0

    goal_starts: list[tuple[float, str]] = []
    for uuid, states in per_goal.items():
        end_label = next((lbl for lbl in ("SUCCEEDED", "ABORTED", "CANCELED")
                          if lbl in states), "UNKNOWN")
        start_t = states.get("EXECUTING", states.get("ACCEPTED", 0.0))
        goal_starts.append((start_t, end_label))
    goal_starts.sort(key=lambda x: x[0])

    trials: list[Trial] = []
    for i, (start_t, end_label) in enumerate(goal_starts):
        split_start = 0.0 if i == 0 else start_t
        split_end = goal_starts[i + 1][0] if i + 1 < len(goal_starts) else bag_end
        trials.append(Trial(i + 1, split_start, split_end, end_label))

    return t0_ns, wt, wf, wtau, wns, img_ns, trials


def detect_spikes(wt, wf, wtau, wns, trials):
    spikes: list[Spike] = []
    for tr in trials:
        mask = (wt >= tr.start_t) & (wt <= tr.end_t)
        idxs = np.where(mask)[0]
        if len(idxs) < 20:
            continue
        t_local = wt[idxs] - tr.start_t
        f_mag = np.linalg.norm(wf[idxs], axis=1)
        tau_mag = np.linalg.norm(wtau[idxs], axis=1)
        baseline_F = float(f_mag[:20].mean())
        baseline_T = float(tau_mag[:20].mean())
        f_resid = f_mag - baseline_F
        t_resid = tau_mag - baseline_T

        dt = float(np.median(np.diff(t_local))) if len(t_local) > 1 else 1.0
        min_dist_samples = max(1, int(PEAK_MIN_DISTANCE_S / dt))

        f_peaks, _ = find_peaks(f_resid,
                                prominence=PEAK_FORCE_PROMINENCE_N,
                                distance=min_dist_samples)
        t_peaks, _ = find_peaks(t_resid,
                                prominence=PEAK_TORQUE_PROMINENCE_NM,
                                distance=min_dist_samples)

        # Merge: union by index, with kind tag
        events: dict[int, str] = {}
        for p in f_peaks:
            events[int(p)] = "F"
        for p in t_peaks:
            events[int(p)] = "B" if int(p) in events else "T"
        # Promote any F-only peak that has a T peak within DEDUPE_WINDOW_S to "B"
        if len(f_peaks) and len(t_peaks):
            t_peak_times = t_local[t_peaks]
            for p in f_peaks:
                if events[int(p)] != "F":
                    continue
                if np.any(np.abs(t_peak_times - t_local[p]) <= DEDUPE_WINDOW_S):
                    events[int(p)] = "B"
            f_peak_times = t_local[f_peaks]
            for p in t_peaks:
                if events[int(p)] != "T":
                    continue
                if np.any(np.abs(f_peak_times - t_local[p]) <= DEDUPE_WINDOW_S):
                    events[int(p)] = "B"

        # Sort by local time, dedupe near-duplicates within DEDUPE_WINDOW_S
        ordered = sorted(events.items(), key=lambda kv: t_local[kv[0]])
        kept: list[tuple[int, str]] = []
        last_time = -1e9
        for p, kind in ordered:
            tt = float(t_local[p])
            if tt - last_time < DEDUPE_WINDOW_S:
                # merge into previous peak; upgrade kind to "B" if different
                if kept and kept[-1][1] != kind:
                    kept[-1] = (kept[-1][0], "B")
                continue
            kept.append((p, kind))
            last_time = tt

        for local_idx, kind in kept:
            global_idx = idxs[local_idx]
            spikes.append(Spike(
                trial_idx=tr.idx,
                t_bag=float(wt[global_idx]),
                t_trial=float(t_local[local_idx]),
                log_time_ns=int(wns[global_idx]),
                kind=kind,
                force_N=float(f_mag[local_idx]),
                torque_Nm=float(tau_mag[local_idx]),
            ))
    return spikes


def fetch_image_at(mcap_path: Path, target_ns: int, search_window_ns: int = 200_000_000):
    """Return (image_ndarray, log_time_ns) of the nearest /center_camera/image."""
    best = None
    best_dt = None
    start = target_ns - search_window_ns
    end = target_ns + search_window_ns
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=["/center_camera/image"], start_time=start, end_time=end
        ):
            dt = abs(message.log_time - target_ns)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = (ros_msg, message.log_time)
    if best is None:
        return None, None
    msg, log_ns = best
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    arr = arr.reshape(msg.height, msg.step // (1 if "16" in msg.encoding else 1))
    # rgb8: width*3 bytes per row; reshape correctly
    channels = 3 if msg.encoding in ("rgb8", "bgr8") else 1
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return arr, log_ns


def annotate(arr: np.ndarray, lines: list[str]) -> Image.Image:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    pad = 10
    line_h = 28
    box_h = pad * 2 + line_h * len(lines)
    box_w = 0
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        box_w = max(box_w, bbox[2] - bbox[0])
    box_w += pad * 2
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 200))
    for i, ln in enumerate(lines):
        draw.text((pad, pad + i * line_h), ln, fill=(255, 255, 0), font=font)
    return img


def process_bag(bag_dir: Path):
    mcap_path = next(bag_dir.glob("*.mcap"))
    print(f"\n=== {bag_dir.name} ===")
    print(f"  mcap: {mcap_path.name}")
    t0_ns, wt, wf, wtau, wns, img_ns, trials = read_wrench_and_trials(mcap_path)
    print(f"  wrench samples: {len(wt)}, image frames: {len(img_ns)}, trials: {len(trials)}")
    spikes = detect_spikes(wt, wf, wtau, wns, trials)
    print(f"  detected spikes: {len(spikes)}")
    if not spikes:
        return

    out_dir = bag_dir / "spikes"
    out_dir.mkdir(exist_ok=True)
    bag_short = bag_dir.name.replace("wrench_calib_", "").replace("_2026-04-23", "")

    # Summary text file alongside images
    summary_path = out_dir / "spikes.txt"
    with open(summary_path, "w") as sf:
        sf.write(f"# Spikes for {bag_dir.name}\n")
        sf.write(f"# kind: F=force only, T=torque only, B=both within {DEDUPE_WINDOW_S}s\n")
        sf.write(f"# thresholds: |dF|>={PEAK_FORCE_PROMINENCE_N}N prominence, "
                 f"|dT|>={PEAK_TORQUE_PROMINENCE_NM}Nm prominence, "
                 f"min-distance {PEAK_MIN_DISTANCE_S}s\n\n")
        sf.write(f"{'idx':>3}  {'trial':>5}  {'t_bag(s)':>9}  {'t_trial(s)':>11}  "
                 f"{'kind':>4}  {'||F||(N)':>10}  {'||tau||(Nm)':>12}  filename\n")

        for i, sp in enumerate(spikes, 1):
            arr, img_log_ns = fetch_image_at(mcap_path, sp.log_time_ns)
            if arr is None:
                print(f"  spike #{i} t_trial={sp.t_trial:.2f}s — no image found")
                continue
            t_img_bag = (img_log_ns - t0_ns) / 1e9

            fname = (f"spike_t{sp.trial_idx}_{sp.t_trial:06.2f}s_{sp.kind}"
                     f"_F{sp.force_N:05.2f}N_T{sp.torque_Nm:05.3f}Nm.png")
            path = out_dir / fname

            lines = [
                f"{bag_short}",
                f"trial {sp.trial_idx}  spike {i}/{len(spikes)}  ({sp.kind})",
                f"t_bag = {sp.t_bag:.3f}s   t_trial = {sp.t_trial:.3f}s",
                f"||F|| = {sp.force_N:6.2f} N    ||tau|| = {sp.torque_Nm:6.3f} N.m",
                f"image t_bag = {t_img_bag:.3f}s   (dt = {(t_img_bag - sp.t_bag)*1000:+.0f}ms)",
            ]
            img = annotate(arr, lines)
            img.save(path)

            sf.write(f"{i:>3}  {sp.trial_idx:>5}  {sp.t_bag:>9.3f}  {sp.t_trial:>11.3f}  "
                     f"{sp.kind:>4}  {sp.force_N:>10.2f}  {sp.torque_Nm:>12.3f}  {fname}\n")

    print(f"  saved {len(spikes)} images + summary -> {out_dir}/")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        sys.exit(2)
    for arg in argv[1:]:
        bag_dir = Path(arg).expanduser().resolve()
        if not bag_dir.is_dir() or not list(bag_dir.glob("*.mcap")):
            print(f"Skipping (no .mcap): {bag_dir}")
            continue
        process_bag(bag_dir)


if __name__ == "__main__":
    main(sys.argv)
