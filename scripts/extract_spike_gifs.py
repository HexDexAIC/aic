#!/usr/bin/env python3
"""Make a ±1s center-camera GIF for every wrench spike detected in a bag.

Reuses the spike detector from extract_spike_images. For each spike, pulls the
/center_camera/image frames within [t_spike - WINDOW_S, t_spike + WINDOW_S],
annotates each frame with bag / trial / spike index, frame time relative to
the spike, and the interpolated ||F||, ||tau|| at that frame, then saves an
animated GIF in <bag_dir>/spikes/.

Run:
    scripts/.venv/bin/python scripts/extract_spike_gifs.py <bag_dir> [<bag_dir2> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_spike_images import (  # noqa: E402
    detect_spikes,
    read_wrench_and_trials,
)

WINDOW_S = 1.0          # ±this many seconds around each spike
RESIZE_WIDTH = 576      # downscale frames before encoding to keep gifs small
GIF_FPS = 12            # roughly matches the simulator camera rate


def interp_at(t_query: float, t_arr: np.ndarray, mag: np.ndarray) -> float:
    if len(t_arr) == 0:
        return 0.0
    return float(np.interp(t_query, t_arr, mag))


def annotate_frame(arr: np.ndarray, header_lines: list[str],
                   footer_lines: list[str], is_spike_frame: bool) -> Image.Image:
    img = Image.fromarray(arr).convert("RGB")
    if RESIZE_WIDTH and img.width > RESIZE_WIDTH:
        new_h = int(img.height * RESIZE_WIDTH / img.width)
        img = img.resize((RESIZE_WIDTH, new_h), Image.BILINEAR)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    pad, line_h = 6, 18

    # Header (top-left): static spike info
    box_w = 0
    for ln in header_lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        box_w = max(box_w, bbox[2] - bbox[0])
    box_w += pad * 2
    box_h = pad * 2 + line_h * len(header_lines)
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0))
    for i, ln in enumerate(header_lines):
        draw.text((pad, pad + i * line_h), ln, fill=(255, 255, 0), font=font)

    # Footer (bottom-left): per-frame info
    box_w = 0
    for ln in footer_lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        box_w = max(box_w, bbox[2] - bbox[0])
    box_w += pad * 2
    fbox_h = pad * 2 + line_h * len(footer_lines)
    draw.rectangle([0, img.height - fbox_h, box_w, img.height], fill=(0, 0, 0))
    color = (255, 64, 64) if is_spike_frame else (255, 255, 0)
    for i, ln in enumerate(footer_lines):
        draw.text((pad, img.height - fbox_h + pad + i * line_h),
                  ln, fill=color, font=font)

    if is_spike_frame:
        # Red border on the spike frame itself
        draw.rectangle([0, 0, img.width - 1, img.height - 1],
                       outline=(255, 0, 0), width=4)

    return img


def make_gif_for_spike(mcap_path: Path, t0_ns: int, spike, wt: np.ndarray,
                       f_mag: np.ndarray, tau_mag: np.ndarray, bag_short: str,
                       n_total: int, out_path: Path) -> int:
    """Return number of frames written (0 if no images in window)."""
    start_ns = spike.log_time_ns - int(WINDOW_S * 1e9)
    end_ns = spike.log_time_ns + int(WINDOW_S * 1e9)

    frames: list[Image.Image] = []
    closest_dt = None
    closest_idx = None

    raw_msgs: list[tuple[int, object]] = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=["/center_camera/image"], start_time=start_ns, end_time=end_ns
        ):
            raw_msgs.append((message.log_time, ros_msg))

    if not raw_msgs:
        return 0

    # Determine which message is closest to the spike time → mark as spike frame
    for i, (log_ns, _) in enumerate(raw_msgs):
        dt = abs(log_ns - spike.log_time_ns)
        if closest_dt is None or dt < closest_dt:
            closest_dt = dt
            closest_idx = i

    header = [
        f"{bag_short}",
        f"trial {spike.trial_idx}  spike {spike.idx}/{n_total}  ({spike.kind})",
        f"PEAK  t_trial={spike.t_trial:.3f}s  t_bag={spike.t_bag:.3f}s",
        f"PEAK  ||F||={spike.force_N:.2f}N  ||tau||={spike.torque_Nm:.3f}Nm",
    ]

    for i, (log_ns, msg) in enumerate(raw_msgs):
        channels = 3 if msg.encoding in ("rgb8", "bgr8") else 1
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, channels)
        if msg.encoding == "bgr8":
            arr = arr[:, :, ::-1]
        if channels == 1:
            arr = np.repeat(arr, 3, axis=2)

        t_frame_bag = (log_ns - t0_ns) / 1e9
        t_rel = t_frame_bag - spike.t_bag
        f_at = interp_at(t_frame_bag, wt, f_mag)
        tau_at = interp_at(t_frame_bag, wt, tau_mag)
        footer = [
            f"frame {i+1}/{len(raw_msgs)}   dt={t_rel:+.3f}s",
            f"||F||={f_at:6.2f}N   ||tau||={tau_at:6.3f}Nm",
        ]
        frames.append(annotate_frame(arr, header, footer,
                                     is_spike_frame=(i == closest_idx)))

    # Convert to palette-mode for compact GIF.
    pal_frames = [fr.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
                  for fr in frames]
    duration_ms = int(1000 / GIF_FPS)
    pal_frames[0].save(out_path, save_all=True, append_images=pal_frames[1:],
                       duration=duration_ms, loop=0, optimize=True, disposal=2)
    return len(frames)


def process_bag(bag_dir: Path):
    mcap_path = next(bag_dir.glob("*.mcap"))
    print(f"\n=== {bag_dir.name} ===")
    t0_ns, wt, wf, wtau, wns, img_ns, trials = read_wrench_and_trials(mcap_path)
    spikes = detect_spikes(wt, wf, wtau, wns, trials)
    print(f"  {len(spikes)} spikes; image frames in bag: {len(img_ns)}")
    if not spikes:
        return

    # Attach an idx attribute for header text
    for i, sp in enumerate(spikes, 1):
        sp.idx = i

    f_mag = np.linalg.norm(wf, axis=1)
    tau_mag = np.linalg.norm(wtau, axis=1)
    out_dir = bag_dir / "spikes"
    out_dir.mkdir(exist_ok=True)
    bag_short = bag_dir.name.replace("wrench_calib_", "").replace("_2026-04-23", "")

    for sp in spikes:
        gif_name = (f"spike_t{sp.trial_idx}_{sp.t_trial:06.2f}s_{sp.kind}"
                    f"_F{sp.force_N:05.2f}N_T{sp.torque_Nm:05.3f}Nm.gif")
        out_path = out_dir / gif_name
        n_frames = make_gif_for_spike(mcap_path, t0_ns, sp, wt, f_mag, tau_mag,
                                      bag_short, len(spikes), out_path)
        print(f"  spike #{sp.idx}: trial {sp.trial_idx}  t_trial={sp.t_trial:6.2f}s  "
              f"kind={sp.kind}  frames={n_frames}  -> {gif_name}")

    print(f"  -> {out_dir}/  ({len(spikes)} GIFs)")


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
