#!/usr/bin/env python3
"""Generate a self-contained HTML report from analyzed wrench-calibration bags.

Reads each bag's spikes/spikes.txt and embeds the plot PNGs + spike GIFs by
relative path. The report file is written one level up so the relative paths
into <bag_dir>/... resolve when opened from disk.

Run:
    scripts/.venv/bin/python scripts/generate_spike_report.py \\
        <out_html> <bag_dir> [<bag_dir2> ...]
"""

from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Spike:
    idx: int
    trial: int
    t_bag: float
    t_trial: float
    kind: str
    force_N: float
    torque_Nm: float
    filename: str  # PNG-style; we substitute .gif for inline animation


def parse_spikes(path: Path) -> list[Spike]:
    spikes: list[Spike] = []
    if not path.exists():
        return spikes
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("idx"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 8:
            continue
        try:
            spikes.append(Spike(
                idx=int(parts[0]),
                trial=int(parts[1]),
                t_bag=float(parts[2]),
                t_trial=float(parts[3]),
                kind=parts[4],
                force_N=float(parts[5]),
                torque_Nm=float(parts[6]),
                filename=parts[7],
            ))
        except (ValueError, IndexError):
            continue
    return spikes


def kind_label(k: str) -> str:
    return {"F": "force", "T": "torque", "B": "both"}.get(k, k)


def per_trial_counts(spikes: list[Spike]) -> dict[int, int]:
    out: dict[int, int] = {}
    for sp in spikes:
        out[sp.trial] = out.get(sp.trial, 0) + 1
    return out


def render_bag_section(bag_dir: Path, out_root: Path) -> str:
    rel = bag_dir.relative_to(out_root)
    spikes = parse_spikes(bag_dir / "spikes" / "spikes.txt")
    short = bag_dir.name

    counts = per_trial_counts(spikes)
    counts_row = " ".join(
        f'<span class="pill">t{t}: {n}</span>' for t, n in sorted(counts.items())
    )

    # Plots (three: force, torque, all-6 combined)
    plots_html = []
    for fname, caption in [
        ("wrench_per_trial.png", "Linear force per trial — ‖F‖, Fx, Fy, Fz"),
        ("torque_per_trial.png", "Torque per trial — ‖τ‖, Tx, Ty, Tz"),
        ("wrench_all6_per_trial.png",
         "All 6 components on twin axes — solid = N, dashed = N·m"),
    ]:
        p = bag_dir / fname
        if not p.exists():
            continue
        plots_html.append(
            f'<figure class="plot">'
            f'<img src="{rel}/{fname}" loading="lazy" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)}</figcaption>'
            f'</figure>'
        )

    # Spike table
    rows = []
    for sp in spikes:
        cls = "row-both" if sp.kind == "B" else (
            "row-torque" if sp.kind == "T" else "row-force")
        gif = sp.filename.replace(".png", ".gif")
        rows.append(
            f'<tr class="{cls}">'
            f'<td class="num">{sp.idx}</td>'
            f'<td class="num">t{sp.trial}</td>'
            f'<td class="num">{sp.t_trial:.3f}</td>'
            f'<td class="num">{sp.t_bag:.3f}</td>'
            f'<td><span class="kind k-{sp.kind}">{sp.kind}</span></td>'
            f'<td class="num">{sp.force_N:.2f}</td>'
            f'<td class="num">{sp.torque_Nm:.3f}</td>'
            f'<td><a href="{rel}/spikes/{gif}">gif</a></td>'
            f'</tr>'
        )
    table_html = (
        '<table class="spikes">'
        '<thead><tr>'
        '<th>#</th><th>trial</th><th>t_trial (s)</th><th>t_bag (s)</th>'
        '<th>kind</th><th>‖F‖ (N)</th><th>‖τ‖ (N·m)</th><th></th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )

    # GIF gallery, grouped by trial
    by_trial: dict[int, list[Spike]] = {}
    for sp in spikes:
        by_trial.setdefault(sp.trial, []).append(sp)
    gallery_blocks = []
    for trial in sorted(by_trial):
        cards = []
        for sp in by_trial[trial]:
            gif = sp.filename.replace(".png", ".gif")
            cards.append(
                f'<figure class="card card-{sp.kind}" id="{short}-spike-{sp.idx}">'
                f'<img src="{rel}/spikes/{gif}" loading="lazy" '
                f'alt="trial {sp.trial} t={sp.t_trial:.2f}s">'
                f'<figcaption>'
                f'<div class="card-line"><b>t{sp.trial} #{sp.idx}</b> '
                f'<span class="kind k-{sp.kind}">{sp.kind}</span></div>'
                f'<div class="card-line">t<sub>trial</sub>={sp.t_trial:.3f}s</div>'
                f'<div class="card-line">‖F‖={sp.force_N:.2f} N · '
                f'‖τ‖={sp.torque_Nm:.3f} N·m</div>'
                f'</figcaption>'
                f'</figure>'
            )
        gallery_blocks.append(
            f'<div class="trial-block">'
            f'<h4>Trial {trial} <span class="muted">'
            f'({len(by_trial[trial])} spikes)</span></h4>'
            f'<div class="grid">{"".join(cards)}</div>'
            f'</div>'
        )

    return f"""
<section class="bag" id="{html.escape(short)}">
  <header class="bag-head">
    <h2>{html.escape(short)}</h2>
    <div class="counts">{counts_row} <span class="pill total">total: {len(spikes)}</span></div>
  </header>

  <h3>Per-trial wrench plots</h3>
  <div class="plot-row">{''.join(plots_html)}</div>

  <h3>Spikes</h3>
  {table_html}

  <h3>Spike GIFs (±1s window)</h3>
  {''.join(gallery_blocks)}
</section>
"""


CSS = """
:root {
  --bg: #fafaf7;
  --panel: #ffffff;
  --fg: #1a1a1a;
  --muted: #6b6b6b;
  --rule: #d9d4c8;
  --accent: #9c3d00;
  --force: #b91c1c;
  --torque: #1d4ed8;
  --both: #6d28d9;
  --code-bg: #f0ece4;
  --shadow: 0 1px 2px rgba(0,0,0,0.05);
}
* { box-sizing: border-box; }
html, body { background: var(--bg); color: var(--fg); margin: 0; }
body {
  font-family: 'Charter', 'Iowan Old Style', Georgia, serif;
  font-size: 16px;
  line-height: 1.55;
  max-width: 1280px;
  margin: 0 auto;
  padding: 32px 36px 80px;
}
h1 { font-size: 28px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 22px; margin: 0; letter-spacing: -0.01em; }
h3 { font-size: 16px; margin: 28px 0 10px; text-transform: uppercase;
     letter-spacing: 0.08em; color: var(--muted); border-bottom: 1px solid var(--rule);
     padding-bottom: 6px; }
h4 { font-size: 15px; margin: 18px 0 8px; }
p { margin: 0 0 12px; }
a { color: var(--accent); }
code, pre, .mono {
  font-family: 'JetBrains Mono', 'IBM Plex Mono', Menlo, monospace;
  font-size: 0.92em;
}
.muted { color: var(--muted); }
.report-meta { color: var(--muted); font-size: 14px; margin-top: 4px; }

.tldr {
  background: var(--panel);
  border-left: 3px solid var(--accent);
  padding: 14px 18px;
  margin: 24px 0;
  box-shadow: var(--shadow);
}
.tldr h3 { margin-top: 0; border-bottom: none; padding-bottom: 0; color: var(--accent); }
.tldr ul { margin: 6px 0 0 18px; padding: 0; }
.tldr li { margin: 4px 0; }

.bag {
  background: var(--panel);
  padding: 20px 22px 8px;
  margin: 28px 0;
  border: 1px solid var(--rule);
  border-radius: 4px;
  box-shadow: var(--shadow);
}
.bag-head {
  display: flex; justify-content: space-between; align-items: baseline;
  flex-wrap: wrap; gap: 12px;
  margin-bottom: 4px;
}
.counts { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
.pill {
  display: inline-block; padding: 2px 8px; margin-left: 4px;
  background: var(--code-bg); border-radius: 9999px; color: var(--fg);
}
.pill.total { background: var(--accent); color: white; }

.plot-row { display: grid; gap: 16px; }
@media (min-width: 1100px) { .plot-row { grid-template-columns: repeat(3, 1fr); } }
.plot { margin: 0; }
.plot img {
  width: 100%; height: auto; display: block;
  border: 1px solid var(--rule); background: white;
}
.plot figcaption { font-size: 12px; color: var(--muted); margin-top: 4px; text-align: center; }

table.spikes {
  border-collapse: collapse;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  width: 100%;
  margin-bottom: 12px;
}
table.spikes th, table.spikes td {
  padding: 4px 10px;
  border-bottom: 1px solid var(--rule);
  text-align: left;
}
table.spikes th { background: var(--code-bg); font-weight: 600; }
table.spikes td.num { text-align: right; }
.row-both { background: rgba(109, 40, 217, 0.06); }
.row-force { background: rgba(185, 28, 28, 0.04); }
.row-torque { background: rgba(29, 78, 216, 0.04); }

.kind {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 3px;
  color: white;
  font-weight: 600;
  font-size: 11px;
}
.k-F { background: var(--force); }
.k-T { background: var(--torque); }
.k-B { background: var(--both); }

.trial-block { margin-top: 14px; }
.grid {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
}
.card {
  margin: 0;
  background: var(--panel);
  border: 1px solid var(--rule);
  border-radius: 3px;
  overflow: hidden;
}
.card-F { border-left: 3px solid var(--force); }
.card-T { border-left: 3px solid var(--torque); }
.card-B { border-left: 3px solid var(--both); }
.card img { width: 100%; height: auto; display: block; }
.card figcaption {
  padding: 6px 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11.5px;
  background: var(--code-bg);
}
.card-line { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.findings table {
  border-collapse: collapse;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  margin: 8px 0 16px;
}
.findings th, .findings td {
  padding: 5px 12px;
  border-bottom: 1px solid var(--rule);
  text-align: left;
}
.findings th { background: var(--code-bg); }
.findings td.num { text-align: right; }
.flag { color: var(--accent); font-weight: 600; }
"""

INTRO = """
<header>
  <h1>Wrench spike analysis — 2026-04-23 calibration</h1>
  <p class="report-meta">
    Two CheatCodeMJ runs, 3 trials each: <code>baseline</code> (clean) and
    <code>bad_offset_2mm</code> (port shifted +2 mm in XY). Center-camera frames
    pulled at every detected wrench spike to compare what force/torque
    transients correspond to visually.
  </p>
</header>

<aside class="tldr">
  <h3>TL;DR</h3>
  <ul>
    <li>Linear force ‖F‖ catches the seat-in event on clean insertions
        (<b>baseline t2: 38 N transient</b>). It does <b>not</b> catch the
        2 mm rim graze.</li>
    <li>Torque ‖τ‖ catches the rim graze
        (<b>2mm-offset t2: 14 N·m transient</b>) with no corresponding force
        spike — opposite signal to baseline t2.</li>
    <li>Working hypothesis: an off-axis plug rim-contacting the port creates a
        moment arm without much pure axial force; ‖τ‖ may be a
        more sensitive contact-event channel than ‖F‖ for off-center misses.</li>
    <li>Detector caveat: a few <span class="kind k-F">F</span>-only spikes near
        end-of-trial are the wrench dropping back to ~0 N as the engine
        resets — false positives from the residual peak finder.</li>
  </ul>
</aside>

<section>
  <h3>Detection method</h3>
  <p>
    Per-trial baseline = mean ‖F‖ / ‖τ‖ over the first 20 wrench samples.
    Residuals (signal − baseline) fed into <code>scipy.signal.find_peaks</code>
    with prominence ≥ 3 N (force) and ≥ 0.5 N·m (torque), minimum 0.3 s
    between peaks. Force-only and torque-only peaks within 0.25 s of each
    other are merged into <span class="kind k-B">B</span> (both). For each
    spike, the closest <code>/center_camera/image</code> frame (always within
    ±20 ms) is annotated with bag/trial/timestamp and saved alongside a
    ±1 s GIF.
  </p>
  <p class="muted">
    Source: <code>scripts/extract_spike_images.py</code>,
    <code>scripts/extract_spike_gifs.py</code>,
    <code>scripts/plot_wrench_per_trial.py</code>.
  </p>
</section>
"""

FINDINGS = """
<section class="findings">
  <h3>Headline finding — F vs τ inversion on trial 2</h3>
  <table>
    <thead>
      <tr><th></th><th>baseline t2</th><th>2mm-offset t2</th><th>interpretation</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>max ‖F‖ (N)</td>
        <td class="num">38.09 <span class="flag">↑</span></td>
        <td class="num">21.55</td>
        <td>baseline registers the seat-in spike; offset run does not</td>
      </tr>
      <tr>
        <td>max ‖τ‖ (N·m)</td>
        <td class="num">2.09</td>
        <td class="num">14.11 <span class="flag">↑</span></td>
        <td>offset run registers ~7× larger torque transient than baseline</td>
      </tr>
      <tr>
        <td>p95 ‖τ‖ (N·m)</td>
        <td class="num">1.32</td>
        <td class="num">1.36</td>
        <td>p95 nearly identical → 14 N·m is a <em>transient</em>, not sustained</td>
      </tr>
    </tbody>
  </table>
  <p>
    Physically consistent with a plug rim-contacting an off-axis port:
    a moment arm, but little axial force. Compare
    <code>baseline/spikes/spike_t2_017.97s_B_F38.09N_T2.086Nm.gif</code>
    against
    <code>bad_offset_2mm/spikes/spike_t2_031.72s_T_F19.24N_T14.111Nm.gif</code>
    in the gallery below.
  </p>
</section>
"""


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        sys.exit(2)
    out_path = Path(argv[1]).expanduser().resolve()
    bag_dirs = [Path(a).expanduser().resolve() for a in argv[2:]]
    out_root = out_path.parent

    sections = [render_bag_section(bd, out_root) for bd in bag_dirs]

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wrench spike analysis — 2026-04-23</title>
  <style>{CSS}</style>
</head>
<body>
{INTRO}
{FINDINGS}
{''.join(sections)}
</body>
</html>
"""
    out_path.write_text(html_doc)
    print(f"Wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv)
