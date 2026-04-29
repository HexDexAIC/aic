#!/usr/bin/env python3
"""Calibration click UI — user clicks 4 corners of the visible SFP slot on
6 carefully-selected frames. Server fits T_canonical_to_visible_mouth and
reports the transform + residual error.

Workflow:
  1. Server picks 6 calibration frames (1 per rail × 1 distance, spanning
     all 5 rails + 1 extra), all at ≈18 cm where the slot is most visible.
  2. User opens http://localhost:8002 in browser.
  3. Per frame: clicks corner 1 (+X+Y), corner 2 (+X-Y), corner 3 (-X-Y),
     corner 4 (-X+Y). Numbered prompts on screen. Can re-click to overwrite
     a corner. Can navigate between frames freely.
  4. When user clicks "Solve calibration", server fits 6-DoF transform
     and writes ~/aic_visible_mouth_calib.json + a verification overlay.

Run: pixi run python scripts/calib_click_server.py
Then open: http://localhost:8002
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from project_gt_port_2d import (
    K_PER_CAM, T_TCP_OPT, state_to_T, quat_to_R,
)

ROOT = Path.home() / "aic_hexdex_sfp300"
GT_POSE_PATH = Path.home() / "aic_gt_port_poses.json"
OFFSET_PATH = Path.home() / "aic_logs" / "tcp_to_plug_offset.json"
CALIB_OUT = Path.home() / "aic_visible_mouth_calib.json"
CLICKS_OUT = Path.home() / "aic_calib_clicks.json"

# 20 calibration frames — diverse rails × 3 distance bands.
# Original 6 (18cm) at indices 0-5: clicks already saved in ~/aic_calib_clicks.json
# and will be restored on server start.
CALIB_FRAMESPEC = [
    # 18cm — slot is sharpest at this distance (12 frames)
    (3,   0.18), (41,  0.18), (78,  0.18), (122, 0.18),    # ORIGINAL 4 train
    (235, 0.18), (272, 0.18),                              # ORIGINAL 2 test
    (17,  0.18), (60,  0.18), (96,  0.18), (138, 0.18),    # NEW 4 train (more rails)
    (175, 0.18), (197, 0.18),                              # NEW 2 val
    # 14cm — medium distance (4 frames)
    (41,  0.14), (96,  0.14), (175, 0.14), (272, 0.14),
    # 10cm — closer, more depth-sensitive (4 frames)
    (3,   0.10), (78,  0.10), (138, 0.10), (235, 0.10),
]

SLOT_W, SLOT_H = 0.0137, 0.0085

# Loaded at startup
CALIB_FRAMES = []
T_TCP_PLUG = None
LOCK = threading.Lock()


def load_calib_frames():
    """Pre-load calibration frames + metadata at startup. Restores prior clicks
    from ~/aic_calib_clicks.json if available so the user keeps progress.
    """
    global T_TCP_PLUG
    gt_pose = json.loads(GT_POSE_PATH.read_text())
    offset = json.loads(OFFSET_PATH.read_text())["sfp"]
    T_TCP_PLUG = np.eye(4)
    T_TCP_PLUG[:3, :3] = quat_to_R(offset["qw"], offset["qx"], offset["qy"], offset["qz"])
    T_TCP_PLUG[:3, 3] = [offset["x"], offset["y"], offset["z"]]

    # Restore prior clicks
    prior_clicks = {}  # (ep, fr) -> [4 clicks]
    if CLICKS_OUT.exists():
        try:
            for cd in json.loads(CLICKS_OUT.read_text()):
                prior_clicks[(cd["ep"], cd["fr"])] = cd["clicks"]
        except Exception:
            pass

    # Find each ep in parquet files
    needed = set(s[0] for s in CALIB_FRAMESPEC)
    ep_to_pq = {}
    for pf in sorted((ROOT / "data" / "chunk-000").glob("*.parquet")):
        tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "observation.state"])
        df = tbl.to_pandas()
        for ep_val in df["episode_index"].unique():
            ep_int = int(ep_val)
            if ep_int in needed and ep_int not in ep_to_pq:
                file_idx = int(pf.stem.replace("file-", ""))
                eg = df[df["episode_index"] == ep_int].sort_values("frame_index").reset_index(drop=True)
                ep_to_pq[ep_int] = (file_idx, eg)

    out = []
    for ep, target_d in CALIB_FRAMESPEC:
        if ep not in ep_to_pq or str(ep) not in gt_pose:
            print(f"WARN: ep{ep} unavailable")
            continue
        file_idx, eg = ep_to_pq[ep]
        states = np.stack(eg["observation.state"].values)
        frames = eg["frame_index"].to_numpy()

        T_settled = np.eye(4)
        T_settled[:3, :3] = np.array(gt_pose[str(ep)]["actual_tcp_R"])
        T_settled[:3, 3] = gt_pose[str(ep)]["actual_tcp_xyz"]
        T_base_port = T_settled @ T_TCP_PLUG
        port_xyz = T_base_port[:3, 3]
        dists = np.linalg.norm(states[:, 0:3] - port_xyz, axis=1)
        valid = dists >= 0.06
        err = np.where(valid, np.abs(dists - target_d), np.inf)
        idx = int(np.argmin(err))
        actual_d = float(dists[idx])
        fr_idx = int(frames[idx])

        # Find row in parquet
        tbl = pq.read_table(ROOT / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet",
                             columns=["episode_index", "frame_index"])
        df_full = tbl.to_pandas()
        row_in_file = df_full[(df_full["episode_index"] == ep) &
                                (df_full["frame_index"] == fr_idx)].index[0]

        cap = cv2.VideoCapture(str(ROOT / "videos" / "observation.images.center" / "chunk-000" / f"file-{file_idx:03d}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row_in_file))
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue

        T_base_tcp = state_to_T(states[idx])
        K = K_PER_CAM["center"]
        T_tcp_opt = T_TCP_OPT["center"]

        # Compute canonical projection (used as click prior + zoom anchor)
        from project_gt_port_2d import state_to_T as s2T
        corners = np.array([
            [+SLOT_W/2, +SLOT_H/2, 0, 1],
            [+SLOT_W/2, -SLOT_H/2, 0, 1],
            [-SLOT_W/2, -SLOT_H/2, 0, 1],
            [-SLOT_W/2, +SLOT_H/2, 0, 1],
            [0, 0, 0, 1],
        ]).T
        T_opt_port = np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T_base_port
        pts3 = T_opt_port @ corners
        Z = pts3[2]
        u = K[0, 0] * pts3[0] / Z + K[0, 2]
        v = K[1, 1] * pts3[1] / Z + K[1, 2]
        canon_proj = np.stack([u, v], axis=1)

        # Restore any previously-saved clicks for this (ep, fr) pair
        prior = prior_clicks.get((ep, fr_idx), [None, None, None, None])

        out.append({
            "ep": ep, "fr": fr_idx, "actual_d": actual_d,
            "img": img,
            "T_base_port": T_base_port,
            "T_base_tcp": T_base_tcp,
            "K": K, "T_tcp_opt": T_tcp_opt,
            "canon_proj": canon_proj,
            "clicks": list(prior),   # 4 corners (image space) — preserved if present
        })
    return out


def save_clicks_now():
    """Persist all current clicks (including partial frames) to disk."""
    dump = []
    for f in CALIB_FRAMES:
        dump.append({"ep": f["ep"], "fr": f["fr"], "clicks": list(f["clicks"])})
    CLICKS_OUT.write_text(json.dumps(dump, indent=2))


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>SFP mouth calibration</title>
<style>
  body { background:#222; color:#eee; font-family:Helvetica,Arial,sans-serif; margin:0; padding:10px; }
  h2 { margin: 0 0 8px 0; }
  #wrap { display:flex; gap:14px; }
  #imgwrap { position:relative; background:#111; border:2px solid #444; }
  #imgwrap img { display:block; max-width:780px; cursor:crosshair; }
  #side { width:340px; }
  button { font-size:14px; padding:6px 12px; margin:2px; }
  .corner { position:absolute; width:14px; height:14px; border-radius:50%;
            border:2px solid white; transform:translate(-50%,-50%); pointer-events:none; }
  .c0 { background:#f00; }
  .c1 { background:#0f0; }
  .c2 { background:#00f; }
  .c3 { background:#ff0; }
  .canon { position:absolute; border:2px dashed #0ff; pointer-events:none; }
  pre { background:#111; color:#cfc; padding:6px; max-height:240px; overflow:auto; font-size:11px; }
  .prompt { font-size:18px; margin:8px 0; padding:6px; background:#333; }
  .corner-label { color:#888; font-size:12px; margin-left:8px; }
</style></head>
<body>
<h2>SFP visible-mouth calibration — click 4 corners per frame</h2>
<div class="prompt" id="prompt">Loading...</div>
<div>Click order: <span style="color:#f00">①+X+Y</span> →
  <span style="color:#0f0">②+X-Y</span> →
  <span style="color:#00f">③-X-Y</span> →
  <span style="color:#ff0">④-X+Y</span>
  <span class="corner-label">(matches port-frame canonical order)</span></div>
<div style="margin:6px 0">
  <button onclick="prevFrame()">← Prev</button>
  <button onclick="nextFrame()">Next →</button>
  <button onclick="resetFrame()">Reset clicks (this frame)</button>
  <button onclick="undoClick()">Undo last click</button>
  <button onclick="solve()" style="background:#063;color:#fff">Solve calibration</button>
  <span id="frameinfo" style="margin-left:14px"></span>
</div>
<div id="wrap">
  <div id="imgwrap">
    <img id="img" />
    <div class="canon" id="canonbox"></div>
    <div class="corner c0" id="c0" style="display:none"></div>
    <div class="corner c1" id="c1" style="display:none"></div>
    <div class="corner c2" id="c2" style="display:none"></div>
    <div class="corner c3" id="c3" style="display:none"></div>
  </div>
  <div id="side">
    <div><b>Tip:</b> the dashed cyan rectangle shows the current canonical projection.
      Click where the actual visible slot corner is (it may differ from canonical).</div>
    <pre id="info"></pre>
    <pre id="result"></pre>
  </div>
</div>

<script>
let curFrame = 0;
let totalFrames = 0;
let imgNatural = [0, 0];
let imgDisp = [0, 0];

const colorNames = ["+X+Y", "+X-Y", "-X-Y", "-X+Y"];

function updatePrompt() {
  const r = fetch(`/state?fr=${curFrame}`).then(r => r.json()).then(d => {
    totalFrames = d.total;
    document.getElementById('prompt').textContent =
      `Frame ${curFrame+1}/${totalFrames}: ep${d.ep} fr${d.fr} d=${d.dist.toFixed(1)}cm — ` +
      `next click: corner #${d.next+1} (${colorNames[d.next]}) — ` +
      `${d.n_clicked}/4 clicked on this frame`;
    document.getElementById('frameinfo').textContent =
      `frame ${curFrame+1}/${totalFrames}, total clicks: ${d.total_clicked}/${4*totalFrames}`;
    document.getElementById('info').textContent = JSON.stringify(d.clicks, null, 1);

    // Position canonical box (cx,cy in image px)
    const canon = d.canon;
    const sx = imgDisp[0] / imgNatural[0]; const sy = imgDisp[1] / imgNatural[1];
    const xs = canon.map(c => c[0]*sx); const ys = canon.map(c => c[1]*sy);
    const x0 = Math.min(...xs); const y0 = Math.min(...ys);
    const x1 = Math.max(...xs); const y1 = Math.max(...ys);
    const cb = document.getElementById('canonbox');
    cb.style.left = x0 + 'px';
    cb.style.top = y0 + 'px';
    cb.style.width = (x1-x0) + 'px';
    cb.style.height = (y1-y0) + 'px';

    // Position corner dots
    for (let i = 0; i < 4; i++) {
      const dot = document.getElementById('c'+i);
      const click = d.clicks[i];
      if (click) {
        dot.style.display = 'block';
        dot.style.left = (click[0]*sx) + 'px';
        dot.style.top = (click[1]*sy) + 'px';
      } else {
        dot.style.display = 'none';
      }
    }
  });
}

function loadImg() {
  const img = document.getElementById('img');
  img.onload = () => {
    imgNatural = [img.naturalWidth, img.naturalHeight];
    const rect = img.getBoundingClientRect();
    imgDisp = [rect.width, rect.height];
    updatePrompt();
  };
  img.src = `/img?fr=${curFrame}&t=${Date.now()}`;
}

document.addEventListener('DOMContentLoaded', () => {
  const img = document.getElementById('img');
  img.onclick = (e) => {
    const rect = img.getBoundingClientRect();
    const sx = imgNatural[0] / rect.width; const sy = imgNatural[1] / rect.height;
    const px = (e.clientX - rect.left) * sx;
    const py = (e.clientY - rect.top) * sy;
    fetch(`/click?fr=${curFrame}&x=${px}&y=${py}`, {method: 'POST'})
      .then(() => updatePrompt());
  };
  loadImg();
});

function nextFrame() {
  if (curFrame < totalFrames - 1) { curFrame++; loadImg(); }
}
function prevFrame() {
  if (curFrame > 0) { curFrame--; loadImg(); }
}
function resetFrame() {
  fetch(`/reset?fr=${curFrame}`, {method: 'POST'}).then(() => updatePrompt());
}
function undoClick() {
  fetch(`/undo?fr=${curFrame}`, {method: 'POST'}).then(() => updatePrompt());
}
function solve() {
  fetch('/solve', {method: 'POST'})
    .then(r => r.json())
    .then(d => {
      document.getElementById('result').textContent = JSON.stringify(d, null, 2);
    });
}
window.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') prevFrame();
  if (e.key === 'ArrowRight') nextFrame();
  if (e.key === 'r') resetFrame();
  if (e.key === 'z') undoClick();
});
</script>
</body></html>
"""


def project_with_T(T_base_port, T_co_mouth, T_base_tcp, K, T_tcp_opt, w=SLOT_W, h=SLOT_H):
    T_base_mouth = T_base_port @ T_co_mouth
    corners = np.array([
        [+w/2, +h/2, 0, 1], [+w/2, -h/2, 0, 1],
        [-w/2, -h/2, 0, 1], [-w/2, +h/2, 0, 1],
    ]).T
    T_opt_mouth = np.linalg.inv(T_base_tcp @ T_tcp_opt) @ T_base_mouth
    pts = T_opt_mouth @ corners
    Z = pts[2]
    if (Z <= 0).any():
        return None
    return np.stack([K[0, 0] * pts[0] / Z + K[0, 2],
                     K[1, 1] * pts[1] / Z + K[1, 2]], axis=1)


def solve_calibration():
    """Find T_canonical_to_mouth that minimizes pixel error to user clicks
    across all calibration frames. Searches over (dx, dy, dz, w_scale, h_scale)
    where w/h scales adjust the rectangle dimensions, and (dx,dy,dz) is a
    translation in canonical port frame.
    """
    from scipy.optimize import minimize

    # Gather all complete (4-clicks) frames
    pairs = []
    for f in CALIB_FRAMES:
        if all(c is not None for c in f["clicks"]):
            user_corners = np.array(f["clicks"])  # (4, 2) image px
            pairs.append({
                "T_base_port": f["T_base_port"],
                "T_base_tcp": f["T_base_tcp"],
                "K": f["K"], "T_tcp_opt": f["T_tcp_opt"],
                "user_corners": user_corners,
            })
    if len(pairs) < 3:
        return {"error": f"need ≥3 fully-clicked frames, have {len(pairs)}"}

    def loss(params):
        dx, dy, dz, w_scale, h_scale = params
        T_co = np.eye(4); T_co[:3, 3] = [dx, dy, dz]
        w = SLOT_W * w_scale; h = SLOT_H * h_scale
        total = 0.0
        for p in pairs:
            proj = project_with_T(p["T_base_port"], T_co, p["T_base_tcp"],
                                    p["K"], p["T_tcp_opt"], w=w, h=h)
            if proj is None:
                return 1e6
            err = (proj - p["user_corners"]).reshape(-1)
            total += float(np.sum(err ** 2))
        return total

    x0 = np.array([0, 0, 0, 1.0, 1.0])
    res = minimize(loss, x0, method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-3, "maxiter": 5000})
    dx, dy, dz, ws, hs = res.x

    # Compute residuals per frame
    T_co = np.eye(4); T_co[:3, 3] = [dx, dy, dz]
    final_w = SLOT_W * ws; final_h = SLOT_H * hs
    per_frame_err = []
    for p in pairs:
        proj = project_with_T(p["T_base_port"], T_co, p["T_base_tcp"],
                                p["K"], p["T_tcp_opt"], w=final_w, h=final_h)
        if proj is not None:
            err = np.linalg.norm(proj - p["user_corners"], axis=1)
            per_frame_err.append({
                "ep": next(f["ep"] for f in CALIB_FRAMES
                            if np.array_equal(f["T_base_port"], p["T_base_port"])),
                "median_corner_err": float(np.median(err)),
                "max_corner_err": float(err.max()),
            })

    result = {
        "n_frames": len(pairs),
        "T_canonical_to_visible_mouth": {
            "dx_mm": float(dx * 1000),
            "dy_mm": float(dy * 1000),
            "dz_mm": float(dz * 1000),
        },
        "rectangle": {
            "width_mm": float(final_w * 1000),
            "height_mm": float(final_h * 1000),
            "w_scale_vs_spec": float(ws),
            "h_scale_vs_spec": float(hs),
        },
        "per_frame_residual_px": per_frame_err,
        "median_overall_corner_err_px": float(np.median(
            [e["median_corner_err"] for e in per_frame_err])),
        "optimizer_status": str(res.message),
        "optimizer_iters": int(res.nit),
    }
    CALIB_OUT.write_text(json.dumps(result, indent=2))
    # Also save raw clicks
    clicks_dump = [{"ep": f["ep"], "fr": f["fr"], "clicks": f["clicks"]}
                    for f in CALIB_FRAMES if all(c is not None for c in f["clicks"])]
    CLICKS_OUT.write_text(json.dumps(clicks_dump, indent=2))
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            data = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif url.path == "/img":
            fr = int(qs.get("fr", ["0"])[0])
            if 0 <= fr < len(CALIB_FRAMES):
                ok, jpgb = cv2.imencode(".jpg", CALIB_FRAMES[fr]["img"],
                                          [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    body = jpgb.tobytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self.send_response(404); self.end_headers()
        elif url.path == "/state":
            fr = int(qs.get("fr", ["0"])[0])
            f = CALIB_FRAMES[fr]
            n_clicked = sum(1 for c in f["clicks"] if c is not None)
            next_idx = next((i for i, c in enumerate(f["clicks"]) if c is None), 4)
            total_clicked = sum(sum(1 for c in fr2["clicks"] if c is not None)
                                  for fr2 in CALIB_FRAMES)
            self._json(200, {
                "ep": f["ep"], "fr": f["fr"], "dist": f["actual_d"] * 100,
                "total": len(CALIB_FRAMES),
                "clicks": f["clicks"],
                "n_clicked": n_clicked,
                "next": next_idx,
                "total_clicked": total_clicked,
                "canon": f["canon_proj"][:4].tolist(),
            })
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path == "/click":
            fr = int(qs.get("fr", ["0"])[0])
            x = float(qs.get("x", ["0"])[0])
            y = float(qs.get("y", ["0"])[0])
            with LOCK:
                f = CALIB_FRAMES[fr]
                next_idx = next((i for i, c in enumerate(f["clicks"]) if c is None), None)
                if next_idx is not None:
                    f["clicks"][next_idx] = [x, y]
                save_clicks_now()
            self._json(200, {"ok": True})
        elif url.path == "/reset":
            fr = int(qs.get("fr", ["0"])[0])
            with LOCK:
                CALIB_FRAMES[fr]["clicks"] = [None, None, None, None]
                save_clicks_now()
            self._json(200, {"ok": True})
        elif url.path == "/undo":
            fr = int(qs.get("fr", ["0"])[0])
            with LOCK:
                clicks = CALIB_FRAMES[fr]["clicks"]
                last_set = max((i for i, c in enumerate(clicks) if c is not None),
                                default=-1)
                if last_set >= 0:
                    clicks[last_set] = None
                save_clicks_now()
            self._json(200, {"ok": True})
        elif url.path == "/solve":
            with LOCK:
                result = solve_calibration()
            self._json(200, result)
        else:
            self.send_response(404); self.end_headers()


def main():
    global CALIB_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()
    print(f"Loading {len(CALIB_FRAMESPEC)} calibration frames...")
    CALIB_FRAMES = load_calib_frames()
    n_clicked = sum(sum(1 for c in f["clicks"] if c is not None) for f in CALIB_FRAMES)
    print(f"  Loaded {len(CALIB_FRAMES)} frames; restored {n_clicked} prior clicks")
    print(f"  Open http://localhost:{args.port}/")
    print(f"  Calibration result will be saved to: {CALIB_OUT}")
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
