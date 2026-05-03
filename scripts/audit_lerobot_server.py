#!/usr/bin/env python3
"""LeRobot-dataset audit viewer.

Reads the HexDexAIC/aic-sfp-300 LeRobotDataset from disk, extracts frames
from the 3 wrist cameras, runs the current YOLO + classical detector on
each, and serves a web UI where the user can:

  - Browse per-episode, per-frame
  - See detector predictions overlaid on all 3 camera views
  - Draw correction boxes (target / distractor / other)
  - Save annotations to ~/aic_audit_annotations.json

Episodes 200-299 are reserved as the EVAL split — viewer marks them
visually but they're equally browsable.

Usage:
    pixi run python scripts/audit_lerobot_server.py [--port 8001]
        [--dataset /home/dell/aic_hexdex_sfp300]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "aic_example_policies"))

from aic_example_policies.ros.port_detector import detect_port as detect_classical
from aic_example_policies.ros.port_detector_yolo import YoloPosePortDetector


YOLO = None
DATASET_DIR: Path = None
ANNOTATIONS_FILE = Path.home() / "aic_audit_annotations.json"
GT_PORT_2D_FILE = Path.home() / "aic_gt_port_2d.json"
GT_PORT_2D = None    # {ep_K -> fr_K -> {cam -> [[u,v]*4 or None]}}
EPISODES_INFO = []   # list of dicts: {episode_idx, frames, length, video_paths}
LOCK = threading.Lock()
EVAL_START = 200     # episodes [EVAL_START, total) reserved as eval

# Cached at startup, looked up O(1) per request.
# (ep_idx, frame_idx) -> (file_idx, frame_in_file)
GLOBAL_FRAME_INDEX = {}
VIDEO_CAPS = {}              # (cam, file_idx) -> opened cv2.VideoCapture
VIDEO_CAP_LOCK = threading.Lock()
DET_CACHE = {}               # (ep_idx, frame_idx, cam) -> (annotated_jpg, det_info)
DET_CACHE_MAX = 96
DET_CACHE_LOCK = threading.Lock()
ANNOTATIONS_CACHE = None
ANNOTATIONS_CACHE_LOCK = threading.Lock()


def get_video_cap(cam, file_idx):
    """Return a per-(cam, file_idx) VideoCapture; reused across requests."""
    chunk_idx = 0
    video_path = DATASET_DIR / "videos" / f"observation.images.{cam}" / \
                 f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
    if not video_path.exists():
        return None
    key = (cam, file_idx)
    with VIDEO_CAP_LOCK:
        if key not in VIDEO_CAPS:
            VIDEO_CAPS[key] = cv2.VideoCapture(str(video_path))
        return VIDEO_CAPS[key]


def cached_load_annotations():
    global ANNOTATIONS_CACHE
    with ANNOTATIONS_CACHE_LOCK:
        if ANNOTATIONS_CACHE is None:
            if ANNOTATIONS_FILE.exists():
                try:
                    ANNOTATIONS_CACHE = json.loads(ANNOTATIONS_FILE.read_text())
                except Exception:
                    ANNOTATIONS_CACHE = {}
            else:
                ANNOTATIONS_CACHE = {}
        return ANNOTATIONS_CACHE


def invalidate_annotations_cache():
    global ANNOTATIONS_CACHE
    with ANNOTATIONS_CACHE_LOCK:
        ANNOTATIONS_CACHE = None


def get_gt_corners(ep, fr, cam):
    """Lookup projected GT port corners (4 [u,v]) for this (ep,fr,cam).
    Returns None if not in GT (failed episode, off-screen, behind cam).
    """
    if GT_PORT_2D is None:
        return None
    ep_d = GT_PORT_2D.get(f"ep_{ep}")
    if not ep_d:
        return None
    fr_d = ep_d.get(f"fr_{fr}")
    if not fr_d:
        return None
    return fr_d.get(cam)


def cache_get_or_compute(ep, fr, cam, port_type, frame_bgr, user_boxes):
    gt_corners = get_gt_corners(ep, fr, cam)
    gt_key = tuple(map(tuple, gt_corners)) if gt_corners else None
    key = (ep, fr, cam, gt_key,
           tuple((b.get("label"), tuple(b["bbox_xyxy"])) for b in user_boxes))
    with DET_CACHE_LOCK:
        if key in DET_CACHE:
            return DET_CACHE[key]
    annotated, info = annotate_frame(frame_bgr, port_type, user_boxes,
                                      gt_corners=gt_corners)
    ok, jpgb = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpg = jpgb.tobytes() if ok else b""
    val = (jpg, info)
    with DET_CACHE_LOCK:
        if len(DET_CACHE) >= DET_CACHE_MAX:
            DET_CACHE.pop(next(iter(DET_CACHE)))
        DET_CACHE[key] = val
    return val


def discover_episodes():
    """One-time scan: reads all parquet files, builds:
       - per-episode length list
       - GLOBAL_FRAME_INDEX[(ep_idx, frame_idx)] -> (file_idx, frame_in_file)
    """
    import pyarrow.parquet as pq
    import re
    info_path = DATASET_DIR / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    total_episodes = info["total_episodes"]
    fps = info["fps"]

    data_dir = DATASET_DIR / "data" / "chunk-000"
    parquets = sorted(data_dir.glob("*.parquet"))
    print(f"  Indexing {len(parquets)} parquet files...")
    ep_lengths = {}
    for pq_path in parquets:
        # Extract file_idx from filename like "file-005.parquet"
        m = re.match(r"file-(\d+)", pq_path.stem)
        if not m:
            continue
        file_idx = int(m.group(1))
        tbl = pq.read_table(pq_path, columns=["episode_index", "frame_index"])
        ep_arr = tbl["episode_index"].to_numpy()
        fr_arr = tbl["frame_index"].to_numpy()
        for i in range(len(ep_arr)):
            ep = int(ep_arr[i]); fr = int(fr_arr[i])
            # i is the row offset within THIS parquet, which == frame offset within
            # the corresponding mp4 file (since one file contains contiguous frames
            # per the LeRobot v3 layout).
            GLOBAL_FRAME_INDEX[(ep, fr)] = (file_idx, i)
            ep_lengths[ep] = max(ep_lengths.get(ep, 0), fr + 1)

    out = []
    for ep_idx in range(total_episodes):
        out.append({
            "episode_idx": ep_idx,
            "length": ep_lengths.get(ep_idx, 0),
            "is_eval": ep_idx >= EVAL_START,
        })
    print(f"  Indexed {sum(ep_lengths.values())} total frames across {len(parquets)} files.")
    return out, fps


def get_frame(ep_idx: int, frame_idx: int, camera: str):
    """Decode a single frame using the cached index. O(1)."""
    entry = GLOBAL_FRAME_INDEX.get((ep_idx, frame_idx))
    if entry is None:
        return None
    file_idx, frame_in_file = entry
    cap = get_video_cap(camera, file_idx)
    if cap is None:
        return None
    with VIDEO_CAP_LOCK:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_file)
        ok, frame_bgr = cap.read()
    return frame_bgr if ok else None


def annotate_frame(img_bgr, port_type: str, user_boxes=None, gt_corners=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = img_bgr.copy()
    info = {"yolo": None, "classical": None, "user": user_boxes or [], "gt": gt_corners}

    # Draw GT-projected port FIRST (yellow polygon) so other overlays sit on top.
    if gt_corners is not None:
        gt_pts = np.array(gt_corners, dtype=np.int32)
        cv2.polylines(out, [gt_pts], True, (0, 255, 255), 2)
        u_min = int(min(p[0] for p in gt_corners))
        v_min = int(min(p[1] for p in gt_corners))
        cv2.putText(out, "GT", (u_min, max(20, v_min - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    yolo_det = YOLO.detect(img_rgb, port_type) if (YOLO and YOLO.available) else None
    if yolo_det is not None:
        if yolo_det.corners_xy is not None:
            box = yolo_det.corners_xy.astype(np.int32)
            cv2.polylines(out, [box], True, (0, 255, 0), 2)
        cv2.putText(out, f"YOLO {yolo_det.score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        info["yolo"] = {"cx": yolo_det.cx, "cy": yolo_det.cy,
                        "w": yolo_det.width, "h": yolo_det.height}

    cls_det = detect_classical(img_rgb, port_type, refine=True)
    if cls_det is not None:
        if cls_det.corners_xy is not None:
            box = cls_det.corners_xy.astype(np.int32)
            cv2.polylines(out, [box], True, (0, 0, 255), 2)
        cv2.putText(out, "classical", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        info["classical"] = {"cx": cls_det.cx, "cy": cls_det.cy,
                             "w": cls_det.width, "h": cls_det.height}

    for b in user_boxes or []:
        color = {"target": (0, 200, 255),
                 "distractor": (255, 0, 200),
                 "other": (180, 180, 180)}.get(b.get("label"), (255, 200, 0))
        x0, y0, x1, y1 = [int(round(v)) for v in b["bbox_xyxy"]]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3)
        cv2.putText(out, b.get("label", "?"), (x0, max(20, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out, info


def load_annotations():
    return cached_load_annotations()


def save_annotations(d):
    ANNOTATIONS_FILE.write_text(json.dumps(d, indent=2))
    invalidate_annotations_cache()


def annotation_key(ep_idx, frame_idx, camera):
    return f"ep{ep_idx:03d}_fr{frame_idx:05d}_{camera}"


HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>AIC Audit</title>
<style>
  body { background:#222; color:#eee; font-family:Helvetica,Arial,sans-serif; margin:0; padding:10px; }
  h2 { margin: 0 0 8px 0; }
  .row { display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap; }
  .panel { background:#111; padding:6px; border:1px solid #333; }
  .cam { position:relative; user-select:none; }
  .cam img { display:block; max-width:520px; cursor:crosshair; }
  .cam .draw { position:absolute; border:3px dashed #0ff; pointer-events:none; }
  .cam h4 { margin: 0 0 4px 0; font-size:14px; }
  #side { width:340px; }
  button, select, input { font-size:14px; padding:5px 8px; margin:2px; }
  input[type=text]{ background:#111; color:#eee; border:1px solid #444; padding:4px 6px; }
  pre { background:#111; color:#cfc; padding:6px; max-height:240px; overflow:auto; font-size:11px; }
  .legend span { padding:2px 8px; margin-right:6px; border-radius:3px; font-size:12px; }
  .eval { background:#603; padding:2px 8px; border-radius:3px; font-size:12px; color:#fff; }
  .train { background:#063; padding:2px 8px; border-radius:3px; font-size:12px; color:#fff; }
  .hint { color:#aaa; font-size:12px; margin-top:6px; }
</style></head>
<body>
<h2>AIC LeRobot dataset auditor</h2>
<div class="legend">
  <span style="background:#ff0;color:#000">GT (projected)</span>
  <span style="background:#0c0;color:#000">YOLO</span>
  <span style="background:#f00;color:#fff">classical</span>
  <span style="background:#fa0;color:#000">target (user)</span>
  <span style="background:#a0f;color:#fff">distractor (user)</span>
  <span style="background:#888;color:#000">other</span>
</div>

<div class="row" style="margin-top:10px">
  <div class="panel" style="width:100%">
    <span>Episode:</span> <input id="ep" type="text" size="4" value="0"/>
    <span>Frame:</span> <input id="fr" type="text" size="5" value="0"/>
    <span id="eplabel" class="train">train</span>
    <button onclick="prevEp()">&laquo;Ep</button>
    <button onclick="nextEp()">Ep&raquo;</button>
    <button onclick="prevFr()">&laquo;Fr</button>
    <button onclick="nextFr()">Fr&raquo;</button>
    <button onclick="loadFrame()">Load</button>
    <span id="counter"></span>
    <div class="hint">Eval split = episodes 200-299. Use shortcuts: arrows = frame nav, J/K = episode nav.</div>
  </div>
</div>

<div class="row" style="margin-top:10px" id="cams">
  <div class="cam panel"><h4 id="h_left">left</h4><img id="img_left" /><div class="draw" id="draw_left" style="display:none"></div></div>
  <div class="cam panel"><h4 id="h_center">center</h4><img id="img_center" /><div class="draw" id="draw_center" style="display:none"></div></div>
  <div class="cam panel"><h4 id="h_right">right</h4><img id="img_right" /><div class="draw" id="draw_right" style="display:none"></div></div>
</div>

<div class="row" style="margin-top:10px">
  <div class="panel" id="side">
    <div>
      <span>Active cam:</span>
      <select id="activecam">
        <option>left</option><option selected>center</option><option>right</option>
      </select>
      <span>Label:</span>
      <select id="label">
        <option>target</option><option>distractor</option><option>other</option>
      </select>
      <button onclick="addBox()">Add box (a)</button>
    </div>
    <div>Comment: <input id="comment" type="text" size="34"/></div>
    <div><b>Boxes (this frame):</b></div>
    <div id="boxlist" style="max-height:160px;overflow:auto;background:#111;padding:6px"></div>
    <button onclick="clearAll()">Clear all this frame</button>
    <hr>
    <pre id="info"></pre>
  </div>
</div>

<script>
let curEp = 0, curFr = 0, curEpLen = 0, totalEps = 0;
let drawing = null;          // currently-being-drawn camera ('left'|'center'|'right')
let lastDrawnCam = 'center'; // camera of most-recently drawn box (used by Add)
let drawStart = null, drawCur = null;
let imgNatural = {}, imgDisp = {};

const cams = ['left', 'center', 'right'];
const activecam = document.getElementById('activecam');
const labelsel = document.getElementById('label');
const eplabel = document.getElementById('eplabel');
const counter = document.getElementById('counter');
const info = document.getElementById('info');
const boxlist = document.getElementById('boxlist');
const comment = document.getElementById('comment');

function loadFrame() {
  curEp = parseInt(document.getElementById('ep').value, 10);
  curFr = parseInt(document.getElementById('fr').value, 10);
  // Optimistic UI: clear boxes immediately, dim images while reloading.
  boxlist.innerHTML = '<i>loading…</i>';
  info.textContent = '';
  for (const c of cams) {
    const img = document.getElementById('img_'+c);
    const drawn = document.getElementById('draw_'+c);
    img.style.opacity = 0.35;
    drawn.style.display = 'none';
    img.onload = () => { img.style.opacity = 1.0; };
    img.src = `/frame?ep=${curEp}&fr=${curFr}&cam=${c}&t=${Date.now()}`;
    img.draggable = false;
    bindCam(c);
  }
  drawStart = null; drawCur = null;
  refreshInfo();
}

async function refreshInfo() {
  const r = await fetch(`/info?ep=${curEp}&fr=${curFr}`);
  const d = await r.json();
  curEpLen = d.length;
  totalEps = d.total_episodes;
  counter.textContent = ` (ep length=${d.length}, eval_start=${d.eval_start}, total_eps=${d.total_episodes})`;
  eplabel.textContent = d.is_eval ? 'EVAL' : 'train';
  eplabel.className = d.is_eval ? 'eval' : 'train';
  info.textContent = `last-clicked cam: ${lastDrawnCam}\nlabel-on-add: ${labelsel.value}`;
  const boxes = d.boxes_per_cam || {};
  let html = '';
  for (const cam of cams) {
    const cb = boxes[cam] || [];
    if (cb.length === 0) continue;
    html += `<div style="margin-bottom:4px"><b>${cam}:</b> ${cb.map((b,i)=>`<span style="background:#222;padding:2px 6px;margin-right:4px">${b.label} <button onclick="delBox('${cam}',${i})" style="padding:1px 5px">x</button></span>`).join('')}</div>`;
  }
  boxlist.innerHTML = html || '<i>(no annotations on this frame)</i>';
}

function bindCam(c) {
  const img = document.getElementById('img_'+c);
  const drawn = document.getElementById('draw_'+c);
  img.onmousedown = e => {
    drawing = c;
    lastDrawnCam = c;
    activecam.value = c;
    const rect = img.getBoundingClientRect();
    drawStart = [e.clientX - rect.left, e.clientY - rect.top];
    drawCur = [drawStart[0], drawStart[1]];
    imgNatural[c] = [img.naturalWidth, img.naturalHeight];
    imgDisp[c] = [rect.width, rect.height];
    // Hide overlays on the OTHER cameras (drawing is per-cam now).
    for (const cc of cams) {
      if (cc !== c) document.getElementById('draw_'+cc).style.display = 'none';
    }
    drawn.style.display = 'block';
    drawn.style.left = drawStart[0]+'px'; drawn.style.top = drawStart[1]+'px';
    drawn.style.width = '1px'; drawn.style.height = '1px';
    drawn.style.borderColor = '#0ff';
    e.preventDefault();
  };
  img.title = `Click and drag on this ${c} camera to draw the port box`;
  // Highlight active cam header.
  for (const cc of cams) {
    const h = document.getElementById('h_'+cc);
    if (cc === lastDrawnCam) {
      h.style.color = '#0ff';
      h.textContent = cc + ' (active)';
    } else {
      h.style.color = '#eee';
      h.textContent = cc;
    }
  }
}
window.addEventListener('mousemove', e => {
  if (!drawing) return;
  const img = document.getElementById('img_'+drawing);
  const drawn = document.getElementById('draw_'+drawing);
  const rect = img.getBoundingClientRect();
  drawCur = [e.clientX - rect.left, e.clientY - rect.top];
  const x = Math.min(drawStart[0], drawCur[0]);
  const y = Math.min(drawStart[1], drawCur[1]);
  const w = Math.max(1, Math.abs(drawStart[0] - drawCur[0]));
  const h = Math.max(1, Math.abs(drawStart[1] - drawCur[1]));
  drawn.style.left = x+'px'; drawn.style.top = y+'px';
  drawn.style.width = w+'px'; drawn.style.height = h+'px';
});
window.addEventListener('mouseup', e => { drawing = null; });

async function addBox() {
  if (!drawStart || !drawCur) { alert('Click and drag on a camera image to draw a box first'); return; }
  // Use lastDrawnCam (the cam where the click happened), NOT the dropdown.
  const c = lastDrawnCam;
  if (!imgNatural[c]) { alert('Click on a camera image first'); return; }
  const sx = imgNatural[c][0] / imgDisp[c][0];
  const sy = imgNatural[c][1] / imgDisp[c][1];
  const x0 = Math.min(drawStart[0], drawCur[0]) * sx;
  const y0 = Math.min(drawStart[1], drawCur[1]) * sy;
  const x1 = Math.max(drawStart[0], drawCur[0]) * sx;
  const y1 = Math.max(drawStart[1], drawCur[1]) * sy;
  await fetch(`/addbox?ep=${curEp}&fr=${curFr}&cam=${c}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({bbox: [x0, y0, x1, y1], label: labelsel.value, comment: comment.value})
  });
  drawStart = null; drawCur = null;
  document.getElementById('draw_'+c).style.display='none';
  comment.value = '';
  // Refresh ONLY the cam we annotated, not all 3 → no flash, no full re-fetch.
  const img = document.getElementById('img_'+c);
  img.style.opacity = 0.35;
  img.onload = () => { img.style.opacity = 1.0; };
  img.src = `/frame?ep=${curEp}&fr=${curFr}&cam=${c}&t=${Date.now()}`;
  refreshInfo();
}
async function delBox(cam, i) {
  await fetch(`/delbox?ep=${curEp}&fr=${curFr}&cam=${cam}&i=${i}`, {method:'POST'});
  loadFrame();
}
window.delBox = delBox;
async function clearAll() {
  if (!confirm('Delete ALL boxes on this frame across all cameras?')) return;
  await fetch(`/clearall?ep=${curEp}&fr=${curFr}`, {method:'POST'});
  loadFrame();
}
function prevEp() { curEp = Math.max(0, curEp - 1); curFr = 0; document.getElementById('ep').value = curEp; document.getElementById('fr').value = 0; loadFrame(); }
function nextEp() { curEp = curEp + 1; curFr = 0; document.getElementById('ep').value = curEp; document.getElementById('fr').value = 0; loadFrame(); }
function prevFr() { curFr = Math.max(0, curFr - 1); document.getElementById('fr').value = curFr; loadFrame(); }
function nextFr() { curFr = Math.min(curEpLen-1, curFr + 1); document.getElementById('fr').value = curFr; loadFrame(); }
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') prevFr();
  if (e.key === 'ArrowRight') nextFr();
  if (e.key === 'j') prevEp();
  if (e.key === 'k') nextEp();
  if (e.key === 'a') addBox();
  if (e.key === '1') labelsel.value = 'target';
  if (e.key === '2') labelsel.value = 'distractor';
  if (e.key === '3') labelsel.value = 'other';
});
loadFrame();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
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
        if url.path == "/" or url.path == "/index.html":
            data = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif url.path == "/frame":
            ep = int(qs.get("ep", ["0"])[0])
            fr = int(qs.get("fr", ["0"])[0])
            cam = qs.get("cam", ["center"])[0]
            frame_bgr = get_frame(ep, fr, cam)
            if frame_bgr is None:
                placeholder = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(placeholder, "frame not available",
                            (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ok, jpgb = cv2.imencode(".jpg", placeholder)
                jpg = jpgb.tobytes()
            else:
                ann = load_annotations().get(annotation_key(ep, fr, cam), {})
                user_boxes = ann.get("boxes", [])
                jpg, _ = cache_get_or_compute(ep, fr, cam, "sfp", frame_bgr, user_boxes)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg)
        elif url.path == "/info":
            ep = int(qs.get("ep", ["0"])[0])
            fr = int(qs.get("fr", ["0"])[0])
            ep_info = next((e for e in EPISODES_INFO if e["episode_idx"] == ep), None)
            length = ep_info["length"] if ep_info else 0
            is_eval = ep_info["is_eval"] if ep_info else False
            saved = load_annotations()
            boxes_per_cam = {}
            for cam in ("left", "center", "right"):
                k = annotation_key(ep, fr, cam)
                if k in saved:
                    boxes_per_cam[cam] = saved[k].get("boxes", [])
            # Don't re-run detectors here; the /frame endpoint already did
            # via the cache. Cheap O(1) reply.
            self._json(200, {
                "episode": ep, "frame": fr, "length": length,
                "is_eval": is_eval, "total_episodes": len(EPISODES_INFO),
                "eval_start": EVAL_START,
                "boxes_per_cam": boxes_per_cam, "det_summary": {},
            })
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path == "/addbox":
            ep = int(qs.get("ep", ["0"])[0])
            fr = int(qs.get("fr", ["0"])[0])
            cam = qs.get("cam", ["center"])[0]
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            with LOCK:
                d = load_annotations()
                k = annotation_key(ep, fr, cam)
                rec = d.get(k) or {"episode": ep, "frame": fr, "camera": cam, "boxes": []}
                rec["boxes"].append({
                    "label": body.get("label", "target"),
                    "bbox_xyxy": body.get("bbox"),
                    "comment": body.get("comment", ""),
                    "timestamp": datetime.now().isoformat(),
                })
                d[k] = rec
                save_annotations(d)
            self._json(200, {"ok": True})
        elif url.path == "/delbox":
            ep = int(qs.get("ep", ["0"])[0])
            fr = int(qs.get("fr", ["0"])[0])
            cam = qs.get("cam", ["center"])[0]
            i = int(qs.get("i", ["0"])[0])
            with LOCK:
                d = load_annotations()
                k = annotation_key(ep, fr, cam)
                rec = d.get(k)
                if rec and 0 <= i < len(rec.get("boxes", [])):
                    rec["boxes"].pop(i)
                    if not rec["boxes"]:
                        d.pop(k, None)
                    save_annotations(d)
            self._json(200, {"ok": True})
        elif url.path == "/clearall":
            ep = int(qs.get("ep", ["0"])[0])
            fr = int(qs.get("fr", ["0"])[0])
            with LOCK:
                d = load_annotations()
                for cam in ("left", "center", "right"):
                    d.pop(annotation_key(ep, fr, cam), None)
                save_annotations(d)
            self._json(200, {"ok": True})
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    global YOLO, DATASET_DIR, EPISODES_INFO, GT_PORT_2D
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--dataset", default=str(Path.home() / "aic_hexdex_sfp300"))
    args = ap.parse_args()

    DATASET_DIR = Path(args.dataset).expanduser()
    if not (DATASET_DIR / "meta" / "info.json").exists():
        sys.exit(f"No info.json under {DATASET_DIR}; finish download first.")

    if GT_PORT_2D_FILE.exists():
        print(f"Loading GT projections from {GT_PORT_2D_FILE}...")
        GT_PORT_2D = json.loads(GT_PORT_2D_FILE.read_text())
        n_eps = len(GT_PORT_2D)
        n_frames = sum(len(v) for v in GT_PORT_2D.values())
        print(f"  Loaded {n_frames} GT-projected frames across {n_eps} episodes")
    else:
        print(f"  (no GT file at {GT_PORT_2D_FILE} — overlay disabled)")

    YOLO = YoloPosePortDetector(conf=0.25)
    print(f"Loading dataset from {DATASET_DIR}...")
    EPISODES_INFO, fps = discover_episodes()
    print(f"  Found {len(EPISODES_INFO)} episodes, fps={fps}")
    print(f"  Eval split: episodes {EVAL_START}–{len(EPISODES_INFO) - 1} "
          f"({len(EPISODES_INFO) - EVAL_START} episodes)")
    print(f"  Annotations: {ANNOTATIONS_FILE}")
    print(f"  Open http://localhost:{args.port}/")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
