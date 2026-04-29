#!/usr/bin/env python3
"""Simple annotation web UI.

Serves saved AIC frames from ~/aic_logs/<run>/trial_*/<frame>_<cam>.jpg
along with the current YOLO + classical detector outputs overlaid.
User clicks-and-drags on the image to mark the actual port location.
Annotations save to ~/aic_user_annotations.json.

Usage:
    pixi run python scripts/annotate_server.py [--port 8000]

Then open http://localhost:8000/ in a browser.

Output JSON schema:
{
  "<image_path>": {
      "port_type": "sfp" | "sc",
      "trial": "trial_01_sfp",
      "camera": "left" | "center" | "right",
      "frame_idx": 0,
      "user_bbox_xyxy": [x_min, y_min, x_max, y_max],
      "user_corners_xy": [[x,y],...]   // optional, future
      "comment": "...",
      "timestamp": "..."
  },
  ...
}
"""
from __future__ import annotations

import argparse
import io
import json
import re
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
ANNOTATIONS_FILE = Path.home() / "aic_user_annotations.json"
FRAMES = []  # list of (image_path, port_type, trial, camera, frame_idx)
ALL_FRAMES = []
LOCK = threading.Lock()


def _set_frames(new_list):
    global FRAMES
    FRAMES = new_list


def discover_frames():
    base = Path.home() / "aic_logs"
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name[0:4].isdigit()])
    if not runs:
        return []
    run_dir = runs[-1]
    out = []
    for trial in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("trial_")):
        task_p = trial / "task.json"
        if not task_p.exists():
            continue
        task = json.loads(task_p.read_text())
        port_type = task["port_type"]
        for jpg in sorted(trial.glob("*_center.jpg")):
            stem = jpg.stem.replace("_center", "")
            try:
                frame_idx = int(stem)
            except ValueError:
                frame_idx = -1
            for cam in ("left", "center", "right"):
                cam_path = jpg.with_name(f"{stem}_{cam}.jpg")
                if cam_path.exists():
                    out.append((cam_path, port_type, trial.name, cam, frame_idx))
    return out


def load_annotations():
    if ANNOTATIONS_FILE.exists():
        try:
            return json.loads(ANNOTATIONS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_annotations(d):
    ANNOTATIONS_FILE.write_text(json.dumps(d, indent=2))


def encode_with_overlays(frame_idx_in_list: int):
    """Read the image, draw current detector outputs and any saved annotation, return JPEG bytes."""
    if not (0 <= frame_idx_in_list < len(FRAMES)):
        return None, None
    img_path, port_type, trial, cam, frame_idx = FRAMES[frame_idx_in_list]
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = img_bgr.copy()

    # Run detectors
    info = {
        "frame_idx": frame_idx_in_list,
        "image_path": str(img_path),
        "trial": trial,
        "camera": cam,
        "port_type": port_type,
        "image_w": img_bgr.shape[1],
        "image_h": img_bgr.shape[0],
        "yolo": None,
        "classical": None,
        "user": None,
    }

    yolo_det = YOLO.detect(img_rgb, port_type) if (YOLO and YOLO.available) else None
    if yolo_det is not None:
        if yolo_det.corners_xy is not None:
            box = yolo_det.corners_xy.astype(np.int32)
            cv2.polylines(overlay, [box], True, (0, 255, 0), 2)
        cv2.putText(overlay, f"YOLO {yolo_det.score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        info["yolo"] = {"cx": yolo_det.cx, "cy": yolo_det.cy,
                        "w": yolo_det.width, "h": yolo_det.height,
                        "score": yolo_det.score}

    cls_det = detect_classical(img_rgb, port_type, refine=True)
    if cls_det is not None:
        if cls_det.corners_xy is not None:
            box = cls_det.corners_xy.astype(np.int32)
            cv2.polylines(overlay, [box], True, (0, 0, 255), 2)
        cv2.putText(overlay, "classical", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        info["classical"] = {"cx": cls_det.cx, "cy": cls_det.cy,
                             "w": cls_det.width, "h": cls_det.height,
                             "score": cls_det.score}

    # Show saved annotation: multiple boxes, color by label
    saved = load_annotations()
    s = saved.get(str(img_path))
    if s is not None:
        # Backward compat: legacy single user_bbox_xyxy
        if "boxes" not in s and s.get("user_bbox_xyxy"):
            s = {"boxes": [{"label": "target", "bbox_xyxy": s["user_bbox_xyxy"],
                           "comment": s.get("comment", "")}]}
        boxes = s.get("boxes", [])
        for b in boxes:
            color = {
                "target": (0, 200, 255),       # orange-ish (BGR)
                "distractor": (255, 0, 200),   # magenta
                "other": (180, 180, 180),      # grey
            }.get(b.get("label", "target"), (255, 200, 0))
            x0, y0, x1, y1 = [int(round(v)) for v in b["bbox_xyxy"]]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
            label = b.get("label", "?")
            cv2.putText(overlay, label, (x0, max(20, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        info["user"] = {"boxes": boxes, **{k: v for k, v in s.items() if k != "boxes"}}

    ok, jpg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None, None
    return jpg.tobytes(), info


HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>AIC port annotator</title>
<style>
  body { background:#222; color:#eee; font-family:Helvetica,Arial,sans-serif; margin:0; padding:12px; }
  h2 { margin: 0 0 8px 0; }
  #wrap { display:flex; gap:16px; align-items:flex-start; }
  #imgwrap { position:relative; user-select:none; }
  #img { max-width: 1100px; cursor: crosshair; display:block; }
  #drawn { position:absolute; border:3px dashed #0ff; pointer-events:none; }
  #side { width:430px; }
  button, select { font-size:14px; padding:6px 10px; margin:2px; }
  .hint { color:#aaa; font-size:12px; }
  pre { background:#111; color:#cfc; padding:8px; max-height:280px; overflow:auto; font-size:12px; }
  .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }
  input[type=text]{ background:#111; color:#eee; border:1px solid #444; padding:4px 6px; }
  .legend { font-size:13px; margin-bottom:8px;}
  .legend span { padding:2px 8px; margin-right:6px; border-radius:3px; }
  .boxlist { background:#111; padding:6px; max-height:200px; overflow:auto; border:1px solid #333; }
  .boxitem { padding:4px 6px; margin:2px 0; background:#222; border-left:4px solid #fa0; display:flex; justify-content:space-between; gap:6px; align-items:center; }
  .boxitem.target { border-left-color:#fa0; }
  .boxitem.distractor { border-left-color:#a0f; }
  .boxitem.other { border-left-color:#888; }
  .boxitem .meta { font-size:12px; color:#ccc; flex:1; }
  .boxitem button { padding:2px 6px; font-size:12px; }
</style></head>
<body>
<h2>AIC port annotator (multi-box)</h2>
<div class="legend">
  <span style="background:#0c0;color:#000">YOLO</span>
  <span style="background:#f00;color:#fff">classical</span>
  <span style="background:#fa0;color:#000">your: target</span>
  <span style="background:#a0f;color:#fff">your: distractor</span>
  <span style="background:#888;color:#000">your: other</span>
  <span style="background:#0ff;color:#000">drawing</span>
</div>
<div id="wrap">
  <div id="imgwrap">
    <img id="img" src="" />
    <div id="drawn" style="display:none"></div>
  </div>
  <div id="side">
    <div class="row">
      <button id="prev">&laquo; Prev</button>
      <button id="next">Next &raquo;</button>
      <span id="counter"></span>
    </div>
    <div class="row">
      <span>Jump to:</span>
      <input id="jumpidx" type="text" size="5" />
      <button id="jumpbtn">Go</button>
    </div>
    <div class="row">
      <span>Filter:</span>
      <select id="trialfilter"><option value="">all trials</option></select>
      <select id="camfilter">
        <option value="">all cams</option>
        <option value="left">left</option>
        <option value="center">center</option>
        <option value="right">right</option>
      </select>
      <button id="apply">apply</button>
    </div>
    <hr>
    <div class="hint">
      <b>Click and drag</b> on image to draw a box. Pick a label below, then <b>Add box</b>.<br>
      Each frame can have multiple boxes (target + distractor + other).
    </div>
    <div class="row">
      <span>Label:</span>
      <select id="labelsel">
        <option value="target">target (the actual sfp_port_0 / sc target)</option>
        <option value="distractor">distractor (other port visible)</option>
        <option value="other">other (anything else worth marking)</option>
      </select>
    </div>
    <div class="row">
      Comment: <input id="comment" type="text" size="32" placeholder="optional note for this box" />
    </div>
    <div class="row">
      <button id="addbox">+ Add box</button>
      <button id="clearall">Clear all on this frame</button>
    </div>
    <hr>
    <div><b>Boxes on this frame:</b></div>
    <div id="boxlist" class="boxlist">(none yet)</div>
    <hr>
    <pre id="info"></pre>
  </div>
</div>
<script>
let idx = 0;
let total = 0;
let drawing = false;
let drawStart = null;
let drawCur = null;
let imgNaturalW = 0;
let imgNaturalH = 0;
let imgDispW = 0;
let imgDispH = 0;

const img = document.getElementById('img');
const drawn = document.getElementById('drawn');
const counter = document.getElementById('counter');
const info = document.getElementById('info');
const comment = document.getElementById('comment');
const labelsel = document.getElementById('labelsel');
const boxlist = document.getElementById('boxlist');
const trialFilter = document.getElementById('trialfilter');
const camFilter = document.getElementById('camfilter');

function refreshImage() {
  const ts = Date.now();
  let url = `/image?idx=${idx}&t=${ts}`;
  img.src = url;
}
async function refreshInfo() {
  const r = await fetch(`/info?idx=${idx}`);
  const d = await r.json();
  total = d.total;
  counter.textContent = `frame ${idx+1} of ${total}`;
  info.textContent = JSON.stringify({trial:d.trial, camera:d.camera, port_type:d.port_type, image_path:d.image_path, yolo:d.yolo, classical:d.classical}, null, 2);
  // boxes list
  const boxes = (d.user && d.user.boxes) || [];
  if (!boxes.length) {
    boxlist.textContent = '(none yet)';
  } else {
    boxlist.innerHTML = boxes.map((b, i) => `
      <div class="boxitem ${b.label}">
        <div class="meta">
          <b>${b.label}</b> [${b.bbox_xyxy.map(v=>v.toFixed(0)).join(', ')}]<br>
          ${b.comment ? '<i>' + b.comment + '</i>' : ''}
        </div>
        <button onclick="deleteBox(${i})">x</button>
      </div>`).join('');
  }
}
async function refreshFilters() {
  const r = await fetch('/trials');
  const d = await r.json();
  trialFilter.innerHTML = '<option value="">all trials</option>' +
    d.trials.map(t => `<option value="${t}">${t}</option>`).join('');
}

img.draggable = false;  // disable native image-drag which can intercept

function startDraw(e) {
  // Always reset state on a fresh click — never carry over from prior box.
  drawing = true;
  const rect = img.getBoundingClientRect();
  drawStart = [e.clientX - rect.left, e.clientY - rect.top];
  drawCur = [drawStart[0], drawStart[1]];  // initialize so drawn renders even on a click without drag
  imgDispW = rect.width;
  imgDispH = rect.height;
  imgNaturalW = img.naturalWidth || imgNaturalW;
  imgNaturalH = img.naturalHeight || imgNaturalH;
  drawn.style.display = 'block';
  drawn.style.left = drawStart[0] + 'px';
  drawn.style.top = drawStart[1] + 'px';
  drawn.style.width = '1px';
  drawn.style.height = '1px';
  e.preventDefault();
  e.stopPropagation();
}
function moveDraw(e) {
  if (!drawing) return;
  const rect = img.getBoundingClientRect();
  drawCur = [e.clientX - rect.left, e.clientY - rect.top];
  const x = Math.min(drawStart[0], drawCur[0]);
  const y = Math.min(drawStart[1], drawCur[1]);
  const w = Math.max(1, Math.abs(drawStart[0] - drawCur[0]));
  const h = Math.max(1, Math.abs(drawStart[1] - drawCur[1]));
  drawn.style.left = x + 'px';
  drawn.style.top = y + 'px';
  drawn.style.width = w + 'px';
  drawn.style.height = h + 'px';
}
function endDraw() {
  drawing = false;
}

// Bind on the wrapper so clicks on the dashed-overlay area still trigger a new draw.
const wrap = document.getElementById('imgwrap');
wrap.addEventListener('mousedown', startDraw);
window.addEventListener('mousemove', moveDraw);
window.addEventListener('mouseup', endDraw);

document.getElementById('addbox').onclick = async () => {
  if (!drawStart || !drawCur) { alert('Draw a box first (click and drag on the image)'); return; }
  const sx = imgNaturalW / imgDispW;
  const sy = imgNaturalH / imgDispH;
  const x0 = Math.min(drawStart[0], drawCur[0]) * sx;
  const y0 = Math.min(drawStart[1], drawCur[1]) * sy;
  const x1 = Math.max(drawStart[0], drawCur[0]) * sx;
  const y1 = Math.max(drawStart[1], drawCur[1]) * sy;
  await fetch(`/addbox?idx=${idx}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({bbox: [x0, y0, x1, y1], label: labelsel.value, comment: comment.value})
  });
  drawStart = null; drawCur = null;
  drawn.style.display = 'none';
  comment.value = '';
  refreshImage(); refreshInfo();
};
async function deleteBox(i) {
  await fetch(`/delbox?idx=${idx}&i=${i}`, {method:'POST'});
  refreshImage(); refreshInfo();
}
window.deleteBox = deleteBox;

document.getElementById('clearall').onclick = async () => {
  if (!confirm('Delete all boxes on this frame?')) return;
  await fetch(`/clearall?idx=${idx}`, {method:'POST'});
  drawStart = null; drawCur = null;
  drawn.style.display = 'none';
  refreshImage(); refreshInfo();
};
document.getElementById('prev').onclick = () => { if (idx>0){idx--; drawn.style.display='none'; refreshImage(); refreshInfo();}};
document.getElementById('next').onclick = () => { if (idx<total-1){idx++; drawn.style.display='none'; refreshImage(); refreshInfo();}};
document.getElementById('jumpbtn').onclick = () => {
  const v = parseInt(document.getElementById('jumpidx').value, 10);
  if (!isNaN(v) && v >= 1 && v <= total) { idx = v-1; refreshImage(); refreshInfo(); drawn.style.display='none'; }
};
document.getElementById('apply').onclick = async () => {
  await fetch(`/setfilter?trial=${encodeURIComponent(trialFilter.value)}&cam=${encodeURIComponent(camFilter.value)}`, {method:'POST'});
  idx = 0;
  refreshImage(); refreshInfo();
};
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') document.getElementById('prev').click();
  if (e.key === 'ArrowRight') document.getElementById('next').click();
  if (e.key === 'a') document.getElementById('addbox').click();
  if (e.key === '1') labelsel.value = 'target';
  if (e.key === '2') labelsel.value = 'distractor';
  if (e.key === '3') labelsel.value = 'other';
});
refreshFilters().then(() => { refreshImage(); refreshInfo(); });
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
        if url.path == "/" or url.path == "/index.html":
            data = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif url.path == "/image":
            qs = parse_qs(url.query)
            idx = int(qs.get("idx", ["0"])[0])
            jpg, _ = encode_with_overlays(idx)
            if jpg is None:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpg)
        elif url.path == "/info":
            qs = parse_qs(url.query)
            idx = int(qs.get("idx", ["0"])[0])
            _, info = encode_with_overlays(idx)
            if info is None:
                self._json(404, {"error": "no frame"}); return
            info["total"] = len(FRAMES)
            self._json(200, info)
        elif url.path == "/trials":
            trials = sorted(set(f[2] for f in FRAMES))
            self._json(200, {"trials": trials})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path == "/addbox":
            idx = int(qs.get("idx", ["0"])[0])
            if not (0 <= idx < len(FRAMES)):
                self._json(400, {"error": "bad idx"}); return
            img_path, port_type, trial, cam, frame_idx = FRAMES[idx]
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            new_box = {
                "label": body.get("label", "target"),
                "bbox_xyxy": body.get("bbox"),
                "comment": body.get("comment", ""),
                "timestamp": datetime.now().isoformat(),
            }
            with LOCK:
                d = load_annotations()
                rec = d.get(str(img_path)) or {
                    "port_type": port_type,
                    "trial": trial,
                    "camera": cam,
                    "frame_idx": frame_idx,
                    "boxes": [],
                }
                # Backward compat: migrate any legacy single bbox
                if "boxes" not in rec and rec.get("user_bbox_xyxy"):
                    rec["boxes"] = [{"label": "target", "bbox_xyxy": rec.pop("user_bbox_xyxy"),
                                     "comment": rec.pop("comment", "")}]
                rec.setdefault("boxes", []).append(new_box)
                d[str(img_path)] = rec
                save_annotations(d)
            self._json(200, {"ok": True})
        elif url.path == "/delbox":
            idx = int(qs.get("idx", ["0"])[0])
            i = int(qs.get("i", ["0"])[0])
            if not (0 <= idx < len(FRAMES)):
                self._json(400, {"error": "bad idx"}); return
            img_path, *_ = FRAMES[idx]
            with LOCK:
                d = load_annotations()
                rec = d.get(str(img_path))
                if rec and "boxes" in rec and 0 <= i < len(rec["boxes"]):
                    rec["boxes"].pop(i)
                    if not rec["boxes"]:
                        d.pop(str(img_path), None)
                    save_annotations(d)
            self._json(200, {"ok": True})
        elif url.path == "/clearall":
            idx = int(qs.get("idx", ["0"])[0])
            if not (0 <= idx < len(FRAMES)):
                self._json(400, {"error": "bad idx"}); return
            img_path, *_ = FRAMES[idx]
            with LOCK:
                d = load_annotations()
                d.pop(str(img_path), None)
                save_annotations(d)
            self._json(200, {"ok": True})
        elif url.path == "/setfilter":
            trial = qs.get("trial", [""])[0]
            cam = qs.get("cam", [""])[0]
            with LOCK:
                _set_frames([f for f in ALL_FRAMES if (not trial or f[2] == trial) and (not cam or f[3] == cam)])
            self._json(200, {"total": len(FRAMES)})
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    global YOLO, ALL_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    YOLO = YoloPosePortDetector(conf=0.25)
    ALL_FRAMES = discover_frames()
    _set_frames(list(ALL_FRAMES))
    if not FRAMES:
        sys.exit("No frames found under ~/aic_logs/<run>/trial_*/")
    print(f"Discovered {len(FRAMES)} frames across "
          f"{len(set(f[2] for f in FRAMES))} trials.")
    print(f"Annotations file: {ANNOTATIONS_FILE}")
    print(f"Open http://localhost:{args.port}/ in your browser")
    print("Keys: arrow-left/right = prev/next; s = save")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
