"""Classical OpenCV port detector for SFP and SC.

Two simple, fast detectors that operate on a single RGB image and return
the 2D port-mouth location (center + size + orientation in pixel space).
The 6D pose lift (Block D) is in port_pose.py — this module is purely 2D.

Design notes:
  * SFP port mouth = dark rectangle (~13.7 x 8.5 mm) on a green NIC card.
    Detected by HSV-masking out the PCB green, finding dark blobs, then
    fitting a minAreaRect with aspect-ratio gate.
  * SC port mouth = small circle (~2.5 mm radius) on a grey housing.
    Detected with HoughCircles on the grayscale image, with radius bounds
    derived from the expected camera distance at trial start.
  * The cable starts a few cm from the target port so the port is roughly
    centered in the wrist cameras. We rely on this by sorting candidates
    by distance to image center and taking the closest valid one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class PortDetection2D:
    """A 2D detection of a port mouth in pixel coordinates."""

    port_type: str          # "sfp" or "sc"
    cx: float               # center x (pixels)
    cy: float               # center y (pixels)
    width: float            # rotated-rect width (pixels)
    height: float           # rotated-rect height (pixels)
    angle_deg: float        # rotated-rect angle (degrees, OpenCV convention)
    score: float            # detection confidence (0..1)
    corners_xy: Optional[np.ndarray] = None  # (4, 2) box corners, ordered TL,TR,BR,BL
    contour: Optional[np.ndarray] = None  # raw contour, for visualization


# Per-camera priors derived from user audit annotations (~7 episodes, all rails).
# Used to (a) restrict the search ROI and (b) disambiguate target vs distractor.
# cx_mean / cy_mean are normalized to image dimensions (so they generalize if
# the dataset resolution changes).
PER_CAM_PRIORS = {
    "left": {
        "cx_mean": 813 / 1152, "cy_mean": 305 / 1024,
        "search_radius_frac": 0.35,
        "area_min": 1000, "area_max": 6000,
        # Among multiple candidates, target is BELOW distractor (higher cy).
        "tiebreak": "lower_cy",
    },
    "center": {
        "cx_mean": 407 / 1152, "cy_mean": 313 / 1024,
        "search_radius_frac": 0.35,
        "area_min": 1000, "area_max": 6000,
        # Target is to the RIGHT of distractor (higher cx).
        "tiebreak": "higher_cx",
    },
    "right": {
        "cx_mean": 256 / 1152, "cy_mean": 596 / 1024,
        "search_radius_frac": 0.35,
        "area_min": 800, "area_max": 5000,
        # Target visually ABOVE distractor on the right camera.
        # Image y grows downward, so "above" => smaller cy => lower_cy.
        "tiebreak": "lower_cy",
    },
    None: {
        "cx_mean": 0.5, "cy_mean": 0.4,
        "search_radius_frac": 1.0,
        "area_min": 800, "area_max": 200_000,
        "tiebreak": "centermost",
    },
}


def detect_sfp_port(image_rgb: np.ndarray,
                     y_max_frac: float = 0.85,
                     camera: Optional[str] = None) -> Optional[PortDetection2D]:
    """Find the SFP port mouth in an RGB image.

    If camera ('left'|'center'|'right') is provided, uses per-camera priors
    derived from user annotations: tighter search ROI, area gates, and
    target/distractor tiebreaker. Otherwise falls back to the original
    image-center heuristic.
    """
    h, w = image_rgb.shape[:2]
    y_max = h * y_max_frac
    prior = PER_CAM_PRIORS.get(camera, PER_CAM_PRIORS[None])
    expected_cx = prior["cx_mean"] * w
    expected_cy = prior["cy_mean"] * h
    search_radius = prior["search_radius_frac"] * max(w, h)
    area_min = prior["area_min"]
    area_max = prior["area_max"]
    tiebreak = prior["tiebreak"]
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # NIC card PCB is saturated green. Mask out anything green-ish and bright.
    # The port mouth is dark (low V) and not green. We look for dark blobs.
    not_green = cv2.bitwise_not(
        cv2.inRange(hsv, (35, 50, 30), (90, 255, 255))
    )  # 1 where NOT green
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 70))  # very dark pixels
    cand = cv2.bitwise_and(not_green, dark)

    # Don't morph-close — that was merging the small port mouth into the
    # surrounding NIC-card body so it stopped being a separate contour.
    # Use a light open instead to remove single-pixel noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel, iterations=1)

    # First pass: external contours (the big housing masses).
    # Second pass: also look at INTERNAL holes within those masses, since
    # the port mouth is often a hole inside the NIC-card silhouette.
    contours_all, hierarchy = cv2.findContours(cand, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours_all) if contours_all is not None else []
    if not contours:
        return None

    image_center = np.array([w / 2, h / 2])

    best: Optional[PortDetection2D] = None
    best_score = 0.0
    candidates = []  # collect ALL candidates that pass the gates; pick by tiebreak below
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min or area > area_max:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if rw < 5 or rh < 5:
            continue
        if cy > y_max:
            continue  # gripper region
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        ar = long_side / short_side
        if not (1.0 <= ar <= 2.8):  # slightly wider since right cam is more square-ish
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area
        if solidity < 0.55:
            continue
        # Spatial prior: must be within search_radius of camera-specific expected center.
        d_to_expected = ((cx - expected_cx) ** 2 + (cy - expected_cy) ** 2) ** 0.5
        if d_to_expected > search_radius:
            continue

        # Score combines proximity-to-prior + aspect ratio sanity.
        prior_score = max(0.0, 1.0 - d_to_expected / search_radius)
        ar_score = max(0.0, 1.0 - abs(ar - 1.6) / 1.0)
        score = 0.7 * prior_score + 0.3 * ar_score

        box = cv2.boxPoints(rect)
        sums = box.sum(axis=1)
        diffs = np.diff(box, axis=1).ravel()
        tl = box[np.argmin(sums)]
        br = box[np.argmax(sums)]
        tr = box[np.argmin(diffs)]
        bl = box[np.argmax(diffs)]
        corners = np.stack([tl, tr, br, bl]).astype(np.float32)
        candidates.append({
            "cx": float(cx), "cy": float(cy),
            "rw": float(rw), "rh": float(rh), "ang": float(ang),
            "score": float(score), "corners": corners, "contour": c,
        })

    if not candidates:
        return None

    # Tiebreaker: when multiple ports are visible (target + distractor),
    # apply camera-specific rule.
    if len(candidates) >= 2 and tiebreak in ("higher_cx", "lower_cx", "higher_cy", "lower_cy"):
        # Take the top-2 candidates by initial score, then pick by tiebreak.
        candidates.sort(key=lambda c: c["score"], reverse=True)
        top2 = candidates[:2]
        if tiebreak == "higher_cx":
            best_c = max(top2, key=lambda c: c["cx"])
        elif tiebreak == "lower_cx":
            best_c = min(top2, key=lambda c: c["cx"])
        elif tiebreak == "higher_cy":
            best_c = max(top2, key=lambda c: c["cy"])
        elif tiebreak == "lower_cy":
            best_c = min(top2, key=lambda c: c["cy"])
        else:
            best_c = top2[0]
    else:
        best_c = max(candidates, key=lambda c: c["score"])

    return PortDetection2D(
        port_type="sfp",
        cx=best_c["cx"], cy=best_c["cy"],
        width=best_c["rw"], height=best_c["rh"],
        angle_deg=best_c["ang"],
        score=best_c["score"],
        corners_xy=best_c["corners"],
        contour=best_c["contour"],
    )


def detect_sc_port(image_rgb: np.ndarray,
                    y_max_frac: float = 0.60) -> Optional[PortDetection2D]:
    """Find the SC port mouth in an RGB image.

    The SC port appears as a small bright blue connector body containing
    two darker holes (one of which is the target). Strategy:
      1. Find blue blobs in HSV space.
      2. Within each blue blob, find dark sub-regions (the holes).
      3. Pick the dark sub-region closest to image center as the port mouth.
    Detections in the lower y_max_frac portion of frame are rejected
    (gripper region).
    """
    h, w = image_rgb.shape[:2]
    y_max = h * y_max_frac
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Step 1: HSV blue mask. SC port body is a saturated blue.
    blue_mask = cv2.inRange(hsv, (90, 80, 60), (130, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _detect_sc_port_circular(image_rgb)

    # Pick the LARGEST blue blob — that's the SC port body.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    best: Optional[PortDetection2D] = None
    best_score = 0.0

    # Step 2: For the largest blue blob (the SC port body), find the dark
    # SUB-MASK regions where blue_mask=0 inside the blob's bounding box.
    # These are the actual fiber-hole openings.
    image_center = np.array([w / 2, h / 2])
    for blue_c in contours[:2]:
        ba = cv2.contourArea(blue_c)
        if ba < 50:
            continue
        bx, by, bw, bh = cv2.boundingRect(blue_c)
        # Reject if the blue blob's center is below the gripper exclusion line
        if by + bh / 2 > y_max:
            continue
        # Look at the bbox of the blue blob; find pixels that are DARK
        # AND not_blue (the actual hole interiors).
        gray_roi = cv2.cvtColor(img_bgr[by:by + bh, bx:bx + bw], cv2.COLOR_BGR2GRAY)
        not_blue_roi = (~blue_mask[by:by + bh, bx:bx + bw].astype(bool)).astype(np.uint8) * 255
        dark_roi = (gray_roi < 80).astype(np.uint8) * 255
        hole_mask = cv2.bitwise_and(dark_roi, not_blue_roi)
        # The whole blue contour shape includes the connector body. Holes are
        # surrounded by blue. Erode the hole_mask slightly so we only get
        # interior holes, not the outer contour boundary.
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        hole_mask = cv2.erode(hole_mask, kernel2, iterations=1)
        sub_contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for sc in sub_contours:
            area = cv2.contourArea(sc)
            # Real fiber hole at ~30-50cm distance is ~3-15 px^2.
            if area < 3 or area > 80:
                continue
            rect = cv2.minAreaRect(sc)
            (cx_local, cy_local), (rw, rh), ang = rect
            cx_full = cx_local + bx
            cy_full = cy_local + by
            # The actual port mouth is in the upper portion of the connector
            # (where the cable enters). Prefer detections near the long-axis
            # extreme of the blue blob nearest the cable approach.
            # Heuristic: prefer aspect-square small holes near the top of bbox
            ar = max(rw, rh) / max(min(rw, rh), 1)
            if ar > 2.5:
                continue
            d_center = np.linalg.norm(np.array([cx_full, cy_full]) - image_center)
            center_score = max(0.0, 1.0 - d_center / max(w, h))
            # Boost score for being in the upper half of the blue blob (where
            # the SC mouth is usually located given the typical scene framing).
            top_score = max(0.0, 1.0 - (cy_local / max(bh, 1)))
            score = 0.5 * center_score + 0.5 * top_score
            if score > best_score:
                box = cv2.boxPoints(rect)
                box[:, 0] += bx
                box[:, 1] += by
                # Sort corners TL, TR, BR, BL
                sums = box.sum(axis=1)
                diffs = np.diff(box, axis=1).ravel()
                tl = box[np.argmin(sums)]
                br = box[np.argmax(sums)]
                tr = box[np.argmin(diffs)]
                bl = box[np.argmax(diffs)]
                corners = np.stack([tl, tr, br, bl]).astype(np.float32)
                best_score = score
                best = PortDetection2D(
                    port_type="sc",
                    cx=float(cx_full), cy=float(cy_full),
                    width=float(rw), height=float(rh),
                    angle_deg=float(ang),
                    score=float(score),
                    corners_xy=corners,
                )
    return best


def _detect_sc_port_circular(image_rgb: np.ndarray) -> Optional[PortDetection2D]:
    """Fallback Hough-circle SC detector for backward compat."""
    h, w = image_rgb.shape[:2]
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.0, minDist=40,
        param1=120, param2=30, minRadius=10, maxRadius=120,
    )
    if circles is None:
        return None
    circles = circles[0]
    image_center = np.array([w / 2, h / 2])
    best = None
    best_score = 0.0
    for cx, cy, r in circles:
        mask = np.zeros_like(gray)
        cv2.circle(mask, (int(cx), int(cy)), max(int(r * 0.6), 1), 255, -1)
        if cv2.mean(gray, mask=mask)[0] > 100:
            continue
        d = np.linalg.norm(np.array([cx, cy]) - image_center)
        s = max(0.0, 1.0 - d / max(w, h))
        if s > best_score:
            best_score = s
            best = PortDetection2D(
                port_type="sc", cx=float(cx), cy=float(cy),
                width=float(r * 2), height=float(r * 2), angle_deg=0.0, score=float(s),
            )
    return best


def shrink_corners_by_rim(det: PortDetection2D,
                           shrink_w: float = 0.76,
                           shrink_h: float = 0.65) -> Optional[PortDetection2D]:
    """Shrink the detected bbox corners toward the center by per-axis ratios.

    Rationale: the dark contour the SFP detector finds extends past the
    spec port-mouth into the surrounding housing rim. With SFP_PORT_MOUTH_M
    set to spec (13.7x8.5mm), the detected pixel size needs to be SHRUNK
    to match. Empirical ratio: visible region ~ 1.31x spec width and
    ~ 1.53x spec height (T1 calibration: detected 35x54 px maps to spec
    13.7x8.5mm at 0.405m depth via fx=1236.6).
    """
    if det is None or det.corners_xy is None:
        return det
    cx, cy = det.cx, det.cy
    new_corners = []
    for (x, y) in det.corners_xy:
        dx = x - cx
        dy = y - cy
        new_corners.append([cx + dx * shrink_w, cy + dy * shrink_h])
    new_corners = np.array(new_corners, dtype=np.float32)
    return PortDetection2D(
        port_type=det.port_type,
        cx=cx, cy=cy,
        width=det.width * shrink_w,
        height=det.height * shrink_h,
        angle_deg=det.angle_deg,
        score=det.score,
        corners_xy=new_corners,
        contour=det.contour,
    )


def refine_corners_subpixel(image_rgb: np.ndarray,
                             det: PortDetection2D) -> Optional[PortDetection2D]:
    """Sub-pixel corner refinement using cv2.cornerSubPix.

    The detected corners come from minAreaRect on a coarse contour and
    are snapped to integer pixels. Sub-pixel refinement finds the
    actual gradient-strong edge within ±2 px and updates the corner.
    """
    if det is None or det.corners_xy is None:
        return det
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners = det.corners_xy.astype(np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        refined = cv2.cornerSubPix(gray, corners, winSize=(5, 5),
                                    zeroZone=(-1, -1), criteria=criteria)
    except cv2.error:
        return det
    refined = refined.reshape(-1, 2)
    new_w = float(np.linalg.norm(refined[1] - refined[0]))
    new_h = float(np.linalg.norm(refined[2] - refined[1]))
    new_cx = float(refined[:, 0].mean())
    new_cy = float(refined[:, 1].mean())
    return PortDetection2D(
        port_type=det.port_type,
        cx=new_cx, cy=new_cy,
        width=new_w, height=new_h,
        angle_deg=det.angle_deg,
        score=det.score,
        corners_xy=refined,
        contour=det.contour,
    )


def refine_corners_via_edges(image_rgb: np.ndarray,
                              det: PortDetection2D,
                              margin: float = 1.6) -> Optional[PortDetection2D]:
    """Refine the 4 corners by snapping to inner edges.

    Crops a margin around the detected bbox, runs Canny + Hough line
    detection, clusters lines into horizontal/vertical, picks the inner
    pair on each axis (closest to detected center) — those are the
    actual port-mouth edges. Intersections give precise corners.
    """
    if det is None:
        return None
    h, w = image_rgb.shape[:2]
    half_w = max(int(det.width / 2 * margin), 6)
    half_h = max(int(det.height / 2 * margin), 6)
    cx, cy = int(det.cx), int(det.cy)
    x0 = max(0, cx - half_w)
    y0 = max(0, cy - half_h)
    x1 = min(w, cx + half_w)
    y1 = min(h, cy + half_h)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return det

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

    # Slight blur to suppress noise.
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, 40, 150)

    # Hough line detection. Tune to find ~10-30 short line segments.
    minlen = max(int(min(det.width, det.height) * 0.4), 6)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                            minLineLength=minlen, maxLineGap=4)
    if lines is None or len(lines) < 4:
        return det

    h_pos = []  # y-coordinates of horizontal lines
    v_pos = []  # x-coordinates of vertical lines
    for ln in lines:
        x_a, y_a, x_b, y_b = ln[0]
        dx = x_b - x_a
        dy = y_b - y_a
        ang = abs(np.degrees(np.arctan2(dy, dx)))
        if ang > 90:
            ang = 180 - ang
        if ang < 25:  # ~horizontal
            h_pos.append((y_a + y_b) / 2.0)
        elif ang > 65:  # ~vertical
            v_pos.append((x_a + x_b) / 2.0)

    if len(h_pos) < 2 or len(v_pos) < 2:
        return det

    cx_local = cx - x0
    cy_local = cy - y0

    # Cluster nearby line positions (within 3 px), take the median per cluster.
    def cluster(positions, tol=3.0):
        positions = sorted(positions)
        clusters = []
        cur = [positions[0]]
        for p in positions[1:]:
            if p - cur[-1] <= tol:
                cur.append(p)
            else:
                clusters.append(np.median(cur))
                cur = [p]
        clusters.append(np.median(cur))
        return clusters

    h_clusters = cluster(h_pos)
    v_clusters = cluster(v_pos)
    if len(h_clusters) < 2 or len(v_clusters) < 2:
        return det

    # Find the inner pair: the two closest to detected center, one on each side.
    h_above = [h for h in h_clusters if h < cy_local]
    h_below = [h for h in h_clusters if h > cy_local]
    v_left = [v for v in v_clusters if v < cx_local]
    v_right = [v for v in v_clusters if v > cx_local]
    if not (h_above and h_below and v_left and v_right):
        return det
    top = max(h_above)
    bottom = min(h_below)
    left = max(v_left)
    right = min(v_right)

    new_w = right - left
    new_h = bottom - top
    if new_w < 5 or new_h < 5:
        return det

    new_cx = (left + right) / 2 + x0
    new_cy = (top + bottom) / 2 + y0
    corners = np.array([
        [left + x0, top + y0],
        [right + x0, top + y0],
        [right + x0, bottom + y0],
        [left + x0, bottom + y0],
    ], dtype=np.float32)

    return PortDetection2D(
        port_type=det.port_type,
        cx=float(new_cx), cy=float(new_cy),
        width=float(new_w), height=float(new_h),
        angle_deg=0.0,
        score=det.score,
        corners_xy=corners,
        contour=det.contour,
    )


def detect_port(image_rgb: np.ndarray, port_type: str,
                expected_uv: Optional[tuple] = None,
                expected_radius: float = 200.0,
                refine: bool = False,
                camera: Optional[str] = None) -> Optional[PortDetection2D]:
    """Dispatch by port type. port_type is 'sfp' or 'sc'.

    If expected_uv is provided, prefer detections within `expected_radius`
    pixels of that point (a prior from external knowledge — e.g., the
    cable plug's projected forward position).

    If camera ('left'|'center'|'right') is provided, the SFP detector uses
    per-camera priors derived from user audit annotations.
    """
    pt = port_type.lower()
    if pt == "sfp":
        det = detect_sfp_port(image_rgb, camera=camera)
    elif pt == "sc":
        det = detect_sc_port(image_rgb)
    else:
        raise ValueError(f"Unknown port type {port_type!r}")

    if det is not None and expected_uv is not None:
        d = ((det.cx - expected_uv[0]) ** 2 + (det.cy - expected_uv[1]) ** 2) ** 0.5
        if d > expected_radius:
            return None  # too far from prior; reject

    if det is not None and refine and port_type.lower() == "sfp":
        det = shrink_corners_by_rim(det, shrink_w=0.76, shrink_h=0.65)

    return det


def detect_port_filtered(image_rgb: np.ndarray, port_type: str,
                         expected_uv: Optional[tuple] = None,
                         expected_radius: float = 200.0) -> Optional[PortDetection2D]:
    """Like detect_port but searches multiple candidates and returns the one
    closest to expected_uv (rather than the most-centered).

    Use this when you have a strong prior from cable/TCP position.
    """
    pt = port_type.lower()
    if pt == "sfp":
        return _detect_sfp_port_filtered(image_rgb, expected_uv, expected_radius)
    if pt == "sc":
        # SC detector already prefers the largest blue blob; no easy filter.
        return detect_sc_port(image_rgb)
    return None


def _detect_sfp_port_filtered(image_rgb, expected_uv, expected_radius,
                               y_max_frac=0.60):
    """SFP detector that returns ANY candidate near expected_uv."""
    h, w = image_rgb.shape[:2]
    y_max = h * y_max_frac
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    not_green = cv2.bitwise_not(cv2.inRange(hsv, (35, 50, 30), (90, 255, 255)))
    dark = cv2.inRange(hsv, (0, 0, 0), (179, 255, 70))
    cand = cv2.bitwise_and(not_green, dark)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_dist = float("inf") if expected_uv is not None else None

    for c in contours:
        area = cv2.contourArea(c)
        if area < 800 or area > 200_000:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if rw < 5 or rh < 5:
            continue
        if cy > y_max:
            continue  # exclude gripper region
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        if not (1.2 <= long_side / short_side <= 2.5):
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        if area / hull_area < 0.6:
            continue

        if expected_uv is not None:
            d = ((cx - expected_uv[0]) ** 2 + (cy - expected_uv[1]) ** 2) ** 0.5
            if d > expected_radius:
                continue
            if d < best_dist:
                best_dist = d
                box = cv2.boxPoints(rect)
                sums = box.sum(axis=1)
                diffs = np.diff(box, axis=1).ravel()
                tl = box[np.argmin(sums)]
                br = box[np.argmax(sums)]
                tr = box[np.argmin(diffs)]
                bl = box[np.argmax(diffs)]
                corners = np.stack([tl, tr, br, bl]).astype(np.float32)
                best = PortDetection2D(
                    port_type="sfp",
                    cx=float(cx), cy=float(cy),
                    width=float(rw), height=float(rh),
                    angle_deg=float(ang),
                    score=1.0 - d / expected_radius,
                    corners_xy=corners,
                    contour=c,
                )
    return best


def draw_detection(image_rgb: np.ndarray, det: PortDetection2D) -> np.ndarray:
    """Return a BGR image with the detection drawn on it (for cv2.imwrite)."""
    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    if det is None:
        return out
    if det.port_type == "sfp":
        rect = ((det.cx, det.cy), (det.width, det.height), det.angle_deg)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(out, [box], 0, (0, 255, 0), 2)
    else:
        cv2.circle(out, (int(det.cx), int(det.cy)), int(det.width / 2), (0, 255, 0), 2)
    cv2.putText(
        out, f"{det.port_type} {det.score:.2f}", (int(det.cx), int(det.cy) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )
    return out
