"""Find each duck's EXACT pixel position via color segmentation.

Unlike cell-based detection (VLM reporting "which grid cell"), this finds
the duck's actual centroid pixel — not a cell's center. A cell only ever
gives an approximation good to one cell's size; if the duck sits off-center
within its cell, the gripper aims at empty space next to it. Color
segmentation has no such rounding: it locates the real blob directly.

Deterministic, no VLM/API call, no per-call cost or run-to-run inconsistency.
"""

from __future__ import annotations

import colorsys
import logging

import numpy as np
import cv2

log = logging.getLogger(__name__)


def hex_to_hsv_center(hex_color: str) -> int:
    """OpenCV hue (0-179) at the center of a hex color."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return int(h * 179)


def hex_to_hsv_range(hex_color: str, hue_tol: int = 8, sat_min: int = 80, val_min: int = 60
                      ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Derive an HSV threshold range around a hex color (e.g. cell.yaml's block color).

    hue_tol: +/- tolerance around the hex color's hue, in OpenCV's 0-179 hue
    scale. Kept narrow (default 8) so adjacent duck colors (e.g. red at
    hue~0, yellow at hue~22) don't overlap and false-match each other's
    shadowed/anti-aliased edges.
    sat_min/val_min: minimum saturation/value to accept (filters out washed-out
    /shadowed regions and the gray board itself).

    Does NOT itself handle hue wraparound (red sits at hue~0, which wraps to
    ~179 on the other end) — use build_mask() below, which does.
    """
    hue_cv = hex_to_hsv_center(hex_color)
    lower = (max(0, hue_cv - hue_tol), sat_min, val_min)
    upper = (min(179, hue_cv + hue_tol), 255, 255)
    return lower, upper


def build_mask(hsv_frame, hex_color: str, hue_tol: int = 5, sat_min: int = 80, val_min: int = 60):
    """Threshold mask for hex_color, correctly handling hue wraparound.

    OpenCV hue is circular (0-179, where 179 is adjacent to 0 — both are
    "red"). A color near either end needs its window split into two ranges
    and OR'd together, or half its true pixels get missed. Colors away from
    the wrap boundary use a single plain range.
    """
    hue_cv = hex_to_hsv_center(hex_color)
    lo = hue_cv - hue_tol
    hi = hue_cv + hue_tol

    if lo < 0:
        mask1 = cv2.inRange(hsv_frame, np.array((0, sat_min, val_min)), np.array((hi, 255, 255)))
        mask2 = cv2.inRange(hsv_frame, np.array((179 + lo, sat_min, val_min)), np.array((179, 255, 255)))
        return cv2.bitwise_or(mask1, mask2)
    if hi > 179:
        mask1 = cv2.inRange(hsv_frame, np.array((lo, sat_min, val_min)), np.array((179, 255, 255)))
        mask2 = cv2.inRange(hsv_frame, np.array((0, sat_min, val_min)), np.array((hi - 179, 255, 255)))
        return cv2.bitwise_or(mask1, mask2)
    return cv2.inRange(hsv_frame, np.array((lo, sat_min, val_min)), np.array((hi, 255, 255)))


def find_duck_centroid(frame, hex_color: str, min_area: float = 1000.0
                        ) -> tuple[float, float, float] | None:
    """Find the largest blob matching hex_color in frame.

    Returns (px, py, area) of the largest matching blob's centroid, or None
    if nothing large enough is found. area is returned so callers can sanity
    -check size against the expected duck size (cell.yaml blocks.size).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, hex_color)

    # A glossy/specular block surface can split the color match into several
    # disconnected fragments (confirmed via hover_mask_debug.jpg: "a bunch
    # of white spots" instead of one solid blob), each too small alone to
    # clear min_area even though the block IS there. Dilate to bridge small
    # gaps between fragments before measuring, same fix already applied in
    # find_marker_patch for the same underlying problem.
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Rank by area so a duplicate/second matching object doesn't silently win
    # over the real one — if more than one candidate clears min_area, that's
    # ambiguous (e.g. two yellow objects in frame) and worth flagging rather
    # than picking one silently.
    candidates = sorted(
        ((c, cv2.contourArea(c)) for c in contours), key=lambda t: t[1], reverse=True
    )
    significant = [(c, a) for c, a in candidates if a >= min_area]
    if not significant:
        return None

    if len(significant) > 1:
        log.warning(
            "%d blobs >= min_area matched color %s (areas=%s) — picking the "
            "largest, but this usually means more than one object of this "
            "color is in frame",
            len(significant), hex_color, [round(a) for _, a in significant],
        )

    largest, area = significant[0]
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"], area


def find_all_ducks(frame, colors: dict[str, str], min_area: float = 1000.0
                    ) -> list[dict]:
    """Find every duck's pixel centroid, one per color in `colors`.

    colors: {duck_id: hex_color}, e.g. from cell.yaml's blocks.colors.
    Returns [{id, px, py, area}, ...] for colors that were actually found.
    """
    results = []
    for duck_id, hex_color in colors.items():
        found = find_duck_centroid(frame, hex_color, min_area=min_area)
        if found is not None:
            px, py, area = found
            results.append({"id": duck_id, "px": px, "py": py, "area": area})
    return results


def find_marker_patch(frame, hex_color: str, min_area: float = 20.0,
                       min_solidity: float = 0.6) -> tuple[float, float, float] | None:
    """Find a small, solid colored patch (e.g. foam stuck to the gripper).

    Like find_duck_centroid, but tuned for a marker that may look SMALL at
    some viewing angles (foam patch viewed edge-on) — a low min_area alone
    risks accepting stray noise speckle at that size, so this also requires
    reasonable SOLIDITY (contour area / convex hull area) to reject thin,
    ragged, or fragmented false matches, which noise speckle often is,
    while a real solid patch is not.

    Returns (px, py, area) of the best-qualifying blob, or None if nothing
    passes both the area and solidity checks.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, hex_color)
    # At long camera distance/steep angle, a small real patch can break into
    # several thin fragments each too small to have measurable contour area
    # (confirmed via debug_marker_size.py: 6 fragments all with area==0
    # despite the color match landing correctly on the real patch). Dilate
    # to merge nearby fragments into one blob before measuring, so a
    # genuinely-present-but-fragmented marker isn't missed.
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < min_solidity:
            continue
        candidates.append((c, area, solidity))

    if not candidates:
        return None

    if len(candidates) > 1:
        log.warning(
            "%d solid blobs >= min_area matched color %s (areas=%s) — "
            "picking the largest, but this usually means more than one "
            "object of this color is in frame",
            len(candidates), hex_color, [round(a) for _, a, _ in candidates],
        )

    largest, area, _ = max(candidates, key=lambda t: t[1])
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"], area


def find_black_marker(frame, dark_threshold: int = 60, min_area: float = 20.0,
                       min_solidity: float = 0.5) -> tuple[float, float, float] | None:
    """Find a dark/black marker via brightness thresholding, not hue.

    Black has near-zero saturation, so hue-based matching (build_mask) can
    never reliably find it — this thresholds on grayscale brightness
    instead, same technique used earlier for the grid's black origin
    marker. Use this for a black mark drawn on a block held by the gripper.

    Requires reasonable SOLIDITY (like find_marker_patch) to reject
    irregular shadows, which are also dark but typically not compact/solid
    the way a deliberately drawn mark is.

    Returns (px, py, area) of the best-qualifying blob, or None if nothing
    passes both the area and solidity checks.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < min_solidity:
            continue
        candidates.append((c, area, solidity))

    if not candidates:
        return None

    if len(candidates) > 1:
        log.warning(
            "%d solid dark blobs >= min_area found (areas=%s) — picking the "
            "largest, but this usually means more than one dark object is "
            "in frame (shadows, cables, arm body, etc.)",
            len(candidates), [round(a) for _, a, _ in candidates],
        )

    largest, area, _ = max(candidates, key=lambda t: t[1])
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"], area
