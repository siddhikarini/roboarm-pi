"""Find the pink grid's green-background target cell, to place a picked
green object INSIDE the correct cell instead of a hardcoded position.

The board has a pink grid drawn on it; exactly one cell has a green
background marking it as the target slot. Since the object being picked
is ALSO green, naive full-frame color detection can't tell "the object to
pick" apart from "the cell to place it in" -- both match the same color.

This resolves that ambiguity SPATIALLY instead of by color alone:
  1. Find the pink grid's bounding box (find_pink_grid_bbox).
  2. The target cell = the green blob INSIDE that bbox (find_green_cell).
  3. The object to pick = the green blob OUTSIDE that bbox (on the open
     table, not on the grid) -- see find_object_outside_grid.

Same dilate/combine-fragments approach as vision/board_crop.py, since a
hand-drawn or printed grid line can fragment into multiple contours the
same way the board's red boundary line did.
"""

from __future__ import annotations

import logging

import numpy as np
import cv2

from .color_detect import build_mask

log = logging.getLogger(__name__)

# Sampled via click_sample_color.py against a fresh board capture (14
# clicks across the pink grid lines), hue tightly clustered at 172-175.
PINK_GRID_HEX_DEFAULT = "#714951"


def find_pink_grid_bbox(frame, pink_hex: str = PINK_GRID_HEX_DEFAULT,
                         hue_tol: int = 8, sat_min: int = 70, val_min: int = 50,
                         min_area_frac: float = 0.4) -> tuple[int, int, int, int] | None:
    """Find the pink grid's bounding box in the frame.

    Returns (x0, y0, x1, y1), or None if no sufficiently large pink region
    is found. min_area_frac: fragments smaller than this fraction of the
    LARGEST pink fragment's area are dropped as noise before combining
    bounding boxes (same idea as board_crop.detect_board_rect).

    Confirmed via debugging: a real grid can produce ONE dominant fragment
    plus a small unrelated pink-hue false positive elsewhere in frame
    (e.g. glare/shadow) that's an order of magnitude smaller -- 0.02 was
    far too permissive and let a ~13%-of-max noise strip merge into the
    combined bbox, stretching it across nearly the whole frame. 0.4 keeps
    only fragments that are genuinely comparable in size to the largest
    (i.e. actually part of the same real grid), same logic as
    board_crop.detect_board_rect but tuned tighter since that case's real
    fragments were close in size to begin with (~4500 vs ~4000).
    """
    frame_h, frame_w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, pink_hex, hue_tol=hue_tol, sat_min=sat_min, val_min=val_min)

    # A thin lens/reflection artifact right at the frame's edge (confirmed:
    # an ~6px-wide strip pinned to x=474-481, i.e. the frame's right border)
    # can share a similar hue/sat/val to the real grid, and dilating BEFORE
    # removing it lets it fuse into a single connected blob with the real
    # grid -- at that point min_area_frac can't separate them anymore,
    # since findContours only sees one merged region. A real grid drawn on
    # the board interior shouldn't touch the frame's outer border; zero out
    # a thin margin around the edges BEFORE dilating, so an edge artifact
    # can never survive to merge into the real detection.
    edge_margin = max(2, min(frame_w, frame_h) // 20)
    mask[:edge_margin, :] = 0
    mask[-edge_margin:, :] = 0
    mask[:, :edge_margin] = 0
    mask[:, -edge_margin:] = 0

    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    significant = [c for c, a in zip(contours, areas) if a >= max_area * min_area_frac]

    all_points = np.vstack(significant)
    x, y, w, h = cv2.boundingRect(all_points)

    # A real grid should be a sub-region of the board, not the whole frame.
    # If thresholds are too loose (catching background/shadow/wood-grain
    # pixels that happen to fall in the pink hue range), the combined
    # bbox can balloon out to cover nearly the entire frame -- that's a
    # detection FAILURE, not a real grid, and silently continuing with it
    # breaks "outside the grid" (nothing is outside the whole frame).
    if w * h > 0.85 * frame_w * frame_h:
        log.warning(
            "pink grid bbox (%d,%d)-(%d,%d) covers %.0f%% of the frame -- "
            "likely a false-positive match (thresholds too loose), not a "
            "real grid. Tighten sat_min/val_min or lower hue_tol.",
            x, y, x + w, y + h, 100.0 * w * h / (frame_w * frame_h),
        )
        return None

    return x, y, x + w, y + h


def find_green_cell(frame, grid_bbox: tuple[int, int, int, int], green_hex: str,
                     min_area: float = 200.0) -> tuple[float, float, float] | None:
    """Find the green-background target cell's centroid WITHIN the grid bbox.

    Returns (px, py, area) in FULL-FRAME pixel coordinates (bbox offset
    already added back), or None if no green cell is found inside the grid.
    """
    x0, y0, x1, y1 = grid_bbox
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, green_hex)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    px = m["m10"] / m["m00"] + x0
    py = m["m01"] / m["m00"] + y0
    return px, py, area


def find_object_outside_grid(frame, grid_bbox: tuple[int, int, int, int], color_hex: str,
                              min_area: float = 300.0) -> tuple[float, float, float] | None:
    """Find the largest color_hex blob OUTSIDE the grid bbox (the object to pick).

    Same color as the target cell can appear both on the table (the
    object) and inside the grid (the target cell) -- this filters to only
    blobs whose centroid falls OUTSIDE grid_bbox, so the two never get
    confused regardless of which one happens to be larger/first-found.
    """
    x0, y0, x1, y1 = grid_bbox

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, color_hex)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = sorted(
        ((c, cv2.contourArea(c)) for c in contours), key=lambda t: t[1], reverse=True
    )

    for c, area in candidates:
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        px = m["m10"] / m["m00"]
        py = m["m01"] / m["m00"]
        if x0 <= px <= x1 and y0 <= py <= y1:
            continue  # this blob is the grid cell, not the object -- skip
        return px, py, area

    return None
