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


def find_black_grid_bbox(frame, dark_threshold: int = 60,
                          min_area_frac: float = 0.4) -> tuple[int, int, int, int] | None:
    """Find a BLACK-bordered grid's bounding box (replaces the pink-hue
    version above for a board using a black border instead).

    Black has near-zero saturation, so hue-based matching (build_mask,
    used by find_pink_grid_bbox) can never reliably find it -- same reason
    color_detect.find_black_marker() exists as a brightness-threshold
    alternative to hue matching. This applies that same technique to the
    grid border specifically, keeping find_pink_grid_bbox's other
    safeguards (edge-margin exclusion, min_area_frac fragment filtering,
    frame-coverage sanity check) since a black border line is just as
    prone to breaking into disconnected fragments (dilate needed) and just
    as vulnerable to a thin dark artifact at the frame edge merging in.
    """
    frame_h, frame_w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    # Same edge-artifact guard as find_pink_grid_bbox: a real grid border
    # drawn on the board interior shouldn't touch the frame's outer edge,
    # so zero out a thin margin BEFORE dilating to prevent an edge artifact
    # (shadow, vignette, cable) from fusing into the real border.
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

    if w * h > 0.85 * frame_w * frame_h:
        log.warning(
            "black grid bbox (%d,%d)-(%d,%d) covers %.0f%% of the frame -- "
            "likely a false-positive match (threshold too loose, picking up "
            "shadows/dark background), not a real grid border. Try raising "
            "dark_threshold.",
            x, y, x + w, y + h, 100.0 * w * h / (frame_w * frame_h),
        )
        return None

    return x, y, x + w, y + h


def find_white_grid_bbox(frame, bright_threshold: int = 200,
                          min_area_frac: float = 0.4) -> tuple[int, int, int, int] | None:
    """Find a WHITE-bordered grid's bounding box.

    White has near-zero saturation and very high brightness, so hue-based
    matching can't find it reliably -- use brightness thresholding from
    ABOVE (pixels brighter than bright_threshold) instead of from below
    like find_black_grid_bbox. Same edge-margin, fragment-filter, and
    frame-coverage safeguards as the black/pink versions.
    """
    frame_h, frame_w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, mask = cv2.threshold(gray, bright_threshold, 255, cv2.THRESH_BINARY)

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

    if w * h > 0.85 * frame_w * frame_h:
        log.warning(
            "white grid bbox (%d,%d)-(%d,%d) covers %.0f%% of the frame -- "
            "likely a false-positive (threshold too loose, picking up bright "
            "background/lighting). Try raising bright_threshold.",
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
                              min_area: float = 2000.0) -> tuple[float, float, float] | None:
    """Find the largest color_hex blob OUTSIDE the grid bbox (the object to pick).

    Same color as the target cell can appear both on the table (the
    object) and inside the grid (the target cell) -- this filters to only
    blobs whose centroid falls OUTSIDE grid_bbox, so the two never get
    confused regardless of which one happens to be larger/first-found.
    """
    found = find_objects_outside_grid(frame, grid_bbox, color_hex, min_area=min_area)
    return found[0] if found else None


def find_objects_outside_grid(frame, grid_bbox: tuple[int, int, int, int], color_hex: str,
                               min_area: float = 2000.0) -> list[tuple[float, float, float]]:
    """Find EVERY color_hex blob OUTSIDE the grid bbox (plural version of
    find_object_outside_grid), sorted largest-area first.

    Needed for a multi-object kit plan where the SAME color can appear on
    more than one physical object at once (e.g. two red cubes on the mat).
    Since same-colored objects are functionally interchangeable for a pick
    (matches the physical setup: "either red cube satisfies it" -- there's
    no need to distinguish WHICH one is which, only to find all current
    candidates of that color so a caller can pick from the list and remove
    it before the next detection call).

    Returns a list of (px, py, area) tuples, one per detected blob, largest
    first. Empty list if none found (never raises -- let the caller decide
    if zero matches is an error for their use case).
    """
    x0, y0, x1, y1 = grid_bbox

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, color_hex)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    candidates = sorted(
        ((c, cv2.contourArea(c)) for c in contours), key=lambda t: t[1], reverse=True
    )

    results = []
    h, w = frame.shape[:2]
    edge_margin = 120  # ignore blobs whose centroid is within 120px of frame edge
    for c, area in candidates:
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        px = m["m10"] / m["m00"]
        py = m["m01"] / m["m00"]
        if x0 <= px <= x1 and y0 <= py <= y1:
            continue  # inside grid bbox -- skip
        if px < edge_margin or px > w - edge_margin or py < edge_margin or py > h - edge_margin:
            continue  # too close to frame edge -- likely border noise
        results.append((px, py, area))

    return results


def find_cell_centers_from_lines(frame, grid_bbox: tuple[int, int, int, int],
                                   pink_hex: str, rows: int, cols: int,
                                   hue_tol: int = 12, sat_min: int = 60, val_min: int = 40
                                   ) -> dict[tuple[int, int], tuple[float, float]] | None:
    """Detect cell centers from the ACTUAL pink lines drawn inside the grid,
    rather than evenly dividing the bounding box.

    Uses projection profiles on the pink color mask within the grid bbox:
    - Horizontal projection (sum along columns) → finds row dividers
    - Vertical projection (sum along rows) → finds column dividers

    Returns {(row_idx, col_idx): (px, py)} in full-frame pixel coordinates,
    or None if the line structure can't be reliably detected (falls back to
    even division in the caller).

    row_idx/col_idx are 0-indexed, same convention as grid_cell_pixel_centers.
    """
    x0, y0, x1, y1 = grid_bbox
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, pink_hex, hue_tol=hue_tol, sat_min=sat_min, val_min=val_min)

    # Projection profiles -- sum each axis to find where the lines are
    h_profile = mask.sum(axis=1).astype(float)  # sum along columns -> one value per row
    v_profile = mask.sum(axis=0).astype(float)  # sum along rows -> one value per col

    def _find_dividers(profile, n_dividers):
        """Find n_dividers internal peaks in a projection profile, excluding
        the outer border which appears as strong peaks near the bbox edges."""
        if n_dividers == 0:
            return []
        n = len(profile)
        # Exclude the outermost 10% from consideration -- those are the bbox
        # border itself, not internal dividers. We only want internal lines.
        margin = max(3, n // 10)
        threshold = float(profile.max()) * 0.2
        min_dist = max(1, n // (n_dividers + 2))

        # Simple peak detection, ignoring edge margins
        peaks = []
        for i in range(margin, n - margin):
            if (profile[i] >= threshold and
                    profile[i] >= profile[max(0,i-1)] and
                    profile[i] >= profile[min(n-1,i+1)]):
                peaks.append(i)

        if not peaks:
            return None

        # Non-maximum suppression
        merged = []
        peaks_sorted = sorted(peaks, key=lambda p: -float(profile[p]))
        used = set()
        for p in peaks_sorted:
            if any(abs(p - m) < min_dist for m in used):
                continue
            used.add(p)
            merged.append(p)
        merged.sort()

        if len(merged) < n_dividers:
            return None
        # Take the n_dividers strongest peaks
        top = sorted(merged, key=lambda p: -float(profile[p]))[:n_dividers]
        return sorted(top)

    # For rows cells we need (rows-1) internal dividers + 2 boundaries (top/bottom)
    # For a 3-row grid: 2 internal horizontal lines
    h_dividers = _find_dividers(h_profile, rows - 1)
    v_dividers = _find_dividers(v_profile, cols - 1)

    crop_h, crop_w = crop.shape[:2]

    # Build cell boundary lists (including outer edges)
    if h_dividers is not None:
        h_boundaries = [0] + list(h_dividers) + [crop_h]
    else:
        # Fall back to even division
        h_boundaries = [int(i * crop_h / rows) for i in range(rows + 1)]

    if v_dividers is not None:
        v_boundaries = [0] + list(v_dividers) + [crop_w]
    else:
        v_boundaries = [int(i * crop_w / cols) for i in range(cols + 1)]

    if len(h_boundaries) != rows + 1 or len(v_boundaries) != cols + 1:
        return None

    centers = {}
    for r in range(rows):
        for c in range(cols):
            py = (h_boundaries[r] + h_boundaries[r + 1]) / 2.0 + y0
            px = (v_boundaries[c] + v_boundaries[c + 1]) / 2.0 + x0
            centers[(r, c)] = (px, py)

    return centers


def grid_cell_pixel_centers(grid_bbox: tuple[int, int, int, int], rows: int, cols: int
                             ) -> dict[tuple[int, int], tuple[float, float]]:
    """Pixel-space center of every (row_idx, col_idx) cell in an evenly-
    divided rows x cols grid within grid_bbox.

    row_idx: 0-indexed, TOP-to-BOTTOM in pixel space (0 = top row in the
    image, not yet mapped to a physical near/far arm label -- see
    resolve_named_cells for that mapping).
    col_idx: 0-indexed, LEFT-to-RIGHT in pixel space.

    Returns {(row_idx, col_idx): (px, py)} for every cell.
    """
    x0, y0, x1, y1 = grid_bbox
    cell_w = (x1 - x0) / cols
    cell_h = (y1 - y0) / rows

    centers = {}
    for r in range(rows):
        for c in range(cols):
            px = x0 + (c + 0.5) * cell_w
            py = y0 + (r + 0.5) * cell_h
            centers[(r, c)] = (px, py)
    return centers


def resolve_named_cells(grid_bbox: tuple[int, int, int, int], rows: int, cols: int,
                         transform: dict, row_labels: str = "ABCD",
                         frame=None, pink_hex: str | None = None,
                         ) -> dict[str, tuple[float, float]]:
    """Map cell names (e.g. "A1", "B2") to ARM-frame (x, y) coordinates.

    If frame and pink_hex are provided, tries to detect the actual cell
    boundaries from the pink lines drawn inside the grid (using
    find_cell_centers_from_lines) before falling back to evenly dividing
    the bounding box. This gives more accurate cell centers when the
    physical grid has clearly drawn internal lines.

    Row naming is resolved by ARM DISTANCE -- row_labels[0] ("A") is
    the row FARTHEST from the arm. Column naming uses pixel LEFT-to-right
    order (col 0 -> "1"). Verify column direction matches your physical
    setup and flip if needed.
    """
    if rows > len(row_labels):
        raise ValueError(f"rows={rows} exceeds available row_labels {row_labels!r}")

    from .pixel_arm_transform import apply_transform

    # Try line-based cell center detection first if frame is provided
    centers = None
    if frame is not None and pink_hex is not None:
        centers = find_cell_centers_from_lines(frame, grid_bbox, pink_hex, rows, cols)
        if centers is not None:
            log.info("resolve_named_cells: using line-based cell centers (%dx%d)", rows, cols)
        else:
            log.warning("resolve_named_cells: line detection failed, falling back to even division")

    if centers is None:
        centers = grid_cell_pixel_centers(grid_bbox, rows, cols)

    # Compute each PIXEL ROW's average arm-frame radius (distance from the
    # arm's base), using the middle column as a representative sample --
    # all cells in one pixel row should be roughly equidistant from the
    # arm along that axis for a grid laid flat on one plane.
    row_radius: dict[int, float] = {}
    for r in range(rows):
        radii = []
        for c in range(cols):
            px, py = centers[(r, c)]
            ax, ay = apply_transform(transform, px, py)
            radii.append((ax ** 2 + ay ** 2) ** 0.5)
        row_radius[r] = sum(radii) / len(radii)

    # Sort pixel rows by descending radius: farthest-from-arm row first.
    rows_by_distance = sorted(range(rows), key=lambda r: row_radius[r], reverse=True)

    named: dict[str, tuple[float, float]] = {}
    for label_idx, pixel_row in enumerate(rows_by_distance):
        letter = row_labels[label_idx]
        for c in range(cols):
            px, py = centers[(pixel_row, c)]
            ax, ay = apply_transform(transform, px, py)
            named[f"{letter}{c + 1}"] = (ax, ay)

    return named
    """Map cell names (e.g. "A1", "B2") to ARM-frame (x, y) coordinates.

    Row naming is resolved by ARM DISTANCE, not raw pixel position: each
    pixel-space row is transformed through the pixel_arm_transform, and
    rows are sorted by distance from the arm's base (hypot(x,y)) --
    row_labels[0] ("A") is assigned to whichever row is FARTHEST from the
    arm, matching the physical requirement that A1/A2 are the row
    farthest from the arm. This works regardless of how the camera happens
    to be mounted/rotated relative to the arm, since it's based on the
    real transformed arm coordinates, not an assumption about which pixel
    edge (top/bottom of the image) is "far".

    Column naming (1, 2, ...) is NOT similarly resolved -- there's no
    stated requirement for which physical side is "1" vs "2", so this
    uses pixel LEFT-to-right order (col_idx 0 -> "1", col_idx 1 -> "2",
    etc.) as a default. VERIFY this matches the intended physical
    layout once a live camera frame is available; flip cols_reversed=True
    below if col 1 should instead be on the physical right.

    rows/cols: grid dimensions (e.g. rows=4, cols=2 for an 8-cell A1..D2
    grid). row_labels: one character per row, in "nearest-letter-first"
    order (default "ABCD" for up to 4 rows) -- row_labels[0] always maps
    to the farthest-from-arm row by construction above, independent of
    physical camera orientation.
    """
    if rows > len(row_labels):
        raise ValueError(f"rows={rows} exceeds available row_labels {row_labels!r}")

    from .pixel_arm_transform import apply_transform

    centers = grid_cell_pixel_centers(grid_bbox, rows, cols)

    # Compute each PIXEL ROW's average arm-frame radius (distance from the
    # arm's base), using the middle column as a representative sample --
    # all cells in one pixel row should be roughly equidistant from the
    # arm along that axis for a grid laid flat on one plane.
    row_radius: dict[int, float] = {}
    for r in range(rows):
        radii = []
        for c in range(cols):
            px, py = centers[(r, c)]
            ax, ay = apply_transform(transform, px, py)
            radii.append((ax ** 2 + ay ** 2) ** 0.5)
        row_radius[r] = sum(radii) / len(radii)

    # Sort pixel rows by descending radius: farthest-from-arm row first.
    rows_by_distance = sorted(range(rows), key=lambda r: row_radius[r], reverse=True)

    named: dict[str, tuple[float, float]] = {}
    for label_idx, pixel_row in enumerate(rows_by_distance):
        letter = row_labels[label_idx]
        for c in range(cols):
            px, py = centers[(pixel_row, c)]
            ax, ay = apply_transform(transform, px, py)
            named[f"{letter}{c + 1}"] = (ax, ay)

    return named
