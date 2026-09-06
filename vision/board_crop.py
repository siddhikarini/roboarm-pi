"""Crop captured frames down to just the board, using a drawn red boundary line.

Background clutter (table edges, tiles, cables) outside the board can
confuse detection and waste pixels. A physical red line drawn around the
board's edge lets us find and crop to just its interior automatically.

Red is ALSO a duck color, so this must not get confused with a red duck:
the boundary LINE forms a much larger enclosed contour (board-sized) than
any duck (duck-sized) — picking the largest red contour naturally
distinguishes them by scale. Cropping to the INTERIOR with a margin then
excludes the red line itself from the cropped output, so it can never be
mistaken for a red duck in any frame captured afterward.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import cv2

from .color_detect import build_mask

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "board_crop.json"


def detect_board_rect(frame, red_hex: str = "#ff0000", hue_tol: int = 15,
                       sat_min: int = 60, val_min: int = 40, inset_px: int = 0
                       ) -> tuple[int, int, int, int]:
    """Find the drawn red boundary and return an interior crop rect.

    Returns (x0, y0, x1, y1) — the boundary's bounding box, inset inward by
    inset_px on each side. Only set inset_px > 0 if a red duck is ever in
    play (to exclude the red line itself from being confused with it) —
    with inset_px=0, the crop runs right up to the drawn line, keeping the
    full board.

    hue_tol/sat_min/val_min are wider than color_detect's duck defaults
    since a marker pen's red may be a different shade than the duck's; only
    the LARGEST matching contour is used (the boundary line, which encloses
    a board-sized area — far bigger than any duck), so a looser color match
    here is safe.

    Raises RuntimeError if no sufficiently large red contour is found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = build_mask(hsv, red_hex, hue_tol=hue_tol, sat_min=sat_min, val_min=val_min)
    # A hand-drawn line often breaks into several disconnected arcs rather
    # than one closed loop (confirmed: this camera's capture split the line
    # into 2 large fragments, ~4500 and ~4000 px^2 — using only the single
    # largest cut off part of the true boundary, leaving a visible gap
    # between the detected crop and the real line). Dilate to bridge small
    # gaps AND combine every sufficiently-large fragment's bounding box
    # (not just the biggest one) so the full drawn loop is captured.
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(
            "no red boundary detected — check the line is visible/red enough, "
            "or adjust hue_tol/sat_min/val_min"
        )

    # Combine all fragments with meaningful area (not tiny noise specks) into
    # one bounding box, instead of trusting a single "largest" contour.
    # NOTE: 0.05 was too permissive -- confirmed via grid_place_detect.py's
    # find_pink_grid_bbox hitting the exact same failure mode (a small,
    # unrelated red/pink-hue fragment elsewhere in frame merging into the
    # combined bbox and dragging it out past the real boundary on one
    # side). Raised to 0.4 to only keep fragments genuinely comparable in
    # size to the largest -- i.e. actually part of the same drawn line.
    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas)
    significant = [c for c, a in zip(contours, areas) if a >= max_area * 0.4]

    all_points = np.vstack(significant)
    x, y, w, h = cv2.boundingRect(all_points)

    h_frame, w_frame = frame.shape[:2]
    x0 = max(0, x + inset_px)
    y0 = max(0, y + inset_px)
    x1 = min(w_frame, x + w - inset_px)
    y1 = min(h_frame, y + h - inset_px)

    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(
            f"detected boundary too small after inset ({inset_px}px) to form a "
            f"valid crop — reduce inset_px or check detection"
        )
    return x0, y0, x1, y1


def apply_crop(frame, crop_rect: tuple[int, int, int, int]):
    x0, y0, x1, y1 = crop_rect
    return frame[y0:y1, x0:x1]


def save_crop(crop_rect: tuple[int, int, int, int], frame_shape: tuple[int, int],
              path: Path | str | None = None) -> None:
    p = Path(path or DEFAULT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "crop_rect": list(crop_rect),
        "frame_height": frame_shape[0],
        "frame_width": frame_shape[1],
    }
    with open(p, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_crop(path: Path | str | None = None,
              expected_shape: tuple[int, int] | None = None) -> tuple[int, int, int, int] | None:
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        return None
    with open(p) as fh:
        payload = json.load(fh)

    if expected_shape is not None:
        saved_shape = (payload.get("frame_height"), payload.get("frame_width"))
        if saved_shape != tuple(expected_shape):
            raise RuntimeError(
                f"board_crop was fit at resolution {saved_shape[1]}x{saved_shape[0]} "
                f"but current capture is {expected_shape[1]}x{expected_shape[0]}. "
                f"Re-run calibrate_board_crop.py at the current resolution."
            )
    return tuple(payload["crop_rect"])


def capture_board_frame(index: int | str):
    """Capture a frame and crop it to the board using the saved calibration.

    index: int device index OR a stream URL string (see capture.py).

    Raises if no board_crop calibration exists yet — run
    calibrate_board_crop.py once first.
    """
    from .capture import capture_frame

    frame = capture_frame(index)
    crop_rect = load_crop(expected_shape=frame.shape[:2])
    if crop_rect is None:
        raise RuntimeError(
            "no board_crop calibration found; run calibrate_board_crop.py once for this camera"
        )
    return apply_crop(frame, crop_rect)
