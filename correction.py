"""Position-based pick/place correction utilities.

Loads pick and place settle maps from the calibration run folder and
applies corrections to commanded arm coordinates before execution.

Used by both run_execute.py (local) and laptop_agent.py (MQTT/Pi).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)


def load_settle_maps(run_dir: Path) -> tuple[list, dict | None, list, dict | None, dict]:
    """Load pick and place settle maps from a run folder.

    Returns:
        pick_points, pick_model, place_points, place_model, place_cell_offsets
    """
    pick_points:  list = []
    pick_model:   dict | None = None
    place_points: list = []
    place_model:  dict | None = None
    place_cell_offsets: dict = {}

    pick_path  = run_dir / "pick_settle_map.json"
    place_path = run_dir / "place_settle_map.json"

    if pick_path.exists():
        data = json.loads(pick_path.read_text())
        pick_points = data.get("points", [])
        pick_model  = data.get("model") or None
        if pick_model:
            log.info("Pick settle map: %d pts, model=%s RMSE=%.1fmm",
                     len(pick_points), pick_model["type"], pick_model.get("rmse_mm", 0))
        else:
            log.info("Pick settle map: %d pts (IDW)", len(pick_points))
    else:
        log.warning("No pick_settle_map.json in %s -- no pick corrections", run_dir)

    if place_path.exists():
        data = json.loads(place_path.read_text())
        place_points       = data.get("points", [])
        place_model        = data.get("model") or None
        place_cell_offsets = data.get("cell_offsets", {})
        if place_model:
            log.info("Place settle map: %d pts, model=%s RMSE=%.1fmm",
                     len(place_points), place_model["type"], place_model.get("rmse_mm", 0))
        else:
            log.info("Place settle map: %d pts (IDW), cell_offsets=%s",
                     len(place_points), list(place_cell_offsets.keys()))
    else:
        log.warning("No place_settle_map.json in %s -- no place corrections", run_dir)

    return pick_points, pick_model, place_points, place_model, place_cell_offsets


def get_correction(x: float, y: float, points: list, model: dict | None) -> tuple[float, float]:
    """Return (dx, dy) correction to ADD to the commanded position.

    correction = -error  (if arm overshoots by +10mm, command -10mm less)
    """
    if model:
        t = model.get("type")
        c = model["coeffs"]
        if t == "poly2d":
            ey = (c["a"] + c["b"]*x + c["c"]*y + c["d"]*x*y
                  + c["e"]*x*x + c["f"]*y*y)
            return 0.0, -ey
        if t == "linear_x":
            ey = c["a"] + c["b"]*x
            return 0.0, -ey

    if not points:
        return 0.0, 0.0

    # IDW from nearest 3 points (cube weighting to avoid cross-region contamination)
    dists = sorted([(math.hypot(x - p["target_x"], y - p["target_y"]), p) for p in points])
    nearest = dists[:3]
    if nearest[0][0] < 5.0:
        p = nearest[0][1]
        return -p.get("error_x", 0.0), -p["error_y"]
    total_w = dx_sum = dy_sum = 0.0
    for d, p in nearest:
        w = 1.0 / (d * d * d)
        dx_sum += w * (-p.get("error_x", 0.0))
        dy_sum += w * (-p["error_y"])
        total_w += w
    return dx_sum / total_w, dy_sum / total_w


def get_place_correction(place_x: float, place_cell_offsets: dict) -> tuple[float, float]:
    """Return (dx, dy) place correction based on x-cluster (cell position).

    A1 cluster: x > 320
    B1 cluster: 220 < x <= 320
    C1 cluster: x <= 220
    """
    if place_x > 320:
        cell_key = "A1"
    elif place_x > 220:
        cell_key = "B1"
    else:
        cell_key = "C1"
    dy = place_cell_offsets.get(cell_key, {}).get("dy", 0.0)
    return 0.0, dy


def apply_corrections(
    pick_x: float, pick_y: float,
    place_x: float, place_y: float,
    pick_points: list, pick_model: dict | None,
    place_cell_offsets: dict,
) -> tuple[float, float, float, float]:
    """Apply pick and place corrections, return corrected (pick_x, pick_y, place_x, place_y)."""
    pick_dx,  pick_dy  = get_correction(pick_x, pick_y, pick_points, pick_model)
    place_dx, place_dy = get_place_correction(place_x, place_cell_offsets)

    return (
        pick_x  + pick_dx,
        pick_y  + pick_dy,
        place_x + place_dx,
        place_y + place_dy,
    )
