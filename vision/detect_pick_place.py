"""Pure DETECTION: find the pick position (object) and place position
(target grid cell), no arm/motion code at all.

This is the piece that runs on the Raspberry Pi's side of the real
architecture (see docs/AGENT_MQTT_FLOW.md): the Pi owns the camera and
color detection locally, and only reports back arm-frame coordinates --
never raw pixels, never a motion plan. What to DO with those coordinates
(the move sequence) is a separate concern, handled by execute_pick_place.py
locally on the Pi, driven by move-by-move instructions relayed from the
MCP/EC2 side over MQTT.

Detects:
  - the pink grid's bounding box
  - the target cell (green background) INSIDE the grid -> place position
  - the object to pick (same color family) OUTSIDE the grid -> pick position

Both positions are converted through the SAME pixel_arm_transform (both
sit on the same table plane).

Usage (standalone diagnostic):
    python vision/detect_pick_place.py --transform config/pixel_arm_transform.json --color green --cell-color "#38846f"
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from .board_crop import capture_board_frame
from .grid_place_detect import (
    PINK_GRID_HEX_DEFAULT,
    find_green_cell,
    find_object_outside_grid,
    find_pink_grid_bbox,
)
from .pixel_arm_transform import apply_transform, load_transform


@dataclass
class DetectionResult:
    pick_x: float
    pick_y: float
    place_x: float
    place_y: float
    pick_pixel: tuple[float, float]
    place_pixel: tuple[float, float]
    grid_bbox: tuple[int, int, int, int]


def detect_pick_and_place(
    *,
    transform_path: str,
    object_color_hex: str,
    cell_color_hex: str,
    camera_index: int = 1,
    pink_hex: str = PINK_GRID_HEX_DEFAULT,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    save_debug: bool = True,
) -> DetectionResult:
    """Capture a frame and detect both the pick and place arm positions.

    Raises RuntimeError with a descriptive message if the grid, target
    cell, or object aren't found -- no silent fallback, since a wrong
    guess here means a bad physical move downstream.
    """
    frame = capture_board_frame(camera_index)
    if save_debug:
        cv2.imwrite("detect_raw_frame.jpg", frame)

    transform = load_transform(path=transform_path, expected_shape=frame.shape[:2])
    if transform is None:
        raise RuntimeError(f"No transform found at {transform_path}")

    grid_bbox = find_pink_grid_bbox(frame, pink_hex)
    if grid_bbox is None:
        if save_debug:
            from .color_detect import build_mask
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = build_mask(hsv, pink_hex, hue_tol=8, sat_min=70, val_min=50)
            cv2.imwrite("detect_pink_mask_debug.jpg", mask)
        raise RuntimeError(
            "pink grid not detected. Check detect_raw_frame.jpg and "
            "detect_pink_mask_debug.jpg."
        )

    cell_found = find_green_cell(frame, grid_bbox, cell_color_hex)
    if cell_found is None:
        raise RuntimeError(
            f"no cell found inside the pink grid matching {cell_color_hex}."
        )
    cell_px, cell_py, cell_area = cell_found

    obj_found = find_object_outside_grid(frame, grid_bbox, object_color_hex)
    if obj_found is None:
        raise RuntimeError(
            f"no object found matching {object_color_hex} OUTSIDE the pink grid."
        )
    obj_px, obj_py, obj_area = obj_found

    if save_debug:
        gx0, gy0, gx1, gy1 = grid_bbox
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (gx0, gy0), (gx1, gy1), (255, 0, 255), 2)
        cv2.drawMarker(debug_frame, (int(cell_px), int(cell_py)), (0, 255, 0),
                        cv2.MARKER_CROSS, 30, 3)
        cv2.drawMarker(debug_frame, (int(obj_px), int(obj_py)), (0, 0, 255),
                        cv2.MARKER_CROSS, 30, 3)
        cv2.imwrite("detect_debug.jpg", debug_frame)

    pick_x, pick_y = apply_transform(transform, obj_px, obj_py)
    pick_x, pick_y = pick_x + offset_x, pick_y + offset_y
    place_x, place_y = apply_transform(transform, cell_px, cell_py)
    place_x, place_y = place_x + offset_x, place_y + offset_y

    return DetectionResult(
        pick_x=pick_x, pick_y=pick_y,
        place_x=place_x, place_y=place_y,
        pick_pixel=(obj_px, obj_py), place_pixel=(cell_px, cell_py),
        grid_bbox=grid_bbox,
    )


def main() -> None:
    import argparse

    from sim.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", required=True)
    ap.add_argument("--color", default="green")
    ap.add_argument("--cell-color", required=True)
    ap.add_argument("--pink-hex", default=PINK_GRID_HEX_DEFAULT)
    ap.add_argument("--index", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    duck = cfg.blocks.get(args.color)
    if duck is None:
        raise SystemExit(f"unknown color '{args.color}'. Options: {list(cfg.blocks)}")

    offset_x = float(cfg.vision.get("pixel_arm_offset_x", 0.0))
    offset_y = float(cfg.vision.get("pixel_arm_offset_y", 0.0))

    result = detect_pick_and_place(
        transform_path=args.transform,
        object_color_hex=duck.color,
        cell_color_hex=args.cell_color,
        camera_index=args.index,
        pink_hex=args.pink_hex,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    print(f"pick:  ({result.pick_x:.1f}, {result.pick_y:.1f})  pixel={result.pick_pixel}")
    print(f"place: ({result.place_x:.1f}, {result.place_y:.1f})  pixel={result.place_pixel}")
    print(f"grid bbox: {result.grid_bbox}")
    print("saved detect_raw_frame.jpg / detect_debug.jpg")


if __name__ == "__main__":
    main()
