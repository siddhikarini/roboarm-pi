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
    find_black_grid_bbox,
    find_green_cell,
    find_object_outside_grid,
    find_objects_outside_grid,
    find_pink_grid_bbox,
    find_white_grid_bbox,
    resolve_named_cells,
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


@dataclass
class KitPlacement:
    """One resolved pick/place pair for a multi-object kit plan."""
    color: str
    cell: str          # e.g. "A1"
    pick_x: float
    pick_y: float
    place_x: float
    place_y: float


def detect_kit_plan(
    *,
    transform_path: str,
    placements: list[tuple[str, str]],
    color_hex_map: dict[str, str],
    grid_rows: int,
    grid_cols: int,
    camera_index: int | str = 1,
    pink_hex: str = PINK_GRID_HEX_DEFAULT,
    grid_border: str = "black",
    dark_threshold: int = 60,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    save_debug: bool = True,
) -> list[KitPlacement]:
    """Resolve pick/place coordinates for a WHOLE multi-object plan from a
    SINGLE camera capture ("stow once" approach), instead of one detection
    call per placement.

    placements: [(color, cell_name), ...] in the order they should be
    executed, e.g. [("red", "A1"), ("red", "A2"), ("blue", "B1"), ...].
    color_hex_map: {color_name: hex_color}, e.g. cfg.blocks colors.

    Same color can appear on MULTIPLE placements (e.g. two "red" entries)
    -- since same-colored objects are functionally interchangeable (there's
    no requirement to track WHICH physical red cube goes where, only that
    a red cube goes to each required cell), each placement of a given
    color consumes the NEXT-largest still-unused detected blob of that
    color from this single capture. This mirrors find_objects_outside_grid
    returning multiple candidates precisely for this reason.

    Raises RuntimeError if the grid isn't found, or if there are more
    placements requesting a color than objects of that color were
    detected in the frame -- no silent fallback, since guessing a pick
    position here risks a bad physical move.
    """
    frame = capture_board_frame(camera_index)
    if save_debug:
        cv2.imwrite("detect_kit_raw_frame.jpg", frame)

    transform = load_transform(path=transform_path, expected_shape=frame.shape[:2])
    if transform is None:
        raise RuntimeError(f"No transform found at {transform_path}")

    if grid_border == "black":
        grid_bbox = find_black_grid_bbox(frame, dark_threshold=dark_threshold)
        border_desc = "black grid border"
    elif grid_border == "white":
        grid_bbox = find_white_grid_bbox(frame)
        border_desc = "white grid border"
    else:
        grid_bbox = find_pink_grid_bbox(frame, pink_hex)
        border_desc = "pink grid"
    if grid_bbox is None:
        if save_debug:
            cv2.imwrite("detect_kit_raw_frame_failed.jpg", frame)
        raise RuntimeError(
            f"{border_desc} not detected. Check detect_kit_raw_frame.jpg."
        )

    named_cells = resolve_named_cells(
        grid_bbox, grid_rows, grid_cols, transform,
        frame=frame,
        pink_hex=pink_hex if grid_border == "pink" else None,
    )

    # Detect every candidate object per DISTINCT color used in this plan,
    # once each (not once per placement) -- e.g. if 3 placements all want
    # "red", we still only run find_objects_outside_grid("red") ONE time
    # and pull 3 blobs off that single result list.
    distinct_colors = {color for color, _ in placements}
    candidates_by_color: dict[str, list[tuple[float, float, float]]] = {}
    for color in distinct_colors:
        hex_color = color_hex_map.get(color)
        if hex_color is None:
            raise RuntimeError(f"unknown color '{color}'. Options: {list(color_hex_map)}")
        candidates_by_color[color] = find_objects_outside_grid(frame, grid_bbox, hex_color)

    resolved: list[KitPlacement] = []
    debug_frame = frame.copy() if save_debug else None
    for color, cell_name in placements:
        if cell_name not in named_cells:
            raise RuntimeError(
                f"unknown cell '{cell_name}'. Options: {sorted(named_cells)}"
            )
        remaining = candidates_by_color[color]
        if not remaining:
            raise RuntimeError(
                f"placement needs an object of color '{color}' for cell "
                f"'{cell_name}', but no more unused '{color}' objects were "
                f"detected outside the grid (already consumed by earlier "
                f"placements in this same plan, or fewer objects on the mat "
                f"than requested placements)."
            )
        obj_px, obj_py, obj_area = remaining.pop(0)  # largest remaining first

        pick_x, pick_y = apply_transform(transform, obj_px, obj_py)
        pick_x, pick_y = pick_x + offset_x, pick_y + offset_y
        place_x, place_y = named_cells[cell_name]
        place_x, place_y = place_x + offset_x, place_y + offset_y

        resolved.append(KitPlacement(
            color=color, cell=cell_name,
            pick_x=pick_x, pick_y=pick_y,
            place_x=place_x, place_y=place_y,
        ))

        if debug_frame is not None:
            cv2.drawMarker(debug_frame, (int(obj_px), int(obj_py)), (0, 0, 255),
                            cv2.MARKER_CROSS, 30, 3)
            cv2.putText(debug_frame, cell_name, (int(obj_px) + 10, int(obj_py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if debug_frame is not None:
        gx0, gy0, gx1, gy1 = grid_bbox
        cv2.rectangle(debug_frame, (gx0, gy0), (gx1, gy1), (255, 0, 255), 2)
        cv2.imwrite("detect_kit_debug.jpg", debug_frame)

    return resolved


def detect_pick_and_place(
    *,
    transform_path: str,
    object_color_hex: str,
    cell_color_hex: str,
    camera_index: int | str = 1,
    pink_hex: str = PINK_GRID_HEX_DEFAULT,
    grid_border: str = "black",
    dark_threshold: int = 60,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    save_debug: bool = True,
) -> DetectionResult:
    """Capture a frame and detect both the pick and place arm positions.

    grid_border: "black" (brightness-threshold border, current physical
    board) or "pink" (original hue-matched border) -- the board was
    physically changed from a pink grid line to a black one, and black
    has near-zero saturation so the old hue-based matching can't find it
    (same reason color_detect.find_black_marker exists as a brightness
    alternative to hue matching elsewhere in this codebase).

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

    if grid_border == "black":
        grid_bbox = find_black_grid_bbox(frame, dark_threshold=dark_threshold)
        if grid_bbox is None:
            if save_debug:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, dark_threshold, 255, cv2.THRESH_BINARY_INV)
                cv2.imwrite("detect_black_mask_debug.jpg", mask)
            raise RuntimeError(
                "black grid border not detected. Check detect_raw_frame.jpg "
                "and detect_black_mask_debug.jpg."
            )
    elif grid_border == "white":
        grid_bbox = find_white_grid_bbox(frame)
        if grid_bbox is None:
            if save_debug:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                cv2.imwrite("detect_white_mask_debug.jpg", mask)
            raise RuntimeError(
                "white grid border not detected. Check detect_raw_frame.jpg "
                "and detect_white_mask_debug.jpg."
            )
    else:
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
    ap.add_argument("--grid-border", choices=["black", "pink"], default="black")
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
        grid_border=args.grid_border,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    print(f"pick:  ({result.pick_x:.1f}, {result.pick_y:.1f})  pixel={result.pick_pixel}")
    print(f"place: ({result.place_x:.1f}, {result.place_y:.1f})  pixel={result.place_pixel}")
    print(f"grid bbox: {result.grid_bbox}")
    print("saved detect_raw_frame.jpg / detect_debug.jpg")


if __name__ == "__main__":
    main()
