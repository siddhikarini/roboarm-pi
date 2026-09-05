"""Pixel-position -> arm-mm transform, using a homography (not plain affine).

Standard robotics technique: this is "eye-to-hand calibration" for a FIXED
external camera + objects on a single known plane (the table, at grip_z
height). Because every duck sits at that one fixed height, we only need the
pixel->arm mapping to be accurate AT that height — not for arbitrary height,
which is why calibration deliberately holds the marker at grip_z (see
calibrate_pixel_to_arm_manual.py).

A homography (projective transform, cv2.findHomography) is the mathematically
correct model for "pixels on a camera viewing a flat plane" — it properly
handles the camera's viewing angle (confirmed real parallax via
test_parallax.py: ~17px shift over 42mm of height change at one x/y). Our
earlier plain affine fit (arm = a*px + b*py + c, one constant scale
everywhere) can't represent that; a homography can.

Correspondences come from placing a marker at KNOWN arm positions (e.g.
cell.yaml grid cells, via grid_to_arm, or ground-truth hand-positioned arm
feedback) and recording where it shows up in the camera frame at grip_z —
see calibrate_pixel_to_arm_manual.py. No arm movement needed beyond that, no
separate marker-corner counting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import cv2

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "pixel_arm_transform.json"


def fit_transform(correspondences: list[tuple[float, float, float, float]]) -> dict:
    """Fit a homography mapping pixel (px, py) -> arm (x, y) at grip_z height.

    correspondences: list of (pixel_x, pixel_y, arm_x, arm_y), all measured
    with the marker at the SAME height (grip_z) — a homography models one
    flat plane correctly; mixing heights breaks that assumption.

    Needs at least 4 non-collinear points (a homography has 8 degrees of
    freedom). Uses cv2.findHomography with RANSAC, so extra points beyond 4
    both improve accuracy (least-squares refinement) and let it detect any
    single wildly-off outlier point automatically.
    """
    if len(correspondences) < 4:
        raise ValueError(f"need at least 4 correspondences for a homography, got {len(correspondences)}")

    src = np.array([[c[0], c[1]] for c in correspondences], dtype=np.float32)
    dst = np.array([[c[2], c[3]] for c in correspondences], dtype=np.float32)

    method = cv2.RANSAC if len(correspondences) >= 5 else 0
    H, mask = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=8.0)
    if H is None:
        raise RuntimeError(
            "cv2.findHomography failed to converge — check correspondences "
            "aren't collinear/degenerate"
        )

    return {"H": H.tolist()}


def fit_residuals(transform: dict, correspondences: list[tuple[float, float, float, float]]
                   ) -> list[float]:
    """Per-point error (mm) between the fitted homography and each true arm position."""
    errors = []
    for px, py, true_x, true_y in correspondences:
        pred_x, pred_y = apply_transform(transform, px, py)
        err = ((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2) ** 0.5
        errors.append(err)
    return errors


def apply_transform(transform: dict, px: float, py: float) -> tuple[float, float]:
    H = np.array(transform["H"], dtype=np.float64)
    pt = np.array([[[px, py]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def apply_gripper_offset(x: float, y: float, radial_mm: float = 0.0,
                          tangential_mm: float = 0.0) -> tuple[float, float]:
    """Shift an arm position along the gripper's OWN reach direction, not
    fixed world x/y.

    This arm has no wrist roll -- the gripper is rigidly attached at the
    end, so its jaw orientation always faces directly along the line from
    the base to the tool tip (the base angle, atan2(y, x)). Any mechanical
    offset between the commanded wrist point and the true grip point
    therefore ROTATES with the base angle instead of staying fixed in
    world x/y.

    Confirmed via test_pick_offset_check.py: a flat world-frame offset
    (e.g. fixed +y) worked at one board position but not another, while
    decomposing the same two measurements into radial (toward/away from
    the base) and tangential (perpendicular) components gave consistent
    signs and roughly proportional-to-radius magnitudes across both spots
    -- exactly what a fixed gripper-frame offset would produce.

    radial_mm: positive = further FROM the base (push the point outward
        along the reach direction).
    tangential_mm: positive = counter-clockwise from the radial direction
        (perpendicular, same convention as kinematics.py's atan2(y, x)).
    """
    theta = math.atan2(y, x)
    radial_dir = (math.cos(theta), math.sin(theta))
    tangential_dir = (-math.sin(theta), math.cos(theta))
    new_x = x + radial_mm * radial_dir[0] + tangential_mm * tangential_dir[0]
    new_y = y + radial_mm * radial_dir[1] + tangential_mm * tangential_dir[1]
    return new_x, new_y


def save_transform(transform: dict, frame_shape: tuple[int, int],
                    path: Path | str | None = None) -> None:
    """frame_shape: (height, width) the transform was fit at — pixel-based,
    so this MUST match at use time."""
    p = Path(path or DEFAULT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**transform, "frame_height": frame_shape[0], "frame_width": frame_shape[1]}
    with open(p, "w") as fh:
        json.dump(payload, fh, indent=2)


def load_transform(path: Path | str | None = None,
                    expected_shape: tuple[int, int] | None = None) -> dict | None:
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        return None
    with open(p) as fh:
        payload = json.load(fh)

    if expected_shape is not None:
        saved_shape = (payload.get("frame_height"), payload.get("frame_width"))
        if saved_shape != tuple(expected_shape):
            raise RuntimeError(
                f"pixel_arm_transform was fit at resolution {saved_shape[1]}x{saved_shape[0]} "
                f"but current capture is {expected_shape[1]}x{expected_shape[0]}. "
                f"Re-run calibrate_pixel_to_arm_manual.py at the current resolution."
            )

    return {"H": payload["H"]}
