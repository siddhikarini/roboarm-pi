"""Config loading for the block sorting cell."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .kinematics import Joints, Kinematics

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "cell.yaml"


@dataclass(frozen=True)
class Block:
    id: str
    color: str
    label: str
    start_x: float
    start_y: float


@dataclass(frozen=True)
class Slot:
    id: str
    label: str
    x: float
    y: float


@dataclass(frozen=True)
class StagingSpot:
    id: str
    label: str
    x: float
    y: float


@dataclass(frozen=True)
class GridTransform:
    """Linear map from grid (col, row) to the arm's XYZ frame (mm).

        arm_x = ax_col*col + ax_row*row + ax_c
        arm_y = ay_col*col + ay_row*row + ay_c

    Coefficients are fit empirically from measured points (see cell.yaml
    grid.transform). Positions in cell.yaml are given in grid coordinates and
    converted to arm mm at load time.
    """
    ax_col: float
    ax_row: float
    ax_c: float
    ay_col: float
    ay_row: float
    ay_c: float

    def to_arm(self, col: float, row: float) -> tuple[float, float]:
        ax = self.ax_col * col + self.ax_row * row + self.ax_c
        ay = self.ay_col * col + self.ay_row * row + self.ay_c
        return ax, ay


class CellConfig:
    def __init__(self, raw: dict):
        self.raw = raw
        self.site = raw["site"]
        self.units = raw["units"]
        self.arm = raw["arm"]
        self.connection = raw.get("connection", {})
        self.limits = raw["limits"]
        self.envelope = raw["envelope"]
        self.gripper = raw["gripper"]
        self.heights = raw["heights"]
        self.vision = raw["vision"]

        self.kin = Kinematics(raw["arm"], raw["envelope"])

        # Optional grid transform (grid col/row -> arm XYZ mm).
        self.grid: GridTransform | None = None
        grid_cfg = raw.get("grid")
        if grid_cfg and "transform" in grid_cfg:
            t = grid_cfg["transform"]
            self.grid = GridTransform(
                ax_col=float(t["ax_col"]), ax_row=float(t["ax_row"]), ax_c=float(t["ax_c"]),
                ay_col=float(t["ay_col"]), ay_row=float(t["ay_row"]), ay_c=float(t["ay_c"]),
            )

        # Parse blocks. Positions may be given as grid {col,row} or raw {x,y}.
        block_cfg = raw["blocks"]
        self.block_size = float(block_cfg["size"])
        self.blocks: dict[str, Block] = {}
        for color_def in block_cfg["colors"]:
            bid = color_def["id"]
            start = block_cfg["start_positions"][bid]
            sx, sy = self._resolve_xy(start)
            self.blocks[bid] = Block(
                id=bid,
                color=color_def["color"],
                label=color_def["label"],
                start_x=sx,
                start_y=sy,
            )

        # Parse slots
        self.slots: dict[str, Slot] = {}
        for s in raw["slots"]:
            sx, sy = self._resolve_xy(s)
            self.slots[s["id"]] = Slot(
                id=s["id"],
                label=s["label"],
                x=sx,
                y=sy,
            )

        # Parse staging positions (temporary safe locations for re-sorting)
        self.staging: dict[str, StagingSpot] = {}
        for s in raw.get("staging", []):
            sx, sy = self._resolve_xy(s)
            self.staging[s["id"]] = StagingSpot(
                id=s["id"],
                label=s["label"],
                x=sx,
                y=sy,
            )

        # Home joints
        self.home_joints = Joints.from_dict(raw["home"]["joints"])

        # Stow position (retracted pose before camera captures), if configured.
        self.stow_xyz: tuple[float, float, float] | None = None
        stow_cfg = raw.get("stow")
        if stow_cfg and "xyz" in stow_cfg:
            sx = stow_cfg["xyz"]
            self.stow_xyz = (float(sx["x"]), float(sx["y"]), float(sx["z"]))

    def _resolve_xy(self, spec: dict) -> tuple[float, float]:
        """Resolve a position spec to arm XYZ mm.

        Accepts either grid coordinates {col, row} (converted via the grid
        transform) or raw arm coordinates {x, y}.
        """
        if "col" in spec and "row" in spec:
            if self.grid is None:
                raise ValueError(
                    "Position uses grid col/row but no grid.transform is defined in config."
                )
            return self.grid.to_arm(float(spec["col"]), float(spec["row"]))
        return float(spec["x"]), float(spec["y"])

    def grid_to_arm(self, col: float, row: float) -> tuple[float, float]:
        if self.grid is None:
            raise ValueError("No grid transform configured.")
        return self.grid.to_arm(col, row)

    @property
    def backend_name(self) -> str:
        return os.environ.get("CELL_BACKEND", self.raw.get("backend", "sim"))

    def sanity_check(self) -> list[str]:
        """Verify home position and all slots are reachable."""
        problems: list[str] = []

        # Check home. In passthrough mode the real home is home.xyz (validated
        # against the envelope); the joint home is only for analytic/sim.
        if self.kin.passthrough:
            hx = (self.raw.get("home", {}) or {}).get("xyz")
            if hx is None:
                problems.append("home: passthrough mode requires home.xyz")
            elif self.kin.inverse(float(hx["x"]), float(hx["y"]), float(hx["z"])) is None:
                problems.append(
                    f"home xyz ({hx['x']}, {hx['y']}, {hx['z']}) outside envelope"
                )
        else:
            for msg in self.kin.joint_violations(self.home_joints):
                problems.append(f"home: {msg}")
            for msg in self.kin.envelope_violations(self.home_joints):
                problems.append(f"home: {msg}")

        # Check stow position (retracted pose before camera captures)
        stow_cfg = self.raw.get("stow")
        if stow_cfg and "xyz" in stow_cfg:
            sx = stow_cfg["xyz"]
            if self.kin.inverse(float(sx["x"]), float(sx["y"]), float(sx["z"])) is None:
                problems.append(
                    f"stow xyz ({sx['x']}, {sx['y']}, {sx['z']}) outside envelope"
                )

        # Check all slots are reachable at grip height
        grip_z = float(self.heights["grip_z"])
        for slot in self.slots.values():
            joints = self.kin.inverse(slot.x, slot.y, grip_z)
            if joints is None:
                problems.append(f"slot '{slot.id}' at ({slot.x}, {slot.y}) unreachable")

        # Check all starting block positions are reachable
        for block in self.blocks.values():
            joints = self.kin.inverse(block.start_x, block.start_y, grip_z)
            if joints is None:
                problems.append(
                    f"block '{block.id}' start ({block.start_x}, {block.start_y}) unreachable"
                )

        # Check all staging positions are reachable
        for spot in self.staging.values():
            joints = self.kin.inverse(spot.x, spot.y, grip_z)
            if joints is None:
                problems.append(
                    f"staging '{spot.id}' at ({spot.x}, {spot.y}) unreachable"
                )

        return problems

    def to_scene(self) -> dict:
        """Everything the Three.js scene needs."""
        return {
            "site": self.site,
            "measured": bool(self.arm.get("measured", False)),
            "arm": {
                "base_height": self.kin.base_height,
                "upper_arm": self.kin.upper_arm,
                "forearm": self.kin.forearm,
                "eoat": self.kin.eoat,
            },
            "envelope": self.envelope,
            "gripper": self.gripper,
            "blocks": [
                {
                    "id": b.id,
                    "color": b.color,
                    "label": b.label,
                    "size": self.block_size,
                    "start_x": b.start_x,
                    "start_y": b.start_y,
                }
                for b in self.blocks.values()
            ],
            "slots": [
                {"id": s.id, "label": s.label, "x": s.x, "y": s.y}
                for s in self.slots.values()
            ],
            "heights": self.heights,
            "home_joints": self.home_joints.as_dict(),
        }


def load_config(path: str | os.PathLike | None = None) -> CellConfig:
    p = Path(path or os.environ.get("CELL_CONFIG") or DEFAULT_PATH)
    with open(p) as fh:
        return CellConfig(yaml.safe_load(fh))
