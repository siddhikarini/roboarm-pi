"""Backend protocol for pick-and-place operations.

Extends the basic arm protocol with gripper operations. The simulated backend
synthesizes block detection; the real backend would use a camera + CV.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import CellConfig
from ..kinematics import Joints


@runtime_checkable
class ArmBackend(Protocol):
    name: str

    def connect(self, cfg: CellConfig) -> None:
        """Open the transport. Must be idempotent."""

    def is_connected(self) -> bool: ...

    def home(self, cfg: CellConfig) -> None:
        """Return to home position."""

    def move_to(self, target: Joints, cfg: CellConfig) -> None:
        """Drive to target joint angles. Blocking. Raises on failure."""

    def read_joints(self) -> Joints:
        """Current joint angles. Called at ~20 Hz for the live mirror."""

    def gripper_open(self, cfg: CellConfig) -> None:
        """Open the gripper."""

    def gripper_close(self, cfg: CellConfig) -> None:
        """Close the gripper (grab a block)."""

    def gripper_state(self) -> str:
        """Return 'open', 'closed', or 'gripping'."""

    def detect_blocks(self, cfg: CellConfig) -> list[dict]:
        """Detect blocks on the table using overhead camera.

        Returns list of {id, color, label, x, y} for each visible block.
        In simulation, this returns the simulated block positions.
        In reality, this would capture a frame and run color detection.
        """

    def estop(self) -> None:
        """Cut motion immediately."""

    def release(self) -> None:
        """Close the transport."""
