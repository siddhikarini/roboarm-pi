"""Forward kinematics, inverse kinematics, and envelope checking for pick-and-place.

Same 4-DOF arm as the inspection cell (base yaw + 3 pitch joints), but now
we also need IK: given a target (x, y, z) for the gripper tip, find joint
angles that reach it. For a 4-DOF arm in a vertical plane, analytical IK is
feasible and we use it here.

The gripper always points straight down for pick-and-place. That simplifies
IK enormously: wrist angle is chosen to make the tool vertical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    """Gripper tip position in cell coordinates."""
    x: float
    y: float
    z: float

    @property
    def radius(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class Joints:
    base: float = 0.0
    shoulder: float = 0.0
    elbow: float = 0.0
    wrist: float = 0.0
    # Optional: the (x, y, z) target these joints were solved for. The real
    # RoArm backend uses this to command XYZ directly (T:104) via the arm's
    # own IK, bypassing our link-length model. Simulated backend ignores it.
    target_xyz: tuple[float, float, float] | None = None

    def as_dict(self) -> dict[str, float]:
        return {
            "base": self.base,
            "shoulder": self.shoulder,
            "elbow": self.elbow,
            "wrist": self.wrist,
        }

    @staticmethod
    def from_dict(d: dict) -> "Joints":
        return Joints(
            base=float(d.get("base", 0.0)),
            shoulder=float(d.get("shoulder", 0.0)),
            elbow=float(d.get("elbow", 0.0)),
            wrist=float(d.get("wrist", 0.0)),
        )

    def lerp(self, other: "Joints", t: float) -> "Joints":
        t = max(0.0, min(1.0, t))
        return Joints(
            base=self.base + (other.base - self.base) * t,
            shoulder=self.shoulder + (other.shoulder - self.shoulder) * t,
            elbow=self.elbow + (other.elbow - self.elbow) * t,
            wrist=self.wrist + (other.wrist - self.wrist) * t,
        )

    def max_delta(self, other: "Joints") -> float:
        a, b = self.as_dict(), other.as_dict()
        return max(abs(a[k] - b[k]) for k in a)


class Kinematics:
    """FK, IK, and envelope checks driven by config."""

    def __init__(self, arm: dict, envelope: dict):
        self.base_height = float(arm["base_height"])
        self.upper_arm = float(arm["upper_arm"])
        self.forearm = float(arm["forearm"])
        self.eoat = float(arm["eoat"])
        self.limits = {k: tuple(v) for k, v in arm["joint_limits"].items()}
        self.max_deg_per_sec = float(arm["max_deg_per_sec"])

        self.r_min, self.r_max = envelope["radius"]
        self.z_min, self.z_max = envelope["z"]

        # Passthrough (XYZ) mode: when the real arm does its own IK, we don't
        # use our link-length model. inverse() returns a Joints carrying the
        # target (x,y,z), validated only against the measured envelope. Enabled
        # via arm.ik_mode: passthrough in cell.yaml.
        self.passthrough = str(arm.get("ik_mode", "analytic")).lower() == "passthrough"

    # -- forward kinematics ------------------------------------------------ #

    def forward(self, j: Joints) -> Pose:
        """Joint angles -> gripper tip position."""
        a1 = math.radians(j.shoulder)
        a2 = a1 + math.radians(j.elbow)
        a3 = a2 + math.radians(j.wrist)

        r = (
            self.upper_arm * math.cos(a1)
            + self.forearm * math.cos(a2)
            + self.eoat * math.cos(a3)
        )
        z = (
            self.base_height
            + self.upper_arm * math.sin(a1)
            + self.forearm * math.sin(a2)
            + self.eoat * math.sin(a3)
        )

        yaw = math.radians(j.base)
        return Pose(
            x=r * math.cos(yaw),
            y=r * math.sin(yaw),
            z=z,
        )

    # -- inverse kinematics ------------------------------------------------ #

    def inverse(self, x: float, y: float, z: float) -> Joints | None:
        """Target (x, y, z) -> joint angles, or None if unreachable.

        In passthrough mode (real arm with onboard IK), we skip our link-length
        model entirely: return a Joints carrying the target (x,y,z), validated
        only against the measured envelope. The RoArm backend commands XYZ
        (T:104) directly and the arm solves its own IK.

        In analytic mode (simulation), solve joint angles from link lengths.

        Constraint (analytic): gripper points straight down (tool pitch = -90).
        """
        if self.passthrough:
            # Validate against the measured cylindrical envelope only.
            r = math.hypot(x, y)
            if not (self.r_min <= r <= self.r_max):
                return None
            if not (self.z_min <= z <= self.z_max):
                return None
            return Joints(target_xyz=(x, y, z))

        # Base angle from x, y
        base_deg = math.degrees(math.atan2(y, x))

        # Horizontal distance from base axis
        r = math.hypot(x, y)

        # The eoat hangs straight down, so it contributes (0, -eoat) in the
        # vertical plane. We solve the 2-link (upper_arm + forearm) problem
        # to reach the point where the wrist joint sits.
        wrist_r = r          # wrist is directly above target (tool vertical)
        wrist_z = z + self.eoat - self.base_height  # relative to shoulder

        # 2-link IK in the vertical plane
        L1 = self.upper_arm
        L2 = self.forearm
        d = math.hypot(wrist_r, wrist_z)

        if d > L1 + L2 or d < abs(L1 - L2):
            return None  # unreachable

        # Law of cosines for elbow angle
        cos_elbow = (L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2)
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        elbow_rad = math.acos(cos_elbow)

        # We want elbow-down configuration (more natural for pick-place)
        elbow_deg = -(math.pi - elbow_rad)

        # Shoulder angle
        alpha = math.atan2(wrist_z, wrist_r)
        beta = math.atan2(L2 * math.sin(math.pi - elbow_rad),
                          L1 + L2 * math.cos(math.pi - elbow_rad))
        shoulder_rad = alpha + beta
        shoulder_deg = math.degrees(shoulder_rad)
        elbow_deg_final = math.degrees(elbow_deg)

        # Wrist: tool must point straight down.
        # Cumulative pitch = shoulder + elbow + wrist = -90 degrees
        wrist_deg = -90.0 - shoulder_deg - elbow_deg_final

        joints = Joints(
            base=round(base_deg, 2),
            shoulder=round(shoulder_deg, 2),
            elbow=round(elbow_deg_final, 2),
            wrist=round(wrist_deg, 2),
        )

        # Verify within limits
        if self.joint_violations(joints):
            return None

        return joints

    # -- validation -------------------------------------------------------- #

    def joint_violations(self, j: Joints) -> list[str]:
        out: list[str] = []
        for name, value in j.as_dict().items():
            lo, hi = self.limits[name]
            if not (lo <= value <= hi):
                out.append(f"{name} at {value:.1f} deg exceeds [{lo}, {hi}]")
        return out

    def envelope_violations(self, j: Joints) -> list[str]:
        """Check gripper tip against workspace envelope."""
        out: list[str] = []
        pose = self.forward(j)

        if pose.radius > self.r_max:
            out.append(f"tip at r={pose.radius:.0f}mm exceeds {self.r_max}mm")
        if pose.radius < self.r_min:
            out.append(f"tip at r={pose.radius:.0f}mm is inside {self.r_min}mm")
        if not (self.z_min <= pose.z <= self.z_max):
            out.append(f"tip at z={pose.z:.0f}mm outside [{self.z_min}, {self.z_max}]")

        return out

    def travel_seconds(self, a: Joints, b: Joints) -> float:
        """Estimated wall-clock time for a move."""
        return max(0.3, a.max_delta(b) / self.max_deg_per_sec)
