"""Real RoArm-M2-Pro backend (USB serial control, no camera).

This backend drives a physical Waveshare RoArm-M2-Pro over its USB-C serial
port (the ESP32 accepts the same JSON commands over serial that it does over
WiFi HTTP). It implements the exact same interface as SimulatedBackend, so the
cell manager, kinematics, safety layer, MCP tools, and 3D visualization all
work unchanged. Only the config flag `backend: roarm` selects this.

Transport
---------
Newline-terminated JSON over a serial port (e.g. COM9 @ 115200). Each command
is a JSON object with a "T" (type) field. The firmware ECHOES the command back
first, then emits the real response on a separate line. Feedback requests
(T:105) are answered with a T:1051 packet containing b/s/e/t joint radians.

Command types used here (from Waveshare's documented API):
    T:100  MOVE_INIT            — home / initialize
    T:104  XYZT_GOAL_CTRL       — move tool to (x,y,z) in mm; arm solves its
                                  own IK. PRIMARY path (ik_mode: passthrough).
    T:101  SINGLE_JOINT_CTRL    — set ONE joint in RADIANS (analytic/sim
                                  fallback path only)
    T:105  SERVO_RAD_FEEDBACK   — request feedback (reply is T:1051, has x/y/z)
    T:106  EOAT_HAND_CTRL       — gripper open/close (radians)
    T:210  TORQUE_CTRL          — torque off/on (used for estop)

Motion (passthrough mode)
-------------------------
move_to() receives a Joints carrying target_xyz (set by Kinematics.inverse in
passthrough mode) and sends T:104 with those coordinates. A hard Z floor
clamps the commanded height so the tool can never be driven into the table.

Joint mapping (this codebase <-> RoArm firmware)
-----------------------------------------------
    base     <-> base / b
    shoulder <-> shoulder / s
    elbow    <-> elbow / e
    wrist    <-> hand / t      (the RoArm "hand"/EOAT joint is our "wrist")

Angles in this codebase are DEGREES; the arm speaks RADIANS. This backend is
the translation layer.

Thread safety
-------------
The manager's move_to() polls feedback in a loop while the 20 Hz mirror calls
read_joints() concurrently — both hit the same serial line. An I/O lock
serializes every command+read cycle so the two never interleave and corrupt
each other's data.

Boot reset
----------
Opening the serial port resets the ESP32, which then runs a ~5s init sequence
printing non-JSON log lines. connect() waits past that boot log before
declaring the arm ready.

Vision (deferred)
-----------------
There is no camera yet. `detect_blocks` and block-position tracking return
hardcoded positions from cell.yaml, exactly like the simulated backend seeds
its state. When a camera is added, only `detect_blocks` needs to change.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time

from ..config import CellConfig
from ..kinematics import Joints

log = logging.getLogger(__name__)

# Gripper positions in radians (documented RoArm-M2 defaults).
# Verify against your actual unit and override via cell.yaml gripper section.
GRIPPER_OPEN_RAD_DEFAULT = 1.08
GRIPPER_CLOSE_RAD_DEFAULT = 3.14

# Motion tuning defaults (override via cell.yaml connection section).
MOVE_SPD_DEFAULT = 0.25          # command speed field
MOVE_ACC_DEFAULT = 10            # command acceleration field
ARRIVE_TOL_RAD = 0.05            # "reached target" tolerance (radians)
POLL_INTERVAL_S = 0.05           # feedback poll cadence while moving
MOVE_TIMEOUT_S = 12.0            # give up waiting for a move after this long
# (was 20.0, then upped from an original 12.0 for far/awkward moves — but
# _move_xyz now retries once on timeout, so worst case is 2x this value.
# Brought back down to 12.0 to keep that worst case reasonable; the retry
# (not a longer single wait) is what actually helps a move that's still
# settling.)

# Serial defaults (override via cell.yaml connection section).
SERIAL_PORT_DEFAULT = "COM9"
SERIAL_BAUD_DEFAULT = 115200
SERIAL_READ_TIMEOUT_S = 1.0      # per-line read timeout
BOOT_WAIT_S = 5.0                # time to let the ESP32 finish its init after open
FEEDBACK_READ_ATTEMPTS = 10      # lines to scan for the T:1051 packet per request

# XYZ motion (passthrough mode).
ARRIVE_TOL_MM = 20.0             # "reached target" tolerance in mm
# (was 6.0, then 15.0 -- probe_reach_envelope.py swept 20 x/y/z combos and
# found the arm's REAL settle error is consistently 13-17.5mm across every
# height and distance tested, not growing with reach -- i.e. a flat settle
# bias, not a reach limit. 15.0 sat right on top of that range, so natural
# settles just over it (e.g. 15.4mm) were spuriously rejected, looking like
# a stall/timeout even though the arm had already physically arrived. 20.0
# gives margin above the observed max.)
XYZ_MOVE_SPD_DEFAULT = 0.25      # T:104 speed field
# Hard safety floor: the tool is NEVER commanded below this Z (mm, arm frame).
# Overridable via cell.yaml heights.z_floor. Defaults just below table_z.
Z_FLOOR_DEFAULT = -125.0


class RoArmBackend:
    """Physical RoArm-M2-Pro backend over USB serial."""

    name = "roarm"

    def __init__(self) -> None:
        self._lock = threading.Lock()      # protects block/gripper state
        self._io_lock = threading.Lock()   # serializes serial command+read cycles
        self._connected = False
        self._homed = False
        self._estopped = False

        # Serial transport
        self._serial = None
        self._port = SERIAL_PORT_DEFAULT
        self._baud = SERIAL_BAUD_DEFAULT

        # Gripper state
        self._gripper = "open"  # open | closed | gripping
        self._held_block: str | None = None
        self._gripper_open_rad = GRIPPER_OPEN_RAD_DEFAULT
        self._gripper_close_rad = GRIPPER_CLOSE_RAD_DEFAULT

        # Motion tuning
        self._spd = MOVE_SPD_DEFAULT
        self._acc = MOVE_ACC_DEFAULT

        # Block state tracking (no camera — seeded from config, same as sim)
        self._block_positions: dict[str, dict[str, float]] = {}
        self._block_slots: dict[str, str | None] = {}

        # Cache of last known joints (updated on every feedback read)
        self._last_joints = Joints()
        # Last commanded/known tool XYZ (passthrough mode block tracking)
        self._last_xyz = (0.0, 0.0, 0.0)
        self._z_floor = Z_FLOOR_DEFAULT
        # Current commanded gripper angle. XYZ moves (T:104) include the hand
        # 't' field, so we must send the CURRENT grip here or the move would
        # reopen the gripper and drop whatever it's holding.
        self._current_grip_rad = GRIPPER_OPEN_RAD_DEFAULT

    # -- lifecycle --------------------------------------------------------- #

    def connect(self, cfg: CellConfig) -> None:
        if self._connected:
            return

        try:
            import serial  # pyserial
        except ImportError as e:
            raise RuntimeError(
                "pyserial is required for the roarm backend. "
                "Install with: pip install pyserial"
            ) from e

        conn = cfg.connection or {}
        # Port resolution: config `port`, else env ROARM_PORT, else default.
        self._port = conn.get("port") or os.environ.get("ROARM_PORT") or SERIAL_PORT_DEFAULT
        self._baud = int(conn.get("baud") or os.environ.get("ROARM_BAUD") or SERIAL_BAUD_DEFAULT)

        # Gripper radian config + HARD LIMITS. Every gripper command is clamped
        # to [_grip_min, _grip_max] so it can never be driven past its safe
        # range (which previously jammed it under a screw and overloaded).
        grip = cfg.gripper or {}
        self._grip_min = float(grip.get("min_rad", 0.50))
        self._grip_max = float(grip.get("max_rad", 1.60))
        self._gripper_open_rad = self._clamp_grip(
            float(grip.get("open_rad", 0.60))
        )
        self._gripper_close_rad = self._clamp_grip(
            float(grip.get("close_rad", 1.50))
        )
        self._grip_overload = float(grip.get("overload_torque", 150))
        # Gripper actuation speed/accel. spd:0 can mean "very slow" on this
        # firmware, so use a real value so it closes/opens promptly.
        self._grip_spd = int(grip.get("grip_spd", 200))
        self._grip_acc = int(grip.get("grip_acc", 60))
        # Extra settle time AFTER a gripper actuation finishes (on top of
        # actuate_time_s, which covers the servo's own travel time) --
        # lets the grip mechanically stabilize (block seated in the fingers,
        # any last flex settled) before the caller's next move (e.g. lift)
        # fires. Without this, "close" -> "lift" could happen back-to-back
        # with no real pause, risking a lift starting before the grip has
        # actually settled onto the block.
        self._grip_settle_s = float(grip.get("grip_settle_s", 0.8))
        # Start assuming the gripper is open, so the first moves hold it open.
        self._current_grip_rad = self._gripper_open_rad
        # Motion tuning overrides
        self._spd = float(conn.get("move_spd", XYZ_MOVE_SPD_DEFAULT))
        self._acc = float(conn.get("move_acc", MOVE_ACC_DEFAULT))
        # T:101 single-joint speed (analytic/joint mode fallback).
        self._joint_spd = int(conn.get("joint_spd", 10))

        # Hard Z safety floor (passthrough/XYZ mode). Prefer heights.z_floor,
        # else just below the measured table.
        heights = cfg.heights or {}
        self._z_floor = float(
            heights.get("z_floor", heights.get("table_z", Z_FLOOR_DEFAULT) - 3.0)
        )

        # Seed block state from config (no camera)
        for block in cfg.blocks.values():
            self._block_positions[block.id] = {"x": block.start_x, "y": block.start_y}
            self._block_slots[block.id] = None

        # Open the serial port. This RESETS the ESP32, which then runs its boot
        # init sequence (~5s of non-JSON log lines).
        try:
            self._serial = serial.Serial(
                self._port, self._baud, timeout=SERIAL_READ_TIMEOUT_S
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot open serial port {self._port} @ {self._baud}: {e}. "
                f"Check the arm is powered, the USB cable is connected, and the "
                f"port is correct (ROARM_PORT / connection.port)."
            ) from e

        # Wait for the boot sequence to finish, then flush the boot log.
        boot_wait = float(conn.get("boot_wait_s", BOOT_WAIT_S))
        log.info("serial open on %s; waiting %.1fs for arm boot...", self._port, boot_wait)
        time.sleep(boot_wait)
        try:
            self._serial.reset_input_buffer()
        except Exception:
            pass

        # Verify the arm answers a feedback request.
        try:
            self._read_feedback()
        except Exception as e:
            raise RuntimeError(
                f"Serial port {self._port} opened but the arm did not return "
                f"feedback: {e}. Try re-running (the arm may still be booting)."
            ) from e

        self._connected = True
        self._estopped = False
        log.info("roarm backend connected (serial %s @ %d)", self._port, self._baud)

    def is_connected(self) -> bool:
        return self._connected

    def release(self) -> None:
        self._connected = False
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass

    # -- serial transport -------------------------------------------------- #

    def _write_cmd(self, cmd: dict) -> None:
        """Write one newline-terminated JSON command to the serial port."""
        if self._serial is None:
            raise RuntimeError("roarm backend not connected")
        line = json.dumps(cmd, separators=(",", ":")) + "\n"
        self._serial.write(line.encode("utf-8"))
        self._serial.flush()

    def _send(self, cmd: dict) -> None:
        """Send a fire-and-forget command (no response expected)."""
        with self._io_lock:
            self._write_cmd(cmd)

    def _read_feedback(self, retries: int = 3) -> dict:
        """Request feedback (T:105) and return the T:1051 packet with b/s/e/t.

        The firmware echoes the command first, then sends the real feedback
        on a later line. We scan several lines for a packet containing joint
        fields, skipping the echo and any boot/log noise.

        Occasionally the feedback line is missed within the fixed read
        attempts (serial timing jitter) — retried automatically instead of
        raising on a single transient miss.
        """
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                with self._io_lock:
                    self._write_cmd({"T": 105})
                    for _ in range(FEEDBACK_READ_ATTEMPTS):
                        raw = self._serial.readline().decode("utf-8", errors="ignore").strip()
                        if not raw:
                            continue
                        try:
                            pkt = json.loads(raw)
                        except json.JSONDecodeError:
                            continue  # non-JSON log line
                        # Must be the actual T:1051 feedback packet type, not
                        # just "contains a t/b/s/e key" — a leftover echo of
                        # a T:104 move command (still in the serial buffer
                        # from a just-sent move) ALSO has a "t" field (the
                        # commanded gripper angle), and was previously
                        # mistaken for real feedback, silently returning the
                        # move's TARGET instead of the arm's actual position.
                        if pkt.get("T") == 1051 and any(k in pkt for k in ("b", "s", "e", "t")):
                            return pkt
                        # else it was an echo/ack of some other command — keep reading
                last_error = RuntimeError("no joint feedback packet (T:1051) received")
            except Exception as e:
                last_error = e
            time.sleep(0.2)
        raise last_error or RuntimeError("no joint feedback packet (T:1051) received")

    # -- unit / name conversion -------------------------------------------- #

    # RoArm T:101 single-joint indices.
    #   0 = base, 1 = shoulder, 2 = elbow, 3 = hand (our "wrist")
    _JOINT_INDEX = {"base": 0, "shoulder": 1, "elbow": 2, "wrist": 3}

    def _joint_cmds(self, j: Joints) -> list[dict]:
        """Our Joints (degrees) -> list of T:101 single-joint commands (radians).

        This firmware variant does NOT respond to the combined T:102 command,
        so we drive each joint individually with T:101. Verified on-hardware.
        """
        spd = int(self._joint_spd)
        return [
            {"T": 101, "joint": 0, "rad": math.radians(j.base), "spd": spd, "acc": self._acc},
            {"T": 101, "joint": 1, "rad": math.radians(j.shoulder), "spd": spd, "acc": self._acc},
            {"T": 101, "joint": 2, "rad": math.radians(j.elbow), "spd": spd, "acc": self._acc},
            {"T": 101, "joint": 3, "rad": math.radians(j.wrist), "spd": spd, "acc": self._acc},
        ]

    @staticmethod
    def _feedback_to_joints(fb: dict) -> Joints:
        """RoArm feedback (radians: b/s/e/t) -> our Joints (degrees)."""
        return Joints(
            base=round(math.degrees(float(fb.get("b", 0.0))), 2),
            shoulder=round(math.degrees(float(fb.get("s", 0.0))), 2),
            elbow=round(math.degrees(float(fb.get("e", 0.0))), 2),
            wrist=round(math.degrees(float(fb.get("t", 0.0))), 2),  # t -> wrist
        )

    # -- motion ------------------------------------------------------------ #

    def home(self, cfg: CellConfig, do_init: bool = False) -> None:
        """Move to the safe home pose.

        do_init=True runs the firmware MOVE_INIT (T:100) first — use this ONCE
        at connect/startup. During normal operation (do_init=False), just glide
        to the home pose with a plain move so the arm doesn't re-run its whole
        init routine after every job.
        """
        if do_init:
            self._send({"T": 100})
            time.sleep(0.5)
        # In passthrough mode, home to a safe XYZ pose (home.xyz). Otherwise
        # use the joint home (analytic/sim).
        home_xyz = (cfg.raw.get("home", {}) or {}).get("xyz")
        if cfg.kin.passthrough and home_xyz:
            self._move_xyz(float(home_xyz["x"]), float(home_xyz["y"]), float(home_xyz["z"]))
        else:
            self.move_to(cfg.home_joints, cfg)
        self._homed = True

    def stow(self, cfg: CellConfig) -> None:
        """Move to the retracted stow pose (cell.yaml stow.xyz).

        Call this BEFORE capturing a camera frame for detection, so the
        arm's own body doesn't occlude the board in view. No-op (with a
        warning) if no stow.xyz is configured, or if not in passthrough mode
        (stow is an XYZ-only pose, same as home.xyz).
        """
        if cfg.stow_xyz is None:
            log.warning("stow() called but no stow.xyz configured in cell.yaml; skipping")
            return
        if not cfg.kin.passthrough:
            log.warning("stow() only supported in passthrough (real arm) mode; skipping")
            return
        x, y, z = cfg.stow_xyz
        self._move_xyz(x, y, z)

    @property
    def homed(self) -> bool:
        return self._homed

    def read_joints(self) -> Joints:
        """Current joint angles in degrees. Called ~20 Hz for the live mirror."""
        try:
            fb = self._read_feedback()
            j = self._feedback_to_joints(fb)
            with self._lock:
                self._last_joints = j
                if "x" in fb:
                    self._last_xyz = (float(fb["x"]), float(fb["y"]), float(fb["z"]))
            return j
        except Exception:
            # On a transient read failure, return the last known joints so the
            # 20 Hz mirror doesn't crash the stream.
            with self._lock:
                return self._last_joints

    def move_to(self, target: Joints, cfg: CellConfig) -> None:
        """Drive to a target. Blocks until arrival or timeout.

        Passthrough mode (target.target_xyz set): command XYZ via T:104 and let
        the arm's onboard IK solve it. This is the real-arm path.
        Joint mode (no target_xyz): send per-joint T:101 (sim/analytic path).
        """
        if self._estopped:
            raise RuntimeError("motion refused: arm is in estop")

        if target.target_xyz is not None:
            self._move_xyz(*target.target_xyz)
        else:
            self._move_joints(target)

    def _move_xyz(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Command an absolute XYZ target (T:104) and poll until settled.

        A hard Z floor clamps the commanded height so the tool can never be
        driven below the safe minimum (protects the table).
        """
        if z < self._z_floor:
            raise RuntimeError(
                f"move refused: z={z:.1f} below safety floor {self._z_floor:.1f}"
            )

        # The 't' field in T:104 is the wrist/hand servo, shared with the
        # gripper. Send the CURRENT grip value so the move holds whatever the
        # gripper is set to (open or closed). This is the version that worked
        # in the alignment test where the duck was gripped and lifted.
        for attempt in range(2):  # one retry if the first attempt times out
            self._send({"T": 104, "x": x, "y": y, "z": z,
                        "t": self._current_grip_rad, "spd": self._spd})
            with self._lock:
                self._last_xyz = (x, y, z)  # record intended target for block tracking

            # Require TWO consecutive readings within tolerance, not just one
            # instant snapshot -- a single in-tolerance reading can still be
            # mid-settle (servo still creeping toward true rest position),
            # which was letting a caller (e.g. gripper_open/close right
            # after a move) act while the arm was still physically moving.
            consecutive_ok = 0
            deadline = time.monotonic() + MOVE_TIMEOUT_S
            start = time.monotonic()
            last_progress_log = start
            cx, cy, cz = x, y, z
            while True:
                if self._estopped:
                    raise RuntimeError("motion stopped by estop")
                try:
                    fb = self._read_feedback()
                except Exception:
                    if time.monotonic() > deadline:
                        break
                    time.sleep(POLL_INTERVAL_S)
                    continue

                cx, cy, cz = float(fb.get("x", 0.0)), float(fb.get("y", 0.0)), float(fb.get("z", 0.0))
                with self._lock:
                    self._last_joints = self._feedback_to_joints(fb)
                    self._last_xyz = (cx, cy, cz)

                # Visible progress every ~2s -- without this, a slow-to-settle
                # move looks like silent dead time on the console (confirmed
                # source of "why is there a gap with nothing happening").
                now = time.monotonic()
                if now - last_progress_log > 2.0:
                    log.info(
                        "move_to(xyz): waiting for arrival... at (%.1f,%.1f,%.1f), "
                        "target (%.1f,%.1f,%.1f), off by dx=%.1f dy=%.1f dz=%.1f (%.1fs elapsed)",
                        cx, cy, cz, x, y, z, cx - x, cy - y, cz - z, now - start,
                    )
                    last_progress_log = now

                reached = (abs(cx - x) < ARRIVE_TOL_MM and abs(cy - y) < ARRIVE_TOL_MM
                           and abs(cz - z) < ARRIVE_TOL_MM)
                consecutive_ok = consecutive_ok + 1 if reached else 0
                if consecutive_ok >= 2:
                    return cx, cy, cz
                if time.monotonic() > deadline:
                    break
                time.sleep(POLL_INTERVAL_S)

            if attempt == 0:
                log.warning(
                    "move_to(xyz): arrival TIMEOUT after %.1fs; still at "
                    "(%.1f,%.1f,%.1f), wanted (%.1f,%.1f,%.1f), off by "
                    "dx=%.1f dy=%.1f dz=%.1f — retrying once",
                    MOVE_TIMEOUT_S, cx, cy, cz, x, y, z, cx - x, cy - y, cz - z,
                )
            else:
                log.warning(
                    "move_to(xyz): arrival TIMEOUT after retry; still at "
                    "(%.1f,%.1f,%.1f), wanted (%.1f,%.1f,%.1f), off by "
                    "dx=%.1f dy=%.1f dz=%.1f — continuing anyway",
                    cx, cy, cz, x, y, z, cx - x, cy - y, cz - z,
                )
                return cx, cy, cz

    def _move_joints(self, target: Joints) -> None:
        """Joint-space move via per-joint T:101 (analytic/sim fallback)."""
        for cmd in self._joint_cmds(target):
            self._send(cmd)

        target_rad = {
            "b": math.radians(target.base),
            "s": math.radians(target.shoulder),
            "e": math.radians(target.elbow),
            "t": math.radians(target.wrist),
        }
        deadline = time.monotonic() + MOVE_TIMEOUT_S
        while True:
            if self._estopped:
                raise RuntimeError("motion stopped by estop")
            try:
                fb = self._read_feedback()
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError("move_to: timed out reading feedback")
                time.sleep(POLL_INTERVAL_S)
                continue

            with self._lock:
                self._last_joints = self._feedback_to_joints(fb)

            reached = all(
                abs(float(fb.get(k, 0.0)) - target_rad[k]) < ARRIVE_TOL_RAD
                for k in ("b", "s", "e", "t")
            )
            if reached:
                return
            if time.monotonic() > deadline:
                log.warning("move_to: arrival timeout; continuing anyway")
                return
            time.sleep(POLL_INTERVAL_S)

    # -- gripper ----------------------------------------------------------- #

    def _tool_xy(self, cfg: CellConfig) -> tuple[float, float]:
        """Current tool (x, y) for block tracking.

        Passthrough mode: use the last commanded/known XYZ (accurate).
        Analytic mode: use forward kinematics on the last joints.
        """
        if cfg.kin.passthrough:
            return self._last_xyz[0], self._last_xyz[1]
        pose = cfg.kin.forward(self._last_joints)
        return pose.x, pose.y

    def _clamp_grip(self, rad: float) -> float:
        """Clamp a gripper target to the safe hard limits [min_rad, max_rad].

        This is the single guard that prevents the gripper from EVER being
        driven past its physical range (which jammed it under a screw and
        overloaded the servo). Every gripper command passes through here.
        """
        clamped = max(self._grip_min, min(self._grip_max, rad))
        if clamped != rad:
            log.warning("gripper target %.3f clamped to safe range [%.2f, %.2f] -> %.3f",
                        rad, self._grip_min, self._grip_max, clamped)
        return clamped

    def _read_hand_torque(self) -> float | None:
        """Read the hand servo torque (torH) from feedback, if available."""
        try:
            fb = self._read_feedback()
            return float(fb.get("torH", 0.0))
        except Exception:
            return None

    def _command_gripper(self, rad: float, protect_overload: bool = False) -> None:
        """Send a clamped gripper command.

        The hard clamp (_clamp_grip) is the real safety: it prevents ever
        driving the gripper past its physical range (which was the original
        jam cause). We do NOT auto-release torque on high readings anymore —
        the hand servo shows large transient torque during normal open/close
        motion, and releasing on that made the gripper go limp instead of
        opening. The clamp alone keeps it safe.

        protect_overload is kept for signature compatibility but no longer
        triggers a torque release.
        """
        safe = self._clamp_grip(rad)
        # Use the configured grip speed/accel. spd:0 can mean "very slow" on
        # this firmware, which was causing gripper moves to lag behind the
        # actuate_time_s wait — visibly showing up a step late.
        self._send({"T": 106, "cmd": safe, "spd": self._grip_spd, "acc": self._grip_acc})
        # Remember the grip so subsequent XYZ moves hold it (don't reopen).
        self._current_grip_rad = safe
        time.sleep(0.3)

    def _rehold_xyz_after_grip(self, cfg: CellConfig, z_before: float | None) -> None:
        """Re-assert the last commanded XYZ after a gripper actuation.

        The gripper's jaw angle IS the 't' field sent with every T:104 XYZ
        move (see _move_xyz) -- there's no separate wrist-pitch servo here.
        A parallel-jaw gripper's fingertip generally doesn't stay on a pure
        vertical line as the jaws swing open/closed; it can trace a small
        arc, which can shift the effective grip height slightly as 't'
        changes. Logs the actual measured shift so this is confirmed by real
        numbers rather than assumed, then re-drives the same target XYZ
        (holding the NEW grip angle) so the arm actively corrects back
        instead of leaving any shift uncorrected.
        """
        if not cfg.kin.passthrough:
            return
        try:
            fb = self._read_feedback()
            z_after = float(fb.get("z"))
        except Exception:
            return
        if z_before is not None:
            shift = z_after - z_before
            if abs(shift) > 1.0:
                log.info("gripper actuation shifted z by %.1fmm (%.1f -> %.1f); re-holding",
                          shift, z_before, z_after)
        x, y, z = self._last_xyz
        try:
            self._move_xyz(x, y, z)
        except Exception:
            log.exception("failed to re-hold XYZ after gripper actuation")

    def gripper_open(self, cfg: CellConfig) -> None:
        z_before = None
        if cfg.kin.passthrough:
            try:
                z_before = float(self._read_feedback().get("z"))
            except Exception:
                pass
        self._command_gripper(self._gripper_open_rad)
        time.sleep(float(cfg.gripper.get("actuate_time_s", 0.4)))
        self._rehold_xyz_after_grip(cfg, z_before)
        time.sleep(self._grip_settle_s)  # let the release mechanically settle
        with self._lock:
            if self._held_block:
                # Releasing: record the block at the current tool position.
                tx, ty = self._tool_xy(cfg)
                self._block_positions[self._held_block] = {
                    "x": round(tx, 1),
                    "y": round(ty, 1),
                }
                self._held_block = None
            self._gripper = "open"

    def gripper_close(self, cfg: CellConfig) -> None:
        z_before = None
        if cfg.kin.passthrough:
            try:
                z_before = float(self._read_feedback().get("z"))
            except Exception:
                pass
        self._command_gripper(self._gripper_close_rad)
        time.sleep(float(cfg.gripper.get("actuate_time_s", 0.4)))
        self._rehold_xyz_after_grip(cfg, z_before)
        # Let the grip mechanically settle onto the block BEFORE returning
        # control to the caller -- without this, "close" -> "lift" could
        # fire back-to-back with no real pause, risking a lift starting
        # before the fingers have actually seated around the block.
        time.sleep(self._grip_settle_s)
        with self._lock:
            # Without a sensor, infer grab from proximity to a known block.
            tx, ty = self._tool_xy(cfg)
            grabbed = self._find_block_at(tx, ty, cfg.block_size)
            if grabbed:
                self._held_block = grabbed
                self._gripper = "gripping"
            else:
                self._gripper = "closed"

    def gripper_state(self) -> str:
        with self._lock:
            return self._gripper

    def _find_block_at(self, x: float, y: float, block_size: float) -> str | None:
        grab_radius = block_size * 0.8
        for bid, pos in self._block_positions.items():
            if bid == self._held_block:
                continue
            dx = pos["x"] - x
            dy = pos["y"] - y
            if (dx * dx + dy * dy) < grab_radius * grab_radius:
                return bid
        return None

    # -- vision (hardcoded — no camera) ------------------------------------ #

    def detect_blocks(self, cfg: CellConfig) -> list[dict]:
        """Return duck positions: live camera+color-detection if enabled, else config.

        vision.use_camera in cell.yaml gates this. When enabled, captures a
        frame, finds each duck's EXACT pixel centroid via color segmentation
        (vision/color_detect.py — no VLM, no grid cells, no per-call cost),
        and converts pixel -> arm mm via the calibrated pixel_arm_transform
        (see calibrate_pixel_to_arm.py) — falling back to config-seeded
        positions if the camera/detection fails, so a transient error never
        blocks a pick/place run.
        """
        vision_cfg = cfg.vision or {}
        if vision_cfg.get("use_camera", False):
            try:
                return self._detect_blocks_camera(cfg, vision_cfg)
            except Exception:
                log.exception("camera detection failed; falling back to config positions")

        with self._lock:
            results = []
            for block in cfg.blocks.values():
                if block.id == self._held_block:
                    continue
                pos = self._block_positions.get(block.id)
                if pos:
                    results.append({
                        "id": block.id,
                        "color": block.color,
                        "label": block.label,
                        "x": pos["x"],
                        "y": pos["y"],
                        "confidence": 1.0,
                    })
            return results

    def _detect_blocks_camera(self, cfg: CellConfig, vision_cfg: dict) -> list[dict]:
        """Capture + color-detect each duck's EXACT pixel centroid, converted to arm mm.

        Deterministic classical CV, no VLM/API call: color segmentation
        (vision/color_detect.py) finds each duck's real centroid pixel, not
        a cell approximation — this is what lets the gripper land exactly on
        the duck regardless of where within its area it happens to sit.
        pixel_arm_transform (calibrated once via calibrate_pixel_to_arm.py)
        converts that pixel straight to arm mm. Raises if the camera fails
        or no calibration exists — caller (detect_blocks) handles the
        fallback to config positions.
        """
        from vision.board_crop import capture_board_frame
        from vision.color_detect import find_duck_centroid
        from vision.pixel_arm_transform import apply_transform, load_transform

        raw_index = vision_cfg.get("camera_index", 1)
        try:
            index: int | str = int(raw_index)
        except (TypeError, ValueError):
            index = str(raw_index)  # e.g. an IP camera stream URL
        min_area = float(vision_cfg.get("min_duck_area", 1000.0))

        # Retract out of the camera's view before capturing, so the arm's own
        # body doesn't occlude the board. Opt-out via vision.stow_before_detect
        # if no stow.xyz is configured or the arm doesn't need to move out of
        # frame for this camera's mounting position.
        if vision_cfg.get("stow_before_detect", True) and cfg.stow_xyz is not None:
            self.stow(cfg)

        # Cropped to just the board (via the drawn boundary line, see
        # calibrate_board_crop.py) — MUST match what calibrate_pixel_to_arm.py
        # used, or pixel positions won't correspond to the fitted transform.
        frame = capture_board_frame(index)

        transform = load_transform(expected_shape=frame.shape[:2])
        if transform is None:
            raise RuntimeError(
                "no pixel_arm_transform found; run calibrate_pixel_to_arm.py once for this camera"
            )

        with self._lock:
            held = self._held_block

        results = []
        for block in cfg.blocks.values():
            if block.id == held:
                continue  # can't see a duck that's currently in the gripper
            found = find_duck_centroid(frame, block.color, min_area=min_area)
            if found is None:
                continue  # this duck isn't visible on the board right now
            px, py, area = found
            x, y = apply_transform(transform, px, py)
            x += float(vision_cfg.get("pixel_arm_offset_x", 0.0))
            y += float(vision_cfg.get("pixel_arm_offset_y", 0.0))
            log.info(
                "%s: pixel=(%.1f, %.1f) area=%.0f -> arm (%.1f, %.1f)",
                block.id, px, py, area, x, y,
            )
            results.append({
                "id": block.id,
                "color": block.color,
                "label": block.label,
                "x": x,
                "y": y,
                "confidence": 1.0,  # deterministic CV, not a probabilistic estimate
            })

        with self._lock:
            for r in results:
                self._block_positions[r["id"]] = {"x": r["x"], "y": r["y"]}

        return results

    # -- state query ------------------------------------------------------- #

    def get_block_positions(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return dict(self._block_positions)

    def get_held_block(self) -> str | None:
        with self._lock:
            return self._held_block

    def get_block_slots(self) -> dict[str, str | None]:
        with self._lock:
            return dict(self._block_slots)

    def set_block_slot(self, block_id: str, slot_id: str | None) -> None:
        with self._lock:
            self._block_slots[block_id] = slot_id

    def reset_blocks(self, cfg: CellConfig) -> None:
        """Reset tracked block state to starting positions.

        Physical repositioning is driven by the manager's pick-and-place reset
        job (which calls move_to/gripper). This just resets the tracked state.
        """
        with self._lock:
            self._held_block = None
            self._gripper = "open"
            for block in cfg.blocks.values():
                self._block_positions[block.id] = {
                    "x": block.start_x,
                    "y": block.start_y,
                }
                self._block_slots[block.id] = None

    # -- safety ------------------------------------------------------------ #

    def estop(self) -> None:
        self._estopped = True
        try:
            # Torque off to freeze / go limp immediately.
            self._send({"T": 210, "cmd": 0})
        except Exception:
            log.exception("estop command failed to send")

    def clear_estop(self) -> None:
        self._estopped = False
        try:
            self._send({"T": 210, "cmd": 1})  # torque back on
        except Exception:
            log.exception("clear_estop command failed to send")
