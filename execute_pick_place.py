"""Pure EXECUTION: run the pick-and-place motion recipe given ALREADY-KNOWN
pick/place arm coordinates. No camera, no color detection, no vision code
at all.

This is the piece that runs on the Raspberry Pi in the real architecture
(see docs/AGENT_MQTT_FLOW.md), one motion primitive at a time, each
primitive dispatched by an instruction relayed from the MCP/EC2 side over
MQTT (T:104 move / T:106 gripper / etc. never leave the Pi -- only the
"done"/"ok" result per primitive crosses the internet). The MCP side owns
WHAT to do and in what order (the planning); this module owns HOW to
physically do one step, given a target it's already been told.

Same 9-step recipe already used by sim/manager.py's execute_sort, factored
out here as a standalone, camera-free function so it can be exercised
directly (for local testing) or driven step-by-step by an MQTT command
loop, without needing sim/manager.py's full planning/job-tracking machinery.

Usage (standalone diagnostic -- runs the FULL recipe locally, no MQTT):
    python execute_pick_place.py --pick-x 383 --pick-y 90 --place-x 233 --place-y 199 --place-z -62.7
"""

from __future__ import annotations

import time

from sim.config import CellConfig
from sim.backends.roarm import RoArmBackend


def run_pick_and_place(
    backend: RoArmBackend,
    cfg: CellConfig,
    *,
    pick_x: float,
    pick_y: float,
    place_x: float,
    place_y: float,
    grip_z: float | None = None,
    place_z: float | None = None,
    pause: callable = lambda msg: None,
) -> None:
    """Run the full approach->open->lower->close->lift->carry->lower->open->lift recipe.

    grip_z: height to close the gripper at (default: cfg.heights.grip_z).
    place_z: height to lower to when releasing (default: same as grip_z) --
        pass a different value (e.g. heights.grid_top_z) when the place
        target sits at a different height than the pick target (e.g.
        placing into a raised grid platform instead of the bare table).
    pause: optional callback invoked with a step label BEFORE each step
        (e.g. an input() prompt for interactive stepping, or a no-op for
        --auto / MQTT-driven execution where each step is already a
        separate incoming command).
    """
    approach_z = float(cfg.heights["approach_z"])
    lift_z = float(cfg.heights["lift_z"])
    grip_z = grip_z if grip_z is not None else float(cfg.heights["grip_z"])
    place_z = place_z if place_z is not None else grip_z

    j_pick_approach = cfg.kin.inverse(pick_x, pick_y, approach_z)
    j_pick_grip = cfg.kin.inverse(pick_x, pick_y, grip_z)
    j_pick_lift = cfg.kin.inverse(pick_x, pick_y, lift_z)
    j_place_approach = cfg.kin.inverse(place_x, place_y, lift_z)
    j_place_lower = cfg.kin.inverse(place_x, place_y, place_z)
    for label, j in [("pick.approach", j_pick_approach), ("pick.grip", j_pick_grip),
                      ("pick.lift", j_pick_lift), ("place.approach", j_place_approach),
                      ("place.lower", j_place_lower)]:
        if j is None:
            raise RuntimeError(f"{label} UNREACHABLE / outside envelope")

    pause(f"approach pick at ({pick_x:.1f}, {pick_y:.1f})")
    backend.move_to(j_pick_approach, cfg)
    backend.gripper_open(cfg)

    pause("lower + CLOSE gripper")
    backend.move_to(j_pick_grip, cfg)
    # Log actual settled position so we can verify the arm reached the block
    try:
        import logging as _log
        _log.getLogger("execute_pick_place").info(
            "pick settle: (%.1f, %.1f, %.1f)  target was (%.1f, %.1f, %.1f)  "
            "off by dx=%.1f dy=%.1f dz=%.1f",
            backend._last_xyz[0], backend._last_xyz[1], backend._last_xyz[2],
            pick_x, pick_y, float(cfg.heights["grip_z"]),
            backend._last_xyz[0] - pick_x,
            backend._last_xyz[1] - pick_y,
            backend._last_xyz[2] - float(cfg.heights["grip_z"]),
        )
    except Exception:
        pass
    backend.gripper_close(cfg)
    time.sleep(0.5)

    pause("lift")
    backend.move_to(j_pick_lift, cfg)

    pause(f"carry to place ({place_x:.1f}, {place_y:.1f})")
    backend.move_to(j_place_approach, cfg)

    pause(f"lower (z={place_z:.1f}) and RELEASE")
    backend.move_to(j_place_lower, cfg)
    backend.gripper_open(cfg)
    time.sleep(0.5)

    pause("lift away")
    # Log actual place settle position
    try:
        import logging as _log
        _log.getLogger("execute_pick_place").info(
            "place settle: (%.1f, %.1f, %.1f)  target was (%.1f, %.1f, %.1f)  "
            "off by dx=%.1f dy=%.1f dz=%.1f",
            backend._last_xyz[0], backend._last_xyz[1], backend._last_xyz[2],
            place_x, place_y, place_z,
            backend._last_xyz[0] - place_x,
            backend._last_xyz[1] - place_y,
            backend._last_xyz[2] - place_z,
        )
    except Exception:
        pass
    # Pull slightly toward the arm's base (radially inward) before lifting --
    # this ensures the open jaws clear the placed block's sides, which extend
    # upward ~40mm from grid_top_z. Going straight up from the place position
    # drags the jaws through the block body if the jaws are still around it.
    import math as _math
    theta = _math.atan2(place_y, place_x)
    clear_x = place_x - 15.0 * _math.cos(theta)  # pull 15mm toward base
    clear_y = place_y - 15.0 * _math.sin(theta)
    j_clear = cfg.kin.inverse(clear_x, clear_y, lift_z)
    backend.move_to(j_clear if j_clear is not None else j_place_approach, cfg)


def main() -> None:
    import argparse
    from sim.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--pick-x", type=float, required=True)
    ap.add_argument("--pick-y", type=float, required=True)
    ap.add_argument("--place-x", type=float, required=True)
    ap.add_argument("--place-y", type=float, required=True)
    ap.add_argument("--grip-z", type=float, default=None)
    ap.add_argument("--place-z", type=float, default=None)
    ap.add_argument("--auto", action="store_true", help="skip step-by-step pauses")
    args = ap.parse_args()

    cfg = load_config()
    if not cfg.kin.passthrough:
        print("WARNING: ik_mode is not passthrough. This test expects the real arm.")

    def pause(msg: str):
        if args.auto:
            print(f"  [auto] {msg}")
        else:
            input(f"  >>> {msg} — press Enter (Ctrl+C to abort) ")

    b = RoArmBackend()
    try:
        pause("connect + home")
        b.connect(cfg)
        b.home(cfg)

        run_pick_and_place(
            b, cfg,
            pick_x=args.pick_x, pick_y=args.pick_y,
            place_x=args.place_x, place_y=args.place_y,
            grip_z=args.grip_z, place_z=args.place_z,
            pause=pause,
        )

        pause("return home")
        b.home(cfg)
        print("\ndone.")
    except KeyboardInterrupt:
        print("\naborted, sending estop")
        try:
            b.estop()
        except Exception:
            pass
    finally:
        b.release()


if __name__ == "__main__":
    main()
