"""Booth laptop agent — the local half of the MQTT relay.

This runs on the laptop physically near the RoArm-M2-Pro (same WiFi LAN).
It is the ONLY thing that talks to the arm. It:

  1. Connects OUTBOUND to the MQTT broker on EC2 (NAT-friendly — no inbound
     ports, no port forwarding needed on the booth network).
  2. Subscribes to robo/cmd. For each command, runs the corresponding
     RoArmBackend method LOCALLY over the LAN (the tight feedback-poll loop
     never crosses the internet).
  3. Publishes the result to robo/result with the same correlation id.
  4. Streams live joint/gripper state to robo/state at ~10 Hz so the EC2
     side and the 3D visualization stay in sync.

Run it:
    python laptop_agent.py
Config comes from cell.yaml (connection.mqtt + connection.host for the arm)
and can be overridden with env vars:
    MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD   (broker)
    ROARM_HOST                                           (arm IP)

The manager/kinematics/MCP tools do NOT run here — only the arm backend.
This process is a thin executor: broker command in, arm motion out.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from sim.config import load_config
from sim.kinematics import Joints
from sim.backends.roarm import RoArmBackend
from sim import mqtt_protocol as proto
from execute_pick_place import run_pick_and_place
from vision.detect_pick_place import detect_kit_plan, detect_pick_and_place

log = logging.getLogger("laptop_agent")

# Guards against accidentally running TWO laptop_agent.py instances at
# once -- confirmed as a recurring real problem this session: both
# instances subscribe to the same robo/cmd topic and race to answer every
# command, causing serial port contention (slow/jittery arm settling,
# repeated arrival timeouts during calibration) and confusing, split log
# output across two processes. A simple PID lock file at startup catches
# this immediately with a clear error, instead of silently running two
# competing processes until someone notices by inspecting `ps aux`.
_LOCK_PATH = Path(__file__).resolve().parent / ".laptop_agent.lock"


def _acquire_singleton_lock() -> None:
    if _LOCK_PATH.exists():
        try:
            old_pid = int(_LOCK_PATH.read_text().strip())
        except (ValueError, OSError):
            old_pid = None
        if old_pid is not None:
            try:
                os.kill(old_pid, 0)  # signal 0: just checks if the pid exists
                alive = True
            except OSError:
                alive = False
            if alive:
                sys.exit(
                    f"another laptop_agent.py is already running (pid {old_pid}) "
                    f"-- kill it first, or delete {_LOCK_PATH} if it's stale "
                    f"(e.g. the old process crashed without cleaning up)."
                )
        # Lock file exists but its pid is gone -- stale, safe to overwrite.
    _LOCK_PATH.write_text(str(os.getpid()))


def _release_singleton_lock() -> None:
    try:
        if _LOCK_PATH.exists() and _LOCK_PATH.read_text().strip() == str(os.getpid()):
            _LOCK_PATH.unlink()
    except OSError:
        pass

# Default pixel->arm transform path. Uses the same live config path every
# other script in this repo defaults to (config/pixel_arm_transform.json),
# so re-running calibrate_pixel_to_arm_auto.py updates what this agent uses
# automatically -- no separate copy/sync step needed on the Pi.
from vision.pixel_arm_transform import DEFAULT_PATH as _TRANSFORM_PATH


class LaptopAgent:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.backend = RoArmBackend()
        self._client = None
        self._stop = threading.Event()
        self._arm_connected = False

    def _resolve_camera_index(self) -> int | str:
        """cell.yaml's vision.camera_index as either an int (local USB
        webcam device index) or a URL string (e.g. a phone running IP
        Webcam, streamed over the LAN) -- see vision/capture.py. Needed on
        the Pi, where a Windows-only "Link to Windows" phone bridge isn't
        available.
        """
        raw = self.cfg.vision.get("camera_index", 1)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return str(raw)

    # -- broker setup ------------------------------------------------------ #

    def start(self) -> None:
        import paho.mqtt.client as mqtt

        conn = self.cfg.connection or {}
        mqtt_cfg = conn.get("mqtt", {}) if isinstance(conn.get("mqtt"), dict) else {}
        host = mqtt_cfg.get("host") or os.environ.get("MQTT_HOST") or "localhost"
        use_tls = bool(mqtt_cfg.get("use_tls", False))
        default_port = 8883 if use_tls else 1883
        port = int(mqtt_cfg.get("port") or os.environ.get("MQTT_PORT") or default_port)
        username = mqtt_cfg.get("username") or os.environ.get("MQTT_USERNAME")
        password = mqtt_cfg.get("password") or os.environ.get("MQTT_PASSWORD")
        client_id = mqtt_cfg.get("client_id") or f"laptop-agent-{os.getpid()}"

        # paho-mqtt >=2.0 defaults to CallbackAPIVersion.VERSION2, whose
        # on_connect/on_message signatures differ (5 args, not 4) from what
        # this code writes below -- silently mismatched callbacks don't
        # crash, they just never fire (paho swallows the TypeError
        # internally), which looked exactly like "connecting... then
        # nothing" with no visible error. Pin VERSION1 explicitly so the
        # existing 4-arg callback signatures actually get called.
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
            client_id=client_id, clean_session=True,
        )

        if use_tls:
            # AWS IoT Core (and most managed MQTT brokers) authenticate via
            # mutual TLS certs, NOT username/password. This was previously
            # MISSING entirely from laptop_agent.py -- connect() was
            # calling client.connect() plaintext against IoT Core's TLS-only
            # port 8883, which just hangs forever with no error (the broker
            # never completes a handshake with a non-TLS client). See
            # sim/backends/mqtt_proxy.py's connect() for the EC2-side
            # equivalent this mirrors.
            ca_cert = mqtt_cfg.get("ca_cert")
            cert_file = mqtt_cfg.get("cert_file")
            key_file = mqtt_cfg.get("key_file")
            if not (ca_cert and cert_file and key_file):
                raise RuntimeError(
                    "mqtt.use_tls is true but ca_cert/cert_file/key_file are "
                    "not all set in cell.yaml connection.mqtt -- required for "
                    "AWS IoT Core's mutual-TLS auth."
                )
            client.tls_set(ca_certs=ca_cert, certfile=cert_file, keyfile=key_file)
        elif username:
            client.username_pw_set(username, password)

        client.on_connect = self._on_connect
        client.on_message = self._on_message
        # Last-will: if the agent drops, publish an offline state so EC2 knows.
        client.will_set(
            proto.TOPIC_STATE,
            json.dumps({"connected": False, "ts": time.time()}),
            qos=proto.QOS_STATE,
            retain=False,
        )

        log.info("connecting to broker %s:%d ...", host, port)
        client.connect(host, port, keepalive=30)
        self._client = client

        # Background state publisher
        state_thread = threading.Thread(target=self._state_loop, daemon=True)
        state_thread.start()

        client.loop_forever()

    def _on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            log.error("broker connection failed rc=%s", rc)
            return
        client.subscribe(proto.TOPIC_CMD, qos=proto.QOS_CMD)
        log.info("connected; subscribed to %s", proto.TOPIC_CMD)

    # -- command handling -------------------------------------------------- #

    def _on_message(self, client, userdata, msg):
        try:
            cmd = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            log.warning("ignoring malformed command")
            return

        cmd_id = cmd.get("id")
        op = cmd.get("op")
        log.info("cmd %s: %s", cmd_id, op)

        result = {"id": cmd_id, "ok": False}
        try:
            self._dispatch_with_watchdog(op, cmd, result)
            result["ok"] = True
        except Exception as e:
            log.exception("command %s (%s) failed", cmd_id, op)
            result["error"] = str(e)

        # Attach current state to the result
        try:
            j = self.backend.read_joints()
            result["joints"] = j.as_dict()
            result["gripper"] = self.backend.gripper_state()
            result["held_block"] = self.backend.get_held_block()
        except Exception:
            pass

        self._client.publish(proto.TOPIC_RESULT, json.dumps(result), qos=proto.QOS_RESULT)

    def _dispatch_with_watchdog(self, op: str, cmd: dict, result: dict,
                                 timeout_s: float = 40.0) -> None:
        """Run _dispatch() with a hard wall-clock watchdog.

        Confirmed real issue: a serial read can occasionally get "wedged"
        at the OS/driver level (seen twice in a row after an abrupt Ctrl+C
        left the port in a bad state) -- Python's own retry/timeout logic
        inside roarm.py's _read_feedback() has a theoretical worst case of
        ~30s, but the actual observed hang was 60+ seconds with ZERO
        progress, meaning the underlying os-level read() itself was stuck
        below where our timeouts can reach. A wedged read like that can
        block forever with no exception ever raised on its own.

        Forcibly closing the serial port from a separate watchdog thread
        is a standard trick to unstick a hung blocking read on Linux --
        closing the file descriptor out from under a blocked read() call
        makes it raise an exception, breaking the hang without needing a
        physical USB replug. If the dispatch finishes normally before the
        timeout, the watchdog is cancelled and does nothing.
        """
        # Scale timeout for multi-placement ops
        if op in (proto.OP_EXECUTE_KIT_PLAN, proto.OP_EXECUTE_PLAN):
            n = len(cmd.get("placements") or [])
            timeout_s = max(timeout_s, n * 60.0)

        done = threading.Event()

        def watchdog():
            if not done.wait(timeout_s):
                log.error(
                    "command watchdog: op '%s' did not complete within %.0fs -- "
                    "forcibly closing the serial port to break a suspected "
                    "wedged read (this will make the current dispatch call "
                    "raise; the NEXT command will need a fresh connect)",
                    op, timeout_s,
                )
                try:
                    if self.backend._serial is not None:
                        self.backend._serial.close()
                except Exception:
                    log.exception("watchdog: failed to force-close serial port")
                self._arm_connected = False
                # Also reset the backend's OWN internal connected flag --
                # without this, RoArmBackend.connect()'s own
                # "if self._connected: return" guard short-circuits on the
                # NEXT command (since that flag was never touched by this
                # watchdog, only self._serial was closed), skipping the
                # actual reconnect entirely and crashing with
                # PortNotOpenError on the first write attempt. Confirmed via
                # a real run: the log showed "received 'gripper' before
                # connect -- auto-connecting first" (so laptop_agent.py's
                # OWN flag was correctly False) immediately followed by
                # PortNotOpenError -- meaning connect() was called but
                # silently did nothing.
                self.backend._connected = False

        watchdog_thread = threading.Thread(target=watchdog, daemon=True)
        watchdog_thread.start()
        try:
            self._dispatch(op, cmd, result)
        finally:
            done.set()

    def _dispatch(self, op: str, cmd: dict, result: dict) -> None:
        if op == proto.OP_CONNECT:
            self.backend.connect(self.cfg)
            self._arm_connected = True
            return

        # Every other op assumes connect() already ran (it initializes
        # gripper limits, motion speed, etc. on the backend instance) --
        # confirmed via a real crash (AttributeError: no _grip_min) when a
        # "gripper" command arrived before any "connect" had been sent in
        # this process's lifetime. Auto-connect here instead of crashing
        # with a confusing internal error, so an out-of-order command from
        # a client is still handled correctly.
        if not self._arm_connected:
            log.info("received '%s' before connect -- auto-connecting first", op)
            self.backend.connect(self.cfg)
            self._arm_connected = True

        if op == proto.OP_HOME:
            self.backend.home(self.cfg)
            return

        if op == proto.OP_INIT:
            # Re-runs the firmware's own MOVE_INIT (T:100) before gliding to
            # the XYZ home pose -- use this on demand (e.g. arm left in an
            # odd pose) without changing OP_HOME's regular behavior.
            self.backend.home(self.cfg, do_init=True)
            return

        if op == proto.OP_MOVE_TO:
            j = cmd.get("joints") or {}
            target = Joints.from_dict(j)
            self.backend.move_to(target, self.cfg)
            return

        if op == proto.OP_GRIPPER:
            action = cmd.get("action")
            if action == "open":
                self.backend.gripper_open(self.cfg)
            elif action == "close":
                self.backend.gripper_close(self.cfg)
            else:
                raise ValueError(f"unknown gripper action: {action}")
            return

        if op == proto.OP_ESTOP:
            self.backend.estop()
            return

        if op == proto.OP_CLEAR_ESTOP:
            self.backend.clear_estop()
            return

        if op == proto.OP_DETECT_PICK_PLACE:
            color = cmd.get("color")
            cell_color = cmd.get("cell_color")
            if not color or not cell_color:
                raise ValueError("detect_pick_place requires 'color' and 'cell_color'")
            duck = self.cfg.blocks.get(color)
            if duck is None:
                raise ValueError(f"unknown color '{color}'. Options: {list(self.cfg.blocks)}")

            camera_index = self._resolve_camera_index()
            offset_x = float(self.cfg.vision.get("pixel_arm_offset_x", 0.0))
            offset_y = float(self.cfg.vision.get("pixel_arm_offset_y", 0.0))
            # Board was physically changed from a pink grid line to a black
            # border -- configurable via cell.yaml (vision.grid_border) so
            # a future board change doesn't need a code edit here again.
            grid_border = self.cfg.vision.get("grid_border", "black")

            if self.cfg.stow_xyz is not None:
                self.backend.stow(self.cfg)

            detection = detect_pick_and_place(
                transform_path=str(_TRANSFORM_PATH),
                object_color_hex=duck.color,
                cell_color_hex=cell_color,
                camera_index=camera_index,
                grid_border=grid_border,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            result["pick_x"] = detection.pick_x
            result["pick_y"] = detection.pick_y
            result["place_x"] = detection.place_x
            result["place_y"] = detection.place_y
            # Log the detected coordinates directly (not just sent over MQTT)
            # so a run's console log alone is enough to check afterward
            # whether a detected target was near the reach envelope's edge
            # -- confirmed useful after a run where the arm ended up fully
            # outstretched and there was no local record of what pick/place
            # position it had actually been given.
            log.info(
                "detect_pick_place -> pick=(%.1f, %.1f) place=(%.1f, %.1f)",
                detection.pick_x, detection.pick_y, detection.place_x, detection.place_y,
            )
            return

        if op == proto.OP_EXECUTE_KIT_PLAN:
            placements_raw = cmd.get("placements")
            if not placements_raw:
                raise ValueError("execute_kit_plan requires a non-empty 'placements' list")
            placements = [(p["color"], p["cell"]) for p in placements_raw]

            color_hex_map = {bid: b.color for bid, b in self.cfg.blocks.items()}
            camera_index = self._resolve_camera_index()
            offset_x = float(self.cfg.vision.get("pixel_arm_offset_x", 0.0))
            offset_y = float(self.cfg.vision.get("pixel_arm_offset_y", 0.0))
            grid_border = self.cfg.vision.get("grid_border", "black")
            grid_rows = int(self.cfg.vision.get("kit_grid_rows", 4))
            grid_cols = int(self.cfg.vision.get("kit_grid_cols", 2))
            pink_hex = self.cfg.vision.get("pink_grid_hex", "#714951")
            # Place height: prefer a dedicated grid_top_z (raised platform)
            # if configured, else fall back to the normal grip_z (same
            # logic as pick_into_grid_cell's place_z on the EC2 manager
            # side -- kept consistent here since this Pi-local plan runs
            # entirely without that manager).
            place_z = float(self.cfg.heights.get("grid_top_z", self.cfg.heights["grip_z"]))

            if self.cfg.stow_xyz is not None:
                self.backend.stow(self.cfg)

            # ONE capture resolves every placement in the plan ("stow
            # once") -- deliberately not re-detecting per placement, since
            # same-colored objects are interchangeable (no need to track
            # which specific one was picked) and re-stowing/re-capturing
            # for every one of up to 6 placements would add real time
            # without a correctness benefit, PROVIDED objects don't shift
            # between detection and pick -- if that assumption turns out
            # wrong in practice, switch back to one detect_pick_place call
            # per placement instead.
            resolved = detect_kit_plan(
                transform_path=str(_TRANSFORM_PATH),
                placements=placements,
                color_hex_map=color_hex_map,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                camera_index=camera_index,
                grid_border=grid_border,
                pink_hex=pink_hex,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            log.info("execute_kit_plan: resolved %d placement(s)", len(resolved))

            completed = []
            for i, placement in enumerate(resolved):
                log.info(
                    "execute_kit_plan: placement %d/%d -- %s -> %s "
                    "pick=(%.1f,%.1f) place=(%.1f,%.1f)",
                    i + 1, len(resolved), placement.color, placement.cell,
                    placement.pick_x, placement.pick_y,
                    placement.place_x, placement.place_y,
                )
                run_pick_and_place(
                    self.backend, self.cfg,
                    pick_x=placement.pick_x, pick_y=placement.pick_y,
                    place_x=placement.place_x, place_y=placement.place_y,
                    place_z=place_z,
                )
                completed.append({"color": placement.color, "cell": placement.cell})

            result["completed"] = completed
            return

        if op == proto.OP_DETECT_KIT_PLAN:
            # Detect all blocks and return coordinates without executing.
            # Command: {"placements": [{"color":"green","cell":"A1","seq":1}, ...]}
            # Result:  {"placements": [{"color":"green","cell":"A1","seq":1,
            #           "pick_x":..,"pick_y":..,"place_x":..,"place_y":..}, ...]}
            placements_raw = cmd.get("placements")
            if not placements_raw:
                raise ValueError("detect_kit_plan requires a non-empty 'placements' list")

            color_hex_map = {bid: b.color for bid, b in self.cfg.blocks.items()}
            camera_index  = self._resolve_camera_index()
            offset_x = float(self.cfg.vision.get("pixel_arm_offset_x", 0.0))
            offset_y = float(self.cfg.vision.get("pixel_arm_offset_y", 0.0))
            grid_border = self.cfg.vision.get("grid_border", "black")
            grid_rows   = int(self.cfg.vision.get("kit_grid_rows", 4))
            grid_cols   = int(self.cfg.vision.get("kit_grid_cols", 2))
            pink_hex    = self.cfg.vision.get("pink_grid_hex", "#714951")

            if self.cfg.stow_xyz is not None:
                self.backend.stow(self.cfg)

            placements_input = [(p["color"], p["cell"]) for p in placements_raw]
            resolved = detect_kit_plan(
                transform_path=str(_TRANSFORM_PATH),
                placements=placements_input,
                color_hex_map=color_hex_map,
                grid_rows=grid_rows, grid_cols=grid_cols,
                camera_index=camera_index, grid_border=grid_border,
                pink_hex=pink_hex, offset_x=offset_x, offset_y=offset_y,
            )

            # Merge seq from input into resolved output
            seq_map = {(p["color"], p["cell"]): p.get("seq", i)
                       for i, p in enumerate(placements_raw)}
            out = []
            for r in resolved:
                out.append({
                    "color":   r.color,
                    "cell":    r.cell,
                    "seq":     seq_map.get((r.color, r.cell), 0),
                    "pick_x":  round(r.pick_x,  2),
                    "pick_y":  round(r.pick_y,  2),
                    "place_x": round(r.place_x, 2),
                    "place_y": round(r.place_y, 2),
                })
            result["placements"] = out
            log.info("detect_kit_plan: returned %d placement(s) with coordinates", len(out))
            return

        if op == proto.OP_EXECUTE_PLAN:
            # Execute a pre-resolved plan (coordinates provided by cloud agent).
            placements_raw = cmd.get("placements")
            if not placements_raw:
                raise ValueError("execute_plan requires a non-empty 'placements' list")

            # Sort by seq
            ordered = sorted(placements_raw, key=lambda p: p.get("seq", 0))
            place_z = float(self.cfg.heights.get("grid_top_z", self.cfg.heights["grip_z"]))

            # Load corrections from config/ (Pi-side) or most recent runs/ folder
            from pathlib import Path as _Path
            from correction import load_settle_maps, apply_corrections
            config_dir = _Path("config")
            # Prefer config/ settle maps (copied from latest calibration)
            if (config_dir / "pick_settle_map.json").exists():
                pick_points, pick_model, _, _, place_cell_offsets = \
                    load_settle_maps(config_dir)
            else:
                # Fall back to latest runs/ folder
                runs_dir = _Path("runs")
                run_candidates = sorted(
                    [d for d in runs_dir.iterdir()
                     if d.is_dir() and (d / "pixel_arm_transform.json").exists()],
                    reverse=True,
                ) if runs_dir.exists() else []
                if run_candidates:
                    pick_points, pick_model, _, _, place_cell_offsets = \
                        load_settle_maps(run_candidates[0])
                else:
                    pick_points, pick_model, place_cell_offsets = [], None, {}
                    log.warning("execute_plan: no settle maps found, running without corrections")

            completed = []
            for i, p in enumerate(ordered):
                raw_pick_x  = float(p["pick_x"])
                raw_pick_y  = float(p["pick_y"])
                raw_place_x = float(p["place_x"])
                raw_place_y = float(p["place_y"])

                pick_x, pick_y, place_x, place_y = apply_corrections(
                    raw_pick_x, raw_pick_y, raw_place_x, raw_place_y,
                    pick_points, pick_model, place_cell_offsets,
                )
                log.info(
                    "execute_plan: %d/%d -- %s -> %s "
                    "pick=(%.1f,%.1f) [corr: %+.1f,%+.1f] "
                    "place=(%.1f,%.1f) [corr: %+.1f,%+.1f]",
                    i+1, len(ordered), p.get("color","?"), p.get("cell","?"),
                    pick_x, pick_y, pick_x-raw_pick_x, pick_y-raw_pick_y,
                    place_x, place_y, place_x-raw_place_x, place_y-raw_place_y,
                )
                run_pick_and_place(
                    self.backend, self.cfg,
                    pick_x=pick_x, pick_y=pick_y,
                    place_x=place_x, place_y=place_y,
                    place_z=place_z,
                )
                completed.append({"color": p.get("color"), "cell": p.get("cell"), "seq": p.get("seq")})

            result["completed"] = completed
            log.info("execute_plan: completed %d placement(s)", len(completed))
            return

        raise ValueError(f"unknown op: {op}")

    # -- live state publisher ---------------------------------------------- #

    def _state_loop(self) -> None:
        period = 1.0 / proto.STATE_PUBLISH_HZ
        while not self._stop.is_set():
            if self._client is not None and self._arm_connected:
                try:
                    j = self.backend.read_joints()
                    payload = {
                        "ts": time.time(),
                        "joints": j.as_dict(),
                        "gripper": self.backend.gripper_state(),
                        "held_block": self.backend.get_held_block(),
                        "connected": self.backend.is_connected(),
                        "block_positions": self.backend.get_block_positions(),
                    }
                    self._client.publish(
                        proto.TOPIC_STATE, json.dumps(payload), qos=proto.QOS_STATE
                    )
                except Exception:
                    pass
            time.sleep(period)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    _acquire_singleton_lock()
    try:
        agent = LaptopAgent()
        agent.start()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        _release_singleton_lock()


if __name__ == "__main__":
    main()
