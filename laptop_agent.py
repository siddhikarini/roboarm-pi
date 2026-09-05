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
import threading
import time

from sim.config import load_config
from sim.kinematics import Joints
from sim.backends.roarm import RoArmBackend
from sim import mqtt_protocol as proto
from vision.detect_pick_place import detect_pick_and_place

log = logging.getLogger("laptop_agent")

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
            self._dispatch(op, cmd, result)
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

            if self.cfg.stow_xyz is not None:
                self.backend.stow(self.cfg)

            detection = detect_pick_and_place(
                transform_path=str(_TRANSFORM_PATH),
                object_color_hex=duck.color,
                cell_color_hex=cell_color,
                camera_index=camera_index,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            result["pick_x"] = detection.pick_x
            result["pick_y"] = detection.pick_y
            result["place_x"] = detection.place_x
            result["place_y"] = detection.place_y
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
    agent = LaptopAgent()
    try:
        agent.start()
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    main()
