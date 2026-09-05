"""Shared MQTT protocol for the distributed block-sorting architecture.

This module is imported by BOTH sides of the relay:
  - The EC2 side (MqttProxyBackend inside the MCP server)
  - The booth laptop side (the standalone arm agent)

It defines the topic names and message schemas so both ends stay in sync.

Architecture
------------
    Karini → MCP server (EC2) → MqttProxyBackend
                                     │ publishes commands
                                     ▼
                              MQTT Broker (EC2)
                                     │
                                     ▼
                        Laptop agent (subscribes)
                                     │ runs roarm.py locally on the LAN
                                     ▼
                              RoArm-M2-Pro (WiFi)

Why MQTT: both EC2 and the laptop connect OUTBOUND to the broker, so the
laptop never needs a public IP or port forwarding. NAT allows outbound.

Latency design: the EC2 side sends one command per motion primitive
(move_to / gripper / home) and waits for a single "done" reply. The tight
feedback-polling loop (T:105 every ~50ms) runs on the LAPTOP over the LAN,
never across the internet. So each primitive is ONE internet round-trip,
not dozens.

Topics
------
    robo/cmd       EC2 publishes commands here; laptop subscribes.
    robo/result    Laptop publishes command results here; EC2 subscribes.
    robo/state     Laptop publishes live joint/gripper state (~10 Hz);
                   EC2 subscribes to feed read_joints() and the 3D viz.

Message shapes
--------------
Command (robo/cmd), published by EC2:
    {
      "id": "cmd_<hex>",         # correlation id
      "op": "move_to" | "gripper" | "home" | "estop" | "clear_estop" | "connect" | "detect" | "detect_pick_place",
      # op-specific payload:
      "joints": {"base":.., "shoulder":.., "elbow":.., "wrist":..},  # move_to
      "action": "open" | "close",                                    # gripper
      # "detect" has no extra payload -- the laptop runs its own camera
      # pipeline and reports back whatever it currently sees.
      "color": "green",                       # detect_pick_place: which
      "cell_color": "#38846f",                # cfg.blocks color to pick,
                                               # and the target cell's own
                                               # background color
    }

Result (robo/result), published by laptop:
    {
      "id": "cmd_<hex>",         # echoes the command id
      "ok": true | false,
      "error": "<message>",      # present when ok is false
      # op-specific extras:
      "joints": {...},           # current joints after the op (degrees)
      "gripper": "open|closed|gripping",
      "held_block": "red" | null,
      "blocks": [{"id":.., "color":.., "x":.., "y":..}, ...],  # detect only
      "pick_x": .., "pick_y": .., "place_x": .., "place_y": ..,  # detect_pick_place only
    }

State (robo/state), published by laptop continuously:
    {
      "ts": <epoch seconds>,
      "joints": {"base":.., "shoulder":.., "elbow":.., "wrist":..},  # degrees
      "gripper": "open|closed|gripping",
      "held_block": "red" | null,
      "connected": true | false,
      "block_positions": {"red": {"x":.., "y":..}, ...},
    }
"""

from __future__ import annotations

import secrets

# -- topics ---------------------------------------------------------------- #

TOPIC_CMD = "robo/cmd"
TOPIC_RESULT = "robo/result"
TOPIC_STATE = "robo/state"

# QoS 1 = at-least-once delivery. Good for commands (we dedupe by id) and
# results. State uses QoS 0 (fire-and-forget, freshest wins).
QOS_CMD = 1
QOS_RESULT = 1
QOS_STATE = 0

# How often the laptop publishes robo/state (Hz).
STATE_PUBLISH_HZ = 10

# Default seconds the EC2 side waits for a command result before giving up.
DEFAULT_CMD_TIMEOUT_S = 20.0


# -- operation names ------------------------------------------------------- #

OP_CONNECT = "connect"
OP_HOME = "home"
OP_MOVE_TO = "move_to"
OP_GRIPPER = "gripper"
OP_ESTOP = "estop"
OP_CLEAR_ESTOP = "clear_estop"
OP_DETECT = "detect"
# Distinct from OP_DETECT (which detects each cell.yaml block color's own
# position): this detects a PICK position (an object) and a PLACE position
# (a target cell inside a pink grid, marked by its own background color) --
# see vision/detect_pick_place.py. Kept as a separate op instead of
# overloading OP_DETECT's result shape (list of {id,color,x,y} blocks) with
# a different, unrelated shape (single pick_x/pick_y/place_x/place_y).
OP_DETECT_PICK_PLACE = "detect_pick_place"


def new_cmd_id() -> str:
    """Generate a unique correlation id for a command."""
    return "cmd_" + secrets.token_hex(6)
