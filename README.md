# roboarm-pi

Raspberry Pi-side agent for the RoArm-M2-Pro block-sorting cell. Runs
locally on the Pi wired to the arm (USB serial) and camera (Link to
Windows / USB webcam), and relays commands to/from the MCP/EC2 side over
MQTT (AWS IoT Core or a local broker).

This is a PARED-DOWN subset of the full project (see the main `demo`
repo/workspace) -- only the pieces the Pi actually needs to run
`laptop_agent.py`. No MCP server, no fastmcp, no VLM/OpenAI code, no 3D
viz bridge -- those live on EC2 and aren't needed here.

## What's included

- `laptop_agent.py` -- the agent: subscribes to `robo/cmd`, runs the
  corresponding arm/camera action LOCALLY, publishes the result to
  `robo/result`, and streams live state to `robo/state`.
- `sim/` -- config loading, kinematics, MQTT protocol, and the real arm
  backend (`sim/backends/roarm.py`, USB serial control).
- `vision/` -- camera capture, board cropping, color-based detection, and
  pixel->arm coordinate conversion.
- `config/cell.yaml` -- geometry, safety limits, MQTT connection settings.
- `config/pixel_arm_transform.json` -- the calibrated pixel->arm mapping
  (re-run `calibrate_pixel_to_arm_auto.py` from the main repo and copy the
  updated file here whenever the camera moves).
- `config/board_crop.json` -- the board's cropped region (same
  re-run-and-copy note applies if the camera moves).

## NOT included (stays on EC2 / the main dev machine)

- `sim/mcp_server.py`, `sim/manager.py`, `sim/bridge.py`,
  `sim/backends/mqtt_proxy.py`, `sim/backends/simulated.py`
- Any `calibrate_*.py` / `test_*.py` / debug scripts -- those are
  dev-time tools, run from the main repo against the real hardware, not
  part of the running Pi agent.
- IoT Core certificates (`.pem`/`.crt`/`.key`) -- deliver these
  separately (scp, or AWS Secrets Manager), never commit them.

## Setup on the Pi

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Edit `config/cell.yaml`:
- `connection.port` -- the arm's serial port (e.g. `/dev/ttyUSB0`)
- `connection.mqtt` -- broker host/port, TLS cert paths (see the main
  repo's cell.yaml comments for the AWS IoT Core setup)
- `vision.camera_index` -- verify with `python -m vision.capture --index 0`
  (adjust index until you get a real frame; camera enumeration can differ
  on the Pi vs. the dev machine)

## Running

```bash
python laptop_agent.py
```

Runs forever, reconnecting to the broker and processing commands until
stopped (Ctrl+C).

## Updating calibration

Whenever the camera or board physically moves, re-run calibration on the
DEV MACHINE (not the Pi) against the real arm + camera, then copy the
updated files here:

```
config/pixel_arm_transform.json
config/board_crop.json
```
