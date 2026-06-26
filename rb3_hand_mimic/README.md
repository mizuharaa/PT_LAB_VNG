# rb3_hand_mimic

Real-time robotic hand mimic demo for the **Qualcomm RB3 Gen 2** (Any ARM-based device).
Currently this only works on local device for detection purposes, to properly communicate with the robotics hand we need a proper SDK file for arm64 arch.
A visitor faces their palm at the RB3 **tracking camera**; the software detects
hand landmarks, maps the pose to finger curls, corrects mirroring/handedness,
smooths and safety-clamps the values, and drives the **VNG × Paxini** robotic
hand over USB/serial.

Priorities, in order: **low latency → stable tracking → correct mirroring →
safe commands → no crashes → diagnostics → clean modular code.**

---

## Architecture

```
RB3 tracking camera
  └─ OpenCV latest-frame capture thread        src/camera.py
       └─ MediaPipe hand landmarks             src/hand_tracker.py   (pluggable)
            └─ select best hand                src/hand_tracker.py
                 └─ handedness + mirror xform  src/transform.py
                      └─ landmarks → curls     src/gesture_mapper.py
                           └─ calibration norm src/calibration.py
                                └─ smoothing   src/smoothing.py
                                     └─ safety  src/safety.py
                                          └─ Paxini hand controller  src/hand_controller.py
```

Each stage is isolated so the tracker (MediaPipe today) or the hand protocol
(unknown today) can be swapped without touching the rest of the pipeline.

```
rb3_hand_mimic/
  README.md  requirements.txt  config.yaml  main.py
  src/   camera, hand_tracker, gesture_mapper, calibration, transform,
         smoothing, hand_controller, safety, diagnostics, utils
  tools/ discover_cameras, test_camera, discover_hand_usb, test_serial_hand,
         record_calibration
```

---

## Installation (on the RB3 Gen 2)

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git v4l-utils tmux

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

> **If MediaPipe will not install on the RB3 (ARM64):** the tracker sits behind
> an interface (`src/hand_tracker.py`). You can bring the rest of the system up
> with `--dry-run` immediately, and later add a TFLite/ONNX/QNN tracker without
> changing the pipeline. If OpenCV's pip wheel misbehaves, use the distro build:
> `sudo apt install -y python3-opencv`.

Add your user to the `video` and `dialout` groups so cameras and serial ports
are accessible without root (log out/in afterwards):

```bash
sudo usermod -aG video,dialout "$USER"
```

---

## Bring-up sequence

### 1. Find the tracking camera

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
python tools/discover_cameras.py
```

`discover_cameras.py` probes indexes 0–8, prints resolution/FPS/name, and saves
one frame per camera to `debug_frames/`. Open those JPEGs and pick the **smaller
tracking camera** (lower resolution, wider/mono look) — not the big main lens.
Then either keep `camera.index: "auto"` (it biases toward tracking-style names)
or set the index explicitly in `config.yaml`.

Live-verify a specific camera:

```bash
python tools/test_camera.py --camera 2 --width 640 --height 480 --fps 30
python tools/test_camera.py --camera 2 --width 320 --height 240 --fps 60
```

### 2. Find the robotic hand USB/serial port

```bash
lsusb
ls -l /dev/serial/by-id/
ls /dev/ttyUSB* /dev/ttyACM*
dmesg -w            # watch while you replug the hand
python tools/discover_hand_usb.py --watch
```

`--watch` detects a newly-appearing port when you replug the hand. Set
`hand.port` in `config.yaml` (or leave `"auto"`).

### 3. Safe protocol test (no camera needed)

```bash
python tools/test_serial_hand.py --pose open
python tools/test_serial_hand.py --pose half
python tools/test_serial_hand.py --pose closed --yes
python tools/test_serial_hand.py --pose open --dry-run   # prints packets only
```

All values are clamped to `[0,1]`; `closed`/`random` ask for confirmation
unless `--yes`. The tool always finishes by sending the rest pose.

### 4. Calibrate to the user / camera placement

```bash
python tools/record_calibration.py --config config.yaml
```

Show an open palm (held ~1s), then a fist (~1s). Per-finger open/closed ranges
are written to `calibration.yaml`. The tool warns if open/closed are too close.

### 5. Run

```bash
# Dry run (no hardware) with debug window:
python main.py --dry-run --debug

# Real run with debug overlay:
python main.py --config config.yaml --debug

# Force a camera index, dry-run:
python main.py --camera-index 2 --dry-run --debug
```

---

## CLI

```
python main.py --config config.yaml --debug
python main.py --config config.yaml --headless
python main.py --dry-run --debug
python main.py --camera-index 2 --dry-run --debug
python main.py --list-cameras
python main.py --list-serial
python main.py --no-window
python main.py --log-level INFO
```

| Flag | Effect |
|------|--------|
| `--config PATH` | config file (default `config.yaml`) |
| `--debug` | show window, draw landmarks + curl bars + FPS/latency, DEBUG logs |
| `--headless` | no GUI; logs a status line every few seconds (for tmux/systemd) |
| `--no-window` | force-disable the window even in debug |
| `--dry-run` | use `MockHandController` (prints commands, no hardware) |
| `--camera-index N` | force a camera index, overriding config |
| `--list-cameras` | run discovery and exit |
| `--list-serial` | list candidate serial ports and exit |
| `--log-level` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

---

## Debug vs. headless

**Debug** shows the OpenCV window with MediaPipe landmarks, per-finger curl
bars, FPS, latency, selected camera index, detected handedness, serial status,
tracking state (active/holding/returning/rest), and the current command values.

**Headless** (demo install inside the CAD enclosure) runs without a GUI and logs
a status line every `debug.headless_status_seconds`. Run it under tmux so it
survives SSH disconnects:

```bash
tmux new -s handdemo
source .venv/bin/activate
python main.py --config config.yaml --headless
# detach: Ctrl-b then d   |   reattach: tmux attach -t handdemo
```

For unattended auto-start, wrap the same command in a `systemd` service.

---

## Mirroring / handedness (read this if the robot mirrors wrong)

The demo assumes the visitor's **palm faces the camera**. Several independent
flips can each invert left/right; all are in `config.yaml → transform`:

- `camera_mirror`: is the captured image a selfie (mirrored)? When true, the
  tracker's reported handedness is corrected back to the **physical** hand.
- `output_mirror`: mirror the *finger order* (index↔ring, thumb↔pinky) if the
  robot's finger indexing runs opposite to the human hand.
- `invert_fingers`: set true if the robot **opens** when the user **closes**.
- `robot_hand` / `user_hand`: declared handedness (logged; drives preview).
- `finger_order`: reorder the 5 values to match the hardware packet order.

Finger curl is a per-finger scalar, so closing your index always closes the
robot's index regardless of mirroring — mirroring mainly matters for the preview
overlay and for hardware that expects the opposite finger ordering. The debug
overlay prints `cam_mirror`, `out_mirror`, detected handedness, and live values
so you can correct it on-site in seconds.

---

## Robotic hand protocol (Paxini — configurable)

The exact protocol is abstracted in `src/hand_controller.py`. Choose a wire
format in `config.yaml → hand.protocol`:

1. **csv** — `thumb,index,middle,ring,pinky\n`
2. **labeled_csv** — `FINGER,thumb,index,middle,ring,pinky\n`
3. **json** — `{"thumb":0.1,"index":0.7,"middle":0.5,"ring":0.2,"pinky":0.0}\n`

`value_type: normalized` sends `0..1`; `value_type: servo` maps to
`servo_min..servo_max` (e.g. 0–180). To add a custom packet/checksum later,
add a branch to `format_command()` or subclass `BaseHandController` — nothing
upstream changes. The serial controller auto-detects the port
(`/dev/serial/by-id` → `ttyACM*` → `ttyUSB*`), reconnects with backoff, uses
low-timeout non-blocking writes, and rate-limits commands.

---

## Safety behavior

- Every output is clamped to `[output_min, output_max]` and rate-limited per
  finger (`max_step_per_command`) to avoid servo slamming.
- **Tracking lost:** hold the last pose for `hold_seconds`, then ease to the
  rest pose over `return_seconds`.
- **Serial disconnect:** never crashes; keeps tracking and retries with backoff.
- **Watchdog:** stale poses beyond `watchdog_seconds` force return-to-rest.
- **Ctrl+C / SIGTERM:** sends the rest pose, then closes camera and serial port.

---

## Performance tuning

- Use the **tracking camera** first (lower latency than the main lens).
- Start at **640×480 @ 30 FPS**; try **320×240 @ 60 FPS** if the sensor supports
  it and you want more responsiveness.
- Keep MediaPipe `model_complexity: 0` for speed.
- Improve **lighting** if landmarks flicker — the single biggest stability win.
- Keep camera **`buffer_size: 1`** and the latest-frame-only capture (default):
  the AI loop always processes the freshest frame and drops the rest.
- Use **`--headless`** in production to drop GUI overhead.
- Keep **`command_rate_hz: 30`** — sending faster rarely helps and can saturate
  the serial link.
- If output feels **jittery**, lower `smoothing.ema_alpha` or raise `deadband`.
  If it feels **laggy**, raise `ema_alpha` (toward 1.0) or lower `deadband`.
- Serial I/O runs through a non-blocking, rate-limited writer so it never blocks
  camera/tracking.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No working camera found` | `python tools/discover_cameras.py`; set `camera.index`; check `video` group |
| `MediaPipe is not installed` | `pip install mediapipe`, or run `--dry-run`, or add an alt tracker |
| Robot mirrored / opposite | flip `transform.output_mirror` / `invert_fingers` (see overlay) |
| Robot finger order wrong | set `hand`/`transform.finger_order` to the hardware order |
| `serial: DISCONNECTED` | check `--list-serial`, `dialout` group, `hand.port`; demo keeps running |
| Jittery fingers | lower `ema_alpha`, raise `deadband`, improve lighting |
| Laggy fingers | raise `ema_alpha`, lower `deadband`, drop to 320×240@60 |

---

## License / attribution

Internal demo for VNG × Paxini on Qualcomm RB3 Gen 2. MediaPipe Hands © Google.
