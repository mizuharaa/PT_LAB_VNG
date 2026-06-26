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

The pipeline runs on **three threads** so that hand *detection* is never blocked
by hand *control*:

```
[camera thread]   src/camera.py        latest-frame-only capture (drops stale frames)
      │ Camera.read()
      ▼
[detection thread]  ── THE HOT PATH (keep it fast) ──     src/pipeline.py:DetectionWorker
   MediaPipe landmarks      src/hand_tracker.py  (pluggable backend)
     → select best hand     src/hand_tracker.py
     → handedness/mirror     src/transform.py
     → landmarks → curls     src/gesture_mapper.py
     → calibration norm      src/calibration.py
     → smoothing             src/smoothing.py
     → publish PoseSample ──┐
                            │  LatestSlot  (newest pose wins; stale ones dropped)
[control thread]  ◄─────────┘                            src/pipeline.py:ControlWorker
   safety clamp/watchdog    src/safety.py
     → hand controller       src/hand_controller.py  (SDK / serial / mock)
```

**Why split detection from control?** The robotic-hand SDK is the slow, blocking,
not-yet-portable part — the real VNG × Paxini SDK ships **x86-only today and does
not run on the RB3 (aarch64)**. Coupling it to detection would drag tracking
latency down to SDK speed. Instead the detection thread runs as fast as the
camera + model allow and drops poses the control side can't keep up with, so
responsiveness is bounded by detection, not by hardware I/O. MediaPipe inference
and serial/SDK I/O both release the GIL during native work, so Python threads
give real overlap here; the `LatestSlot` boundary is also where this could become
a process/queue later if a backend needs it.

Each stage is isolated so the tracker (MediaPipe today) or the hand backend
(SDK/serial) can be swapped without touching the rest of the pipeline.

```
rb3_hand_mimic/
  README.md  requirements.txt  config.yaml  main.py
  src/   camera, hand_tracker, gesture_mapper, calibration, transform,
         smoothing, safety, hand_controller, pipeline, metrics,
         diagnostics, utils
  tools/ discover_cameras, test_camera, discover_hand_usb, test_serial_hand,
         record_calibration
```

`pipeline.py` holds the async runtime (LatestSlot + DetectionWorker +
ControlWorker); `metrics.py` measures detection/end-to-end responsiveness.

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
>
> **Headless RB3 (no display):** prefer `opencv-python-headless` to avoid pulling
> `libGL`/X11 system deps — `pip install opencv-python-headless`. The demo's
> `--headless` / `--no-window` / `--benchmark` modes never call `cv2.imshow`.
> The whole stack is pure Python and runs identically on x86_64 and aarch64; only
> the binary wheels differ per architecture.

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
python main.py --benchmark 15           # 15s detection-responsiveness report
python main.py --benchmark --debug      # benchmark with the live window
python main.py --camera-index 2 --dry-run --debug
python main.py --list-cameras
python main.py --list-serial
python main.py --no-window
python main.py --log-level INFO
```

| Flag | Effect |
|------|--------|
| `--config PATH` | config file (default `config.yaml`) |
| `--debug` | show window, draw landmarks + curl bars + responsiveness overlay, DEBUG logs |
| `--headless` | no GUI; logs a status line every few seconds (for tmux/systemd) |
| `--no-window` | force-disable the window even in debug |
| `--dry-run` | use `MockHandController` (prints commands, no hardware) |
| `--benchmark [SECONDS]` | measure detection responsiveness; prints latency percentiles. Omit the value to run until Ctrl+C. No window unless `--debug` is also set |
| `--camera-index N` | force a camera index, overriding config |
| `--list-cameras` | run discovery and exit |
| `--list-serial` | list candidate serial ports and exit |
| `--log-level` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

### Measuring responsiveness

`--benchmark` is the fastest way to answer “is detection fast enough?” on a new
machine or chip. It runs the full async pipeline (control side uses the
placeholder SDK by default, so no hardware is needed) and prints, per second:

```
detect 58.9fps lat p50/p90 12.4/16.1ms | e2e p50/p90 18.0/24.3ms | ctrl 119.4hz det-rate 96% cmds 441
```

and a percentile summary on exit:

- **detection latency** — ms per frame in the tracker+mapper hot path (the number
  to drive down for sensitivity to movement)
- **end-to-end latency** — camera capture → command handed to the controller
- **pose age at control** — how stale a pose is when control consumes it
- **detection rate** — % of processed frames where a hand was found

The same numbers appear live in the `--debug` overlay and in each `--headless`
status line, so you can compare x86 dev vs. the RB3 directly.

---

## Debug vs. headless

**Debug** shows the OpenCV window with MediaPipe landmarks, per-finger curl
bars, detection FPS, detection/end-to-end latency, control rate, selected camera
index, detected handedness, hand-controller status, tracking state
(active/holding/returning/rest), and the current command values.

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

## Robotic hand control (backends — configurable)

Control is abstracted in `src/hand_controller.py` behind `BaseHandController`.
Every backend implements one method, `_apply(values)`, that receives normalized
per-finger curls (`0..1`); rate-limiting lives in the base class. Pick a backend
in `config.yaml → hand.controller`:

- **`sdk`** *(default)* — drives the hand through a vendor SDK (`HandSdk`). The
  real VNG × Paxini SDK is **x86-only today and does not run on the RB3
  (aarch64)**, so this currently uses **`PlaceholderHandSdk`**: it validates and
  logs the finger commands and runs on *any* architecture, so the full async
  pipeline works end-to-end right now. When the ARM64 binding arrives,
  implement `HandSdk` (`connect` / `set_curls` / `is_connected` / `disconnect`),
  wire it into `load_hand_sdk()`, and set `hand.sdk.library` — nothing upstream
  changes. The placeholder methods are marked `TODO(vng-paxini-arm64)` 1:1 with
  the calls the production SDK is expected to expose.
- **`serial`** — text protocol over a serial/USB port (`pyserial`). Choose a wire
  format in `hand.protocol`:
  1. **csv** — `thumb,index,middle,ring,pinky\n`
  2. **labeled_csv** — `FINGER,thumb,index,middle,ring,pinky\n`
  3. **json** — `{"thumb":0.1,"index":0.7,...}\n`

  `value_type: normalized` sends `0..1`; `value_type: servo` maps to
  `servo_min..servo_max`. The serial controller auto-detects the port
  (`/dev/serial/by-id` → `ttyACM*` → `ttyUSB*`), reconnects with backoff, uses
  low-timeout non-blocking writes, and rate-limits commands.
- **`mock`** — logs commands only (also forced by `--dry-run`).

Because control lives on its own thread, swapping the x86 SDK for an ARM64 one
(or even a slow gRPC shim to a co-located x86 helper) changes **only** the
backend — detection latency is unaffected.

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
| `hand: DISCONNECTED` | serial backend: check `--list-serial`, `dialout` group, `hand.port`. SDK backend: real ARM64 SDK not wired yet (placeholder always connects). Demo keeps running either way |
| Detection feels slow | run `--benchmark`; drop to 320×240@60, keep `model_complexity: 0`, improve lighting |
| Jittery fingers | lower `ema_alpha`, raise `deadband`, improve lighting |
| Laggy fingers | raise `ema_alpha`, lower `deadband`, drop to 320×240@60 |

---

## License / attribution

Internal demo for VNG × Paxini on Qualcomm RB3 Gen 2. MediaPipe Hands © Google.
