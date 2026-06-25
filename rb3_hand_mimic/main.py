#!/usr/bin/env python3
"""rb3_hand_mimic -- real-time robotic hand mimic for the Qualcomm RB3 Gen 2.

Pipeline:
  tracking camera -> latest-frame capture -> MediaPipe landmarks -> select hand
  -> handedness/mirror transform -> finger curls -> calibration normalize
  -> smoothing/deadband -> safety clamp/watchdog -> Paxini hand controller

Run examples:
  python main.py --config config.yaml --debug
  python main.py --config config.yaml --headless
  python main.py --dry-run --debug
  python main.py --camera-index 2 --dry-run --debug
  python main.py --list-cameras
  python main.py --list-serial
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import Optional

# Ensure local package import works regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.calibration import load_calibration  # noqa: E402
from src.camera import (  # noqa: E402
    CameraConfig,
    discover_cameras,
    open_camera_from_config,
)
from src.diagnostics import Diagnostics, HeadlessReporter, draw_overlay  # noqa: E402
from src.gesture_mapper import GestureMapper, HandPose  # noqa: E402
from src.hand_controller import (  # noqa: E402
    HandConfig,
    create_controller,
    list_serial_ports,
)
from src.hand_tracker import TrackerConfig, create_tracker  # noqa: E402
from src.safety import SafetyConfig, SafetyManager  # noqa: E402
from src.smoothing import Smoother, SmoothingConfig  # noqa: E402
from src.transform import Transform, TransformConfig  # noqa: E402
from src.utils import get_logger, load_config, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time robotic hand mimic for the Qualcomm RB3 Gen 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    p.add_argument("--config", default=default_cfg, help="Path to config.yaml")
    p.add_argument("--debug", action="store_true", help="Show window + verbose overlay")
    p.add_argument("--headless", action="store_true", help="No GUI; log status periodically")
    p.add_argument("--no-window", action="store_true", help="Disable the OpenCV window")
    p.add_argument("--dry-run", action="store_true", help="Use MockHandController (no hardware)")
    p.add_argument("--camera-index", type=int, default=None, help="Force a camera index")
    p.add_argument("--log-level", default=None, help="DEBUG|INFO|WARNING|ERROR")
    p.add_argument("--list-cameras", action="store_true", help="Discover cameras and exit")
    p.add_argument("--list-serial", action="store_true", help="List serial ports and exit")
    return p.parse_args()


class HandMimicApp:
    """Owns all pipeline objects and the main loop, with clean shutdown."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cfg = load_config(args.config)

        log_level = args.log_level or self.cfg.get("logging", {}).get("level", "INFO")
        if args.debug and not args.log_level:
            log_level = "DEBUG"
        self.log = setup_logging(
            level=log_level,
            log_file=self.cfg.get("logging", {}).get("file"),
            fmt=self.cfg.get("logging", {}).get("format"),
        )
        self.logger = get_logger("main")

        # Build typed configs.
        self.cam_cfg = CameraConfig.from_dict(self.cfg.get("camera", {}))
        self.tracker_cfg = TrackerConfig.from_dict(self.cfg.get("tracker", {}))
        self.transform_cfg = TransformConfig.from_dict(self.cfg.get("transform", {}))
        self.smoothing_cfg = SmoothingConfig.from_dict(self.cfg.get("smoothing", {}))
        self.safety_cfg = SafetyConfig.from_dict(self.cfg.get("safety", {}))
        self.hand_cfg = HandConfig.from_dict(self.cfg.get("hand", {}))
        self.debug_cfg = self.cfg.get("debug", {}) or {}

        # Pipeline components (created in setup()).
        self.camera = None
        self.tracker = None
        self.mapper = GestureMapper()
        self.calibration = load_calibration(self.cfg, args.config)
        self.transform = Transform(self.transform_cfg)
        self.smoother = Smoother(self.smoothing_cfg)
        self.safety = SafetyManager(self.safety_cfg, self.hand_cfg.rest_pose)
        self.controller = None

        self.diag = Diagnostics()
        self.diag.camera_mirror = self.transform_cfg.camera_mirror
        self.diag.output_mirror = self.transform_cfg.output_mirror

        self._running = False
        self._last_frame_id = -1

        # Window policy: debug shows it; headless/no-window never does.
        self.show_window = (
            (args.debug or self.debug_cfg.get("show_window", False))
            and not args.headless
            and not args.no_window
        )
        self.headless_reporter = HeadlessReporter(
            float(self.debug_cfg.get("headless_status_seconds", 3.0))
        )

    # -- setup / teardown ---------------------------------------------------
    def setup(self) -> None:
        self.logger.info("starting rb3_hand_mimic (dry_run=%s, headless=%s)",
                         self.args.dry_run, self.args.headless)

        # Controller first: if hardware missing and not dry-run, it degrades to
        # reconnect attempts rather than crashing.
        self.controller = create_controller(
            self.hand_cfg,
            finger_order=self.transform_cfg.finger_order,
            force_mock=self.args.dry_run,
        )
        self.controller.connect()
        self.controller.send_rest()

        # Tracker (may raise a clear SystemExit if MediaPipe is missing).
        self.tracker = create_tracker(self.tracker_cfg)

        # Camera.
        self.camera = open_camera_from_config(self.cam_cfg, self.args.camera_index)
        self.camera.start()
        self.diag.camera_index = self.camera.index

        if self.calibration.source == "defaults":
            self.logger.warning("using DEFAULT calibration ranges; run "
                                "tools/record_calibration.py for best accuracy")

    def teardown(self) -> None:
        self.logger.info("shutting down; sending rest pose")
        try:
            if self.controller is not None:
                self.controller.send_rest()
                time.sleep(0.05)
                self.controller.close()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("controller shutdown error: %s", exc)
        try:
            if self.camera is not None:
                self.camera.release()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("camera shutdown error: %s", exc)
        try:
            if self.tracker is not None:
                self.tracker.close()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("tracker shutdown error: %s", exc)
        if self.show_window:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        self._running = True
        self.setup()
        self.logger.info("entering main loop")
        try:
            while self._running:
                self._iterate()
        finally:
            self.teardown()

    def _iterate(self) -> None:
        frame, frame_id = self.camera.read()
        if frame is None:
            time.sleep(0.002)
            return

        # Only process genuinely new frames -- don't reprocess duplicates.
        is_new = frame_id != self._last_frame_id
        self._last_frame_id = frame_id

        t0 = time.perf_counter()
        pose: Optional[HandPose] = None
        selected_hand = None

        if is_new:
            self.diag.tick()
            hands = self.tracker.process(frame)
            selected_hand = self.tracker.select_best(hands, self.tracker_cfg)
            if selected_hand is not None:
                raw_pose = self.mapper.map(selected_hand)
                norm_pose = self.calibration.normalize(raw_pose)
                tf_pose = self.transform.apply(norm_pose)
                pose = self.smoother.apply(tf_pose)
                self.safety.update_pose(pose.as_dict())
                self.diag.handedness = pose.handedness
                self.diag.selected = True
            else:
                self.diag.selected = False

        # Safety state machine + controller send run every iteration so the
        # watchdog / return-to-rest behaves even when no new frame arrived.
        command = self.safety.compute_output()
        self.controller.send(command)

        # Update diagnostics.
        self.diag.command = command
        self.diag.track_state = self.safety.current_state()
        self.diag.serial_connected = self.controller.is_connected()
        self.diag.latency_ms = (time.perf_counter() - t0) * 1000.0

        if self.show_window and is_new:
            self._render(frame, selected_hand, pose)
        elif self.args.headless:
            self.headless_reporter.maybe_log(self.diag)

    def _render(self, frame, selected_hand, pose) -> None:
        import cv2

        # Draw landmarks on the ORIGINAL (unmirrored) frame; a later horizontal
        # flip then keeps the skeleton aligned with the mirrored preview image.
        if (
            selected_hand is not None
            and self.debug_cfg.get("draw_landmarks", True)
            and hasattr(self.tracker, "draw")
        ):
            try:
                self.tracker.draw(frame, selected_hand)
            except Exception as exc:  # noqa: BLE001 - drawing must never crash demo
                self.logger.debug("landmark draw failed: %s", exc)

        # Flip for selfie preview, then draw text overlay (so text isn't mirrored).
        display = cv2.flip(frame, 1) if self.cam_cfg.mirror_preview else frame

        draw_overlay(
            display,
            self.diag,
            pose,
            draw_curl_bars=self.debug_cfg.get("draw_curl_bars", True),
            draw_fps=self.debug_cfg.get("draw_fps", True),
        )

        window = self.debug_cfg.get("window_name", "rb3_hand_mimic")
        cv2.imshow(window, display)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):  # ESC or q
            self.logger.info("quit requested from window")
            self._running = False

    def stop(self) -> None:
        self._running = False


def _handle_list_modes(args: argparse.Namespace, cfg: dict) -> bool:
    """Handle --list-cameras / --list-serial; return True if we should exit."""
    if args.list_cameras:
        cam_cfg = CameraConfig.from_dict(cfg.get("camera", {}))
        print("Discovering cameras (indexes 0-8)...\n")
        infos = discover_cameras(cam_cfg, save_debug_dir="debug_frames")
        print(f"\n{'idx':>3}  {'reads':>5}  {'res':>9}  {'fps':>5}  name")
        for i in infos:
            res = f"{i.width}x{i.height}" if i.reads else "-"
            print(f"{i.index:>3}  {('yes' if i.reads else 'no'):>5}  "
                  f"{res:>9}  {i.fps:>5}  {i.name or i.note}")
        print("\nSet camera.index in config.yaml or pass --camera-index.")
        return True
    if args.list_serial:
        ports = list_serial_ports()
        print("Serial port candidates:")
        if not ports:
            print("  (none found under /dev/serial/by-id, /dev/ttyACM*, /dev/ttyUSB*)")
        for p in ports:
            print(f"  {p}")
        return True
    return False


def main() -> int:
    args = parse_args()

    # Bootstrap logging early using config if available.
    try:
        cfg_preview = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        # list modes can still partially work, but config is needed; report.
        setup_logging("INFO")
        get_logger("main").error("%s", exc)
        return 2

    setup_logging(
        level=args.log_level or cfg_preview.get("logging", {}).get("level", "INFO"),
        log_file=cfg_preview.get("logging", {}).get("file"),
        fmt=cfg_preview.get("logging", {}).get("format"),
    )

    if _handle_list_modes(args, cfg_preview):
        return 0

    app = HandMimicApp(args)

    # Clean shutdown on SIGINT/SIGTERM -> rest pose + close ports.
    def _signal(signum, _frame):
        get_logger("main").info("signal %s received; stopping", signum)
        app.stop()

    signal.signal(signal.SIGINT, _signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal)

    try:
        app.run()
    except KeyboardInterrupt:
        app.stop()
    except Exception as exc:  # noqa: BLE001 - top-level guard, demo must not crash silently
        get_logger("main").exception("fatal error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
