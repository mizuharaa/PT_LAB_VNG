#!/usr/bin/env python3
"""rb3_hand_mimic -- real-time robotic hand mimic for the Qualcomm RB3 Gen 2.

Async pipeline (detection is decoupled from hand control):

  camera thread  -> latest-frame capture
  detection thread (HOT PATH): MediaPipe landmarks -> select hand ->
      handedness/mirror transform -> finger curls -> calibration -> smoothing
      -> publishes the latest PoseSample
  control thread: safety clamp/watchdog -> hand controller (SDK / serial / mock)

Detection never waits on hand control, so a slow or not-yet-ported hand SDK
cannot reduce tracking responsiveness. See src/pipeline.py for the rationale.

Run examples:
  python main.py --config config.yaml --debug
  python main.py --config config.yaml --headless
  python main.py --dry-run --debug
  python main.py --benchmark 15            # 15s detection-responsiveness report
  python main.py --benchmark --debug       # benchmark with the live window
  python main.py --camera-index 2 --dry-run --debug
  python main.py --list-cameras
  python main.py --list-serial
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from typing import List, Optional

# Ensure local package import works regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.calibration import load_calibration  # noqa: E402
from src.camera import (  # noqa: E402
    Camera,
    CameraConfig,
    discover_cameras,
    open_camera_from_config,
)
from src.debug_menu import DebugMenu, MenuItem  # noqa: E402
from src.diagnostics import Diagnostics, HeadlessReporter, draw_overlay  # noqa: E402
from src.gesture_mapper import GestureMapper, compute_finger_confidence  # noqa: E402
from src.gesture_recognizer import classify  # noqa: E402
from src.hand_controller import (  # noqa: E402
    BaseHandController,
    HandConfig,
    create_controller,
    list_serial_ports,
)
from src.fusion import FusionConfig  # noqa: E402
from src.hand_tracker import (  # noqa: E402
    BaseHandTracker,
    TrackerConfig,
    create_tracker,
)
from src.metrics import Metrics  # noqa: E402
from src.pipeline import (  # noqa: E402
    ControlWorker,
    DetectionWorker,
    FusionWorker,
    LatestSlot,
)
from src.safety import SafetyConfig, SafetyManager  # noqa: E402
from src.smoothing import Smoother, SmoothingConfig  # noqa: E402
from src.transform import Transform, TransformConfig  # noqa: E402
from src.utils import FINGERS, clamp, get_logger, load_config, setup_logging  # noqa: E402


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
    p.add_argument(
        "--benchmark", nargs="?", type=float, const=0.0, default=None,
        metavar="SECONDS",
        help="Measure detection responsiveness. Optional duration in seconds "
             "(omit value to run until Ctrl+C). No window unless --debug is also set.",
    )
    p.add_argument("--camera-index", type=int, default=None, help="Force the primary camera index")
    p.add_argument("--camera2-index", type=int, default=None,
                   help="Enable the second (fusion) camera at this index")
    p.add_argument("--no-dual", action="store_true",
                   help="Force single-camera mode (ignore camera2/fusion)")
    p.add_argument("--log-level", default=None, help="DEBUG|INFO|WARNING|ERROR")
    p.add_argument("--list-cameras", action="store_true", help="Discover cameras and exit")
    p.add_argument("--list-serial", action="store_true", help="List serial ports and exit")
    return p.parse_args()


class HandMimicApp:
    """Owns the pipeline objects and the two async workers, with clean shutdown."""

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
        self.pipeline_cfg = self.cfg.get("pipeline", {}) or {}

        # Run modes.
        self.benchmark = args.benchmark is not None
        self.benchmark_seconds = float(args.benchmark or 0.0)

        # Shared, stateless pipeline objects (safe to use from one thread each).
        self.mapper = GestureMapper()
        self.calibration = load_calibration(self.cfg, args.config)
        self.safety = SafetyManager(self.safety_cfg, self.hand_cfg.rest_pose)

        # Async machinery.
        self.metrics = Metrics(window=int(self.pipeline_cfg.get("metrics_window", 150)))
        self.control_tick_hz = float(self.pipeline_cfg.get("control_tick_hz", 120.0))
        self.fusion_cfg = FusionConfig.from_dict(self.cfg.get("fusion", {}))

        # Dual-camera (opt-in): a second camera at a known offset sees fingers
        # the first view occludes; their per-finger curls are fused. Off by
        # default, so the standard path stays single-camera and lightweight.
        cam2 = self.cfg.get("camera2", {}) or {}
        self.dual = (
            (bool(cam2.get("enabled", False)) or args.camera2_index is not None)
            and not args.no_dual
        )
        # Secondary camera config = primary's settings overlaid with camera2 keys.
        self.cam2_cfg = CameraConfig.from_dict({**(self.cfg.get("camera", {}) or {}), **cam2})
        # Camera index selections (None = auto-discover). Editable via the debug
        # menu; camera/dual changes are applied by rebuilding the detection layer.
        self.cam0_index: Optional[int] = args.camera_index
        if args.camera2_index is not None:
            self.cam1_index: Optional[int] = args.camera2_index
        else:
            _c1 = cam2.get("index", 1)
            self.cam1_index = _c1 if isinstance(_c1, int) else None

        # Pipeline objects created in setup() (lists support 1 or 2 cameras).
        self.controller: Optional[BaseHandController] = None
        self.cameras: List[Camera] = []
        self.detections: List[DetectionWorker] = []
        self.det_slots: List[LatestSlot] = []
        self.control_slot = LatestSlot()
        self.fusion: Optional[FusionWorker] = None
        self.control: Optional[ControlWorker] = None

        self.diag = Diagnostics()
        self.diag.camera_mirror = self.transform_cfg.camera_mirror
        self.diag.output_mirror = self.transform_cfg.output_mirror

        self._running = False

        # Window policy: debug shows it; headless/no-window/benchmark never do
        # (benchmark wants raw detection speed unless you explicitly add --debug).
        want_window = args.debug or self.debug_cfg.get("show_window", False)
        self.show_window = (
            want_window
            and not args.headless
            and not args.no_window
            and not (self.benchmark and not args.debug)
        )
        self.headless_reporter = HeadlessReporter(
            float(self.debug_cfg.get("headless_status_seconds", 3.0))
        )
        self.benchmark_report_seconds = float(
            self.pipeline_cfg.get("benchmark_report_seconds", 1.0)
        )

        # In-window debug menu (press 'm' in the debug window).
        self.menu = self._build_menu()

    # -- debug menu ---------------------------------------------------------
    def _build_menu(self) -> DebugMenu:
        """Construct the debug-menu items bound to this app's live state."""
        index_cycle: List[Optional[int]] = [None, 0, 1, 2, 3, 4, 5, 6, 7]

        def fmt_idx(v: Optional[int]) -> str:
            return "auto" if v is None else str(v)

        def cycle_idx(cur: Optional[int], delta: int) -> Optional[int]:
            pos = index_cycle.index(cur) if cur in index_cycle else 0
            return index_cycle[(pos + delta) % len(index_cycle)]

        def set_cam0(d: int) -> None:
            self.cam0_index = cycle_idx(self.cam0_index, d)

        def set_cam1(d: int) -> None:
            self.cam1_index = cycle_idx(self.cam1_index, d)

        def toggle_dual(_d: int) -> None:
            self.dual = not self.dual

        def toggle_mirror(_d: int) -> None:
            self.cam_cfg.mirror_preview = not self.cam_cfg.mirror_preview

        def toggle_landmarks(_d: int) -> None:
            self.debug_cfg["draw_landmarks"] = not self.debug_cfg.get("draw_landmarks", True)

        def toggle_out_mirror(_d: int) -> None:
            # Transform objects hold this cfg by reference, so flipping it is live.
            self.transform_cfg.output_mirror = not self.transform_cfg.output_mirror
            self.diag.output_mirror = self.transform_cfg.output_mirror

        def adj_fusion(d: int) -> None:
            self.fusion_cfg.min_confidence = clamp(
                round(self.fusion_cfg.min_confidence + d * 0.05, 2), 0.0, 1.0)

        items = [
            MenuItem("Dual camera", lambda: "ON" if self.dual else "OFF", toggle_dual, pending=True),
            MenuItem("Camera 0 index", lambda: fmt_idx(self.cam0_index), set_cam0, pending=True),
            MenuItem("Camera 1 index", lambda: fmt_idx(self.cam1_index), set_cam1, pending=True),
            MenuItem("APPLY camera changes", lambda: "[Enter]", lambda _d: None,
                     on_select=lambda: "apply_cameras"),
            MenuItem("Mirror preview",
                     lambda: "ON" if self.cam_cfg.mirror_preview else "OFF", toggle_mirror),
            MenuItem("Draw landmarks",
                     lambda: "ON" if self.debug_cfg.get("draw_landmarks", True) else "OFF",
                     toggle_landmarks),
            MenuItem("Output mirror (robot)",
                     lambda: "ON" if self.transform_cfg.output_mirror else "OFF", toggle_out_mirror),
            MenuItem("Fusion min-confidence",
                     lambda: f"{self.fusion_cfg.min_confidence:.2f}", adj_fusion),
        ]
        return DebugMenu(items)

    # -- setup / teardown ---------------------------------------------------
    def setup(self) -> None:
        self.logger.info(
            "starting rb3_hand_mimic (dry_run=%s, headless=%s, benchmark=%s, async)",
            self.args.dry_run, self.args.headless, self.benchmark,
        )

        # Controller first: if hardware/SDK is missing it degrades gracefully
        # (placeholder SDK / reconnect attempts) rather than crashing.
        self.controller = create_controller(
            self.hand_cfg,
            finger_order=self.transform_cfg.finger_order,
            force_mock=self.args.dry_run,
        )
        self.controller.connect()
        self.controller.send_rest()

        if self.calibration.source == "defaults":
            self.logger.warning("using DEFAULT calibration ranges; run "
                                "tools/record_calibration.py for best accuracy")

        self._build_detection_layer()
        self.control = ControlWorker(
            controller=self.controller,
            safety=self.safety,
            slot=self.control_slot,
            metrics=self.metrics,
            tick_hz=self.control_tick_hz,
        )

    def _build_detection_layer(self) -> None:
        """(Re)build cameras + per-camera detection workers + optional fusion.

        Sets self.cameras / self.detections / self.det_slots / self.fusion and
        points self.control_slot at whatever the control thread should consume
        (the lone detection slot, or the fusion output). Safe to call again at
        runtime after tearing the previous layer down (see _reconfigure_cameras).
        """
        self.cameras = []
        self.detections = []
        self.det_slots = []
        self.fusion = None

        # Each camera gets its OWN tracker + transform + smoother (all stateful)
        # and its own slot; mapper and calibration are stateless and shared.
        specs = [(self.cam0_index, self.cam_cfg)]
        if self.dual:
            specs.append((self.cam1_index, self.cam2_cfg))

        # Open the cameras in PARALLEL -- the slow part on Windows is the device
        # open (a non-DirectShow webcam falls back to MSMF at ~10s), so opening
        # both at once makes boot ~= the slowest single camera, not the sum.
        # Trackers are created serially afterwards (MediaPipe's first-run TFLite
        # init isn't safe to trigger from two threads at once).
        cams: List = [None] * len(specs)

        def _open(slot_i: int, forced_index, cam_cfg) -> None:
            try:
                camera = open_camera_from_config(cam_cfg, forced_index)
                camera.start()
                cams[slot_i] = camera
            except Exception as exc:  # noqa: BLE001
                cams[slot_i] = exc

        threads = [
            threading.Thread(target=_open, args=(i, fi, cfg), name=f"cam-open-{i}")
            for i, (fi, cfg) in enumerate(specs)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, camera in enumerate(cams):
            if not isinstance(camera, Camera):
                if i == 0:  # primary camera is required
                    raise camera if isinstance(camera, BaseException) else RuntimeError(
                        "primary camera failed to open")
                self.logger.warning("secondary camera unavailable (%s); "
                                    "continuing single-camera", camera)
                continue
            worker = DetectionWorker(
                camera=camera,
                tracker=create_tracker(self.tracker_cfg),
                mapper=self.mapper,
                calibration=self.calibration,
                transform=Transform(self.transform_cfg),
                smoother=Smoother(self.smoothing_cfg),
                tracker_cfg=self.tracker_cfg,
                slot=LatestSlot(),
                metrics=self.metrics,
                keep_debug_frames=self.show_window,
            )
            self.cameras.append(camera)
            self.detections.append(worker)
            self.det_slots.append(worker.slot)

        self.diag.camera_index = self.cameras[0].index

        # With one camera the detection worker feeds the control slot directly;
        # with two, a fusion worker combines them into a fresh output slot.
        if len(self.detections) > 1:
            self.fusion = FusionWorker(
                in_slots=self.det_slots,
                out_slot=LatestSlot(),
                fusion_cfg=self.fusion_cfg,
                tick_hz=self.control_tick_hz,
            )
            self.control_slot = self.fusion.out_slot
            self.logger.info("dual-camera fusion enabled (%d cameras, strategy=%s)",
                             len(self.detections), self.fusion_cfg.strategy)
        else:
            self.control_slot = self.det_slots[0]

    def _reconfigure_cameras(self) -> None:
        """Apply pending camera/dual changes by rebuilding the detection layer
        while the controller + control thread keep running (safety eases the
        hand to rest during the brief camera reopen)."""
        self.logger.info("reconfiguring cameras (dual=%s cam0=%s cam1=%s) ...",
                         self.dual, self.cam0_index, self.cam1_index)
        layer = list(self.detections) + ([self.fusion] if self.fusion else [])
        for w in layer:
            w.stop()
        for w in layer:
            if w.is_alive():
                w.join(timeout=1.0)
        for camera in self.cameras:
            try:
                camera.release()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("camera release error: %s", exc)
        for w in self.detections:
            try:
                w.tracker.close()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("tracker close error: %s", exc)

        try:
            self._build_detection_layer()
        except Exception as exc:  # noqa: BLE001
            self.logger.error("camera rebuild failed: %s; stopping", exc)
            self._running = False
            return

        for w in self.detections:
            w.start()
        if self.fusion is not None:
            self.fusion.start()
        if self.control is not None:
            self.control.set_slot(self.control_slot)
        self.logger.info("reconfigure done (%d camera(s) active)", len(self.detections))

    def teardown(self) -> None:
        self.logger.info("shutting down; stopping workers and sending rest pose")
        # Stop detection first so no new poses arrive, then fusion, then control.
        workers: List = list(self.detections)
        if self.fusion is not None:
            workers.append(self.fusion)
        if self.control is not None:
            workers.append(self.control)
        for worker in workers:
            worker.stop()
        for worker in workers:
            if worker.is_alive():
                worker.join(timeout=1.0)

        try:
            if self.controller is not None:
                self.controller.send_rest()
                time.sleep(0.05)
                self.controller.close()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("controller shutdown error: %s", exc)
        for camera in self.cameras:
            try:
                camera.release()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("camera shutdown error: %s", exc)
        for worker in self.detections:
            try:
                worker.tracker.close()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("tracker shutdown error: %s", exc)
        if self.show_window:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass

        if self.benchmark:
            print("\n" + self.metrics.summary())

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        self._running = True
        self.setup()
        assert self.control is not None
        for worker in self.detections:
            worker.start()
        if self.fusion is not None:
            self.fusion.start()
        self.control.start()
        self.logger.info("entering main loop (%d camera(s), workers running)",
                         len(self.detections))

        deadline = (
            time.perf_counter() + self.benchmark_seconds
            if self.benchmark and self.benchmark_seconds > 0
            else None
        )
        last_report = 0.0
        try:
            while self._running:
                alive = (all(w.is_alive() for w in self.detections)
                         and self.control.is_alive()
                         and (self.fusion is None or self.fusion.is_alive()))
                if not alive:
                    self.logger.error("a worker thread died; exiting")
                    break

                if self.show_window:
                    self._render()
                else:
                    now = time.perf_counter()
                    interval = (
                        self.benchmark_report_seconds if self.benchmark
                        else self.headless_reporter.interval
                    )
                    if now - last_report >= interval:
                        last_report = now
                        self._refresh_diag()
                        if self.benchmark:
                            self.logger.info("%s", self.metrics.report_line())
                        else:
                            self.headless_reporter.maybe_log(self.diag)
                    time.sleep(0.01)

                if deadline is not None and time.perf_counter() >= deadline:
                    self.logger.info("benchmark duration reached; stopping")
                    break
        finally:
            self.teardown()

    def _refresh_diag(self) -> None:
        """Pull the latest control + metrics state into the diagnostics snapshot."""
        self.diag.update_from_metrics(self.metrics.snapshot())
        if self.control is not None:
            command, track_state, connected, handedness = self.control.status()
            self.diag.command = command
            self.diag.track_state = track_state
            self.diag.serial_connected = connected
            self.diag.handedness = handedness
            # Classify the commanded pose for the debug/headless hand-state view.
            self.diag.hand_state = classify(command, handedness)

    def _render(self) -> None:
        import cv2

        if not self.detections:
            time.sleep(0.005)
            return

        self._refresh_diag()

        # One panel per camera: landmarks + that camera's own raw weights.
        panels: List = []
        primary_pose = None
        for i, worker in enumerate(self.detections):
            frame, selected_hand, pose = worker.debug_snapshot()
            if frame is None:
                continue
            if i == 0:
                primary_pose = pose
            # Draw landmarks on the ORIGINAL frame; the later flip keeps the
            # skeleton aligned with the mirrored selfie preview.
            if (selected_hand is not None
                    and self.debug_cfg.get("draw_landmarks", True)
                    and hasattr(worker.tracker, "draw")):
                try:
                    worker.tracker.draw(frame, selected_hand)
                except Exception as exc:  # noqa: BLE001 - drawing must never crash the demo
                    self.logger.debug("landmark draw failed: %s", exc)
            disp = cv2.flip(frame, 1) if self.cam_cfg.mirror_preview else frame
            conf = compute_finger_confidence(selected_hand) if selected_hand is not None else None
            self._annotate_camera_panel(disp, i, pose, conf)
            panels.append(disp)

        if not panels:
            time.sleep(0.005)
            return

        # Dual requested but the 2nd camera isn't up: show a clear placeholder
        # panel (instead of silently staying single) so it's visible on screen
        # and you can pick a working index in the menu.
        if self.dual and len(panels) < 2:
            panels.append(self._placeholder_panel(panels[0]))

        self.diag.selected = primary_pose is not None
        if primary_pose is not None:
            self.diag.handedness = primary_pose.handedness
            self.diag.hand_state = classify(primary_pose.as_dict(), primary_pose.handedness)

        # Side-by-side composite when more than one camera is active. The main
        # overlay (fused command bars + gesture + metrics) is drawn over it.
        display = panels[0] if len(panels) == 1 else self._hconcat(panels)
        draw_overlay(
            display,
            self.diag,
            primary_pose,
            draw_curl_bars=self.debug_cfg.get("draw_curl_bars", True),
            draw_fps=self.debug_cfg.get("draw_fps", True),
        )

        if not self.menu.visible:
            hint_x = display.shape[1] - 96
            cv2.putText(display, "[m] menu", (hint_x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, "[m] menu", (hint_x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1, cv2.LINE_AA)
        self.menu.render(display)
        window = self.debug_cfg.get("window_name", "rb3_hand_mimic")
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        action = self.menu.handle_key(key)
        if action == "apply_cameras":
            self._reconfigure_cameras()
        elif action is None and key in (27, ord("q")):  # ESC/q (menu didn't consume)
            self.logger.info("quit requested from window")
            self._running = False

    def _annotate_camera_panel(self, panel, i: int, pose, conf=None) -> None:
        """Label a camera panel and (in dual mode) print that camera's raw weights
        and per-finger view confidence (higher = this camera sees that finger
        better, so fusion trusts it more)."""
        import cv2

        idx = self.cameras[i].index if i < len(self.cameras) else i
        label = f"cam{i} (idx {idx})"
        y0 = panel.shape[0] - 12
        cv2.putText(panel, label, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, label, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
        if len(self.detections) > 1:
            curls = pose.as_dict() if pose is not None else {f: 0.0 for f in FINGERS}
            wtxt = "raw  " + " ".join(f"{f[0].upper()}{curls.get(f, 0.0):.2f}" for f in FINGERS)
            ctxt = "conf " + " ".join(f"{f[0].upper()}{(conf or {}).get(f, 0.0):.2f}" for f in FINGERS)
            cv2.putText(panel, wtxt, (8, panel.shape[0] - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, wtxt, (8, panel.shape[0] - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, ctxt, (8, panel.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, ctxt, (8, panel.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 220, 120), 1, cv2.LINE_AA)

    def _placeholder_panel(self, ref):
        """A 'camera unavailable' panel, shown when dual is on but cam1 isn't up."""
        import cv2
        import numpy as np

        h, w = ref.shape[0], ref.shape[1]
        panel = np.full((h, w, 3), 35, dtype=np.uint8)
        idx = "auto" if self.cam1_index is None else str(self.cam1_index)
        lines = [
            f"cam1 (idx {idx}): UNAVAILABLE",
            "open [m] menu -> set 'Camera 1 index'",
            "-> 'APPLY camera changes' (Enter)",
        ]
        y = h // 2 - 24
        for s in lines:
            cv2.putText(panel, s, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 210), 2, cv2.LINE_AA)
            y += 34
        return panel

    @staticmethod
    def _hconcat(panels: List):
        """Horizontally stack panels, normalizing to a common height."""
        import cv2

        h = min(p.shape[0] for p in panels)
        resized = [
            p if p.shape[0] == h
            else cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
            for p in panels
        ]
        return cv2.hconcat(resized)

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
