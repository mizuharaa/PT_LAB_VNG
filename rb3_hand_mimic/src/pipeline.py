"""Async detection/control runtime.

The pipeline is split into two threads connected by a single latest-value slot:

    [camera thread]            already in camera.py (latest-frame-only capture)
         |
         v   Camera.read()
    +-------------------+      DetectionWorker  (the HOT PATH -- keep it fast)
    | tracker -> mapper |      runs as fast as the camera + model allow:
    | -> calibration    |      landmarks, finger curls, calibration, transform,
    | -> transform      |      smoothing. Produces a PoseSample.
    | -> smoothing      |
    +-------------------+
         |
         v   LatestSlot.put()  (lock-free-ish single-slot mailbox; newest wins)
    +-------------------+      ControlWorker
    | safety state mc   |      consumes the freshest PoseSample, runs the safety
    | -> hand SDK/serial|      state machine + watchdog, and drives the hand
    +-------------------+      controller (serial / SDK placeholder / mock).

Why two threads instead of one loop:
  * Detection responsiveness must not depend on hand-control speed. The control
    side talks to the robotic-hand SDK -- which today only ships for x86 and is
    not yet ported to the RB3's aarch64. A slow, blocking, or stubbed SDK call
    must never stall landmark detection. Decoupling them via LatestSlot means
    detection always runs at full speed and control always acts on the *latest*
    pose, dropping any it could not keep up with.
  * MediaPipe inference and serial/SDK I/O both release the GIL while doing
    native work, so Python threads give real overlap here without the IPC cost
    of moving video frames between processes. The LatestSlot boundary is the
    seam where this could later become a process/queue if a backend needs it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .fusion import FusionConfig, fuse_curls
from .gesture_mapper import compute_finger_confidence
from .metrics import Metrics
from .utils import get_logger

log = get_logger("pipeline")


@dataclass
class PoseSample:
    """One detection result handed from the detection thread to control.

    `values` is the final normalized finger pose (post calibration/transform/
    smoothing), keyed by finger name, or None when no hand was detected on this
    frame. The timestamps let the control side compute end-to-end latency and
    decide whether a pose is fresh enough to act on.
    """

    frame_id: int
    capture_ts: float          # perf_counter when the frame was captured
    detect_ts: float           # perf_counter when detection finished
    detected: bool
    handedness: str = ""
    confidence: float = 0.0
    values: Optional[Dict[str, float]] = None
    finger_conf: Optional[Dict[str, float]] = None  # per-finger view reliability


class LatestSlot:
    """Single-slot mailbox: producer overwrites, consumer reads the newest value.

    Unlike a queue, old values are never buffered -- the consumer always sees the
    freshest pose and stale ones are simply dropped. A monotonically increasing
    sequence number lets the consumer tell a brand-new value from a re-read.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[PoseSample] = None
        self._seq = 0
        self._last_read_seq = 0

    def put(self, value: PoseSample) -> None:
        with self._lock:
            self._value = value
            self._seq += 1

    def get_fresh(self) -> Tuple[Optional[PoseSample], bool]:
        """Return (latest_value, is_new_since_last_read)."""
        with self._lock:
            is_new = self._seq != self._last_read_seq
            self._last_read_seq = self._seq
            return self._value, is_new


class DetectionWorker(threading.Thread):
    """Runs the detection hot path and publishes PoseSamples to a LatestSlot."""

    def __init__(
        self,
        camera,
        tracker,
        mapper,
        calibration,
        transform,
        smoother,
        tracker_cfg,
        slot: LatestSlot,
        metrics: Metrics,
        keep_debug_frames: bool = False,
    ) -> None:
        super().__init__(name="detection", daemon=True)
        self.camera = camera
        self.tracker = tracker
        self.mapper = mapper
        self.calibration = calibration
        self.transform = transform
        self.smoother = smoother
        self.tracker_cfg = tracker_cfg
        self.slot = slot
        self.metrics = metrics
        self.keep_debug_frames = keep_debug_frames

        self._running = threading.Event()
        self._last_frame_id = -1

        # Latest frame + selected hand + pose, for the main thread's overlay.
        self._dbg_lock = threading.Lock()
        self._dbg_frame = None
        self._dbg_hand = None
        self._dbg_pose = None

    def run(self) -> None:
        self._running.set()
        log.info("detection worker started")
        while self._running.is_set():
            frame, frame_id = self.camera.read()
            if frame is None or frame_id == self._last_frame_id:
                # No frame yet, or no *new* frame -- never reprocess a duplicate.
                time.sleep(0.001)
                continue
            self._last_frame_id = frame_id
            capture_ts = self.camera.last_capture_ts()

            t0 = time.perf_counter()
            hands = self.tracker.process(frame)
            hand = self.tracker.select_best(hands, self.tracker_cfg)

            pose = None
            finger_conf = None
            if hand is not None:
                raw = self.mapper.map(hand)
                norm = self.calibration.normalize(raw)
                tf = self.transform.apply(norm)
                pose = self.smoother.apply(tf)
                finger_conf = compute_finger_confidence(hand)
            detect_ts = time.perf_counter()

            sample = PoseSample(
                frame_id=frame_id,
                capture_ts=capture_ts,
                detect_ts=detect_ts,
                detected=hand is not None,
                handedness=pose.handedness if pose is not None else "",
                confidence=pose.confidence if pose is not None else 0.0,
                values=pose.as_dict() if pose is not None else None,
                finger_conf=finger_conf,
            )
            self.slot.put(sample)
            self.metrics.record_detection((detect_ts - t0) * 1000.0, hand is not None)

            if self.keep_debug_frames:
                with self._dbg_lock:
                    self._dbg_frame = frame
                    self._dbg_hand = hand
                    self._dbg_pose = pose
        log.info("detection worker stopped")

    def debug_snapshot(self):
        """Return (frame, selected_hand, pose) for overlay rendering."""
        with self._dbg_lock:
            return self._dbg_frame, self._dbg_hand, self._dbg_pose

    def stop(self) -> None:
        self._running.clear()


class ControlWorker(threading.Thread):
    """Consumes the latest PoseSample, runs safety, and drives the controller.

    This thread owns the safety state machine (watchdog / hold / return-to-rest)
    and the hand controller. It re-evaluates safety every tick -- even when no
    new pose arrives -- so the watchdog still fires on tracking loss. The
    controller's own rate limiter caps how often commands actually go out.
    """

    def __init__(
        self,
        controller,
        safety,
        slot: LatestSlot,
        metrics: Metrics,
        tick_hz: float = 120.0,
    ) -> None:
        super().__init__(name="control", daemon=True)
        self.controller = controller
        self.safety = safety
        self.slot = slot
        self.metrics = metrics
        self._period = 1.0 / tick_hz if tick_hz > 0 else 0.0

        self._running = threading.Event()
        # Shared state for diagnostics (read by the main thread).
        self._state_lock = threading.Lock()
        self._command: Dict[str, float] = {}
        self._track_state = "rest"
        self._connected = False
        self._handedness = "Unknown"

    def run(self) -> None:
        self._running.set()
        log.info("control worker started")
        while self._running.is_set():
            loop_start = time.perf_counter()
            sample, is_new = self.slot.get_fresh()
            now = time.perf_counter()

            # Only feed a *new* pose where a hand was actually detected. New
            # "no hand" samples are intentionally ignored so the watchdog can
            # observe the absence and ease back to rest.
            if is_new and sample is not None and sample.detected and sample.values:
                self.safety.update_pose(sample.values, now=now)
                with self._state_lock:
                    self._handedness = sample.handedness or "Unknown"

            command = self.safety.compute_output(now=now)
            sent = self.controller.send(command)
            self.metrics.record_control(sent)

            # Only time real detections through to the hardware -- held/rest
            # commands (no fresh hand) would otherwise skew the latency stats.
            if sent and is_new and sample is not None and sample.detected:
                self.metrics.record_e2e(
                    (now - sample.capture_ts) * 1000.0,
                    (now - sample.detect_ts) * 1000.0,
                )

            with self._state_lock:
                self._command = command
                self._track_state = self.safety.current_state()
                self._connected = self.controller.is_connected()

            if self._period > 0:
                elapsed = time.perf_counter() - loop_start
                if elapsed < self._period:
                    time.sleep(self._period - elapsed)
        log.info("control worker stopped")

    def status(self) -> Tuple[Dict[str, float], str, bool, str]:
        """Return (last_command, track_state, controller_connected, handedness)."""
        with self._state_lock:
            return (dict(self._command), self._track_state,
                    self._connected, self._handedness)

    def set_slot(self, slot: LatestSlot) -> None:
        """Repoint the control input at a new slot (used after a live camera
        rebuild). A single reference swap is atomic in CPython; the new slot
        brings its own freshness tracking."""
        self.slot = slot

    def stop(self) -> None:
        self._running.clear()


class FusionWorker(threading.Thread):
    """Fuses the latest PoseSample from several detection slots into one and
    publishes it to the control slot. Only created when >1 camera is active;
    with a single camera the detection worker writes the control slot directly,
    so this stage costs nothing in the default configuration.
    """

    def __init__(
        self,
        in_slots: List[LatestSlot],
        out_slot: LatestSlot,
        fusion_cfg: FusionConfig,
        tick_hz: float = 120.0,
    ) -> None:
        super().__init__(name="fusion", daemon=True)
        self.in_slots = in_slots
        self.out_slot = out_slot
        self.fusion_cfg = fusion_cfg
        self._period = 1.0 / tick_hz if tick_hz > 0 else 0.0
        self._running = threading.Event()
        self._seq = 0

    def run(self) -> None:
        self._running.set()
        log.info("fusion worker started (%d cameras)", len(self.in_slots))
        while self._running.is_set():
            loop_start = time.perf_counter()
            fetched = [slot.get_fresh() for slot in self.in_slots]
            if any(is_new for _, is_new in fetched):
                sources = []
                capture_ts = detect_ts = 0.0
                handedness = "Unknown"
                confidence = 0.0
                for sample, _ in fetched:
                    if sample is not None and sample.detected and sample.values:
                        sources.append((sample.values, sample.confidence, sample.finger_conf))
                        # Carry timestamps/handedness from the freshest detection.
                        if sample.detect_ts > detect_ts:
                            detect_ts = sample.detect_ts
                            capture_ts = sample.capture_ts
                            handedness = sample.handedness
                            confidence = sample.confidence
                    else:
                        sources.append((None, 0.0, None))

                fused = fuse_curls(sources, self.fusion_cfg)
                now = time.perf_counter()
                self._seq += 1
                self.out_slot.put(PoseSample(
                    frame_id=self._seq,
                    capture_ts=capture_ts or now,
                    detect_ts=detect_ts or now,
                    detected=fused is not None,
                    handedness=handedness,
                    confidence=confidence,
                    values=fused,
                ))

            if self._period > 0:
                elapsed = time.perf_counter() - loop_start
                if elapsed < self._period:
                    time.sleep(self._period - elapsed)
        log.info("fusion worker stopped")

    def stop(self) -> None:
        self._running.clear()
