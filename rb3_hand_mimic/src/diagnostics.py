"""Diagnostics: debug overlay rendering + headless status logging.

Kept separate from the pipeline so the real-time loop stays lean and so the
overlay can be disabled with zero cost in headless mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .gesture_mapper import HandPose
from .utils import FINGERS, FpsMeter, get_logger

log = get_logger("diag")


@dataclass
class Diagnostics:
    """Mutable snapshot of pipeline state for display/logging."""

    camera_index: int = -1
    fps: float = 0.0
    latency_ms: float = 0.0
    handedness: str = "-"
    selected: bool = False
    serial_connected: bool = False
    track_state: str = "rest"
    command: Dict[str, float] = field(default_factory=lambda: {f: 0.0 for f in FINGERS})
    camera_mirror: bool = False
    output_mirror: bool = False
    fps_meter: FpsMeter = field(default_factory=lambda: FpsMeter(30))

    # Async-pipeline responsiveness metrics (populated from metrics.snapshot()).
    detect_fps: float = 0.0
    detect_p50_ms: float = 0.0
    detect_p90_ms: float = 0.0
    e2e_p50_ms: float = 0.0
    control_hz: float = 0.0
    detection_rate: float = 0.0

    def tick(self) -> None:
        self.fps_meter.tick()
        self.fps = self.fps_meter.fps()

    def update_from_metrics(self, snap: Dict) -> None:
        """Copy the headline numbers out of a metrics.Metrics snapshot dict."""
        self.detect_fps = float(snap.get("detect_fps", 0.0))
        self.control_hz = float(snap.get("control_hz", 0.0))
        self.detection_rate = float(snap.get("detection_rate", 0.0))
        det = snap.get("detect_ms")
        e2e = snap.get("e2e_ms")
        if det is not None:
            self.detect_p50_ms = det.p50
            self.detect_p90_ms = det.p90
        if e2e is not None:
            self.e2e_p50_ms = e2e.p50

    # -- headless logging ---------------------------------------------------
    def status_line(self) -> str:
        cmd = " ".join(f"{f[0]}:{self.command.get(f, 0.0):.2f}" for f in FINGERS)
        return (
            f"cam={self.camera_index} det={self.detect_fps:4.1f}fps "
            f"lat p50/p90={self.detect_p50_ms:4.1f}/{self.detect_p90_ms:4.1f}ms "
            f"e2e={self.e2e_p50_ms:5.1f}ms ctrl={self.control_hz:4.0f}hz "
            f"hand={self.handedness} sel={'Y' if self.selected else 'N'} "
            f"hw={'up' if self.serial_connected else 'DOWN'} "
            f"track={self.track_state} | {cmd}"
        )


def draw_overlay(
    frame,
    diag: Diagnostics,
    pose: Optional[HandPose],
    draw_curl_bars: bool = True,
    draw_fps: bool = True,
):
    """Render the debug overlay onto a BGR frame (returns the same frame).

    Imports OpenCV locally so headless mode never needs the GUI symbols.
    """
    import cv2

    h, w = frame.shape[:2]
    green = (0, 220, 0)
    red = (0, 0, 220)
    white = (255, 255, 255)
    yellow = (0, 220, 220)

    def put(text: str, y: int, color=white) -> None:
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if draw_fps:
        put(f"cam {diag.camera_index}   detect {diag.detect_fps:4.1f} fps", 20, yellow)
        put(f"det lat p50/p90 {diag.detect_p50_ms:4.1f}/{diag.detect_p90_ms:4.1f} ms   "
            f"e2e {diag.e2e_p50_ms:5.1f} ms   ctrl {diag.control_hz:4.0f} hz", 40, yellow)

    hw_color = green if diag.serial_connected else red
    put(f"hand: {'connected' if diag.serial_connected else 'DISCONNECTED'}", 60, hw_color)

    track_color = green if diag.track_state == "active" else yellow
    put(f"track: {diag.track_state}   hand: {diag.handedness}   sel: {'Y' if diag.selected else 'N'}", 80, track_color)

    put(f"cam_mirror: {diag.camera_mirror}   out_mirror: {diag.output_mirror}", 100)

    if draw_curl_bars:
        _draw_curl_bars(frame, diag.command, origin_y=h - 110)

    return frame


def _draw_curl_bars(frame, command: Dict[str, float], origin_y: int) -> None:
    import cv2

    bar_w = 26
    gap = 10
    max_h = 80
    x = 12
    for finger in FINGERS:
        val = max(0.0, min(1.0, command.get(finger, 0.0)))
        bar_h = int(val * max_h)
        # background track
        cv2.rectangle(frame, (x, origin_y), (x + bar_w, origin_y + max_h), (60, 60, 60), 1)
        # filled portion (green->red as it closes)
        color = (0, int(220 * (1 - val)), int(220 * val))
        cv2.rectangle(
            frame,
            (x, origin_y + (max_h - bar_h)),
            (x + bar_w, origin_y + max_h),
            color,
            -1,
        )
        cv2.putText(frame, finger[0].upper(), (x + 6, origin_y + max_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{val:.2f}", (x - 2, origin_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        x += bar_w + gap


class HeadlessReporter:
    """Logs a status line every N seconds in headless mode."""

    def __init__(self, interval_seconds: float = 3.0) -> None:
        self.interval = interval_seconds
        self._last = 0.0

    def maybe_log(self, diag: Diagnostics) -> None:
        now = time.perf_counter()
        if now - self._last >= self.interval:
            self._last = now
            log.info("%s", diag.status_line())
