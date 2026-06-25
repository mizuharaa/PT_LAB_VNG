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

    def tick(self) -> None:
        self.fps_meter.tick()
        self.fps = self.fps_meter.fps()

    # -- headless logging ---------------------------------------------------
    def status_line(self) -> str:
        cmd = " ".join(f"{f[0]}:{self.command.get(f, 0.0):.2f}" for f in FINGERS)
        return (
            f"cam={self.camera_index} fps={self.fps:4.1f} "
            f"lat={self.latency_ms:5.1f}ms hand={self.handedness} "
            f"sel={'Y' if self.selected else 'N'} "
            f"serial={'up' if self.serial_connected else 'DOWN'} "
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
        put(f"cam {diag.camera_index}  {diag.fps:4.1f} fps  {diag.latency_ms:4.1f} ms", 20, yellow)

    serial_color = green if diag.serial_connected else red
    put(f"serial: {'connected' if diag.serial_connected else 'DISCONNECTED'}", 40, serial_color)

    track_color = green if diag.track_state == "active" else yellow
    put(f"track: {diag.track_state}   hand: {diag.handedness}   sel: {'Y' if diag.selected else 'N'}", 60, track_color)

    put(f"cam_mirror: {diag.camera_mirror}   out_mirror: {diag.output_mirror}", 80)

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
