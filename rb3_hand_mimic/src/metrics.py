"""Responsiveness metrics for the detection/control pipeline.

The whole point of the async revamp is to make hand *detection* as fast and as
responsive as possible, independent of the (possibly slow / not-yet-ported)
hand-control SDK. To know whether we are succeeding we measure, separately:

  * detection latency   -- ms spent per frame inside the tracker+mapper hot path
  * detection rate      -- frames actually processed per second (detect FPS)
  * end-to-end latency  -- camera capture timestamp -> command handed to control
  * pose age            -- how stale a pose is by the time control consumes it
  * control rate        -- how often the control thread re-evaluates safety

This module is dependency-light (stdlib only) and thread-safe: the detection
thread and the control thread both write into the same Metrics instance, while
the main thread reads snapshots for the overlay / benchmark report.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Linear-interpolation percentile (q in [0, 1]) over a sorted list."""
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


@dataclass
class StatSummary:
    """Summary of one rolling latency channel (all values in ms)."""

    count: int = 0
    mean: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    max: float = 0.0


class _Rolling:
    """Fixed-window sample buffer that can summarize itself. Not thread-safe on
    its own; the owning Metrics holds the lock."""

    def __init__(self, window: int) -> None:
        self._v: Deque[float] = deque(maxlen=window)

    def add(self, x: float) -> None:
        self._v.append(x)

    def summary(self) -> StatSummary:
        if not self._v:
            return StatSummary()
        vals = sorted(self._v)
        return StatSummary(
            count=len(vals),
            mean=sum(vals) / len(vals),
            p50=_percentile(vals, 0.50),
            p90=_percentile(vals, 0.90),
            p99=_percentile(vals, 0.99),
            max=vals[-1],
        )


class _Rate:
    """Sliding-window event-rate estimator (events/second)."""

    def __init__(self, window: int) -> None:
        self._ticks: Deque[float] = deque(maxlen=window)

    def tick(self, now: float) -> None:
        self._ticks.append(now)

    def rate(self) -> float:
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        if span <= 0:
            return 0.0
        return (len(self._ticks) - 1) / span


class Metrics:
    """Thread-safe collection of pipeline responsiveness metrics."""

    def __init__(self, window: int = 150) -> None:
        self._lock = threading.Lock()
        self._detect_ms = _Rolling(window)
        self._e2e_ms = _Rolling(window)
        self._pose_age_ms = _Rolling(window)
        self._detect_rate = _Rate(min(window, 90))
        self._control_rate = _Rate(min(window, 90))

        self.frames_processed = 0
        self.hands_detected = 0
        self.commands_sent = 0
        self.control_ticks = 0
        self._t_start = time.perf_counter()

    # -- detection thread ---------------------------------------------------
    def record_detection(self, latency_ms: float, detected: bool) -> None:
        now = time.perf_counter()
        with self._lock:
            self._detect_ms.add(latency_ms)
            self._detect_rate.tick(now)
            self.frames_processed += 1
            if detected:
                self.hands_detected += 1

    # -- control thread -----------------------------------------------------
    def record_control(self, sent: bool) -> None:
        now = time.perf_counter()
        with self._lock:
            self._control_rate.tick(now)
            self.control_ticks += 1
            if sent:
                self.commands_sent += 1

    def record_e2e(self, e2e_ms: float, pose_age_ms: float) -> None:
        with self._lock:
            self._e2e_ms.add(e2e_ms)
            self._pose_age_ms.add(pose_age_ms)

    # -- readers ------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            detect = self._detect_ms.summary()
            e2e = self._e2e_ms.summary()
            pose_age = self._pose_age_ms.summary()
            return {
                "detect_fps": self._detect_rate.rate(),
                "control_hz": self._control_rate.rate(),
                "detect_ms": detect,
                "e2e_ms": e2e,
                "pose_age_ms": pose_age,
                "frames_processed": self.frames_processed,
                "hands_detected": self.hands_detected,
                "detection_rate": (
                    self.hands_detected / self.frames_processed
                    if self.frames_processed else 0.0
                ),
                "commands_sent": self.commands_sent,
                "control_ticks": self.control_ticks,
                "uptime_s": time.perf_counter() - self._t_start,
            }

    def report_line(self) -> str:
        """One-line snapshot for headless logging / benchmark ticks."""
        s = self.snapshot()
        d: StatSummary = s["detect_ms"]
        e: StatSummary = s["e2e_ms"]
        return (
            f"detect {s['detect_fps']:5.1f}fps "
            f"lat p50/p90 {d.p50:4.1f}/{d.p90:4.1f}ms | "
            f"e2e p50/p90 {e.p50:5.1f}/{e.p90:5.1f}ms | "
            f"ctrl {s['control_hz']:5.1f}hz "
            f"det-rate {s['detection_rate']*100:4.0f}% "
            f"cmds {s['commands_sent']}"
        )

    def summary(self) -> str:
        """Multi-line summary printed at the end of a benchmark run."""
        s = self.snapshot()
        d: StatSummary = s["detect_ms"]
        e: StatSummary = s["e2e_ms"]
        a: StatSummary = s["pose_age_ms"]
        return (
            "=== responsiveness summary ===\n"
            f"  uptime              : {s['uptime_s']:.1f} s\n"
            f"  frames processed    : {s['frames_processed']} "
            f"({s['detect_fps']:.1f} fps avg over window)\n"
            f"  hand detection rate : {s['detection_rate']*100:.1f}% "
            f"({s['hands_detected']}/{s['frames_processed']})\n"
            f"  detection latency   : "
            f"mean {d.mean:.1f}  p50 {d.p50:.1f}  p90 {d.p90:.1f}  "
            f"p99 {d.p99:.1f}  max {d.max:.1f}  (ms)\n"
            f"  end-to-end latency  : "
            f"mean {e.mean:.1f}  p50 {e.p50:.1f}  p90 {e.p90:.1f}  "
            f"p99 {e.p99:.1f}  max {e.max:.1f}  (ms, capture->command)\n"
            f"  pose age at control : "
            f"mean {a.mean:.1f}  p50 {a.p50:.1f}  p90 {a.p90:.1f}  (ms)\n"
            f"  control rate        : {s['control_hz']:.1f} hz "
            f"({s['commands_sent']} commands sent)\n"
        )
