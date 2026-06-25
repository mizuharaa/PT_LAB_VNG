"""Safety layer between the smoothed pose and the hardware.

Responsibilities (project priority #4 & #5):
  * Clamp every output to [output_min, output_max].
  * Rate-limit per-finger change (max_step_per_command) to avoid servo slamming.
  * Tracking-loss policy: hold the last pose for `hold_seconds`, then ease back
    to the rest pose over `return_seconds`.
  * Watchdog: if no fresh pose arrives within `watchdog_seconds`, force the
    return-to-rest behavior so stale commands aren't repeated forever.

The SafetyManager is a pure state machine over time; it does not do any I/O.
Feeding it the current time explicitly keeps it deterministic and testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .utils import FINGERS, clamp, get_logger, lerp

log = get_logger("safety")


class TrackState(Enum):
    ACTIVE = "active"      # fresh pose available this tick
    HOLDING = "holding"    # tracking lost, holding last pose
    RETURNING = "returning"  # easing back toward rest
    REST = "rest"          # settled at rest pose


@dataclass
class SafetyConfig:
    output_min: float = 0.0
    output_max: float = 1.0
    hold_seconds: float = 0.5
    return_seconds: float = 1.0
    watchdog_seconds: float = 1.5
    max_step_per_command: float = 0.25

    @classmethod
    def from_dict(cls, d: Dict) -> "SafetyConfig":
        return cls(
            output_min=float(d.get("output_min", 0.0)),
            output_max=float(d.get("output_max", 1.0)),
            hold_seconds=float(d.get("hold_seconds", 0.5)),
            return_seconds=float(d.get("return_seconds", 1.0)),
            watchdog_seconds=float(d.get("watchdog_seconds", 1.5)),
            max_step_per_command=float(d.get("max_step_per_command", 0.25)),
        )


class SafetyManager:
    def __init__(self, cfg: SafetyConfig, rest_pose: Dict[str, float]) -> None:
        self.cfg = cfg
        self.rest_pose = {f: clamp(rest_pose.get(f, 0.0), 0.0, 1.0) for f in FINGERS}
        self._output: Dict[str, float] = dict(self.rest_pose)
        self._last_good: Dict[str, float] = dict(self.rest_pose)
        self._last_update_ts: float = 0.0   # last time we got a fresh pose
        self._return_start_ts: float = 0.0
        self._return_from: Dict[str, float] = dict(self.rest_pose)
        self.state = TrackState.REST

    # -- inputs -------------------------------------------------------------
    def update_pose(self, values: Dict[str, float], now: Optional[float] = None) -> None:
        """Register a fresh, valid pose."""
        now = now if now is not None else time.perf_counter()
        self._last_good = {f: clamp(values.get(f, 0.0), 0.0, 1.0) for f in FINGERS}
        self._last_update_ts = now
        if self.state != TrackState.ACTIVE:
            log.debug("tracking re-acquired -> ACTIVE")
        self.state = TrackState.ACTIVE

    # -- output -------------------------------------------------------------
    def compute_output(self, now: Optional[float] = None) -> Dict[str, float]:
        """Return the safe command for this tick based on time + last pose."""
        now = now if now is not None else time.perf_counter()
        age = now - self._last_update_ts if self._last_update_ts else 1e9

        if self.state == TrackState.ACTIVE:
            if age <= self.cfg.hold_seconds:
                target = self._last_good
            else:
                self._enter_holding(now)
                target = self._last_good
        elif self.state == TrackState.HOLDING:
            if age <= self.cfg.hold_seconds and age < self.cfg.watchdog_seconds:
                target = self._last_good
            else:
                self._enter_returning(now)
                target = self._return_target(now)
        elif self.state == TrackState.RETURNING:
            target = self._return_target(now)
        else:  # REST
            target = self.rest_pose

        # Watchdog override: stale beyond watchdog always returns to rest.
        if age >= self.cfg.watchdog_seconds and self.state in (TrackState.ACTIVE, TrackState.HOLDING):
            log.warning("watchdog: no pose for %.2fs -> returning to rest", age)
            self._enter_returning(now)
            target = self._return_target(now)

        self._output = self._rate_limit_and_clamp(target)
        return dict(self._output)

    # -- state transitions --------------------------------------------------
    def _enter_holding(self, now: float) -> None:
        if self.state != TrackState.HOLDING:
            log.info("tracking lost -> HOLDING last pose for up to %.2fs", self.cfg.hold_seconds)
        self.state = TrackState.HOLDING

    def _enter_returning(self, now: float) -> None:
        if self.state != TrackState.RETURNING:
            log.info("returning to rest pose over %.2fs", self.cfg.return_seconds)
            self._return_start_ts = now
            self._return_from = dict(self._output)
        self.state = TrackState.RETURNING

    def _return_target(self, now: float) -> Dict[str, float]:
        if self.cfg.return_seconds <= 0:
            self.state = TrackState.REST
            return self.rest_pose
        t = (now - self._return_start_ts) / self.cfg.return_seconds
        if t >= 1.0:
            self.state = TrackState.REST
            return self.rest_pose
        return {f: lerp(self._return_from[f], self.rest_pose[f], t) for f in FINGERS}

    # -- clamping / rate limiting ------------------------------------------
    def _rate_limit_and_clamp(self, target: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        step = self.cfg.max_step_per_command
        for f in FINGERS:
            desired = clamp(target.get(f, 0.0), self.cfg.output_min, self.cfg.output_max)
            prev = self._output.get(f, desired)
            if step > 0:
                delta = clamp(desired - prev, -step, step)
                val = prev + delta
            else:
                val = desired
            out[f] = clamp(val, self.cfg.output_min, self.cfg.output_max)
        return out

    def current_state(self) -> str:
        return self.state.value
