"""Smoothing filters to balance servo-jitter vs. demo responsiveness.

Provides:
  * EMA (exponential moving average) -- default, one tunable alpha.
  * One Euro filter -- adaptive: smooths when still, responsive when moving.
  * Deadband -- ignores sub-threshold changes to stop micro-jitter.

All filters operate per-finger on the canonical FINGERS order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .gesture_mapper import HandPose
from .utils import FINGERS, clamp, get_logger

log = get_logger("smoothing")


@dataclass
class SmoothingConfig:
    filter: str = "ema"
    ema_alpha: float = 0.5
    deadband: float = 0.02
    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict) -> "SmoothingConfig":
        one_euro = d.get("one_euro", {}) or {}
        return cls(
            filter=str(d.get("filter", "ema")),
            ema_alpha=float(d.get("ema_alpha", 0.5)),
            deadband=float(d.get("deadband", 0.02)),
            min_cutoff=float(one_euro.get("min_cutoff", 1.0)),
            beta=float(one_euro.get("beta", 0.02)),
            d_cutoff=float(one_euro.get("d_cutoff", 1.0)),
        )


class _OneEuro:
    """Single-channel One Euro filter (Casiez et al., 2012)."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 0:
            dt = 1e-3
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class Smoother:
    """Per-finger smoothing with deadband. Stateful across frames."""

    def __init__(self, cfg: SmoothingConfig) -> None:
        self.cfg = cfg
        self._last: Dict[str, float] = {}
        self._euro: Dict[str, _OneEuro] = {}
        if cfg.filter == "one_euro":
            self._euro = {
                f: _OneEuro(cfg.min_cutoff, cfg.beta, cfg.d_cutoff) for f in FINGERS
            }

    def reset(self) -> None:
        self._last.clear()
        for f in self._euro:
            self._euro[f] = _OneEuro(self.cfg.min_cutoff, self.cfg.beta, self.cfg.d_cutoff)

    def apply(self, pose: HandPose) -> HandPose:
        out: Dict[str, float] = {}
        for finger in FINGERS:
            raw = clamp(getattr(pose, finger), 0.0, 1.0)
            prev = self._last.get(finger)

            if self.cfg.filter == "none":
                val = raw
            elif self.cfg.filter == "one_euro":
                val = clamp(self._euro[finger](raw, pose.timestamp), 0.0, 1.0)
            else:  # ema (default)
                if prev is None:
                    val = raw
                else:
                    a = clamp(self.cfg.ema_alpha, 0.0, 1.0)
                    val = a * raw + (1.0 - a) * prev

            # Deadband: suppress tiny changes to avoid servo chatter.
            if prev is not None and abs(val - prev) < self.cfg.deadband:
                val = prev

            self._last[finger] = val
            out[finger] = val

        return HandPose(
            thumb=out["thumb"],
            index=out["index"],
            middle=out["middle"],
            ring=out["ring"],
            pinky=out["pinky"],
            confidence=pose.confidence,
            handedness=pose.handedness,
            timestamp=pose.timestamp,
        )
