"""Shared utilities: config loading, logging, geometry, timing helpers.

This module has no heavy dependencies (only numpy + PyYAML) so it can be
imported from CLI tools without pulling in OpenCV/MediaPipe.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "PyYAML is required. Install with: pip install PyYAML\n"
        f"(import error: {exc})"
    )

LOGGER_NAME = "rb3_hand_mimic"

# Canonical finger order used internally everywhere in the pipeline. The
# hardware protocol order is handled separately in transform.py.
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dict.

    Raises a clear error if the file is missing or malformed instead of an
    opaque traceback.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Pass --config <path> or copy the shipped config.yaml."
        )
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse YAML config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping/dict.")
    return data


def save_yaml(path: str, data: Dict[str, Any]) -> None:
    """Write a dict to a YAML file (used by calibration tooling)."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


def config_relative_path(config_path: str, target: str) -> str:
    """Resolve `target` relative to the directory holding `config_path`.

    Keeps the project free of hardcoded absolute paths: a calibration file
    referenced from config.yaml lives next to it regardless of CWD.
    """
    if os.path.isabs(target):
        return target
    base = os.path.dirname(os.path.abspath(config_path))
    return os.path.join(base, target)


def deep_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Fetch a nested config value via dotted path, e.g. "camera.width"."""
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """Configure and return the package logger.

    Safe to call multiple times; existing handlers are cleared first.
    """
    fmt = fmt or "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(fmt)

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child of the package logger."""
    base = logging.getLogger(LOGGER_NAME)
    return base.getChild(name) if name else base


# -----------------------------------------------------------------------------
# Math / geometry helpers
# -----------------------------------------------------------------------------
def clamp(value: float, low: float, high: float) -> float:
    """Clamp a scalar to [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation; t is clamped to [0, 1]."""
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t


def normalize_range(value: float, lo: float, hi: float) -> float:
    """Map `value` from [lo, hi] to [0, 1] with clamping.

    Robust to inverted ranges (lo > hi) which can happen with noisy
    calibration captures.
    """
    if hi == lo:
        return 0.0
    if lo > hi:  # inverted; flip so normalization stays monotonic
        lo, hi = hi, lo
        value = lo + hi - value
    return clamp((value - lo) / (hi - lo), 0.0, 1.0)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the angle in degrees between two vectors.

    Scale-invariant, so it works regardless of how big the hand appears in
    the frame. Guards against zero-length vectors.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos_a = float(np.dot(v1, v2) / (n1 * n2))
    cos_a = clamp(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


# -----------------------------------------------------------------------------
# Timing / FPS
# -----------------------------------------------------------------------------
class FpsMeter:
    """Sliding-window FPS estimator based on inter-tick timestamps."""

    def __init__(self, window: int = 30) -> None:
        self._ticks: Deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._ticks.append(time.perf_counter())

    def fps(self) -> float:
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        if span <= 0:
            return 0.0
        return (len(self._ticks) - 1) / span


@dataclass
class RateLimiter:
    """Simple monotonic rate limiter. `ready()` returns True at most rate_hz."""

    rate_hz: float
    _last: float = field(default=0.0)

    def ready(self) -> bool:
        if self.rate_hz <= 0:
            return True
        now = time.perf_counter()
        if now - self._last >= (1.0 / self.rate_hz):
            self._last = now
            return True
        return False
