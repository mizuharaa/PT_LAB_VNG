"""Per-user / per-camera calibration of raw finger curls.

A raw curl from gesture_mapper is mapped to a normalized curl using per-finger
(open, closed) reference values. This adapts to different hand sizes, camera
placement, and lighting.

Calibration data lives in its own YAML file (default calibration.yaml next to
config.yaml) and is produced by tools/record_calibration.py. If it is missing
or invalid we fall back to the safe defaults embedded in config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .gesture_mapper import HandPose
from .utils import FINGERS, get_logger, load_config, normalize_range, save_yaml

log = get_logger("calibration")


@dataclass
class Calibration:
    """Per-finger (open, closed) raw-curl reference points."""

    # finger -> (raw_open, raw_closed)
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    source: str = "defaults"

    def is_valid(self, min_separation: float = 0.15) -> bool:
        if set(self.ranges.keys()) != set(FINGERS):
            return False
        for finger, (lo, hi) in self.ranges.items():
            if abs(hi - lo) < min_separation:
                log.warning(
                    "calibration for %s is suspicious: open=%.3f closed=%.3f "
                    "(separation < %.3f)", finger, lo, hi, min_separation,
                )
                return False
        return True

    def normalize(self, pose: HandPose) -> HandPose:
        """Return a new HandPose with raw curls normalized to [0, 1]."""
        norm: Dict[str, float] = {}
        for finger in FINGERS:
            raw = getattr(pose, finger)
            lo, hi = self.ranges.get(finger, (0.0, 1.0))
            norm[finger] = normalize_range(raw, lo, hi)
        return HandPose(
            thumb=norm["thumb"],
            index=norm["index"],
            middle=norm["middle"],
            ring=norm["ring"],
            pinky=norm["pinky"],
            confidence=pose.confidence,
            handedness=pose.handedness,
            timestamp=pose.timestamp,
        )

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "ranges": {f: {"open": lo, "closed": hi} for f, (lo, hi) in self.ranges.items()},
        }

    @classmethod
    def from_defaults(cls, defaults: Dict[str, Dict[str, float]]) -> "Calibration":
        ranges: Dict[str, Tuple[float, float]] = {}
        for finger in FINGERS:
            d = defaults.get(finger, {})
            ranges[finger] = (float(d.get("open", 0.1)), float(d.get("closed", 0.9)))
        return cls(ranges=ranges, source="defaults")

    @classmethod
    def from_file(cls, path: str) -> Optional["Calibration"]:
        if not os.path.isfile(path):
            return None
        try:
            data = load_config(path)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("could not read calibration file %s: %s", path, exc)
            return None
        raw_ranges = data.get("ranges", {})
        ranges: Dict[str, Tuple[float, float]] = {}
        for finger in FINGERS:
            d = raw_ranges.get(finger, {})
            if "open" not in d or "closed" not in d:
                log.warning("calibration file missing finger %s; ignoring file", finger)
                return None
            ranges[finger] = (float(d["open"]), float(d["closed"]))
        return cls(ranges=ranges, source=path)

    def save(self, path: str) -> None:
        save_yaml(path, self.to_dict())
        log.info("saved calibration to %s", path)


def load_calibration(cfg: Dict, config_path: str) -> Calibration:
    """Load calibration honoring config; fall back to defaults safely.

    `cfg` is the full parsed config dict. `config_path` anchors relative paths.
    """
    from .utils import config_relative_path

    cal_cfg = cfg.get("calibration", {}) or {}
    defaults = cal_cfg.get("defaults", {}) or {}
    min_sep = float(cal_cfg.get("min_separation", 0.15))
    fallback = Calibration.from_defaults(defaults)

    if not cal_cfg.get("enabled", True):
        log.info("calibration disabled in config; using default ranges")
        return fallback

    cal_file = cal_cfg.get("file")
    if cal_file:
        path = config_relative_path(config_path, cal_file)
        loaded = Calibration.from_file(path)
        if loaded is not None and loaded.is_valid(min_sep):
            log.info("loaded calibration from %s", path)
            return loaded
        if loaded is not None:
            log.warning("calibration in %s failed validation; using defaults", path)
        else:
            log.info("no calibration file at %s; using safe defaults", path)

    return fallback
