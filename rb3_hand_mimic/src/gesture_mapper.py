"""Convert 21 hand landmarks into 5 normalized finger curl values.

Geometry-based (not raw y-coordinate) so it is robust to hand position,
rotation, and scale:

  * index/middle/ring/pinky: average the joint angles at the MCP and PIP
    joints. A straight finger has ~180 deg at each joint; a curled finger has
    much smaller angles. We map the average to a raw curl in [0, 1].
  * thumb: the thumb abducts sideways rather than folding like the others, so
    we combine its IP/MCP joint angles with the tip-to-pinky-MCP distance
    (normalized by palm width). When the thumb folds across the palm the tip
    moves toward the pinky side.

The raw curl produced here is later normalized per-user by calibration.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .hand_tracker import HandLandmarks
from .utils import FINGERS, angle_between, clamp, get_logger, normalize_range

log = get_logger("mapper")

# MediaPipe landmark indices.
WRIST = 0
THUMB = (1, 2, 3, 4)        # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)        # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)

# Joint-angle range used to normalize curl. A fully straight finger sits near
# 180 deg; a tightly curled finger reaches roughly 60-90 deg at the joints.
ANGLE_STRAIGHT = 180.0
ANGLE_CURLED = 70.0


@dataclass
class HandPose:
    """Normalized finger curls in [0, 1]; 0 = open, 1 = closed."""

    thumb: float
    index: float
    middle: float
    ring: float
    pinky: float
    confidence: float
    handedness: str
    timestamp: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "thumb": self.thumb,
            "index": self.index,
            "middle": self.middle,
            "ring": self.ring,
            "pinky": self.pinky,
        }

    def values(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in FINGERS], dtype=np.float32)

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, float],
        confidence: float = 1.0,
        handedness: str = "Unknown",
        timestamp: float = 0.0,
    ) -> "HandPose":
        return cls(
            thumb=float(d.get("thumb", 0.0)),
            index=float(d.get("index", 0.0)),
            middle=float(d.get("middle", 0.0)),
            ring=float(d.get("ring", 0.0)),
            pinky=float(d.get("pinky", 0.0)),
            confidence=confidence,
            handedness=handedness,
            timestamp=timestamp,
        )


def _palm_width(pts: np.ndarray) -> float:
    """Distance between index MCP (5) and pinky MCP (17): a scale reference."""
    return float(np.linalg.norm(pts[5] - pts[17])) + 1e-6


def _finger_curl_4pt(pts: np.ndarray, idx: tuple) -> float:
    """Raw curl for a 4-landmark finger using MCP & PIP joint angles.

    idx = (mcp, pip, dip, tip). Uses 3D landmarks so the estimate degrades
    gracefully when the finger points toward the camera.
    """
    mcp, pip, dip, tip = (pts[i] for i in idx)
    # Angle at MCP joint: vector toward wrist-side is approximated by (mcp->pip)
    # reversed against (mcp->wrist) is unstable, so use the finger's own chain:
    angle_mcp = angle_between(pip - mcp, dip - pip)  # bend between 1st & 2nd seg
    angle_pip = angle_between(dip - pip, tip - dip)  # bend between 2nd & 3rd seg
    # A straight finger -> both angles near 0 deg of *deviation*; angle_between
    # of consecutive segment directions is ~0 when straight and grows when bent.
    # Convert "deviation from straight" into curl: 0 deviation -> open.
    deviation = (angle_mcp + angle_pip) * 0.5  # 0..~120
    # Map deviation [0, (180-ANGLE_CURLED)] -> curl [0, 1].
    max_dev = ANGLE_STRAIGHT - ANGLE_CURLED
    return clamp(deviation / max_dev, 0.0, 1.0)


def _thumb_curl(pts: np.ndarray) -> float:
    """Raw thumb curl combining joint bend and fold-across-palm distance."""
    cmc, mcp, ip, tip = (pts[i] for i in THUMB)
    # Joint bend (deviation from straight), same convention as other fingers.
    angle_mcp = angle_between(ip - mcp, mcp - cmc)
    angle_ip = angle_between(tip - ip, ip - mcp)
    deviation = (angle_mcp + angle_ip) * 0.5
    max_dev = ANGLE_STRAIGHT - ANGLE_CURLED
    bend_curl = clamp(deviation / max_dev, 0.0, 1.0)

    # Fold metric: thumb tip approaches the pinky MCP when closing across palm.
    palm_w = _palm_width(pts)
    tip_to_pinky = float(np.linalg.norm(pts[4] - pts[17])) / palm_w
    # Open thumb: tip far from pinky MCP (ratio ~1.2+); closed: ratio ~0.4.
    fold_curl = normalize_range(tip_to_pinky, 1.2, 0.45)

    # Weight the fold metric a bit higher: it captures abduction the joint
    # angles miss. Tunable, but works well with palm-facing-camera demos.
    return clamp(0.4 * bend_curl + 0.6 * fold_curl, 0.0, 1.0)


def compute_raw_curls(hand: HandLandmarks) -> Dict[str, float]:
    """Compute raw (un-calibrated) curl values for all five fingers."""
    pts = hand.points
    return {
        "thumb": _thumb_curl(pts),
        "index": _finger_curl_4pt(pts, INDEX),
        "middle": _finger_curl_4pt(pts, MIDDLE),
        "ring": _finger_curl_4pt(pts, RING),
        "pinky": _finger_curl_4pt(pts, PINKY),
    }


class GestureMapper:
    """Stateless mapper from landmarks to raw HandPose (pre-calibration)."""

    def map(self, hand: HandLandmarks) -> HandPose:
        raw = compute_raw_curls(hand)
        return HandPose(
            thumb=raw["thumb"],
            index=raw["index"],
            middle=raw["middle"],
            ring=raw["ring"],
            pinky=raw["pinky"],
            confidence=hand.score,
            handedness=hand.handedness,
            timestamp=time.time(),
        )
