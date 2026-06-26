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
INDEX_MCP = 5
PINKY_MCP = 17

# Finger curl blends two cues, both from 3D world landmarks:
#   1. ANGLE: summed flexion at MCP + PIP + DIP. Sensitive to any joint moving.
#   2. FOLD : how far the fingertip has folded back toward its own MCP knuckle
#            (tip->MCP distance / finger length). ~1.0 straight, ~0.3 curled.
# The fold cue matters because (a) it lifts the mid-range so a *partial* bend
# already carries real weight, and (b) it stays meaningful when fingers tuck
# behind the palm and the joint angles "mesh"/collapse -- the tip-to-base
# distance still shrinks even when the per-joint angles get unreliable. Tuned so
# a comfortable (not maximal) close reaches ~1.0 and a half-bend reads ~0.5;
# per-user calibration then fine-tunes. Lower FINGER_FULL_DEG = more gain so you
# don't have to crush a full fist to hit 1.0.
FINGER_FULL_DEG = 210.0      # divisor mapping summed finger flexion -> [0, 1]
FINGER_FOLD_OPEN = 1.00      # tip/MCP-distance ratio when the finger is straight
FINGER_FOLD_CLOSED = 0.35    # ...and when fully folded
FINGER_W_ANGLE = 0.55        # blend weight: joint-angle cue
FINGER_W_FOLD = 0.45         # blend weight: fold-distance cue (occlusion-robust)
THUMB_FLEX_FULL_DEG = 100.0  # divisor for thumb's 2-joint flexion sum

# Thumb adduction (folding across the palm) measured as tip->index-MCP distance
# normalized by palm width. Far (abducted/open) ~1.3, near (adducted/closed)
# ~0.5. This is what makes sideways ("horizontal") thumb motion register.
THUMB_ADD_OPEN = 1.30
THUMB_ADD_CLOSED = 0.50
# Blend of thumb flexion vs adduction. Adduction is weighted a bit higher so
# horizontal thumb motion is captured, while flexion still closes it in a fist.
THUMB_W_FLEX = 0.45
THUMB_W_ADD = 0.55


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
    return float(np.linalg.norm(pts[INDEX_MCP] - pts[PINKY_MCP])) + 1e-6


def _finger_curl_4pt(pts: np.ndarray, idx: tuple) -> float:
    """Raw curl for a 4-landmark finger: blend of joint-angle and fold-distance.

    idx = (mcp, pip, dip, tip). The angle cue sums the bend at MCP+PIP+DIP (each
    ~0 deg straight, growing as it bends). The fold cue is the fingertip's
    distance back to its own MCP, normalized by the finger's length (~1.0
    straight, ~0.3 folded), which lifts partial bends and survives occlusion.
    Both come from 3D world landmarks, so the result is largely orientation
    invariant.
    """
    mcp, pip, dip, tip = (pts[i] for i in idx)
    bend_mcp = angle_between(mcp - pts[WRIST], pip - mcp)  # knuckle flexion vs palm
    bend_pip = angle_between(pip - mcp, dip - pip)         # 1st interphalangeal
    bend_dip = angle_between(dip - pip, tip - dip)         # 2nd interphalangeal
    angle_curl = clamp((bend_mcp + bend_pip + bend_dip) / FINGER_FULL_DEG, 0.0, 1.0)

    finger_len = (
        float(np.linalg.norm(pip - mcp))
        + float(np.linalg.norm(dip - pip))
        + float(np.linalg.norm(tip - dip))
        + 1e-6
    )
    fold_ratio = float(np.linalg.norm(tip - mcp)) / finger_len
    fold_curl = normalize_range(fold_ratio, FINGER_FOLD_OPEN, FINGER_FOLD_CLOSED)

    return clamp(FINGER_W_ANGLE * angle_curl + FINGER_W_FOLD * fold_curl, 0.0, 1.0)


def _thumb_curl(pts: np.ndarray) -> float:
    """Raw thumb curl: blend of joint flexion and adduction across the palm.

    Pure flexion barely moves when the thumb sweeps sideways, so we add an
    adduction term (thumb tip distance to the index MCP, normalized by palm
    width). That makes horizontal thumb motion register, while the flexion term
    still closes the thumb in a fist.
    """
    cmc, mcp, ip, tip = (pts[i] for i in THUMB)
    bend_mcp = angle_between(mcp - cmc, ip - mcp)
    bend_ip = angle_between(ip - mcp, tip - ip)
    flex = clamp((bend_mcp + bend_ip) / THUMB_FLEX_FULL_DEG, 0.0, 1.0)

    palm_w = _palm_width(pts)
    tip_to_index = float(np.linalg.norm(pts[4] - pts[INDEX_MCP])) / palm_w
    add = normalize_range(tip_to_index, THUMB_ADD_OPEN, THUMB_ADD_CLOSED)

    return clamp(THUMB_W_FLEX * flex + THUMB_W_ADD * add, 0.0, 1.0)


_FINGER_IDX = {
    "thumb": THUMB, "index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY,
}


def _curl_one(pts: np.ndarray, name: str) -> float:
    """Curl for one finger from a single landmark array (2D or 3D)."""
    if name == "thumb":
        return _thumb_curl(pts)
    return _finger_curl_4pt(pts, _FINGER_IDX[name])


def compute_raw_curls(hand) -> Dict[str, float]:
    """Per-finger raw curl, blending an in-plane 2D-image estimate with the 3D
    world estimate by how well each finger is seen in the image.

    Why blend: the 2D image landmarks are crisp and reliable for a finger bent
    *within* the image plane -- e.g. a 90-degree bend at the PIP ("hook"), which
    the 3D world model tends to under-rotate, especially for a single finger bent
    on its own. So when a finger is seen side-on (high view confidence) we trust
    the 2D angle; when it is foreshortened (pointing toward/away from the camera)
    we fall back to the orientation-robust 3D estimate.
    """
    world = hand.world_points if hand.world_points is not None else hand.points
    img = hand.points
    aspect = float(getattr(hand, "aspect", 1.0)) or 1.0

    # In-plane 2D copy: de-stretch normalized x and flatten z into the image plane.
    img2d = np.asarray(img, dtype=np.float32).copy()
    img2d[:, 0] *= aspect
    img2d[:, 2] = 0.0

    conf = compute_finger_confidence(hand)  # per-finger "seen side-on" weight
    out: Dict[str, float] = {}
    for name in FINGERS:
        w = clamp(conf.get(name, 0.0), 0.0, 1.0)
        curl_2d = _curl_one(img2d, name)
        curl_3d = _curl_one(world, name)
        out[name] = clamp(w * curl_2d + (1.0 - w) * curl_3d, 0.0, 1.0)
    return out


def _path_len(pts: np.ndarray, idx: tuple, dims: int) -> float:
    """Summed length of the landmark chain `idx`, using the first `dims` coords."""
    total = 0.0
    for k in range(len(idx) - 1):
        total += float(np.linalg.norm(pts[idx[k + 1]][:dims] - pts[idx[k]][:dims]))
    return total


def compute_finger_confidence(hand) -> Dict[str, float]:
    """Per-finger view reliability in [0, 1] from image-vs-world foreshortening.

    Each finger's chain length is measured in the 2D image and in 3D world
    space, both normalized by palm width. When a finger lies in the image plane
    (seen side-on) the two match -> confidence ~1. When it points toward/away
    from the camera -- exactly what a curling finger does when the PALM faces the
    lens -- the image projection shrinks -> low confidence. Multi-camera fusion
    uses this to trust whichever camera sees each finger side-on.
    """
    img = hand.points
    world = hand.world_points if hand.world_points is not None else hand.points
    palm_img = float(np.linalg.norm(img[INDEX_MCP][:2] - img[PINKY_MCP][:2])) + 1e-6
    palm_world = float(np.linalg.norm(world[INDEX_MCP] - world[PINKY_MCP])) + 1e-6
    conf: Dict[str, float] = {}
    for name, idx in _FINGER_IDX.items():
        ext_img = _path_len(img, idx, 2) / palm_img
        ext_world = _path_len(world, idx, 3) / palm_world
        conf[name] = clamp(ext_img / (ext_world + 1e-6), 0.0, 1.0)
    return conf


class GestureMapper:
    """Stateless mapper from landmarks to raw HandPose (pre-calibration)."""

    def map(self, hand: HandLandmarks) -> HandPose:
        # Per-finger blend of 2D-image and 3D-world curl (see compute_raw_curls):
        # accurate for in-plane bends, orientation-robust when foreshortened.
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
