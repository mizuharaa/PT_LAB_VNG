"""Mirroring and handedness transform -- the trickiest correctness concern.

There are several independent "flips" that can each invert left/right:

  1. camera_mirror  : the captured image is a selfie view (mirrored). When the
     image is mirrored, MediaPipe's reported handedness is also mirrored, so a
     real right hand is reported as "Left". We correct the reported handedness
     so downstream logic reasons about the *physical* hand.
  2. user_hand vs robot_hand : a right human hand driving a left robot hand (or
     vice versa) needs the finger order preserved but the *curl semantics* are
     identical per finger -- thumb still maps to thumb. Handedness mismatch
     does NOT swap fingers; it only matters for spatial mirroring of the
     preview and any future per-side asymmetry. We expose it for clarity and
     logging and to drive `output_mirror` defaults.
  3. output_mirror : explicitly mirror the finger *order* if the robot's finger
     indexing runs opposite to the human hand. This swaps index<->pinky etc.
  4. invert_fingers: robot closes when the human opens (servo direction).
  5. finger_order  : reorder the 5 values to match the hardware packet order.

Because finger curl is a per-finger scalar (not a spatial coordinate), the
mimic is mostly robust to mirroring: closing your index closes the robot index.
The places mirroring actually matters are (a) the preview overlay and (b) when
the hardware expects the opposite finger ordering -- handled by output_mirror
and finger_order. We keep all of this explicit and logged so the demo can be
corrected on-site quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .gesture_mapper import HandPose
from .utils import FINGERS, clamp, get_logger

log = get_logger("transform")

# Mirroring a hand swaps fingers symmetrically about the middle finger.
_MIRROR_MAP = {
    "thumb": "pinky",
    "index": "ring",
    "middle": "middle",
    "ring": "index",
    "pinky": "thumb",
}


@dataclass
class TransformConfig:
    camera_mirror: bool = True
    output_mirror: bool = False
    user_hand: str = "auto"
    robot_hand: str = "right"
    palm_facing_camera: bool = True
    invert_fingers: bool = False
    finger_order: List[str] = field(default_factory=lambda: list(FINGERS))

    @classmethod
    def from_dict(cls, d: Dict) -> "TransformConfig":
        order: List[str] = [str(f) for f in d.get("finger_order", list(FINGERS))]
        # Validate finger order; fall back to canonical order if malformed.
        if sorted(order) != sorted(FINGERS):
            log.warning("transform.finger_order %r invalid; using default", order)
            order = list(FINGERS)
        return cls(
            camera_mirror=bool(d.get("camera_mirror", True)),
            output_mirror=bool(d.get("output_mirror", False)),
            user_hand=str(d.get("user_hand", "auto")),
            robot_hand=str(d.get("robot_hand", "right")),
            palm_facing_camera=bool(d.get("palm_facing_camera", True)),
            invert_fingers=bool(d.get("invert_fingers", False)),
            finger_order=order,
        )


def correct_handedness(reported: str, camera_mirror: bool) -> str:
    """Return the physical handedness given MediaPipe's reported label.

    MediaPipe reports handedness assuming a non-mirrored image. If our capture
    is mirrored (selfie), a physical right hand is reported as "Left", so we
    flip the label back to physical space.
    """
    if reported not in ("Left", "Right"):
        return reported
    if camera_mirror:
        return "Right" if reported == "Left" else "Left"
    return reported


class Transform:
    """Apply mirroring / handedness / ordering to a normalized HandPose."""

    def __init__(self, cfg: TransformConfig) -> None:
        self.cfg = cfg
        self._logged_once = False

    def physical_handedness(self, pose: HandPose) -> str:
        return correct_handedness(pose.handedness, self.cfg.camera_mirror)

    def apply(self, pose: HandPose) -> HandPose:
        """Return a HandPose whose values reflect mirroring + inversion.

        Note: this does NOT reorder into hardware order -- that happens at the
        controller boundary via `to_ordered_list` so the rest of the pipeline
        always sees canonical finger order.
        """
        values = pose.as_dict()

        if self.cfg.output_mirror:
            # Swap fingers symmetrically (index<->ring, thumb<->pinky).
            values = {dst: values[src] for dst, src in _MIRROR_MAP.items()}

        if self.cfg.invert_fingers:
            values = {f: clamp(1.0 - v, 0.0, 1.0) for f, v in values.items()}

        if not self._logged_once:
            log.info(
                "transform: camera_mirror=%s output_mirror=%s invert=%s "
                "robot_hand=%s physical_handedness=%s",
                self.cfg.camera_mirror, self.cfg.output_mirror,
                self.cfg.invert_fingers, self.cfg.robot_hand,
                self.physical_handedness(pose),
            )
            self._logged_once = True

        return HandPose(
            thumb=values["thumb"],
            index=values["index"],
            middle=values["middle"],
            ring=values["ring"],
            pinky=values["pinky"],
            confidence=pose.confidence,
            handedness=self.physical_handedness(pose),
            timestamp=pose.timestamp,
        )

    def to_ordered_list(self, pose: HandPose) -> List[float]:
        """Return finger values in the hardware finger_order."""
        d = pose.as_dict()
        return [d[f] for f in self.cfg.finger_order]
