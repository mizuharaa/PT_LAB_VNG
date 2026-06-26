"""Human-readable hand-state recognition for the debug view.

Turns the five normalized finger curls into:
  * a per-finger discrete state (extended / half / bent) plus the curl amount,
  * a named gesture (fist, open palm, peace, pointing, three/four up, ...),
  * the physical handedness (we emphasize LEFT -- the Paxini hand is a left hand,
    so getting the left hand right is the priority).

This is intentionally cheap (operates on five floats) so it can run every frame
without hurting the detection hot path.

Limitations (documented honestly):
  * "fingers crossed" and dynamic gestures like "wave" cannot be determined from
    curl magnitudes alone -- they need lateral landmark geometry / temporal
    motion. Those are flagged as TODO and will become reliable once the second
    camera + fusion (and a short motion history) are in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .utils import FINGERS, clamp

# Curl thresholds for the discrete per-finger state.
EXTENDED_MAX = 0.30   # curl <= this -> finger is straight/extended
BENT_MIN = 0.65       # curl >= this -> finger is folded/bent; between -> half

STATE_EXTENDED = "extended"
STATE_HALF = "half"
STATE_BENT = "bent"


@dataclass
class FingerState:
    name: str
    curl: float
    state: str

    def short(self) -> str:
        tag = {"extended": "ext", "half": "half", "bent": "bent"}[self.state]
        return f"{self.name[0].upper()}:{tag} {self.curl:.2f}"


@dataclass
class HandState:
    handedness: str                       # "Left" | "Right" | "Unknown"
    gesture: str                          # human-readable label
    fingers: List[FingerState] = field(default_factory=list)
    extended: List[str] = field(default_factory=list)  # names fully extended

    def is_left(self) -> bool:
        return self.handedness.lower().startswith("l")

    def summary(self) -> str:
        hand = self.handedness.upper() if self.handedness != "Unknown" else "?"
        states = "  ".join(f.short() for f in self.fingers)
        return f"{hand} hand | {self.gesture} | {states}"


def _finger_state(curl: float) -> str:
    if curl <= EXTENDED_MAX:
        return STATE_EXTENDED
    if curl >= BENT_MIN:
        return STATE_BENT
    return STATE_HALF


# Gesture lookup keyed by the frozenset of *fully extended* finger names. Only
# unambiguous, curl-derivable poses are listed; anything else falls back to a
# generic "<n> fingers up" / partial description.
_GESTURES: Dict[frozenset, str] = {
    frozenset(): "Fist",
    frozenset(FINGERS): "Open palm",
    frozenset({"index"}): "Pointing (index)",
    frozenset({"middle"}): "Middle finger",
    frozenset({"pinky"}): "Pinky up",
    frozenset({"thumb"}): "Thumbs up",
    frozenset({"index", "middle"}): "Peace / Victory",
    frozenset({"thumb", "index"}): "L / Finger gun",
    frozenset({"thumb", "pinky"}): "Call me",
    frozenset({"index", "pinky"}): "Rock on",
    frozenset({"thumb", "index", "pinky"}): "I love you",
    frozenset({"index", "middle", "ring"}): "Three up",
    frozenset({"index", "middle", "ring", "pinky"}): "Four up",
    frozenset({"thumb", "index", "middle"}): "Three (OK-spread)",
}


def classify(curls: Dict[str, float], handedness: str = "Unknown") -> HandState:
    """Classify a pose (finger -> curl 0..1) into a HandState."""
    fingers: List[FingerState] = []
    extended: List[str] = []
    for f in FINGERS:
        c = clamp(float(curls.get(f, 0.0)), 0.0, 1.0)
        st = _finger_state(c)
        fingers.append(FingerState(name=f, curl=c, state=st))
        if st == STATE_EXTENDED:
            extended.append(f)

    key = frozenset(extended)
    if key in _GESTURES:
        gesture = _GESTURES[key]
    else:
        n = len(extended)
        half = [f.name for f in fingers if f.state == STATE_HALF]
        if n == 0:
            gesture = "Closing / fist-ish"
        elif half:
            gesture = f"{n} up (+{len(half)} half: {','.join(h[0].upper() for h in half)})"
        else:
            gesture = f"{n} fingers up ({','.join(e[0].upper() for e in extended)})"

    # TODO(2-cam/motion): detect crossed fingers (needs lateral landmark overlap)
    # and dynamic gestures like "wave" (needs a short motion history).
    return HandState(handedness=handedness, gesture=gesture,
                     fingers=fingers, extended=extended)
