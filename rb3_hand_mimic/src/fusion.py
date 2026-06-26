"""Multi-camera curl fusion (placeholder strategy).

Motivation: when fingers fold behind the palm, their landmarks bunch together
("mesh") and a single camera under-reports the curl. A second camera at a known
offset sees the hand from a different angle, so for any given finger at least one
view is usually unobstructed. Fusing the two per-finger curls recovers weight
the single view loses.

This module only does the *math* on plain curl dicts (finger -> 0..1) plus a
confidence per source, so it has no dependency on the pipeline/camera modules
(keeps it cycle-free and unit-testable). The pipeline's FusionWorker adapts
PoseSamples to/from this.

Fusion is per-FINGER, not per-hand: each finger is a confidence-weighted average
across the cameras, where the weight is the camera's hand score times that
finger's reliability in that view. The per-finger reliability (computed in
gesture_mapper.compute_finger_confidence) drops when a finger points toward/away
from the camera (foreshortened) -- which is precisely the palm-facing curl case.
So a finger that is ambiguous to one camera is taken from the camera that sees it
side-on. This is the key to making palm-facing detection as good as back-of-hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .utils import FINGERS, clamp, get_logger

log = get_logger("fusion")


@dataclass
class FusionConfig:
    strategy: str = "confidence_weighted"
    min_confidence: float = 0.3   # ignore a source below this detection score

    @classmethod
    def from_dict(cls, d: Dict) -> "FusionConfig":
        d = d or {}
        return cls(
            strategy=str(d.get("strategy", "confidence_weighted")),
            min_confidence=float(d.get("min_confidence", 0.3)),
        )


# A source per camera: (curls or None, hand detection score, per-finger conf or None).
Source = Tuple[Optional[Dict[str, float]], float, Optional[Dict[str, float]]]


def fuse_curls(sources: List[Source], cfg: FusionConfig) -> Optional[Dict[str, float]]:
    """Fuse per-finger curls from several camera views into one curl dict.

    Each finger is averaged across cameras weighted by (hand_score x
    finger_confidence), so the camera that sees a given finger best dominates
    that finger. Returns the fused finger->curl mapping, or None if no source
    had a usable detection.
    """
    usable = [(c, max(0.0, hw), fc) for c, hw, fc in sources
              if c is not None and hw >= cfg.min_confidence]
    if not usable:
        return None

    fused: Dict[str, float] = {}
    for f in FINGERS:
        num = 0.0
        den = 0.0
        for curls, hw, fconf in usable:
            # Per-finger weight: whole-hand score x how well this camera sees
            # this finger (1.0 if no per-finger confidence was supplied). A small
            # floor keeps a finger from going weightless across all cameras.
            fw = hw * (fconf.get(f, 1.0) if fconf else 1.0)
            fw = max(fw, 1e-3)
            num += clamp(curls.get(f, 0.0), 0.0, 1.0) * fw
            den += fw
        fused[f] = clamp(num / den, 0.0, 1.0) if den > 0 else 0.0
    return fused
