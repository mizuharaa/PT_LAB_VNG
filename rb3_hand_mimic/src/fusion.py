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

Current strategy is a PLACEHOLDER: per-source confidence-weighted average, where
confidence is the hand detection score. The real win will come from *per-finger*
confidence (landmark presence/visibility, or "is this finger occluded in this
view") so an occluded finger is taken from whichever camera can actually see it.
That hook is marked TODO(per-finger-confidence).
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


# A source = (curls or None if that camera saw no hand, detection confidence).
Source = Tuple[Optional[Dict[str, float]], float]


def fuse_curls(sources: List[Source], cfg: FusionConfig) -> Optional[Dict[str, float]]:
    """Fuse per-finger curls from several camera views into one curl dict.

    Returns the fused finger->curl mapping, or None if no source had a usable
    detection. With a single usable source this is just that source's curls.
    """
    usable = [(c, max(0.0, w)) for c, w in sources
              if c is not None and w >= cfg.min_confidence]
    if not usable:
        return None
    if len(usable) == 1:
        return {f: clamp(usable[0][0].get(f, 0.0), 0.0, 1.0) for f in FINGERS}

    # PLACEHOLDER: per-source confidence-weighted average per finger.
    # TODO(per-finger-confidence): weight each finger by how well *that* finger
    # is seen in each view (landmark presence / occlusion), not just the whole-
    # hand score, so an occluded finger is taken from the camera that sees it.
    total_w = sum(w for _, w in usable) or 1.0
    fused: Dict[str, float] = {}
    for f in FINGERS:
        s = sum(clamp(c.get(f, 0.0), 0.0, 1.0) * w for c, w in usable)
        fused[f] = clamp(s / total_w, 0.0, 1.0)
    return fused
