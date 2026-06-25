"""Hand landmark tracking behind a swappable interface.

MediaPipe Hands is the default backend (fastest to prototype, CPU-friendly).
If MediaPipe will not install on the RB3 (ARM64), implement another
`BaseHandTracker` (TFLite / ONNX / Qualcomm QNN-SNPE) and select it via
`tracker.backend` in config -- the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .utils import get_logger

log = get_logger("tracker")


@dataclass
class HandLandmarks:
    """A single detected hand.

    points: (21, 3) array of normalized landmarks (x, y in [0,1], z relative).
    handedness: "Left" or "Right" as reported by the tracker (image space).
    score: detection/handedness confidence in [0, 1].
    bbox: (x_min, y_min, x_max, y_max) in normalized coords.
    """

    points: np.ndarray
    handedness: str
    score: float
    bbox: Tuple[float, float, float, float]

    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class TrackerConfig:
    backend: str = "mediapipe"
    max_num_hands: int = 2
    model_complexity: int = 0
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    select: str = "largest"      # "largest" | "left" | "right"
    min_hand_score: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "TrackerConfig":
        return cls(
            backend=str(d.get("backend", "mediapipe")),
            max_num_hands=int(d.get("max_num_hands", 2)),
            model_complexity=int(d.get("model_complexity", 0)),
            min_detection_confidence=float(d.get("min_detection_confidence", 0.6)),
            min_tracking_confidence=float(d.get("min_tracking_confidence", 0.5)),
            select=str(d.get("select", "largest")),
            min_hand_score=float(d.get("min_hand_score", 0.5)),
        )


class BaseHandTracker(ABC):
    """Interface every tracker backend must implement."""

    @abstractmethod
    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """Return all detected hands for a BGR frame (may be empty)."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""

    # -- shared selection logic --------------------------------------------
    def select_best(
        self, hands: List[HandLandmarks], cfg: TrackerConfig
    ) -> Optional[HandLandmarks]:
        """Pick one hand to drive the robot from possibly many detections."""
        candidates = [h for h in hands if h.score >= cfg.min_hand_score]
        if not candidates:
            return None

        if cfg.select in ("left", "right"):
            want = cfg.select.capitalize()
            matching = [h for h in candidates if h.handedness == want]
            if matching:
                chosen = max(matching, key=lambda h: h.area())
                log.debug("selected %s hand (prefer=%s)", chosen.handedness, cfg.select)
                return chosen
            # fall through to largest if the preferred hand isn't present

        chosen = max(candidates, key=lambda h: h.area())
        log.debug(
            "selected %s hand by largest bbox (area=%.3f, score=%.2f)",
            chosen.handedness, chosen.area(), chosen.score,
        )
        return chosen


# -----------------------------------------------------------------------------
# MediaPipe backend
# -----------------------------------------------------------------------------
class MediaPipeHandTracker(BaseHandTracker):
    """MediaPipe Hands wrapper. Imports lazily so the rest of the project (and
    --dry-run bring-up) works even when MediaPipe is not installed.
    """

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        try:
            import mediapipe as mp  # noqa: WPS433 (intentional lazy import)
        except ImportError as exc:
            raise SystemExit(
                "MediaPipe is not installed.\n"
                "Install with: pip install mediapipe\n"
                "If MediaPipe is unavailable on this RB3/ARM64 image, implement an\n"
                "alternative BaseHandTracker (TFLite/ONNX/QNN) and set\n"
                "tracker.backend accordingly. The pipeline still runs with\n"
                "--dry-run for bring-up.\n"
                f"(import error: {exc})"
            )
        self._mp = mp
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            model_complexity=cfg.model_complexity,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        log.info(
            "MediaPipe Hands ready (max_hands=%d, model_complexity=%d)",
            cfg.max_num_hands, cfg.model_complexity,
        )

    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        import cv2  # local import; OpenCV already a hard dep of camera.py

        # MediaPipe expects RGB; marking the array read-only is a documented
        # micro-optimization that lets it pass by reference.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        out: List[HandLandmarks] = []
        if not result.multi_hand_landmarks:
            return out

        handedness_list = result.multi_handedness or []
        for i, lm in enumerate(result.multi_hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            label = "Unknown"
            score = 1.0
            if i < len(handedness_list) and handedness_list[i].classification:
                cls = handedness_list[i].classification[0]
                label = cls.label  # "Left" / "Right"
                score = float(cls.score)
            xs, ys = pts[:, 0], pts[:, 1]
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            out.append(HandLandmarks(points=pts, handedness=label, score=score, bbox=bbox))
        return out

    def draw(self, frame_bgr: np.ndarray, hand: HandLandmarks) -> None:
        """Draw landmark connections for a selected hand onto the frame."""
        # Rebuild a MediaPipe landmark proto so we can reuse its nice renderer.
        from mediapipe.framework.formats import landmark_pb2

        nl = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in hand.points:
            nl.landmark.add(x=float(x), y=float(y), z=float(z))
        self.drawing.draw_landmarks(
            frame_bgr,
            nl,
            self._mp_hands.HAND_CONNECTIONS,
            self.drawing_styles.get_default_hand_landmarks_style(),
            self.drawing_styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        try:
            self._hands.close()
        except Exception:  # noqa: BLE001
            pass


def create_tracker(cfg: TrackerConfig) -> BaseHandTracker:
    """Factory: instantiate the configured tracker backend."""
    backend = cfg.backend.lower()
    if backend == "mediapipe":
        return MediaPipeHandTracker(cfg)
    raise ValueError(
        f"Unknown tracker backend '{cfg.backend}'. Supported: 'mediapipe'. "
        "Add a new BaseHandTracker subclass to support others."
    )
