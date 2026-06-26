"""Hand landmark tracking behind a swappable interface.

MediaPipe Hands is the default backend (fastest to prototype, CPU-friendly).
If MediaPipe will not install on the RB3 (ARM64), implement another
`BaseHandTracker` (TFLite / ONNX / Qualcomm QNN-SNPE) and select it via
`tracker.backend` in config -- the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import os
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

    points: (21, 3) array of normalized image landmarks (x, y in [0,1], z rel).
    world_points: (21, 3) metric, view-normalized landmarks in meters relative
        to the hand's geometric center (MediaPipe `hand_world_landmarks`). These
        are far more robust than image landmarks for joint-angle/curl estimation
        because they don't change as the hand rotates or moves in frame. May be
        None if the backend doesn't provide them.
    handedness: "Left" or "Right" as reported by the tracker (image space).
    score: detection/handedness confidence in [0, 1].
    bbox: (x_min, y_min, x_max, y_max) in normalized coords.
    """

    points: np.ndarray
    handedness: str
    score: float
    bbox: Tuple[float, float, float, float]
    world_points: Optional[np.ndarray] = None
    aspect: float = 1.0   # source frame width/height, to de-stretch 2D x coords

    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)


@dataclass
class TrackerConfig:
    backend: str = "mediapipe"
    max_num_hands: int = 2
    model_complexity: int = 0   # legacy; unused by the Tasks API (model file fixes this)
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    select: str = "largest"      # "largest" | "left" | "right"
    min_hand_score: float = 0.5
    model_path: str = "models/hand_landmarker.task"

    @classmethod
    def from_dict(cls, d: dict) -> "TrackerConfig":
        return cls(
            backend=str(d.get("backend", "mediapipe")),
            max_num_hands=int(d.get("max_num_hands", 2)),
            model_complexity=int(d.get("model_complexity", 0)),
            min_detection_confidence=float(d.get("min_detection_confidence", 0.6)),
            min_tracking_confidence=float(d.get("min_tracking_confidence", 0.5)),
            min_presence_confidence=float(d.get("min_presence_confidence", 0.5)),
            select=str(d.get("select", "largest")),
            min_hand_score=float(d.get("min_hand_score", 0.5)),
            model_path=str(d.get("model_path", "models/hand_landmarker.task")),
        )


class BaseHandTracker(ABC):
    """Interface every tracker backend must implement."""

    @abstractmethod
    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """Return all detected hands for a BGR frame (may be empty)."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""

    def draw(self, frame_bgr: np.ndarray, hand: HandLandmarks) -> None:
        """Render landmarks onto a frame, in place. Backends that can draw
        override this; the default is a no-op so callers need no isinstance
        check or hasattr guard."""
        return None

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
    """MediaPipe Tasks `HandLandmarker` backend (VIDEO running mode).

    The legacy `mediapipe.solutions.hands` API was removed in recent mediapipe
    releases, so this loads the `hand_landmarker.task` model bundle through the
    modern Tasks Vision API. Imports are lazy so the rest of the project (and
    --dry-run bring-up) still works when mediapipe is not installed.
    """

    # Standard 21-landmark hand skeleton (index pairs). Used for manual overlay
    # rendering since the legacy solutions.drawing_utils renderer no longer ships.
    _CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),            # index
        (5, 9), (9, 10), (10, 11), (11, 12),       # middle
        (9, 13), (13, 14), (14, 15), (15, 16),     # ring
        (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
        (0, 17),                                   # palm base
    )

    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        try:
            import mediapipe as mp  # noqa: WPS433 (intentional lazy import)
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
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

        model_path = self._resolve_model_path(cfg.model_path)
        if not os.path.isfile(model_path):
            raise SystemExit(
                f"Hand landmark model not found: {model_path}\n"
                "Download it once (~7.8 MB):\n"
                "  curl -L --create-dirs -o models/hand_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task\n"
                "or set tracker.model_path in config.yaml to an existing .task file."
            )

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        log.info(
            "MediaPipe Tasks HandLandmarker ready (num_hands=%d, model=%s)",
            cfg.max_num_hands, os.path.basename(model_path),
        )

    @staticmethod
    def _resolve_model_path(path: str) -> str:
        """Resolve a model path relative to the project root (parent of src/)."""
        if os.path.isabs(path):
            return path
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, path)

    def _next_timestamp_ms(self) -> int:
        """Strictly-increasing timestamp required by VIDEO running mode."""
        ts = int(time.perf_counter() * 1000.0)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def process(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        import cv2  # local import; OpenCV already a hard dep of camera.py

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        h, w = frame_bgr.shape[:2]
        aspect = (w / h) if h else 1.0
        out: List[HandLandmarks] = []
        hand_landmarks = result.hand_landmarks or []
        world_landmarks = result.hand_world_landmarks or []
        handedness = result.handedness or []
        for i, lm_list in enumerate(hand_landmarks):
            pts = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
            world = None
            if i < len(world_landmarks) and world_landmarks[i]:
                world = np.array([[p.x, p.y, p.z] for p in world_landmarks[i]],
                                 dtype=np.float32)
            label = "Unknown"
            score = 1.0
            if i < len(handedness) and handedness[i]:
                cat = handedness[i][0]
                label = cat.category_name or "Unknown"  # "Left" / "Right"
                score = float(cat.score)
            xs, ys = pts[:, 0], pts[:, 1]
            bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            out.append(HandLandmarks(points=pts, handedness=label, score=score,
                                     bbox=bbox, world_points=world, aspect=aspect))
        return out

    def draw(self, frame_bgr: np.ndarray, hand: HandLandmarks) -> None:
        """Render the hand skeleton onto a BGR frame (manual; in place)."""
        import cv2

        h, w = frame_bgr.shape[:2]
        pix = [(int(x * w), int(y * h)) for x, y, _ in hand.points]
        for a, b in self._CONNECTIONS:
            cv2.line(frame_bgr, pix[a], pix[b], (255, 255, 255), 2, cv2.LINE_AA)
        for px, py in pix:
            cv2.circle(frame_bgr, (px, py), 3, (0, 220, 0), -1, cv2.LINE_AA)

    def close(self) -> None:
        try:
            self._landmarker.close()
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
