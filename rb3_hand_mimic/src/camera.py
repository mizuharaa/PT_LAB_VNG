"""Camera discovery and a latest-frame-only threaded capture.

Design goals (per project priorities #1 responsiveness):
  * Never let the AI loop fall behind the camera: the capture thread keeps only
    the most recent frame and drops everything older.
  * Do not hardcode index 0. Auto-discovery enumerates /dev/video*, tries
    OpenCV indexes 0..8, and scores cameras to pick the RB3 tracking camera.
  * Backend selection (V4L2/GStreamer) is configurable for the RB3.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .utils import get_logger

log = get_logger("camera")

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "OpenCV is required. Install with: pip install opencv-python\n"
        "On the RB3 you can also use: sudo apt install -y python3-opencv\n"
        f"(import error: {exc})"
    )


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------
@dataclass
class CameraConfig:
    preferred: str = "tracking"
    index: str = "auto"          # "auto" or stringified int
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "auto"
    mirror_preview: bool = True
    rotate_degrees: int = 0
    buffer_size: int = 1
    prefer_indices: List[int] = field(default_factory=list)
    avoid_indices: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict) -> "CameraConfig":
        return cls(
            preferred=str(d.get("preferred", "tracking")),
            index=str(d.get("index", "auto")),
            width=int(d.get("width", 640)),
            height=int(d.get("height", 480)),
            fps=int(d.get("fps", 30)),
            backend=str(d.get("backend", "auto")),
            mirror_preview=bool(d.get("mirror_preview", True)),
            rotate_degrees=int(d.get("rotate_degrees", 0)),
            buffer_size=int(d.get("buffer_size", 1)),
            prefer_indices=list(d.get("prefer_indices", []) or []),
            avoid_indices=list(d.get("avoid_indices", []) or []),
        )


@dataclass
class CameraInfo:
    """Result of probing a single camera index."""

    index: int
    opened: bool
    reads: bool
    width: int
    height: int
    fps: float
    device: str = ""
    name: str = ""
    note: str = ""


# -----------------------------------------------------------------------------
# Backend helpers
# -----------------------------------------------------------------------------
def _backend_candidates(backend: str) -> List[int]:
    """Ordered OpenCV backend flags to try for a config backend string.

    Explicit choices ("v4l2"/"gstreamer") are honored exactly. "auto"/"any"
    stays portable across architectures and OSes:
      * Linux (x86_64 *and* aarch64/RB3): try V4L2 first (USB/UVC webcams), then
        fall back to CAP_ANY -- which lets OpenCV reach a GStreamer/MIPI sensor
        when V4L2 cannot open it, instead of hard-failing.
      * Other OSes (Windows/macOS used for dev): CAP_ANY only. V4L2 does not
        exist off Linux, so we must not force it on bare "posix" (e.g. macOS).
    """
    b = (backend or "auto").lower()
    if b == "v4l2":
        return [cv2.CAP_V4L2]
    if b == "gstreamer":
        return [cv2.CAP_GSTREAMER]
    # auto / any (and any unknown value): platform-aware with graceful fallback.
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def _open_capture(index: int, backend: str) -> Optional["cv2.VideoCapture"]:
    """Open a VideoCapture, trying each candidate backend in order.

    Returns an opened capture, or None if every backend failed. Centralizes the
    cross-platform backend fallback so probe/open/debug-frame behave identically.
    """
    for api in _backend_candidates(backend):
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def list_video_devices() -> List[str]:
    """Return sorted /dev/video* device paths (empty on non-Linux)."""
    return sorted(glob.glob("/dev/video*"))


def v4l2_list_devices() -> Dict[int, str]:
    """Map /dev/videoN -> human-readable device name via `v4l2-ctl`.

    Returns {} if v4l2-ctl is unavailable. Output format example:

        Qualcomm Tracking Camera (usb-...):
                /dev/video2
                /dev/video3
    """
    mapping: Dict[int, str] = {}
    if shutil.which("v4l2-ctl") is None:
        return mapping
    try:
        out = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        log.debug("v4l2-ctl failed: %s", exc)
        return mapping

    current_name = ""
    for line in out.splitlines():
        if not line.strip():
            continue
        if not line.startswith((" ", "\t")):
            current_name = line.strip().rstrip(":")
        else:
            dev = line.strip()
            if dev.startswith("/dev/video"):
                try:
                    idx = int(dev.replace("/dev/video", ""))
                    mapping[idx] = current_name
                except ValueError:
                    pass
    return mapping


def _looks_like_tracking(name: str) -> bool:
    """Heuristic: does this camera name look like a tracking/low-res sensor?"""
    n = name.lower()
    keywords = ("track", "tracking", "stereo", "mono", "ir", "fisheye", "global")
    return any(k in n for k in keywords)


def _looks_like_main(name: str) -> bool:
    n = name.lower()
    keywords = ("main", "hires", "high", "rgb", "color", "imx", "primary")
    return any(k in n for k in keywords)


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------
def probe_camera(
    index: int,
    backend: str = "auto",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    read_frames: int = 10,
    name: str = "",
) -> CameraInfo:
    """Open a camera index, attempt to read frames, and estimate FPS."""
    device = f"/dev/video{index}" if sys.platform.startswith("linux") else f"index {index}"
    cap = _open_capture(index, backend)
    if cap is None:
        return CameraInfo(index, False, False, 0, 0, 0.0, device, name, "did not open")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    ok_count = 0
    start = time.perf_counter()
    actual_w = actual_h = 0
    for _ in range(read_frames):
        ok, frame = cap.read()
        if ok and frame is not None:
            ok_count += 1
            actual_h, actual_w = frame.shape[:2]
    elapsed = time.perf_counter() - start
    est_fps = (ok_count / elapsed) if elapsed > 0 and ok_count else 0.0

    cap.release()
    return CameraInfo(
        index=index,
        opened=True,
        reads=ok_count > 0,
        width=actual_w,
        height=actual_h,
        fps=round(est_fps, 1),
        device=device,
        name=name,
        note="ok" if ok_count else "opened but no frames",
    )


def discover_cameras(
    cfg: CameraConfig,
    max_index: int = 8,
    save_debug_dir: Optional[str] = None,
) -> List[CameraInfo]:
    """Probe indexes 0..max_index, annotate with v4l2 names, optionally save
    one debug frame per working camera.
    """
    names = v4l2_list_devices()
    results: List[CameraInfo] = []

    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)

    for idx in range(max_index + 1):
        info = probe_camera(
            idx,
            backend=cfg.backend,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            name=names.get(idx, ""),
        )
        results.append(info)
        if info.reads:
            log.info(
                "camera %d: %s  %dx%d ~%.1ffps  [%s]",
                idx, info.device, info.width, info.height, info.fps,
                info.name or "unknown",
            )
            if save_debug_dir:
                _save_debug_frame(idx, cfg, save_debug_dir, info.name)
        else:
            log.debug("camera %d: %s", idx, info.note)
    return results


def _save_debug_frame(index: int, cfg: CameraConfig, out_dir: str, name: str) -> None:
    cap = _open_capture(index, cfg.backend)
    if cap is None:
        return
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        ok, frame = cap.read()
        if ok and frame is not None:
            safe_name = "".join(c if c.isalnum() else "_" for c in name)[:40]
            path = os.path.join(out_dir, f"cam{index}_{safe_name or 'unknown'}.jpg")
            cv2.imwrite(path, frame)
            log.info("saved debug frame: %s", path)
    finally:
        cap.release()


def select_camera_index(cfg: CameraConfig, infos: List[CameraInfo]) -> Optional[int]:
    """Pick the best camera index given config preference and probe results.

    Scoring favors the tracking camera when preferred=="tracking", honors
    prefer/avoid index lists, and falls back to the first working camera.
    """
    working = [i for i in infos if i.reads]
    if not working:
        return None

    def score(info: CameraInfo) -> float:
        s = 0.0
        if info.index in cfg.avoid_indices:
            s -= 100.0
        if info.index in cfg.prefer_indices:
            s += 50.0
        if cfg.preferred == "tracking":
            if _looks_like_tracking(info.name):
                s += 30.0
            if _looks_like_main(info.name):
                s -= 10.0
        elif cfg.preferred == "main":
            if _looks_like_main(info.name):
                s += 30.0
            if _looks_like_tracking(info.name):
                s -= 10.0
        # Tie-break: prefer the camera that actually reached the requested fps.
        s += min(info.fps, 60.0) * 0.1
        # Slight bias to lower indices for determinism.
        s -= info.index * 0.01
        return s

    best = max(working, key=score)
    log.info(
        "auto-selected camera %d (%s) for preferred=%s",
        best.index, best.name or "unknown", cfg.preferred,
    )
    return best.index


# -----------------------------------------------------------------------------
# Threaded latest-frame capture
# -----------------------------------------------------------------------------
class Camera:
    """Threaded camera that always exposes the most recent frame.

    Old frames are dropped (never queued) so downstream stages process the
    freshest image. `read()` returns (frame, frame_id) where frame_id lets the
    consumer detect duplicate/stale frames.
    """

    def __init__(self, index: int, cfg: CameraConfig) -> None:
        self.index = index
        self.cfg = cfg
        self._cap: Optional["cv2.VideoCapture"] = None
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_capture_ts = 0.0

    # -- lifecycle ----------------------------------------------------------
    def open(self) -> None:
        cap = _open_capture(self.index, self.cfg.backend)
        if cap is None:
            raise RuntimeError(
                f"Could not open camera index {self.index}.\n"
                "Hints: run `python tools/discover_cameras.py`, check that the "
                "device exists under /dev/video*, and that no other process is "
                "using it."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        # buffer_size=1 keeps latency low where the backend honors it.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, self.cfg.buffer_size))
        except Exception:  # noqa: BLE001 - not all backends support it
            pass
        self._cap = cap

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        log.info("opened camera %d at %dx%d (reported fps=%.1f)", self.index, w, h, fps)

    def start(self) -> "Camera":
        if self._cap is None:
            self.open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera-capture", daemon=True)
        self._thread.start()
        # Wait briefly for the first frame so consumers don't spin on None.
        deadline = time.perf_counter() + 3.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._frame is not None:
                    break
            time.sleep(0.005)
        return self

    def _loop(self) -> None:
        assert self._cap is not None
        rotate_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        rot = rotate_map.get(self.cfg.rotate_degrees % 360)
        consecutive_failures = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                if consecutive_failures % 30 == 0:
                    log.warning("camera %d read failed (%d times)", self.index, consecutive_failures)
                time.sleep(0.005)
                continue
            consecutive_failures = 0
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._last_capture_ts = time.perf_counter()

    def read(self) -> Tuple[Optional["cv2.typing.MatLike"], int]:
        """Return (latest_frame_copy_ref, frame_id). frame may be None early on."""
        with self._lock:
            return self._frame, self._frame_id

    def last_capture_ts(self) -> float:
        with self._lock:
            return self._last_capture_ts

    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log.info("camera %d released", self.index)

    # context manager sugar
    def __enter__(self) -> "Camera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.release()


def open_camera_from_config(
    cfg: CameraConfig,
    forced_index: Optional[int] = None,
) -> Camera:
    """Resolve the camera index (explicit > forced > auto) and return a Camera.

    `forced_index` comes from the CLI (`--camera-index`) and overrides config.
    """
    if forced_index is not None:
        log.info("using CLI-forced camera index %d", forced_index)
        return Camera(forced_index, cfg)

    if str(cfg.index).lower() != "auto":
        try:
            idx = int(cfg.index)
            log.info("using config camera index %d", idx)
            return Camera(idx, cfg)
        except ValueError:
            log.warning("config camera.index=%r invalid, falling back to auto", cfg.index)

    log.info("auto-discovering camera (preferred=%s)...", cfg.preferred)
    infos = discover_cameras(cfg)
    idx = select_camera_index(cfg, infos)
    if idx is None:
        raise RuntimeError(
            "No working camera found. Run `python tools/discover_cameras.py` and "
            "set camera.index explicitly in config.yaml."
        )
    return Camera(idx, cfg)
