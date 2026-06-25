#!/usr/bin/env python3
"""Open a chosen camera, show the live feed, and print measured FPS.

Usage:
  python tools/test_camera.py --camera 2
  python tools/test_camera.py --camera 2 --width 320 --height 240 --fps 60
  python tools/test_camera.py --camera 2 --no-window   # headless FPS check

Press q or ESC to quit (windowed mode).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import Camera, CameraConfig  # noqa: E402
from src.utils import FpsMeter, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Live-test a single camera.")
    ap.add_argument("--camera", type=int, required=True, help="Camera index to open")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="Auto-stop after N seconds (0 = run until q)")
    args = ap.parse_args()

    setup_logging("INFO")

    cfg = CameraConfig(
        index=str(args.camera),
        width=args.width,
        height=args.height,
        fps=args.fps,
        backend=args.backend,
        rotate_degrees=args.rotate,
        mirror_preview=True,
    )

    cam = Camera(args.camera, cfg)
    cam.start()

    meter = FpsMeter(60)
    last_print = time.perf_counter()
    start = time.perf_counter()
    use_window = not args.no_window

    if use_window:
        try:
            import cv2
        except ImportError:
            use_window = False

    try:
        while True:
            frame, fid = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue
            meter.tick()

            now = time.perf_counter()
            if now - last_print >= 1.0:
                print(f"fps={meter.fps():.1f}  frame_id={fid}  size={frame.shape[1]}x{frame.shape[0]}")
                last_print = now

            if use_window:
                disp = cv2.flip(frame, 1) if cfg.mirror_preview else frame
                cv2.putText(disp, f"{meter.fps():.1f} fps", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
                cv2.imshow(f"test_camera {args.camera}", disp)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break

            if args.seconds > 0 and (now - start) >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if use_window:
            import cv2
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
