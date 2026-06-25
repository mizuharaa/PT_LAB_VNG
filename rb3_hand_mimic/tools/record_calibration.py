#!/usr/bin/env python3
"""Guided open/closed-hand calibration.

Captures stable per-finger raw curl averages for an OPEN palm and a CLOSED
fist, validates the separation, and writes the result to the calibration file
referenced by config.yaml (default calibration.yaml).

Usage:
  python tools/record_calibration.py --config config.yaml
  python tools/record_calibration.py --config config.yaml --camera-index 2
  python tools/record_calibration.py --config config.yaml --no-window
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration import Calibration  # noqa: E402
from src.camera import CameraConfig, open_camera_from_config  # noqa: E402
from src.gesture_mapper import GestureMapper  # noqa: E402
from src.hand_tracker import TrackerConfig, create_tracker  # noqa: E402
from src.utils import (  # noqa: E402
    FINGERS,
    config_relative_path,
    get_logger,
    load_config,
    setup_logging,
)

log = get_logger("calib-tool")


def capture_average(
    camera, tracker, tracker_cfg, mapper, seconds: float, show_window: bool, label: str
) -> Optional[Dict[str, float]]:
    """Average raw curls over `seconds` of frames where a hand is detected."""
    samples: Dict[str, List[float]] = {f: [] for f in FINGERS}
    cv2 = None
    if show_window:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            show_window = False

    end = time.perf_counter() + seconds
    last_id = -1
    detected_frames = 0
    while time.perf_counter() < end:
        frame, fid = camera.read()
        if frame is None or fid == last_id:
            time.sleep(0.003)
            continue
        last_id = fid
        hands = tracker.process(frame)
        hand = tracker.select_best(hands, tracker_cfg)
        if hand is not None:
            pose = mapper.map(hand)
            for f in FINGERS:
                samples[f].append(getattr(pose, f))
            detected_frames += 1
        if show_window and cv2 is not None:
            disp = cv2.flip(frame, 1)
            color = (0, 220, 0) if hand is not None else (0, 0, 220)
            cv2.putText(disp, f"{label}: hold still ({detected_frames} samples)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.imshow("record_calibration", disp)
            cv2.waitKey(1)

    if detected_frames < 5:
        log.warning("only %d frames had a detected hand for '%s'", detected_frames, label)
        return None
    return {f: (sum(v) / len(v)) if v else 0.0 for f, v in samples.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Record open/closed hand calibration.")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--camera-index", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=1.0, help="Averaging window per pose")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    setup_logging("INFO")
    cfg = load_config(args.config)

    cam_cfg = CameraConfig.from_dict(cfg.get("camera", {}))
    tracker_cfg = TrackerConfig.from_dict(cfg.get("tracker", {}))
    cal_cfg = cfg.get("calibration", {}) or {}
    min_sep = float(cal_cfg.get("min_separation", 0.15))
    show_window = not args.no_window

    mapper = GestureMapper()
    tracker = create_tracker(tracker_cfg)
    camera = open_camera_from_config(cam_cfg, args.camera_index)
    camera.start()

    try:
        print("\n=== Calibration ===")
        print("Face your palm toward the TRACKING camera.\n")

        input("Step 1/2: Show an OPEN palm (fingers spread). Press ENTER, then hold still...")
        open_vals = capture_average(camera, tracker, tracker_cfg, mapper,
                                    args.seconds, show_window, "OPEN palm")
        if open_vals is None:
            print("Could not capture open pose (no hand detected). Improve lighting / framing.")
            return 1
        print("  open raw:", {k: round(v, 3) for k, v in open_vals.items()})

        input("\nStep 2/2: Make a tight FIST. Press ENTER, then hold still...")
        closed_vals = capture_average(camera, tracker, tracker_cfg, mapper,
                                      args.seconds, show_window, "CLOSED fist")
        if closed_vals is None:
            print("Could not capture closed pose (no hand detected).")
            return 1
        print("  closed raw:", {k: round(v, 3) for k, v in closed_vals.items()})

        # Build calibration (open -> 0.0, closed -> 1.0).
        ranges = {f: (open_vals[f], closed_vals[f]) for f in FINGERS}
        calib = Calibration(ranges=ranges, source="recorded")

        print("\n=== Validation ===")
        weak = []
        for f in FINGERS:
            sep = abs(closed_vals[f] - open_vals[f])
            flag = "OK" if sep >= min_sep else "WEAK"
            if sep < min_sep:
                weak.append(f)
            print(f"  {f:<7} open={open_vals[f]:.3f} closed={closed_vals[f]:.3f} "
                  f"sep={sep:.3f} [{flag}]")
        if weak:
            print(f"\nWARNING: weak separation for {weak}. The robot may not move these")
            print("fingers fully. Re-run with clearer open/closed poses and good lighting.")
            ans = input("Save anyway? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("Not saved.")
                return 0

        cal_file = cal_cfg.get("file", "calibration.yaml")
        out_path = config_relative_path(args.config, cal_file)
        calib.save(out_path)
        print(f"\nSaved calibration to: {out_path}")
        print("Run the demo:  python main.py --config config.yaml --debug")
    finally:
        camera.release()
        tracker.close()
        if show_window:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
