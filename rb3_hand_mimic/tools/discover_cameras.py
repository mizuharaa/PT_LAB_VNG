#!/usr/bin/env python3
"""Enumerate and probe cameras to identify the RB3 tracking camera.

Tries OpenCV indexes 0-8, reports open/read status, resolution, estimated FPS,
and the v4l2 device name, and saves one debug frame per working camera to
debug_frames/ so you can visually tell the tracking camera from the main lens.

Usage:
  python tools/discover_cameras.py
  python tools/discover_cameras.py --config config.yaml --max-index 8
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import (  # noqa: E402
    CameraConfig,
    discover_cameras,
    list_video_devices,
    select_camera_index,
    v4l2_list_devices,
)
from src.utils import load_config, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Discover and probe cameras.")
    ap.add_argument("--config", default=None, help="Optional config.yaml for camera settings")
    ap.add_argument("--max-index", type=int, default=8)
    ap.add_argument("--out-dir", default="debug_frames")
    args = ap.parse_args()

    setup_logging("INFO")

    if args.config and os.path.isfile(args.config):
        cam_cfg = CameraConfig.from_dict(load_config(args.config).get("camera", {}))
    else:
        cam_cfg = CameraConfig()

    print("=== /dev/video* devices ===")
    devs = list_video_devices()
    print("\n".join(devs) if devs else "(none or non-Linux platform)")

    print("\n=== v4l2-ctl --list-devices ===")
    names = v4l2_list_devices()
    if names:
        for idx, name in sorted(names.items()):
            print(f"  /dev/video{idx}: {name}")
    else:
        print("  (v4l2-ctl not available or no devices)")

    print(f"\n=== Probing camera indexes 0..{args.max_index} ===")
    infos = discover_cameras(cam_cfg, max_index=args.max_index, save_debug_dir=args.out_dir)

    print(f"\n{'idx':>3}  {'reads':>5}  {'resolution':>11}  {'fps':>5}  name/note")
    print("-" * 60)
    for i in infos:
        res = f"{i.width}x{i.height}" if i.reads else "-"
        print(f"{i.index:>3}  {('yes' if i.reads else 'no'):>5}  "
              f"{res:>11}  {i.fps:>5}  {i.name or i.note}")

    chosen = select_camera_index(cam_cfg, infos)
    print("\n=== Recommendation ===")
    if chosen is not None:
        print(f"Auto-selected camera index: {chosen} (preferred='{cam_cfg.preferred}')")
        print(f"Inspect the saved frames in '{args.out_dir}/' to confirm this is the")
        print("smaller TRACKING camera (not the big main lens). If wrong, set")
        print("camera.index explicitly in config.yaml or use --camera-index.")
    else:
        print("No working camera found. Check cabling, permissions (video group),")
        print("and that no other process holds the device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
