#!/usr/bin/env python3
"""Send safe test poses to the robotic hand to verify the serial protocol.

Poses:
  open    -> all fingers 0.0
  closed  -> all fingers 1.0   (asks for confirmation unless --yes)
  half    -> all fingers 0.5
  random  -> safe random values (asks for confirmation unless --yes)

All values are clamped to [0, 1] before formatting; nothing unsafe is sent.

Usage:
  python tools/test_serial_hand.py --pose open
  python tools/test_serial_hand.py --pose closed --yes
  python tools/test_serial_hand.py --pose random --port /dev/ttyACM0
  python tools/test_serial_hand.py --pose half --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_controller import HandConfig, create_controller  # noqa: E402
from src.utils import FINGERS, clamp, load_config, setup_logging  # noqa: E402

# Deterministic "random" sequence so the tool needs no Date/random at import.
_PSEUDO = [0.2, 0.8, 0.5, 0.1, 0.9, 0.35, 0.65, 0.45]


def build_pose(name: str, step: int = 0) -> dict:
    if name == "open":
        return {f: 0.0 for f in FINGERS}
    if name == "closed":
        return {f: 1.0 for f in FINGERS}
    if name == "half":
        return {f: 0.5 for f in FINGERS}
    if name == "random":
        return {f: clamp(_PSEUDO[(step + i) % len(_PSEUDO)], 0.0, 1.0)
                for i, f in enumerate(FINGERS)}
    raise ValueError(f"unknown pose {name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Send safe test poses to the robotic hand.")
    ap.add_argument("--pose", required=True, choices=["open", "closed", "half", "random"])
    ap.add_argument("--config", default=None, help="Optional config.yaml for hand settings")
    ap.add_argument("--port", default=None, help="Override hand.port (e.g. /dev/ttyACM0 or auto)")
    ap.add_argument("--dry-run", action="store_true", help="Use mock controller (no hardware)")
    ap.add_argument("--repeat", type=int, default=1, help="Send the pose N times")
    ap.add_argument("--interval", type=float, default=0.5, help="Seconds between repeats")
    ap.add_argument("--yes", action="store_true", help="Skip confirmation for closed/random")
    args = ap.parse_args()

    setup_logging("INFO")

    if args.config and os.path.isfile(args.config):
        hand_dict = load_config(args.config).get("hand", {})
    else:
        hand_dict = {}
    hand_cfg = HandConfig.from_dict(hand_dict)
    # This tool specifically exercises the SERIAL protocol, regardless of which
    # backend config.yaml selects for the demo (default is now "sdk"). --dry-run
    # still forces the mock controller via create_controller(force_mock=...).
    hand_cfg.controller = "serial"
    if args.port:
        hand_cfg.port = args.port

    if args.pose in ("closed", "random") and not args.yes and not args.dry_run:
        ans = input(f"Pose '{args.pose}' will move the hand. Continue? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 0

    controller = create_controller(hand_cfg, force_mock=args.dry_run)
    controller.connect()

    try:
        for n in range(args.repeat):
            pose = build_pose(args.pose, step=n)
            ok = controller.send(pose, force=True)
            print(f"[{n+1}/{args.repeat}] sent {args.pose}: "
                  + ", ".join(f"{k}={v:.2f}" for k, v in pose.items())
                  + (" (ok)" if ok else " (NOT SENT - port down?)"))
            if n < args.repeat - 1:
                time.sleep(args.interval)
        # Always leave the hand safe.
        time.sleep(0.1)
        controller.send_rest()
        print("Sent rest pose. Done.")
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
