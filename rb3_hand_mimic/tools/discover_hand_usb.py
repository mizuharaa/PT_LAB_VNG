#!/usr/bin/env python3
"""Discover the USB connection for the VNG x Paxini robotic hand.

Lists USB devices (lsusb), stable by-id serial symlinks, and ttyACM*/ttyUSB*
ports. Optionally watches for a NEW port appearing so you can identify the hand
by unplugging and replugging it.

Usage:
  python tools/discover_hand_usb.py
  python tools/discover_hand_usb.py --watch
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_controller import list_serial_ports  # noqa: E402
from src.utils import setup_logging  # noqa: E402


def _run(cmd: list) -> str:
    if shutil.which(cmd[0]) is None:
        return f"({cmd[0]} not available)"
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return f"({cmd[0]} failed: {exc})"


def snapshot() -> set:
    ports = set(glob.glob("/dev/ttyACM*")) | set(glob.glob("/dev/ttyUSB*"))
    ports |= set(glob.glob("/dev/serial/by-id/*"))
    return ports


def main() -> int:
    ap = argparse.ArgumentParser(description="Discover the robotic hand USB/serial port.")
    ap.add_argument("--watch", action="store_true",
                    help="Watch for a new port (unplug/replug the hand to identify it)")
    ap.add_argument("--watch-seconds", type=float, default=30.0)
    args = ap.parse_args()

    setup_logging("INFO")

    print("=== lsusb ===")
    print(_run(["lsusb"]))

    print("\n=== /dev/serial/by-id/ ===")
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    print("\n".join(by_id) if by_id else "(none)")

    print("\n=== /dev/ttyACM* and /dev/ttyUSB* ===")
    tty = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
    print("\n".join(tty) if tty else "(none)")

    print("\n=== Candidate ports (preferred order) ===")
    cands = list_serial_ports()
    if cands:
        for c in cands:
            print(f"  {c}")
        print(f"\nLikely hand port: {cands[0]}")
        print("Set hand.port in config.yaml (or keep 'auto').")
    else:
        print("  (none found)")

    if args.watch:
        print(f"\n=== Watching for a NEW port for {args.watch_seconds:.0f}s ===")
        print("UNPLUG the robotic hand now, wait ~2s, then REPLUG it...")
        before = snapshot()
        deadline = time.perf_counter() + args.watch_seconds
        found = None
        while time.perf_counter() < deadline:
            now = snapshot()
            new = now - before
            if new:
                found = sorted(new)
                break
            before |= now  # account for ports that disappear on unplug
            time.sleep(0.3)
        if found:
            print("\nNEW port(s) detected (this is almost certainly the hand):")
            for f in found:
                print(f"  {f}")
        else:
            print("\nNo new port detected within the time window.")
            print("Try `dmesg -w` in another terminal while replugging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
