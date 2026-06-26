"""Placeholder for the Paxini DexH15 hand SDK protocol.

Target deployment topology
--------------------------
    Camera (OV9282) ─► MediaPipe (this app, RB3 / ARM64, native)
        ─► map landmarks → 11 normalized joint angles [0..1000]
            ─► **DexH15 C++ SDK (runs inside an x86 VM)**
                ─► sdk.setJointPositionsAngle(angles)
                    ─► robotic hand moves

The DexH15 SDK is a C++ library that only runs on x86, so it cannot be loaded
in-process on the RB3 (aarch64). The integration therefore crosses a machine
boundary: this app computes the 11 joint angles on the RB3 and ships them to a
small bridge process inside the x86 VM, which calls the real
`setJointPositionsAngle()` on the SDK.

This module is a PLACEHOLDER for that boundary. It:
  * defines the 11-joint layout and the [0..1000] angle convention,
  * maps our 5 normalized finger curls → 11 joint angles,
  * models the RB3→VM transport (DexH15Bridge) with a documented wire format,
  * exposes a `DexH15HandSdk` that plugs into the project's `HandSdk` interface.

Nothing here talks to real hardware yet. Everything marked ``TODO(dexh15)`` is a
spot to wire the real SDK / transport once it is available. The exact joint
order, ranges, and direction MUST be confirmed against the DexH15 SDK docs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .hand_controller import HandSdk
from .utils import FINGERS, clamp, get_logger

log = get_logger("dexh15")

# Normalized joint command range expected by setJointPositionsAngle().
ANGLE_MIN = 0
ANGLE_MAX = 1000

# -----------------------------------------------------------------------------
# Joint layout (PLACEHOLDER -- confirm against the real DexH15 SDK)
# -----------------------------------------------------------------------------
# 11 controllable joints. We currently estimate one flexion scalar per finger
# (+ a thumb that also carries adduction), so several joints are driven from the
# same source until richer per-joint estimation (or the 2nd camera) is wired.
# Index in this tuple == joint index passed to setJointPositionsAngle().
JOINT_NAMES: List[str] = [
    "thumb_rotation",  # 0  thumb opposition / CMC rotation
    "thumb_mcp",       # 1  thumb MCP flexion
    "thumb_ip",        # 2  thumb IP flexion
    "index_mcp",       # 3  index MCP flexion
    "index_pip",       # 4  index PIP flexion
    "middle_mcp",      # 5  middle MCP flexion
    "middle_pip",      # 6  middle PIP flexion
    "ring_mcp",        # 7  ring MCP flexion
    "ring_pip",        # 8  ring PIP flexion
    "pinky_mcp",       # 9  pinky MCP flexion
    "pinky_pip",       # 10 pinky PIP flexion
]
NUM_JOINTS = len(JOINT_NAMES)

# How each of the 11 joints is driven from our per-finger curl (0..1). Most
# joints map 1:1 to the owning finger's curl; the thumb's three joints share the
# thumb curl. ``scale`` lets a joint use less than the full travel (e.g. a PIP
# that should not fully close). TODO(dexh15): replace with the real per-joint
# kinematics once joint limits/coupling are known.
@dataclass(frozen=True)
class _JointDrive:
    finger: str        # which finger curl drives this joint
    scale: float = 1.0  # fraction of [0..1000] travel to use


_JOINT_DRIVE: Dict[str, _JointDrive] = {
    "thumb_rotation": _JointDrive("thumb", 0.8),
    "thumb_mcp": _JointDrive("thumb", 1.0),
    "thumb_ip": _JointDrive("thumb", 0.9),
    "index_mcp": _JointDrive("index", 1.0),
    "index_pip": _JointDrive("index", 0.95),
    "middle_mcp": _JointDrive("middle", 1.0),
    "middle_pip": _JointDrive("middle", 0.95),
    "ring_mcp": _JointDrive("ring", 1.0),
    "ring_pip": _JointDrive("ring", 0.95),
    "pinky_mcp": _JointDrive("pinky", 1.0),
    "pinky_pip": _JointDrive("pinky", 0.95),
}


def map_curls_to_joints(curls: Dict[str, float]) -> List[int]:
    """Map 5 normalized finger curls (0..1) → 11 joint angles in [0..1000].

    PLACEHOLDER kinematics: each joint takes its owning finger's curl, scaled and
    quantized to the integer command range. Returns a list of length NUM_JOINTS
    ordered to match JOINT_NAMES / the SDK joint indices.
    """
    out: List[int] = []
    for name in JOINT_NAMES:
        drive = _JOINT_DRIVE[name]
        curl = clamp(float(curls.get(drive.finger, 0.0)), 0.0, 1.0)
        angle = int(round(curl * drive.scale * ANGLE_MAX))
        out.append(max(ANGLE_MIN, min(ANGLE_MAX, angle)))
    return out


# -----------------------------------------------------------------------------
# Transport bridge: RB3 (this app) ─► x86 VM (real DexH15 C++ SDK)
# -----------------------------------------------------------------------------
@dataclass
class DexH15Config:
    """Connection settings for the bridge to the x86 VM hosting the SDK."""

    host: str = "127.0.0.1"   # the x86 VM address reachable from the RB3
    port: int = 50515
    connect_timeout: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict) -> "DexH15Config":
        d = d or {}
        return cls(
            host=str(d.get("host", "127.0.0.1")),
            port=int(d.get("port", 50515)),
            connect_timeout=float(d.get("connect_timeout", 1.0)),
        )


class DexH15Bridge:
    """Placeholder client for the bridge process inside the x86 VM.

    Wire format (proposed, newline-delimited JSON over TCP) -- one object per
    command, mirroring the SDK calls the bridge will make:

        {"cmd": "setJointPositionsAngle", "angles": [<11 ints 0..1000>]}
        {"cmd": "getJointPositionsAngle"}
        {"cmd": "ping"}

    The real bridge unpacks ``angles`` and calls the C++
    ``setJointPositionsAngle()``. Today this class only validates and logs; no
    socket is opened. TODO(dexh15): open a TCP socket to (host, port) and send
    these frames.
    """

    def __init__(self, cfg: DexH15Config) -> None:
        self.cfg = cfg
        self._connected = False
        self._sock = None
        self._count = 0
        self._log_every = 30

    def connect(self) -> bool:
        # TODO(dexh15): open a real socket, e.g.
        #   self._sock = socket.create_connection((cfg.host, cfg.port),
        #                                          timeout=cfg.connect_timeout)
        self._connected = True
        log.info(
            "DexH15Bridge placeholder connected (NO TRANSPORT). Would reach the "
            "x86 VM SDK at %s:%d. Commands are validated + logged only.",
            self.cfg.host, self.cfg.port,
        )
        return True

    def is_connected(self) -> bool:
        return self._connected

    def _send(self, message: Dict) -> bool:
        """Serialize + 'send' one command frame. Placeholder logs it."""
        if not self._connected:
            return False
        frame = json.dumps(message, separators=(",", ":")) + "\n"
        # TODO(dexh15): self._sock.sendall(frame.encode("utf-8"))
        self._count += 1
        if self._count % self._log_every == 0:
            log.debug("DexH15 -> %s", frame.rstrip("\n"))
        return True

    def setJointPositionsAngle(self, angles: List[int]) -> bool:  # noqa: N802 (matches SDK)
        """Command all 11 joints. `angles` are integers in [0..1000]."""
        if len(angles) != NUM_JOINTS:
            log.error("expected %d joint angles, got %d; dropping command",
                      NUM_JOINTS, len(angles))
            return False
        clamped = [max(ANGLE_MIN, min(ANGLE_MAX, int(a))) for a in angles]
        return self._send({"cmd": "setJointPositionsAngle", "angles": clamped})

    def disconnect(self) -> None:
        # TODO(dexh15): close the socket if open.
        self._connected = False
        log.info("DexH15Bridge placeholder disconnected (sent %d commands)", self._count)


# -----------------------------------------------------------------------------
# HandSdk adapter -- plugs DexH15 into the project's controller abstraction
# -----------------------------------------------------------------------------
# hand_controller imports this module only lazily (inside load_hand_sdk), so
# importing HandSdk at module level here does not create a runtime import cycle.
class DexH15HandSdk(HandSdk):
    """Drives the DexH15 hand: 5 finger curls → 11 joints → bridge → x86 SDK."""

    def __init__(self, cfg: Optional[DexH15Config] = None,
                 finger_order: Optional[List[str]] = None) -> None:
        self.cfg = cfg or DexH15Config()
        self.finger_order = finger_order or list(FINGERS)
        self._bridge = DexH15Bridge(self.cfg)

    def connect(self) -> bool:
        return self._bridge.connect()

    def set_curls(self, curls: Dict[str, float]) -> bool:
        angles = map_curls_to_joints(curls)
        return self._bridge.setJointPositionsAngle(angles)

    def is_connected(self) -> bool:
        return self._bridge.is_connected()

    def disconnect(self) -> None:
        self._bridge.disconnect()
