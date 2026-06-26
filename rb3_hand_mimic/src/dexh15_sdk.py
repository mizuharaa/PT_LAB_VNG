"""Placeholder for the Paxini DexH15 (GMH15) dexterous-hand SDK / protocol.

Aligned to the vendor docs (DexHandSDK V3.0.0 "API Reference" + "Getting Started"):

  * The real SDK is **C++ only** (DexHandSDK-3.0.0-Linux.deb, namespace
    paxini::bot::dexh15) and requires an **x86 / x86_64** CPU on Ubuntu 22.04 --
    so it does NOT run natively on the RB3 (aarch64) and has no Python binding.
  * The hand itself is a serial device speaking **Modbus-RTU** (RS-485 / USB-485,
    e.g. /dev/ttyUSB0, slave 0x79, baud up to 4,000,000) or **USB 2.0 / USB-CDC**
    (/dev/ttyACM0). Because that is just serial I/O, a pure-Python driver runs on
    *any* architecture -- including the RB3 -- which is why driving the hand at
    the protocol level (the route the other engineer took) is the path that lets
    everything run on the RB3 with no x86 VM.

Two command interfaces the hand exposes (this module supports both):

  * "motor"  -> 7 raw motor positions (setMotorTargetPosition / Dex15MotorPosition).
                These are the actual registers; simplest to drive over the raw
                protocol from Python, and they map almost 1:1 from our 5 curls.
                DEFAULT.
  * "joint"  -> 11 normalized joint angles (setJointPositionsAngle). A convenience
                layer that runs joint->motor inverse kinematics *inside* the SDK,
                so a protocol-level Python driver would have to re-derive that.

This file still touches no hardware. Replace DexH15Bridge's TODO bodies with the
engineer's pyserial/Modbus implementation (or a binding to the C++ SDK) and the
rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .hand_controller import HandSdk
from .utils import FINGERS, clamp, get_logger

log = get_logger("dexh15")

# -----------------------------------------------------------------------------
# Value ranges (from the API Reference)
# -----------------------------------------------------------------------------
# Normalized joint angles: [0, 1000], EXCEPT thumb_joint1 (lateral swing) is
# centered at 0 with range [-500, 500].
ANGLE_MIN = 0
ANGLE_MAX = 1000
THUMB_SWING_MIN = -500
THUMB_SWING_MAX = 500
# Raw motor positions span [0, 10000] (forwardKinematicParse). 0 == open/zero
# pose; larger == more flexed. We scale to `motor_full` (<= 10000) so we never
# slam to an unverified hardware maximum.
MOTOR_MIN = 0
MOTOR_MAX = 10000

# -----------------------------------------------------------------------------
# Joint layout -- 11 normalized joints, exact order from setJointPositionsAngle.
# index(2) -> middle(2) -> ring(2) -> little(2) -> thumb(3)
# -----------------------------------------------------------------------------
JOINT_NAMES: List[str] = [
    "index_joint1",   # 0  proximal flex   [0,1000]
    "index_joint2",   # 1  end flex        [0,1000]
    "middle_joint1",  # 2                  [0,1000]
    "middle_joint2",  # 3                  [0,1000]
    "ring_joint1",    # 4                  [0,1000]
    "ring_joint2",    # 5                  [0,1000]
    "little_joint1",  # 6                  [0,1000]
    "little_joint2",  # 7                  [0,1000]
    "thumb_joint1",   # 8  lateral swing   [-500,500]
    "thumb_joint2",   # 9  proximal flex   [0,1000]
    "thumb_joint3",   # 10 end flex        [0,1000]
]
NUM_JOINTS = len(JOINT_NAMES)

# -----------------------------------------------------------------------------
# Motor layout -- 7 motors, order from Dex15MotorPosition (Getting Started 4.1.2)
# -----------------------------------------------------------------------------
MOTOR_NAMES: List[str] = [
    "index_flex",        # 1 index-finger flexion
    "middle_flex",       # 2 middle-finger flexion
    "ring_flex",         # 3 ring-finger flexion
    "pinky_flex",        # 4 little-finger flexion
    "four_finger_base",  # 5 common base of the four fingers (spread/abduction)
    "thumb_lower",       # 6 thumb lower joint
    "thumb_upper",       # 7 thumb upper joint
]
NUM_MOTORS = len(MOTOR_NAMES)


def map_curls_to_motors(curls: Dict[str, float], motor_full: int = 8000) -> List[int]:
    """Map 5 normalized finger curls (0..1) -> 7 raw motor positions.

    PLACEHOLDER kinematics. `motor_full` is the position commanded at curl == 1
    (kept below the 10000 ceiling until per-unit limits are confirmed). Motor 5
    (four-finger base / spread) has no single-camera estimate, so it stays at
    its neutral 0; wire it up once finger-abduction is estimated or the 2nd
    camera is in. Returns NUM_MOTORS ints in MOTOR order.
    """
    def pos(name: str) -> int:
        v = clamp(float(curls.get(name, 0.0)), 0.0, 1.0)
        return int(max(MOTOR_MIN, min(MOTOR_MAX, round(v * motor_full))))

    return [
        pos("index"),
        pos("middle"),
        pos("ring"),
        pos("pinky"),
        0,                 # four_finger_base: neutral (TODO: finger spread)
        pos("thumb"),      # thumb_lower
        pos("thumb"),      # thumb_upper
    ]


def map_curls_to_joints(curls: Dict[str, float]) -> List[int]:
    """Map 5 normalized finger curls (0..1) -> 11 normalized joint angles.

    PLACEHOLDER. Each finger's curl drives both of its joints; thumb curl drives
    thumb_joint2/3 (flexion). thumb_joint1 (lateral swing, [-500,500]) has no
    single-camera estimate, so it stays neutral at 0 (TODO: thumb adduction).
    Returns NUM_JOINTS ints, matching JOINT_NAMES / SDK joint order.
    """
    def ang(name: str, scale: float = 1.0) -> int:
        v = clamp(float(curls.get(name, 0.0)), 0.0, 1.0)
        return int(max(ANGLE_MIN, min(ANGLE_MAX, round(v * scale * ANGLE_MAX))))

    return [
        ang("index"), ang("index", 0.95),
        ang("middle"), ang("middle", 0.95),
        ang("ring"), ang("ring", 0.95),
        ang("pinky"), ang("pinky", 0.95),
        0,                                  # thumb_joint1: neutral swing [-500,500]
        ang("thumb"), ang("thumb", 0.9),    # thumb_joint2, thumb_joint3
    ]


# -----------------------------------------------------------------------------
# Transport bridge: RB3 (this app) <-> the hand (or an x86-VM SDK relay)
# -----------------------------------------------------------------------------
@dataclass
class DexH15Config:
    """Connection + command settings for the DexH15 hand."""

    host: str = "127.0.0.1"     # x86-VM relay address (only if using the C++ SDK)
    port: int = 50515
    connect_timeout: float = 1.0
    interface: str = "motor"    # "motor" (7 positions) | "joint" (11 angles)
    slave_address: int = 0x79   # Modbus slave id of the hand
    motor_full: int = 8000      # motor position commanded at curl == 1

    @classmethod
    def from_dict(cls, d: Dict) -> "DexH15Config":
        d = d or {}
        return cls(
            host=str(d.get("host", "127.0.0.1")),
            port=int(d.get("port", 50515)),
            connect_timeout=float(d.get("connect_timeout", 1.0)),
            interface=str(d.get("interface", "motor")).lower(),
            slave_address=int(d.get("slave_address", 0x79)),
            motor_full=int(d.get("motor_full", 8000)),
        )


class DexH15Bridge:
    """Placeholder transport to the hand. Validates + logs; opens no port yet.

    Proposed wire format (newline-delimited JSON over TCP/serial), one command
    per line, mirroring the SDK calls a real driver makes:

        {"cmd":"setMotorTargetPosition","slave":121,"pos":[<7 ints 0..10000>]}
        {"cmd":"setJointPositionsAngle","slave":121,"angles":[<11 ints>]}

    Real init handshake the driver MUST perform before streaming commands
    (Getting Started 3.2.1) -- otherwise the hand will not move:
        open serial -> initModbusDevice(slave) -> initMotorPosition(slave)
        -> setMotorControlMode(slave, POSITION) -> enableMotor(slave)
        ... stream commands ...
        -> disableMotor(slave) -> close serial
    """

    def __init__(self, cfg: DexH15Config) -> None:
        self.cfg = cfg
        self._connected = False
        self._count = 0
        self._log_every = 30

    def connect(self) -> bool:
        # TODO(dexh15): open the serial/Modbus port (or socket to the x86-VM
        # relay) and run the init handshake described above.
        self._connected = True
        log.info(
            "DexH15Bridge placeholder connected (NO TRANSPORT). interface=%s "
            "slave=0x%02X. Commands validated + logged only.",
            self.cfg.interface, self.cfg.slave_address,
        )
        return True

    def is_connected(self) -> bool:
        return self._connected

    def _send(self, message: Dict) -> bool:
        if not self._connected:
            return False
        frame = json.dumps(message, separators=(",", ":")) + "\n"
        # TODO(dexh15): write `frame` (or the equivalent Modbus register writes).
        self._count += 1
        if self._count % self._log_every == 0:
            log.debug("DexH15 -> %s", frame.rstrip("\n"))
        return True

    def setMotorTargetPosition(self, positions: List[int]) -> bool:  # noqa: N802 (SDK name)
        """Command all 7 motor positions (raw, [0..10000])."""
        if len(positions) != NUM_MOTORS:
            log.error("expected %d motor positions, got %d", NUM_MOTORS, len(positions))
            return False
        clamped = [int(max(MOTOR_MIN, min(MOTOR_MAX, p))) for p in positions]
        return self._send({"cmd": "setMotorTargetPosition",
                           "slave": self.cfg.slave_address, "pos": clamped})

    def setJointPositionsAngle(self, angles: List[int]) -> bool:  # noqa: N802 (SDK name)
        """Command all 11 normalized joint angles."""
        if len(angles) != NUM_JOINTS:
            log.error("expected %d joint angles, got %d", NUM_JOINTS, len(angles))
            return False
        return self._send({"cmd": "setJointPositionsAngle",
                           "slave": self.cfg.slave_address, "angles": list(angles)})

    def disconnect(self) -> None:
        # TODO(dexh15): disableMotor + close the port.
        self._connected = False
        log.info("DexH15Bridge placeholder disconnected (sent %d commands)", self._count)


# -----------------------------------------------------------------------------
# HandSdk adapter -- plugs DexH15 into the project's controller abstraction
# -----------------------------------------------------------------------------
class DexH15HandSdk(HandSdk):
    """Drives the DexH15 hand from 5 finger curls via the configured interface
    (7 motor positions by default, or 11 joint angles)."""

    def __init__(self, cfg: Optional[DexH15Config] = None,
                 finger_order: Optional[List[str]] = None) -> None:
        self.cfg = cfg or DexH15Config()
        self.finger_order = finger_order or list(FINGERS)
        self._bridge = DexH15Bridge(self.cfg)

    def connect(self) -> bool:
        return self._bridge.connect()

    def set_curls(self, curls: Dict[str, float]) -> bool:
        if self.cfg.interface == "joint":
            return self._bridge.setJointPositionsAngle(map_curls_to_joints(curls))
        return self._bridge.setMotorTargetPosition(
            map_curls_to_motors(curls, self.cfg.motor_full))

    def is_connected(self) -> bool:
        return self._bridge.is_connected()

    def disconnect(self) -> None:
        self._bridge.disconnect()
