"""Robotic hand controller abstraction.

The exact VNG x Paxini protocol is unknown at build time, so the controller is
fully abstracted and the serial packet format is configurable (csv /
labeled_csv / json). When the real protocol arrives, either pick the matching
format in config or add a new `_format_*` method / Protocol subclass -- nothing
upstream changes.

Controllers:
  * BaseHandController  -- interface.
  * MockHandController  -- logs commands; used for --dry-run / no hardware.
  * SerialHandController -- pyserial, port auto-detect, reconnect w/ backoff,
                            low-timeout non-blocking writes, command rate limit.
"""

from __future__ import annotations

import glob
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .utils import FINGERS, RateLimiter, clamp, get_logger

log = get_logger("hand")


@dataclass
class HandConfig:
    controller: str = "serial"
    port: str = "auto"
    baudrate: int = 115200
    timeout: float = 0.02
    command_rate_hz: float = 30.0
    protocol: str = "csv"
    newline: bool = True
    value_type: str = "normalized"   # "normalized" | "servo"
    servo_min: int = 0
    servo_max: int = 180
    decimals: int = 3
    rest_pose: Dict[str, float] = field(
        default_factory=lambda: {f: 0.0 for f in FINGERS}
    )

    @classmethod
    def from_dict(cls, d: Dict) -> "HandConfig":
        rest = d.get("rest_pose", {}) or {}
        return cls(
            controller=str(d.get("controller", "serial")),
            port=str(d.get("port", "auto")),
            baudrate=int(d.get("baudrate", 115200)),
            timeout=float(d.get("timeout", 0.02)),
            command_rate_hz=float(d.get("command_rate_hz", 30.0)),
            protocol=str(d.get("protocol", "csv")),
            newline=bool(d.get("newline", True)),
            value_type=str(d.get("value_type", "normalized")),
            servo_min=int(d.get("servo_min", 0)),
            servo_max=int(d.get("servo_max", 180)),
            decimals=int(d.get("decimals", 3)),
            rest_pose={f: float(rest.get(f, 0.0)) for f in FINGERS},
        )


# -----------------------------------------------------------------------------
# Packet formatting (shared by all controllers)
# -----------------------------------------------------------------------------
def _to_servo(value: float, lo: int, hi: int) -> int:
    return int(round(lo + clamp(value, 0.0, 1.0) * (hi - lo)))


def format_command(values: Dict[str, float], order: List[str], cfg: HandConfig) -> bytes:
    """Serialize normalized finger values into the configured wire format.

    `values` is keyed by finger name; `order` defines hardware finger order.
    """
    ordered = [clamp(values[f], 0.0, 1.0) for f in order]

    if cfg.value_type == "servo":
        nums = [_to_servo(v, cfg.servo_min, cfg.servo_max) for v in ordered]
        str_vals = [str(n) for n in nums]
    else:  # normalized
        str_vals = [f"{v:.{cfg.decimals}f}" for v in ordered]

    if cfg.protocol == "json":
        payload = json.dumps({f: (nums[i] if cfg.value_type == "servo" else float(str_vals[i]))
                              for i, f in enumerate(order)})
    elif cfg.protocol == "labeled_csv":
        payload = "FINGER," + ",".join(str_vals)
    else:  # csv (default)
        payload = ",".join(str_vals)

    if cfg.newline:
        payload += "\n"
    return payload.encode("utf-8")


# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------
class BaseHandController(ABC):
    def __init__(self, cfg: HandConfig, finger_order: Optional[List[str]] = None) -> None:
        self.cfg = cfg
        self.finger_order = finger_order or list(FINGERS)
        self._limiter = RateLimiter(cfg.command_rate_hz)
        self._last_sent: Optional[Dict[str, float]] = None

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def _write(self, packet: bytes) -> bool:
        """Return True on success, False on (recoverable) failure."""

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    def send(self, values: Dict[str, float], force: bool = False) -> bool:
        """Rate-limited send. `force` bypasses the rate limiter (rest poses)."""
        if not force and not self._limiter.ready():
            return False
        packet = format_command(values, self.finger_order, self.cfg)
        ok = self._write(packet)
        if ok:
            self._last_sent = dict(values)
        return ok

    def send_rest(self) -> bool:
        """Send the configured rest/open pose (always forced)."""
        return self.send(dict(self.cfg.rest_pose), force=True)


# -----------------------------------------------------------------------------
# Mock
# -----------------------------------------------------------------------------
class MockHandController(BaseHandController):
    """Dry-run controller: formats and logs commands, no hardware."""

    def __init__(self, cfg: HandConfig, finger_order: Optional[List[str]] = None) -> None:
        super().__init__(cfg, finger_order)
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        log.info("MockHandController active (dry-run, protocol=%s)", self.cfg.protocol)

    def _write(self, packet: bytes) -> bool:
        log.debug("MOCK -> %s", packet.decode("utf-8").rstrip("\n"))
        return True

    def is_connected(self) -> bool:
        return self._connected

    def close(self) -> None:
        self._connected = False
        log.info("MockHandController closed")


# -----------------------------------------------------------------------------
# Serial
# -----------------------------------------------------------------------------
def list_serial_ports() -> List[str]:
    """Return candidate serial ports, preferring stable by-id symlinks."""
    candidates: List[str] = []
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    candidates.extend(by_id)
    candidates.extend(sorted(glob.glob("/dev/ttyACM*")))
    candidates.extend(sorted(glob.glob("/dev/ttyUSB*")))
    # On non-Linux dev machines, fall back to pyserial enumeration.
    if not candidates:
        try:
            from serial.tools import list_ports
            candidates = [p.device for p in list_ports.comports()]
        except Exception:  # noqa: BLE001
            pass
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def auto_detect_port() -> Optional[str]:
    ports = list_serial_ports()
    if ports:
        log.info("auto-detected serial port: %s (candidates: %s)", ports[0], ports)
        return ports[0]
    log.warning("no serial ports found under /dev/serial/by-id, ttyACM*, ttyUSB*")
    return None


class SerialHandController(BaseHandController):
    """pyserial-backed controller with reconnect/backoff and safe writes."""

    def __init__(self, cfg: HandConfig, finger_order: Optional[List[str]] = None) -> None:
        super().__init__(cfg, finger_order)
        self._serial = None
        self._port: Optional[str] = None
        self._backoff = 0.5
        self._backoff_max = 5.0
        self._next_retry = 0.0
        try:
            import serial  # noqa: WPS433
            self._serial_mod = serial
        except ImportError as exc:
            raise SystemExit(
                "pyserial is required for the serial controller.\n"
                "Install with: pip install pyserial\n"
                "Or run with --dry-run to use the mock controller.\n"
                f"(import error: {exc})"
            )

    def _resolve_port(self) -> Optional[str]:
        if self.cfg.port and self.cfg.port.lower() != "auto":
            return self.cfg.port
        return auto_detect_port()

    def connect(self) -> None:
        port = self._resolve_port()
        if port is None:
            log.warning("no serial port available; will retry with backoff")
            self._schedule_retry()
            return
        try:
            self._serial = self._serial_mod.Serial(
                port=port,
                baudrate=self.cfg.baudrate,
                timeout=self.cfg.timeout,
                write_timeout=self.cfg.timeout,
            )
            self._port = port
            self._backoff = 0.5
            log.info("connected to robotic hand on %s @ %d baud", port, self.cfg.baudrate)
        except Exception as exc:  # noqa: BLE001 - serial raises many subtypes
            log.warning("failed to open serial port %s: %s", port, exc)
            self._serial = None
            self._schedule_retry()

    def _schedule_retry(self) -> None:
        self._next_retry = time.perf_counter() + self._backoff
        self._backoff = min(self._backoff * 2.0, self._backoff_max)

    def _maybe_reconnect(self) -> None:
        if self._serial is not None:
            return
        if time.perf_counter() >= self._next_retry:
            log.info("attempting serial reconnect...")
            self.connect()

    def is_connected(self) -> bool:
        return self._serial is not None and getattr(self._serial, "is_open", False)

    def _write(self, packet: bytes) -> bool:
        # Never block the real-time loop: reconnection is attempted opportun-
        # istically and write failures degrade gracefully to a retry schedule.
        if not self.is_connected():
            self._maybe_reconnect()
            if not self.is_connected():
                return False
        try:
            self._serial.write(packet)
            return True
        except Exception as exc:  # noqa: BLE001 - SerialException, write timeout, etc.
            log.warning("serial write failed (%s); dropping connection", exc)
            self._safe_close_port()
            self._schedule_retry()
            return False

    def _safe_close_port(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # noqa: BLE001
                pass
        self._serial = None

    def close(self) -> None:
        self._safe_close_port()
        log.info("serial controller closed")


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------
def create_controller(
    cfg: HandConfig,
    finger_order: Optional[List[str]] = None,
    force_mock: bool = False,
) -> BaseHandController:
    """Instantiate the configured controller. `force_mock` honors --dry-run."""
    if force_mock or cfg.controller.lower() == "mock":
        return MockHandController(cfg, finger_order)
    if cfg.controller.lower() == "serial":
        return SerialHandController(cfg, finger_order)
    raise ValueError(
        f"Unknown hand.controller '{cfg.controller}'. Supported: 'serial', 'mock'."
    )
