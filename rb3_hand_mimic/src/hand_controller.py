"""Robotic hand controller abstraction.

The exact VNG x Paxini protocol is unknown at build time, so the controller is
fully abstracted. Every backend implements one method -- `_apply(values)` --
that takes normalized per-finger curls (0..1) and drives the hardware however
it likes (serial bytes, a vendor SDK call, or nothing). Rate-limiting lives in
the base class, so a backend only worries about delivery.

Controllers:
  * BaseHandController   -- interface (send / send_rest / rate-limit).
  * MockHandController   -- logs commands; used for --dry-run / no hardware.
  * SerialHandController -- pyserial, port auto-detect, reconnect w/ backoff,
                            low-timeout non-blocking writes.
  * SdkHandController    -- drives the robotic hand through a vendor SDK
                            (HandSdk). The real VNG x Paxini SDK is currently
                            x86-only and not yet ported to the RB3 (aarch64),
                            so the default backing is PlaceholderHandSdk: it
                            validates and logs commands and lets the whole async
                            pipeline run on any architecture today. Swap in the
                            real binding by implementing HandSdk.
"""

from __future__ import annotations

import glob
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    sdk_library: str = "placeholder"   # which HandSdk binding to load (sdk backend)
    rest_pose: Dict[str, float] = field(
        default_factory=lambda: {f: 0.0 for f in FINGERS}
    )

    @classmethod
    def from_dict(cls, d: Dict) -> "HandConfig":
        rest = d.get("rest_pose", {}) or {}
        sdk = d.get("sdk", {}) or {}
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
            sdk_library=str(sdk.get("library", "placeholder")),
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
    def _apply(self, values: Dict[str, float]) -> bool:
        """Deliver one command to the hardware. Return True on success, False on
        (recoverable) failure. `values` are normalized curls keyed by finger."""

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    def send(self, values: Dict[str, float], force: bool = False) -> bool:
        """Rate-limited send. `force` bypasses the rate limiter (rest poses)."""
        if not force and not self._limiter.ready():
            return False
        ok = self._apply(values)
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

    def _apply(self, values: Dict[str, float]) -> bool:
        packet = format_command(values, self.finger_order, self.cfg)
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
        self._serial: Optional[Any] = None   # pyserial Serial; Any avoids a hard dep
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

    def _apply(self, values: Dict[str, float]) -> bool:
        return self._write(format_command(values, self.finger_order, self.cfg))

    def _write(self, packet: bytes) -> bool:
        # Never block the real-time loop: reconnection is attempted opportun-
        # istically and write failures degrade gracefully to a retry schedule.
        if not self.is_connected():
            self._maybe_reconnect()
            if not self.is_connected():
                return False
        serial_obj = self._serial
        if serial_obj is None:
            return False
        try:
            serial_obj.write(packet)
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
# Vendor SDK backend (ARM64 target)
# -----------------------------------------------------------------------------
class HandSdk(ABC):
    """Interface the robotic-hand vendor SDK must satisfy.

    This is the seam between our pipeline and the VNG x Paxini hand SDK. The
    real SDK is currently distributed as an x86 build and does NOT run on the
    Qualcomm RB3 Gen 2 (aarch64). When an ARM64 build (or a ctypes/cffi binding,
    or a gRPC shim to a co-located x86 helper) becomes available, implement this
    interface and hand it to SdkHandController -- nothing else in the pipeline
    changes.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Open the device. Return True on success."""

    @abstractmethod
    def set_curls(self, curls: Dict[str, float]) -> bool:
        """Command finger curls (finger name -> 0..1). Return True on success."""

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...


class PlaceholderHandSdk(HandSdk):
    """No-hardware stand-in for the real ARM64 hand SDK.

    Lets the full async detection->control pipeline run on any architecture
    today (x86 dev box or the RB3) before the ARM64 SDK exists. It validates the
    finger values and logs them at a throttled rate; it never touches hardware.

    TODO(vng-paxini-arm64): replace with the real SDK binding. Each method below
    maps 1:1 to a call the production SDK is expected to expose.
    """

    def __init__(self, finger_order: Optional[List[str]] = None, log_every: int = 30) -> None:
        self.finger_order = finger_order or list(FINGERS)
        self._connected = False
        self._count = 0
        self._log_every = max(1, log_every)

    def connect(self) -> bool:
        # TODO: sdk.open() / device handshake on the ARM64 build.
        self._connected = True
        log.info(
            "PlaceholderHandSdk connected (NO HARDWARE). Real ARM64 SDK pending; "
            "commands are logged only. finger_order=%s", self.finger_order,
        )
        return True

    def set_curls(self, curls: Dict[str, float]) -> bool:
        # TODO: translate normalized curls -> SDK joint/servo targets and push.
        if not self._connected:
            return False
        self._count += 1
        if self._count % self._log_every == 0:
            ordered = ", ".join(f"{f}:{clamp(curls.get(f, 0.0), 0.0, 1.0):.2f}"
                                for f in self.finger_order)
            log.debug("SDK(placeholder) <- %s", ordered)
        return True

    def is_connected(self) -> bool:
        return self._connected

    def disconnect(self) -> None:
        # TODO: sdk.close() on the ARM64 build.
        self._connected = False
        log.info("PlaceholderHandSdk disconnected (sent %d commands)", self._count)


def load_hand_sdk(cfg: HandConfig, finger_order: Optional[List[str]] = None) -> HandSdk:
    """Resolve the HandSdk implementation to use.

    Today this always returns the placeholder. When the ARM64 SDK ships, branch
    on cfg.sdk_library here (import the binding, construct it) and fall back to
    the placeholder if the import fails -- so the demo still runs anywhere.
    """
    lib = (cfg.sdk_library or "placeholder").lower()
    if lib in ("placeholder", "", "mock", "none"):
        return PlaceholderHandSdk(finger_order)
    # TODO(vng-paxini-arm64): import and construct the real binding here, e.g.
    #   from paxini_arm64 import PaxiniHand
    #   return PaxiniHandSdkAdapter(PaxiniHand(...))
    log.warning(
        "hand.sdk.library=%r requested but no ARM64 binding is wired in yet; "
        "falling back to PlaceholderHandSdk.", cfg.sdk_library,
    )
    return PlaceholderHandSdk(finger_order)


class SdkHandController(BaseHandController):
    """Controller that drives the robotic hand through a HandSdk implementation."""

    def __init__(
        self,
        cfg: HandConfig,
        finger_order: Optional[List[str]] = None,
        sdk: Optional[HandSdk] = None,
    ) -> None:
        super().__init__(cfg, finger_order)
        self._sdk = sdk or load_hand_sdk(cfg, self.finger_order)

    def connect(self) -> None:
        self._sdk.connect()

    def _apply(self, values: Dict[str, float]) -> bool:
        return self._sdk.set_curls(values)

    def is_connected(self) -> bool:
        return self._sdk.is_connected()

    def close(self) -> None:
        self._sdk.disconnect()


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
    backend = cfg.controller.lower()
    if backend == "serial":
        return SerialHandController(cfg, finger_order)
    if backend == "sdk":
        return SdkHandController(cfg, finger_order)
    raise ValueError(
        f"Unknown hand.controller '{cfg.controller}'. "
        "Supported: 'sdk', 'serial', 'mock'."
    )
