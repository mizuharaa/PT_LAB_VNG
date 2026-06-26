"""In-window debug configuration menu (keyboard-driven OpenCV overlay).

Toggle with ``m``. Navigate with ``w``/``s``, change a value with ``a``/``d``,
and trigger an item's action (e.g. APPLY) with Enter. The menu is purely a view
+ key dispatcher: each MenuItem holds callbacks supplied by the app, so this
module stays decoupled from the pipeline.

Settings split into two kinds:
  * live   -- take effect immediately (mirror, landmarks, fusion confidence...),
  * pending -- camera index / dual-camera changes that need the detection layer
    to be rebuilt; they are marked ``*`` and committed by the APPLY item.

Arrow keys are unreliable across OpenCV/OS builds, so navigation uses w/a/s/d.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

# waitKey codes we treat as "activate" (Enter / e).
_ENTER_KEYS = (13, 10, ord("e"))
_UP_KEYS = (ord("w"),)
_DOWN_KEYS = (ord("s"),)
_LEFT_KEYS = (ord("a"),)
_RIGHT_KEYS = (ord("d"),)
_TOGGLE_KEYS = (ord("m"),)
_ESC = 27


@dataclass
class MenuItem:
    label: str
    value_fn: Callable[[], str]                 # current value as text
    on_change: Callable[[int], None]            # a/d -> delta -1/+1
    on_select: Optional[Callable[[], Optional[str]]] = None  # Enter -> action str
    pending: bool = False                        # needs APPLY to take effect


class DebugMenu:
    """Keyboard-driven overlay menu. Returns action strings to the app."""

    def __init__(self, items: List[MenuItem]) -> None:
        self.items = items
        self.index = 0
        self.visible = False

    def handle_key(self, key: int) -> Optional[str]:
        """Process one waitKey code.

        Returns an action string ("apply_cameras", ...) when an item requests
        it, the sentinel "consumed" when the menu handled the key, or None when
        the key should fall through to app-level handling (quit, etc.).
        """
        if key in (255, -1):  # no key
            return "consumed" if self.visible else None

        if not self.visible:
            if key in _TOGGLE_KEYS:
                self.visible = True
                return "consumed"
            return None

        # Menu is open.
        if key in _TOGGLE_KEYS or key == _ESC:
            self.visible = False
            return "consumed"
        if key in _UP_KEYS:
            self.index = (self.index - 1) % len(self.items)
        elif key in _DOWN_KEYS:
            self.index = (self.index + 1) % len(self.items)
        elif key in _LEFT_KEYS:
            self.items[self.index].on_change(-1)
        elif key in _RIGHT_KEYS:
            self.items[self.index].on_change(+1)
        elif key in _ENTER_KEYS:
            item = self.items[self.index]
            if item.on_select is not None:
                return item.on_select() or "consumed"
        return "consumed"

    def render(self, frame) -> None:
        if not self.visible:
            return
        import cv2

        rows = [it.label for it in self.items]
        title = "== DEBUG MENU =="
        footer = "w/s move   a/d change   Enter apply   m/Esc close"
        # Panel geometry (top-right corner).
        pad = 10
        line_h = 22
        width = 360
        height = pad * 2 + line_h * (len(rows) + 3)
        fh, fw = frame.shape[:2]
        x0 = max(0, fw - width - 12)
        y0 = 12

        # Translucent dark panel.
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (90, 90, 90), 1)

        def text(s, x, y, color=(255, 255, 255), scale=0.5):
            cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

        y = y0 + pad + 16
        text(title, x0 + pad, y, (0, 220, 220), 0.55)
        y += line_h
        for i, it in enumerate(self.items):
            sel = (i == self.index)
            marker = ">" if sel else " "
            star = " *" if it.pending else ""
            color = (0, 220, 0) if sel else (230, 230, 230)
            text(f"{marker} {it.label}: {it.value_fn()}{star}", x0 + pad, y, color)
            y += line_h
        y += line_h
        text(footer, x0 + pad, y, (180, 180, 180), 0.42)
