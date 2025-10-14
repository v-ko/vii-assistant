from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, TypedDict

from pynput import keyboard, mouse


@dataclass(slots=True)
class _WindowGeometry:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def to_relative(self, x: int, y: int) -> tuple[int, int]:
        return x - self.left, y - self.top


@dataclass(slots=True)
class _FocusState:
    """Track whether the pointer is within the recorded window."""

    inside: bool = False
    last_x: int | None = None
    last_y: int | None = None
    last_relative_x: int | None = None
    last_relative_y: int | None = None


class SessionRecorderConfig(TypedDict, total=False):
    window_geometry: tuple[int, int, int, int]
    screen: str
    client_type: str


class SessionRecorder:
    """Record mouse and keyboard events via pynput."""

    def __init__(
        self,
        event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        *,
        config: Optional[SessionRecorderConfig] = None,
    ) -> None:
        self._event_callback = event_callback
        self._keyboard_listener = None
        self._mouse_listener = None
        self._active = False
        self._focus = _FocusState()

        cfg = dict(config) if config else {}
        self._window_geometry = self._coerce_geometry(cfg.get("window_geometry"))

        try:
            self._pointer_controller: Optional[mouse.Controller] = mouse.Controller()
        except Exception:
            self._pointer_controller = None

        if self._window_geometry is None:
            self._focus.inside = True
            print(
                "[SessionRecorder] Window geometry not provided; "
                "recording all input events."
            )

    def start(self) -> None:
        if self._active:
            return

        def relative_coords(x: Any, y: Any) -> tuple[int, int] | None:
            if self._window_geometry is None:
                return None
            try:
                abs_x = int(x)
                abs_y = int(y)
            except (TypeError, ValueError):
                return None
            return self._window_geometry.to_relative(abs_x, abs_y)

        def emit(
            event_type: str, payload: dict[str, Any], *, force: bool = False
        ) -> None:
            should_gate = self._window_geometry is not None and not force
            if should_gate and not self._focus.inside:
                return

            added_relative = False
            if "x" in payload and "y" in payload:
                rel = relative_coords(payload["x"], payload["y"])
                if rel is not None:
                    abs_x = int(payload["x"])
                    abs_y = int(payload["y"])
                    payload["screen_x"] = abs_x
                    payload["screen_y"] = abs_y
                    payload["x"], payload["y"] = rel
                    added_relative = True

            if (
                not added_relative
                and self._window_geometry is not None
                and self._focus.last_relative_x is not None
            ):
                payload.setdefault("pointer_x", self._focus.last_relative_x)
                payload.setdefault("pointer_y", self._focus.last_relative_y)

            event = {
                "event_type": event_type,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # print(f"[SessionRecorder] {event_type}: {json.dumps(payload)}")
            if self._event_callback:
                self._event_callback(event)

        def announce_focus_change(
            new_state: bool,
            x: int,
            y: int,
        ) -> None:
            payload: dict[str, Any] = {"inside": new_state, "x": x, "y": y}
            emit("active_window_focus", payload, force=True)

        def update_focus(x: int, y: int) -> None:
            inside = (
                self._window_geometry.contains(x, y) if self._window_geometry else True
            )
            relative = relative_coords(x, y)
            if inside != self._focus.inside:
                self._focus.inside = inside
                announce_focus_change(inside, x, y)
            self._focus.last_x = x
            self._focus.last_y = y
            if relative is not None:
                self._focus.last_relative_x, self._focus.last_relative_y = relative

        def refresh_focus_from_pointer() -> None:
            if self._pointer_controller is None or self._window_geometry is None:
                return
            try:
                px, py = self._pointer_controller.position
            except Exception:
                return
            update_focus(int(px), int(py))

        def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
            refresh_focus_from_pointer()
            if not self._focus.inside:
                return
            emit("keyboard_press", {"key": self._key_to_str(key)})

        def on_release(key: keyboard.Key | keyboard.KeyCode) -> None:
            refresh_focus_from_pointer()
            if not self._focus.inside:
                return
            emit("keyboard_release", {"key": self._key_to_str(key)})

        def on_click(x: int, y: int, button: mouse.Button, pressed: bool) -> None:
            abs_x, abs_y = int(x), int(y)
            update_focus(abs_x, abs_y)
            emit(
                "mouse_click",
                {
                    "x": abs_x,
                    "y": abs_y,
                    "button": str(button),
                    "pressed": pressed,
                },
            )

        def on_scroll(x: int, y: int, dx: int, dy: int) -> None:
            abs_x, abs_y = int(x), int(y)
            update_focus(abs_x, abs_y)
            emit(
                "mouse_scroll",
                {
                    "x": abs_x,
                    "y": abs_y,
                    "dx": int(dx),
                    "dy": int(dy),
                },
            )

        def on_move(x: int, y: int) -> None:
            abs_x, abs_y = int(x), int(y)
            update_focus(abs_x, abs_y)

        self._keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        self._mouse_listener = mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll,
        )

        self._keyboard_listener.start()
        self._mouse_listener.start()
        self._active = True

        refresh_focus_from_pointer()

    def stop(self) -> None:
        if not self._active:
            return
        for listener in (self._keyboard_listener, self._mouse_listener):
            if listener is None:
                continue
            listener.stop()
            listener.join()
        self._keyboard_listener = None
        self._mouse_listener = None
        self._active = False
        self._focus = _FocusState(inside=self._window_geometry is None)

    def _key_to_str(self, key: Any) -> str:
        if hasattr(key, "char") and key.char is not None:
            return key.char
        return str(key)

    def _coerce_geometry(self, value: Any) -> _WindowGeometry | None:
        if value is None:
            return None
        if isinstance(value, _WindowGeometry):
            return value

        raw: tuple[Any, Any, Any, Any] | None = None
        if isinstance(value, dict):
            if {"x", "y", "width", "height"}.issubset(value.keys()):
                raw = (
                    value["x"],
                    value["y"],
                    value["width"],
                    value["height"],
                )
            elif {"left", "top", "width", "height"}.issubset(value.keys()):
                raw = (
                    value["left"],
                    value["top"],
                    value["width"],
                    value["height"],
                )
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            raw = tuple(value)  # type: ignore[assignment]

        if raw is None:
            return None

        try:
            left, top, width, height = (int(v) for v in raw)
        except (TypeError, ValueError):
            return None

        if width <= 0 or height <= 0:
            return None

        return _WindowGeometry(left=left, top=top, width=width, height=height)
