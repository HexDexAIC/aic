#
#  Copyright (C) 2026 Hariharan Ravichandran
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Keyboard teleop input for shared-autonomy policy execution.

A small pynput-backed listener that tracks which control keys are currently
held and exposes time-integrated cartesian deltas to a caller running at any
control rate. Decouples teleop sensitivity from OS keyboard auto-repeat: as
long as a key is held, ``get_delta(dt)`` returns ``step_per_sec * dt`` along
the corresponding axis.

Used by the ``TeleopAssist`` policy wrapper. Standalone — has no ROS deps and
can be unit-tested or driven from a plain Python REPL.

Key map (base frame):

    W / S       ±x         linear (default 40 mm/s when held)
    A / D       ±y         linear
    R / F       ±z         linear
    Q / E       yaw  ∓     angular (default 28 deg/s when held; q=+yaw, e=-yaw)
    I / K       pitch ∓
    J / L       roll  ∓
    SPACE       toggle pause mode (policy frozen, teleop drives alone)
    TAB         exit pause / re-engage policy
    ESC         emergency stop (publishes a hold then exits)

Step rates can be overridden via env vars TELEOP_LIN_RATE (m/s) and
TELEOP_ANG_RATE (rad/s).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from pynput import keyboard
except ImportError:  # pragma: no cover - pynput is in pixi env
    keyboard = None  # type: ignore[assignment]


DEFAULT_LIN_RATE_MPS = float(os.environ.get("TELEOP_LIN_RATE", "0.04"))
DEFAULT_ANG_RATE_RPS = float(os.environ.get("TELEOP_ANG_RATE", "0.5"))


@dataclass
class TeleopState:
    """Snapshot of teleop intent at a moment in time."""

    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0
    mode: str = "delta"          # "delta" | "pause" | "stop"
    active: bool = False         # True if any movement key was held during this read

    def is_zero(self) -> bool:
        return (
            self.dx == 0.0 and self.dy == 0.0 and self.dz == 0.0
            and self.droll == 0.0 and self.dpitch == 0.0 and self.dyaw == 0.0
        )


# Key → axis sign. Holding a key = continuous +/-1 along that axis.
LINEAR_KEYS = {
    "w": ("x", +1.0),
    "s": ("x", -1.0),
    "a": ("y", +1.0),
    "d": ("y", -1.0),
    "r": ("z", +1.0),
    "f": ("z", -1.0),
}

ANGULAR_KEYS = {
    "q": ("yaw",   +1.0),
    "e": ("yaw",   -1.0),
    "i": ("pitch", +1.0),
    "k": ("pitch", -1.0),
    "j": ("roll",  +1.0),
    "l": ("roll",  -1.0),
}


class KeyboardTeleop:
    """Listen to keyboard, expose time-integrated cartesian deltas.

    Usage::

        teleop = KeyboardTeleop().start()
        try:
            t_prev = time.monotonic()
            while running:
                t_now = time.monotonic()
                state = teleop.get_delta(dt=t_now - t_prev)
                t_prev = t_now
                if state.mode == "stop":
                    break
                # ... apply state.dx, state.dy, ... to commanded pose ...
        finally:
            teleop.stop()
    """

    def __init__(
        self,
        lin_rate_mps: float = DEFAULT_LIN_RATE_MPS,
        ang_rate_rps: float = DEFAULT_ANG_RATE_RPS,
    ) -> None:
        if keyboard is None:
            raise RuntimeError(
                "pynput is not available; KeyboardTeleop cannot start. "
                "Install pynput in the pixi env."
            )
        self.lin_rate = lin_rate_mps
        self.ang_rate = ang_rate_rps
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._mode = "delta"
        self._listener: Optional[keyboard.Listener] = None

    # ── Public API ────────────────────────────────────────────────────

    def start(self) -> "KeyboardTeleop":
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._listener.daemon = True
        self._listener.start()
        return self

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_delta(self, dt: float) -> TeleopState:
        """Return integrated delta over `dt` seconds, given currently-held keys.

        Pause mode still returns the accumulated delta (so the operator can
        nudge while paused — this is what makes pause useful for fine
        positioning). Stop mode returns zeros + mode='stop'.
        """
        with self._lock:
            held = set(self._held)
            mode = self._mode

        if mode == "stop":
            return TeleopState(mode="stop")

        state = TeleopState(mode=mode)
        for k in held:
            if k in LINEAR_KEYS:
                axis, sign = LINEAR_KEYS[k]
                step = sign * self.lin_rate * dt
                if axis == "x": state.dx += step
                elif axis == "y": state.dy += step
                elif axis == "z": state.dz += step
                state.active = True
            elif k in ANGULAR_KEYS:
                axis, sign = ANGULAR_KEYS[k]
                step = sign * self.ang_rate * dt
                if axis == "roll":  state.droll  += step
                elif axis == "pitch": state.dpitch += step
                elif axis == "yaw":   state.dyaw   += step
                state.active = True
        return state

    def get_mode(self) -> str:
        with self._lock:
            return self._mode

    def force_stop(self) -> None:
        """Programmatic emergency stop (for non-keyboard exit paths)."""
        with self._lock:
            self._mode = "stop"

    # ── Internal: key handlers ────────────────────────────────────────

    def _key_to_char(self, key) -> Optional[str]:
        # Letter keys arrive as KeyCode(char='w'); special keys as Key.space etc.
        try:
            ch = key.char
            return ch.lower() if ch else None
        except AttributeError:
            return None

    def _on_press(self, key) -> None:
        # Special toggles first.
        if key == keyboard.Key.space:
            with self._lock:
                self._mode = "pause" if self._mode == "delta" else "delta"
            return
        if key == keyboard.Key.tab:
            with self._lock:
                self._mode = "delta"
            return
        if key == keyboard.Key.esc:
            with self._lock:
                self._mode = "stop"
            return

        ch = self._key_to_char(key)
        if ch is None:
            return
        if ch in LINEAR_KEYS or ch in ANGULAR_KEYS:
            with self._lock:
                self._held.add(ch)

    def _on_release(self, key) -> None:
        ch = self._key_to_char(key)
        if ch is None:
            return
        with self._lock:
            self._held.discard(ch)


# ── Smoke test ────────────────────────────────────────────────────────
def _smoke_test() -> None:  # pragma: no cover
    """Run interactively; prints state every 200 ms until ESC."""
    teleop = KeyboardTeleop().start()
    try:
        t_prev = time.monotonic()
        print("Keyboard teleop smoke test. Press ESC to exit.")
        print("WASD/RF for xyz, QE/IK/JL for yaw/pitch/roll, SPACE pause, TAB resume.")
        while True:
            time.sleep(0.2)
            t_now = time.monotonic()
            state = teleop.get_delta(dt=t_now - t_prev)
            t_prev = t_now
            if state.mode == "stop":
                print("STOP")
                break
            if state.active or state.mode != "delta":
                print(
                    f"mode={state.mode}  "
                    f"d=({state.dx:+.4f}, {state.dy:+.4f}, {state.dz:+.4f}) m  "
                    f"r=({state.droll:+.4f}, {state.dpitch:+.4f}, {state.dyaw:+.4f}) rad"
                )
    finally:
        teleop.stop()


if __name__ == "__main__":
    _smoke_test()
