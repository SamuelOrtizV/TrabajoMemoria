"""Keyboard control helpers for the game (WASD + combos).

This module exposes simple functions to emulate common movement inputs
using DirectInput scan codes via PressKey/ReleaseKey.

Controls mapping:
- 0: none
- 1: A
- 2: D
- 3: W
- 4: S
- 5: A+W
- 6: A+S
- 7: D+W
- 8: D+S
"""

from .game_control import PressKey, ReleaseKey, W, A, S, D, CTRL, N, Y, B
import time

# Delay between turning actions. Larger values produce longer/stronger turns.
# Must be < 1/FPS.
SLEEP_TIME = 0.09


def reset_race(sleep_time: float = 2.0) -> None:
    """Reset the race session using hotkeys (CTRL+N, then CTRL+Y).

    The timings here are conservative to ensure the game registers the keypresses.
    """
    # Press CTRL + N to reset the session
    PressKey(CTRL)
    PressKey(N)
    time.sleep(0.1)
    ReleaseKey(N)
    ReleaseKey(CTRL)

    time.sleep(sleep_time)  # Wait for the environment to reset

    # Press CTRL + Y to start the race
    PressKey(CTRL)
    PressKey(Y)
    time.sleep(0.1)
    ReleaseKey(Y)
    ReleaseKey(CTRL)


def go_to_pits() -> None:
    """Send the car to pits using hotkey (CTRL+B)."""
    PressKey(CTRL)
    PressKey(B)
    time.sleep(0.1)
    ReleaseKey(B)
    ReleaseKey(CTRL)


# Action primitives ---------------------------------------------------------------------

def none() -> None:
    """Release all steering/brake keys (no action)."""
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def move_left() -> None:
    """Press 'A' briefly to steer left."""
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    time.sleep(SLEEP_TIME)
    ReleaseKey(A)


def move_right() -> None:
    """Press 'D' briefly to steer right."""
    PressKey(D)
    ReleaseKey(S)
    ReleaseKey(A)
    time.sleep(SLEEP_TIME)
    ReleaseKey(D)


def move_forward() -> None:
    """Hold 'W' to accelerate forward (release other keys)."""
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def move_back() -> None:
    """Press 'S' briefly to brake/reverse."""
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    time.sleep(SLEEP_TIME)
    ReleaseKey(S)


def move_left_forward() -> None:
    """Press 'A' + 'W' briefly (left + forward)."""
    PressKey(A)
    PressKey(W)
    ReleaseKey(S)
    ReleaseKey(D)
    time.sleep(SLEEP_TIME)
    ReleaseKey(A)


def move_left_back() -> None:
    """Press 'A' + 'S' briefly (left + back)."""
    PressKey(A)
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(SLEEP_TIME)
    ReleaseKey(A)
    ReleaseKey(S)


def move_right_forward() -> None:
    """Press 'D' + 'W' briefly (right + forward)."""
    PressKey(D)
    PressKey(W)
    ReleaseKey(S)
    ReleaseKey(A)
    time.sleep(SLEEP_TIME)
    ReleaseKey(D)


def move_right_back() -> None:
    """Press 'D' + 'S' briefly (right + back)."""
    PressKey(D)
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    time.sleep(SLEEP_TIME)
    ReleaseKey(D)
    ReleaseKey(S)


def move(direction: int) -> None:
    """Execute a discrete movement action by id (0..8).

    :param direction: action id (see module docstring mapping).
    :raises ValueError: if id is not between 0 and 8.
    """
    if direction == 0:
        none()
    elif direction == 1:
        move_left()
    elif direction == 2:
        move_right()
    elif direction == 3:
        move_forward()
    elif direction == 4:
        move_back()
    elif direction == 5:
        move_left_forward()
    elif direction == 6:
        move_left_back()
    elif direction == 7:
        move_right_forward()
    elif direction == 8:
        move_right_back()
    else:
        raise ValueError("direction must be between 0 and 8")