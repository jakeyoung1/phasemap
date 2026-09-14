"""Pitch geometry, attacking-frame transforms, and named zones.

All positions are metres. Pitch frame: x runs 0-105 from the left goal line to the
right goal line, y runs 0-68 from the bottom touchline to the top touchline.

Analysis happens in the attacking frame: the team in possession always attacks the
goal at x = 105 and its left side is high y. That keeps "forward", "left wing" and
"deepest defender" meaning the same thing in every phase.
"""

from __future__ import annotations

import math

import numpy as np

LENGTH = 105.0
WIDTH = 68.0
BOX_DEPTH = 16.5
BOX_WIDTH = 40.32
SIX_YARD_DEPTH = 5.5
SIX_YARD_WIDTH = 18.32
GOAL_WIDTH = 7.32
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_SPOT = 11.0
ZONE14_DEPTH = 18.5  # metres in front of the box that count as "zone 14"


def to_attacking_frame(xy: np.ndarray, direction: int) -> np.ndarray:
    """Return a copy of `xy` (shape (..., 2)) turned so play goes toward x = 105."""
    out = np.array(xy, dtype=float, copy=True)
    if direction == -1:
        out[..., 0] = LENGTH - out[..., 0]
        out[..., 1] = WIDTH - out[..., 1]
    elif direction != 1:
        raise ValueError(f"direction must be 1 or -1, got {direction}")
    return out


def third(x: float) -> str:
    if x < LENGTH / 3:
        return "defensive third"
    if x < 2 * LENGTH / 3:
        return "middle third"
    return "final third"


def lane(y: float) -> str:
    """Five vertical lanes seen from the attacking team (high y is their left)."""
    offset = y - WIDTH / 2
    if offset > BOX_WIDTH / 2:
        return "left wing"
    if offset > SIX_YARD_WIDTH / 2:
        return "left half-space"
    if offset >= -SIX_YARD_WIDTH / 2:
        return "central"
    if offset >= -BOX_WIDTH / 2:
        return "right half-space"
    return "right wing"


def in_box(x: float, y: float) -> bool:
    return x >= LENGTH - BOX_DEPTH and abs(y - WIDTH / 2) <= BOX_WIDTH / 2


def zone(x: float, y: float) -> str:
    """Coach-readable location label in the attacking frame."""
    if in_box(x, y):
        return "penalty box"
    if LENGTH - BOX_DEPTH - ZONE14_DEPTH <= x < LENGTH - BOX_DEPTH and lane(y) == "central":
        return "zone 14"
    return f"{third(x)}, {lane(y)}"


def goal_view(x: float, y: float) -> tuple[float, float]:
    """Distance (m) to the centre of the goal at x = 105 and the angle (deg) its mouth covers."""
    dx = LENGTH - x
    near = (dx, WIDTH / 2 - GOAL_WIDTH / 2 - y)
    far = (dx, WIDTH / 2 + GOAL_WIDTH / 2 - y)
    cross = near[0] * far[1] - near[1] * far[0]
    dot = near[0] * far[0] + near[1] * far[1]
    return math.hypot(dx, WIDTH / 2 - y), math.degrees(abs(math.atan2(cross, dot)))
