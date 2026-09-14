"""Velocity, speed, distance, and sprints from noisy positional data."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

PLAYER_MAX_SPEED = 12.0  # m/s; anything faster is a tracking glitch
SPRINT_SPEED = 7.0  # m/s (25.2 km/h)
HIGH_SPEED = 5.5  # m/s (19.8 km/h)
SMOOTH_WINDOW = 7  # frames (0.28 s at 25 fps)


def finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs for each block of consecutive True values."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def velocity(
    xy: np.ndarray,
    fps: float,
    window: int = SMOOTH_WINDOW,
    max_speed: float | None = PLAYER_MAX_SPEED,
) -> np.ndarray:
    """Smoothed velocity in m/s for positions shaped (T, 2) or (T, N, 2).

    Each unbroken stretch of known positions is differentiated on its own with a
    Savitzky-Golay filter, so gaps never leak into neighbouring frames. Readings above
    `max_speed` are dropped as glitches.
    """
    arr = np.asarray(xy, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[:, None, :]
    vel = np.full(arr.shape, np.nan)
    dt = 1.0 / fps
    for j in range(arr.shape[1]):
        known = np.isfinite(arr[:, j, :]).all(axis=1)
        for a, b in finite_runs(known):
            seg = arr[a:b, j, :]
            if b - a >= window:
                vel[a:b, j, :] = savgol_filter(seg, window, polyorder=1, deriv=1, delta=dt, axis=0)
            elif b - a >= 2:
                vel[a:b, j, :] = np.gradient(seg, dt, axis=0)
    if max_speed is not None:
        with np.errstate(invalid="ignore"):
            vel[np.hypot(vel[..., 0], vel[..., 1]) > max_speed] = np.nan
    return vel[:, 0, :] if single else vel


def speed(vel: np.ndarray) -> np.ndarray:
    return np.hypot(vel[..., 0], vel[..., 1])


def distance_covered(vel: np.ndarray, fps: float) -> np.ndarray:
    """Metres covered: speed integrated over the frames where it is known."""
    return np.nansum(speed(vel), axis=0) / fps


def sprints(
    spd: np.ndarray, fps: float, threshold: float = SPRINT_SPEED, min_seconds: float = 1.0
) -> list[tuple[int, int]]:
    """Frame ranges where speed stays above `threshold` for at least `min_seconds`."""
    fast = np.nan_to_num(np.asarray(spd, dtype=float), nan=0.0) > threshold
    return [(a, b) for a, b in finite_runs(fast) if (b - a) / fps >= min_seconds]
