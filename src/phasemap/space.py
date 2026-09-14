"""Pressure, passing lanes, and a simple pitch-control surface.

Pitch control is an arrival-time model: each player keeps moving on their current
velocity for a reaction time, then runs straight at a fixed top speed. The team that
reaches a spot first controls it, softened into a probability with a logistic curve.
The parameters (0.7 s reaction, 5 m/s top speed, 0.45 s arrival spread) come from the
Spearman-style model in Laurie Shaw's Friends of Tracking code; comparing team-best
arrival times directly is a further simplification of that model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import pitch

REACTION_S = 0.7
TOP_SPEED = 5.0
ARRIVAL_SIGMA_S = 0.45
LANE_RADIUS = 2.0
LANE_GROWTH = 0.05


def nearest(point, others: np.ndarray) -> tuple[float, int]:
    """Distance to the closest player in `others` (rows of x, y) and that row's index."""
    rows = np.asarray(others, dtype=float).reshape(-1, 2)
    dist = np.hypot(rows[:, 0] - point[0], rows[:, 1] - point[1])
    if not np.isfinite(dist).any():
        return float("nan"), -1
    i = int(np.nanargmin(dist))
    return float(dist[i]), i


def count_within(point, others: np.ndarray, radius: float) -> int:
    rows = np.asarray(others, dtype=float).reshape(-1, 2)
    dist = np.hypot(rows[:, 0] - point[0], rows[:, 1] - point[1])
    with np.errstate(invalid="ignore"):
        return int(np.count_nonzero(dist <= radius))


def lane_blockers(
    start, end, opponents: np.ndarray, radius: float = LANE_RADIUS, growth: float = LANE_GROWTH
) -> list[int]:
    """Rows of `opponents` close enough to the straight line start->end to contest a pass.

    The allowed distance widens the further along the line a defender stands, since a
    longer ball flight gives more time to step across. Players level with the receiver
    (last 10% of the line) count as marking the receiver, not blocking the lane.
    """
    a = np.asarray(start, dtype=float)
    ab = np.asarray(end, dtype=float) - a
    length = float(np.hypot(*ab))
    opp = np.asarray(opponents, dtype=float).reshape(-1, 2)
    if length < 1e-6 or len(opp) == 0:
        return []
    t = ((opp - a) @ ab) / length**2
    closest = a + np.outer(np.clip(t, 0.0, 1.0), ab)
    dist = np.hypot(*(opp - closest).T)
    with np.errstate(invalid="ignore"):
        hit = (t > 0.05) & (t < 0.9) & (dist <= radius + growth * t * length)
    return [int(i) for i in np.flatnonzero(hit)]


@dataclass(frozen=True)
class ControlGrid:
    xs: np.ndarray  # (nx,) cell-centre x in metres
    ys: np.ndarray  # (ny,) cell-centre y in metres
    prob: np.ndarray  # (ny, nx) chance the attacking team controls each cell

    def at(self, x: float, y: float) -> float:
        return float(self.prob[np.abs(self.ys - y).argmin(), np.abs(self.xs - x).argmin()])

    def share(
        self,
        x_min: float = 0.0,
        x_max: float = pitch.LENGTH,
        y_min: float = 0.0,
        y_max: float = pitch.WIDTH,
    ) -> float:
        """Average attacking control over the cells whose centres fall in the rectangle."""
        rows = (self.ys >= y_min) & (self.ys <= y_max)
        cols = (self.xs >= x_min) & (self.xs <= x_max)
        return float(self.prob[np.ix_(rows, cols)].mean())


def _team_arrival(targets: np.ndarray, pos, vel) -> np.ndarray:
    """Earliest time (s) any player on the team can reach each target."""
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    vel = np.zeros_like(pos) if vel is None else np.nan_to_num(np.asarray(vel, dtype=float).reshape(-1, 2))
    on_pitch = np.isfinite(pos).all(axis=1)
    if not on_pitch.any():
        return np.full(len(targets), np.inf)
    launch = pos[on_pitch] + vel[on_pitch] * REACTION_S
    gap = np.hypot(
        targets[:, None, 0] - launch[None, :, 0],
        targets[:, None, 1] - launch[None, :, 1],
    )
    return REACTION_S + gap.min(axis=1) / TOP_SPEED


def control_grid(att_pos, att_vel, def_pos, def_vel, nx: int = 53, ny: int = 34) -> ControlGrid:
    """Attacking team's control probability on an nx-by-ny grid (about 2 m cells)."""
    xs = (np.arange(nx) + 0.5) * pitch.LENGTH / nx
    ys = (np.arange(ny) + 0.5) * pitch.WIDTH / ny
    gx, gy = np.meshgrid(xs, ys)
    targets = np.column_stack([gx.ravel(), gy.ravel()])
    lead = _team_arrival(targets, def_pos, def_vel) - _team_arrival(targets, att_pos, att_vel)
    steep = np.pi / (np.sqrt(3.0) * ARRIVAL_SIGMA_S)
    prob = 1.0 / (1.0 + np.exp(-steep * lead))
    return ControlGrid(xs=xs, ys=ys, prob=prob.reshape(ny, nx))
