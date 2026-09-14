"""Team shape and back-line structure for a single frame, in the attacking frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from . import pitch


@dataclass(frozen=True)
class Shape:
    width: float  # m between the widest outfield players
    depth: float  # m between the deepest and highest outfield players
    area: float  # m² inside the outfield players' convex hull
    centroid: tuple[float, float]


@dataclass(frozen=True)
class BackLine:
    ids: tuple[str, ...]  # from the attacking team's left (high y) to its right
    height: float  # m from the defended goal line to the line's average x
    gaps: tuple[float, ...]  # distance between neighbours, in `ids` order
    widest_gap: float
    widest_pair: tuple[str, str]
    last_defender_x: float  # x of the deepest outfield defender


def team_shape(xy: np.ndarray) -> Shape | None:
    """Shape of one team's outfield players (rows of x, y; NaN rows ignored)."""
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None
    try:
        area = float(ConvexHull(pts).volume)  # scipy reports a 2-D hull's area as volume
    except QhullError:
        area = 0.0
    return Shape(
        width=float(np.ptp(pts[:, 1])),
        depth=float(np.ptp(pts[:, 0])),
        area=area,
        centroid=(float(pts[:, 0].mean()), float(pts[:, 1].mean())),
    )


def back_line(xy: np.ndarray, ids: list[str], size: int = 4) -> BackLine | None:
    """The `size` deepest outfield defenders, who protect the goal at x = 105."""
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    present = np.flatnonzero(np.isfinite(pts).all(axis=1))
    if len(present) < max(size, 2):
        return None
    deepest = present[np.argsort(-pts[present, 0], kind="stable")[:size]]
    order = deepest[np.argsort(-pts[deepest, 1], kind="stable")]
    line = pts[order]
    gaps = np.hypot(*np.diff(line, axis=0).T)
    k = int(np.argmax(gaps))
    return BackLine(
        ids=tuple(ids[i] for i in order),
        height=float(pitch.LENGTH - line[:, 0].mean()),
        gaps=tuple(float(g) for g in gaps),
        widest_gap=float(gaps[k]),
        widest_pair=(ids[order[k]], ids[order[k + 1]]),
        last_defender_x=float(line[:, 0].max()),
    )
