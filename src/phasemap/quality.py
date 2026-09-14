"""Data-quality facts: places where the tracking itself is doubtful."""

from __future__ import annotations

import numpy as np

from . import kinematics
from .view import Draft, PhaseView, r1

MERGED_M = 0.1  # two players this close are almost certainly sharing one track
MERGED_MIN_S = 1.0


def merged_tracks(v: PhaseView) -> list[Draft]:
    """Pairs of players who share one tracked position for a stretch, usually an occlusion in a crowd."""
    out = []
    ids = v.match.player_ids
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            gap = np.hypot(*(v.xy[:, a] - v.xy[:, b]).T)
            with np.errstate(invalid="ignore"):
                runs = [(s, e) for s, e in kinematics.finite_runs(gap < MERGED_M) if (e - s) / v.fps >= MERGED_MIN_S]
            if not runs:
                continue
            secs = sum(e - s for s, e in runs) / v.fps
            text = (
                f"Tracking check: {ids[a]} and {ids[b]} share one tracked position for {secs:.1f} s of this window, "
                "which usually means a single track is covering both players. Treat their positions, speeds "
                "and spacing here with caution."
            )
            out.append(Draft("data quality", runs[0][0], text, (ids[a], ids[b]), {"seconds": r1(secs)}))
    return out
