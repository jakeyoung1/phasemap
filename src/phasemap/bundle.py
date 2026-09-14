"""Everything a report needs about one analysed phase, as strict JSON."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from .evidence import Ledger, Verification
from .view import TEAM, PhaseView, finite


def clean(value):
    """Swap NaN and infinity for None so the result is valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def _point(p) -> list[float] | None:
    return [round(float(p[0]), 1), round(float(p[1]), 1)] if finite(p) else None


def build(
    view: PhaseView,
    ledger: Ledger,
    analysis: dict | None,
    verification: Verification | None,
    backend: str | None,
) -> dict:
    match, phase = view.match, view.phase
    keepers = {view.keeper("home"), view.keeper("away")}
    cols = [c for c in range(len(match.player_ids)) if np.isfinite(view.xy[:, c, 0]).any()]
    events = []
    for ev in phase.events:
        i0, i1 = view.i(ev.start_frame), view.i(ev.end_frame)
        events.append(
            {
                "t": round(view.t(i0), 2),
                "t_end": round(view.t(i1), 2),
                "type": ev.type,
                "label": ev.label,
                "player": ev.player,
                "receiver": ev.receiver,
                "start": _point(view.action_point(ev.player, i0, ev.start_xy)),
                "end": _point(view.action_point(ev.receiver, i1, ev.end_xy)),
            }
        )
    return clean(
        {
            "meta": {
                "match": match.name,
                "source": match.source,
                "phase": phase.id,
                "team": phase.team,
                "team_label": TEAM[phase.team],
                "period": phase.period,
                "clock_start": match.clock_label(phase.start_frame),
                "clock_end": match.clock_label(phase.end_frame),
                "outcome": phase.outcome,
                "fps": view.fps,
                "duration_s": round(view.n / view.fps, 2),
                "won_t": round(view.t(view.won_i), 2),
                "final_t": round(view.t(view.final_i), 2),
                "score_before": match.score_before(phase.start_frame),
                "score_after": match.score_before(phase.end_frame + 1),
                "backend": backend,
                "generated": dt.date.today().isoformat(),
            },
            "players": [
                {"id": match.player_ids[c], "team": match.player_teams[c], "keeper": match.player_ids[c] in keepers}
                for c in cols
            ],
            "track": {
                "xy": np.round(view.xy[:, cols].reshape(view.n, -1), 1).tolist(),
                "ball": [_point(b) for b in view.ball],
            },
            "events": events,
            "facts": ledger.to_json(),
            "analysis": analysis,
            "verification": verification.to_json() if verification else None,
        }
    )
