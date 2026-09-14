"""Assemble the evidence ledger for one phase."""

from __future__ import annotations

from collections import Counter

from . import context, offball, onball, quality
from .evidence import Ledger
from .match import Match
from .phases import Phase
from .view import Draft, PhaseView


def situation(v: PhaseView) -> list[Draft]:
    match, phase = v.match, v.phase
    before, after = match.score_before(phase.start_frame), match.score_before(phase.end_frame + 1)
    half = {1: "first half", 2: "second half"}.get(phase.period, "extra time")
    text = f"Match situation: {half}, Home {before['home']} Away {before['away']} before this phase"
    values = {"home_before": before["home"], "away_before": before["away"]}
    if after != before:
        text += f", Home {after['home']} Away {after['away']} after it"
        values.update(home_after=after["home"], away_after=after["away"])
    return [Draft("situation", v.won_i, text + ".", (), values)]


DETECTORS = (
    situation,
    quality.merged_tracks,
    onball.possession_start,
    onball.passes,
    onball.carries,
    onball.options,
    onball.shot,
    onball.tempo,
    offball.back_line,
    offball.shifts,
    offball.runs,
    offball.recovery,
    offball.shapes,
    offball.box_numbers,
    offball.control,
)
KIND_ORDER = (
    "situation", "possession start", "data quality", "shape", "control", "pass", "carry", "run",
    "defender shift", "recovery run", "back-line gap", "line", "option", "box numbers", "back line",
    "behind the ball", "shot", "tempo",
)


def build(match: Match, phase: Phase, phases: list[Phase], stats=None) -> tuple[Ledger, PhaseView]:
    """Measure one phase. Pass `stats` from context.match_stats to reuse it across phases."""
    view = PhaseView.build(match, phase)
    rank = {kind: n for n, kind in enumerate(KIND_ORDER)}
    drafts = sorted(
        (d for detect in DETECTORS for d in detect(view)),
        key=lambda d: (d.i, rank.get(d.kind, len(rank))),
    )
    ledger = Ledger()
    for d in drafts:
        ledger.add(d.kind, view.t(d.i), view.frame(d.i), d.text, d.players, **d.values)
    team_s, player_s = stats if stats is not None else context.match_stats(match, phases)
    involved = [pid for pid, _ in Counter(p for d in drafts for p in d.players).most_common()]
    for d in context.facts(match, team_s, player_s, view, involved):
        ledger.add(d.kind, 0.0, None, d.text, d.players, **d.values)
    return ledger, view
