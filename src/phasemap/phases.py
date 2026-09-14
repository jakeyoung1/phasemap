"""Split a match's event stream into possession phases."""

from __future__ import annotations

from dataclasses import dataclass, field

from .match import Event, Match

POSSESSION_TYPES = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST", "BALL OUT", "FAULT RECEIVED"}
ENDING_TYPES = {"SHOT", "BALL LOST", "BALL OUT", "FAULT RECEIVED"}
SHOT_RANK = {"goal": 0, "shot saved": 1, "shot hit woodwork": 1, "shot blocked": 2, "shot off target": 2}


@dataclass
class Phase:
    id: int
    team: str
    period: int
    events: list[Event] = field(default_factory=list)
    outcome: str = "open"

    @property
    def start_frame(self) -> int:
        return self.events[0].start_frame

    @property
    def end_frame(self) -> int:
        return max(e.end_frame for e in self.events)

    @property
    def duration_s(self) -> float:
        return max(e.end_s for e in self.events) - self.events[0].start_s

    @property
    def passes(self) -> list[Event]:
        return [e for e in self.events if e.type == "PASS"]

    @property
    def shot(self) -> Event | None:
        return next((e for e in self.events if e.type == "SHOT"), None)

    @property
    def start_kind(self) -> str:
        return self.events[0].label

    @property
    def ended(self) -> bool:
        return self.outcome != "open"


def _outcome(ev: Event) -> str:
    if ev.type == "SHOT":
        parts = ev.subtype.split("-")
        if "GOAL" in parts:
            return "goal"
        if "BLOCKED" in parts:
            return "shot blocked"
        if "SAVED" in parts:
            return "shot saved"
        if "WOODWORK" in parts:
            return "shot hit woodwork"
        return "shot off target"
    if ev.type == "BALL LOST":
        return "turnover"
    if ev.type == "BALL OUT":
        return "ball out of play"
    return "foul won"


def segment(events: list[Event]) -> list[Phase]:
    """Group consecutive on-ball events by the team in possession.

    A phase ends on a shot, turnover, ball out of play, or foul, when the other team
    shows up on the ball without a tagged turnover, or when the period changes.
    Challenges and cards say nothing about possession and are skipped.
    """
    phases: list[Phase] = []
    current: Phase | None = None
    for ev in events:
        if ev.type not in POSSESSION_TYPES:
            continue
        if current is None or current.ended or ev.team != current.team or ev.period != current.period:
            if current is not None and not current.ended:
                current.outcome = "period ended" if ev.period != current.period else "possession lost"
            current = Phase(id=len(phases), team=ev.team, period=ev.period)
            phases.append(current)
        current.events.append(ev)
        if ev.type in ENDING_TYPES:
            current.outcome = _outcome(ev)
    return phases


def notable(phases: list[Phase], limit: int | None = None) -> list[Phase]:
    """Phases that produced a shot, goals first, then longer passing moves."""
    shots = [p for p in phases if p.outcome in SHOT_RANK]
    shots.sort(key=lambda p: (SHOT_RANK[p.outcome], -len(p.passes), p.start_frame))
    return shots[:limit] if limit else shots


def window(
    match: Match, phase: Phase, lead_s: float = 4.0, tail_s: float = 1.5, max_s: float = 40.0
) -> tuple[int, int]:
    """Tracking rows [start, stop) to measure and replay for a phase.

    Adds a lead-in (how the ball was won) and a short tail (what happened next), keeps
    at most the final `max_s` seconds of long possessions, and never crosses a period.
    """
    first, last = match.period_rows[phase.period]
    stop = min(match.row(phase.end_frame) + int(tail_s * match.fps) + 1, last)
    start = match.row(phase.start_frame) - int(lead_s * match.fps)
    start = max(start, stop - int((max_s + tail_s) * match.fps), first)
    return start, stop
