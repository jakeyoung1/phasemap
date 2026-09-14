"""Per-phase working view: attacking-frame positions, velocities, and on-ball spans."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import kinematics, pitch, space
from .match import Event, Match, other
from .phases import Phase, window

TEAM = {"home": "Home", "away": "Away"}


def r1(value: float) -> float:
    return round(float(value), 1)


def finite(point) -> bool:
    return bool(np.isfinite(point).all())


def unique(*names) -> tuple[str, ...]:
    """Drop empties and repeats, keeping first-seen order."""
    return tuple(dict.fromkeys(n for n in names if n))


@dataclass(frozen=True)
class Draft:
    """A fact before it gets an ID. Detectors return these; the ledger numbers them."""

    kind: str
    i: int  # window index the fact refers to
    text: str
    players: tuple[str, ...] = ()
    values: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Span:
    """A stretch where one attacker had the ball at their feet."""

    player: str
    arrive: int  # window index when the ball reached them
    release: int  # window index of their next action
    action: Event


@dataclass
class PhaseView:
    match: Match
    phase: Phase
    start: int  # first tracking row in the window
    stop: int  # one past the last row
    xy: np.ndarray  # (W, P, 2) attacking frame
    vel: np.ndarray  # (W, P, 2) m/s in the attacking frame
    ball: np.ndarray  # (W, 2) attacking frame

    @classmethod
    def build(cls, match: Match, phase: Phase, **window_kw) -> PhaseView:
        start, stop = window(match, phase, **window_kw)
        direction = match.attack_dir[(phase.team, phase.period)]
        xy = pitch.to_attacking_frame(match.xy[start:stop], direction)
        ball = pitch.to_attacking_frame(match.ball[start:stop], direction)
        return cls(match, phase, start, stop, xy, kinematics.velocity(xy, match.fps), ball)

    @property
    def fps(self) -> float:
        return self.match.fps

    @property
    def n(self) -> int:
        return self.stop - self.start

    @property
    def attack(self) -> str:
        return self.phase.team

    @property
    def defence(self) -> str:
        return other(self.phase.team)

    @property
    def direction(self) -> int:
        return self.match.attack_dir[(self.phase.team, self.phase.period)]

    def i(self, frame: int) -> int:
        """Window index for a provider frame, clamped into the window."""
        return int(np.clip(self.match.row(frame) - self.start, 0, self.n - 1))

    def t(self, i: int) -> float:
        return i / self.fps

    def frame(self, i: int) -> int:
        return int(self.match.frames[self.start + i])

    def keeper(self, team: str) -> str:
        return self.match.goalkeepers[(team, self.phase.period)]

    def cols(self, team: str, outfield: bool = False) -> np.ndarray:
        cols = self.match.team_cols(team)
        if outfield:
            cols = cols[cols != self.match.col(self.keeper(team))]
        return cols

    def ids(self, cols) -> list[str]:
        return [self.match.player_ids[c] for c in cols]

    def pos(self, player: str, i: int) -> np.ndarray:
        return self.xy[i, self.match.col(player)]

    def nearest(self, point, i: int, cols) -> tuple[float, str | None]:
        dist, k = space.nearest(point, self.xy[i, cols])
        return dist, (self.ids(cols)[k] if k >= 0 else None)

    def event_point(self, xy) -> np.ndarray:
        if xy is None:
            return np.array([np.nan, np.nan])
        return pitch.to_attacking_frame(np.array(xy, dtype=float), self.direction)

    def action_point(self, player: str | None, i: int, fallback) -> np.ndarray:
        """Where a player stood at window index i, else the event's own coordinates."""
        if player is not None and player in self.match.player_ids:
            spot = self.pos(player, i)
            if finite(spot):
                return spot
        return self.event_point(fallback)

    @cached_property
    def actions(self) -> list[Event]:
        """On-ball decisions in order: passes and the shot."""
        return [e for e in self.phase.events if e.type in ("PASS", "SHOT")]

    @property
    def last_action(self) -> Event:
        return self.actions[-1] if self.actions else self.phase.events[-1]

    @property
    def won_i(self) -> int:
        return self.i(self.phase.start_frame)

    @property
    def final_i(self) -> int:
        return self.i(self.last_action.start_frame)

    @property
    def final_pass(self) -> Event | None:
        passes = self.phase.passes
        return passes[-1] if passes else None

    @cached_property
    def spans(self) -> list[Span]:
        out = []
        events = self.phase.events
        for prev, cur in zip(events, events[1:]):
            if cur.player is None:
                continue
            if prev.type == "PASS" and prev.receiver == cur.player:
                arrive = prev.end_frame
            elif prev.player == cur.player and prev.type in ("RECOVERY", "SET PIECE"):
                arrive = prev.start_frame
            else:
                continue
            out.append(Span(cur.player, self.i(arrive), self.i(cur.start_frame), cur))
        return out
