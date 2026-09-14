"""Provider-neutral match container: tracking arrays, events, and team orientation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from . import pitch

TEAMS = ("home", "away")
HALF_STARTS_S = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}


def other(team: str) -> str:
    return "away" if team == "home" else "home"


@dataclass(frozen=True)
class Event:
    index: int
    team: str  # "home" | "away"
    type: str  # upper-case provider type, e.g. "PASS", "SHOT"
    subtype: str  # "" when absent, e.g. "ON TARGET-GOAL"
    period: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    player: str | None  # e.g. "H10"
    receiver: str | None
    start_xy: tuple[float, float] | None  # pitch frame, metres
    end_xy: tuple[float, float] | None

    @property
    def is_goal(self) -> bool:
        return self.type == "SHOT" and "GOAL" in self.subtype.split("-")

    @property
    def label(self) -> str:
        kind = self.type.lower()
        return f"{kind} ({self.subtype.lower()})" if self.subtype else kind


@dataclass
class Match:
    name: str
    source: str  # attribution line for anything published
    fps: float
    frames: np.ndarray  # (T,) provider frame numbers, contiguous
    periods: np.ndarray  # (T,)
    clock_s: np.ndarray  # (T,) provider clock in seconds
    player_ids: list[str]  # "H11", "A25", ...
    player_teams: list[str]  # "home" / "away", aligned with player_ids
    xy: np.ndarray  # (T, P, 2) metres in the pitch frame, NaN when off the pitch
    ball: np.ndarray  # (T, 2) metres, NaN when unknown
    events: list[Event]

    def __post_init__(self) -> None:
        if len(self.frames) > 1 and not np.all(np.diff(self.frames) == 1):
            raise ValueError("tracking frames must be contiguous")
        self._col = {pid: i for i, pid in enumerate(self.player_ids)}

    def row(self, frame: int) -> int:
        """Array row for a provider frame number, clamped to the tracked range."""
        return int(np.clip(frame - self.frames[0], 0, len(self.frames) - 1))

    def col(self, player_id: str) -> int:
        return self._col[player_id]

    def team_of(self, player_id: str) -> str:
        return self.player_teams[self._col[player_id]]

    def team_cols(self, team: str) -> np.ndarray:
        return np.array([i for i, t in enumerate(self.player_teams) if t == team], dtype=int)

    @cached_property
    def period_rows(self) -> dict[int, tuple[int, int]]:
        """First row and one-past-last row of each period."""
        out = {}
        for p in np.unique(self.periods):
            rows = np.flatnonzero(self.periods == p)
            out[int(p)] = (int(rows[0]), int(rows[-1]) + 1)
        return out

    @cached_property
    def attack_dir(self) -> dict[tuple[str, int], int]:
        """+1 when a team attacks toward x = 105 in a period, else -1.

        Read from each period's first second, when both teams line up in their own half
        for kick-off: the team whose players sit further left attacks to the right.
        """
        out: dict[tuple[str, int], int] = {}
        for period, (first, stop) in self.period_rows.items():
            rows = np.arange(first, min(stop, first + int(self.fps)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                home_x = np.nanmedian(self.xy[np.ix_(rows, self.team_cols("home"))][..., 0])
                away_x = np.nanmedian(self.xy[np.ix_(rows, self.team_cols("away"))][..., 0])
            home = 1 if home_x < away_x else -1
            out[("home", period)], out[("away", period)] = home, -home
        return out

    @cached_property
    def goalkeepers(self) -> dict[tuple[str, int], str]:
        """Each team's keeper per period: the regular who stays closest to their own goal line."""
        out: dict[tuple[str, int], str] = {}
        for period, (first, stop) in self.period_rows.items():
            rows = np.arange(first, stop, max(1, int(self.fps)))
            for team in TEAMS:
                cols = self.team_cols(team)
                x = self.xy[np.ix_(rows, cols)][..., 0]
                own_goal = 0.0 if self.attack_dir[(team, period)] == 1 else pitch.LENGTH
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    depth = np.nanmedian(np.abs(x - own_goal), axis=0)
                depth[np.isfinite(x).mean(axis=0) < 0.5] = np.inf
                out[(team, period)] = self.player_ids[cols[int(np.argmin(depth))]]
        return out

    def clock_label(self, frame: int) -> str:
        """Broadcast-style match clock, e.g. "47:44" early in the second half."""
        r = self.row(frame)
        period = int(self.periods[r])
        elapsed = self.clock_s[r] - self.clock_s[self.period_rows[period][0]]
        total = int(HALF_STARTS_S.get(period, 0) + elapsed)
        return f"{total // 60}:{total % 60:02d}"

    def score_before(self, frame: int) -> dict[str, int]:
        score = {team: 0 for team in TEAMS}
        for ev in self.events:
            if ev.is_goal and ev.start_frame < frame:
                score[ev.team] += 1
        return score
