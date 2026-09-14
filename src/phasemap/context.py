"""Match context: team and player tendencies across the whole game.

Coaches judge a moment against what a team or player usually does. Every number here
comes from this one match, so it describes the game, not a season.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import kinematics, pitch, shape
from .match import TEAMS, Match, other
from .phases import Phase
from .view import TEAM, Draft, PhaseView, r1

SAMPLE_S = 1.0  # seconds between shape samples
PHYSICAL_STEP = 5  # frames between samples for distance and sprints
MAX_PLAYERS = 8
EVENT_COUNTS = ("passes", "forward_passes", "ball_lost", "shots", "goals", "duels_won", "duels_lost")


def _outfield(match: Match, team: str, period: int) -> np.ndarray:
    cols = match.team_cols(team)
    return cols[cols != match.col(match.goalkeepers[(team, period)])]


def _owner_by_row(match: Match, phases: list[Phase]) -> np.ndarray:
    """0 while home has the ball, 1 for away, -1 when neither does."""
    owner = np.full(len(match.frames), -1, dtype=np.int8)
    for p in phases:
        owner[match.row(p.start_frame) : match.row(p.end_frame) + 1] = TEAMS.index(p.team)
    return owner


def team_stats(match: Match, phases: list[Phase]) -> dict[str, dict[str, float]]:
    owner = _owner_by_row(match, phases)
    held = max(1, int(np.count_nonzero(owner >= 0)))
    heights: dict[str, list[float]] = {t: [] for t in TEAMS}
    widths: dict[str, list[float]] = {t: [] for t in TEAMS}
    for r in range(0, len(match.frames), max(1, int(SAMPLE_S * match.fps))):
        if owner[r] < 0:
            continue
        attack = TEAMS[owner[r]]
        defence = other(attack)
        period = int(match.periods[r])
        xy = pitch.to_attacking_frame(match.xy[r], match.attack_dir[(attack, period)])
        d_cols, a_cols = _outfield(match, defence, period), _outfield(match, attack, period)
        line = shape.back_line(xy[d_cols], [match.player_ids[c] for c in d_cols])
        if line is not None:
            heights[defence].append(line.height)
        spread = shape.team_shape(xy[a_cols])
        if spread is not None:
            widths[attack].append(spread.width)
    stats = {}
    for k, team in enumerate(TEAMS):
        mine = [e for e in match.events if e.team == team]
        recoveries = [e for e in mine if e.type == "RECOVERY" and e.start_xy]
        high = sum(
            1
            for e in recoveries
            if pitch.to_attacking_frame(np.array(e.start_xy), match.attack_dir[(team, e.period)])[0] >= pitch.LENGTH / 2
        )
        stats[team] = {
            "possession_pct": 100 * np.count_nonzero(owner == k) / held,
            "shots": sum(e.type == "SHOT" for e in mine),
            "goals": sum(e.is_goal for e in mine),
            "high_recovery_pct": 100 * high / max(1, len(recoveries)),
            "line_height_m": float(np.mean(heights[team])) if heights[team] else float("nan"),
            "width_m": float(np.mean(widths[team])) if widths[team] else float("nan"),
        }
    return stats


def player_stats(match: Match) -> dict[str, dict[str, float]]:
    counts = {pid: Counter() for pid in match.player_ids}
    for e in match.events:
        if e.player not in counts:
            continue
        c = counts[e.player]
        if e.type == "PASS":
            c["passes"] += 1
            if e.start_xy and e.end_xy and (e.end_xy[0] - e.start_xy[0]) * match.attack_dir[(e.team, e.period)] >= 5:
                c["forward_passes"] += 1
        elif e.type == "BALL LOST":
            c["ball_lost"] += 1
        elif e.type == "SHOT":
            c["shots"] += 1
            c["goals"] += int(e.is_goal)
        elif e.type == "CHALLENGE":
            parts = e.subtype.split("-")
            if "WON" in parts:
                c["duels_won"] += 1
            elif "LOST" in parts:
                c["duels_lost"] += 1
    fps = match.fps / PHYSICAL_STEP
    sampled = match.xy[::PHYSICAL_STEP]
    spd = kinematics.speed(kinematics.velocity(sampled, fps))
    per_second = max(1, int(round(fps)))
    out = {}
    for col, pid in enumerate(match.player_ids):
        s = spd[:, col]
        filled = np.nan_to_num(s)
        smooth = np.convolve(filled, np.ones(per_second) / per_second, mode="valid") if len(s) >= per_second else filled
        out[pid] = {
            **{key: counts[pid][key] for key in EVENT_COUNTS},
            "minutes": float(np.isfinite(sampled[:, col, 0]).sum() / fps / 60),
            "distance_km": float(np.nansum(s) / fps / 1000),
            "sprints": len(kinematics.sprints(s, fps, threshold=kinematics.SPRINT_SPEED)),
            "top_speed_kmh": float(smooth.max() * 3.6) if len(smooth) else 0.0,
        }
    return out


def match_stats(match: Match, phases: list[Phase]):
    return team_stats(match, phases), player_stats(match)


def facts(match: Match, team_s: dict, player_s: dict, v: PhaseView, involved: list[str]) -> list[Draft]:
    out = []
    for team in (v.attack, v.defence):
        s = team_s[team]
        parts = [
            f"{s['possession_pct']:.0f}% of possession time",
            f"shots {s['shots']}, goals {s['goals']}",
            f"{s['high_recovery_pct']:.0f}% of ball recoveries in the opponent half",
        ]
        values = {
            "possession_pct": round(s["possession_pct"]),
            "shots": s["shots"],
            "goals": s["goals"],
            "high_recovery_pct": round(s["high_recovery_pct"]),
        }
        if np.isfinite(s["line_height_m"]):
            parts.append(f"back line averaged {s['line_height_m']:.1f} m from goal out of possession")
            values["line_height_m"] = r1(s["line_height_m"])
        if np.isfinite(s["width_m"]):
            parts.append(f"outfield averaged {s['width_m']:.1f} m wide in possession")
            values["width_m"] = r1(s["width_m"])
        out.append(Draft("context", 0, f"Match context, {TEAM[team]}: " + "; ".join(parts) + ".", (), values))
    for pid in involved[:MAX_PLAYERS]:
        s = player_s.get(pid)
        if s is None:
            continue
        text = (
            f"Match context, {pid} ({TEAM[match.team_of(pid)]}): {s['minutes']:.0f} minutes tracked; "
            f"passes completed {s['passes']} (forward {s['forward_passes']}); ball lost {s['ball_lost']}; "
            f"shots {s['shots']} (goals {s['goals']}); duels won {s['duels_won']}, lost {s['duels_lost']}; "
            f"sprints {s['sprints']}; top sustained speed {s['top_speed_kmh']:.1f} km/h; "
            f"distance {s['distance_km']:.1f} km."
        )
        values = {key: s[key] for key in (*EVENT_COUNTS, "sprints")}
        values.update(minutes=round(s["minutes"]), top_speed_kmh=r1(s["top_speed_kmh"]), distance_km=r1(s["distance_km"]))
        out.append(Draft("context", 0, text, (pid,), values))
    return out
