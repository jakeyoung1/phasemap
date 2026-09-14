"""On-ball facts: how possession started, passes, carries, options not taken, the shot, tempo."""

from __future__ import annotations

import numpy as np

from . import kinematics, pitch, space
from .quality import MERGED_M
from .view import TEAM, Draft, PhaseView, finite, r1, unique

MIN_CARRY_M = 5.0
MIN_CARRY_S = 0.5
OPEN_SPACE_M = 3.0
DECISIONS_WITH_OPTIONS = 3
OPTIONS_PER_DECISION = 3
MAX_BALL_SPEED_MS = 45.0
POSTS = (
    (pitch.LENGTH, pitch.WIDTH / 2 - pitch.GOAL_WIDTH / 2),
    (pitch.LENGTH, pitch.WIDTH / 2 + pitch.GOAL_WIDTH / 2),
)


def _in_triangle(p, a, b, c) -> bool:
    def side(o, u, w):
        return (u[0] - o[0]) * (w[1] - o[1]) - (u[1] - o[1]) * (w[0] - o[0])

    d = (side(a, b, p), side(b, c, p), side(c, a, p))
    return not (min(d) < 0 < max(d))


def possession_start(v: PhaseView) -> list[Draft]:
    first = v.phase.events[0]
    i = v.won_i
    ball = v.action_point(first.player, i, first.start_xy)
    if not finite(ball):
        return []
    dxy = v.xy[i, v.cols(v.defence, outfield=True)]
    present = np.isfinite(dxy).all(axis=1)
    goal_side = int(np.count_nonzero(dxy[present, 0] > ball[0]))
    outfield = int(present.sum())
    dist, _ = pitch.goal_view(*ball)
    verb = "restarted play with a" if first.type == "SET PIECE" else "won the ball with a"
    text = (
        f"{first.player or TEAM[v.attack]} {verb} {first.label} in {pitch.zone(*ball)}, {dist:.1f} m from goal. "
        f"{goal_side} of {TEAM[v.defence]}'s {outfield} outfield players were goal-side of the ball."
    )
    values = {"goal_distance_m": r1(dist), "goal_side": goal_side, "outfield": outfield}
    return [Draft("possession start", i, text, unique(first.player), values)]


def passes(v: PhaseView) -> list[Draft]:
    out = []
    everyone = v.cols(v.defence)
    outfield = v.cols(v.defence, outfield=True)
    for ev in v.phase.passes:
        i0, i1 = v.i(ev.start_frame), v.i(ev.end_frame)
        start = v.action_point(ev.player, i0, ev.start_xy)
        end = v.action_point(ev.receiver, i1, ev.end_xy)
        if not (finite(start) and finite(end)):
            continue
        length = float(np.hypot(*(end - start)))
        progress = float(end[0] - start[0])
        travel = max(ev.end_frame - ev.start_frame, 1) / v.fps
        speed = length / travel * 3.6
        heading = "forward" if progress >= 5 else "backward" if progress <= -5 else "square"
        kind = f" ({ev.subtype.lower()})" if ev.subtype else ""
        text = (
            f"{ev.player} passed{kind} to {ev.receiver or 'space'}: {length:.1f} m {heading} "
            f"({progress:+.1f} m toward goal), {pitch.zone(*start)} to {pitch.zone(*end)}, "
            f"{travel:.1f} s of ball travel at an average {speed:.1f} km/h."
        )
        values = {"length_m": r1(length), "progress_m": r1(progress), "travel_s": r1(travel), "speed_kmh": r1(speed)}
        players = [ev.player, ev.receiver]
        p_gap, p_near = v.nearest(start, i0, everyone)
        if p_near:
            text += f" Nearest defender to {ev.player} at release: {p_near}, {p_gap:.1f} m."
            values["passer_space_m"] = r1(p_gap)
            players.append(p_near)
        r_gap, r_near = v.nearest(end, i1, everyone)
        if r_near and ev.receiver:
            text += f" {ev.receiver} received it with {r_near} {r_gap:.1f} m away."
            values["receiver_space_m"] = r1(r_gap)
            players.append(r_near)
        with np.errstate(invalid="ignore"):
            xs = v.xy[i0, outfield, 0]
            bypassed = int(np.count_nonzero((xs > start[0]) & (xs < end[0])))
        values["bypassed"] = bypassed
        if bypassed:
            text += (
                f" {bypassed} {TEAM[v.defence]} outfield player{'' if bypassed == 1 else 's'} stood between "
                "passer and receiver along the length of the pitch."
            )
        lane = [v.ids(outfield)[k] for k in space.lane_blockers(start, end, v.xy[i0, outfield])]
        values["lane_defenders"] = len(lane)
        if lane:
            text += f" Straight-line passing lane contested by {', '.join(lane)}."
            players += lane
        out.append(Draft("pass", i0, text, unique(*players), values))
    return out


def carries(v: PhaseView) -> list[Draft]:
    out = []
    for span in v.spans:
        a, b = span.arrive, span.release
        secs = (b - a) / v.fps
        col = v.match.col(span.player)
        p0, p1 = v.xy[a, col], v.xy[b, col]
        if secs < MIN_CARRY_S or not (finite(p0) and finite(p1)):
            continue
        dist = float(np.hypot(*(p1 - p0)))
        if dist < MIN_CARRY_M:
            continue
        progress = float(p1[0] - p0[0])
        text = (
            f"{span.player} carried the ball {dist:.1f} m ({progress:+.1f} m toward goal) in {secs:.1f} s, "
            f"{pitch.zone(*p0)} to {pitch.zone(*p1)}"
        )
        values = {"distance_m": r1(dist), "progress_m": r1(progress), "seconds": r1(secs)}
        spd = kinematics.speed(v.vel[a : b + 1, col])
        if np.isfinite(spd).any():
            top = float(np.nanmax(spd)) * 3.6
            text += f", reaching {top:.1f} km/h"
            values["top_speed_kmh"] = r1(top)
        text += f", before the {span.action.label}."
        out.append(Draft("carry", a, text, (span.player,), values))
    return out


def options(v: PhaseView) -> list[Draft]:
    """Teammates the ball carrier did not pick at each of the final decisions."""
    out = []
    mates = v.cols(v.attack, outfield=True)
    opponents = v.cols(v.defence)
    opponent_ids = v.ids(opponents)
    for ev in v.actions[-DECISIONS_WITH_OPTIONS:]:
        if ev.player is None:
            continue
        i = v.i(ev.start_frame)
        origin = v.action_point(ev.player, i, ev.start_xy)
        if not finite(origin):
            continue
        rows = []
        for col, mate in zip(mates, v.ids(mates)):
            spot = v.xy[i, col]
            if mate in (ev.player, ev.receiver) or not finite(spot):
                continue
            gap, marker = v.nearest(spot, i, opponents)
            if marker is None:
                continue
            lane = [opponent_ids[k] for k in space.lane_blockers(origin, spot, v.xy[i, opponents])]
            ahead = float(spot[0] - origin[0])
            rows.append((not lane and gap >= OPEN_SPACE_M, ahead, gap, mate, spot, lane, marker))
        rows.sort(key=lambda r: (not r[0], -min(r[1], 30.0), -r[2]))
        chosen = "shot" if ev.type == "SHOT" else f"passed to {ev.receiver}"
        shown: list[np.ndarray] = []
        for _, ahead, gap, mate, spot, lane, marker in rows:
            if len(shown) == OPTIONS_PER_DECISION:
                break
            if any(np.hypot(*(spot - s)) < MERGED_M for s in shown):
                continue  # two tracks on one spot; the data-quality fact covers it
            shown.append(spot)
            dist = float(np.hypot(*(spot - origin)))
            goal_dist, _ = pitch.goal_view(*spot)
            lane_text = "clear" if not lane else "contested by " + ", ".join(lane)
            text = (
                f"Option not taken when {ev.player} {chosen}: {mate}, {dist:.1f} m away and "
                f"{abs(ahead):.1f} m {'ahead of' if ahead >= 0 else 'behind'} the ball, in {pitch.zone(*spot)}, "
                f"{goal_dist:.1f} m from goal, nearest defender {marker} at {gap:.1f} m, "
                f"straight-line passing lane {lane_text}."
            )
            values = {
                "distance_m": r1(dist),
                "ahead_m": r1(ahead),
                "goal_distance_m": r1(goal_dist),
                "space_m": r1(gap),
                "lane_defenders": len(lane),
            }
            out.append(Draft("option", i, text, unique(ev.player, mate, marker, *lane, ev.receiver), values))
    return out


def shot(v: PhaseView) -> list[Draft]:
    ev = v.phase.shot
    if ev is None or ev.player is None:
        return []
    i = v.i(ev.start_frame)
    origin = v.action_point(ev.player, i, ev.start_xy)
    if not finite(origin):
        return []
    dist, angle = pitch.goal_view(*origin)
    header = "HEAD" in ev.subtype.split("-")
    text = (
        f"{ev.player} {'headed' if header else 'shot'} from {dist:.1f} m in {pitch.zone(*origin)}, "
        f"with {angle:.1f}° of goal mouth in view. Outcome: {v.phase.outcome}."
    )
    values = {"distance_m": r1(dist), "angle_deg": r1(angle)}
    players = [ev.player]
    outfield = v.cols(v.defence, outfield=True)
    gap, marker = v.nearest(origin, i, outfield)
    if marker:
        text += f" Nearest outfield defender: {marker} at {gap:.1f} m."
        values["defender_space_m"] = r1(gap)
        players.append(marker)
    blocking = [
        pid for pid, spot in zip(v.ids(outfield), v.xy[i, outfield]) if finite(spot) and _in_triangle(spot, origin, *POSTS)
    ]
    text += f" {len(blocking)} outfield defenders inside the triangle between ball and posts"
    text += f" ({', '.join(blocking)})." if blocking else "."
    values["triangle_defenders"] = len(blocking)
    players += blocking
    keeper = v.keeper(v.defence)
    spot = v.pos(keeper, i)
    if finite(spot):
        off_line = pitch.LENGTH - float(spot[0])
        to_ball = float(np.hypot(*(spot - origin)))
        text += f" Keeper {keeper} was {off_line:.1f} m off the goal line and {to_ball:.1f} m from the ball."
        values.update(keeper_off_line_m=r1(off_line), keeper_to_ball_m=r1(to_ball))
        players.append(keeper)
    flight = v.ball[i : i + int(0.6 * v.fps) + 1]
    if len(flight) >= 3:
        ball_speed = kinematics.speed(kinematics.velocity(flight, v.fps, max_speed=None))
        if np.isfinite(ball_speed).any() and np.nanmax(ball_speed) < MAX_BALL_SPEED_MS:
            top = float(np.nanmax(ball_speed)) * 3.6
            text += f" Tracked ball speed peaked at {top:.1f} km/h."
            values["ball_speed_kmh"] = r1(top)
    return [Draft("shot", i, text, unique(*players), values)]


def tempo(v: PhaseView) -> list[Draft]:
    first, last = v.phase.events[0], v.last_action
    start = v.action_point(first.player, v.won_i, first.start_xy)
    end = v.action_point(last.player, v.final_i, last.start_xy)
    if not (finite(start) and finite(end)):
        return []
    secs = (v.final_i - v.won_i) / v.fps
    count = len(v.phase.passes)
    progress = float(end[0] - start[0])
    text = (
        f"{TEAM[v.attack]} took {secs:.1f} s and {count} pass{'' if count == 1 else 'es'} from the {first.label} "
        f"to the {last.label}, moving the ball {progress:+.1f} m toward goal. Outcome: {v.phase.outcome}."
    )
    return [Draft("tempo", v.final_i, text, (), {"seconds": r1(secs), "passes": count, "progress_m": r1(progress)})]
