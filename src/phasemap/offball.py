"""Off-ball facts: back-line gaps, defenders shifted across, runs, recovery, shape, space."""

from __future__ import annotations

import numpy as np

from . import kinematics, pitch, shape, space
from .view import TEAM, Draft, PhaseView, finite, r1, unique

SAMPLE_EVERY = 5  # frames between shape samples (0.2 s at 25 fps)
GAP_REPORT_M = 12.0
GAP_GROWTH_M = 4.0
SHIFT_REPORT_M = 8.0
SHIFT_WINDOW_S = 3.0
MAX_SHIFTS = 2
RUN_SPEED = kinematics.HIGH_SPEED
RUN_MIN_S = 1.0
RUN_LEAD_S = 2.0
MAX_RUNS = 5
MAX_RECOVERY_RUNS = 3
RECOVERY_MIN_M = 5.0
BOX_Y = (pitch.WIDTH / 2 - pitch.BOX_WIDTH / 2, pitch.WIDTH / 2 + pitch.BOX_WIDTH / 2)


def _line_at(v: PhaseView, i: int) -> shape.BackLine | None:
    cols = v.cols(v.defence, outfield=True)
    return shape.back_line(v.xy[i, cols], v.ids(cols))


def _fast_stretches(v: PhaseView, col: int, lo: int, hi: int, holding: np.ndarray | None = None):
    """Window index ranges [a, b) where a player ran at high speed for long enough."""
    spd = kinematics.speed(v.vel[lo:hi, col])
    if holding is not None:
        spd = np.where(holding[lo:hi], 0.0, spd)
    return [(lo + a, lo + b) for a, b in kinematics.sprints(spd, v.fps, threshold=RUN_SPEED, min_seconds=RUN_MIN_S)]


def back_line(v: PhaseView) -> list[Draft]:
    samples = [(i, _line_at(v, i)) for i in range(v.won_i, v.final_i + 1, SAMPLE_EVERY)]
    samples = [(i, line) for i, line in samples if line is not None]
    if not samples:
        return []
    out = []
    side = TEAM[v.defence]
    first = samples[0][1]
    iw, widest = max(samples, key=lambda s: s[1].widest_gap)
    if widest.widest_gap >= GAP_REPORT_M and widest.widest_gap - first.widest_gap >= GAP_GROWTH_M:
        a0, b0 = first.widest_pair
        a1, b1 = widest.widest_pair
        text = (
            f"{side}'s back line: widest gap {first.widest_gap:.1f} m ({a0}-{b0}) when possession started, "
            f"growing to {widest.widest_gap:.1f} m between {a1} and {b1}."
        )
        values = {"gap_start_m": r1(first.widest_gap), "gap_max_m": r1(widest.widest_gap)}
        out.append(Draft("back-line gap", iw, text, unique(a0, b0, a1, b1), values))
    end = _line_at(v, v.final_i)
    if end is not None:
        a, b = end.widest_pair
        text = (
            f"At the final action {side}'s back line was {end.height:.1f} m from its goal line "
            f"({first.height:.1f} m when possession started); widest gap {end.widest_gap:.1f} m between {a} and {b}."
        )
        values = {"height_end_m": r1(end.height), "height_start_m": r1(first.height), "gap_m": r1(end.widest_gap)}
        middle = (v.pos(a, v.final_i) + v.pos(b, v.final_i)) / 2
        dist, runner = v.nearest(middle, v.final_i, v.cols(v.attack, outfield=True))
        if runner:
            text += f" Nearest {TEAM[v.attack]} player to the middle of that gap: {runner}, {dist:.1f} m."
            values["attacker_to_gap_m"] = r1(dist)
        out.append(Draft("back line", v.final_i, text, unique(a, b, runner), values))
    fp = v.final_pass
    if fp is not None and fp.receiver:
        i = v.i(fp.start_frame)
        line = _line_at(v, i)
        spot = v.action_point(fp.receiver, i, None)
        if line is not None and finite(spot):
            beyond = float(spot[0] - line.last_defender_x)
            where = "beyond" if beyond > 0 else "short of"
            text = (
                f"When {fp.player} released the final pass, {fp.receiver} was {abs(beyond):.1f} m {where} "
                f"{side}'s deepest outfield defender."
            )
            out.append(Draft("line", i, text, unique(fp.player, fp.receiver), {"beyond_line_m": r1(beyond)}))
    return out


def shifts(v: PhaseView) -> list[Draft]:
    """Defenders who moved furthest across the pitch within a few seconds."""
    cols = v.cols(v.defence, outfield=True)
    idx = np.arange(max(0, v.won_i - int(RUN_LEAD_S * v.fps)), v.final_i + 1, SAMPLE_EVERY)
    if len(idx) < 2:
        return []
    ys = v.xy[idx][:, cols, 1]
    horizon = max(1, int(SHIFT_WINDOW_S * v.fps / SAMPLE_EVERY))
    found = []
    for k, pid in enumerate(v.ids(cols)):
        best = None  # (shift, -duration, start sample, end sample): biggest shift, then quickest
        for s in range(len(idx) - 1):
            if not np.isfinite(ys[s, k]):
                continue
            ahead = np.abs(ys[s + 1 : s + 1 + horizon, k] - ys[s, k])
            if not np.isfinite(ahead).any():
                continue
            j = int(np.nanargmax(ahead))
            candidate = (round(float(ahead[j]), 1), -(j + 1), s, s + 1 + j)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is not None and best[0] >= SHIFT_REPORT_M:
            found.append((best[0], pid, int(idx[best[2]]), int(idx[best[3]])))
    found.sort(key=lambda f: -f[0])
    out = []
    for shift, pid, a, b in found[:MAX_SHIFTS]:
        col = v.match.col(pid)
        mates = cols[cols != col]
        p0, p1 = v.xy[a, col], v.xy[b, col]
        g0, _ = space.nearest(p0, v.xy[a, mates])
        g1, _ = space.nearest(p1, v.xy[b, mates])
        if not (np.isfinite(g0) and np.isfinite(g1)):
            continue
        secs = (b - a) / v.fps
        text = (
            f"{pid} shifted {shift:.1f} m across the pitch in {secs:.1f} s, {pitch.zone(*p0)} to {pitch.zone(*p1)}; "
            f"distance to the nearest {TEAM[v.defence]} outfield teammate went from {g0:.1f} m to {g1:.1f} m."
        )
        values = {"shift_m": shift, "seconds": r1(secs), "teammate_gap_start_m": r1(g0), "teammate_gap_end_m": r1(g1)}
        out.append(Draft("defender shift", b, text, (pid,), values))
    return out


def runs(v: PhaseView) -> list[Draft]:
    """High-speed runs by attackers without the ball, runs into the final third first."""
    lo, hi = max(0, v.won_i - int(RUN_LEAD_S * v.fps)), v.final_i + 1
    everyone = v.cols(v.defence)
    cols = v.cols(v.attack, outfield=True)
    found = []
    for col, pid in zip(cols, v.ids(cols)):
        holding = np.zeros(v.n, dtype=bool)
        for span in v.spans:
            if span.player == pid:
                holding[span.arrive : span.release + 1] = True
        for a, b in _fast_stretches(v, col, lo, hi, holding):
            p0, p1 = v.xy[a, col], v.xy[b - 1, col]
            if finite(p0) and finite(p1):
                found.append((bool(p1[0] >= 2 * pitch.LENGTH / 3), float(np.hypot(*(p1 - p0))), pid, col, a, b))
    found.sort(key=lambda f: (not f[0], -f[1]))
    out = []
    for _, dist, pid, col, a, b in found[:MAX_RUNS]:
        p0, p1 = v.xy[a, col], v.xy[b - 1, col]
        top = float(np.nanmax(kinematics.speed(v.vel[a:b, col]))) * 3.6
        progress = float(p1[0] - p0[0])
        text = (
            f"{pid} ran {dist:.1f} m without the ball ({progress:+.1f} m toward goal), "
            f"{pitch.zone(*p0)} to {pitch.zone(*p1)}, reaching {top:.1f} km/h."
        )
        values = {"distance_m": r1(dist), "progress_m": r1(progress), "top_speed_kmh": r1(top)}
        m0, n0 = v.nearest(p0, a, everyone)
        m1, n1 = v.nearest(p1, b - 1, everyone)
        if n0 and n1:
            text += f" Nearest defender at the start: {n0}, {m0:.1f} m; at the end: {n1}, {m1:.1f} m."
            values.update(marker_start_m=r1(m0), marker_end_m=r1(m1))
        line = _line_at(v, b - 1)
        if line is not None and p1[0] > line.last_defender_x:
            beyond = float(p1[0] - line.last_defender_x)
            text += f" The run ended {beyond:.1f} m beyond {TEAM[v.defence]}'s deepest outfield defender."
            values["beyond_line_m"] = r1(beyond)
        out.append(Draft("run", a, text, unique(pid, n0, n1), values))
    return out


def recovery(v: PhaseView) -> list[Draft]:
    """Defenders sprinting back toward their own goal, and who was still caught upfield at the end."""
    cols = v.cols(v.defence, outfield=True)
    ids = v.ids(cols)
    found = []
    for col, pid in zip(cols, ids):
        for a, b in _fast_stretches(v, col, v.won_i, v.final_i + 1):
            p0, p1 = v.xy[a, col], v.xy[b - 1, col]
            if finite(p0) and finite(p1) and p1[0] - p0[0] >= RECOVERY_MIN_M:
                found.append((float(p1[0] - p0[0]), pid, col, a, b))
    found.sort(key=lambda f: -f[0])
    out = []
    for back, pid, col, a, b in found[:MAX_RECOVERY_RUNS]:
        p0, p1 = v.xy[a, col], v.xy[b - 1, col]
        dist = float(np.hypot(*(p1 - p0)))
        top = float(np.nanmax(kinematics.speed(v.vel[a:b, col]))) * 3.6
        text = (
            f"{pid} ran back {dist:.1f} m ({back:.1f} m toward their own goal), "
            f"{pitch.zone(*p0)} to {pitch.zone(*p1)}, reaching {top:.1f} km/h."
        )
        values = {"distance_m": r1(dist), "toward_goal_m": r1(back), "top_speed_kmh": r1(top)}
        out.append(Draft("recovery run", a, text, (pid,), values))
    last = v.last_action
    ball = v.action_point(last.player, v.final_i, last.start_xy)
    # Inside the box nearly every defender is upfield of the ball, so the count says nothing there.
    if finite(ball) and not pitch.in_box(*ball):
        xy = v.xy[v.final_i, cols]
        present = np.isfinite(xy).all(axis=1)
        behind = [pid for pid, spot, ok in zip(ids, xy, present) if ok and spot[0] < ball[0]]
        text = (
            f"At the final action ({last.player}, {last.label}), {len(behind)} of {TEAM[v.defence]}'s "
            f"{int(present.sum())} outfield players were further from their own goal than the ball"
        )
        text += f": {', '.join(behind)}." if behind else "."
        values = {"behind_ball": len(behind), "outfield": int(present.sum())}
        out.append(Draft("behind the ball", v.final_i, text, unique(last.player, *behind), values))
    return out


def shapes(v: PhaseView) -> list[Draft]:
    out = []
    for moment, i in (("when possession started", v.won_i), ("at the final action", v.final_i)):
        dfn = shape.team_shape(v.xy[i, v.cols(v.defence, outfield=True)])
        att = shape.team_shape(v.xy[i, v.cols(v.attack, outfield=True)])
        if dfn is None or att is None:
            continue
        text = (
            f"Shape {moment}: {TEAM[v.defence]} outfield {dfn.width:.1f} m wide and {dfn.depth:.1f} m deep, "
            f"covering {dfn.area:.0f} m²; {TEAM[v.attack]} outfield {att.width:.1f} m wide and {att.depth:.1f} m deep."
        )
        values = {
            "defence_width_m": r1(dfn.width),
            "defence_depth_m": r1(dfn.depth),
            "defence_area_m2": round(dfn.area),
            "attack_width_m": r1(att.width),
            "attack_depth_m": r1(att.depth),
        }
        out.append(Draft("shape", i, text, (), values))
    return out


def _box_count(xy: np.ndarray) -> int:
    return sum(1 for x, y in xy if np.isfinite(x) and np.isfinite(y) and pitch.in_box(x, y))


def box_numbers(v: PhaseView) -> list[Draft]:
    out = []
    for moment, ev in (("final pass", v.final_pass), ("shot", v.phase.shot)):
        if ev is None:
            continue
        i = v.i(ev.start_frame)
        if ev.type == "PASS":
            target = v.action_point(ev.receiver, v.i(ev.end_frame), ev.end_xy)
        else:
            target = v.action_point(ev.player, i, ev.start_xy)
        if not finite(target) or target[0] < 2 * pitch.LENGTH / 3:
            continue
        att = _box_count(v.xy[i, v.cols(v.attack, outfield=True)])
        dfn = _box_count(v.xy[i, v.cols(v.defence, outfield=True)])
        text = (
            f"At the {moment} ({ev.player}), {att} {TEAM[v.attack]} players were in the penalty box "
            f"against {dfn} {TEAM[v.defence]} outfield players."
        )
        out.append(Draft("box numbers", i, text, unique(ev.player), {"attackers_in_box": att, "defenders_in_box": dfn}))
    return out


def control(v: PhaseView) -> list[Draft]:
    moments = [("possession start", v.won_i)]
    if v.final_pass is not None:
        moments.append(("final pass", v.i(v.final_pass.start_frame)))
    if v.phase.shot is not None:
        moments.append(("shot", v.i(v.phase.shot.start_frame)))
    att, dfn = v.cols(v.attack), v.cols(v.defence)
    out = []
    for moment, i in moments:
        grid = space.control_grid(v.xy[i, att], v.vel[i, att], v.xy[i, dfn], v.vel[i, dfn])
        box = grid.share(x_min=pitch.LENGTH - pitch.BOX_DEPTH, y_min=BOX_Y[0], y_max=BOX_Y[1]) * 100
        third = grid.share(x_min=2 * pitch.LENGTH / 3) * 100
        text = (
            f"At the {moment}, {TEAM[v.attack]} controlled {box:.0f}% of the penalty box and "
            f"{third:.0f}% of the final third (arrival-time pitch control)."
        )
        out.append(Draft("control", i, text, (), {"box_pct": round(box), "final_third_pct": round(third)}))
    return out
