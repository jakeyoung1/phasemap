import numpy as np

from phasemap import phases
from phasemap.match import Event, Match


def ev(team, type_, subtype="", period=1, frame=0):
    return Event(
        index=frame, team=team, type=type_, subtype=subtype, period=period,
        start_frame=frame, end_frame=frame + 5, start_s=frame / 25, end_s=(frame + 5) / 25,
        player=None, receiver=None, start_xy=None, end_xy=None,
    )


EVENTS = [
    ev("home", "SET PIECE", "KICK OFF", frame=1),
    ev("home", "PASS", frame=10),
    ev("away", "CHALLENGE", "GROUND-WON", frame=20),
    ev("home", "BALL LOST", "INTERCEPTION", frame=30),
    ev("away", "RECOVERY", "INTERCEPTION", frame=31),
    ev("away", "PASS", frame=40),
    ev("away", "PASS", frame=50),
    ev("away", "SHOT", "ON TARGET-GOAL", frame=60),
    ev("home", "SET PIECE", "KICK OFF", frame=100),
    ev("home", "PASS", frame=110),
    ev("away", "PASS", frame=120),
    ev("away", "PASS", period=2, frame=200),
    ev("away", "SHOT", "ON TARGET-SAVED", period=2, frame=210),
]


def test_segment_boundaries_and_outcomes():
    result = phases.segment(EVENTS)
    assert [(p.team, p.period, len(p.events), p.outcome) for p in result] == [
        ("home", 1, 3, "turnover"),
        ("away", 1, 4, "goal"),
        ("home", 1, 2, "possession lost"),
        ("away", 1, 1, "period ended"),
        ("away", 2, 2, "shot saved"),
    ]
    goal = result[1]
    assert goal.start_kind == "recovery (interception)"
    assert len(goal.passes) == 2
    assert goal.shot.subtype == "ON TARGET-GOAL"
    assert goal.start_frame == 31 and goal.end_frame == 65


def test_notable_puts_goals_first():
    result = phases.segment(EVENTS)
    assert [p.outcome for p in phases.notable(result)] == ["goal", "shot saved"]
    assert len(phases.notable(result, limit=1)) == 1


def tiny_match(periods):
    n = len(periods)
    return Match(
        name="t", source="t", fps=25.0, frames=np.arange(1, n + 1), periods=np.array(periods),
        clock_s=np.arange(n) / 25, player_ids=[], player_teams=[], xy=np.zeros((n, 0, 2)),
        ball=np.zeros((n, 2)), events=[],
    )


def test_window_stays_inside_period():
    match = tiny_match([1] * 300 + [2] * 300)
    phase = phases.Phase(id=0, team="home", period=2, events=[ev("home", "PASS", period=2, frame=310)])
    assert phases.window(match, phase) == (300, 309 + 5 + 37 + 1)


def test_window_caps_long_possessions():
    match = tiny_match([1] * 300 + [2] * 300)
    phase = phases.Phase(
        id=0, team="home", period=1, events=[ev("home", "PASS", frame=20), ev("home", "PASS", frame=285)]
    )
    assert phases.window(match, phase, max_s=4.0) == (300 - int(5.5 * 25), 300)
