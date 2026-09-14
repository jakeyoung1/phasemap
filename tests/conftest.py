import numpy as np
import pytest

from phasemap.match import Event, Match

FPS = 25.0
N = 200


def _static(x, y):
    return np.tile([float(x), float(y)], (N, 1))


def _path(keys):
    """Piecewise-linear track through (row, x, y) keyframes, held still outside them."""
    rows = np.arange(N)
    frames = [k[0] for k in keys]
    return np.stack(
        [np.interp(rows, frames, [k[1] for k in keys]), np.interp(rows, frames, [k[2] for k in keys])], axis=1
    )


def _event(index, type_, subtype, start, end, player, receiver, start_xy, end_xy):
    return Event(
        index=index, team="home", type=type_, subtype=subtype, period=1, start_frame=start, end_frame=end,
        start_s=start / FPS, end_s=end / FPS, player=player, receiver=receiver, start_xy=start_xy, end_xy=end_xy,
    )


@pytest.fixture
def through_ball():
    """Home wins the ball, plays a through ball to a runner who carries into the box and scores.

    Away's right-sided defender A4 drifts wide before the pass, opening the gap to A2.
    Frame f sits in tracking row f - 1.
    """
    tracks = {
        "H1": _static(5, 34),
        "H2": _static(50, 34),
        "H3": _path([(0, 60, 20), (50, 60, 20), (125, 84, 20), (175, 94, 30)]),
        "H4": _static(70, 50),
        "A1": _static(103, 34),
        "A2": _static(80, 30),
        "A3": _static(80, 45),
        "A4": _path([(0, 78, 15), (60, 78, 15), (110, 78, 5)]),
        "A5": _static(82, 58),
        "A6": _static(60, 34),
    }
    ids = list(tracks)
    events = [
        _event(0, "RECOVERY", "INTERCEPTION", 11, 11, "H2", None, (50.0, 34.0), None),
        _event(1, "PASS", "", 101, 126, "H2", "H3", (50.0, 34.0), (84.0, 20.0)),
        _event(2, "SHOT", "ON TARGET-GOAL", 176, 186, "H3", None, (94.0, 30.0), (105.0, 34.0)),
    ]
    return Match(
        name="Through ball", source="synthetic", fps=FPS, frames=np.arange(1, N + 1), periods=np.ones(N, dtype=int),
        clock_s=np.arange(1, N + 1) / FPS, player_ids=ids, player_teams=["home" if p[0] == "H" else "away" for p in ids],
        xy=np.stack([tracks[p] for p in ids], axis=1), ball=np.full((N, 2), np.nan), events=events,
    )
