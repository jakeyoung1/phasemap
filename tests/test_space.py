import numpy as np
import pytest

from phasemap import space


def test_nearest_skips_missing_players():
    others = np.array([[np.nan, np.nan], [3, 4], [10, 0]], float)
    dist, idx = space.nearest((0, 0), others)
    assert dist == pytest.approx(5)
    assert idx == 1


def test_nearest_with_nobody():
    dist, idx = space.nearest((0, 0), np.full((2, 2), np.nan))
    assert np.isnan(dist) and idx == -1


def test_count_within():
    others = np.array([[1, 0], [0, 2.5], [np.nan, np.nan], [6, 0]], float)
    assert space.count_within((0, 0), others, 3.0) == 2


def test_lane_blockers():
    opponents = np.array(
        [
            [60, 35],  # on the line, halfway along
            [60, 40],  # 6 m off the line
            [70.5, 34],  # level with the receiver: marking, not blocking
            [np.nan, np.nan],
            [52, 36.5],  # near the passer, 2.5 m off the line (allowed 2.1 m)
        ],
        float,
    )
    assert space.lane_blockers((50, 34), (70, 34), opponents) == [0]


def test_control_grid_favours_closer_team():
    grid = space.control_grid(
        att_pos=np.array([[80.0, 34.0]]),
        att_vel=np.zeros((1, 2)),
        def_pos=np.array([[20.0, 34.0]]),
        def_vel=np.zeros((1, 2)),
    )
    assert grid.prob.shape == (34, 53)
    assert grid.at(80, 34) > 0.95
    assert grid.at(20, 34) < 0.05
    row = grid.prob[np.abs(grid.ys - 34).argmin()]
    assert np.all(np.diff(row) >= -1e-12)
    assert 0.2 < grid.share(x_min=40, x_max=65) < 0.8


def test_running_start_wins_space_ahead():
    att, dfn = np.array([[50.0, 34.0]]), np.array([[60.0, 34.0]])
    still = space.control_grid(att, np.zeros((1, 2)), dfn, np.zeros((1, 2)))
    running = space.control_grid(att, np.array([[7.0, 0.0]]), dfn, np.zeros((1, 2)))
    assert running.at(58, 34) > still.at(58, 34)
