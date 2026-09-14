import numpy as np
import pytest

from phasemap import shape


def test_team_shape_square():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [np.nan, np.nan]], float)
    s = shape.team_shape(pts)
    assert s.width == pytest.approx(10)
    assert s.depth == pytest.approx(10)
    assert s.area == pytest.approx(100)
    assert s.centroid == pytest.approx((5, 5))


def test_team_shape_collinear_has_zero_area():
    pts = np.array([[0, 0], [5, 0], [10, 0]], float)
    assert shape.team_shape(pts).area == 0.0


def test_team_shape_needs_three_players():
    assert shape.team_shape(np.array([[0, 0], [1, 1]], float)) is None


def test_back_line_orders_left_to_right_and_finds_widest_gap():
    ids = ["A2", "A3", "A4", "A5", "A8"]
    pts = np.array(
        [
            [90, 60],  # left back
            [92, 40],  # left centre back
            [91, 24],  # right centre back
            [89, 8],  # right back
            [75, 34],  # midfielder, not in the back four
        ],
        float,
    )
    line = shape.back_line(pts, ids)
    assert line.ids == ("A2", "A3", "A4", "A5")
    assert line.widest_pair == ("A2", "A3")
    assert line.widest_gap == pytest.approx(np.hypot(2, 20))
    assert line.height == pytest.approx(105 - 90.5)
    assert line.last_defender_x == 92


def test_back_line_ignores_missing_players():
    ids = ["A1", "A2", "A3", "A4", "A5"]
    pts = np.array([[90, 60], [np.nan, np.nan], [91, 24], [89, 8], [88, 40]], float)
    line = shape.back_line(pts, ids)
    assert "A2" not in line.ids
    assert shape.back_line(pts[:3], ids[:3]) is None
