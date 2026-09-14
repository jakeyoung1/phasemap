import math

import numpy as np
import pytest

from phasemap import pitch


def test_attacking_frame_turns_half_circle():
    pts = np.array([[0.0, 0.0], [105.0, 68.0], [30.0, 10.0]])
    flipped = pitch.to_attacking_frame(pts, -1)
    assert np.allclose(flipped, [[105, 68], [0, 0], [75, 58]])
    assert np.allclose(pitch.to_attacking_frame(pts, 1), pts)
    assert flipped is not pts


def test_attacking_frame_rejects_bad_direction():
    with pytest.raises(ValueError):
        pitch.to_attacking_frame(np.zeros((1, 2)), 0)


@pytest.mark.parametrize(
    "y, expected",
    [
        (66, "left wing"),
        (50, "left half-space"),
        (34, "central"),
        (20, "right half-space"),
        (2, "right wing"),
    ],
)
def test_lanes_from_attacker_view(y, expected):
    assert pitch.lane(y) == expected


def test_zone_labels():
    assert pitch.zone(95, 34) == "penalty box"
    assert pitch.zone(80, 34) == "zone 14"
    assert pitch.zone(80, 60) == "final third, left wing"
    assert pitch.zone(10, 34) == "defensive third, central"


def test_goal_view_from_penalty_spot():
    dist, angle = pitch.goal_view(105 - 11, 34)
    assert dist == pytest.approx(11.0)
    assert angle == pytest.approx(math.degrees(2 * math.atan(3.66 / 11)))
