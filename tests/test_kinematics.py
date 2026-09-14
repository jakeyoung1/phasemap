import numpy as np
import pytest

from phasemap import kinematics as kin

FPS = 25.0


def test_constant_speed_line():
    t = np.arange(100) / FPS
    xy = np.stack([5.0 * t, np.full_like(t, 30.0)], axis=1)
    vel = kin.velocity(xy, FPS)
    assert np.allclose(kin.speed(vel), 5.0)
    assert kin.distance_covered(vel, FPS) == pytest.approx(20.0, abs=0.3)


def test_gaps_split_runs_and_short_runs_use_gradient():
    t = np.arange(40) / FPS
    xy = np.stack([3.0 * t, np.zeros_like(t)], axis=1)
    xy[10:13] = np.nan
    xy[13:16] = [[100.0, 0.0], [100.12, 0.0], [100.24, 0.0]]
    xy[16] = np.nan
    vel = kin.velocity(xy, FPS)
    assert np.isnan(vel[10:13]).all() and np.isnan(vel[16]).all()
    assert np.allclose(kin.speed(vel[13:16]), 3.0)
    assert np.allclose(kin.speed(vel[:10]), 3.0)
    assert np.allclose(kin.speed(vel[17:]), 3.0)


def test_glitch_jump_is_discarded():
    xy = np.zeros((30, 2))
    xy[15:] = [5.0, 0.0]
    vel = kin.velocity(xy, FPS)
    assert np.isnan(vel[15]).all()
    assert np.allclose(kin.speed(vel[:8]), 0.0)


def test_multi_player_shape_is_kept():
    xy = np.zeros((20, 3, 2))
    xy[:, 1, 0] = np.arange(20) * 0.2  # 5 m/s
    vel = kin.velocity(xy, FPS)
    assert vel.shape == (20, 3, 2)
    assert np.allclose(kin.speed(vel)[:, 1], 5.0)
    assert np.allclose(kin.distance_covered(vel, FPS), [0.0, 4.0, 0.0])


def test_sprints_need_duration():
    spd = np.zeros(100)
    spd[10:40] = 7.5  # 1.2 s
    spd[60:80] = 8.0  # 0.8 s
    spd[85] = np.nan
    assert kin.sprints(spd, FPS) == [(10, 40)]
