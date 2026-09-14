import numpy as np
import pytest

from phasemap.ingest import metrica
from phasemap.match import Match

HOME = """,,,Home,,Home,,,
,,,11,,1,,,
Period,Frame,Time [s],Player11,,Player1,,Ball,
1,1,0.04,0.95,0.5,0.40,0.25,0.5,0.5
1,2,0.08,0.95,0.5,0.41,0.25,NaN,NaN
2,3,0.12,0.05,0.5,0.60,0.75,0.5,0.5
2,4,0.16,0.05,0.5,NaN,NaN,0.52,0.5
"""

AWAY = """,,,Away,,Away,,,
,,,25,,26,,,
Period,Frame,Time [s],Player25,,Player 26,,Ball,
1,1,0.04,0.05,0.5,0.6,0.5,0.5,0.5
1,2,0.08,0.05,0.5,0.6,0.5,0.49,0.51
2,3,0.12,0.95,0.5,0.4,0.5,0.5,0.5
2,4,0.16,0.95,0.5,0.4,0.5,0.52,0.5
"""

EVENTS = """Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y
Home,PASS,,1,1,0.04,2,0.08,Player1,Player11,0.4,0.25,0.9,0.5
Away,SHOT,HEAD-ON TARGET-GOAL,2,3,0.12,4,0.16,Player 26,,0.4,0.5,-0.01,0.5
"""


@pytest.fixture
def match(tmp_path):
    folder = tmp_path / "Sample_Game_9"
    folder.mkdir()
    files = metrica.game_files("Sample_Game_9")
    (folder / files["home"]).write_text(HOME)
    (folder / files["away"]).write_text(AWAY)
    (folder / files["events"]).write_text(EVENTS)
    return metrica.load(folder)


def test_players_and_coordinates(match):
    assert match.player_ids == ["H11", "H1", "A25", "A26"]
    assert match.player_teams == ["home", "home", "away", "away"]
    assert match.xy.shape == (4, 4, 2)
    assert np.allclose(match.xy[0, 1], [42.0, 51.0])
    assert np.isnan(match.xy[3, 1]).all()
    assert list(match.frames) == [1, 2, 3, 4]
    assert list(match.periods) == [1, 1, 2, 2]


def test_ball_falls_back_to_away_file(match):
    assert np.allclose(match.ball[1], [0.49 * 105, 0.49 * 68])


def test_events(match):
    first, goal = match.events
    assert first.player == "H1" and first.receiver == "H11" and first.subtype == ""
    assert first.label == "pass"
    assert goal.player == "A26" and goal.receiver is None
    assert goal.is_goal
    assert goal.end_xy == (-1.05, 34.0)


def test_orientation_and_keepers(match):
    assert match.attack_dir == {("home", 1): -1, ("away", 1): 1, ("home", 2): 1, ("away", 2): -1}
    assert match.goalkeepers[("home", 1)] == "H11"
    assert match.goalkeepers[("away", 2)] == "A25"


def test_clock_and_score(match):
    assert match.clock_label(3) == "45:00"
    assert match.score_before(3) == {"home": 0, "away": 0}
    assert match.score_before(4) == {"home": 0, "away": 1}


def test_frames_must_be_contiguous():
    with pytest.raises(ValueError):
        Match(
            name="x", source="x", fps=25.0, frames=np.array([1, 3]), periods=np.array([1, 1]),
            clock_s=np.zeros(2), player_ids=[], player_teams=[], xy=np.zeros((2, 0, 2)),
            ball=np.zeros((2, 2)), events=[],
        )
