from phasemap import phases, quality
from phasemap.view import PhaseView


def test_merged_tracks_are_flagged(through_ball):
    h4, a3 = through_ball.col("H4"), through_ball.col("A3")
    through_ball.xy[20:80, h4] = through_ball.xy[20:80, a3]
    phase = phases.segment(through_ball.events)[0]
    drafts = quality.merged_tracks(PhaseView.build(through_ball, phase))
    assert len(drafts) == 1
    assert drafts[0].players == ("H4", "A3")
    assert drafts[0].values == {"seconds": 2.4}
    assert drafts[0].i == 20


def test_clean_tracks_raise_nothing(through_ball):
    phase = phases.segment(through_ball.events)[0]
    assert quality.merged_tracks(PhaseView.build(through_ball, phase)) == []
