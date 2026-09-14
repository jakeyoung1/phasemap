from pathlib import Path

import pytest

from phasemap import facts, phases
from phasemap.evidence import check_claim

GAME = Path(__file__).resolve().parents[1] / "data" / "metrica" / "Sample_Game_2"


def first(ledger, kind):
    return next(f for f in ledger if f.kind == kind)


def every(ledger, kind):
    return [f for f in ledger if f.kind == kind]


@pytest.fixture
def ledger(through_ball):
    found = phases.segment(through_ball.events)
    assert [p.outcome for p in found] == ["goal"]
    built, _ = facts.build(through_ball, found[0], found)
    return built


def test_every_fact_states_only_its_own_numbers(ledger):
    for fact in ledger:
        check = check_claim(fact.id, fact.text, [fact.id], ledger)
        assert check.status == "verified", (fact.text, check.problems)


def test_detectors_fire(ledger):
    kinds = {f.kind for f in ledger}
    expected = {
        "possession start", "pass", "carry", "option", "shot", "tempo", "back-line gap", "back line", "line",
        "defender shift", "run", "shape", "box numbers", "control", "context", "situation",
    }
    assert expected <= kinds, expected - kinds


def test_facts_are_time_ordered_and_numbered(ledger):
    timed = [f.t for f in ledger if f.frame is not None]
    assert timed == sorted(timed)
    assert [f.id for f in ledger] == [f"F{k}" for k in range(1, len(ledger) + 1)]


def test_pass_measurements(ledger):
    p = first(ledger, "pass").values
    assert p["length_m"] == pytest.approx(36.8, abs=0.1)
    assert p["progress_m"] == pytest.approx(34.0)
    assert p["bypassed"] == 5 and p["lane_defenders"] == 0
    assert p["passer_space_m"] == pytest.approx(10.0)
    assert p["receiver_space_m"] == pytest.approx(10.8, abs=0.1)
    assert p["speed_kmh"] == pytest.approx(132.4, abs=0.2)


def test_shot_measurements(ledger):
    s = first(ledger, "shot")
    assert s.values["distance_m"] == pytest.approx(11.7, abs=0.1)
    assert s.values["angle_deg"] == pytest.approx(33.1, abs=0.1)
    assert s.values["keeper_off_line_m"] == pytest.approx(2.0)
    assert s.values["triangle_defenders"] == 0
    assert "A1" in s.players


def test_options_rank_open_lanes_and_flag_blockers(ledger):
    opts = every(ledger, "option")
    assert len(opts) == 3
    assert opts[0].players[:2] == ("H2", "H4") and opts[0].values["lane_defenders"] == 0
    at_shot = [o for o in opts if o.players[0] == "H3"]
    assert [o.players[1] for o in at_shot] == ["H4", "H2"]
    assert at_shot[1].values["lane_defenders"] == 2


def test_gap_shift_and_run(ledger):
    gap = first(ledger, "back-line gap").values
    assert gap["gap_start_m"] == pytest.approx(15.1) and gap["gap_max_m"] == pytest.approx(25.1)
    shift = first(ledger, "defender shift")
    assert shift.players == ("A4",)
    assert shift.values["shift_m"] == pytest.approx(10.0) and shift.values["seconds"] == pytest.approx(2.0)
    run = first(ledger, "run")
    assert run.players[0] == "H3"
    assert run.values["beyond_line_m"] == pytest.approx(1.7, abs=0.2)
    assert run.values["top_speed_kmh"] == pytest.approx(28.8, abs=0.5)
    assert first(ledger, "line").values["beyond_line_m"] == pytest.approx(-6.0)


def test_numbers_at_the_end(ledger):
    assert not every(ledger, "behind the ball")  # shot from inside the box, where the count says nothing
    assert every(ledger, "box numbers")[-1].values == {"attackers_in_box": 1.0, "defenders_in_box": 0.0}
    tempo = first(ledger, "tempo").values
    assert tempo["seconds"] == pytest.approx(6.6) and tempo["passes"] == 1


def test_situation_reports_score_change(ledger):
    expected = {"home_before": 0.0, "away_before": 0.0, "home_after": 1.0, "away_after": 0.0}
    assert first(ledger, "situation").values == expected


def test_context_facts_have_no_frame(ledger):
    ctx = every(ledger, "context")
    assert ctx and all(f.frame is None for f in ctx)
    assert ctx[0].text.startswith("Match context, Home")


@pytest.mark.skipif(not GAME.exists(), reason="Metrica sample game 2 not downloaded")
def test_real_goal_phase_builds_clean_ledger():
    from phasemap.ingest import metrica

    match = metrica.load(GAME)
    all_phases = phases.segment(match.events)
    goal = phases.notable(all_phases)[0]
    built, _ = facts.build(match, goal, all_phases)
    assert len(built) >= 15
    assert "shot" in {f.kind for f in built}
    for fact in built:
        assert check_claim(fact.id, fact.text, [fact.id], built).status == "verified", fact.text
        for key, value in fact.values.items():
            if key.endswith("speed_kmh") and fact.kind not in ("pass", "shot"):
                assert value < 45, fact.text
