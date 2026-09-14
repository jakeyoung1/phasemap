import pytest

from phasemap.evidence import Ledger, numbers_in, verify


@pytest.fixture
def ledger():
    led = Ledger()
    led.add(
        "gap", 4.2, 1255, "Gap between A18 and A19 grew from 9.1 m to 17.3 m in 2.4 s",
        players=("A18", "A19"), gap_before_m=9.1, gap_after_m=17.3, seconds=2.4,
    )
    led.add("pass", 6.6, 1310, "H7 passed 21.6 m at 65.5 km/h", players=("H7", "H9"), length_m=21.6, speed_kmh=65.5)
    led.add("shot", 8.1, 1350, "H9 shot from 11.8 m with 2 defenders in the lane", players=("H9",), distance_m=11.8, blockers=2)
    return led


def test_ledger_ids_and_prompt(ledger):
    assert [f.id for f in ledger] == ["F1", "F2", "F3"]
    assert "F2" in ledger and "F9" not in ledger
    assert ledger.prompt_block().splitlines()[1] == "F2 [t=6.6s, pass (H7, H9)] H7 passed 21.6 m at 65.5 km/h"
    assert ledger.to_json()[2]["values"] == {"distance_m": 11.8, "blockers": 2.0}


def test_numbers_ignore_ids_clock_and_ordinals():
    text = "At 47:44 H9 (#9) made his 2nd run, 7.9 m in 1.2s, a 3v2 in F12"
    assert numbers_in(text) == [7.9, 1.2, 3.0, 2.0]


def test_verified_claim(ledger):
    analysis = {"causal_chain": [{"label": "Gap opens", "detail": "A18 and A19 split from 9.1 m to 17.3 m", "evidence": ["F1"], "t": 4.2}]}
    check = verify(analysis, ledger).checks[0]
    assert check.status == "verified", check.problems


def test_rounding_and_speed_units_are_accepted(ledger):
    claim = {"detail": "H7's 22 m pass travelled at about 18.2 m/s", "evidence": ["F2"]}
    check = verify({"attack": {"what_worked": [claim]}}, ledger).checks[0]
    assert check.status == "verified", check.problems


def test_number_from_uncited_fact_is_partial(ledger):
    claim = {"detail": "H9 shot from 11.8 m", "evidence": ["F2"]}
    check = verify({"attack": {"what_worked": [claim]}}, ledger).checks[0]
    assert check.status == "partial"
    assert check.problems == ["11.8 is measured, but not in the cited facts"]


def test_invented_number_and_unknown_id_are_unsupported(ledger):
    claim = {"detail": "The back line was 30 m high", "evidence": ["F99"]}
    check = verify({"defence": {"what_went_wrong": [claim]}}, ledger).checks[0]
    assert check.status == "unsupported"
    assert "30 is not in any measurement" in check.problems
    assert any("F99" in p for p in check.problems)


def test_missing_evidence_is_unsupported(ledger):
    claim = {"detail": "The press was poorly organised", "evidence": []}
    assert verify({"defence": {"what_went_wrong": [claim]}}, ledger).checks[0].status == "unsupported"


def test_player_not_in_any_fact_is_unsupported(ledger):
    claim = {"player": "H4", "detail": "H4 should have tracked A19", "evidence": ["F1"]}
    check = verify({"attack": {"recommendations": [claim]}}, ledger).checks[0]
    assert check.status == "unsupported"
    assert check.problems == ["H4 is not in any measurement"]


def test_key_moment_time_must_sit_near_evidence(ledger):
    moment = {"t": 30.0, "label": "Pass into the gap", "detail": "H7 finds H9", "why_it_mattered": "It broke the line", "evidence": ["F2"]}
    check = verify({"key_moment": moment}, ledger).checks[0]
    assert check.status == "partial"
    assert check.problems == ["time 30s is not near its evidence"]


def test_headline_numbers_checked_against_all_facts(ledger):
    assert verify({"headline": "A 17.3 m gap let H9 shoot from 12 m"}, ledger).checks[0].status == "verified"
    assert verify({"headline": "A 40 m gap"}, ledger).checks[0].status == "unsupported"


def test_score_and_counts(ledger):
    analysis = {
        "headline": "A 17.3 m gap",
        "causal_chain": [{"label": "Gap", "detail": "A18 and A19 split", "evidence": ["F1"]}],
        "attack": {"what_worked": [{"detail": "H9 shot from 11.8 m", "evidence": ["F2"]}]},
        "defence": {"what_went_wrong": [{"detail": "Nobody tracked the run", "evidence": []}]},
    }
    result = verify(analysis, ledger)
    assert result.counts() == {"verified": 2, "partial": 1, "unsupported": 1}
    assert result.score() == pytest.approx(0.625)
    assert result.to_json()["counts"]["partial"] == 1
