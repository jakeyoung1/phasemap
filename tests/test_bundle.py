import json

from phasemap import bundle, facts, phases
from phasemap.evidence import verify


def test_bundle_is_strict_json(through_ball):
    found = phases.segment(through_ball.events)
    ledger, view = facts.build(through_ball, found[0], found)
    analysis = {
        "headline": "H3 finished a through ball",
        "causal_chain": [{"label": "Run", "detail": "H3 ran in behind", "t": 2.0, "evidence": ["F1"]}],
    }
    data = json.loads(json.dumps(bundle.build(view, ledger, analysis, verify(analysis, ledger), "cli"), allow_nan=False))
    assert data["meta"]["outcome"] == "goal" and data["meta"]["team"] == "home"
    assert data["meta"]["score_after"] == {"home": 1, "away": 0}
    assert len(data["track"]["xy"]) == view.n
    assert len(data["track"]["xy"][0]) == 2 * len(data["players"])
    assert data["track"]["ball"][0] is None
    assert [p["id"] for p in data["players"] if p["keeper"]] == ["H1", "A1"]
    assert data["events"][1]["player"] == "H2" and data["events"][1]["end"] == [84.0, 20.0]
    counts = data["verification"]["counts"]
    assert sum(counts.values()) == len(data["verification"]["checks"])


def test_clean_replaces_non_finite_numbers():
    assert bundle.clean({"a": [1.0, float("nan")], "b": {"c": float("inf")}, "d": (2, "x")}) == {
        "a": [1.0, None],
        "b": {"c": None},
        "d": [2, "x"],
    }
