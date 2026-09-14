import json
from pathlib import Path

import pytest

from phasemap import cli

GAME = Path(__file__).resolve().parents[1] / "data" / "metrica" / "Sample_Game_2"
needs_data = pytest.mark.skipif(not GAME.exists(), reason="Metrica sample game 2 not downloaded")


@needs_data
def test_phases_lists_goals_first(capsys):
    assert cli.main(["phases", "--game", str(GAME), "--top", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].split()[:2] == ["id", "team"]
    assert len(lines) == 4 and all(" goal " in line for line in lines[1:])


@needs_data
def test_analyze_without_reasoning_writes_report(tmp_path, capsys):
    assert cli.main(["analyze", "33", "--game", str(GAME), "--backend", "none", "--out", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "phase-33.json").read_text())
    assert data["analysis"] is None and data["meta"]["outcome"] == "goal"
    assert (tmp_path / "phase-33.html").read_text().startswith("<!doctype html>")
    assert "Report:" in capsys.readouterr().out


@needs_data
def test_failed_reasoning_still_writes_report(tmp_path, capsys, monkeypatch):
    def refuse(ledger, view, backend):
        raise cli.reason.ReasoningError("claude CLI failed: OAuth session expired")

    monkeypatch.setattr(cli.reason, "analyze", refuse)
    assert cli.main(["analyze", "33", "--game", str(GAME), "--out", str(tmp_path)]) == 0
    assert "OAuth session expired" in capsys.readouterr().err
    assert (tmp_path / "phase-33.html").exists()


@needs_data
def test_render_combines_bundles(tmp_path):
    for pid in ("215", "33"):
        assert cli.main(["analyze", pid, "--game", str(GAME), "--backend", "none", "--out", str(tmp_path)]) == 0
    target = tmp_path / "goals.html"
    args = ["render", str(tmp_path / "phase-215.json"), str(tmp_path / "phase-33.json"), "--output", str(target)]
    assert cli.main(args) == 0
    assert "<title>PhaseMap Sample Game 2 Goals</title>" in target.read_text()


def test_missing_game_explains_the_fix(tmp_path):
    with pytest.raises(SystemExit, match="phasemap fetch"):
        cli.main(["phases", "--game", str(tmp_path / "Sample_Game_9")])
