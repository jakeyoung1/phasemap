import json
import subprocess

import pytest

from phasemap import facts, phases, reason


@pytest.fixture
def built(through_ball):
    found = phases.segment(through_ball.events)
    return facts.build(through_ball, found[0], found)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_schema_objects_are_closed_and_fully_required():
    objects = [n for n in _walk(reason.SCHEMA) if n.get("type") == "object"]
    assert len(objects) >= 8
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def test_prompt_carries_every_fact_and_the_timeline(built):
    ledger, view = built
    prompt = reason.build_prompt(ledger, view)
    for fact in ledger:
        assert f"{fact.id} [" in prompt
    assert "Event timeline:" in prompt and "H2: pass to H3" in prompt
    assert "Outcome: goal" in prompt
    assert "—" not in reason.SYSTEM and "—" not in prompt


def test_parse_cli_prefers_structured_output():
    payload = {"is_error": False, "structured_output": {"headline": "x"}, "result": ""}
    assert reason.parse_cli_output(json.dumps(payload), 0) == {"headline": "x"}


def test_parse_cli_falls_back_to_json_in_result_text():
    payload = {"is_error": False, "result": 'Here you go:\n{"headline": "y"}\n'}
    assert reason.parse_cli_output(json.dumps(payload), 0) == {"headline": "y"}


def test_parse_cli_errors_are_readable():
    payload = {"is_error": True, "result": "Failed to authenticate: OAuth session expired"}
    with pytest.raises(reason.ReasoningError, match="OAuth session expired"):
        reason.parse_cli_output(json.dumps(payload), 1)
    with pytest.raises(reason.ReasoningError, match="no JSON"):
        reason.parse_cli_output("segfault", 139, "boom")


def test_normalize_fills_gaps_and_needs_headline():
    raw = {
        "headline": "A gap opened",
        "attack": {"what_worked": [{"detail": "d", "evidence": []}, "junk"]},
        "confidence": "certain",
    }
    out = reason.normalize(raw)
    assert out["attack"]["what_worked"] == [{"detail": "d", "evidence": []}]
    assert out["attack"]["recommendations"] == [] and out["defence"]["what_went_wrong"] == []
    assert out["causal_chain"] == [] and out["key_moment"] is None and out["confidence"] == "low"
    with pytest.raises(reason.ReasoningError):
        reason.normalize({"summary": "no headline"})


def test_pick_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert reason.pick_backend() == "api"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(reason.shutil, "which", lambda name: "/usr/bin/claude")
    assert reason.pick_backend() == "cli"
    monkeypatch.setattr(reason.shutil, "which", lambda name: None)
    with pytest.raises(reason.ReasoningError):
        reason.pick_backend()


def test_cli_call_is_sandboxed(monkeypatch, built):
    seen = {}

    def fake_run(cmd, input, capture_output, text, timeout, env):
        seen.update(cmd=cmd, input=input, env=env)
        out = json.dumps({"is_error": False, "structured_output": {"headline": "ok"}})
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(reason.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(reason.subprocess, "run", fake_run)
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "desktop")
    ledger, view = built
    analysis, backend = reason.analyze(ledger, view, backend="cli")
    assert backend == "cli" and analysis["headline"] == "ok"
    cmd = seen["cmd"]
    assert cmd[:2] == ["/usr/bin/claude", "-p"]
    assert "--safe-mode" in cmd and cmd[cmd.index("--tools") + 1] == ""
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == reason.SCHEMA
    assert "CLAUDECODE" not in seen["env"] and "CLAUDE_CODE_ENTRYPOINT" not in seen["env"]
    assert "Measured facts:" in seen["input"]
