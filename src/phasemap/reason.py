"""Reasoning layer: turn a phase's evidence ledger into a cited tactical analysis.

Two ways to reach Claude: the Anthropic API (ANTHROPIC_API_KEY) or a logged-in
`claude` CLI in print mode. Both are asked for JSON that follows SCHEMA.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from . import pitch
from .evidence import Ledger
from .view import TEAM, PhaseView, finite

API_MODEL = "claude-opus-5"
CLI_MODEL = "opus"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CLI_TIMEOUT_S = 900


class ReasoningError(RuntimeError):
    pass


def _obj(**props) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_TEXT = {"type": "string"}
_IDS = {"type": "array", "items": {"type": "string"}}
_CLAIM = _obj(detail=_TEXT, evidence=_IDS)
_ADVICE = _obj(player=_TEXT, detail=_TEXT, evidence=_IDS)

SCHEMA = _obj(
    headline=_TEXT,
    summary=_TEXT,
    causal_chain={"type": "array", "items": _obj(label=_TEXT, detail=_TEXT, t={"type": "number"}, evidence=_IDS)},
    key_moment=_obj(t={"type": "number"}, label=_TEXT, detail=_TEXT, why_it_mattered=_TEXT, evidence=_IDS),
    attack=_obj(what_worked={"type": "array", "items": _CLAIM}, recommendations={"type": "array", "items": _ADVICE}),
    defence=_obj(what_went_wrong={"type": "array", "items": _CLAIM}, recommendations={"type": "array", "items": _ADVICE}),
    alternatives={"type": "array", "items": _obj(detail=_TEXT, tradeoff=_TEXT, evidence=_IDS)},
    intent_caveats={"type": "array", "items": _CLAIM},
    data_limits={"type": "array", "items": _TEXT},
    confidence={"type": "string", "enum": ["low", "medium", "high"]},
)

SYSTEM = """You are PhaseMap's tactical analyst. You explain why one attacking phase of a soccer match developed the way it did, for a coach, using only measurements taken from tracking and event data.

Evidence rules
- Support every claim with the IDs of the facts it rests on (F1, F2, ...). Cite the facts that contain the numbers and players you mention.
- Use only numbers that appear in the cited facts. Do not calculate new ones such as differences, totals or averages.
- Refer to players by ID: H for Home, A for Away, then the shirt number. Do not invent names or official positions, and describe where a player was only as the facts do.
- Lanes such as "left wing" or "right half-space" are named from the attacking team's point of view.
- t is seconds from the start of the replay window. Give each causal step and the key moment the t of the fact it describes.

Reasoning rules
- Keep what happened separate from why. Tracking data cannot show intent. When a player acted with a defender close by at release, the action may have been forced rather than chosen; say so in intent_caveats instead of presenting it as a deliberate choice.
- Judge alternatives only from the option facts. Passing lanes are straight-line estimates for ground passes, so a lofted ball can beat a contested lane.
- Match context facts cover this match only. Use them to say whether a moment fits the game's pattern, never as proof of a player's general quality.
- Pitch control percentages come from a simple arrival-time model. Treat them as indicative, not exact.
- A phase can be decided by things this data cannot measure, such as first touch, body shape, weight of pass or the finish itself. When that is likely, say so in data_limits rather than stretching the numbers.

Writing
- Plain coaching language, short sentences, no hype, no em dashes.
- Players are anonymous IDs. Write "H1" or "they", never "he" or "she", and never guess at anything the data does not carry.
- Keep fact IDs out of the headline and summary; cite them only in the evidence fields.
- headline: one sentence of at most 20 words. summary: two or three sentences.
- causal_chain: three to six ordered steps from how possession started to the outcome, each with a label of two to five words.
- key_moment: the one moment that most changed the phase.
- attack.what_worked and defence.what_went_wrong: two to four items each.
- attack.recommendations and defence.recommendations: two or three concrete changes each. Set player to the player ID when the advice is for one player, otherwise an empty string.
- alternatives: one to three options at the key decision, each with its trade-off.
- intent_caveats: one to three places where the data cannot separate a choice from a forced action.
- data_limits: two to five short statements of what this data cannot show about this phase.
- confidence: low, medium or high for the explanation as a whole.
"""


def build_prompt(ledger: Ledger, view: PhaseView) -> str:
    match, phase = view.match, view.phase
    half = {1: "first half", 2: "second half"}.get(phase.period, f"period {phase.period}")
    lines = [
        f"Match: {match.name}. Source: {match.source}.",
        f"Phase: {TEAM[phase.team]} in possession, {match.clock_label(phase.start_frame)} to "
        f"{match.clock_label(phase.end_frame)} of the {half}. Outcome: {phase.outcome}.",
        f"Replay window: t=0.0s to t={view.n / view.fps:.1f}s. Possession started at t={view.t(view.won_i):.1f}s, "
        f"final action at t={view.t(view.final_i):.1f}s.",
        f"All positions use {TEAM[phase.team]}'s attacking view: they attack the goal at the far end, "
        "and lanes are named from their side.",
        "",
        "Event timeline:",
    ]
    for ev in phase.events:
        i = view.i(ev.start_frame)
        spot = view.action_point(ev.player, i, ev.start_xy)
        where = f" in {pitch.zone(*spot)}" if finite(spot) else ""
        target = f" to {ev.receiver}" if ev.receiver else ""
        lines.append(f"t={view.t(i):.1f}s {ev.player or TEAM[ev.team]}: {ev.label}{target}{where}")
    lines += ["", "Measured facts:", *(f.prompt_line() for f in ledger if f.frame is not None)]
    lines += ["", "Match context (this match only):", *(f.prompt_line() for f in ledger if f.frame is None)]
    lines += ["", f"Explain why this phase ended as it did ({phase.outcome}). Cite fact IDs for every claim."]
    return "\n".join(lines)


def pick_backend() -> str:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "api"
    if shutil.which("claude"):
        return "cli"
    raise ReasoningError("No way to reach Claude: set ANTHROPIC_API_KEY, or install and log in to the claude CLI.")


def analyze(ledger: Ledger, view: PhaseView, backend: str = "auto") -> tuple[dict, str]:
    """Run the reasoning layer. Returns the normalized analysis and the backend used."""
    prompt = build_prompt(ledger, view)
    chosen = pick_backend() if backend == "auto" else backend
    if chosen == "api":
        raw = call_api(SYSTEM, prompt)
    elif chosen == "cli":
        raw = call_cli(SYSTEM, prompt)
    else:
        raise ValueError(f"unknown backend: {backend}")
    return normalize(raw), chosen


def call_api(system: str, prompt: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    try:
        with client.beta.messages.stream(
            model=API_MODEL,
            max_tokens=32000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.APIStatusError as exc:
        raise ReasoningError(f"Claude API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise ReasoningError("Could not reach the Claude API") from exc
    if message.stop_reason == "refusal":
        raise ReasoningError("Claude declined to analyze this phase")
    if message.stop_reason == "max_tokens":
        raise ReasoningError("The analysis hit the output limit before finishing")
    return extract_json(next((b.text for b in message.content if b.type == "text"), ""))


def call_cli(system: str, prompt: str, timeout: float = CLI_TIMEOUT_S) -> dict:
    exe = shutil.which("claude")
    if exe is None:
        raise ReasoningError("claude CLI not found on PATH")
    cmd = [
        exe, "-p", "--safe-mode", "--tools", "", "--no-session-persistence", "--model", CLI_MODEL,
        "--output-format", "json", "--system-prompt", system, "--json-schema", json.dumps(SCHEMA),
    ]
    # Session variables from a host Claude Code process would make the child try the host's login.
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE" and not k.startswith("CLAUDE_CODE_")}
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise ReasoningError(f"claude CLI did not finish within {timeout:.0f} s") from exc
    return parse_cli_output(proc.stdout, proc.returncode, proc.stderr)


def parse_cli_output(stdout: str, returncode: int, stderr: str = "") -> dict:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        detail = (stderr or stdout).strip()[:300]
        raise ReasoningError(f"claude CLI gave no JSON (exit {returncode}): {detail}") from exc
    if payload.get("is_error"):
        raise ReasoningError(f"claude CLI failed: {payload.get('result')}")
    if isinstance(payload.get("structured_output"), dict):
        return payload["structured_output"]
    if isinstance(payload.get("result"), str):
        return extract_json(payload["result"])
    raise ReasoningError("claude CLI returned no analysis")


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ReasoningError("response contained no JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ReasoningError(f"response JSON did not parse: {exc}") from exc
    if not isinstance(data, dict):
        raise ReasoningError("response JSON was not an object")
    return data


_LISTS = ("causal_chain", "alternatives", "intent_caveats", "data_limits")
_SIDES = {"attack": ("what_worked", "recommendations"), "defence": ("what_went_wrong", "recommendations")}


def normalize(data: dict) -> dict:
    """Coerce a model response into the shape the verifier and report expect."""
    if not isinstance(data, dict) or not isinstance(data.get("headline"), str) or not data["headline"].strip():
        raise ReasoningError("analysis has no headline")
    out = dict(data)
    out["summary"] = out["summary"] if isinstance(out.get("summary"), str) else ""
    for key in _LISTS:
        items = out.get(key) if isinstance(out.get(key), list) else []
        out[key] = [x for x in items if isinstance(x, str if key == "data_limits" else dict)]
    for side, keys in _SIDES.items():
        block = out.get(side) if isinstance(out.get(side), dict) else {}
        out[side] = {k: [x for x in (block.get(k) or []) if isinstance(x, dict)] for k in keys}
    if not isinstance(out.get("key_moment"), dict):
        out["key_moment"] = None
    if out.get("confidence") not in ("low", "medium", "high"):
        out["confidence"] = "low"
    return out
