"""Evidence ledger and claim checking.

Every measured fact gets an ID. The AI layer has to cite IDs for each claim, and each
claim is then checked mechanically: cited IDs must exist, every number in the claim
must match a cited measurement, and every player it names must appear in the evidence.
This checks grounding, not whether the tactical reasoning is sound.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterator

CLOCK = re.compile(r"\b\d{1,3}:\d{2}\b")
TOKENS = re.compile(r"\b[HAF]\d+\b|#\d+|\b\d+(?:st|nd|rd|th)\b|\bzone \d+\b")
VERSUS = re.compile(r"(\d)\s*vs?\.?\s*(\d)")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
PLAYER = re.compile(r"\b[HA]\d{1,2}\b")
TIME_SLACK_S = 2.0


@dataclass(frozen=True)
class Fact:
    id: str
    kind: str
    t: float  # seconds from the start of the replay window
    frame: int | None  # provider frame to seek to; None for match-level context
    text: str  # plain statement; every number in it is also in `values`
    values: dict[str, float] = field(default_factory=dict)
    players: tuple[str, ...] = ()

    def prompt_line(self) -> str:
        who = f" ({', '.join(self.players)})" if self.players else ""
        when = f"t={self.t:.1f}s" if self.frame is not None else "match"
        return f"{self.id} [{when}, {self.kind}{who}] {self.text}"


class Ledger:
    def __init__(self) -> None:
        self._facts: dict[str, Fact] = {}

    def add(self, kind: str, t: float, frame: int | None, text: str, players=(), **values: float) -> Fact:
        fid = f"F{len(self._facts) + 1}"
        self._facts[fid] = Fact(
            id=fid,
            kind=kind,
            t=round(float(t), 1),
            frame=None if frame is None else int(frame),
            text=text,
            values={k: float(v) for k, v in values.items()},
            players=tuple(players),
        )
        return self._facts[fid]

    def __contains__(self, fid: object) -> bool:
        return fid in self._facts

    def __getitem__(self, fid: str) -> Fact:
        return self._facts[fid]

    def __iter__(self) -> Iterator[Fact]:
        return iter(self._facts.values())

    def __len__(self) -> int:
        return len(self._facts)

    def prompt_block(self) -> str:
        return "\n".join(f.prompt_line() for f in self)

    def to_json(self) -> list[dict]:
        return [asdict(f) for f in self]


def numbers_in(text: str) -> list[float]:
    """Numbers a claim asserts, skipping clock times, player/fact IDs, shirt numbers, ordinals."""
    cleaned = VERSUS.sub(r"\1 v \2", TOKENS.sub(" ", CLOCK.sub(" ", text)))
    return [float(n) for n in NUMBER.findall(cleaned)]


def _candidates(facts) -> list[float]:
    """Every number a set of facts can back, including km/h <-> m/s conversions."""
    out = []
    for f in facts:
        if f.frame is not None:
            out.append(f.t)
        for key, value in f.values.items():
            out.append(value)
            if key.endswith("_kmh"):
                out.append(value / 3.6)
            elif key.endswith("_ms"):
                out.append(value * 3.6)
    return out


def _matches(n: float, v: float) -> bool:
    n, v = abs(n), abs(v)
    if abs(n - v) <= max(0.051, 0.05 * v):
        return True
    return n.is_integer() and v >= 0.5 and round(v) == n


@dataclass
class Check:
    path: str  # where the claim sits in the analysis, e.g. "causal_chain[2]"
    text: str
    evidence: list[str]
    status: str  # "verified" | "partial" | "unsupported"
    problems: list[str] = field(default_factory=list)


def check_claim(path: str, text: str, evidence: list, ledger: Ledger) -> Check:
    hard: list[str] = []
    soft: list[str] = []
    evidence = [str(e) for e in evidence]
    unknown = [e for e in evidence if e not in ledger]
    cited = [ledger[e] for e in evidence if e in ledger]
    if unknown:
        hard.append(f"cites unknown evidence: {', '.join(unknown)}")
    if not cited:
        hard.append("no valid evidence cited")
    cited_values, all_values = _candidates(cited), _candidates(ledger)
    for n in numbers_in(text):
        if any(_matches(n, v) for v in cited_values):
            continue
        if any(_matches(n, v) for v in all_values):
            soft.append(f"{n:g} is measured, but not in the cited facts")
        else:
            hard.append(f"{n:g} is not in any measurement")
    cited_players = {p for f in cited for p in f.players}
    known_players = {p for f in ledger for p in f.players}
    for pid in sorted(set(PLAYER.findall(text))):
        if pid not in known_players:
            hard.append(f"{pid} is not in any measurement")
        elif pid not in cited_players:
            soft.append(f"{pid} is not in the cited facts")
    status = "unsupported" if hard else "partial" if soft else "verified"
    return Check(path, text, evidence, status, hard + soft)


def check_free_text(path: str, text: str, ledger: Ledger) -> Check:
    """Headline-style text needs no citations, but its numbers and players must exist."""
    values = _candidates(ledger)
    problems = [f"{n:g} is not in any measurement" for n in numbers_in(text) if not any(_matches(n, v) for v in values)]
    known = {p for f in ledger for p in f.players}
    problems += [f"{pid} is not in any measurement" for pid in sorted(set(PLAYER.findall(text))) if pid not in known]
    return Check(path, text, [], "unsupported" if problems else "verified", problems)


def iter_claims(analysis: dict):
    """(path, text, evidence, t) for every evidence-backed item in an analysis."""
    for i, step in enumerate(analysis.get("causal_chain") or []):
        yield f"causal_chain[{i}]", f"{step.get('label', '')}: {step.get('detail', '')}", step.get("evidence", []), step.get("t")
    moment = analysis.get("key_moment")
    if moment:
        text = f"{moment.get('label', '')}: {moment.get('detail', '')} {moment.get('why_it_mattered', '')}"
        yield "key_moment", text, moment.get("evidence", []), moment.get("t")
    for side, key in (("attack", "what_worked"), ("attack", "recommendations"), ("defence", "what_went_wrong"), ("defence", "recommendations")):
        for i, item in enumerate((analysis.get(side) or {}).get(key) or []):
            yield f"{side}.{key}[{i}]", item.get("detail", ""), item.get("evidence", []), None
    for i, alt in enumerate(analysis.get("alternatives") or []):
        yield f"alternatives[{i}]", f"{alt.get('detail', '')} {alt.get('tradeoff', '')}", alt.get("evidence", []), None
    for i, caveat in enumerate(analysis.get("intent_caveats") or []):
        yield f"intent_caveats[{i}]", caveat.get("detail", ""), caveat.get("evidence", []), None


@dataclass
class Verification:
    checks: list[Check]

    def counts(self) -> dict[str, int]:
        out = {"verified": 0, "partial": 0, "unsupported": 0}
        for c in self.checks:
            out[c.status] += 1
        return out

    def score(self) -> float:
        """Share of checked items that are grounded, with partial items counting half."""
        if not self.checks:
            return 0.0
        c = self.counts()
        return (c["verified"] + 0.5 * c["partial"]) / len(self.checks)

    def by_path(self) -> dict[str, Check]:
        return {c.path: c for c in self.checks}

    def to_json(self) -> dict:
        return {"score": round(self.score(), 3), "counts": self.counts(), "checks": [asdict(c) for c in self.checks]}


def verify(analysis: dict, ledger: Ledger) -> Verification:
    checks = [check_free_text(key, analysis[key], ledger) for key in ("headline", "summary") if analysis.get(key)]
    for path, text, evidence, t in iter_claims(analysis):
        check = check_claim(path, text, evidence, ledger)
        times = [ledger[e].t for e in evidence if e in ledger and ledger[e].frame is not None]
        if isinstance(t, (int, float)) and times and min(abs(t - x) for x in times) > TIME_SLACK_S:
            check.problems.append(f"time {t:g}s is not near its evidence")
            if check.status == "verified":
                check.status = "partial"
        checks.append(check)
    return Verification(checks)
