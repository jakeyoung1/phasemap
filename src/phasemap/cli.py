"""Command line: fetch data, list phases, and analyse a phase into a report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import bundle, context, facts, phases, reason, report
from .evidence import verify
from .ingest import metrica

DEFAULT_GAME = Path("data/metrica/Sample_Game_2")
DEFAULT_OUT = Path("out")


def _load(game: Path):
    if not game.exists():
        raise SystemExit(f"No data at {game}. Download it with: phasemap fetch --game {game.name} --dest {game.parent}")
    match = metrica.load(game)
    return match, phases.segment(match.events)


def cmd_fetch(args) -> int:
    print(f"Saved {args.game} to {metrica.fetch(args.game, args.dest)}")
    return 0


def cmd_phases(args) -> int:
    match, found = _load(args.game)
    picked = found if args.all else phases.notable(found, args.top)
    print(f"{'id':>4}  {'team':<5} {'clock':<13} {'outcome':<17} {'passes':>6}  secs")
    for p in picked:
        clock = f"{match.clock_label(p.start_frame)}-{match.clock_label(p.end_frame)}"
        print(f"{p.id:>4}  {p.team:<5} {clock:<13} {p.outcome:<17} {len(p.passes):>6}  {p.duration_s:4.1f}")
    return 0


def cmd_analyze(args) -> int:
    match, found = _load(args.game)
    phase = next((p for p in found if p.id == args.phase), None)
    if phase is None:
        raise SystemExit(f"No phase {args.phase}. List them with: phasemap phases")
    ledger, view = facts.build(match, phase, found, context.match_stats(match, found))
    analysis = verification = backend = None
    if args.backend != "none":
        try:
            analysis, backend = reason.analyze(ledger, view, backend=args.backend)
        except reason.ReasoningError as exc:
            print(f"Written breakdown skipped: {exc}", file=sys.stderr)
        else:
            verification = verify(analysis, ledger)
    data = bundle.build(view, ledger, analysis, verification, backend)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"phase-{phase.id}"
    stem.with_suffix(".json").write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    page = report.write(data, stem.with_suffix(".html"), fragment=args.fragment)
    print(f"{len(ledger)} facts measured for phase {phase.id} ({phase.team}, {phase.outcome})")
    if verification is not None:
        counts = verification.counts()
        print(
            f"Grounding {verification.score():.0%}: {counts['verified']} backed, "
            f"{counts['partial']} partly backed, {counts['unsupported']} not backed"
        )
    print(f"Report: {page}")
    return 0


def cmd_render(args) -> int:
    sources = [Path(b) for b in args.bundles]
    data = [json.loads(p.read_text(encoding="utf-8")) for p in sources]
    if args.output:
        target = Path(args.output)
    elif len(sources) == 1:
        target = sources[0].with_suffix(".html")
    else:
        target = sources[0].parent / "phases.html"
    print(f"Report: {report.write(data, target, fragment=args.fragment)}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="phasemap", description="Measure and explain phases of play from tracking data.")
    sub = root.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download a Metrica sample game")
    fetch.add_argument("--game", default="Sample_Game_2")
    fetch.add_argument("--dest", type=Path, default=Path("data/metrica"))
    fetch.set_defaults(run=cmd_fetch)

    listing = sub.add_parser("phases", help="list phases that ended in a shot, goals first")
    listing.add_argument("--game", type=Path, default=DEFAULT_GAME)
    listing.add_argument("--top", type=int, default=15)
    listing.add_argument("--all", action="store_true", help="list every phase, not only shots")
    listing.set_defaults(run=cmd_phases)

    analyze = sub.add_parser("analyze", help="measure one phase, explain it, and write a report")
    analyze.add_argument("phase", type=int)
    analyze.add_argument("--game", type=Path, default=DEFAULT_GAME)
    analyze.add_argument("--backend", choices=("auto", "api", "cli", "none"), default="auto")
    analyze.add_argument("--out", type=Path, default=DEFAULT_OUT)
    analyze.add_argument("--fragment", action="store_true", help="leave out html/head/body tags")
    analyze.set_defaults(run=cmd_analyze)

    render = sub.add_parser("render", help="rebuild a report from saved bundles; several share one page")
    render.add_argument("bundles", nargs="+")
    render.add_argument("--output")
    render.add_argument("--fragment", action="store_true", help="leave out html/head/body tags")
    render.set_defaults(run=cmd_render)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
