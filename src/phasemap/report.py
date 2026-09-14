"""Render analysed phase bundles as one self-contained HTML page."""

from __future__ import annotations

import html
import json
from pathlib import Path

TEMPLATES = Path(__file__).parent / "templates"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400"
    "&family=Big+Shoulders+Display:wght@600;800&family=IBM+Plex+Mono:wght@400;500&display=swap"
)


def _bundles(data: dict | list[dict]) -> list[dict]:
    items = data if isinstance(data, list) else [data]
    if not items:
        raise ValueError("nothing to render")
    return sorted(items, key=lambda b: b["meta"]["phase"])


def title(data: dict | list[dict]) -> str:
    bundles = _bundles(data)
    if len(bundles) == 1:
        meta = bundles[0]["meta"]
        return f"PhaseMap {meta['clock_end']} {meta['outcome'].title()}"
    kind = "Goals" if all(b["meta"]["outcome"] == "goal" for b in bundles) else "Phases"
    return f"PhaseMap {bundles[0]['meta']['match']} {kind}"


def _script_json(data) -> str:
    # "<" never appears outside JSON strings, so escaping it keeps "</script>" out of the page.
    return json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")


def render(data: dict | list[dict], fragment: bool = False) -> str:
    """A full HTML document, or a fragment without html/head/body for hosts that add their own.

    Several bundles become one page with a switcher, ordered as they happened in the match.
    """
    bundles = _bundles(data)
    head = (
        f"<title>{html.escape(title(bundles))}</title>\n"
        f'<link rel="stylesheet" href="{FONTS}">\n'
        f"<style>\n{(TEMPLATES / 'report.css').read_text(encoding='utf-8')}</style>\n"
    )
    markup, script = (TEMPLATES / "report.html").read_text(encoding="utf-8").split("{{JS}}")
    body = markup.replace("{{DATA}}", _script_json(bundles)) + (TEMPLATES / "report.js").read_text(encoding="utf-8") + script
    if fragment:
        return head + body
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"{head}</head>\n<body>\n{body}</body>\n</html>\n"
    )


def write(data: dict | list[dict], path: str | Path, fragment: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(data, fragment=fragment), encoding="utf-8")
    return path
