"""Load Metrica Sports sample games (CSV tracking + events) into a Match.

Metrica coordinates run 0-1 with (0, 0) at the top-left corner of a 105 x 68 m pitch.
They are converted to metres with y flipped so that y grows toward the top touchline.
"""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .. import pitch
from ..match import Event, Match

SOURCE = "Metrica Sports sample data (github.com/metrica-sports/sample-data), anonymized"
RAW_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/{game}/{file}"
FPS = 25.0


def game_files(game: str) -> dict[str, str]:
    return {
        "home": f"{game}_RawTrackingData_Home_Team.csv",
        "away": f"{game}_RawTrackingData_Away_Team.csv",
        "events": f"{game}_RawEventsData.csv",
    }


def fetch(game: str, dest: str | Path) -> Path:
    """Download one CSV sample game into dest/game, skipping files already present."""
    try:  # use the OS certificate store; python.org builds often lack one
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    folder = Path(dest) / game
    folder.mkdir(parents=True, exist_ok=True)
    for name in game_files(game).values():
        target = folder / name
        if target.exists():
            continue
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(RAW_URL.format(game=game, file=name), partial)
        partial.rename(target)
    return folder


def _to_metres(x, y) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(x, dtype=float) * pitch.LENGTH, (1.0 - np.asarray(y, dtype=float)) * pitch.WIDTH


def _read_tracking(path: Path, prefix: str):
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # team names
        next(reader)  # jersey numbers
        header = next(reader)
    # A player's x column carries the name ("Player11", sometimes "Player 26"); y follows it.
    players = [(i, h.replace(" ", "")[len("Player") :]) for i, h in enumerate(header) if h.strip().startswith("Player")]
    ball_col = header.index("Ball")
    raw = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=float)
    xs, ys = _to_metres(raw[:, [i for i, _ in players]], raw[:, [i + 1 for i, _ in players]])
    bx, by = _to_metres(raw[:, ball_col], raw[:, ball_col + 1])
    return {
        "frames": raw[:, 1].astype(int),
        "periods": raw[:, 0].astype(int),
        "clock": raw[:, 2],
        "ids": [f"{prefix}{number}" for _, number in players],
        "xy": np.stack([xs, ys], axis=-1).astype(np.float32),
        "ball": np.stack([bx, by], axis=-1).astype(np.float32),
    }


def _player(value, team: str, id_by_number: dict[str, str]) -> str | None:
    if pd.isna(value):
        return None
    number = str(value).replace(" ", "").removeprefix("Player")
    return id_by_number.get(number, f"{'H' if team == 'home' else 'A'}{number}")


def _point(x, y) -> tuple[float, float] | None:
    if pd.isna(x) or pd.isna(y):
        return None
    mx, my = _to_metres(x, y)
    return (round(float(mx), 2), round(float(my), 2))


def _read_events(path: Path, id_by_number: dict[str, str]) -> list[Event]:
    events = []
    for i, r in enumerate(pd.read_csv(path).to_dict("records")):
        team = str(r["Team"]).strip().lower()
        events.append(
            Event(
                index=i,
                team=team,
                type=str(r["Type"]).strip().upper(),
                subtype="" if pd.isna(r["Subtype"]) else str(r["Subtype"]).strip().upper(),
                period=int(r["Period"]),
                start_frame=int(r["Start Frame"]),
                end_frame=int(r["End Frame"]),
                start_s=float(r["Start Time [s]"]),
                end_s=float(r["End Time [s]"]),
                player=_player(r["From"], team, id_by_number),
                receiver=_player(r["To"], team, id_by_number),
                start_xy=_point(r["Start X"], r["Start Y"]),
                end_xy=_point(r["End X"], r["End Y"]),
            )
        )
    return events


def load(folder: str | Path) -> Match:
    """Load a sample game folder such as data/metrica/Sample_Game_2."""
    folder = Path(folder)
    files = game_files(folder.name)
    home = _read_tracking(folder / files["home"], "H")
    away = _read_tracking(folder / files["away"], "A")
    if not np.array_equal(home["frames"], away["frames"]):
        raise ValueError("home and away tracking files cover different frames")
    ids = home["ids"] + away["ids"]
    numbers = [pid[1:] for pid in ids]
    id_by_number = {n: pid for n, pid in zip(numbers, ids) if numbers.count(n) == 1}
    return Match(
        name=folder.name.replace("_", " "),
        source=SOURCE,
        fps=FPS,
        frames=home["frames"],
        periods=home["periods"],
        clock_s=home["clock"],
        player_ids=ids,
        player_teams=["home"] * len(home["ids"]) + ["away"] * len(away["ids"]),
        xy=np.concatenate([home["xy"], away["xy"]], axis=1),
        ball=np.where(np.isfinite(home["ball"]), home["ball"], away["ball"]),
        events=_read_events(folder / files["events"], id_by_number),
    )
