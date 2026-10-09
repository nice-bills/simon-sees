"""Reading recorded playtest sessions back: shared by scripts/playtest_report.py and scripts/export_labels.py.

A session folder (see recorder.py) holds events.jsonl and feedback.jsonl. Here they become:
    Session.rounds        the `round` events, in order
    Session.corrections   {(round_id, pid): feedback row}, the players' standing "Wrong call"s (retractions applied)
    Session.notes         end-of-game notes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .recorder import read_jsonl


@dataclass
class Session:
    dir: Path
    meta: dict
    events: list[dict]
    feedback: list[dict]
    corrections: dict[tuple[str, int], dict] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.dir.name

    @property
    def rounds(self) -> list[dict]:
        return [e for e in self.events if e.get("event") == "round"]

    @property
    def errors(self) -> list[dict]:
        return [e for e in self.events if e.get("event") == "error"]

    @property
    def notes(self) -> list[dict]:
        return [f for f in self.feedback if f.get("kind") == "note"]


def standing_corrections(feedback: Iterable[dict]) -> dict[tuple[str, int], dict]:
    """The last word per (round, player): a later tap replaces an earlier one; a retraction removes it."""
    out: dict[tuple[str, int], dict] = {}
    for f in feedback:
        if f.get("kind") != "wrong_call" or f.get("round_id") is None or f.get("pid") is None:
            continue
        k = (str(f["round_id"]), int(f["pid"]))
        if f.get("retracted"):
            out.pop(k, None)
        else:
            out[k] = f
    return out


def load_session(folder: str | Path) -> Session:
    folder = Path(folder)
    try:
        meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    fb = list(read_jsonl(folder / "feedback.jsonl"))
    return Session(folder, meta, list(read_jsonl(folder / "events.jsonl")), fb, standing_corrections(fb))


def find_sessions(paths: Iterable[str | Path]) -> list[Path]:
    """Accepts session folders, a `sessions/` folder, or a folder that contains `sessions/` (a pulled dataset)."""
    found: list[Path] = []
    for p in map(Path, paths):
        if (p / "session.json").exists():
            found.append(p)
        elif (p / "sessions").is_dir():
            found += find_sessions([p / "sessions"])
        elif p.is_dir():
            found += sorted(c for c in p.iterdir() if (c / "session.json").exists())
    seen, out = set(), []
    for f in found:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def load_sessions(paths: Iterable[str | Path]) -> list[Session]:
    return [load_session(p) for p in find_sessions(paths)]


# --------------------------------------------------------------------------- what a verdict says about the picture


def label_name(kind: str) -> str:
    return "moved" if kind == "freeze" else "did"


def answer_from_verdict(kind: str, simon: bool, verdict: str) -> bool:
    """The yes/no answer to the referee's own question that a verdict implies.
    pose, Simon said:      out <-> did NOT do it
    pose, trick (no Simon): out <-> DID do it
    freeze:                out <-> moved"""
    out = verdict == "out"
    if kind == "freeze":
        return out
    return (not out) if simon else out
