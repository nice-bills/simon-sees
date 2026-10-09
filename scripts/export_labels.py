"""Turn recorded playtest sessions into a JSONL dataset for fine-tuning the pose referee (e.g. on Tinker).

    python scripts/export_labels.py                              # ./playtests (pulled) or ./sessions -> labels.jsonl
    python scripts/export_labels.py playtests --out data/labels.jsonl --human-only
    python scripts/export_labels.py sessions --min-margin 0.3    # model labels only when d1 was confident

One row per player crop the referee judged:

    {"image": "sessions/<s>/crops/<round>_p0_f0.jpg",  # relative to the output file (--absolute for full paths)
     "images": [...],                                  # freeze rounds: the BEFORE and AFTER crops, in order
     "command": "one_foot", "question": "Look at the person in this photo. Is that person standing on one foot?",
     "kind": "pose" | "freeze", "simon": true, "label_name": "did" | "moved",
     "label": true, "answer": "yes",                    # the answer to the question
     "source": "model" | "human_correction",
     "p_model": 0.82, "session": ..., "round_id": ..., "pid": 0, "frame": 0}

`source: human_correction` rows come from a player tapping "Wrong call": the verdict was flipped by a person,
and the label is what the corrected verdict implies (pose + Simon said: out = did not do it; trick: out = did it;
freeze: out = moved). `source: model` rows are d1's own answer (P >= 0.5) on calls nobody disputed: pseudo
labels, not ground truth. A correction replaces the model row for that player and round; it is applied to every
frame of the round (the pose is assumed to hold across the ~0.5 s between frames).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simon_sees.playtest_data import Session, answer_from_verdict, label_name, load_sessions  # noqa: E402


def rows_for(session: Session, min_margin: float = 0.0, human_only: bool = False) -> list[dict]:
    rows = []
    for r in session.rounds:
        kind, simon, rid = r.get("kind", "pose"), bool(r.get("simon", True)), r.get("round_id")
        base = {"command": r.get("command"), "question": r.get("question"), "kind": kind, "simon": simon,
                "label_name": label_name(kind), "session": session.name, "round_id": rid}
        for p in r.get("players", []):
            crops = [c for c in (p.get("crops") or [])]
            if not p.get("seen") or not any(crops):
                continue
            fix = session.corrections.get((str(rid), int(p["pid"])))
            if fix is not None:
                label, source = answer_from_verdict(kind, simon, fix.get("correct", "")), "human_correction"
            elif human_only:
                continue
            else:
                source = "model"
                label = None  # per frame below
            common = {**base, "pid": p["pid"], "source": source}
            if fix is not None:
                common["given"] = p.get("verdict")
                common["corrected_to"] = fix.get("correct")
            if kind == "freeze":
                pm = p.get("p_moved")
                if source == "model":
                    if pm is None or abs(pm - 0.5) < min_margin:
                        continue
                    label = pm >= 0.5
                imgs = [session.dir / c for c in crops if c]
                rows.append({"image": imgs[0], "images": imgs, **common, "label": bool(label),
                             "answer": "yes" if label else "no", "p_model": pm, "frame": None})
                continue
            per = p.get("p_per_frame") or [p.get("p")] * len(crops)
            for fi, c in enumerate(crops):
                if not c:
                    continue
                pf = per[fi] if fi < len(per) else None
                lab = label
                if source == "model":
                    if pf is None or abs(pf - 0.5) < min_margin:
                        continue
                    lab = pf >= 0.5
                img = session.dir / c
                rows.append({"image": img, "images": [img], **common, "label": bool(lab),
                             "answer": "yes" if lab else "no", "p_model": pf, "frame": fi})
    return rows


def export(paths: list[str], out: str | Path, min_margin: float = 0.0, human_only: bool = False,
           absolute: bool = False) -> tuple[Path, Counter]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base = out.parent.resolve()
    counts: Counter = Counter()

    def rel(p: Path) -> str:
        p = Path(p).resolve()
        return str(p) if absolute else os.path.relpath(p, base)

    with out.open("w", encoding="utf-8") as f:
        for s in load_sessions(paths):
            for row in rows_for(s, min_margin, human_only):
                row["image"] = rel(row["image"])
                row["images"] = [rel(x) for x in row["images"]]
                counts[row["source"]] += 1
                f.write(json.dumps(row) + "\n")
    return out, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*", help="session folders, sessions/, or a pulled dataset (default: playtests/ or sessions/)")
    ap.add_argument("--out", default="labels.jsonl")
    ap.add_argument("--human-only", action="store_true", help="only rows a player corrected")
    ap.add_argument("--min-margin", type=float, default=0.0, help="drop model rows with |P - 0.5| below this")
    ap.add_argument("--absolute", action="store_true", help="absolute image paths instead of relative to --out")
    a = ap.parse_args(argv)
    paths = a.paths or [p for p in ("playtests", "sessions") if Path(p).is_dir()]
    if not paths:
        print("no sessions found: pass a folder (playtests/, sessions/ or one session)", file=sys.stderr)
        return 1
    out, counts = export(paths, a.out, a.min_margin, a.human_only, a.absolute)
    print(f"{out}: {sum(counts.values())} rows ({counts['human_correction']} human corrections, {counts['model']} model labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
