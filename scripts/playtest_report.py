"""Summarise recorded playtest sessions into a markdown report: what to fix next.

    python scripts/playtest_report.py playtests --out playtests/report.md
    python scripts/playtest_report.py sessions/20261009-153000-ab12cd        # one session

Sections: overview (models, device, mode) · accuracy vs the players' corrections · per-command error rates ·
latency per stage (p50/p95) · identity (players lost / not seen, low match scores, suspected swaps) ·
crashes and errors · the corrected calls (with crop paths to look at) · the players' notes.

Accuracy is measured against human corrections: a call nobody flagged counts as right, so it is an upper bound,
and it only means something for sessions where the players actually used "Wrong call".
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simon_sees.playtest_data import Session, load_sessions  # noqa: E402

STAGES = [("segment", "SAM (segment)"), ("embed", "DINO (embed)"), ("match", "match"), ("d1", "d1 (decide)"),
          ("vision_total", "vision total"), ("round_trip", "remote round trip"), ("look", "look (wall)"),
          ("tts_call", "TTS (the call)"), ("round_wall", "whole round")]


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v / 1000:.1f} s" if v >= 10_000 else f"{v:.0f} ms"


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def suspected_swaps(s: Session, min_iou: float = 0.5) -> list[dict]:
    """Two players whose boxes traded places between consecutive looks: A now stands where B was and B where A
    was. A heuristic (people do move), so it is listed for a look at the crops, not counted as fact."""
    out = []
    prev: dict[int, list] = {}
    prev_where = None
    for e in s.events:
        looks: list[tuple[str, dict[int, list]]] = []
        if e.get("event") == "locate":
            looks.append((f"{e.get('game_id')} locate", {int(k): v.get("box") for k, v in (e.get("players") or {}).items()
                                                         if v.get("seen") and v.get("box")}))
        elif e.get("event") == "round":
            nframes = max((len(p.get("box_per_frame") or []) for p in e.get("players", [])), default=0)
            for fi in range(nframes):
                looks.append((f"{e.get('round_id')} frame {fi}",
                              {int(p["pid"]): (p.get("box_per_frame") or [None] * nframes)[fi] for p in e.get("players", [])
                               if fi < len(p.get("box_per_frame") or []) and (p.get("box_per_frame") or [])[fi]}))
        elif e.get("event") == "game_start":
            prev, prev_where = {}, None
        for where, boxes in looks:
            pids = sorted(boxes)
            for i, a in enumerate(pids):
                for b in pids[i + 1:]:
                    if a in prev and b in prev and iou(boxes[a], prev[b]) >= min_iou and iou(boxes[b], prev[a]) >= min_iou \
                            and iou(boxes[a], prev[a]) < min_iou:
                        out.append({"where": where, "after": prev_where, "pids": (a, b)})
            prev.update(boxes)
            prev_where = where
    return out


def report(sessions: list[Session]) -> str:
    L: list[str] = ["# Simon Sees playtest report", ""]
    if not sessions:
        return "# Simon Sees playtest report\n\nNo sessions found.\n"
    rounds = [(s, r) for s in sessions for r in s.rounds]
    judged = [(s, r, p) for s, r in rounds for p in r.get("players", []) if p.get("seen")]
    unseen = [(s, r, p) for s, r in rounds for p in r.get("players", []) if not p.get("seen")]
    corrections = [(s, k, f) for s in sessions for k, f in s.corrections.items()]
    games = sum(len(s.meta.get("games") or []) for s in sessions)
    with_fb = [s for s in sessions if s.corrections or s.notes]

    # ---- overview
    L += ["## Overview", "",
          f"- **{len(sessions)} session(s)**, {games} game(s), {len(rounds)} round(s), {len(judged)} player verdicts "
          f"({len(unseen)} more where the player was not seen)",
          f"- players gave feedback in {len(with_fb)} session(s): {len(corrections)} standing 'Wrong call' correction(s), "
          f"{sum(len(s.notes) for s in sessions)} note(s)", ""]
    L += ["| session | mode | backend | segmenter | embedder | decider | device | games | rounds |", "|---|---|---|---|---|---|---|---|---|"]
    for s in sessions:
        m = s.meta.get("models") or next((g.get("models") for g in s.meta.get("games") or [] if g.get("models")), {})
        d = s.meta.get("device") or {}
        dev = d.get("decider") or d.get("segmenter") or d.get("remote") or d.get("requested") or "?"
        if d.get("gpu"):
            dev += f" ({d['gpu']})"
        L.append(f"| {s.name} | {s.meta.get('mode', '?')} | {s.meta.get('backend', '?')} | {m.get('segmenter', '?')} | "
                 f"{m.get('embedder', '?')} | {m.get('decider', '?')} | {dev} | {len(s.meta.get('games') or [])} | {len(s.rounds)} |")
    L.append("")

    # ---- accuracy
    fixed = {(s.name, k) for s, k, _ in corrections}
    false_out = sum(1 for _, _, f in corrections if f.get("given") == "out" and f.get("correct") == "safe")
    missed = sum(1 for _, _, f in corrections if f.get("given") == "safe" and f.get("correct") == "out")
    L += ["## Accuracy vs the players' corrections", ""]
    if judged:
        acc = 1 - len(fixed) / len(judged)
        L.append(f"- **{acc:.1%}** of {len(judged)} verdicts stood (upper bound: an unflagged call counts as right)")
        fb_names = {s.name for s in with_fb}
        jf = [x for x in judged if x[0].name in fb_names]
        if jf and len(jf) != len(judged):
            L.append(f"- in sessions with feedback only: **{1 - len(fixed) / len(jf):.1%}** of {len(jf)}")
        L.append(f"- wrongly sent off (said out, should be safe): **{false_out}** · missed (said safe, should be out): **{missed}**")
        unsure = [x for x in judged if x[2].get("uncertain")]
        if unsure:
            L.append(f"- {len(unsure)} verdict(s) were 'unsure → safe' (benefit of the doubt); "
                     f"{sum(1 for s, r, p in unsure if (s.name, (str(r.get('round_id')), int(p['pid']))) in fixed)} of them were corrected")
    else:
        L.append("No judged verdicts yet.")
    L.append("")

    # ---- per command
    L += ["## Per-command error rates", "", "| command | type | verdicts | corrected | error rate | outs | mean P |",
          "|---|---|---|---|---|---|---|"]
    by: dict[tuple, list] = defaultdict(list)
    for s, r, p in judged:
        typ = "freeze" if r.get("kind") == "freeze" else ("simon says" if r.get("simon") else "trick")
        by[(r.get("command"), typ)].append((s, r, p))
    for (cmd, typ), xs in sorted(by.items(), key=lambda kv: (-sum(1 for s, r, p in kv[1] if (s.name, (str(r.get("round_id")), int(p["pid"]))) in fixed) / len(kv[1]), str(kv[0]))):
        n_fix = sum(1 for s, r, p in xs if (s.name, (str(r.get("round_id")), int(p["pid"]))) in fixed)
        ps = [p.get("p") for _, _, p in xs if p.get("p") is not None]
        L.append(f"| {cmd} | {typ} | {len(xs)} | {n_fix} | {n_fix / len(xs):.0%} | {sum(1 for _, _, p in xs if p.get('out'))} | "
                 f"{(sum(ps) / len(ps)):.2f} |" if ps else f"| {cmd} | {typ} | {len(xs)} | {n_fix} | {n_fix / len(xs):.0%} | 0 | – |")
    L.append("")

    # ---- latency
    L += ["## Latency per stage", "", "| stage | n | p50 | p95 | max |", "|---|---|---|---|---|"]
    for key, label in STAGES:
        xs = [r.get("timings_ms", {}).get(key) for _, r in rounds]
        xs = [x for x in xs if x is not None and (x > 0 or key in ("match",))]
        if xs:
            L.append(f"| {label} | {len(xs)} | {fmt_ms(pct(xs, .5))} | {fmt_ms(pct(xs, .95))} | {fmt_ms(max(xs))} |")
    says = [e.get("tts_ms") for s in sessions for e in s.events if e.get("event") == "say" and e.get("tts_ms") is not None]
    if says:
        L.append(f"| TTS (every line) | {len(says)} | {fmt_ms(pct(says, .5))} | {fmt_ms(pct(says, .95))} | {fmt_ms(max(says))} |")
    lu = [e.get("latency_ms", {}).get("total") for s in sessions for e in s.events if e.get("event") == "lineup"]
    lu = [x for x in lu if x is not None]
    if lu:
        L.append(f"| line-up (vision) | {len(lu)} | {fmt_ms(pct(lu, .5))} | {fmt_ms(pct(lu, .95))} | {fmt_ms(max(lu))} |")
    L.append("")

    # ---- identity
    L += ["## Identity", ""]
    pr = [(s, r, p) for s, r in rounds for p in r.get("players", [])]
    frames_total = sum(len(p.get("seen_per_frame") or []) for _, _, p in pr)
    frames_missed = sum(sum(1 for x in (p.get("seen_per_frame") or []) if not x) for _, _, p in pr)
    L.append(f"- not seen in a round (verdict skipped, Simon says 'I lost you'): **{len(unseen)}** of {len(pr)} player-rounds")
    if frames_total:
        L.append(f"- not matched in a judged frame: {frames_missed} of {frames_total} ({frames_missed / frames_total:.0%})")
    lost_by = Counter(f"{p.get('name')} ({s.name})" for s, r, p in unseen)
    if lost_by:
        L.append("- lost most: " + ", ".join(f"{k} ×{v}" for k, v in lost_by.most_common(5)))
    locates = [e for s in sessions for e in s.events if e.get("event") == "locate"]
    missing_at_start = sum(1 for e in locates for v in (e.get("players") or {}).values() if not v.get("seen"))
    if locates:
        L.append(f"- missing at the pre-game look: {missing_at_start} player-look(s) over {len(locates)} look(s)")
    sims = [x for _, _, p in pr for x in (p.get("sim_per_frame") or []) if x is not None]
    if sims:
        thr = next((s.meta.get("requested", {}).get("match_threshold") for s in sessions if s.meta.get("requested")), 0.45) or 0.45
        low = sum(1 for _, _, p in pr for x, seen in zip(p.get("sim_per_frame") or [], p.get("seen_per_frame") or [])
                  if seen and x is not None and x < thr + 0.1)
        L.append(f"- match score (cosine) p5 / p50: {pct(sims, .05):.2f} / {pct(sims, .5):.2f}; "
                 f"{low} matched frame(s) within 0.1 of the threshold ({thr})")
    swaps = [(s, w) for s in sessions for w in suspected_swaps(s)]
    L.append(f"- suspected identity swaps (two players' boxes traded places): **{len(swaps)}**")
    for s, w in swaps[:10]:
        L.append(f"  - {s.name}: players {w['pids'][0]} ↔ {w['pids'][1]} at {w['where']} (after {w['after']})")
    L.append("")

    # ---- crashes
    errs = [(s, e) for s in sessions for e in s.errors]
    L += ["## Crashes and errors", ""]
    stopped = sum(1 for s in sessions for e in s.events if e.get("event") == "game_stopped")
    unfinished = sum(1 for s in sessions for g in (s.meta.get("games") or []) if not g.get("ended"))
    L.append(f"- {len(errs)} error event(s); {stopped} game(s) stopped from the phone; {unfinished} game(s) never reached full time")
    for (where, msg), n in Counter((e.get("where"), (e.get("message") or "")[:160]) for _, e in errs).most_common(10):
        L.append(f"  - ×{n} `{where}`: {msg}")
    first_tb = next((e for _, e in errs if e.get("traceback")), None)
    if first_tb:
        tail = "\n".join(first_tb["traceback"].strip().splitlines()[-8:])
        L += ["", "First traceback (last lines):", "", "```", tail, "```"]
    L.append("")

    # ---- corrected calls
    L += ["## The corrected calls", ""]
    if corrections:
        L += ["| session | round | command | type | player | said | should be | P | crops |", "|---|---|---|---|---|---|---|---|---|"]
        index = {(s.name, str(r.get("round_id"))): r for s, r in rounds}
        for s, (rid, pid), f in corrections:
            r = index.get((s.name, rid), {})
            p = next((x for x in r.get("players", []) if int(x["pid"]) == pid), {})
            typ = "freeze" if r.get("kind") == "freeze" else ("simon says" if r.get("simon") else "trick")
            crops = " ".join(f"`{s.name}/{c}`" for c in (p.get("crops") or []) if c)
            P = p.get("p")
            L.append(f"| {s.name} | {rid} | {r.get('command', f.get('command'))} | {typ} | {f.get('player')} | {f.get('given')} | "
                     f"{f.get('correct')} | {'–' if P is None else f'{P:.2f}'} | {crops} |")
    else:
        L.append("No corrections.")
    L.append("")

    # ---- notes
    L += ["## The players' notes", ""]
    notes = [(s, n) for s in sessions for n in s.notes]
    for s, n in notes:
        L.append(f"> {n.get('text', '').strip()}\n>\n> — {s.name}, game {n.get('game_id')}\n")
    if not notes:
        L.append("No notes.")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="summarise Simon Sees playtest sessions")
    ap.add_argument("paths", nargs="*", help="playtests/, sessions/, or session folders (default: playtests/ or sessions/)")
    ap.add_argument("--out", help="write the markdown here as well as printing it")
    a = ap.parse_args(argv)
    paths = a.paths or [p for p in ("playtests", "sessions") if Path(p).is_dir()]
    md = report(load_sessions(paths))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
