"""Turn logs/decisions.jsonl (from real games) into the numbers the write-up needs.

    python scripts/summarize_log.py logs/decisions.jsonl

Prints: games, rounds, decisions, latency percentiles per method, how often the referee was unsure,
eliminations by reason, and a CSV skeleton (decisions_to_label.csv) you can fill with ground truth
from your video to measure referee accuracy in the field.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main(path: str) -> int:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    games = [r for r in rows if r["event"] == "game_start"]
    decisions = [r for r in rows if r["event"] == "decision"]
    verdicts = [r for r in rows if r["event"] == "verdict"]
    print(f"games: {len(games)}  rounds: {len({(r['game_id'], r['round']) for r in verdicts})}  "
          f"player-decisions: {len(verdicts)}")
    by_method = defaultdict(list)
    seen = set()
    for r in decisions:
        key = (r["game_id"], r["round"], r.get("method"))
        if key in seen:  # one batch call serves every player in the round: count its latency once
            continue
        seen.add(key)
        by_method[(r.get("model"), r.get("method"))].append(r["latency_ms"])
    print("\nbatch latency (one call judges every player in the round):")
    print("| model | method | calls | median ms | p90 ms |\n|---|---|---|---|---|")
    for (model, method), lat in by_method.items():
        print(f"| {model} | {method} | {len(lat)} | {statistics.median(lat):.1f} | {pct(lat, 0.9):.1f} |")
    reasons = Counter(r["reason"] for r in verdicts)
    unsure = sum(1 for r in verdicts if r.get("uncertain"))
    print(f"\nverdict reasons: {dict(reasons)}")
    print(f"benefit-of-the-doubt calls (uncertain band): {unsure}/{len(verdicts)}")
    out = Path(path).with_name("decisions_to_label.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "round", "call", "player", "p_action", "out", "truth_did_it (fill 1/0 from video)", "frames"])
        frames = {(d["game_id"], d["round"], d["player"]): ";".join(d.get("frames") or []) for d in decisions}
        for r in verdicts:
            w.writerow([r["game_id"], r["round"], r["call"], r["player"], r.get("p_action"), r["out"], "",
                        frames.get((r["game_id"], r["round"], r["player"]), "")])
    print(f"\nwrote {out}: fill the truth column from your video, then compute field accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "logs/decisions.jsonl"))
