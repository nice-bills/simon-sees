"""Turn per-example predictions into the results table (bench/results-tinker.{md,json}).

    python bench/referee_report.py --preds name=path.jsonl [name=path.jsonl ...] --split test [--json out.json]

Test labels come from the human check (data/referee_test_verified.jsonl); dropped examples are excluded.
Latency: mean wall time per batch call / per example, as recorded by bench/referee_eval.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from referee_eval import GAME, metrics, verified_labels  # noqa: E402


def load_preds(path: Path) -> tuple[dict[str, float], list]:
    ps, lat = {}, []
    for r in map(json.loads, open(path)):
        if "_latency" in r:
            lat += r["_latency"]
        else:
            ps[r["id"]] = r["p"]
    return ps, lat


def truth(split: str) -> dict[str, dict]:
    if split == "test":
        return {k: v for k, v in verified_labels().items() if not v.get("drop")}
    return {r["id"]: r for r in map(json.loads, open(GAME))}


def evaluate(path: Path, split: str) -> dict:
    ps, lat = load_preds(path)
    t = truth(split)
    preds = [{"command": v["command"], "label": bool(v["label"]), "p": ps[k]} for k, v in t.items() if k in ps]
    m = metrics(preds)
    m["missing"] = sum(1 for k in t if k not in ps)
    if lat:
        n = sum(c for _, c in lat)
        m["latency"] = {"batches": len(lat), "examples": n, "mean_batch_ms": round(statistics.mean(ms for ms, _ in lat), 1),
                        "mean_batch_size": round(n / len(lat), 1),
                        "ms_per_example": round(sum(ms for ms, _ in lat) / n, 1)}
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    out = {}
    for spec in a.preds:
        name, path = spec.split("=", 1)
        out[name] = evaluate(Path(path), a.split)
        m = out[name]
        print(f"{name:28s} n={m['n']:4d} missing={m['missing']:3d} acc={m['accuracy']} macroF1={m['macro_f1']} "
              f"fp={m['fp']} fn={m['fn']} " + " ".join(f"{c}={v['accuracy']}" for c, v in m["per_command"].items()))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
