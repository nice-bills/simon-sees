"""Referee latency: one crop, and one round (all players' crops in one batched call), on the game-slice crops.

    python bench/referee_latency.py --model tinker:<tinker:// path> --repeats 5 --out bench/latency-tinker.json
    python bench/referee_latency.py --model d1:LiquidAI/d1-omni-600M --repeats 3 --out bench/latency-d1.json

Round sizes 3 and 6 players (the sample photos have 3 players per frame). Wall time per call, measured after
one warm-up call. For Tinker this includes the network round trip from where this runs to Tinker's servers.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from referee_eval import GAME, question  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sizes", default="1,3,6")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(GAME)]
    items = [(Image.open(ROOT / r["crop"]).convert("RGB"), r["command"]) for r in rows]
    if a.model.startswith("tinker"):
        import tinker

        from simon_sees.tinker_referee import TinkerScorer, get_tokenizer, referee_prompt

        sc = tinker.ServiceClient(**({"api_key": os.environ["TINKER_API_KEY"]} if os.environ.get("TINKER_API_KEY") else {}))
        path = a.model.split(":", 1)[1] if ":" in a.model else None
        cl = sc.create_sampling_client(model_path=path) if path else sc.create_sampling_client(base_model="Qwen/Qwen3.5-4B")
        s = TinkerScorer(client=cl, tokenizer=get_tokenizer())

        def call(batch):
            return s.score([(im, referee_prompt(question(c)["instructions"], question(c).get("criteria"))) for im, c in batch])
    elif a.model.startswith("d1:"):
        from simon_sees.decider import D1Decider, Request

        d = D1Decider(a.model[3:], device="cpu")

        def call(batch):
            return d.decide([Request([im], {"did": question(c)}) for im, c in batch])
    else:
        raise SystemExit("model: tinker:<path> | d1:<hf id>")
    sizes = [int(x) for x in a.sizes.split(",")]
    call(items[:1])  # warm-up
    out = {"model": a.model, "cpu_count": os.cpu_count(), "sizes": {}}
    k = 0
    for n in sizes:
        ms = []
        for _ in range(a.repeats):
            batch = [items[(k + j) % len(items)] for j in range(n)]
            k += n
            t0 = time.perf_counter()
            call(batch)
            ms.append(1e3 * (time.perf_counter() - t0))
        out["sizes"][str(n)] = {"calls": len(ms), "median_ms": round(statistics.median(ms), 1),
                                "min_ms": round(min(ms), 1), "max_ms": round(max(ms), 1),
                                "median_ms_per_crop": round(statistics.median(ms) / n, 1)}
        print(n, out["sizes"][str(n)], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
