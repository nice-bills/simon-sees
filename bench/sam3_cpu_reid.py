"""Measure the real vision stack on the sample frames: SAM 3 person counts + latency, and whether the
DINO match keeps every enrolled player correct (especially across 004_swapped-places).

    python bench/sam3_cpu_reid.py --segmenter sam3 --embedder auto --out bench/results-sam3-cpu.json

Ground truth for the sample frames (checked by eye and by torso pixel colour, players numbered left to right
at the line-up 000: P1 navy sweater, P2 floral dress, P3 checked shirt; the colours are only how a human labels
the frames, the pipeline never sees them as words):
    004_swapped-places   left->right the people are P3, P1, P2
    006_centre-player... P2 has left the frame; P1 left, P3 right
    every other frame    P1, P2, P3 in line-up order
If the weights SIGBUS on a network filesystem, set HF_HOME to a copy on local disk (e.g. /tmp/hf).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from simon_sees.engine import players_from_lineup  # noqa: E402
from simon_sees.pipeline import build_stack, process_batch  # noqa: E402
from simon_sees.replay import fit  # noqa: E402
from simon_sees.vision import VisionConfig  # noqa: E402

TRUTH = {  # frame stem prefix -> pids left to right (0-based), None = use line-up order
    "004": [2, 0, 1],
    "006": [0, 2],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmenter", default="sam3")
    ap.add_argument("--embedder", default="auto")
    ap.add_argument("--frames", default=str(ROOT / "samples/people"))
    ap.add_argument("--side", type=int, default=640)
    ap.add_argument("--out", default=str(ROOT / "bench/results-sam3-cpu.json"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    import torch

    t0 = time.perf_counter()
    stack = build_stack(VisionConfig(segmenter=a.segmenter, embedder=a.embedder))
    load_s = time.perf_counter() - t0
    print("models", stack.models, f"load {load_s:.1f}s", flush=True)

    paths = sorted(Path(a.frames).glob("*.jpg"))
    frames = {p.stem: fit(Image.open(p), a.side) for p in paths}

    # 1) SAM alone, per frame (first call is warm-up and reported separately)
    seg = []
    for i, (stem, im) in enumerate(frames.items()):
        t = time.perf_counter()
        dets = stack.segmenter.segment([im])[0]
        ms = 1e3 * (time.perf_counter() - t)
        seg.append({"frame": stem, "people": len(dets), "segment_ms": round(ms, 1), "warmup": i == 0,
                    "scores": [round(d.score, 3) for d in sorted(dets, key=lambda d: d.cx)],
                    "has_masks": all(d.mask is not None for d in dets)})
        print("segment", seg[-1], flush=True)

    # 2) enrol on the line-up, then observe every later frame in order (tracking state carries over)
    stems = list(frames)
    enrol = process_batch(stack, {"kind": "enrol"}, [frames[stems[0]]])
    people = enrol["people"]
    players, roster = players_from_lineup(people, [])
    print("enrol", len(people), enrol["latency_ms"], flush=True)
    reid = []
    for stem in stems[1:]:
        r = process_batch(stack, {"kind": "observe"}, [frames[stem]], list(roster.identities.values()))
        roster.apply(r["updates"])
        fp = r["frames"][0]["players"]
        seen = sorted(((v["box"][0] + v["box"][2]) / 2, int(pid)) for pid, v in fp.items() if v["seen"])
        got = [pid for _, pid in seen]
        want = TRUTH.get(stem[:3], list(range(len(people))))
        reid.append({"frame": stem, "left_to_right": [f"P{p + 1}" for p in got],
                     "expected": [f"P{p + 1}" for p in want], "correct": got == want,
                     "sims": {f"P{int(k) + 1}": v["sim"] for k, v in fp.items()}, "latency_ms": r["latency_ms"]})
        print("observe", reid[-1], flush=True)

    warm = [s["segment_ms"] for s in seg if not s["warmup"]]
    out = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M %Z"),
        "machine": {"platform": platform.platform(), "cpus": os.cpu_count(), "torch": torch.__version__,
                    "threads": torch.get_num_threads()},
        "models": stack.models, "frame_side_px": a.side, "load_s": round(load_s, 1),
        "segment": seg,
        "segment_ms_median_warm": round(statistics.median(warm), 1) if warm else None,
        "enrol": {"people": len(people), "latency_ms": enrol["latency_ms"]},
        "reid": reid, "reid_all_correct": all(x["correct"] for x in reid),
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("WROTE", a.out, "median warm segment ms", out["segment_ms_median_warm"], "all correct", out["reid_all_correct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
