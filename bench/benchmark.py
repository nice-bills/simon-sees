"""Decision model vs generative VLM on the same referee questions and the same frames.

For every labeled row we ask the game's own per-crop question ("Look at the person in this photo.
Is that person standing on one foot?") two ways. Each image should be one player's crop, the way the
game asks it after SAM 3 has segmented the frame:

  * d1 (decision model): one forward pass, calibrated P(yes), zero output tokens.
  * a generative open VLM (default LiquidAI/LFM2.5-VL-1.6B; d1-3B's own backbone is LiquidAI/LFM2.5-VL-3B,
    and google/gemma-3-4b-it works too): chat prompt + "Answer yes or no.", greedy decode, parse the word.

Outputs: <out>/results.csv (one row per model x item), <out>/summary.csv and <out>/summary.md (the table).

    python bench/benchmark.py --labels bench/labels.csv \
        --d1 LiquidAI/d1-3B --gen LiquidAI/LFM2.5-VL-3B --out bench/results

labels.csv columns:
    id, image, image_before, player, command, question, label
  - image: path (relative to the CSV) of the frame; for freeze rows also image_before (the earlier frame)
  - player: optional free-text note (ignored by the question); command: an id from simon_sees/commands.py
  - question: optional free text that overrides the command's question
  - label: 1 if the true answer is yes, else 0
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from simon_sees.commands import BY_ID  # noqa: E402
from simon_sees.decider import action_question, pick_device, pick_dtype  # noqa: E402


def load_items(path: Path, side: int) -> list[dict]:
    items = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("image"):
                continue
            imgs = []
            if row.get("image_before"):
                imgs.append(path.parent / row["image_before"])
            imgs.append(path.parent / row["image"])
            if row.get("question"):
                q = {"type": "noul", "instructions": row["question"]}
            else:
                cmd = BY_ID[row["command"]]
                q = action_question(cmd.check, cmd.true_hint, cmd.false_hint)
            pics = []
            for p in imgs:
                im = Image.open(p).convert("RGB")
                im.thumbnail((side, side))
                pics.append(im)
            items.append({"id": row.get("id") or Path(row["image"]).stem, "images": pics, "question": q,
                          "label": int(row["label"]), "command": row.get("command", "")})
    return items


def bench_d1(model_id: str, items: list[dict], device: str, dtype: str, repeats: int) -> list[dict]:
    import torch
    from transformers import AutoModel

    dev = pick_device(device)
    dt = pick_dtype(model_id, dev, dtype)
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dt).to(dev).eval()
    load_s = time.perf_counter() - t0
    out = []
    with torch.inference_mode():
        model.system_one(None, {"q": items[0]["question"]}, images=items[0]["images"])  # warm-up, not timed
        for it in items:
            times, p = [], None
            for _ in range(repeats):
                t = time.perf_counter()
                r = model.system_one(None, {"q": it["question"]}, images=it["images"])
                if dev == "cuda":
                    torch.cuda.synchronize()
                times.append(1e3 * (time.perf_counter() - t))
                p = float(r["answers"]["q"]["noul"])
            out.append({"model": model_id, "kind": "decision (d1)", "id": it["id"], "command": it["command"],
                        "label": it["label"], "p_yes": round(p, 5), "answer": "", "pred": int(p >= 0.5),
                        "latency_ms": round(statistics.median(times), 2), "output_tokens": 0,
                        "device": dev, "dtype": str(dt).replace("torch.", ""), "load_s": round(load_s, 1)})
            print(f"  {model_id} {it['id']}: p={p:.3f} label={it['label']} {out[-1]['latency_ms']} ms", flush=True)
    del model
    gc.collect()
    return out


def bench_generative(model_id: str, items: list[dict], device: str, dtype: str, repeats: int, max_new_tokens: int) -> list[dict]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dev = pick_device(device)
    dt = pick_dtype(model_id, dev, dtype)
    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True, dtype=dt).to(dev).eval()
    load_s = time.perf_counter() - t0

    def ask(it):
        content = [{"type": "image", "image": im} for im in it["images"]]
        content.append({"type": "text", "text": it["question"]["instructions"] + " Answer with only one word: yes or no."})
        msgs = [{"role": "user", "content": content}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt").to(dev)
        t = time.perf_counter()
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = 1e3 * (time.perf_counter() - t)
        new = ids[0][inputs["input_ids"].shape[-1]:]
        return proc.decode(new, skip_special_tokens=True).strip(), ms, int(new.shape[-1])

    out = []
    with torch.inference_mode():
        ask(items[0])  # warm-up
        for it in items:
            times, text, ntok = [], "", 0
            for _ in range(repeats):
                text, ms, ntok = ask(it)
                times.append(ms)
            word = text.lower().strip(" .!\"'\n").split()
            word = word[0].strip(".,!") if word else ""
            pred = 1 if word.startswith("yes") else 0 if word.startswith("no") else -1  # -1 = unparseable, counts wrong
            out.append({"model": model_id, "kind": "generative VLM", "id": it["id"], "command": it["command"],
                        "label": it["label"], "p_yes": "", "answer": text[:40], "pred": pred,
                        "latency_ms": round(statistics.median(times), 2), "output_tokens": ntok,
                        "device": dev, "dtype": str(dt).replace("torch.", ""), "load_s": round(load_s, 1)})
            print(f"  {model_id} {it['id']}: {text!r} label={it['label']} {out[-1]['latency_ms']} ms", flush=True)
    del model
    gc.collect()
    return out


def summarize(rows: list[dict]) -> list[dict]:
    summ = []
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == model]
        lat = sorted(r["latency_ms"] for r in rs)
        acc = sum(int(r["pred"]) == int(r["label"]) for r in rs) / len(rs)
        brier = ""
        if rs[0]["p_yes"] != "":
            brier = round(sum((float(r["p_yes"]) - int(r["label"])) ** 2 for r in rs) / len(rs), 4)
        summ.append({"model": model, "kind": rs[0]["kind"], "n": len(rs), "accuracy": round(acc, 3),
                     "median_ms": round(statistics.median(lat), 1), "p90_ms": round(lat[min(len(lat) - 1, int(0.9 * len(lat)))], 1),
                     "output_tokens": round(statistics.mean(int(r["output_tokens"]) for r in rs), 1),
                     "brier": brier, "unparseable": sum(int(r["pred"]) == -1 for r in rs),
                     "device": rs[0]["device"], "dtype": rs[0]["dtype"]})
    return summ


def write(out: Path, rows: list[dict], summ: list[dict], note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("results.csv", rows), ("summary.csv", summ)):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    lines = [f"# Referee benchmark\n", note, "",
             "| model | type | n | accuracy | median latency | p90 latency | output tokens | Brier (lower is better) | device |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summ:
        lines.append(f"| {s['model']} | {s['kind']} | {s['n']} | {s['accuracy']:.0%} | {s['median_ms']} ms | {s['p90_ms']} ms | "
                     f"{s['output_tokens']} | {s['brier'] if s['brier'] != '' else 'n/a'} | {s['device']} ({s['dtype']}) |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--d1", nargs="*", default=["LiquidAI/d1-3B"])
    ap.add_argument("--gen", nargs="*", default=["LiquidAI/LFM2.5-VL-1.6B"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--side", type=int, default=384)
    ap.add_argument("--repeats", type=int, default=3, help="timed calls per item; the median is reported")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "results")
    ap.add_argument("--note", default="")
    ap.add_argument("--merge", nargs="*", type=Path, help="results.csv files from earlier runs to include")
    a = ap.parse_args()

    import torch

    items = load_items(a.labels, a.side)
    print(f"{len(items)} labeled items from {a.labels}")
    rows: list[dict] = []
    if a.merge:  # results of earlier runs (e.g. one model per process on a small machine)
        for path in a.merge:
            with path.open(newline="") as f:
                rows += [dict(r, latency_ms=float(r["latency_ms"])) for r in csv.DictReader(f)]
    for m in a.d1:
        rows += bench_d1(m, items, a.device, a.dtype, a.repeats)
        write(a.out, rows, summarize(rows), "partial")
    for m in a.gen:
        rows += bench_generative(m, items, a.device, a.dtype, a.repeats, a.max_new_tokens)
        write(a.out, rows, summarize(rows), "partial")
    note = a.note or (f"{len(items)} items from `{a.labels.name}`, frames resized to {a.side}px, median of {a.repeats} "
                      f"warm calls per item, one question per call. Host: {platform.machine()} / {platform.system()}, "
                      f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads.")
    write(a.out, rows, summarize(rows), note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
