"""Score referee models on the same examples and write per-example predictions + metrics.

    python bench/referee_eval.py --data data/referee --split test --model tinker-base   --out bench/results/preds-base.jsonl
    python bench/referee_eval.py --data data/referee --split test --model tinker:tinker://.../sampler_weights/final
    python bench/referee_eval.py --data data/referee --split test --model d1:LiquidAI/d1-omni-600M
    python bench/referee_eval.py --data data/referee --split game  ...   # the 7 samples/people frames' players

Every model gets the game's own question for the command (commands.py `check` + its yes/no hints):
d1 through decider.D1Decider (noul question), the Qwen models through simon_sees.tinker_referee (the
same text, P(yes) from the first answer token's logprobs). The decision threshold is 0.5 for all.
`metrics()` reads the human-verified labels (data/referee_test_verified.jsonl) when present.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simon_sees.commands import BY_ID  # noqa: E402
from simon_sees.decider import Request, action_question  # noqa: E402

VERIFIED = ROOT / "data" / "referee_test_verified.jsonl"
GAME = ROOT / "data" / "referee_game_slice.jsonl"


def question(cmd: str) -> dict:
    c = BY_ID[cmd]
    return action_question(c.check, c.true_hint, c.false_hint)


def load_rows(data: Path, split: str, manifest: str = "manifest.jsonl") -> list[dict]:
    if split == "game":
        rows = [json.loads(l) for l in open(GAME)]
        for r in rows:
            r["path"] = str(ROOT / r["crop"])
        return rows
    rows = [json.loads(l) for l in open(data / manifest) if json.loads(l)["split"] == split]
    if split == "test":  # rows to score = every human-checked test example (labels applied in final_rows)
        v = verified_labels()
        if v:
            rows = [r for r in rows if r["id"] in v]
    for r in rows:
        r["path"] = str(data / r["crop"])
    return rows


def verified_labels() -> dict[str, dict]:
    if not VERIFIED.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, open(VERIFIED))}


def final_rows(rows: list[dict], split: str) -> list[dict]:
    """Apply the human check: test rows take the verified label; rows marked drop are removed."""
    if split != "test":
        return rows
    v = verified_labels()
    if not v:
        return rows
    out = []
    for r in rows:
        h = v.get(r["id"])
        if h is None or h.get("drop"):
            continue
        out.append({**r, "label": bool(h["label"]), "rule_label": r["label"]})
    return out


def metrics(preds: list[dict], thr: float = 0.5) -> dict:
    def f1(tp, fp, fn):
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    tp = sum(1 for p in preds if p["label"] and p["p"] >= thr)
    tn = sum(1 for p in preds if not p["label"] and p["p"] < thr)
    fp = sum(1 for p in preds if not p["label"] and p["p"] >= thr)
    fn = sum(1 for p in preds if p["label"] and p["p"] < thr)
    n = len(preds)
    per = defaultdict(list)
    for p in preds:
        per[p["command"]].append(p)
    out = {"n": n, "accuracy": round((tp + tn) / n, 4) if n else None,
           "macro_f1": round((f1(tp, fp, fn) + f1(tn, fn, fp)) / 2, 4) if n else None,
           "tp": tp, "tn": tn, "fp": fp, "fn": fn, "per_command": {}}
    for c, ps in sorted(per.items()):
        ok = sum(1 for p in ps if (p["p"] >= thr) == p["label"])
        out["per_command"][c] = {"n": len(ps), "pos": sum(bool(p["label"]) for p in ps), "accuracy": round(ok / len(ps), 4)}
    return out


def score_tinker(rows, model_path: str | None, batch: int = 64, base_model: str = "Qwen/Qwen3.5-4B"):
    import tinker

    from simon_sees.tinker_referee import TinkerScorer, get_tokenizer, referee_prompt

    import os

    sc = tinker.ServiceClient(**({"api_key": os.environ["TINKER_API_KEY"]} if os.environ.get("TINKER_API_KEY") else {}))
    cl = sc.create_sampling_client(model_path=model_path) if model_path else sc.create_sampling_client(base_model=base_model)
    s = TinkerScorer(client=cl, tokenizer=get_tokenizer())
    ps, lat = [], []
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        items = []
        for r in part:
            q = question(r["command"])
            items.append((Image.open(r["path"]).convert("RGB"), referee_prompt(q["instructions"], q.get("criteria"))))
        t0 = time.perf_counter()
        ps += s.score(items)
        lat.append((1e3 * (time.perf_counter() - t0), len(items)))
    return ps, lat


def score_d1(rows, model_id: str, batch: int = 8):
    from simon_sees.decider import D1Decider

    d = D1Decider(model_id, device="cpu")
    ps, lat = [], []
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        reqs = [Request([Image.open(r["path"]).convert("RGB")], {"did": question(r["command"])}) for r in part]
        t0 = time.perf_counter()
        res = d.decide(reqs)
        lat.append((1e3 * (time.perf_counter() - t0), len(reqs)))
        ps += [x.p("did") for x in res]
    return ps, lat


def score_lfm(rows, model_id: str):
    """A small generative VLM (LFM2.5-VL-450M) asked the same prompt; P(yes) from the first token's logits."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from simon_sees.tinker_referee import answer_token_ids, referee_prompt

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True, dtype=torch.float32).eval()
    yes, no = answer_token_ids(proc.tokenizer)
    ps, lat = [], []
    with torch.inference_mode():
        for r in rows:
            q = question(r["command"])
            msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(r["path"]).convert("RGB")},
                                                 {"type": "text", "text": referee_prompt(q["instructions"], q.get("criteria"))}]}]
            inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                              return_tensors="pt")
            t0 = time.perf_counter()
            lp = torch.log_softmax(model(**inputs).logits[0, -1].float(), -1)
            lat.append((1e3 * (time.perf_counter() - t0), 1))
            py = sum(float(lp[i].exp()) for i in yes)
            pn = sum(float(lp[i].exp()) for i in no)
            ps.append(py / (py + pn) if py + pn > 0 else 0.5)
    return ps, lat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "referee")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", required=True, help="tinker-base | tinker:<tinker:// path> | d1:<hf id>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--skip-scored", action="store_true", help="skip ids already in --out (resume / top up)")
    args = ap.parse_args(argv)
    rows = load_rows(args.data, args.split, args.manifest)
    if args.count:
        rows = rows[args.start:args.start + args.count]
    if args.skip_scored and args.out.exists():
        done = {r.get("id") for r in map(json.loads, open(args.out))}
        rows = [r for r in rows if r["id"] not in done]
    if args.model == "tinker-base":
        ps, lat = score_tinker(rows, None)
    elif args.model.startswith("tinker-base:"):
        ps, lat = score_tinker(rows, None, base_model=args.model[len("tinker-base:"):])
    elif args.model.startswith("tinker:"):
        ps, lat = score_tinker(rows, args.model[len("tinker:"):])
    elif args.model.startswith("lfm:"):
        ps, lat = score_lfm(rows, args.model[4:])
    elif args.model.startswith("d1:"):
        ps, lat = score_d1(rows, args.model[3:])
    else:
        raise SystemExit(f"unknown model {args.model}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for r, p in zip(rows, ps):
            f.write(json.dumps({"id": r["id"], "command": r["command"], "label": r["label"], "p": round(float(p), 5)}) + "\n")
        f.write(json.dumps({"_latency": lat, "model": args.model}) + "\n")
    scored = final_rows([{**r, "p": p} for r, p in zip(rows, ps)], args.split)
    print(json.dumps(metrics([{"command": r["command"], "label": bool(r["label"]), "p": r["p"]} for r in scored]))[:400])


if __name__ == "__main__":
    main()
