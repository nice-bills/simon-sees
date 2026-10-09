"""Fine-tune the referee on Tinker: LoRA SFT of Qwen3.5-4B on (player crop, Simon command) -> yes/no.

    python scripts/train_referee.py --data data/referee --run runs/referee-v1 [--epochs 2] [--lr 4e-4] [--rank 32]

The loss is cross-entropy on the answer word ("yes"/"no") and <|im_end|> only; the prompt is exactly the
game's (simon_sees.tinker_referee.referee_prompt over commands.py's check + hints). Horizontal flips are a free
augmentation (every rule is left/right symmetric). Every --eval-every steps the weights are saved for sampling
and scored on the val split (accuracy at P(yes) >= 0.5); the best val checkpoint is the one to evaluate on
test. Progress, checkpoints and token counts go to {run}/log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simon_sees.commands import BY_ID  # noqa: E402
from simon_sees.decider import action_question  # noqa: E402
from simon_sees.tinker_referee import (BASE_MODEL, TinkerScorer, get_tokenizer, image_tokens,  # noqa: E402
                                       referee_prompt, training_datum)


def prompt_for(cmd: str) -> str:
    c = BY_ID[cmd]
    q = action_question(c.check, c.true_hint, c.false_hint)
    return referee_prompt(q["instructions"], q.get("criteria"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "referee")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="manifest.jsonl", help="manifest file inside --data (v2: manifest-v2.jsonl)")
    ap.add_argument("--oversample-rare", type=int, default=1,
                    help="repeat the positives of commands with < 60 train positives this many times")
    args = ap.parse_args(argv)

    import tinker

    args.run.mkdir(parents=True, exist_ok=True)
    log = open(args.run / "log.jsonl", "a")

    def emit(**kw):
        kw["t"] = round(time.time(), 1)
        log.write(json.dumps(kw) + "\n")
        log.flush()
        print(json.dumps(kw)[:300], flush=True)

    rows = [json.loads(l) for l in open(args.data / args.manifest)]
    train = [r for r in rows if r["split"] == "train"]
    if args.oversample_rare > 1:
        n_pos = {c: sum(1 for r in train if r["command"] == c and r["label"]) for c in {r["command"] for r in train}}
        train += [r for r in train if r["label"] and n_pos[r["command"]] < 60] * (args.oversample_rare - 1)
    val = [r for r in rows if r["split"] == "val"]
    tok = get_tokenizer()
    rng = random.Random(args.seed)
    steps_per_epoch = len(train) // args.batch
    total_steps = int(steps_per_epoch * args.epochs)
    emit(event="start", train=len(train), val=len(val), steps=total_steps, **{k: str(v) for k, v in vars(args).items()})

    import os

    sc = tinker.ServiceClient(**({"api_key": os.environ["TINKER_API_KEY"]} if os.environ.get("TINKER_API_KEY") else {}))
    tc = sc.create_lora_training_client(base_model=args.base, rank=args.rank, seed=args.seed)

    def load(r, flip=False):
        im = Image.open(args.data / r["crop"]).convert("RGB")
        return im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else im

    val_items = [(load(r), prompt_for(r["command"])) for r in val]

    def evaluate(step):
        path = tc.save_weights_for_sampler(name=f"step{step:04d}").result().path
        scorer = TinkerScorer(client=tc.create_sampling_client(path) if hasattr(tc, "create_sampling_client")
                              else sc.create_sampling_client(model_path=path), tokenizer=tok)
        ps = scorer.score(val_items)
        acc = sum((p >= 0.5) == r["label"] for p, r in zip(ps, val)) / len(val)
        nll = -sum(__import__("math").log(max(1e-6, p if r["label"] else 1 - p)) for p, r in zip(ps, val)) / len(val)
        emit(event="eval", step=step, val_acc=round(acc, 4), val_nll=round(nll, 4), sampler_path=path)
        return acc, path

    order: list[int] = []
    tokens_seen = 0
    step = 0
    best = (-1.0, None, -1)
    while step < total_steps:
        if len(order) < args.batch:
            idx = list(range(len(train)))
            rng.shuffle(idx)
            order += idx
        batch_idx, order = order[:args.batch], order[args.batch:]
        lr = args.lr * max(0.0, 1 - step / total_steps)  # linear decay to 0
        data = []
        for i in batch_idx:
            r = train[i]
            img = load(r, flip=rng.random() < 0.5)
            data.append(training_datum(img, prompt_for(r["command"]), bool(r["label"]), tok))
            tokens_seen += image_tokens(img) + 90
        t0 = time.time()
        fb = tc.forward_backward(data, "cross_entropy")
        op = tc.optim_step(tinker.AdamParams(learning_rate=lr))
        out = fb.result()
        op.result()
        loss = None
        try:
            lw = [(float(sum(o["logprobs"].data[j] * w for j, w in enumerate(d.loss_fn_inputs["weights"].data))),
                   float(sum(d.loss_fn_inputs["weights"].data)))
                  for o, d in zip(out.loss_fn_outputs, data)]
            loss = -sum(a for a, _ in lw) / max(1e-9, sum(b for _, b in lw))
        except Exception:  # noqa: BLE001
            pass
        step += 1
        emit(event="step", step=step, lr=round(lr, 7), loss=None if loss is None else round(loss, 4),
             tokens=tokens_seen, s=round(time.time() - t0, 2))
        if step % args.eval_every == 0 or step == total_steps:
            acc, path = evaluate(step)
            if acc >= best[0]:
                best = (acc, path, step)
    state = tc.save_state(name="final").result().path
    emit(event="done", best_val_acc=best[0], best_sampler_path=best[1], best_step=best[2], state_path=state,
         tokens=tokens_seen, steps=step)


if __name__ == "__main__":
    main()
