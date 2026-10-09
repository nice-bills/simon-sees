# Benchmark: decision model vs generative VLM as a referee

Same frames, same question, two kinds of open model:

| | decision model (d1) | generative VLM |
|---|---|---|
| how it answers | one forward pass, reads P(yes) at the answer slot | prompt + "answer yes or no", decode tokens, parse the word |
| output | calibrated probability, threshold it | a word (or a sentence you have to parse) |
| tokens generated | 0 | ≥ 1 (more if it gets chatty) |

## 1. Build a labeled set from your own field test

Run a game with `--save-frames captures` (frames stay on your laptop). Afterwards:

```bash
python scripts/summarize_log.py logs/decisions.jsonl     # writes logs/decisions_to_label.csv
```

Watch your video, and write rows in `bench/labels.csv` (format in `labels.example.csv`): one row per
(frame, player, command) with `label` 1 if the player really did it. For freeze rows put the earlier frame
in `image_before` and label 1 if the player really moved. 40–100 rows is enough for a blog-post table.

## 2. Run

```bash
pip install -r requirements-model.txt
# Apple Silicon / NVIDIA laptop:
python bench/benchmark.py --labels bench/labels.csv \
    --d1 LiquidAI/d1-3B LiquidAI/d1-omni-600M \
    --gen LiquidAI/LFM2.5-VL-3B \
    --out bench/results
```

`LiquidAI/LFM2.5-VL-3B` is d1-3B's own base model, so the comparison isolates the decision head vs
generation. Swap in `google/gemma-3-4b-it` (accept its licence on Hugging Face first) or
`LiquidAI/LFM2.5-VL-1.6B` for other baselines.

On a small machine, run one model per process and merge:

```bash
python bench/benchmark.py --labels bench/labels.csv --d1 LiquidAI/d1-3B --gen --out bench/r_d1
python bench/benchmark.py --labels bench/labels.csv --d1 --gen LiquidAI/LFM2.5-VL-3B --out bench/r_gen \
    --merge bench/r_d1/results.csv
```

Outputs `results.csv` (every answer), `summary.csv` and `summary.md` (paste into the post).

## Sandbox smoke run (CPU, 2 threads, stock photos)

`results-sandbox-cpu.md` is a smoke test on 23 hand-labeled rows from 7 stock photos (not committed:
copyrighted) on a 2-vCPU, no-GPU sandbox. It proves the pipeline end to end; it is **not** the headline
number. Your laptop/GPU run on your own field frames is.
