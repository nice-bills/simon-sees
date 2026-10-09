# Sandbox smoke run: CPU only, 2 vCPU, 8 GB RAM (2026-10-08)

> Measured before the identity redesign, with whole-frame questions. The finding below that d1-omni-600M
> could not answer clothing-conditional questions is why the game now finds people with SAM 3, tells them
> apart by their line-up photo (DINO embeddings) and asks d1 about one player's own crop. Kept as a record.

**This is a pipeline smoke test, not the headline benchmark.** 23 hand-labeled referee questions over 7
stock photos, run with `bench/benchmark.py` in a no-GPU cloud sandbox (linux aarch64, torch 2.14.1+cpu,
transformers 5.17.0, 2 threads, fp32, frames resized to 384 px, one question per call, 1 timed warm call per
item). The photos are not committed (copyright); their URLs and the labels are in `labels-sandbox-stock.csv`.

d1-3B is missing because it could not run in this sandbox: its weights are 6.2 GB in bf16 (12.5 GB in fp32,
the card's CPU default) and the box had 8 GB of RAM; loading succeeded (30 s), but the first forward pass died
with SIGBUS while paging the memory-mapped weights from the network filesystem.

| model | type | n | accuracy | median latency | p90 latency | tokens generated | Brier | device |
|---|---|---|---|---|---|---|---|---|
| LiquidAI/d1-omni-600M (experimental) | decision | 23 | 78% (18/23) | 2552.1 ms | 3018.9 ms | 0 | 0.2096 | cpu (fp32) |
| LiquidAI/LFM2.5-VL-450M | generative VLM | 23 | 87% (20/23) | 2582.5 ms | 2677.0 ms | 2.1 | n/a | cpu (fp32) |

Where each model was wrong:

| item | truth | d1-omni-600M P(yes) | LFM2.5-VL-450M answer |
|---|---|---|---|
| tree1: woman in tree pose, one foot | yes | 0.812 ✓ | "No" ✗ |
| head1: man, hands on head | yes | 0.000 ✗ | "Yes" ✓ |
| head2: man, hands behind head | yes | 0.000 ✗ | "Yes" ✓ |
| head2: is he standing on one foot? | no | 0.996 ✗ | "No" ✓ |
| sit1: seated man, standing on one foot? | no | 0.680 ✗ | "No" ✓ |
| head2: "is there a person wearing a white t-shirt?" | yes | 0.000 ✗ | "Yes" ✓ |
| freeze, same photo twice (tree1): did she move? | no | 0.000 ✓ | "Yes." ✗ |
| freeze, same photo twice (head1): did he move? | no | 0.000 ✓ | "Yes." ✗ |

Other observations from the same session:

- d1-omni-600M answered "no" to every clothing-conditional presence question tried ("is there a person
  wearing a white t-shirt?", 3 phrasings x 3 photos where the person was plainly there) but was right every
  time on "is there a person in this photo?" (1.00 on people, 0.00 on a photo of cats). Hence the game's
  `--visibility any_person` mode, selected automatically for omni models.
- Several questions over one image cost more than one on the omni encoder (1 question ≈ 2.9 s, 4 questions
  ≈ 5.7 s on this CPU), unlike d1-3B whose card reports 3 questions at ≈1.3x one.
- End-to-end dry run with d1-omni-600M (1 player, replayed photos): one pose round = 2 frames x (1 player
  question + 1 visibility question) in one batched call = 6.2 s on this CPU.
- Kokoro-82M TTS on the same CPU: 2.4 s to synthesise a 3.05 s line; 33 fixed lines prewarmed in 49 s.
