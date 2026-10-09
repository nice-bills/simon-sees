# The Simon Sees referee, fine-tuned on Tinker: results

Measured numbers only. Machine-readable copy: `bench/results-tinker.json`. Scripts: `bench/referee_eval.py`
(predictions), `bench/referee_report.py` (tables), `bench/referee_latency.py` (latency), `scripts/train_referee.py`
(training), `scripts/mine_hard_negatives.py` (v2 data), `scripts/export_referee.py` (weights).

**The task.** One player crop, one Simon command, one question, exactly as the game asks it: *"Look at the person in
this photo. Is that person standing on one foot? Yes means: ... No means: ... Answer with one word: yes or no."*
P(yes) comes from the first answer token's logprobs (one generated token); the decision threshold is 0.5 for every
model. Six pose commands: `airplane`, `arms_up`, `hands_head`, `one_foot`, `squat`, `touch_toes`.

**The referee now:** Qwen3.5-4B + a rank-32 LoRA (v2), `tinker://7bfed040-1481-5605-ad1f-a7cdee9763ca:train:0/sampler_weights/step0100`.

## TEST: 222 human-verified COCO crops (81 yes / 141 no, 210 images never used for training)

| model | n | accuracy | macro-F1 | false pos | false neg |
|---|---|---|---|---|---|
| untuned-4b | 222 | 87.4% | 0.871 | 27 | 1 |
| untuned-35b | 222 | 88.7% | 0.882 | 19 | 6 |
| d1-omni-600m | 222 | 63.1% | 0.629 | 65 | 17 |
| tuned-v1 | 222 | 87.8% | 0.874 | 22 | 5 |
| tuned-v2 | 222 | 93.2% | 0.928 | 9 | 6 |

- v2 vs the untuned 4B: right on 20 crops the untuned model got wrong, wrong on 7 it got right (exact McNemar
  p = 0.019). Bootstrap 95% interval for v2's accuracy: 89.6–96.4%.
- v2 vs the untuned 35B (Qwen3.6-35B-A3B, ~9x the parameters, the train/val label-cleaning teacher): 18 vs 8,
  p = 0.076. v2 is ahead on this set but the difference is not significant at n = 222.
- v1 (the first run) was a wash: 87.8% vs 87.4% untuned.
- d1-omni-600M is the experimental 600M d1 checkpoint (the only d1 that fits this 8 GB box). d1-3B was not measured here.

### Per command (TEST accuracy)

| model | airplane (n=31, 16 yes) | arms_up (n=32, 14 yes) | hands_head (n=36, 6 yes) | one_foot (n=33, 12 yes) | squat (n=46, 25 yes) | touch_toes (n=44, 8 yes) |
|---|---|---|---|---|---|---|
| untuned-4b | 100.0% | 84.4% | 94.4% | 90.9% | 84.8% | 75.0% |
| untuned-35b | 96.8% | 78.1% | 100.0% | 87.9% | 93.5% | 77.3% |
| d1-omni-600m | 71.0% | 71.9% | 72.2% | 39.4% | 69.6% | 54.5% |
| tuned-v1 | 96.8% | 84.4% | 97.2% | 84.9% | 89.1% | 77.3% |
| tuned-v2 | 96.8% | 84.4% | 97.2% | 93.9% | 89.1% | 97.7% |

v2's gain is almost all false positives removed: 27 → 9 against the untuned 4B. `touch_toes` went from 75% to
97.7% and `one_foot` from 90.9% to 93.9%. `arms_up` did not move (84.4%): its errors are now 4 false positives and
1 miss, mostly people mid-jump with arms up (labelled no or ambiguous by hand). v2's 15 errors: arms_up 4 FP + 1 FN,
squat 3 FP + 2 FN, one_foot 2 FP, airplane 1 FN, hands_head 1 FN, touch_toes 1 FN.

## The real-game slice: 119 crops (7 AI-generated sample photos, SAM 3 crops, every player x every command, 9 yes)

| model | n | accuracy | macro-F1 | false pos | false neg |
|---|---|---|---|---|---|
| untuned-4b | 119 | 99.2% | 0.971 | 1 | 0 |
| d1-omni-600m | 119 | 77.3% | 0.602 | 25 | 2 |
| tuned-v1 | 119 | 100.0% | 1.000 | 0 | 0 |
| tuned-v2 | 119 | 98.3% | 0.940 | 1 | 1 |

Every Qwen model is at 98–100% here; the slice is small and easy (9 positives) and does not separate them. It does
show d1-omni-600M's false-positive problem on the game's own crops (25 FP, mostly `airplane` and `arms_up`). v2's two
errors: player 2 of frame 002 asked `arms_up` (P = 0.56, untuned 4B: 0.78), and the squatting player 2 of frame 005
(P = 0.22). The untuned 35B was not run on this slice.

## Latency (wall time per call, median; game-slice crops, 384 px long side)

| referee | where | 1 crop | round of 3 crops (one batched call) | round of 6 crops | per crop at 6 |
|---|---|---|---|---|---|
| Qwen3.5-4B + LoRA v2 | Tinker (remote) | 2,885 ms | 3,006 ms | 2,879 ms | 480 ms |
| d1-omni-600M | this CPU (2 vCPU, fp32) | 2,683 ms | 7,939 ms | 16,507 ms | 2,751 ms |

5 calls per size for Tinker, 3 for d1, after one warm-up. The Tinker number is dominated by a fixed ~2.9 s round
trip (network from this sandbox + queueing + prefill), so a round of 3 or 6 players costs the same as one player;
d1 on this CPU grows linearly with players. Tinker was measured while another job on the same account was running.
For scale, in the earlier CPU replay demo SAM 3 took 19.3 s per frame on the same box, so the referee is not the
bottleneck on CPU either way. Not measured: d1-3B, d1 on a GPU (Liquid's card: 17 ms on an RTX 4090), the local
Qwen weights.

## Cost

Tinker list prices (https://tinker-docs.thinkingmachines.ai/tinker/models.json, fetched 2026-10-09), per million
tokens: Qwen3.5-4B prefill $0.33, sample $1.005, train $0.737; Qwen3.6-35B-A3B prefill $0.54, sample $1.335.
A request is ~182 tokens (image ~90 tokens at 384 px + the prompt), plus one sampled token.

- **Per crop:** ~$0.000061. **Per game:** 10 rounds x 6 players (60 crops) ≈ **$0.004**; 20 rounds x 10 players ≈ $0.012.
- **Everything so far (estimate):** training v1 $0.73 + v2 $0.75; ~4,840 Qwen3.5-4B scoring requests (evals,
  in-training val, latency, demo) $0.30; ~2,990 Qwen3.6-35B-A3B teacher requests $0.30. **Total ≈ $2.1**, of which
  the v2 iteration and its evaluation ≈ $0.9.
- These are estimates from logged token counts and the rate card. `rest_client.get_billing_usage` returned no rows
  for this account, so there is no billed figure to quote.

## Training

| | v1 | v2 (the referee) |
|---|---|---|
| base | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-4B |
| LoRA rank | 32 | 32 |
| learning rate | 4e-4, linear decay to 0 | 2e-4, linear decay to 0 |
| epochs / steps / batch | 3 / 174 / 32 | 2 / 176 / 32 |
| train examples | 1,884 | 2,604 unique (2,844 with rare positives x3) |
| val examples | 138 | 222 (v1 val + 84 near-miss negatives) |
| tokens trained | 987,902 | 1,023,508 |
| checkpoint chosen on val | step 174 (val 100%) | step 100 (val 97.3%; val NLL lowest, 0.102) |

Loss: cross-entropy on the answer word and `<|im_end|>` only; random horizontal flips. Every 25 steps the weights
are saved for sampling and scored on val; the best val checkpoint is the one scored on TEST, once.
v2 val accuracy by step: 25: 93.7%, 50: 91.0%, 75: 94.1%, **100: 97.3%**, 125: 96.0%, 150: 94.1%, 175/176: 93.7%.
On the v2 val set the untuned 4B scores 84.2% and v1 83.3%.

**What changed in v2.** v1's TEST errors were nearly all false positives on "near-miss" poses: a skateboarder or
surfer crouched forward read as *touching toes*, a baseball slide as *touching toes*, sitting cross-legged as
*squatting*, a cyclist or runner as *standing on one foot*. v1's negatives were mostly easy (upright, arms down), and
its val set had none of these, so it hit 100% val while TEST did not move. `scripts/mine_hard_negatives.py` finds
COCO people whose keypoints are close to a pose but decisively fail a key condition (bent over but knees deeply bent
or hands above the knees; hips at ankle height; an ankle raised but the torso leaning or the standing leg bent; one
arm up but the other down; and so on) and adds 120 per command to train and 14 to val as labelled negatives, from
train/val images only. A visual spot check of 42 of them found them to be real negatives, with a few borderline crops. The learning rate was
halved and epochs cut from 3 to 2 because v1's loss went to ~0 (it was memorising), and the positives of the rarest
commands (`hands_head` 9, `touch_toes` 19, `airplane` 36) were repeated 3x.

**Honest caveat on TEST.** TEST was scored once per model, and no TEST photo is in train or val (split by COCO
image; the 210 test images are excluded from mining). But the near-miss finders were designed after looking at v1's
TEST errors, so the categories of hard negatives were informed by TEST. Treat v2's TEST gain as somewhat optimistic;
the gain on the untouched val set (84% → 97%) and on the error categories is consistent with it.

## Data pipeline and licences

- **Labels:** COCO 2017 person keypoints (© COCO Consortium, CC BY 4.0) turned into yes / clear-no / ambiguous by
  explicit geometric rules (`simon_sees/pose_rules.py`, table in `data/README.md`); ambiguous people are dropped.
- **Crops:** made exactly as the game crops a player (box + 12% margin, background dimmed by the COCO mask, other
  people painted grey, 384 px long side).
- **Label cleaning (train/val only):** an example is kept only if the untuned Qwen3.6-35B-A3B agrees with the rule
  label. The v2 near-miss negatives are not teacher-filtered (the teacher makes the same false positives).
- **TEST:** 262 candidates looked at by hand, 40 dropped as ambiguous, 15 rule labels flipped → 222 examples.
- **Images:** Flickr photos under their own CC licences (train/val counts: CC BY-NC-ND 805, BY-NC-SA 758, BY 461,
  BY-NC 398, BY-SA 239, BY-ND 153, no known restrictions 12). Each manifest row carries its licence and Flickr URL.
  Images are downloaded for training/evaluation only and not redistributed; `data/referee/` and `models/` are git-ignored.
- **The fine-tuned weights:** a LoRA on Qwen3.5-4B (Apache-2.0). Trained on images that are partly non-commercial
  licensed; check that before any commercial use.

## What failed or is thin

- **`hand_mouth` and `wave` were dropped.** Their rules do not work on COCO: a hand near the face is mostly a phone
  call, a drink or a batting stance, and one raised hand is mostly a sport swing (1 in 16 and 3 in 16 rule positives
  were real on the human check). Both commands are still in the game; with the Tinker referee they are judged by the
  fine-tuned model on a command it was never trained or tested on, so their accuracy is unmeasured.
- **Thin rare-pose positives.** Train has 9 `hands_head`, 19 `touch_toes` and 36 `airplane` positives after
  cleaning; TEST has 6, 8 and 16. The per-command numbers for these rest on a handful of examples.
- **v1's val set was too easy** (100% val, no TEST gain). v2's val adds near-miss negatives; v2 still never sees
  the motion-blur, mid-air and outdoor-light cases a field game will produce.
- **`arms_up` did not improve** (84.4% for every 4B variant). Mid-jump arms-up poses are the residual errors.
- **Not covered by the referee:** `freeze` (frame differencing), `sit_down`, and the prop commands (`touch_green`,
  `touch_ground`, `hug_tree`).
- **Local serving is untested.** The LoRA was downloaded (`scripts/export_referee.py download`, 278 MiB), but
  converting/merging it and running Qwen3.5-4B locally needs more RAM than this 8 GB sandbox has. See `docs/tinker.md`.
- **No field test yet.** All numbers are on COCO photos and AI-generated sample photos.
