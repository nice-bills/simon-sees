---
title: I read 321 Touch Grass entries. None of them could referee a game of Simon Says. So I built one that can.
published: false
tags: devchallenge, hf26challenge, opensource, ai
cover_image: [COVER IMAGE: a photo of the phone on a stump with friends mid-pose behind it]
---

<!--
DRAFT. Everything in [SQUARE BRACKETS] is a placeholder for something only you can supply: a number you
measured, a moment from the field test, a link. Do not publish with any [ ] left. Never fill a number
you did not measure.
-->

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 1: Touch Grass](https://dev.to/challenges/hacktoberfest-week1-2026-10-05)*

[PERSONAL OPENING, 3–4 sentences, in your own voice. Suggested shape: the last time you played a dumb
playground game as an adult, who was there, what it felt like, and the moment someone pulled out a
phone and it ended. Example of the *shape* only: "The last time my friends and I played Simon Says we
were [age/occasion]. It lasted [how long] and ended in an argument about whether [name] had moved."]

Before building anything for this challenge, I read the field. As of October 8 there were **321 valid
Week 1 entries**. **130** of them are outdoor quests, missions, scavenger hunts or games. By my count
(a keyword scan of every post, then reading each hit), only **three** are built for more than one person
at a time, and all three are photo games: you go off, take a picture, and an AI judges it afterwards.

Not one of them watches people play *together, live*. That's the game I wanted: the kind you play on a
patch of grass with friends, where the only screen is a phone leaning against a tree and nobody is
looking at it.

## What I Built

**Simon Sees** is Simon Says for a group of friends outdoors, refereed live by an open AI model running
on a laptop.

You prop a phone on a stump or a backpack, pointed at the players. Everyone stands in a row and you tap
**Line up**: the phone shows one photo card per person, numbered left to right. Type or say a name on a
card if you like. Then you walk away from the screen, and a voice takes over:

> **"Simon says stand on one foot."** … *two seconds* … **"Nice. Everyone's still in."**
> **"Hands on your head!"** … **"Number 2, Simon didn't say! You're out."**
> **"Simon says freeze."** … **"Kofi, you moved. You're out."**

The phone streams a few frames a second to the laptop. Open models find every person (**SAM 3**), work out
which player is which by comparing them with their line-up photo (**DINOv3** embeddings), and then an
open-weight referee looks at each player's own crop and answers one yes/no question (a **Qwen3.5-4B** I
fine-tuned on **Tinker** for exactly this, or Liquid's **d1** decision model fully offline): *"Look at
the person in this photo. Is that person standing on one foot?"* It comes back with a
probability, the game applies the rules, and Simon calls out who's out. Last one standing wins.

The screen is for setup only: aim the camera, line up, press Start. The rest of the game is voices,
laughing, and people wobbling on one leg in the grass.

It's for anyone with a phone, a laptop and two or more friends: a family at a picnic, kids at a birthday,
a group at the park who have run out of things to say. [ADD: who you actually played with.]

## Demo

[VIDEO: 60–90 s. Suggested cut: 5 s of the phone being propped up → a trick round where someone gets
caught → a freeze round → the winner announcement → one shot of the laptop log scrolling.]

[GIF of a single round, if the video is long]

No friends or field nearby? There's a desk mode: `python -m simon_sees serve --replay my_clip.mp4`
plays a recorded clip as the camera so you can watch the referee work without going outside (though
going outside is the point).

## Code

{% github nice-bills/simon-sees %}

MIT-licensed code. Runs with `pip install -r requirements-model.txt && python -m simon_sees serve`;
the phone joins over a Cloudflare tunnel or, fully offline, over the laptop's own Wi-Fi/hotspot with a
local HTTPS certificate.

## How I Built It

### Who is who: segment, embed, match

My first version asked the referee about clothes: players registered as "red shirt" or "blue cap", and every
question started *"Look only at the person wearing a red shirt…"*. Testing killed it (the numbers are further
down): the small decision model said "no" to every clothing-conditional question, even with that person plainly
in the photo. So the game no longer describes anyone. It identifies people the way a visual search engine finds
an object:

1. **Enrolment.** At the line-up, **[SAM 3](https://huggingface.co/facebook/sam3)** segments every person in the
   frame from the text prompt `"person"`. Each mask becomes a photo card, numbered left to right, and each masked
   crop is embedded with **DINOv3**. That vector *is* the player.
2. **Segmentation.** Every round, SAM 3 finds everyone in the new frame.
3. **Embedding match.** Each person's crop is embedded and compared with every player's small gallery (the line-up
   vector plus a few confident sightings, so it follows pose and light). A Hungarian assignment pairs players and
   people one to one. Below the threshold, a player is "not seen", never guessed.
4. **Judging per crop.** Only then does d1 see anything, and what it sees is one player's crop with the
   background dimmed by their mask.

The segment-then-embed pattern is Maziyar Panahi's: he showed DINOv3 + SAM 3.1 turning a surgical tray into a
visual search engine, one instrument as the query and the rest ranked by similarity, all local
([post](https://x.com/MaziyarPanahi/status/2106368674500358351)). In Simon Sees the query is your line-up photo.

I could only measure this on a CPU so far: a 2-vCPU sandbox, SAM 3 at 560 px, and DINOv2-small standing in
because my DINOv3 access was still pending. On the 7 sample photos (AI-generated park shots of three people, not
a field recording) SAM 3 got the head count right in 7 of 7 frames at a median **16.6 s per frame** (fine for a
test, hopeless for a game; that's what a GPU is for). Re-identification was right in 5 of 6 frames after the
line-up, **including the frame where all three players swapped places** (cosine 0.94–0.96 to the right person),
with no identity swaps. The one miss: a player in a deep squat, straight after the swap, scored 0.30 against her
standing photo and was reported "not seen", which the rules turn into a warning, not an elimination.

### The hero: a decision model, not a chatbot

The referee is **[Liquid AI's d1-3B](https://huggingface.co/LiquidAI/d1-3B)**, an open-weight *decision
model* released on October 7. You hand it a state (here, one player's crop) and a set of typed questions,
and it returns **calibrated probabilities in a single forward pass, generating zero tokens**:

```python
model.system_one(None, {
    "did": {"type": "noul",
            "instructions": "Look at the person in this photo. Is that person standing on one foot?",
            "criteria": {"true": "one foot clearly lifted off the ground", "false": "both feet on the ground"}},
}, images=[player_crop])
# {"answers": {"did": {"type": "noul", "noul": 0.97}}, "usage": {..., "output_tokens": 0}}
```

That shape is exactly what a referee needs. I don't want a paragraph about the scene; I want
P(this player is on one foot) as a number I can threshold. Three things made it fit:

1. **All players in one batch.** Every seen player's crop for a frame goes into the same batched call, so a
   round costs one model call per frame, not one per player.
2. **Two images, one question.** For "freeze" I give it the same player's BEFORE and AFTER crops and ask
   *"These two photos show the same person a moment apart. Did that person move?"*. (A plain pixel-difference
   check inside that player's box runs alongside as a logged second opinion. It only takes over if you swap in a
   model that can't take two images.)
3. **Probabilities, not words.** P < 0.35 when Simon said it: out. P > 0.65 when Simon didn't: out. In
   between, the referee isn't sure and gives the benefit of the doubt, like a fair human would.

### The game loop

```
call → speak → wait a beat (2.2 s) → grab frames → SAM 3 → DINO match → one batched d1 call per frame → rules → announce
```

The engine is plain Python with a pure `judge()` function, so the rules are unit-tested with a mocked
model (66 pytest tests). Some rules came from playing, not planning:

- **Trick rounds never use "freeze"**, because standing still is what everyone is already doing.
- **If everyone is caught at once, the round doesn't count.** (Playground law.)
- **Someone the camera can't see (or can't recognise) is warned, never eliminated.** "Number 3, I lost you."
- **No trick in round one.** People need to learn the rhythm first.

Every decision is written to `decisions.jsonl`: the question, per-frame probabilities, latency, verdict.
That log is where the numbers below come from.

### Voice and ears, also open

Simon's voice is **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** (Apache-2.0) on the laptop,
streamed to the phone as WAV; if it isn't installed, the phone's built-in speech synthesis takes over.
Saying your name for your photo card uses **faster-whisper**. The phone page is vanilla JS: camera → 640 px
JPEG → WebSocket, ~3 frames a second.

### Decision model vs generative VLM: the numbers

I benchmarked the same referee questions on the same frames two ways: d1, and a generative open VLM
asked to answer "yes" or "no". The fairest pairing is **d1-3B vs LFM2.5-VL-3B, its own base model**: same
backbone, decision head vs text generation.

**Headline (my laptop, [HARDWARE e.g. M-series Mac, mps / RTX xxxx], [N] labeled frames from our field test):**

| model | type | accuracy | median latency / decision | p90 | tokens generated |
|---|---|---|---|---|---|
| LiquidAI/d1-3B | decision | [RESULT]% | [RESULT] ms | [RESULT] ms | 0 |
| LiquidAI/LFM2.5-VL-3B | generative | [RESULT]% | [RESULT] ms | [RESULT] ms | [RESULT] |
| LiquidAI/d1-omni-600M | decision (experimental) | [RESULT]% | [RESULT] ms | [RESULT] ms | 0 |

[ONE OR TWO SENTENCES interpreting the table honestly: where d1 won, where it didn't.]

For context, Liquid's own card reports d1-3B at **17 ms** for a 384 px image on an RTX 4090 and **62 ms** on
an Apple M5 Pro. A Simon Says round gives players a two-second beat anyway, so anything under a few
hundred milliseconds is invisible.

**And on a machine with no GPU at all.** Before I had the laptop set up, I ran the benchmark in a 2-vCPU,
no-GPU cloud sandbox on 23 hand-labeled questions over 7 stock photos (people in tree pose, hands on head,
sitting on grass, plus negatives). d1-3B wouldn't run there: the box has 8 GB of RAM and the weights are
6.2 GB in bf16 (12.5 GB in fp32, the card's CPU default). So this compares the two small models:

| model | type | accuracy | median latency | tokens generated |
|---|---|---|---|---|
| LiquidAI/d1-omni-600M (experimental) | decision | 78% (18/23) | 2,552 ms | 0 |
| LiquidAI/LFM2.5-VL-450M | generative | 87% (20/23) | 2,583 ms | ~2 |

That surprised me, and I'm leaving it in. On a CPU this small, a 2-token answer costs almost nothing
extra, so the decision model's speed advantage disappears, and the experimental 600M encoder was less
accurate than a 450M generative model. It also said "no" to every clothing-specific presence question I tried
("is there a person wearing a white t-shirt?", three phrasings, three photos where that person was plainly
there), while "is there a person?" was right every time.
That result is why the game stopped describing people at all: SAM 3 finds them, DINO tells them apart, and d1
is only ever asked about one person's own crop (see "Who is who" above). These numbers are from whole-frame
questions, before that change.
The generative model had its own failure: shown the *same* photo twice, it said the person had moved
(2/2), while d1 correctly said no (2/2). For a freeze referee, that's the error that matters.

[UPDATE THIS PARAGRAPH once you have the laptop numbers: does d1-3B close the accuracy gap? Is it faster?]

### Best Use of Tinker: fine-tuning the referee

The referee only ever answers one question: *is this person, in this crop, doing this pose?* On the 8 GB, no-GPU box
I built this in, the d1 that fits (`d1-omni-600M`) got that right **63%** of the time on a test set I checked by
hand, and most of its mistakes were the worst kind for a referee: calling people out for poses they weren't doing.
So I fine-tuned a referee on **Tinker**: a rank-32 LoRA on **Qwen3.5-4B** (open weights), trained on exactly the
game's prompt, with P(yes) read from the logprobs of one generated token so the game can threshold it like d1.

**The data.** COCO 2017 person keypoints (CC BY 4.0) gave me labels without hand-labelling thousands of photos:
explicit geometric rules turn 17 keypoints into *yes*, *clearly no* or *ambiguous* per command (both wrists 0.3
torso-lengths above the nose and arms straight = reaching for the sky, and so on), ambiguous people are dropped,
and each person is cropped exactly the way the game crops a player. The untuned **Qwen3.6-35B-A3B** on Tinker then
acted as a label cleaner: a train/val example was kept only if it agreed with the rule. The test set never touched
the rules or the teacher: I looked at all 262 candidate crops, flipped 15 labels and dropped 40 → **222 test crops
from 210 photos that are never used for training.**

**The numbers** (threshold 0.5 for every model, `bench/results-tinker.md`):

| referee | test accuracy | macro-F1 | false "did it" | missed |
|---|---|---|---|---|
| d1-omni-600M, CPU | 63.1% | 0.629 | 65 | 17 |
| Qwen3.5-4B, untuned | 87.4% | 0.871 | 27 | 1 |
| Qwen3.6-35B-A3B, untuned | 88.7% | 0.882 | 19 | 6 |
| Qwen3.5-4B + LoRA, first run | 87.8% | 0.874 | 22 | 5 |
| **Qwen3.5-4B + LoRA, second run** | **93.2%** | **0.928** | **9** | **6** |

**The first run didn't work, and that was the useful part.** It hit 100% on validation and moved test by 0.4
points. The val set was too easy, and so were the negatives: mostly people standing upright with their arms down.
The test errors were all near-misses: a crouched skateboarder or a baseball player sliding into base "touching
their toes", someone sitting cross-legged "squatting", a cyclist "on one foot". For the second run I mined 720
near-miss negatives from COCO keypoints (bent over but knees deeply bent, hips at ankle height, an ankle raised but
the body leaning...), added more of them to val, repeated the rarest positives, halved the learning rate and
trained 2 epochs instead of 3. False positives fell from 27 to 9; *touch your toes* went from 75% to 98%. Against
the untuned 4B that's significant (exact McNemar p = 0.019); against the untuned 35B, a model with ~9x the
parameters, it's ahead but not significantly (p = 0.08). Honesty note: I picked the near-miss categories after
looking at the first run's test errors, so the test gain is probably a little optimistic.

**Speed and cost.** One batched Tinker call per round takes ~2.9 s from my sandbox whether it's 1, 3 or 6 players;
d1-omni-600M on that CPU takes 2.7 s *per player*. At Tinker's list prices a crop costs about $0.00006, so a
10-round game with 6 friends is under half a cent, and both training runs plus every evaluation came to about $2.

**What didn't work.** "Cover your mouth" and "wave" had rules too, but on COCO a hand near the face is mostly a
phone call or a drink, and one raised hand is mostly a tennis swing; I dropped both from training rather than teach
the model the wrong thing. The rarest poses still have thin data (9 training positives for hands-on-head), and
`arms_up` didn't improve at all: people mid-jump with their arms up are its remaining errors.

**Still open, and yours to keep.** The LoRA downloads from Tinker (`scripts/export_referee.py`) and merges into
a plain Qwen3.5-4B checkpoint for transformers or vLLM, so the referee doesn't depend on any service. I couldn't
test that local path on an 8 GB box; it's documented in `docs/tinker.md` and marked untested.

[FIELD RESULT for the Tinker referee: X of Y calls matched the video, median Z ms per round from `summarize_log.py`]

## Taking it outside

[FIELD TEST, the most important section. Suggested beats, with real details only:]

- **Where and who:** [place, time of day, how many people, ages]
- **Setup time:** [minutes from arrival to first round]
- **How many rounds / games:** [N games, M rounds, from decisions.jsonl via `scripts/summarize_log.py`]
- **Referee latency in the field:** median [RESULT] ms per batch (`summarize_log.py` prints it)
- **How often the referee was right:** [X of Y calls matched what we saw on video], with [N] benefit-of-the-doubt calls
- **The best moment:** [the call everyone argued about, the person who fell for a trick round three times]
- **What broke:** [sun glare, two people who look alike, players bunching up at the line-up, someone walking out of frame, wind, the phone sliding]
- **What I changed afterwards:** [a threshold, a command that never worked, a rule]
- **How long anyone looked at a screen:** [roughly, e.g. "about 30 seconds at the start"]

[PHOTO: the group mid-round, from behind the phone]

## Why Does Open Innovation Matter?

**It's a camera pointed at your friends.** That sentence is why this had to be open and local. With the
laptop setup, no frame ever leaves devices we own (the optional Modal GPU backend sends frames to my own Modal
workspace instead, and the post says so if I use it): no cloud vision API, no account, no upload, nothing
stored unless I opt in with `--save-frames`. I wouldn't point a closed API at my friends' kids at a
birthday party, and I wouldn't expect anyone else to.

**Parks don't have good signal.** The weights download once. After that the phone can join the laptop's
hotspot with mobile data off, and the whole game runs in a field with zero bars.

**It's free to play all afternoon.** Each round is a couple of vision calls per player; an afternoon is
[RESULT from your log] decisions. Locally that costs some battery. On a per-call vision API it would be a
bill, and a reason to stop playing.

**I could look inside and change it.** The d1 weights *and* the inference code are on Hugging Face. I read
`runner.py` to learn that a request with images gets its own pass with the image as the shared trunk, which
is why the game batches all players into one request per frame instead of one call per player. I swapped
in a second model with one flag, and benchmarked both against a generative model on my own frames. A closed
endpoint gives you none of that.

A licence note, because "open" has fine print: the code is MIT. d1 is released under Liquid's **LFM Open
License v1.0**, which allows free use, modification and redistribution, including commercial use by
companies under US$10M in annual revenue. That's open enough for a picnic game, and I say so in the README.

## My Agent Session

[OPTIONAL: DevRelay embed or link, `{% agent_session … %}`. Delete this section if you don't include one.]

## Prize Categories

- Overall
- Best Use of Tinker (the fine-tuned referee: see "Best Use of Tinker: fine-tuning the referee" above)
<!-- Keep a line only if you actually used it, and add evidence in the post:
- Best Use of DigitalOcean: [only if you ran the referee on a GPU Droplet; add a screenshot + the latency you got there]
- Best Use of Sentry Agent Tracing: [only if you ran with SENTRY_DSN; add a trace screenshot of gen_ai.evaluate spans and what you learned]
-->

[CLOSING LINE, one sentence, your voice. E.g. what the group asked to play next.]
