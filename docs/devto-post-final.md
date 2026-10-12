---
title: "I read 321 Touch Grass entries. None of them could referee a game of Simon Says. So I built one that can."
published: false
tags: devchallenge, hf26challenge, opensource, ai
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 1: Touch Grass](https://dev.to/challenges/hacktoberfest-week1-2026-10-05)*

Playground games rarely end because people get bored. They end when one person checks a phone, then another, and the circle on the grass turns into a row of people looking down. I wanted to give the phone the one job in Simon Says that nobody wants, the referee, so everyone else can stay in the game.

Before building anything, I read the field. On October 8 there were **321 valid Week 1 entries**, and **130** of them were outdoor quests, missions, scavenger hunts or games. Almost all are built for one person and a phone: you go somewhere, take a photo, and an AI judges it afterwards. I didn't find one that watches a group of people play together, live. That's the game I wanted: a patch of grass, a few friends, and a phone leaning against a tree that nobody looks at.

## What I Built

**Simon Sees** is Simon Says outdoors, refereed live by open models.

Prop a phone on a stump or a backpack, pointed at the players. Everyone stands in a row and you tap **Line up**: the phone shows one photo card per person, numbered left to right, and you can type or say a name on any card. Then you walk away from the screen and a voice takes over:

> **"Simon says stand on one foot."** … *a beat* … **"Nice. Everyone's still in."**
> **"Hands on your head!"** … **"Number 2, Simon didn't say! You're out."**
> **"Simon says freeze."** … **"Number 3, you moved. You're out."**

The phone streams a few frames a second to a laptop. **SAM 3** finds every person, **DINO** embeddings work out which player is which by comparing each one with their line-up photo, and a referee looks at each player's own crop and answers one yes/no question: *"Look at the person in this photo. Is that person standing on one foot?"* The referee is a **Qwen3.5-4B I fine-tuned on Tinker** for exactly that question, or Liquid's **d1** decision model if you want the whole game on your own laptop. It returns a probability, the game applies the rules, and Simon calls out who's out. Last one standing wins.

The screen is for setup only: aim, line up, press Start. The rest is voices, laughing, and people wobbling on one leg. Playing alone works too: whoever is in the picture is the player.

It's for anyone with a phone, a laptop and some space: a family at a picnic, kids at a birthday, a group at the park who have run out of things to say.

## Demo

![Simon Sees replay demo: the line-up shows one photo card per player, Simon calls a command, the referee's per-player decisions appear in Ref's notes, and a winner is announced](https://raw.githubusercontent.com/nice-bills/simon-sees/main/demo/simon-sees-demo.gif)

This is a **replay demo**: the phone page at 390×844 playing a game over AI-generated sample photos, with the real models running on a 2-vCPU, no-GPU box, sped up 3.9x to fit in a minute. With Simon's voice: [simon-sees-demo.mp4](https://github.com/nice-bills/simon-sees/blob/main/demo/simon-sees-demo.mp4).

In that round ("Simon says put your hands on your head") the fine-tuned referee gave the three players P = 0.00, 0.995 and 0.00. Only the middle player had her hands on her head, so she wins after one round. An earlier recording with `d1-omni-600M` gave 0.12, 0.71 and 0.14 on the same frame.

So far Simon Sees has been tested on recorded and replayed footage and an end-to-end test suite; a real game outdoors is the next step. You can run the same desk mode yourself: `python -m simon_sees serve --replay my_clip.mp4` plays a recorded clip as the camera.

## Code

{% github nice-bills/simon-sees %}

Repo: [github.com/nice-bills/simon-sees](https://github.com/nice-bills/simon-sees), MIT-licensed. `pip install -r requirements-model.txt && python -m simon_sees serve`, then open the page on your phone over a Cloudflare tunnel or, fully offline, over the laptop's own Wi-Fi or hotspot with a local HTTPS certificate. `./run_playtest.sh` does the install, server and tunnel in one go.

## How I Built It

### Who is who: segment, embed, match

My first version asked the referee about clothes: players registered as "red shirt" or "blue cap", and every question began *"Look only at the person wearing a red shirt…"*. Testing killed it. The small decision model said "no" to every clothing-specific question I tried ("is there a person wearing a white t-shirt?", three phrasings, three photos where that person was plainly there), while "is there a person?" was right every time. So the game no longer describes anyone. It identifies people the way a visual search engine finds an object:

1. **Enrolment.** At the line-up, **[SAM 3](https://huggingface.co/facebook/sam3)** segments every person in the frame from the text prompt `"person"`. Each mask becomes a photo card, and each masked crop is embedded with **DINOv3** (DINOv2-small if you don't have DINOv3 access). That vector *is* the player.
2. **Segmentation.** Every round, SAM 3 finds everyone in the new frame.
3. **Embedding match.** Each person's crop is compared with every player's small gallery: the line-up vector plus a few confident sightings, so it follows pose and light. A Hungarian assignment pairs players and people one to one. Below the threshold, a player is "not seen", never guessed.
4. **Judging per crop.** Only then does the referee see anything: one player's crop with the background dimmed by their mask, so a neighbour's raised arm doesn't count.

The segment-then-embed pattern is Maziyar Panahi's: he showed DINOv3 + SAM 3.1 turning a surgical tray into a visual search engine, one instrument as the query and the rest ranked by similarity, all local ([post](https://x.com/MaziyarPanahi/status/2106368674500358351)). In Simon Sees the query is your line-up photo.

I measured this on a CPU: a 2-vCPU sandbox, SAM 3 at 560 px, DINOv2-small. On 7 AI-generated park photos of three people, SAM 3 got the head count right in 7 of 7 frames at a median **16.6 s per frame**, fine for a test and hopeless for a game; live play wants a GPU or a Mac. Re-identification was right in 5 of 6 frames after the line-up, **including the frame where all three players swapped places** (cosine 0.94–0.96 to the right person), with no identity swaps. The one miss: a player in a deep squat, straight after the swap, scored 0.30 against her standing photo and was reported "not seen", which the rules turn into a warning, not an elimination.

### Best Use of Tinker: fine-tuning the referee

The referee answers one question: *is this person, in this crop, doing this pose?* On the 8 GB, no-GPU box I built this in, the d1 that fits (`d1-omni-600M`) got that right **63%** of the time on a hand-checked test set, and most of its mistakes were the worst kind for a referee: calling people out for poses they weren't doing. So I fine-tuned a referee on **Tinker**: a rank-32 LoRA on **Qwen3.5-4B** (open weights, Apache-2.0), trained on the game's exact prompt, with P(yes) read from the logprobs of one generated token so the game can threshold it like a decision model.

**The data.** COCO 2017 person keypoints (CC BY 4.0) gave me labels without hand-labelling thousands of photos. Explicit geometric rules turn 17 keypoints into *yes*, *clearly no* or *ambiguous* per command (both wrists 0.3 torso-lengths above the nose with arms straight means reaching for the sky, and so on). Ambiguous people are dropped, and each person is cropped exactly the way the game crops a player. The untuned **Qwen3.6-35B-A3B** on Tinker then cleaned the labels: a train/val example was kept only if it agreed with the rule. The test set never touched the rules or the teacher. I looked at all 262 candidate crops, flipped 15 labels and dropped 40, leaving **222 test crops from 210 photos never used for training.**

**The numbers** (threshold 0.5 for every model, six pose commands, [`bench/results-tinker.md`](https://github.com/nice-bills/simon-sees/blob/main/bench/results-tinker.md)):

| referee | test accuracy | macro-F1 | false "did it" | missed |
|---|---|---|---|---|
| d1-omni-600M, CPU | 63.1% | 0.629 | 65 | 17 |
| Qwen3.5-4B, untuned | 87.4% | 0.871 | 27 | 1 |
| Qwen3.6-35B-A3B, untuned | 88.7% | 0.882 | 19 | 6 |
| Qwen3.5-4B + LoRA, first run | 87.8% | 0.874 | 22 | 5 |
| **Qwen3.5-4B + LoRA, second run** | **93.2%** | **0.928** | **9** | **6** |

**The first run didn't work, and that was the useful part.** It hit 100% on validation and moved test by 0.4 points. The validation set was too easy, and so were the negatives: mostly people standing upright with their arms down. The test errors were all near-misses: a crouched skateboarder or a baseball player sliding into base "touching their toes", someone sitting cross-legged "squatting", a cyclist "on one foot". For the second run I mined 720 near-miss negatives from COCO keypoints (bent over but knees deeply bent, hips at ankle height, an ankle raised but the body leaning), added more of them to validation, repeated the rarest positives, halved the learning rate and trained 2 epochs instead of 3.

False "did it" calls fell from 27 to 9, and *touch your toes* went from 75% to 98%. Against the untuned 4B that's significant (exact McNemar p = 0.019; bootstrap 95% interval for the tuned model 89.6–96.4%). Against the untuned 35B, a model with ~9x the parameters, it's ahead but not significantly (p = 0.08). One caveat: I chose the near-miss categories after looking at the first run's test errors, so the test gain is probably a little optimistic.

**Speed and cost.** One batched Tinker call per round takes ~2.9 s from my sandbox whether it's 1, 3 or 6 players (480 ms per crop at 6). `d1-omni-600M` on that CPU takes ~2.7 s *per player*, so a round of 6 takes 16.5 s. At Tinker's list prices a crop costs about $0.00006, a 10-round game with 6 friends is under half a cent, and both training runs plus every evaluation came to about $2.

**What didn't work.** "Cover your mouth" and "wave" had rules too, but on COCO a hand near the face is mostly a phone call or a drink, and one raised hand is mostly a tennis swing. I dropped both from training rather than teach the model the wrong thing. The rarest poses still have thin data (9 training positives for hands-on-head), and `arms_up` didn't improve at all: people mid-jump with their arms up are its remaining errors.

**Yours to keep.** The LoRA downloads from Tinker (`scripts/export_referee.py`) and merges into a plain Qwen3.5-4B checkpoint for transformers or vLLM, so the referee doesn't have to depend on any service. That local path needs more RAM than my 8 GB box, so it's documented in `docs/tinker.md` and marked untested.

### The fully local referee: a decision model, not a chatbot

Without a Tinker key, the game falls back to **[Liquid AI's d1](https://huggingface.co/LiquidAI/d1-3B)**, an open-weight *decision model*, running on your own machine. You hand it a state (one player's crop) and a set of typed questions, and it returns **calibrated probabilities in a single forward pass, generating zero tokens**:

```python
model.system_one(None, {
    "did": {"type": "noul",
            "instructions": "Look at the person in this photo. Is that person standing on one foot?",
            "criteria": {"true": "one foot clearly lifted off the ground", "false": "both feet on the ground"}},
}, images=[player_crop])
# {"answers": {"did": {"type": "noul", "noul": 0.97}}, "usage": {..., "output_tokens": 0}}
```

That's the shape a referee needs: not a paragraph about the scene, but P(this player is on one foot) as a number to threshold. Every seen player's crop goes into one batched request per frame. For "freeze", d1 gets the same player's BEFORE and AFTER crops and one question: *"These two photos show the same person a moment apart. Did that person move?"* (The Tinker referee sees one crop at a time, so with it, freeze is judged by a pixel-difference check inside the player's box, labelled `FALLBACK` in the log.)

In an early whole-frame test on the CPU box, a generative VLM (`LFM2.5-VL-450M`) shown the *same* photo twice said the person had moved (2 of 2), while d1 correctly said no (2 of 2). For a freeze referee, that's the error that matters. The larger `d1-3B` (6.2 GB in bf16) didn't fit in that box's 8 GB of RAM, so I have no d1-3B numbers of my own; Liquid's model card reports **17 ms** for a 384 px image on an RTX 4090 and **62 ms** on an Apple M5 Pro.

### The rules

**Simon said it and P(did it) < 0.35: out. Simon didn't say it and P(did it) > 0.65: out.** In between, the referee isn't sure and gives the benefit of the doubt, like a fair human would. The engine is plain Python with a pure `judge()` function, so the rules are unit-tested with a mocked model, and end-to-end tests drive the phone page in a browser at desktop size and at 390×844. A few rules come straight from how the game is played:

- **Trick rounds never use "freeze"**, because standing still is what everyone is already doing.
- **If everyone is caught at once, the round doesn't count.** (Playground law.)
- **Someone the camera can't see or recognise is warned, never eliminated.** "Number 3, I lost you."
- **No trick in round one.** People need to learn the rhythm first.

Every decision is written to `decisions.jsonl`: the question, per-frame probabilities, latency per stage, verdict. Players can tap **Wrong call** on any verdict; `python -m simon_sees upload` sends recorded sessions to a *private* Hugging Face dataset, and `scripts/export_labels.py` turns the corrections into training rows for the next fine-tune.

### Voice and ears, also open

Simon's voice is **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** (Apache-2.0) on the laptop, streamed to the phone; if it isn't installed, the phone's built-in speech synthesis takes over. Saying your name for your photo card uses **faster-whisper**. The phone page is vanilla JS with self-hosted fonts: camera, 640 px JPEG, WebSocket, about 3 frames a second.

## Why Does Open Innovation Matter?

**It's a camera pointed at your friends.** That sentence is why this had to be built on open models. With the local d1 referee, no frame leaves devices you own: no cloud vision API, no account, nothing stored unless you opt in. With the Tinker referee, the only thing that leaves the laptop is each player's dimmed crop, sent for one yes/no answer to a model I trained and control, and you can take the weights home with `export_referee.py`. I wouldn't point a closed vision API at a kids' birthday party, and I wouldn't expect anyone else to.

**I could fix the referee instead of prompting around it.** An off-the-shelf model said "you did it" to a crouched skateboarder and a cyclist. With open weights I trained that out: 27 false calls down to 9, for about $2. With a closed endpoint, the best I could have done was rephrase the question and hope.

**Parks don't have good signal.** With d1, the weights download once. After that the phone joins the laptop's hotspot with mobile data off, and the game runs in a field with zero bars.

**It's cheap enough to play all afternoon.** Locally it costs some battery. On Tinker it's under half a cent for a 10-round game of six. Neither is a reason to stop playing.

**I could look inside and swap parts.** The d1 weights *and* its inference code are on Hugging Face; reading its `runner.py` is how I learned to batch every player into one request per frame. Swapping the referee, the segmenter or the embedder is one flag, and every swap is logged.

A licence note, because "open" has fine print: the code is MIT and Qwen3.5-4B is Apache-2.0. d1 is under Liquid's **LFM Open License v1.0**, which allows free use, modification and redistribution, including commercial use by companies under US$10M in annual revenue. SAM 3 and DINOv3 are gated under Meta's own licences. The referee's training images come from COCO/Flickr under their own CC licences, many non-commercial, so the fine-tuned weights aren't in the repo. The README spells all of it out.

## Prize Categories

- Overall
- Best Use of Tinker: the referee is a Qwen3.5-4B LoRA fine-tuned on Tinker, scoring 93.2% against 87.4% for the untuned model and 63.1% for the local `d1-omni-600M` baseline on 222 hand-checked crops, with false "you did it" calls cut from 27 to 9.

Prop the phone up, step back, and let Simon be the one everybody argues with.
