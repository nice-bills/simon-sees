# Playtesting Simon Sees

You run the game on your laptop, your phone is the camera and the speaker, and your friends play. While you
play, the laptop quietly records what the referee saw and decided. Afterwards one command sends that recording
to a **private** Hugging Face dataset so it can be studied and the referee improved.

If a call is wrong, tap **Wrong call**. Those taps are the most useful thing you can send.

## 1. What you need

- A laptop (macOS, Linux or Windows) with **Python 3.10 or newer** (`python3 --version`). 16 GB of RAM is
  comfortable for the real models.
- A phone with a browser (Safari or Chrome), on the **same Wi-Fi** as the laptop.
- A free **Hugging Face** account: <https://huggingface.co/join>.
- About **10 GB of free disk** for the models (downloaded once). Not needed for the `mock` test run.
- 2–5 friends, some space outdoors or in a big room, and a way to prop the phone up 3–5 m away.

## 2. Get the code

Either unzip the file you were sent:

```bash
unzip simon-sees-src.zip && cd simon-sees
```

or clone it:

```bash
git clone https://github.com/nice-bills/simon-sees && cd simon-sees
```

## 3. Install, and set your Hugging Face token

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: py -m venv .venv; .venv\Scripts\activate
pip install -r requirements-model.txt                   # the models' libraries (a few minutes)
```

(`run_playtest.sh` in step 5 does this for you if you prefer.)

The token lets the laptop download the models and upload your sessions. It needs **write** access:

1. Open <https://huggingface.co/settings/tokens> and click **Create new token**.
2. Choose **Fine-grained**, name it `simon-sees`.
3. Under **Repositories**, tick **Write access to contents/settings of all repos under your personal
   namespace** (this also lets it create the private playtest dataset). Also tick **Read access to contents
   of all public gated repos you can access**, so it can download SAM 3.
   (Simpler alternative: choose the **Write** token type instead of Fine-grained.)
4. Click **Create token** and copy it (starts with `hf_`).
5. Accept the SAM 3 licence once, logged in: <https://huggingface.co/facebook/sam3> → **Agree**.
   (DINOv3 is optional: <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>. Without it the
   game uses DINOv2 automatically.)

Then, in the terminal you will run the game from:

```bash
export HF_TOKEN=hf_your_token_here                      # Windows PowerShell: $env:HF_TOKEN = "hf_..."
```

Never paste the token into a file you share, a chat, or the code. It only needs to live in that terminal.

The sessions go to `nice-bill/simon-sees-playtests` by default. If your Hugging Face username is not
`nice-bill`, point it at your own account (it is created for you, private):

```bash
export SIMON_HF_REPO=your-username/simon-sees-playtests
```

## 4. Start the server and open it on the phone

```bash
python -m simon_sees serve                              # first start downloads the models (~6–10 GB)
```

Wait until it says `Uvicorn running on http://0.0.0.0:8000`.

**Phones only allow a web page to use the camera over HTTPS**, so the plain `http://<laptop-ip>:8000` will not
work on the phone. The easiest fix is a free Cloudflare tunnel, in a second terminal:

```bash
# install once: macOS: brew install cloudflared · Windows: winget install Cloudflare.cloudflared
# Linux: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
cloudflared tunnel --url http://localhost:8000
```

It prints a link like `https://some-random-words.trycloudflare.com`. Open that on the phone. (No internet
where you play? [phone-setup.md](phone-setup.md), option C, makes a local HTTPS certificate instead, so the
phone talks to the laptop over Wi-Fi or the phone's hotspot only.)

### Which models? (speed matters)

| how you start it | what referees | speed | use it for |
|---|---|---|---|
| `python -m simon_sees serve` | SAM 3 → DINOv3 → d1-3B, on the laptop | **Apple Silicon or NVIDIA GPU: fine. CPU only: very slow** (SAM 3 took ~17 s per frame on a 2-core machine, so each round takes about a minute) | the real test, if your laptop has a GPU or an M-series chip |
| `python -m simon_sees serve --segmenter fallback --embedder dinov2 --model LiquidAI/d1-omni-600M` | lighter models (RT-DETR, DINOv2-small, d1-omni-600M) | lighter on CPU, but not timed end to end yet (d1-omni-600M alone took ~2.5 s per question on 2 cores); expect several seconds per round | an ordinary laptop with no GPU |
| `python -m simon_sees serve --segmenter mock --embedder mock --decider mock` | nobody: the calls are random | instant, no downloads | checking the phone link, camera, recording and upload work, before the real thing |
| `python -m simon_sees serve --backend modal` | the full models on a cloud GPU | fast | if you have set up Modal ([modal.md](modal.md)): `export SIMON_VISION_URL=...` first |

## 5. Or do steps 3–4 in one go

```bash
./run_playtest.sh          # or: ./run_playtest.sh fast   ·   ./run_playtest.sh mock
```

Windows: `.\run_playtest.ps1` (or `.\run_playtest.ps1 fast`). It installs everything the first time, starts the
server, starts the Cloudflare tunnel if `cloudflared` is installed and prints the phone link, and when you
press **Ctrl+C** at the end it uploads the session for you (or makes a zip if `HF_TOKEN` is not set).

## 6. Play

1. On the phone: **Turn camera on**, flip to the back camera, prop the phone so everyone is in shot head to toe.
2. Everyone lines up facing the phone, a little apart. Tap **Line up**, type names if you like, tap **Start**.
3. Play. Turn the volume up.
4. **When the referee gets one wrong, tap Wrong call.**
   - On a red "out" screen, **Wrong call** means "that player should not be out". One tap.
   - On a green or yellow screen, **Wrong call** opens *Ref's notes*: tap **Wrong call** next to the player
     who should have gone out.
   - Missed the moment? Open **Ref's notes** any time: the last call is listed, one row per player.
   - Tapped by mistake? Tap **Undo**.
5. At the end, write a sentence in **Anything Simon got wrong?** and tap **Send note**. Play again as often as
   you like.

Everything is recorded on the laptop in `sessions/` (frames, each player's crop, timings, your corrections).
To play without recording: `SIMON_RECORD=0 python -m simon_sees serve`.

## 7. Send the sessions

Stop the server (Ctrl+C), then:

```bash
python -m simon_sees upload
```

It prints the link of your private dataset. Running it again only sends what is new. To have it upload by
itself at the end of every game, start the server with `SIMON_UPLOAD=hf python -m simon_sees serve`.

If the upload fails (no token, no internet), make a zip and send it in the chat instead:

```bash
python -m simon_sees bundle
```

**Privacy:** the sessions contain photos of the people who played. Ask them first. The dataset is created
private and the upload refuses to send to a public repo. Delete `sessions/` on the laptop when you are done.
