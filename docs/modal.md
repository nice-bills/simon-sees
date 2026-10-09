# Running the vision stack on a Modal GPU

By default everything runs on your laptop. On a CPU-only laptop SAM 3 is slow (about 16.6 s per frame at
560 px on the 2-vCPU sandbox this was built in, see `bench/results-sam3-cpu.md`), so the game ships a second
backend: the game, the roster, the voice and the phone page stay on your laptop, and each frame batch goes
over HTTPS to one GPU container on [Modal](https://modal.com) that runs

    SAM 3.1 (facebook/sam3.1, "person")  ->  DINOv3 embeddings  ->  match to the line-up  ->  LiquidAI/d1-3B per crop

`deploy/modal_app.py` is that container. It is stateless: the laptop sends the frames plus the enrolled
players' embeddings, and gets back the same dict the local pipeline produces.

> **Status: written, not yet deployed.** The Modal app was written in a sandbox where the Modal CLI could not
> authenticate, so it has never been deployed or timed. No GPU latency numbers exist yet. Run it from your
> laptop and fill in the table at the bottom.

## 1. One-time setup (on your laptop)

```bash
pip install modal
modal token set --token-id <ak-...> --token-secret <as-...>   # from modal.com > Settings > API tokens
                                                               # (or `modal setup` to do it in the browser)
```

Accept the model licences on Hugging Face with the account whose token you will give Modal:

- SAM 3.1: <https://huggingface.co/facebook/sam3.1> (required)
- DINOv3: <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m> (optional; without it the
  container falls back to `facebook/dinov2-small` and reports the embedder as `FALLBACK facebook/dinov2-small`)
- LiquidAI/d1-3B is ungated (LFM Open License v1.0; read its revenue threshold in the README licence section)

```bash
modal secret create huggingface HF_TOKEN=hf_xxx
```

## 2. Deploy

```bash
modal deploy deploy/modal_app.py
# -> https://<workspace>--simon-sees-vision-web.modal.run
```

Knobs, read at deploy time: `SIMON_GPU=L4` (default; `A10`, `L40S`, `A100` also work), `SIMON_MODEL=LiquidAI/d1-3B`
(or `LiquidAI/d1-omni-600M`), `SIMON_SEGMENTER=sam3.1` (or `sam3` for the transformers model), `EMBEDDER=auto`.
For example `SIMON_GPU=A10 modal deploy deploy/modal_app.py`.

The endpoint requires a Modal **proxy token** (`requires_proxy_auth=True`), so nobody else can spend your GPU
time. Create one at modal.com > Settings > Proxy auth tokens, or `modal workspace proxy-tokens create`.

Check it:

```bash
curl -H "Authorization: Bearer $TOKEN_ID.$TOKEN_SECRET" https://<workspace>--simon-sees-vision-web.modal.run/info
```

The first call cold-starts the container: image pull, weight download into the `simon-sees-hf-cache` volume
(first deploy only), and loading SAM 3.1 + DINO + d1-3B onto the GPU. Expect that to take minutes the very first
time; later cold starts skip the download.

## 3. Play with it

```bash
export SIMON_VISION_URL=https://<workspace>--simon-sees-vision-web.modal.run
export SIMON_VISION_TOKEN=$TOKEN_ID.$TOKEN_SECRET       # or SIMON_MODAL_KEY=... SIMON_MODAL_SECRET=...
python -m simon_sees serve                              # --backend modal is the default once SIMON_VISION_URL is set
```

`/api/info` on the laptop then reports the models the container loaded. Each batch's log entry carries
`latency_ms.round_trip` (laptop to GPU and back) next to the container's own `segment / embed / match / decide`.
`SIMON_VISION_TIMEOUT` (default 300 s) bounds each call, generous so the first, cold call does not time out.

## Cost notes

Modal bills GPU containers per second while they run, including load time and an idle window after the last
request. List prices on <https://modal.com/pricing> (checked 2026-10-08): L4 $0.000222/s (about $0.80/hour),
A10 $0.000306/s (about $1.10/hour), L40S $0.000542/s (about $1.95/hour), plus small CPU and memory charges. The
Starter plan includes $30 of compute per month.

- `scaledown_window=120` in `deploy/modal_app.py`: the container stays warm (and billed) for 2 minutes after
  the last batch, long enough to bridge the pauses between rounds. Lower it to save money, raise it to avoid cold
  starts mid-game.
- `max_containers=1`: one game, one GPU. A 30-minute afternoon game on an L4 that stays warm the whole time is
  about 30 x $0.0133 = $0.40 of GPU time at list price, before CPU/memory and the cold start. That is arithmetic
  on the list price, not a measurement.
- Nothing runs, and nothing is billed, when nobody is playing. `modal app stop simon-sees-vision` stops it for good.

## Measured on Modal

| GPU | cold start | segment | embed | decide (d1-3B) | round trip |
|---|---|---|---|---|---|
| _not run yet_ | [fill in] | [fill in] | [fill in] | [fill in] | [fill in] |

## Privacy

Frames leave your laptop on this backend: they go to your own Modal workspace, are processed in memory and not
stored (the app writes nothing but model weights to its volume). If that is not acceptable for your group, stay
on the local backend.
