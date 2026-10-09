# Optional: run the referee on a DigitalOcean GPU Droplet (+ Sentry tracing)

The whole point of Simon Sees is that it runs on a laptop you own, offline. This path exists for when you
don't have a GPU laptop or Apple Silicon Mac: rent a GPU for the afternoon of the game, point the phone at
it, delete the Droplet afterwards.

**Privacy trade-off:** with this setup the camera frames go to a server you rent. They are processed in
memory and never written to disk unless you set `SIMON_SAVE_FRAMES`, but they do leave your phone.

## 1. Create the Droplet

1. DigitalOcean control panel → **Create → GPU Droplets**.
2. Pick a single-GPU plan (d1-3B needs ~7 GB of VRAM in bf16, so any of the offered NVIDIA GPUs is
   plenty) and the **AI/ML Ready** image (NVIDIA drivers + Docker preinstalled).
3. Add your SSH key, create, and note the public IP.

## 2. Run it

```bash
ssh root@<droplet-ip>
git clone https://github.com/nice-bills/simon-sees && cd simon-sees
docker build -t simon-sees .
docker run -d --gpus all -p 8000:8000 \
  -v hf-cache:/home/simon/.cache/huggingface \
  -e SIMON_DEVICE=cuda \
  -e SENTRY_DSN="$SENTRY_DSN" \
  --name simon simon-sees python -m simon_sees serve --host 0.0.0.0 --port 8000 --compile
docker logs -f simon      # first start downloads ~6 GB of weights, then "warming up"
```

`--compile` turns on CUDA graphs for single questions, as the d1-3B model card recommends for NVIDIA.

## 3. HTTPS for the phone camera

Quickest: on the Droplet, `cloudflared tunnel --url http://localhost:8000` and open the printed
`https://…trycloudflare.com` URL on the phone (see docs/phone-setup.md). For something permanent, put
Caddy in front with a domain name (`caddy reverse-proxy --from simon.example.com --to localhost:8000`).

## 4. Sentry Agent Tracing (optional)

Set `SENTRY_DSN` (and `pip install sentry-sdk`, already in requirements-extras.txt). Each round becomes an
`invoke_agent Simon Sees referee` transaction, and every d1 call becomes a child `gen_ai.evaluate` span
with the model id, input tokens, `output_tokens = 0`, number of questions/images and latency. Images are
never attached; question text and answers are attached only with `SENTRY_SEND_PROMPTS=1`.

In Sentry: **Insights → Agents** shows the spans; filter `span.op:gen_ai.evaluate` to plot d1 latency
per round. Screenshot that for the write-up.

## 5. Tear down

`docker rm -f simon`, then destroy the Droplet in the control panel (GPU Droplets bill by the hour while
they exist, even when idle).
