"""Simon Sees vision + referee on a Modal GPU: SAM 3.1 -> DINOv3 -> LiquidAI d1-3B behind one web endpoint.

Run these from YOUR laptop (see docs/modal.md), not from a CI box or a sandbox:

    pip install modal && modal token set --token-id <id> --token-secret <secret>
    modal secret create huggingface HF_TOKEN=hf_...      # an HF token whose account accepted the SAM 3.1 gate
    modal deploy deploy/modal_app.py                     # prints https://<workspace>--simon-sees-vision-web.modal.run

Then on the laptop that runs the game server:

    export SIMON_VISION_URL=https://<workspace>--simon-sees-vision-web.modal.run
    export SIMON_VISION_TOKEN=<proxy token id>.<proxy token secret>     # or SIMON_MODAL_KEY + SIMON_MODAL_SECRET
    python -m simon_sees serve --backend modal

The endpoint is stateless. The server POSTs one frame batch plus the enrolled players' embeddings, and
gets back exactly the dict `simon_sees.pipeline.process_batch` returns locally:

    GET  /info    -> {"models": {"segmenter", "embedder", "decider"}, "gpu": ...}
    POST /batch   {"task": {...}, "frames": [<jpeg base64>...], "identities": [Identity.to_json()...]} -> dict

Overridable with env vars at deploy time: SIMON_GPU (default L4), SIMON_MODEL (default LiquidAI/d1-3B),
SIMON_SEGMENTER (sam3.1 | sam3, default sam3.1), EMBEDDER (auto | dinov3 | dinov2, default auto: DINOv3,
falling back to DINOv2-small, labelled FALLBACK, if the DINOv3 gate is not accepted on that HF account).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
GPU = os.environ.get("SIMON_GPU", "L4")
MODEL_ID = os.environ.get("SIMON_MODEL", "LiquidAI/d1-3B")
SEGMENTER = os.environ.get("SIMON_SEGMENTER", "sam3.1")
EMBEDDER = os.environ.get("EMBEDDER", "auto")
CACHE = "/cache"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch>=2.7", "torchvision", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers==5.17.0", "accelerate>=1.0", "safetensors>=0.6", "huggingface_hub>=1.0", "hf_xet",
        "pillow>=10", "numpy>=1.26", "scipy", "fastapi>=0.110", "opencv-python-headless",
        # SAM 3.1 has no transformers integration; it ships in Meta's repo (build_sam3_predictor)
        "git+https://github.com/facebookresearch/sam3",
    )
    .env({"HF_HOME": CACHE, "HF_HUB_ENABLE_HF_TRANSFER": "0", "SIMON_SEGMENTER": SEGMENTER, "EMBEDDER": EMBEDDER,
          "SIMON_MODEL": MODEL_ID})
    .add_local_python_source("simon_sees")
)

app = modal.App("simon-sees-vision", image=image)
weights = modal.Volume.from_name("simon-sees-hf-cache", create_if_missing=True)


@app.cls(
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={CACHE: weights},
    timeout=600,
    scaledown_window=120,  # an idle container stays warm (and billed) this long; a game pauses <30 s between batches
    max_containers=1,  # one game = one GPU; raise it if several groups play at once
)
@modal.concurrent(max_inputs=1)
class Vision:
    @modal.enter()
    def load(self) -> None:
        import logging

        from simon_sees.decider import D1Decider
        from simon_sees.identity import MatchConfig
        from simon_sees.pipeline import build_stack
        from simon_sees.vision import VisionConfig

        logging.basicConfig(level=logging.INFO)
        cfg = VisionConfig(segmenter=os.environ["SIMON_SEGMENTER"], embedder=os.environ["EMBEDDER"], device="cuda")
        decider = D1Decider(os.environ["SIMON_MODEL"], device="cuda", dtype="bf16")
        if hasattr(decider, "warmup"):
            decider.warmup((640, 480))
        self.stack = build_stack(cfg, decider, MatchConfig())
        weights.commit()  # keep downloaded weights for the next cold start

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        import torch
        from fastapi import FastAPI, HTTPException

        from simon_sees.identity import Identity
        from simon_sees.pipeline import from_b64, process_batch

        api = FastAPI(title="simon-sees-vision")
        stack = self.stack

        @api.get("/info")
        def info() -> dict:
            return {"models": stack.models, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

        @api.post("/batch")
        def batch(body: dict) -> dict:
            frames = [from_b64(f) for f in body.get("frames", [])]
            if not frames:
                raise HTTPException(400, "no frames")
            idents = [Identity.from_json(d) for d in body.get("identities", [])]
            return process_batch(stack, body.get("task", {}), frames, idents)

        return api
