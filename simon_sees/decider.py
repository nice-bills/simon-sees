"""The referee's judgement: a decision model that answers typed questions about one player's own photo.

The hero is Liquid AI's open d1 decision model (LiquidAI/d1-3B, or the experimental
LiquidAI/d1-omni-600M). It reads the frame(s) and every question in ONE forward pass and
returns calibrated probabilities, with zero generated tokens:

    model.system_one(None, {"p0_obey": {"type": "noul", "instructions": "..."}}, images=[img])
    -> {"answers": {"p0_obey": {"type": "noul", "noul": 0.93}}, "usage": {"input_tokens": n, "output_tokens": 0}}

`system_one_batch([(state, questions, images), ...])` takes several such requests in one call
(requests with images each get their own pass; the image is read once for all its questions).

The decision model never has to find anyone: SAM/DINOv3 (vision.py, identity.py) already cut out each
player's own crop, so every question is about "the person in this photo" (pose rounds) or "the person
in these two photos" (freeze: the same player's BEFORE and AFTER crops). No clothing, no positions.

Everything else in the app talks to the small `Decider` interface below, so tests and dry runs
can swap in a mock without torch installed.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

log = logging.getLogger(__name__)


@dataclass
class Request:
    """One pass: the images are the whole state, the questions are asked about them."""
    images: list  # PIL images, in order (before, after for freeze): one player's crops
    questions: dict[str, dict]  # name -> {"type": "noul"|"choice"|"score", "instructions", "criteria"?}
    meta: dict = field(default_factory=dict)  # {"player": pid, "frame": i, "kind": "pose"|"freeze"}: logs + mocks


@dataclass
class Result:
    answers: dict[str, dict]  # name -> the model's typed answer (noul: {"noul": p})
    latency_ms: float  # wall time of the batch call this request was part of
    input_tokens: int = 0
    raw: dict = field(default_factory=dict)

    def p(self, name: str) -> float:
        """P(yes) for a noul answer; for a choice, the top option's confidence."""
        a = self.answers[name]
        if a.get("type") == "noul" or "noul" in a:
            return float(a["noul"])
        return float(a.get("confidence", 0.0))

    def choice(self, name: str) -> str | None:
        return self.answers[name].get("choice")


class Decider(Protocol):
    name: str
    multi_image: bool  # can one request carry several images (needed for the freeze pair check)

    def decide(self, requests: Sequence[Request]) -> list[Result]: ...


# --------------------------------------------------------------------------- question builders


def noul(instructions: str, yes: str | None = None, no: str | None = None) -> dict:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if yes and no:
        q["criteria"] = {"true": yes, "false": no}
    return q


def action_question(check: str, yes: str = "", no: str = "") -> dict:
    """Pose/touch round, one player's crop: "Look at the person in this photo. Is that person standing on one foot?"."""
    return noul(f"Look at the person in this photo. {check}", yes or None, no or None)


def freeze_question(check: str, yes: str = "", no: str = "") -> dict:
    """Freeze round: the same player's BEFORE and AFTER crops in one request."""
    return noul(f"These two photos show the same person a moment apart. {check}", yes or None, no or None)


# --------------------------------------------------------------------------- d1


def pick_device(pref: str = "auto") -> str:
    import torch

    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(model_id: str, device: str, pref: str = "auto"):
    """The model cards' advice: d1-3B in bf16 on GPU/mps and fp32 on CPU; d1-omni-600M was trained in
    fp32 and should run fp16 on GPU (its card warns bf16 flips some answers)."""
    import torch

    if pref and pref != "auto":
        return {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}[pref]
    if device == "cpu":
        return torch.float32
    return torch.float16 if "omni" in model_id.lower() else torch.bfloat16


class D1Decider:
    """LiquidAI d1 through `transformers` with `trust_remote_code=True` (needs transformers>=5.14,
    d1-omni-600M needs >=5.15). Loading runs the model repo's own code, so the pinned revisions
    (D1_REVISIONS) are loaded exactly, never whatever is latest on the Hub."""

    multi_image = True

    # model repo commit each release was tested against (trust_remote_code runs repo code: pin it)
    D1_REVISIONS = {"LiquidAI/d1-3B": "051bcc464b01b9f92942b364d9586b0ef5912432",
                    "LiquidAI/d1-omni-600M": "02b55d7076f15129e59ab3f94783f32c4b088674"}

    def __init__(self, model_id: str = "LiquidAI/d1-3B", device: str = "auto", dtype: str = "auto",
                 compile: bool = False, tracer: Callable | None = None):
        import torch
        from transformers import AutoModel

        self.model_id = model_id
        self.name = model_id
        self.device = pick_device(device)
        self.dtype = pick_dtype(model_id, self.device, dtype)
        revision = self.D1_REVISIONS.get(model_id)
        if revision is None:
            raise ValueError(f"no trusted revision is configured for {model_id!r}; add and review its commit in D1_REVISIONS")
        t0 = time.perf_counter()
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, revision=revision,
                                               dtype=self.dtype).to(self.device)
        self.model.eval()
        if compile and self.device == "cuda" and hasattr(self.model, "compile"):
            # the d1-3B card: CUDA graphs for single questions on NVIDIA
            self.model.compile(mode="reduce-overhead")
        self.load_s = time.perf_counter() - t0
        self._torch = torch
        self._tracer = tracer
        log.info("loaded %s on %s (%s) in %.1fs", model_id, self.device, self.dtype, self.load_s)

    def decide(self, requests: Sequence[Request]) -> list[Result]:
        if not requests:
            return []
        batch = [(None, r.questions, list(r.images)) for r in requests]
        t0 = time.perf_counter()
        with self._torch.inference_mode():
            out = self.model.system_one_batch(batch)
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        ms = 1e3 * (time.perf_counter() - t0)
        results = [Result(answers=o["answers"], latency_ms=ms,
                          input_tokens=int(o.get("usage", {}).get("input_tokens", 0)), raw=o) for o in out]
        if self._tracer:
            self._tracer(self.model_id, requests, results, ms)
        return results

    def warmup(self, size: tuple[int, int] = (384, 288)) -> float:
        """The first call of a shape pays kernel selection: warm the shapes the game serves."""
        from PIL import Image

        img = Image.new("RGB", size, (110, 140, 90))
        t0 = time.perf_counter()
        crop = img.crop((0, 0, size[0] // 3, size[1]))  # a person-crop-shaped image
        self.decide([Request([crop], {"did": action_question("Is that person standing on one foot?")})] * 2)
        self.decide([Request([crop, crop], {"moved": freeze_question("Did that person move?")})])
        return time.perf_counter() - t0


# --------------------------------------------------------------------------- mocks


class MockDecider:
    """No model: answers come from `script(question_name, request.meta) -> p`, or are random (seeded).
    Used by the tests and by `--decider mock` dry runs so the whole loop runs on any laptop."""

    multi_image = True

    def __init__(self, script: Callable[[str, dict], float] | None = None, seed: int = 7,
                 latency_ms: float = 0.0, multi_image: bool = True):
        self.name = "mock"
        self.script = script
        self.rng = random.Random(seed)
        self.latency_ms = latency_ms
        self.multi_image = multi_image
        self.calls: list[list[Request]] = []

    def _p(self, name: str, meta: dict) -> float:
        if self.script:
            return float(self.script(name, meta))
        if name == "moved":
            return self.rng.choice([0.04, 0.08, 0.12, 0.9])
        return self.rng.choice([0.03, 0.2, 0.55, 0.8, 0.92, 0.97, 0.98])

    def decide(self, requests: Sequence[Request]) -> list[Result]:
        self.calls.append(list(requests))
        if self.latency_ms:
            time.sleep(self.latency_ms / 1e3)
        out = []
        for i, r in enumerate(requests):
            answers = {}
            for name, q in r.questions.items():
                if q["type"] == "choice":
                    opts = list(q["criteria"])
                    pick = opts[(sum(map(ord, name)) + i) % max(1, len(opts) - 1)]  # never "absent" by default
                    answers[name] = {"type": "choice", "choice": pick, "confidence": 0.8,
                                     "probabilities": {o: (0.8 if o == pick else 0.2 / (len(opts) - 1)) for o in opts}}
                else:
                    answers[name] = {"type": "noul", "noul": self._p(name, r.meta)}
            out.append(Result(answers=answers, latency_ms=self.latency_ms))
        return out


def tinker_available() -> bool:
    """A fine-tuned referee checkpoint on Tinker is configured and the SDK + key are present."""
    if not os.environ.get("SIMON_TINKER_MODEL"):
        return False
    if not (os.environ.get("TINKER_API_KEY") or os.environ.get("SIMON_TINKER_API_KEY")):
        return False
    try:
        import tinker  # noqa: F401
    except ImportError:
        return False
    return True


def local_referee_available() -> bool:
    w = os.environ.get("SIMON_REFEREE_WEIGHTS")
    return bool(w) and os.path.isdir(w)


def make_decider(kind: str, model_id: str, device: str = "auto", dtype: str = "auto", compile: bool = False,
                 tracer: Callable | None = None, seed: int = 7, tinker_model: str | None = None) -> Decider:
    """kind: d1 | mock | tinker (the fine-tuned Qwen3.5-4B referee on Tinker) | local-qwen (the same
    weights served locally) | auto (tinker if configured, else local weights, else d1)."""
    if kind == "mock":
        return MockDecider(seed=seed)
    if kind == "auto":
        if tinker_model or tinker_available():
            try:
                return make_decider("tinker", model_id, device, dtype, compile, tracer, seed, tinker_model)
            except Exception as e:  # noqa: BLE001
                log.warning("tinker referee unavailable (%s: %s); trying the next one", type(e).__name__, str(e)[:160])
        if local_referee_available():
            try:
                return make_decider("local-qwen", model_id, device, dtype, compile, tracer, seed)
            except Exception as e:  # noqa: BLE001
                log.warning("local referee weights failed (%s: %s); FALLBACK to d1", type(e).__name__, str(e)[:160])
        return D1Decider(model_id, device=device, dtype=dtype, compile=compile, tracer=tracer)
    if kind == "tinker":
        from .tinker_referee import TinkerDecider

        return TinkerDecider(model_path=tinker_model or os.environ.get("SIMON_TINKER_MODEL"), tracer=tracer)
    if kind == "local-qwen":
        from .tinker_referee import LocalQwenDecider

        return LocalQwenDecider(device=device, tracer=tracer)
    if kind == "d1":
        return D1Decider(model_id, device=device, dtype=dtype, compile=compile, tracer=tracer)
    raise ValueError(f"unknown decider {kind!r} (use auto, tinker, local-qwen, d1 or mock)")


def default_model_id() -> str:
    return os.environ.get("SIMON_MODEL", "LiquidAI/d1-3B")
