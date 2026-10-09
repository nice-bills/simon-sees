"""The fine-tuned referee: Qwen3.5-4B + a LoRA trained on Tinker, asked one yes/no question per player crop.

    p = P(yes | crop, "Look at the person in this photo. Is that person standing on one foot? ...")

P(yes) is read from the logprobs of the first answer token (top-k at the first sampled position), so one
request costs one generated token. The same prompt format is used for training (scripts/train_referee.py),
evaluation (bench/referee_eval.py) and the game (`--decider tinker`), so what was measured is what plays.

Two ways to run it:
  * Tinker sampling (default when SIMON_TINKER_MODEL is a tinker:// path): `TinkerDecider`.
    Needs the `tinker` SDK and TINKER_API_KEY. Batches a round's crops as concurrent requests.
  * Locally, open weights: `LocalQwenDecider` loads Qwen/Qwen3.5-4B with transformers, applies the exported
    LoRA adapter with peft (or loads a merged checkpoint), and scores the same prompt. See docs/tinker.md.
"""

from __future__ import annotations

import io
import logging
import math
import os
import time
from typing import Any, Callable, Sequence

from PIL import Image

from .decider import Request, Result

log = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3.5-4B"
PATCH, MERGE = 16, 2  # Qwen3.5 vision: 16 px patches, 2x2 merge -> one token per 32x32 px
MIN_PIXELS, MAX_PIXELS = 65536, 16777216
YES_WORDS = ("yes", "Yes", " yes", " Yes", "YES")
NO_WORDS = ("no", "No", " no", " No", "NO")


# --------------------------------------------------------------------------- the prompt


def referee_prompt(instructions: str, criteria: dict | None = None) -> str:
    """The one prompt format. `instructions` is the game's question ("Look at the person in this photo. Is
    that person ...?"), `criteria` the command's {"true": ..., "false": ...} hints."""
    lines = [instructions.strip()]
    if criteria and criteria.get("true") and criteria.get("false"):
        lines.append(f"Yes means: {criteria['true']}. No means: {criteria['false']}.")
    lines.append("Answer with one word: yes or no.")
    return "\n".join(lines)


def smart_resize(height: int, width: int, factor: int = PATCH * MERGE, min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Qwen2-VL / Qwen3.5 image resizing (same as transformers' Qwen2VLImageProcessor)."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_tokens(img: Image.Image) -> int:
    h, w = smart_resize(img.height, img.width)
    return (h // (PATCH * MERGE)) * (w // (PATCH * MERGE))


def jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


_TOKENIZERS: dict[str, Any] = {}


def get_tokenizer(name: str | None = None):
    """The Qwen3.5 tokenizer (tokenizer.json from the HF repo, or a local dir via SIMON_TINKER_TOKENIZER)."""
    name = name or os.environ.get("SIMON_TINKER_TOKENIZER") or BASE_MODEL
    if name not in _TOKENIZERS:
        from transformers import AutoTokenizer

        _TOKENIZERS[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZERS[name]


def answer_token_ids(tok) -> tuple[set[int], set[int]]:
    def first(w):
        ids = tok.encode(w, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    yes = {i for i in map(first, YES_WORDS) if i is not None}
    no = {i for i in map(first, NO_WORDS) if i is not None}
    return yes, no


def prompt_segments(text: str) -> tuple[str, str]:
    """Qwen3.5 chat template, one user turn with one image, thinking disabled (the HF template with
    enable_thinking=False; the tinker-cookbook `qwen3_5_disable_thinking` renderer)."""
    before = "<|im_start|>user\n<|vision_start|>"
    after = f"<|vision_end|>{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return before, after


def build_model_input(img: Image.Image, text: str, tok, answer: str | None = None):
    """tinker.ModelInput for the prompt (and, for training, the answer + <|im_end|>). Returns
    (model_input, n_prompt_tokens, answer_token_ids)."""
    import tinker

    before, after = prompt_segments(text)
    a = tok.encode(before, add_special_tokens=False)
    b = tok.encode(after, add_special_tokens=False)
    chunk = tinker.types.ImageChunk(data=jpeg_bytes(img), format="jpeg", expected_tokens=image_tokens(img))
    ans: list[int] = []
    if answer is not None:
        ans = tok.encode(answer, add_special_tokens=False) + tok.encode("<|im_end|>", add_special_tokens=False)
    chunks = [tinker.types.EncodedTextChunk(tokens=a), chunk, tinker.types.EncodedTextChunk(tokens=b + ans)]
    n_prompt = len(a) + chunk.expected_tokens + len(b)
    return tinker.ModelInput(chunks=chunks), n_prompt, ans


def training_datum(img: Image.Image, text: str, label: bool, tok):
    """One SFT example: loss only on the answer word and <|im_end|> (next-token targets, right-shifted)."""
    import tinker

    answer = "yes" if label else "no"
    before, after = prompt_segments(text)
    a = tok.encode(before, add_special_tokens=False)
    b = tok.encode(after, add_special_tokens=False)
    ans = tok.encode(answer, add_special_tokens=False) + tok.encode("<|im_end|>", add_special_tokens=False)
    n_img = image_tokens(img)
    chunk = tinker.types.ImageChunk(data=jpeg_bytes(img), format="jpeg", expected_tokens=n_img)
    full_text_tail = b + ans
    # input = everything but the last token; target[i] = token at position i+1
    inp = tinker.ModelInput(chunks=[tinker.types.EncodedTextChunk(tokens=a), chunk,
                                    tinker.types.EncodedTextChunk(tokens=full_text_tail[:-1])])
    seq: list[int] = a + [0] * n_img + full_text_tail  # 0 stands in for image positions (weight 0)
    targets = seq[1:]
    weights = [0.0] * len(targets)
    for k in range(len(ans)):
        weights[len(targets) - len(ans) + k] = 1.0
    return tinker.Datum(model_input=inp, loss_fn_inputs={
        "target_tokens": tinker.TensorData(data=targets, dtype="int64", shape=[len(targets)]),
        "weights": tinker.TensorData(data=weights, dtype="float32", shape=[len(weights)]),
    })


def p_yes_from_topk(topk: Sequence[tuple[int, float]], yes_ids: set[int], no_ids: set[int]) -> float:
    """P(yes) / (P(yes) + P(no)) over the first answer position's top-k. 0.5 if neither shows up."""
    py = sum(math.exp(lp) for t, lp in topk if t in yes_ids)
    pn = sum(math.exp(lp) for t, lp in topk if t in no_ids)
    if py + pn <= 0:
        return 0.5
    return py / (py + pn)


# --------------------------------------------------------------------------- Tinker sampling


class TinkerScorer:
    """Scores (image, prompt) pairs with a Tinker sampling client: one generated token, top-20 logprobs."""

    def __init__(self, model_path: str | None = None, base_model: str = BASE_MODEL, client: Any = None,
                 tokenizer: Any = None, topk: int = 20):
        self.model_path = model_path
        self.base_model = base_model
        self.tok = tokenizer or get_tokenizer()
        self.yes_ids, self.no_ids = answer_token_ids(self.tok)
        self.topk = topk
        if client is None:
            import tinker

            sc = tinker.ServiceClient(**({"api_key": os.environ["SIMON_TINKER_API_KEY"]}
                                         if os.environ.get("SIMON_TINKER_API_KEY") else {}))
            client = (sc.create_sampling_client(model_path=model_path) if model_path
                      else sc.create_sampling_client(base_model=base_model))
        self.client = client

    def score(self, items: Sequence[tuple[Image.Image, str]]) -> list[float]:
        """All requests are sent at once (Tinker batches them server-side), then gathered."""
        import tinker

        params = tinker.SamplingParams(max_tokens=1, temperature=0.0)
        futs = []
        for img, text in items:
            mi, _, _ = build_model_input(img, text, self.tok)
            futs.append(self.client.sample(prompt=mi, num_samples=1, sampling_params=params,
                                           topk_sample_logprobs=self.topk))
        out = []
        for f in futs:
            r = f.result()
            seq = r.sequences[0]
            topk = (seq.topk_logprobs or [None])[0] or []
            if not topk and seq.tokens:  # no top-k: fall back to the sampled token itself
                t = seq.tokens[0]
                p = math.exp(seq.logprobs[0]) if seq.logprobs else 1.0
                out.append(p if t in self.yes_ids else (1 - p) if t in self.no_ids else 0.5)
                continue
            out.append(p_yes_from_topk(topk, self.yes_ids, self.no_ids))
        return out


class TinkerDecider:
    """`--decider tinker`: the fine-tuned referee via Tinker. Pose questions only (multi_image=False:
    freeze rounds fall back to pixel differencing, which is how freeze is judged anyway)."""

    multi_image = False

    def __init__(self, model_path: str | None = None, base_model: str = BASE_MODEL, scorer: TinkerScorer | None = None,
                 tracer: Callable | None = None):
        self.model_path = model_path or os.environ.get("SIMON_TINKER_MODEL")
        self.scorer = scorer or TinkerScorer(self.model_path, base_model)
        self.name = f"{base_model} + LoRA (Tinker {self.model_path})" if self.model_path else f"{base_model} (Tinker, untuned)"
        self._tracer = tracer
        self.spans: list[dict] = []  # last batch's timing spans, picked up by the recorder/event log

    def decide(self, requests: Sequence[Request]) -> list[Result]:
        if not requests:
            return []
        items, index = [], []
        for ri, r in enumerate(requests):
            for qname, q in r.questions.items():
                items.append((r.images[-1], referee_prompt(q["instructions"], q.get("criteria"))))
                index.append((ri, qname))
        t0 = time.perf_counter()
        ps = self.scorer.score(items)
        ms = 1e3 * (time.perf_counter() - t0)
        self.spans = [{"span": "tinker.sample", "requests": len(items), "ms": round(ms, 1),
                       "model": self.model_path or self.scorer.base_model}]
        answers: list[dict] = [{} for _ in requests]
        for (ri, qname), p in zip(index, ps):
            answers[ri][qname] = {"type": "noul", "noul": float(p)}
        results = [Result(answers=a, latency_ms=ms, raw={"backend": "tinker", "spans": self.spans}) for a in answers]
        if self._tracer:
            self._tracer(self.name, requests, results, ms)
        return results


# --------------------------------------------------------------------------- local open weights


class LocalQwenDecider:
    """The same referee without Tinker: Qwen3.5-4B + the downloaded LoRA (peft), or a merged checkpoint.
    SIMON_REFEREE_WEIGHTS = a dir with adapter_config.json (LoRA) or config.json (merged)."""

    multi_image = False

    def __init__(self, weights: str | None = None, base_model: str = BASE_MODEL, device: str = "auto",
                 tracer: Callable | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from .decider import pick_device

        weights = weights or os.environ.get("SIMON_REFEREE_WEIGHTS")
        if not weights:
            raise RuntimeError("LocalQwenDecider needs SIMON_REFEREE_WEIGHTS (see docs/tinker.md)")
        self.device = pick_device(device)
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        is_adapter = os.path.exists(os.path.join(weights, "adapter_config.json"))
        src = base_model if is_adapter else weights
        self.processor = AutoProcessor.from_pretrained(src)
        model = AutoModelForImageTextToText.from_pretrained(src, dtype=dtype)
        if is_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, weights).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.tok = self.processor.tokenizer
        self.yes_ids, self.no_ids = answer_token_ids(self.tok)
        self.name = f"{base_model} + LoRA (local {os.path.basename(weights.rstrip('/'))})"
        self._torch = torch
        self._tracer = tracer

    def _p(self, img: Image.Image, text: str) -> float:
        torch = self._torch
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[prompt], images=[img], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        py = sum(math.exp(lp[i].item()) for i in self.yes_ids)
        pn = sum(math.exp(lp[i].item()) for i in self.no_ids)
        return py / (py + pn) if py + pn > 0 else 0.5

    def decide(self, requests: Sequence[Request]) -> list[Result]:
        t0 = time.perf_counter()
        answers = []
        for r in requests:
            answers.append({q: {"type": "noul", "noul": self._p(r.images[-1], referee_prompt(v["instructions"], v.get("criteria")))}
                            for q, v in r.questions.items()})
        ms = 1e3 * (time.perf_counter() - t0)
        results = [Result(answers=a, latency_ms=ms, raw={"backend": "local-qwen"}) for a in answers]
        if self._tracer:
            self._tracer(self.name, requests, results, ms)
        return results
