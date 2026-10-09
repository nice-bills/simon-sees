"""The referee's eyes, part 1: find every person in a frame, and describe what's inside each mask.

    SEGMENTER  sam3     SAM 3 through transformers (Sam3Model + Sam3Processor), text prompt "person".
                        Gated: accept the licence at https://huggingface.co/facebook/sam3
               sam3.1   SAM 3.1 checkpoints through Meta's own repo (github.com/facebookresearch/sam3,
                        `build_sam3_predictor(version="sam3.1")`). No transformers integration. CUDA only.
                        Gated: https://huggingface.co/facebook/sam3.1
               fallback FALLBACK: RT-DETR (PekingU/rtdetr_r18vd, Apache-2.0, ungated), COCO class "person".
                        Boxes only, no masks. Lets the app run before the SAM gates are accepted.
               mock     tests / CI only: fixed vertical bands, no model.
    EMBEDDER   dinov3   DINOv3 ViT (default facebook/dinov3-vitb16-pretrain-lvd1689m, gated) through
                        transformers AutoModel; the pooled CLS token is the person's signature.
               dinov2   FALLBACK: facebook/dinov2-small (Apache-2.0, ungated).
               mock     tests / CI only: a coarse pixel-layout signature, numpy only.

This is the pattern from Maziyar Panahi's DINOv3 + SAM demo: SAM isolates each person, DINOv3
compares what's inside each mask. Identity comes from the embedding alone, never from a text description of the person.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

SAM3_ID = "facebook/sam3"
SAM31_ID = "facebook/sam3.1"
DINOV3_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV2_ID = "facebook/dinov2-small"
FALLBACK_DETECTOR_ID = "PekingU/rtdetr_r18vd"


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels of the frame it came from
    score: float
    mask: np.ndarray | None = None  # HxW bool, frame-sized; None for box-only detectors (FALLBACK)
    track_id: int | None = None  # SAM 3.1 keeps an object id across the frames of one batch

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


class Segmenter(Protocol):
    name: str
    has_masks: bool

    def segment(self, images: Sequence[Image.Image]) -> list[list[Detection]]: ...


class Embedder(Protocol):
    name: str

    def embed(self, crops: Sequence[Image.Image]) -> np.ndarray: ...  # (n, d), L2-normalised


def _device(pref: str = "auto") -> str:
    import torch

    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(device: str, pref: str = "auto"):
    import torch

    if pref and pref != "auto":
        return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[pref]
    return torch.float32 if device == "cpu" else torch.bfloat16


def keep_people(dets: list[Detection], frame_h: int, min_score: float, min_height: float, max_people: int) -> list[Detection]:
    """Drop weak hits and tiny far-away figures. min_height is a fraction of the frame height: 0.12 keeps a
    player who steps back 8-10 m from a propped-up phone (0.25 used to drop anyone past ~5 m, playtest)."""
    out = [d for d in dets if d.score >= min_score and d.height >= min_height * frame_h]
    out.sort(key=lambda d: -d.score)
    return out[:max_people]


# --------------------------------------------------------------------------- segmenters


class Sam3Segmenter:
    """SAM 3 image model through transformers, prompted with the text "person"."""

    has_masks = True

    def __init__(self, model_id: str = SAM3_ID, device: str = "auto", dtype: str = "auto", prompt: str = "person",
                 threshold: float = 0.5, min_height: float = 0.12, max_people: int = 8, image_size: int | None = None):
        import torch
        from transformers import Sam3Config, Sam3Model, Sam3Processor

        self.device = _device(device)
        # fp32 on CPU: bf16 matmuls on a CPU are far slower than fp32 (measured: >5 min vs 17 s per frame)
        self.dtype = _dtype(self.device, dtype)
        # SAM 3 is trained at 1008 px. On a CPU we default to 560 px (the transformers docs' "custom
        # resolution" option, which warns accuracy may drop); people standing 3-5 m away still segment cleanly.
        env = os.environ.get("SAM3_IMAGE_SIZE")
        self.image_size = int(image_size or (env and int(env)) or (560 if self.device == "cpu" else 1008))
        self.name = model_id if self.image_size == 1008 else f"{model_id}@{self.image_size}px"
        self.prompt, self.threshold, self.min_height, self.max_people = prompt, threshold, min_height, max_people
        if self.image_size != 1008:
            config = Sam3Config.from_pretrained(model_id)
            config.image_size = self.image_size
            self.processor = Sam3Processor.from_pretrained(model_id, size={"height": self.image_size, "width": self.image_size})
            self.model = Sam3Model.from_pretrained(model_id, config=config, dtype=self.dtype)
        else:
            self.processor = Sam3Processor.from_pretrained(model_id)
            self.model = Sam3Model.from_pretrained(model_id, dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self._torch = torch

    def segment(self, images: Sequence[Image.Image]) -> list[list[Detection]]:
        torch = self._torch
        out: list[list[Detection]] = []
        for img in images:
            # the documented text-prompt path (transformers SAM3 docs, "Text-Only Prompts")
            inputs = self.processor(images=img, text=self.prompt, return_tensors="pt").to(self.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            with torch.inference_mode():
                res = self.model(**inputs)
            r = self.processor.post_process_instance_segmentation(
                res, threshold=self.threshold, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist())[0]
            dets = [Detection(tuple(float(v) for v in b), float(s), m.cpu().numpy().astype(bool))
                    for b, s, m in zip(r["boxes"], r["scores"], r["masks"])]
            out.append(keep_people(dets, img.height, self.threshold, self.min_height, self.max_people))
        return out


class Sam31Segmenter:
    """SAM 3.1 (Object Multiplex) via Meta's sam3 package. Its public predictor is a video predictor,
    which suits us: a frame batch is written as a tiny JPEG "video", prompted once with "person" on the
    first frame, and propagated, so every object keeps the same id across the batch for free.
    Needs `pip install git+https://github.com/facebookresearch/sam3` and a CUDA GPU (it is the Modal path)."""

    has_masks = True

    def __init__(self, prompt: str = "person", compile: bool = False, min_height: float = 0.12, max_people: int = 8,
                 threshold: float = 0.5):
        from sam3.model_builder import build_sam3_predictor  # type: ignore

        self.name = SAM31_ID
        self.prompt, self.min_height, self.max_people, self.threshold = prompt, min_height, max_people, threshold
        self.predictor = build_sam3_predictor(version="sam3.1", compile=compile, max_num_objects=16)

    def segment(self, images: Sequence[Image.Image]) -> list[list[Detection]]:
        with tempfile.TemporaryDirectory() as tmp:
            for i, img in enumerate(images):
                img.convert("RGB").save(Path(tmp) / f"{i:05d}.jpg", quality=92)
            sid = self.predictor.handle_request({"type": "start_session", "resource_path": tmp})["session_id"]
            try:
                first = self.predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0,
                                                       "text": self.prompt})
                per_frame: dict[int, dict] = {first["frame_index"]: first["outputs"]}
                if len(images) > 1:
                    for o in self.predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
                        per_frame[o["frame_index"]] = o["outputs"]
            finally:
                self.predictor.handle_request({"type": "close_session", "session_id": sid})
        out = []
        for i, img in enumerate(images):
            o = per_frame.get(i) or {}
            masks = np.asarray(o.get("out_binary_masks", np.zeros((0, img.height, img.width), bool)))
            probs = np.asarray(o.get("out_probs", np.ones(len(masks))))
            ids = np.asarray(o.get("out_obj_ids", np.arange(len(masks))))
            dets = []
            for m, p, oid in zip(masks, probs, ids):
                ys, xs = np.nonzero(m)
                if not len(xs):
                    continue
                dets.append(Detection((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
                                      float(p), m.astype(bool), int(oid)))
            out.append(keep_people(dets, img.height, self.threshold, self.min_height, self.max_people))
        return out


class FallbackDetector:
    """FALLBACK person finder: RT-DETR (Apache-2.0, ungated) boxes for COCO "person". No masks."""

    has_masks = False

    def __init__(self, model_id: str = FALLBACK_DETECTOR_ID, device: str = "auto", threshold: float = 0.5,
                 min_height: float = 0.12, max_people: int = 8):
        import torch
        from transformers import AutoImageProcessor, RTDetrForObjectDetection

        self.name = f"FALLBACK {model_id}"
        self.model_id = model_id
        self.device = _device(device)
        self.threshold, self.min_height, self.max_people = threshold, min_height, max_people
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.person = labels.get("person", 0)
        self._torch = torch

    def segment(self, images: Sequence[Image.Image]) -> list[list[Detection]]:
        torch = self._torch
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            res = self.model(**inputs)
        rs = self.processor.post_process_object_detection(res, target_sizes=[im.size[::-1] for im in images],
                                                          threshold=self.threshold)
        out = []
        for img, r in zip(images, rs):
            dets = [Detection(tuple(float(v) for v in b), float(s)) for s, l, b in zip(r["scores"], r["labels"], r["boxes"])
                    if int(l) == self.person]
            out.append(keep_people(dets, img.height, self.threshold, self.min_height, self.max_people))
        return out


class MockSegmenter:
    """Tests / CI: either a script `frame_index -> [Detection]`, or `n` equal vertical bands."""

    has_masks = False

    def __init__(self, script=None, n: int = 3):
        self.name = "mock"
        self.script = script
        self.n = n
        self.calls = 0

    def segment(self, images: Sequence[Image.Image]) -> list[list[Detection]]:
        out = []
        for img in images:
            i = self.calls
            self.calls += 1
            if self.script is not None:
                out.append(list(self.script(i, img)))
                continue
            w, h = img.size
            band = w / (self.n + 1)
            out.append([Detection((band * (k + 0.5), h * 0.15, band * (k + 1.5), h * 0.95), 0.9) for k in range(self.n)])
        return out


# --------------------------------------------------------------------------- embedders


class HFEmbedder:
    """DINOv3 (or the DINOv2 FALLBACK) through transformers: the pooled CLS token of each person crop."""

    def __init__(self, model_id: str = DINOV3_ID, device: str = "auto", dtype: str = "auto", fallback: bool = False):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.model_id = model_id
        self.name = f"FALLBACK {model_id}" if fallback else model_id
        self.device = _device(device)
        self.dtype = _dtype(self.device, dtype)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        self._torch = torch

    def embed(self, crops: Sequence[Image.Image]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 1), np.float32)
        torch = self._torch
        inputs = self.processor(images=list(crops), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(pixel_values=inputs.pixel_values.to(self.dtype))
        v = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        v = v.float().cpu().numpy()
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


class MockEmbedder:
    """Tests / CI: a 4x2 grid of mean RGB values per crop. Deterministic, numpy only. Not used for real play."""

    name = "mock"

    def embed(self, crops: Sequence[Image.Image]) -> np.ndarray:
        rows = []
        for c in crops:
            a = np.asarray(c.convert("RGB").resize((8, 16)), dtype=np.float32) / 255.0
            cells = [a[y:y + 4, x:x + 4].mean(axis=(0, 1)) for y in range(0, 16, 4) for x in range(0, 8, 4)]
            v = np.concatenate(cells) - 0.5
            rows.append(v)
        v = np.asarray(rows, dtype=np.float32).reshape(len(crops), -1)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


# --------------------------------------------------------------------------- crops


def _paint(a: np.ndarray, where: np.ndarray, value: float) -> None:
    a[where] = value


def crop_person(img: Image.Image, det: Detection, pad: float = 0.12, background: str = "dim",
                long_side: int | None = 384, others: Sequence[Detection] = ()) -> Image.Image:
    """The player's own picture: their box with a little margin (hands, feet, the ground they touch).

    Isolation, so a player's card and the referee's input show this player and nobody else:
      - with a SAM mask, everything outside the mask is dimmed ("dim", grass/trees stay faintly visible
        for "touch something green") or blanked to grey ("blank");
      - `others` (the other people found in the same frame) are painted flat grey wherever they are not
        this player: their masks when they have one, otherwise the part of their box outside this
        player's box. That is what removes the neighbour who used to stand in the margin of the crop.
    `background="keep"` turns all of it off (raw box crop)."""
    w, h = img.size
    x0, y0, x1, y1 = det.box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    box = (max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px + 0.5)), min(h, int(y1 + py + 0.5)))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        box = (0, 0, w, h)
    src = img.convert("RGB")
    if background in ("dim", "blank"):
        a = np.asarray(src, dtype=np.float32).copy()
        own = det.mask if det.mask is not None and det.mask.shape == (h, w) else None
        if own is not None:
            m = own[..., None].astype(np.float32)
            bg = a * (0.35 if background == "dim" else 0.0) + (0 if background == "dim" else 128)
            a = a * m + bg * (1 - m)
        if others:
            mine = own if own is not None else np.zeros((h, w), bool)
            if own is None:
                bx0, by0, bx1, by1 = (int(round(v)) for v in det.box)
                mine = mine.copy()
                mine[max(0, by0):max(0, by1), max(0, bx0):max(0, bx1)] = True
            for o in others:
                if o is det:
                    continue
                if o.mask is not None and o.mask.shape == (h, w):
                    om = o.mask
                else:
                    om = np.zeros((h, w), bool)
                    ox0, oy0, ox1, oy1 = (int(round(v)) for v in o.box)
                    om[max(0, oy0):max(0, oy1), max(0, ox0):max(0, ox1)] = True
                _paint(a, om & ~mine, 96.0)
        src = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    c = src.crop(box)
    if long_side and max(c.size) != long_side:
        k = long_side / max(c.size)
        c = c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))), Image.Resampling.BILINEAR)
    return c


# --------------------------------------------------------------------------- factory


@dataclass
class VisionConfig:
    segmenter: str = field(default_factory=lambda: os.environ.get("SEGMENTER", "sam3"))  # sam3 | sam3.1 | fallback | mock
    embedder: str = field(default_factory=lambda: os.environ.get("EMBEDDER", "dinov3"))  # dinov3 | dinov2 | mock
    embedder_model: str | None = field(default_factory=lambda: os.environ.get("EMBEDDER_MODEL"))
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    mock_people: int = 3


def make_segmenter(cfg: VisionConfig) -> Segmenter:
    kind = cfg.segmenter.lower()
    if kind == "sam3":
        return Sam3Segmenter(device=cfg.device, dtype=cfg.dtype)
    if kind in ("sam3.1", "sam31"):
        return Sam31Segmenter(compile=cfg.compile)
    if kind == "fallback":
        return FallbackDetector(device=cfg.device)
    if kind == "mock":
        return MockSegmenter(n=cfg.mock_people)
    raise ValueError(f"unknown SEGMENTER {cfg.segmenter!r} (sam3 | sam3.1 | fallback | mock)")


def make_embedder(cfg: VisionConfig) -> Embedder:
    kind = cfg.embedder.lower()
    if kind == "dinov3":
        return HFEmbedder(cfg.embedder_model or DINOV3_ID, cfg.device, cfg.dtype)
    if kind == "dinov2":
        return HFEmbedder(cfg.embedder_model or DINOV2_ID, cfg.device, cfg.dtype, fallback=True)
    if kind == "mock":
        return MockEmbedder()
    raise ValueError(f"unknown EMBEDDER {cfg.embedder!r} (dinov3 | dinov2 | mock)")


def timed(fn, *a, **kw) -> tuple[Any, float]:
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return r, 1e3 * (time.perf_counter() - t0)
