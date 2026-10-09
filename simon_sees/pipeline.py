"""One frame batch in, everything the referee needs out:  SAM -> DINOv3 -> match -> d1.

    process_batch(stack, task, frames, identities) -> dict   (JSON-serialisable)

`task["kind"]`:
    enrol    one frame: every person, ordered left to right, with their photo (JPEG, base64) and embedding
    observe  match the enrolled players in each frame (who is in the picture?)
    pose     observe + one d1 question per seen player per frame, asked about that player's own crop
    freeze   frames = [BEFORE, AFTER]: observe both + one d1 question per player on the pair of their crops
             (pixel differencing inside the player's box is logged as a second opinion / FALLBACK)

The same function runs in-process (`LocalVision`) or on a Modal GPU (`RemoteVision` posts the frames and
the roster's embeddings to deploy/modal_app.py, which calls process_batch and returns this dict), so the
game engine cannot tell the two apart. The roster lives with the game; the GPU is stateless.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .decider import Decider, Request, action_question, freeze_question
from .identity import Identity, MatchConfig, assign, order_left_to_right
from .motion import motion_score, moved_probability, union_box
from .vision import (DINOV2_ID, DINOV3_ID, Detection, Embedder, HFEmbedder, MockEmbedder, MockSegmenter, Segmenter,
                     VisionConfig, crop_person, make_segmenter)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- (de)serialising frames


def jpeg_b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def from_b64(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


# --------------------------------------------------------------------------- the stack


@dataclass
class Stack:
    segmenter: Segmenter
    embedder: Embedder
    decider: Decider | None = None
    match: MatchConfig = field(default_factory=MatchConfig)
    crop_background: str = "dim"  # dim | keep | blank (only matters when the segmenter has masks)

    @property
    def models(self) -> dict:
        return {"segmenter": self.segmenter.name, "embedder": self.embedder.name,
                "decider": getattr(self.decider, "name", None)}


def build_embedder(cfg: VisionConfig) -> Embedder:
    """dinov3 is preferred; `auto` falls back to the ungated DINOv2 (labelled FALLBACK) while the
    DINOv3 gate is not accepted or the weights can't be fetched."""
    kind = cfg.embedder.lower()
    if kind == "mock":
        return MockEmbedder()
    if kind in ("dinov3", "auto"):
        try:
            return HFEmbedder(cfg.embedder_model or DINOV3_ID, cfg.device, cfg.dtype)
        except Exception as e:  # noqa: BLE001 - GatedRepoError, OSError, offline...
            if kind == "dinov3":
                raise
            log.warning("DINOv3 unavailable (%s: %s); FALLBACK to %s", type(e).__name__, str(e)[:120], DINOV2_ID)
    if kind in ("dinov2", "auto"):
        return HFEmbedder(DINOV2_ID if kind == "auto" else (cfg.embedder_model or DINOV2_ID), cfg.device, cfg.dtype,
                          fallback=True)
    raise ValueError(f"unknown EMBEDDER {cfg.embedder!r} (auto | dinov3 | dinov2 | mock)")


def build_segmenter(cfg: VisionConfig) -> Segmenter:
    """sam3 is preferred; `auto` falls back to the ungated RT-DETR person detector (FALLBACK, boxes only)."""
    kind = cfg.segmenter.lower()
    if kind == "auto":
        try:
            return make_segmenter(VisionConfig(segmenter="sam3", device=cfg.device, dtype=cfg.dtype))
        except Exception as e:  # noqa: BLE001
            log.warning("SAM 3 unavailable (%s: %s); FALLBACK to the RT-DETR person detector",
                        type(e).__name__, str(e)[:120])
            return make_segmenter(VisionConfig(segmenter="fallback", device=cfg.device))
    return make_segmenter(cfg)


def build_stack(cfg: VisionConfig, decider: Decider | None = None, match: MatchConfig | None = None) -> Stack:
    return Stack(build_segmenter(cfg), build_embedder(cfg), decider, match or MatchConfig())


# --------------------------------------------------------------------------- the batch


def _round_box(b) -> list[float]:
    return [round(float(v), 1) for v in b]


def process_batch(stack: Stack, task: dict, frames: Sequence[Image.Image],
                  identities: Sequence[Identity] = ()) -> dict:
    kind = task.get("kind", "observe")
    t_all = time.perf_counter()
    lat = {"segment": 0.0, "embed": 0.0, "match": 0.0, "decide": 0.0}

    t0 = time.perf_counter()
    dets_per_frame = stack.segmenter.segment(list(frames))
    lat["segment"] = 1e3 * (time.perf_counter() - t0)

    # two crops per person: for d1 the background is dimmed (context like grass or a tree still counts);
    # for DINO it is blanked to grey so the signature is the person and not whoever stands next to them
    crops_per_frame = [[crop_person(f, d, background=stack.crop_background, others=dets) for d in dets]
                       for f, dets in zip(frames, dets_per_frame)]
    embed_crops = [[crop_person(f, d, pad=0.04, background="blank", long_side=224, others=dets) if d.mask is not None else c
                    for d, c in zip(dets, cs)] for f, dets, cs in zip(frames, dets_per_frame, crops_per_frame)]
    t0 = time.perf_counter()
    flat = [c for cs in embed_crops for c in cs]
    flat_embs = stack.embedder.embed(flat) if flat else np.zeros((0, 1), np.float32)
    lat["embed"] = 1e3 * (time.perf_counter() - t0)
    embs_per_frame, k = [], 0
    for cs in crops_per_frame:
        embs_per_frame.append(flat_embs[k:k + len(cs)])
        k += len(cs)

    out: dict[str, Any] = {"kind": kind, "models": stack.models, "frames": [], "decisions": {}, "updates": {}}

    if kind == "enrol":
        f, dets, crops, embs = frames[0], dets_per_frame[0], crops_per_frame[0], embs_per_frame[0]
        order = order_left_to_right([d.box for d in dets])
        out["people"] = [{"box": _round_box(dets[i].box), "score": round(dets[i].score, 3),
                          "emb": embs[i].astype(np.float32).round(5).tolist(),
                          "photo": jpeg_b64(_thumb(crops[i]), 80), "has_mask": dets[i].mask is not None}
                         for i in order]
        out["frames"].append({"size": list(f.size), "detections": len(dets)})
        lat["total"] = 1e3 * (time.perf_counter() - t_all)
        out["latency_ms"] = {k2: round(v, 1) for k2, v in lat.items()}
        return out

    # --- who is who, frame by frame (tracking: last boxes move forward as we go)
    idents = [Identity(i.pid, list(i.gallery), i.last_box, i.max_gallery) for i in identities]
    seen: list[dict[int, int]] = []  # per frame: pid -> detection index
    t0 = time.perf_counter()
    for f, dets, embs in zip(frames, dets_per_frame, embs_per_frame):
        ms = assign(idents, embs, [d.box for d in dets], stack.match)
        frame_players, frame_seen = {}, {}
        for m in ms:
            if m.det is None:
                frame_players[str(m.pid)] = {"seen": False, "sim": round(m.sim, 4)}
                continue
            d = dets[m.det]
            frame_seen[m.pid] = m.det
            frame_players[str(m.pid)] = {"seen": True, "sim": round(m.sim, 4), "box": _round_box(d.box),
                                         "score": round(d.score, 3)}
            ident = next(i for i in idents if i.pid == m.pid)
            if m.sim >= stack.match.remember_above:
                ident.remember(embs[m.det], d.box)
                out["updates"][str(m.pid)] = {"emb": embs[m.det].astype(np.float32).round(5).tolist(),
                                              "box": _round_box(d.box)}
            else:
                ident.last_box = d.box
                out["updates"].setdefault(str(m.pid), {"emb": None, "box": _round_box(d.box)})
                out["updates"][str(m.pid)]["box"] = _round_box(d.box)
        seen.append(frame_seen)
        out["frames"].append({"size": list(f.size), "detections": len(dets), "players": frame_players})
    lat["match"] = 1e3 * (time.perf_counter() - t0)

    if kind in ("pose", "freeze"):
        requests: list[Request] = []
        crops: dict[int, list[Image.Image]] = {}
        if kind == "pose":
            q = action_question(task["check"], task.get("yes", ""), task.get("no", ""))
            for fi, frame_seen in enumerate(seen):
                for pid, di in frame_seen.items():
                    requests.append(Request([crops_per_frame[fi][di]], {"did": q},
                                            meta={"player": pid, "frame": fi, "kind": "pose"}))
                    crops.setdefault(pid, []).append(crops_per_frame[fi][di])
        else:
            q = freeze_question(task["check"], task.get("yes", ""), task.get("no", ""))
            both = [pid for pid in seen[0] if pid in seen[1]]
            pairs = {pid: (dets_per_frame[0][seen[0][pid]], dets_per_frame[1][seen[1][pid]]) for pid in both}
            # the same box on both frames, so the decision model compares like with like
            for pid, (db, da) in pairs.items():
                ub = union_box(db.box, da.box)
                crops[pid] = [crop_person(frames[0], Detection(ub, db.score, db.mask), background=stack.crop_background,
                                          others=[o for o in dets_per_frame[0] if o is not db]),
                              crop_person(frames[1], Detection(ub, da.score, da.mask), background=stack.crop_background,
                                          others=[o for o in dets_per_frame[1] if o is not da])]
                px = motion_score(frames[0], frames[1], ub)
                out["decisions"][str(pid)] = {"pixel_motion": round(px, 5),
                                              "p_pixel": round(moved_probability(px, task.get("pixel_lo", 0.01),
                                                                                 task.get("pixel_hi", 0.06)), 4)}
            if task.get("use_model", True) and stack.decider is not None and getattr(stack.decider, "multi_image", False):
                for pid in both:
                    requests.append(Request(list(crops[pid]), {"moved": q}, meta={"player": pid, "kind": "freeze"}))
        results = []
        if requests and stack.decider is not None:
            t0 = time.perf_counter()
            results = stack.decider.decide(requests)
            lat["decide"] = 1e3 * (time.perf_counter() - t0)
            spans = getattr(stack.decider, "spans", None)
            if spans:  # the Tinker referee's own timing (network + sampling), next to the whole decide step
                out["spans"] = list(spans)
                lat["tinker"] = round(sum(sp.get("ms", 0.0) for sp in spans), 1)
            out["input_tokens"] = int(results[0].input_tokens) if results else 0
            out["requests"] = len(requests)
        out["question"] = q["instructions"]
        for req, res in zip(requests, results):
            pid = str(req.meta["player"])
            d = out["decisions"].setdefault(pid, {})
            if kind == "pose":
                d.setdefault("p", [None] * len(frames))
                d["p"][req.meta["frame"]] = round(res.p("did"), 4)
            else:
                d["p_moved"] = round(res.p("moved"), 4)
        if task.get("return_crops"):
            out["crops"] = {str(pid): [jpeg_b64(c, 80) for c in cs] for pid, cs in crops.items()}

    lat["total"] = 1e3 * (time.perf_counter() - t_all)
    out["latency_ms"] = {k2: round(v, 1) for k2, v in lat.items()}
    return out


def _thumb(c: Image.Image, h: int = 240) -> Image.Image:
    k = h / c.height
    return c.resize((max(1, round(c.width * k)), h), Image.Resampling.BILINEAR)


# --------------------------------------------------------------------------- backends


class LocalVision:
    """Everything in this process (laptop CPU/GPU/mps, or the sandbox)."""

    backend = "local"

    def __init__(self, stack: Stack):
        self.stack = stack

    @property
    def models(self) -> dict:
        return self.stack.models

    def run(self, task: dict, frames: Sequence[Image.Image], identities: Sequence[Identity] = ()) -> dict:
        return process_batch(self.stack, task, frames, identities)


class RemoteVision:
    """The vision stack + d1 on a remote GPU (deploy/modal_app.py on Modal). One HTTPS POST per frame batch.

    SIMON_VISION_URL     the endpoint printed by `modal deploy deploy/modal_app.py` (…modal.run).
                         SIMON_MODAL_URL is accepted as an older alias.
    SIMON_VISION_TOKEN   "<token id>.<token secret>" of a Modal proxy token, sent as `Authorization: Bearer`
    SIMON_MODAL_KEY/SECRET  the same token as the separate Modal-Key / Modal-Secret headers
    """

    backend = "modal"

    def __init__(self, url: str | None = None, key: str | None = None, secret: str | None = None,
                 token: str | None = None, timeout: float | None = None):
        self.url = (url or os.environ.get("SIMON_VISION_URL") or os.environ.get("SIMON_MODAL_URL", "")).rstrip("/")
        if not self.url:
            raise RuntimeError("--backend modal needs SIMON_VISION_URL (see docs/modal.md)")
        self.token = token or os.environ.get("SIMON_VISION_TOKEN")
        self.key = key or os.environ.get("SIMON_MODAL_KEY")
        self.secret = secret or os.environ.get("SIMON_MODAL_SECRET")
        # the first call after a cold start loads SAM 3.1 + DINOv3 + d1-3B on the GPU
        self.timeout = timeout or float(os.environ.get("SIMON_VISION_TIMEOUT", "300"))
        self._models: dict = {}

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        elif self.key and self.secret:
            h["Modal-Key"], h["Modal-Secret"] = self.key, self.secret
        return h

    def _call(self, path: str, body: dict | None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method="POST" if data is not None else "GET",
                                     headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - the user's own endpoint
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace")
            raise RuntimeError(f"remote vision {path} failed: HTTP {e.code} {detail}") from e

    @property
    def models(self) -> dict:
        if not self._models:
            try:
                self._models = self._call("/info", None).get("models", {})
            except Exception as e:  # noqa: BLE001
                log.warning("remote vision /info failed: %s", e)
                return {"segmenter": "remote (unreachable)", "embedder": "remote (unreachable)",
                        "decider": "remote (unreachable)"}
        return self._models

    def run(self, task: dict, frames: Sequence[Image.Image], identities: Sequence[Identity] = ()) -> dict:
        t0 = time.perf_counter()
        out = self._call("/batch", {"task": task, "frames": [jpeg_b64(f, 85) for f in frames],
                                    "identities": [i.to_json() for i in identities]})
        out.setdefault("latency_ms", {})["round_trip"] = round(1e3 * (time.perf_counter() - t0), 1)
        self._models = out.get("models", self._models)
        return out


def mock_vision(decider: Decider | None = None, people: int = 3) -> LocalVision:
    """No models at all: vertical-band 'people' and a pixel-layout signature. Tests, CI and `--segmenter mock`."""
    return LocalVision(Stack(MockSegmenter(n=people), MockEmbedder(), decider))
