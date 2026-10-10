"""FastAPI server: serves the phone page, takes its camera frames over a WebSocket, runs the line-up and
the game, and sends back what Simon says (as server-side TTS audio when an open TTS is installed).

Frames arrive as binary WebSocket messages (one JPEG each). Control messages are JSON:
    {"type": "lineup"}                                   SAM finds everyone, DINO embeds them, photos come back
    {"type": "start", "names": {"0": "Kofi", "2": "Ama"}, "skip": [1]}     {"type": "stop"}
Server -> phone JSON:
    {"type": "lineup", "players": [{"id", "number", "photo", "box"}], "models": {...}, "view": {"w", "h", "url"}}
    {"type": "lineup", "status": "follow", "players": [{"id", "box" | null, "seen"}], "view": {...}}  (rings follow, lobby)
    {"type": "state", ...}  (players[*].box + view {seq, w, h, url}: the phone's rings, drawn over /api/view/<seq>.jpg)     {"type": "say", "text": ..., "mood": ..., "audio": "/api/audio/<key>.wav" | null}
HTTP:
    POST /api/feedback  {"kind": "wrong_call", "round_id", "pid", "correct": "out"|"safe"}  or  {"kind": "note", "text"}
                        -> feedback.jsonl of the playtest session (recorder.py), linked by round_id

`--backend local` runs SAM/DINO/d1 in this process; `--backend modal` sends each frame batch to the
Modal GPU endpoint (deploy/modal_app.py) and only keeps the game, the roster and the voice here.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image

from . import tracing
from .decider import Decider, make_decider
from .decision_log import DecisionLog
from .engine import Frame, Game, Settings, lineup, players_from_lineup
from .identity import Identity
from .recorder import detect_device, make_recorder
from .replay import fit, iter_frames
from .voice import Ears, Voice, estimate_speech_s, parse_name, wav_duration

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
# a 30 MP image decodes to ~90 MB of RGB: real phone frames never come close, decompression bombs do.
# Pillow only *warns* at its own default (~89 MP), so it is tightened here.
Image.MAX_IMAGE_PIXELS = 30_000_000


class RateLimiter:
    """Per-client token bucket for the unauthenticated-ish frame/audio drop-off points (DoS bound)."""

    def __init__(self, rate_per_s: float, burst: float):
        self.rate, self.burst = rate_per_s, burst
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, seen = self._buckets.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - seen) * self.rate)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False
        self._buckets[key] = (tokens - 1.0, now)
        return True


@dataclass
class Config:
    backend: str = "local"  # local | modal
    decider: str = "auto"  # auto = fine-tuned referee (Tinker or local weights) if configured, else d1
    model_id: str = "LiquidAI/d1-3B"
    segmenter: str = "auto"  # auto (sam3, else FALLBACK) | sam3 | sam3.1 | fallback | mock
    embedder: str = "auto"  # auto (dinov3, else FALLBACK dinov2) | dinov3 | dinov2 | mock
    embedder_model: str | None = None
    match_threshold: float = 0.45
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    tts: str = "auto"
    stt: bool = True
    replay: str | None = None
    replay_fps: float = 3.0
    side: int = 640
    log_path: str | None = "logs/decisions.jsonl"
    frame_timeout_s: float = 4.0
    speech_scale: float = 1.0  # <1 shortens the pause for each spoken line (tests, demos)
    settings: Settings = field(default_factory=Settings)
    record: bool | None = None  # playtest recorder; None = on unless SIMON_RECORD=0
    sessions_dir: str | None = None  # default SIMON_SESSIONS or ./sessions
    upload: str | None = field(default_factory=lambda: os.environ.get("SIMON_UPLOAD") or None)  # "hf" = auto upload
    hf_repo: str | None = None  # default SIMON_HF_REPO or nice-bill/simon-sees-playtests
    # rings follow the players between judged looks (lobby and game): auto = on for --backend local, off for modal
    # (every follow look would be a GPU request); never in a replay (the replay is the camera, kept reproducible)
    follow: str = field(default_factory=lambda: os.environ.get("SIMON_FOLLOW", "auto"))
    # game token: the phone page and every /api + /ws call must present it (?token=... or X-Simon-Token).
    # "" = no auth (the library/test default); `python -m simon_sees serve` generates one unless --no-auth.
    token: str = ""
    max_upload_mb: float = float(os.environ.get("SIMON_MAX_UPLOAD_MB", "16"))


def build_vision(cfg: Config, decider: Decider | None = None):
    """The vision backend the game talks to (pipeline.LocalVision or pipeline.RemoteVision)."""
    from .identity import MatchConfig
    from .pipeline import LocalVision, RemoteVision, Stack, build_stack
    from .vision import MockEmbedder, MockSegmenter, VisionConfig

    if cfg.backend == "modal":
        return RemoteVision()
    if decider is None:
        decider = make_decider(cfg.decider, cfg.model_id, cfg.device, cfg.dtype, cfg.compile,
                               tracer=tracing.decision_tracer, seed=cfg.settings.seed or 7)
        if hasattr(decider, "warmup"):
            log.info("warming up %s ...", cfg.model_id)
            decider.warmup((cfg.side, cfg.side * 3 // 4))
    match = MatchConfig(threshold=cfg.match_threshold)
    if cfg.segmenter == "mock" and cfg.embedder == "mock":
        return LocalVision(Stack(MockSegmenter(), MockEmbedder(), decider, match))
    vcfg = VisionConfig(segmenter=cfg.segmenter, embedder=cfg.embedder, embedder_model=cfg.embedder_model,
                        device=cfg.device, dtype=cfg.dtype, compile=cfg.compile)
    return LocalVision(build_stack(vcfg, decider, match))


class FrameHub:
    """The latest camera frame, and a way to wait for fresh ones."""

    def __init__(self, timeout: float = 4.0):
        self.timeout = timeout
        self.latest: Frame | None = None
        self.count = 0
        self._cond = asyncio.Condition()

    async def put(self, image: Image.Image) -> None:
        async with self._cond:
            self.latest = Frame(image, time.time())
            self.count += 1
            self._cond.notify_all()

    async def fresh(self, timeout: float | None = None) -> Frame:
        """The next frame that arrives after this call (stale frames would judge the past)."""
        timeout = self.timeout if timeout is None else timeout
        start = self.count
        async with self._cond:
            try:
                await asyncio.wait_for(self._cond.wait_for(lambda: self.count > start), timeout)
            except asyncio.TimeoutError:
                if self.latest is None:
                    raise RuntimeError("no camera frames: is the phone page open with the camera on?")
                log.warning("no fresh frame in %.1fs, judging the last one", timeout)
            return self.latest  # type: ignore[return-value]


class Hub:
    """Connected phones/spectators, plus the IO the game talks to."""

    def __init__(self, voice: Voice, frames: FrameHub, speech_scale: float = 1.0, replay=None, recorder=None):
        self.speech_scale = speech_scale
        self.rec = recorder
        self.last_tts_ms: float | None = None
        self.replay = replay
        self.sockets: set[WebSocket] = set()
        self.voice = voice
        self.frames_hub = frames
        self.audio: dict[str, bytes] = {}
        self.last_state: dict = {}

    async def broadcast(self, msg: dict) -> None:
        dead = []
        for ws in list(self.sockets):
            try:
                await ws.send_text(json.dumps(msg, default=str))
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)

    # --- GameIO
    async def say(self, text: str, mood: str = "info") -> None:
        url, seconds = None, estimate_speech_s(text)
        self.last_tts_ms = None
        if self.voice.available:
            try:
                t0 = time.perf_counter()
                wav = await asyncio.to_thread(self.voice.synth, text)
                self.last_tts_ms = round(1e3 * (time.perf_counter() - t0), 1)
                key = hashlib.sha1(text.encode()).hexdigest()[:16]
                self.audio[key] = wav
                url, seconds = f"/api/audio/{key}.wav", wav_duration(wav)
            except Exception as e:  # noqa: BLE001 - never let TTS stop the game
                log.warning("TTS failed (%s); phone will speak it", e)
        log.info("SIMON: %s", text)
        if self.rec is not None:
            self.rec.event("say", text=text, mood=mood, tts=self.voice.engine, tts_ms=self.last_tts_ms,
                           round=self.last_state.get("round"), game_id=self.last_state.get("game_id"))
        await self.broadcast({"type": "say", "text": text, "mood": mood, "audio": url, "seconds": round(seconds, 2)})
        await asyncio.sleep(seconds * self.speech_scale + 0.15)

    async def frames(self, n: int, gap_s: float) -> list[Frame]:
        out = []
        for i in range(n):
            if i:
                await asyncio.sleep(gap_s)
            if self.replay is not None:
                # dry run: every look takes the next replayed frame, so a seeded game is reproducible
                img = await asyncio.to_thread(next, self.replay)
                await self.frames_hub.put(img)
                out.append(self.frames_hub.latest)  # type: ignore[arg-type]
            else:
                out.append(await self.frames_hub.fresh())
        return out

    async def update(self, state: dict) -> None:
        self.last_state = state
        await self.broadcast({"type": "state", **state})


def create_app(cfg: Config, decider: Decider | None = None, vision=None) -> FastAPI:
    tracing.init()
    frames = FrameHub(cfg.frame_timeout_s)
    voice = Voice(cfg.tts)
    ears = Ears(cfg.stt)
    rec = make_recorder(cfg.record, cfg.sessions_dir, meta={
        "mode": "replay" if cfg.replay else "live", "backend": cfg.backend, "seed": cfg.settings.seed,
        "requested": {"decider": cfg.decider, "model": cfg.model_id, "segmenter": cfg.segmenter,
                      "embedder": cfg.embedder, "embedder_model": cfg.embedder_model, "device": cfg.device,
                      "dtype": cfg.dtype, "compile": cfg.compile, "match_threshold": cfg.match_threshold,
                      "side": cfg.side, "tts": cfg.tts, "stt": cfg.stt, "replay": cfg.replay},
        "settings": asdict(cfg.settings), "auto_upload": cfg.upload})
    hub = Hub(voice, frames, cfg.speech_scale, iter_frames(cfg.replay, cfg.side) if cfg.replay else None, rec)
    dlog = DecisionLog(cfg.log_path)
    st: dict[str, Any] = {"vision": vision, "model_error": None, "game": None, "task": None, "games": [],
                          "loading": vision is None, "lineup": None, "photos": {}, "lining_up": False,
                          "rec": rec, "uploading": None, "follow_ids": None, "follow_for": None, "look_ms": None}
    vision_lock = asyncio.Lock()  # one look at a time: line-up, the game's looks, and follow looks
    if vision is None and decider is not None:
        st["vision"], st["loading"] = build_vision(cfg, decider), False

    def note_models() -> None:
        v = st["vision"]
        rec.update_meta(models=dict(getattr(v, "models", {}) or {}), backend=getattr(v, "backend", cfg.backend),
                        device=detect_device(v, cfg.device), tts=voice.engine, stt=ears.engine,
                        model_error=st["model_error"])

    def load_models() -> None:
        try:
            st["vision"] = build_vision(cfg)
        except Exception as e:  # noqa: BLE001
            log.exception("models failed to load")
            st["model_error"] = f"{type(e).__name__}: {e}"
        finally:
            st["loading"] = False
            note_models()

    async def auto_upload(reason: str) -> None:
        """SIMON_UPLOAD=hf: push this session to the private HF dataset. Never breaks the game."""
        if (cfg.upload or "").lower() != "hf" or not rec.enabled:
            return
        if st["uploading"] and not st["uploading"].done():
            await st["uploading"]
        from .upload import upload_sessions

        async def go() -> None:
            try:
                res = await asyncio.to_thread(upload_sessions, rec.dir.parent, cfg.hf_repo, None, [rec.dir])
                log.info("auto upload (%s): %s", reason, res.get("uploaded") or "up to date")
            except Exception as e:  # noqa: BLE001
                log.warning("auto upload failed (%s): %s. Run `python -m simon_sees upload` later.", reason, e)

        st["uploading"] = asyncio.create_task(go())

    def prewarm_voice() -> None:
        """Synthesise every fixed line once, so Simon never pauses mid-game to render a command."""
        if not voice.available:
            return
        from .commands import COMMANDS, Call
        from .engine import SAFE_LINES, TRICK_SAFE

        lines = [Call(c, s).text for c in COMMANDS for s in (True, False)] + SAFE_LINES + TRICK_SAFE + ["Relax."]
        t0 = time.perf_counter()
        for line in lines:
            try:
                voice.synth(line)
            except Exception as e:  # noqa: BLE001
                log.warning("TTS prewarm failed: %s", e)
                return
        log.info("TTS: %d lines ready in %.1fs", len(lines), time.perf_counter() - t0)

    def follow_enabled() -> bool:
        mode = (cfg.follow or "auto").lower()
        if cfg.replay or mode in ("0", "off", "false", "no"):
            return False
        return mode in ("1", "on", "true", "yes") or (mode == "auto" and cfg.backend != "modal")

    async def lobby_follow(frame: Frame) -> bool:
        """Between Line up and Start: find the people on the team sheet again so their rings follow them."""
        people = st["lineup"]
        if not people or st["lining_up"] or vision_lock.locked():
            return False
        if st["follow_for"] is not people:  # a fresh line-up: one identity per person, from their line-up crop
            st["follow_ids"] = [Identity(i, [np.asarray(p["emb"], dtype=np.float32)],
                                         tuple(p["box"]) if p.get("box") else None) for i, p in enumerate(people)]
            st["follow_for"] = people
        ids = st["follow_ids"]
        async with vision_lock:
            t0 = time.perf_counter()
            res = await asyncio.to_thread(st["vision"].run, {"kind": "observe"}, [frame.image], ids)
            st["look_ms"] = 1e3 * (time.perf_counter() - t0)
        if st["lineup"] is not people or st["lining_up"] or (st["task"] and not st["task"].done()):
            return False  # lined up again / the game started meanwhile: this look is stale
        found = (res.get("frames") or [{}])[0].get("players") or {}
        players = []
        for ident in ids:
            info = found.get(str(ident.pid)) or {}
            box = [float(v) for v in info["box"]] if info.get("seen") and info.get("box") else None
            if box:
                ident.last_box = tuple(box)
                people[ident.pid]["box"] = box  # the game starts tracking from here, not the line-up spot
            players.append({"id": ident.pid, "box": box, "seen": bool(box)})
        st["lineup_frame"] = frame
        w, h = frame.image.size
        await hub.broadcast({"type": "lineup", "status": "follow", "players": players,
                             "view": {"w": w, "h": h, "url": f"/api/view/l{int(time.time() * 1000)}.jpg"}})
        return True

    async def follow_loop() -> None:
        """Rings follow the players: a quick look (no referee model) about once a second while nothing else is
        looking. Turns itself down when looks are slow (SAM 3 on a CPU), so it never holds up a judged look."""
        if not follow_enabled():
            return
        every = max(0.3, cfg.settings.follow_every_s)
        last = -1
        while True:
            await asyncio.sleep(every)
            try:
                frame = frames.latest
                if st["vision"] is None or frame is None or frames.count == last or not hub.sockets:
                    continue
                if st["look_ms"] is not None and st["look_ms"] > cfg.settings.follow_max_ms:
                    continue
                last = frames.count
                game = st["game"] if st["task"] and not st["task"].done() else None
                if game is not None:
                    done = await game.follow(frame)
                    if done and game.look_ms is not None:
                        st["look_ms"] = game.look_ms
                else:
                    await lobby_follow(frame)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - following is a nicety; it must never break anything
                log.debug("follow look failed: %s", e)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        if st["vision"] is None:
            threading.Thread(target=load_models, daemon=True).start()
        else:
            note_models()
        threading.Thread(target=prewarm_voice, daemon=True).start()
        follower = asyncio.create_task(follow_loop())
        yield
        follower.cancel()
        if st["task"]:
            st["task"].cancel()
        if st["uploading"] and not st["uploading"].done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(st["uploading"], 30)
        rec.close()

    app = FastAPI(title="Simon Sees", lifespan=lifespan)

    frame_limiter = RateLimiter(rate_per_s=10, burst=20)   # the phone sends 3 fps; scripts get slack
    stt_limiter = RateLimiter(rate_per_s=6 / 60, burst=5)  # one name registration per player, so plenty
    max_upload = int(cfg.max_upload_mb * 1_000_000)
    if max_upload <= 0:
        raise ValueError("max_upload_mb must be positive")

    async def read_upload(upload: UploadFile) -> bytes:
        """Read at most the configured limit, including requests without Content-Length."""
        data = await upload.read(max_upload + 1)
        if len(data) > max_upload:
            raise HTTPException(413, "payload too large")
        return data

    @app.middleware("http")
    async def gate(request: Request, call_next):
        """The game token guards the page and every API route; oversized uploads are refused unread."""
        path = request.url.path
        if cfg.token and (path == "/" or path.startswith("/api/")):
            given = request.query_params.get("token") or request.headers.get("x-simon-token") or ""
            if not secrets.compare_digest(given, cfg.token):
                return JSONResponse({"error": "bad or missing token"}, status_code=401)
        # Enforce the cap while consuming the ASGI stream, so chunked requests cannot evade it.
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_upload:
                return JSONResponse({"error": "payload too large"}, status_code=413)
            body.extend(chunk)
        sent = False

        async def replay_body():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        request._receive = replay_body
        return await call_next(request)

    def info() -> dict:
        v = st["vision"]
        models = dict(getattr(v, "models", {}) or {}) if v is not None and not st["loading"] else {}
        return {"model": models.get("decider") or cfg.model_id, "models": models, "backend": cfg.backend,
                "decider": cfg.decider, "loading": st["loading"], "model_error": st["model_error"],
                "tts": voice.engine, "stt": ears.engine, "replay": bool(cfg.replay), "frames_received": frames.count,
                "tracing": tracing.enabled(), "playing": bool(st["task"] and not st["task"].done()),
                "lineup": len(st["lineup"]) if st["lineup"] else 0,
                "recording": rec.enabled, "session": rec.dir.name if rec.enabled else None}

    @app.get("/api/info")
    async def api_info() -> dict:
        return info()

    @app.get("/api/state")
    async def api_state() -> dict:
        return hub.last_state or {"phase": "lobby"}

    @app.get("/api/audio/{key}.wav")
    async def api_audio(key: str) -> Response:
        if key not in hub.audio:
            raise HTTPException(404)
        return Response(hub.audio[key], media_type="audio/wav", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/photo/{pid}.jpg")
    async def api_photo(pid: int) -> Response:
        """A player's own line-up crop: their card on the team sheet. Kept in memory only."""
        b64 = st["photos"].get(pid)
        if not b64:
            raise HTTPException(404)
        return Response(base64.b64decode(b64), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    def _jpeg(img: Image.Image) -> Response:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=82)
        return Response(buf.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/view/{key}.jpg")
    async def api_view(key: str) -> Response:
        """The frame the HUD rings are drawn on: `l<stamp>` = the line-up frame, a number = the game's last
        judged frame (state["view"]). Before anything was looked at, a replay shows its first frame."""
        game = st["game"]
        if not key.startswith("l") and game is not None and getattr(game, "view_frame", None) is not None:
            return _jpeg(game.view_frame.image)
        if st.get("lineup_frame") is not None:
            return _jpeg(st["lineup_frame"].image)
        if frames.latest is not None:
            return _jpeg(frames.latest.image)
        if cfg.replay:
            try:
                return _jpeg(next(iter_frames(cfg.replay, cfg.side, loop=False)))
            except Exception:  # noqa: BLE001
                pass
        raise HTTPException(404)

    @app.post("/api/stt")
    async def api_stt(request: Request, audio: UploadFile = File(...)) -> dict:
        if not ears.available:
            raise HTTPException(501, "speech recognition not installed; type your name")
        if not stt_limiter.allow(request.client.host if request.client else "?"):
            raise HTTPException(429, "too many transcriptions, try again in a moment")
        payload = await read_upload(audio)
        text = await asyncio.to_thread(ears.transcribe, payload)
        return {"text": text, "name": parse_name(text)}

    @app.post("/api/frame")
    async def api_frame(request: Request, file: UploadFile = File(...)) -> dict:
        """HTTP alternative to the WebSocket for sending a frame (handy for scripts)."""
        if not frame_limiter.allow(request.client.host if request.client else "?"):
            raise HTTPException(429, "too many frames, slow down")
        try:
            img = fit(Image.open(io.BytesIO(await read_upload(file))), cfg.side)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - a torn or oversized (bomb) image
            raise HTTPException(422, f"not a usable image: {type(e).__name__}") from e
        await frames.put(img)
        return {"ok": True, "frames": frames.count}

    def find_round(round_id: str) -> dict | None:
        for g in reversed(st["games"]):
            r = g.rounds.get(round_id)
            if r:
                return r
        return None

    @app.post("/api/feedback")
    async def api_feedback(body: dict = Body(...)) -> dict:
        """The players' corrections: "Wrong call" on a verdict (which player, what it should have been) and the
        end-of-game note. Appended to feedback.jsonl, linked to events.jsonl by round_id."""
        kind = str(body.get("kind") or "")
        game = st["game"]
        if kind == "wrong_call":
            rid = str(body.get("round_id") or "")
            correct = str(body.get("correct") or "")
            if correct not in ("out", "safe"):
                raise HTTPException(422, "correct must be 'out' or 'safe'")
            r = find_round(rid)
            if r is None:
                raise HTTPException(404, f"unknown round {rid!r}")
            try:
                pid = int(body.get("pid"))
            except (TypeError, ValueError):
                raise HTTPException(422, "pid must be a player id") from None
            d = next((x for x in r.get("decisions", []) if x.get("pid") == pid), None)
            if d is None:
                raise HTTPException(404, f"player {pid} was not judged in {rid}")
            row = rec.feedback(kind="wrong_call", round_id=rid, game_id=rid.rsplit("-r", 1)[0], round=r.get("round"),
                               command=r.get("command"), kind_of_round=r.get("kind"), simon=r.get("simon"),
                               pid=pid, player=d.get("player"), number=d.get("number"),
                               given="out" if d.get("out") else "safe", given_reason=d.get("reason"), p=d.get("p"),
                               correct=correct, retracted=bool(body.get("retracted")),
                               where=str(body.get("where") or "")[:16] or None)
        elif kind == "note":
            text = str(body.get("text") or "").strip()[:2000]
            if not text:
                raise HTTPException(422, "empty note")
            row = rec.feedback(kind="note", text=text, game_id=str(body.get("game_id") or (game.id if game else "")) or None,
                               rounds=game.round if game else None)
            asyncio.create_task(auto_upload("note"))
        else:
            raise HTTPException(422, "kind must be 'wrong_call' or 'note'")
        return {"ok": True, "recorded": rec.enabled, "session": rec.dir.name if rec.enabled else None,
                "feedback": row or None}

    def ready() -> str | None:
        if st["vision"] is None:
            return st["model_error"] or "the models are still loading"
        return None

    async def do_lineup() -> None:
        if (err := ready()) or st["lining_up"]:
            if err:
                await hub.broadcast({"type": "error", "message": err})
            return
        if st["task"] and not st["task"].done():
            await hub.broadcast({"type": "error", "message": "a game is already running"})
            return
        st["lining_up"] = True
        await hub.broadcast({"type": "lineup", "status": "looking", "players": []})
        try:
            async with vision_lock:
                t0 = time.perf_counter()
                res = await lineup(st["vision"], hub)
                st["look_ms"] = 1e3 * (time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            log.exception("line-up failed")
            await hub.broadcast({"type": "error", "message": f"line-up failed: {type(e).__name__}: {e}"})
            return
        finally:
            st["lining_up"] = False
        people = res.get("people", [])[:8]
        st["lineup"] = people
        st["photos"] = {i: p["photo"] for i, p in enumerate(people)}
        if rec.enabled:
            stamp_s = time.strftime("%H%M%S")
            crops = []
            for i, p in enumerate(people):
                try:
                    crops.append(rec.crop(Image.open(io.BytesIO(base64.b64decode(p["photo"]))), f"lineup-{stamp_s}_p{i}.jpg"))
                except Exception:  # noqa: BLE001
                    crops.append(None)
            rec.event("lineup", people=len(people), boxes=[p["box"] for p in people], scores=[p.get("score") for p in people],
                      models=res.get("models"), latency_ms=res.get("latency_ms"), crops=crops,
                      frame=rec.frame(frames.latest.image, f"lineup-{stamp_s}.jpg") if frames.latest else None)
        dlog.write("lineup", people=len(people), boxes=[p["box"] for p in people], models=res.get("models"),
                   latency_ms=res.get("latency_ms"))
        stamp = int(time.time() * 1000)
        st["lineup_frame"] = frames.latest
        view = ({"w": frames.latest.image.size[0], "h": frames.latest.image.size[1], "url": f"/api/view/l{stamp}.jpg"}
                if frames.latest is not None else None)
        await hub.broadcast({"type": "lineup", "status": "done", "models": res.get("models"),
                             "latency_ms": res.get("latency_ms"), "view": view,
                             "players": [{"id": i, "number": i + 1, "box": p["box"],
                                          "photo": f"/api/photo/{i}.jpg?v={stamp}"} for i, p in enumerate(people)]})
        if not people:
            await hub.broadcast({"type": "error", "message": "I can't see anyone. Everyone in the picture, head to toe."})

    async def start_game(names: dict, skip: list) -> None:
        if st["task"] and not st["task"].done():
            await hub.broadcast({"type": "error", "message": "a game is already running"})
            return
        if err := ready():
            await hub.broadcast({"type": "error", "message": err})
            return
        if not st["lineup"]:
            await hub.broadcast({"type": "error", "message": "line up first: tap Line up with everyone in the picture"})
            return
        names = {int(k): str(v) for k, v in (names or {}).items()}
        keep = [i for i in range(len(st["lineup"])) if i not in {int(s) for s in (skip or [])}]
        if not keep:
            await hub.broadcast({"type": "error", "message": "nobody left on the team sheet"})
            return
        players, roster = players_from_lineup(st["lineup"], names, keep)
        game = Game(st["vision"], hub, players, roster, cfg.settings, dlog, recorder=rec, vision_lock=vision_lock)
        st["game"] = game
        st["games"].append(game)

        async def run() -> None:
            try:
                await game.run()
            except asyncio.CancelledError:
                rec.event("game_stopped", game_id=game.id, round=game.round)
                raise
            except Exception as e:  # noqa: BLE001
                rec.error("game", e, game_id=game.id, round=game.round, phase=game.phase)
                log.exception("game crashed", extra={"_simon_recorded": True})
                await hub.broadcast({"type": "error", "message": f"{type(e).__name__}: {e}"})
            await auto_upload("game end")

        st["task"] = asyncio.create_task(run())

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        if cfg.token and not secrets.compare_digest(ws.query_params.get("token") or "", cfg.token):
            await ws.close(code=4401)
            return
        await ws.accept()
        hub.sockets.add(ws)
        await ws.send_text(json.dumps({"type": "info", **info()}))
        if hub.last_state:
            await ws.send_text(json.dumps({"type": "state", **hub.last_state}, default=str))
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    if cfg.replay:
                        continue  # dry run: the replay is the camera
                    host = ws.client.host if ws.client else "?"
                    if len(msg["bytes"]) > max_upload or not frame_limiter.allow(host):
                        continue
                    try:
                        img = fit(Image.open(io.BytesIO(msg["bytes"])), cfg.side)
                    except Exception:  # noqa: BLE001 - a torn frame
                        continue
                    await frames.put(img)
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "lineup":
                        asyncio.create_task(do_lineup())
                    elif kind == "start":
                        await start_game(data.get("names") or {}, data.get("skip") or [])
                    elif kind == "stop" and st["game"]:
                        st["game"].stop()
                        if st["task"]:
                            st["task"].cancel()
                        await hub.broadcast({"type": "state", "phase": "stopped"})
                    elif kind == "info":
                        await ws.send_text(json.dumps({"type": "info", **info()}))
        except WebSocketDisconnect:
            pass
        finally:
            hub.sockets.discard(ws)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.state.st = st
    app.state.hub = hub
    app.state.frames = frames
    return app
