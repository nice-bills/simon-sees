"""Command line.

    python -m simon_sees serve                       # phone page + game: SAM 3 -> DINOv3 -> d1-3B, all local
    python -m simon_sees serve --backend modal       # same, but the vision stack + d1 run on a Modal GPU
    python -m simon_sees serve --replay samples/people --segmenter fallback --embedder dinov2   # desk demo, ungated models
    python -m simon_sees dryrun --replay samples/people --decider mock --segmenter mock --embedder mock --fast
    python -m simon_sees dryrun --replay samples/people --names Kofi Ama ""     # line-up from the first frame
    python -m simon_sees upload                      # recorded playtests -> your PRIVATE HF dataset (needs HF_TOKEN)
    python -m simon_sees bundle                      # recorded playtests -> one zip to send by hand

Every serve/dryrun records a playtest session in sessions/ (SIMON_RECORD=0 turns it off; see recorder.py).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

from .engine import Frame, Game, Settings, lineup, players_from_lineup


def _settings(a: argparse.Namespace) -> Settings:
    s = Settings(max_rounds=a.rounds, seed=a.seed, trick_rate=a.trick_rate, outdoors=not a.indoors,
                 freeze_strategy=a.freeze, save_frames=a.save_frames, pose_frames=a.pose_frames)
    if getattr(a, "fast", False):
        s.reaction_s = s.frame_gap_s = s.freeze_grace_s = s.freeze_window_s = s.rest_s = 0.0
    return s


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["local", "modal"],
                   default=os.environ.get("SIMON_BACKEND") or ("modal" if os.environ.get("SIMON_VISION_URL") else "local"),
                   help="local: SAM/DINO/d1 in this process; modal: one HTTPS call per frame batch to SIMON_VISION_URL (deploy/modal_app.py). Defaults to modal when SIMON_VISION_URL is set")
    p.add_argument("--segmenter", default=os.environ.get("SEGMENTER", "auto"),
                   choices=["auto", "sam3", "sam3.1", "fallback", "mock"],
                   help="auto = SAM 3 (facebook/sam3, gated), else the FALLBACK RT-DETR person detector")
    p.add_argument("--embedder", default=os.environ.get("EMBEDDER", "auto"), choices=["auto", "dinov3", "dinov2", "mock"],
                   help="auto = DINOv3 (gated), else the FALLBACK facebook/dinov2-small")
    p.add_argument("--embedder-model", default=os.environ.get("EMBEDDER_MODEL"),
                   help="override the HF id, e.g. facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--match-threshold", type=float, default=float(os.environ.get("SIMON_MATCH_THRESHOLD", "0.45")),
                   help="minimum cosine similarity to recognise an enrolled player")
    p.add_argument("--side", type=int, default=int(os.environ.get("SIMON_SIDE", "640")),
                   help="long side of each camera frame, px")
    p.add_argument("--decider", default=os.environ.get("SIMON_DECIDER", "auto"),
                   choices=["auto", "tinker", "local-qwen", "d1", "mock"],
                   help="auto = the fine-tuned Qwen3.5-4B referee (Tinker if SIMON_TINKER_MODEL + TINKER_API_KEY are set, "
                        "else local weights in SIMON_REFEREE_WEIGHTS), else d1")
    p.add_argument("--model", default=os.environ.get("SIMON_MODEL", "LiquidAI/d1-3B"),
                   help="LiquidAI/d1-3B (default) or LiquidAI/d1-omni-600M")
    p.add_argument("--device", default=os.environ.get("SIMON_DEVICE", "auto"), help="auto | cuda | mps | cpu")
    p.add_argument("--dtype", default=os.environ.get("SIMON_DTYPE", "auto"), help="auto | fp32 | bf16 | fp16")
    p.add_argument("--compile", action="store_true", help="CUDA graphs (NVIDIA only)")
    p.add_argument("--replay", default=os.environ.get("SIMON_REPLAY"), help="folder of images or a video used as the camera")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--trick-rate", type=float, default=0.25)
    p.add_argument("--indoors", action="store_true", help="skip commands that need grass or trees")
    p.add_argument("--freeze", default="pair", choices=["pair", "pixel"], help="pair = d1 sees before+after (default); pixel = fallback")
    p.add_argument("--pose-frames", type=int, default=2, help="frames judged per pose round")
    p.add_argument("--log", default=os.environ.get("SIMON_LOG", "logs/decisions.jsonl"))
    p.add_argument("--save-frames", default=os.environ.get("SIMON_SAVE_FRAMES"),
                   help="keep the judged frames in this folder (local only) to label for bench/")


def _config(a: argparse.Namespace, **kw):
    from .server import Config

    return Config(backend=a.backend, decider=a.decider, model_id=a.model, segmenter=a.segmenter, embedder=a.embedder,
                  embedder_model=a.embedder_model, match_threshold=a.match_threshold, device=a.device, dtype=a.dtype,
                  compile=a.compile, replay=a.replay, side=a.side, log_path=a.log, settings=_settings(a), **kw)


def serve(a: argparse.Namespace) -> int:
    import uvicorn

    from .server import Config, create_app

    cfg = _config(a, tts=a.tts, stt=not a.no_stt, speech_scale=a.speech_scale)
    app = create_app(cfg)
    scheme = "https" if a.ssl_certfile else "http"
    print(f"\n  Simon Sees on {scheme}://{a.host}:{a.port}  (phone: expose it with cloudflared, see docs/phone-setup.md)\n")
    uvicorn.run(app, host=a.host, port=a.port, ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile,
                log_level="info", ws_max_size=4 << 20)
    return 0


class ConsoleIO:
    """GameIO for a terminal dry run: Simon prints, the camera is a replayed folder/video."""

    def __init__(self, source: str, fast: bool, side: int = 640):
        from .replay import iter_frames

        self.it = iter_frames(source, side)
        self.fast = fast
        self.t0 = time.time()

    async def say(self, text: str, mood: str = "info") -> None:
        print(f"[{time.time() - self.t0:6.1f}s] SIMON ({mood}): {text}", flush=True)
        if not self.fast:
            from .voice import estimate_speech_s

            await asyncio.sleep(estimate_speech_s(text))

    async def frames(self, n: int, gap_s: float) -> list[Frame]:
        out = []
        for i in range(n):
            if i and gap_s:
                await asyncio.sleep(gap_s)
            out.append(Frame(next(self.it)))
        return out

    async def update(self, state: dict) -> None:
        pass


def dryrun(a: argparse.Namespace) -> int:
    from . import tracing
    from .decision_log import DecisionLog
    from .server import build_vision

    if not a.replay:
        print("dryrun needs --replay <folder of images or video>", file=sys.stderr)
        return 2
    from dataclasses import asdict

    from .recorder import detect_device, make_recorder

    tracing.init()
    rec = make_recorder(meta={"mode": "dryrun", "replay": a.replay, "seed": a.seed, "settings": asdict(_settings(a))})
    try:
        vision = build_vision(_config(a))
    except Exception as e:
        rec.error("load_models", e)
        rec.close()
        raise
    rec.update_meta(models=vision.models, backend=vision.backend, device=detect_device(vision, a.device))
    print(f"models: {vision.models} | backend: {vision.backend}", flush=True)
    io = ConsoleIO(a.replay, a.fast, a.side)

    async def go():
        res = await lineup(vision, io)  # the first replayed frame is the line-up
        people = res.get("people", [])
        print(f"line-up: {len(people)} people found, left to right "
              f"({res.get('latency_ms', {}).get('total', '?')} ms)", flush=True)
        if not people:
            raise SystemExit("nobody in the line-up frame")
        players, roster = players_from_lineup(people, list(a.names or []))
        game = Game(vision, io, players, roster, _settings(a), DecisionLog(a.log), recorder=rec)
        return game, await game.run()

    try:
        game, winners = asyncio.run(go())
    except Exception as e:
        rec.error("dryrun", e)
        raise
    finally:
        rec.close()
    lat = sorted(game.latencies)
    print(f"\nwinners: {[p.display_name for p in winners]} | rounds: {game.round} | decision batches: {len(lat)}"
          + (f" | median d1 batch: {lat[len(lat) // 2]:.1f} ms" if lat else "") + f" | log: {a.log}"
          + (f" | session: {rec.dir}" if rec.enabled else ""))
    return 0


def upload(a: argparse.Namespace) -> int:
    from .upload import upload_sessions

    try:
        res = upload_sessions(a.sessions, a.repo, sessions=a.only or None, force=a.all, allow_public=a.public_ok)
    except RuntimeError as e:
        print(f"upload failed: {e}", file=sys.stderr)
        print("Fallback: `python -m simon_sees bundle` makes a zip you can send instead.", file=sys.stderr)
        return 1
    for u in res["uploaded"]:
        print(f"  uploaded {u['session']}  ({u['files']} files)")
    if not res["uploaded"]:
        print("  nothing new to upload" + ("" if res["up_to_date"] else f" (no sessions in {a.sessions or 'sessions/'})"))
    print(f"\n  private dataset: {res['url']}")
    return 0


def bundle(a: argparse.Namespace) -> int:
    from .upload import make_bundle

    try:
        out = make_bundle(a.sessions, a.out, a.only or None)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(f"\n  {out.resolve()}  ({out.stat().st_size / 1e6:.1f} MB)\n  Send this zip in the chat.")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=os.environ.get("SIMON_LOGLEVEL", "INFO"), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="simon_sees", description="Simon Says, refereed by SAM 3 + DINOv3 + an open decision model.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the server for the phone page")
    _common(s)
    s.add_argument("--host", default=os.environ.get("SIMON_HOST", "0.0.0.0"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    s.add_argument("--tts", default=os.environ.get("SIMON_TTS", "auto"), choices=["auto", "kokoro", "piper", "browser"])
    s.add_argument("--no-stt", action="store_true")
    s.add_argument("--fast", action="store_true", help="no reaction/rest pauses (tests and demos)")
    s.add_argument("--speech-scale", type=float, default=float(os.environ.get("SIMON_SPEECH_SCALE", "1.0")),
                   help="multiply the pause after each spoken line (e.g. 0.1 in tests)")
    s.add_argument("--ssl-certfile")
    s.add_argument("--ssl-keyfile")
    s.set_defaults(fn=serve)
    d = sub.add_parser("dryrun", help="play a game in the terminal against replayed frames")
    _common(d)
    d.add_argument("--names", nargs="*", default=[],
                   help="names for the people in the line-up frame, left to right (\"\" keeps Player N)")
    d.add_argument("--fast", action="store_true", help="no waiting between calls")
    d.set_defaults(fn=dryrun)
    u = sub.add_parser("upload", help="push recorded playtest sessions to a PRIVATE Hugging Face dataset")
    u.add_argument("--repo", default=None, help="dataset repo id (default SIMON_HF_REPO or nice-bill/simon-sees-playtests)")
    u.add_argument("--sessions", default=None, help="sessions folder (default SIMON_SESSIONS or ./sessions)")
    u.add_argument("--all", action="store_true", help="re-send every file, not only new ones")
    u.add_argument("--public-ok", action="store_true", help="allow uploading to a repo that is public (not recommended)")
    u.add_argument("only", nargs="*", help="only these session folders")
    u.set_defaults(fn=upload)
    b = sub.add_parser("bundle", help="zip recorded playtest sessions to send by hand")
    b.add_argument("--out", default=None, help="zip path (default simon-sees-playtests-<time>.zip)")
    b.add_argument("--sessions", default=None, help="sessions folder (default SIMON_SESSIONS or ./sessions)")
    b.add_argument("only", nargs="*", help="only these session folders")
    b.set_defaults(fn=bundle)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
