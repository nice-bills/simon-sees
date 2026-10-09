"""Playtest recorder: everything needed to see afterwards what the referee got right and wrong.

On by default; `SIMON_RECORD=0` turns it off. One folder per server run (or dry run):

    sessions/<YYYYmmdd-HHMMSS>-<id>/
      session.json     models/backends actually used, device, library versions, settings, seed, games played
      events.jsonl     one line per event: lineup, round (command, trick/freeze, per-stage timings, identity
                       match scores, d1 P(did it), verdict per player), say (TTS time), error (+ traceback)
      feedback.jsonl   the players' own corrections ("Wrong call") and end-of-game notes, linked by round_id
      frames/          the judged frame(s) of every round, JPEG, long side <= 640 px
      crops/           each player's crop per judged frame (what the referee looked at)
      server.log       the server's log while this session was open

Frames are photos of the people playing: they stay on the laptop unless the user runs
`python -m simon_sees upload` (a PRIVATE Hugging Face dataset) or `bundle` (a zip they send themselves).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import secrets
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

FRAME_SIDE = 640
CROP_SIDE = 384
_SECRET_KEYS = ("token", "secret", "key", "password", "dsn")


def enabled_by_env() -> bool:
    return os.environ.get("SIMON_RECORD", "1").strip().lower() not in ("0", "false", "no", "off", "")


def sessions_root() -> Path:
    return Path(os.environ.get("SIMON_SESSIONS", "sessions"))


def versions() -> dict:
    from importlib import metadata

    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("simon-sees", "torch", "transformers", "huggingface_hub", "fastapi", "pillow", "numpy",
                "kokoro", "faster-whisper"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    try:
        from . import __version__

        out.setdefault("simon-sees", __version__)
    except Exception:  # noqa: BLE001
        pass
    return out


def detect_device(vision: Any = None, preferred: str = "auto") -> dict:
    """What actually ran the models: the decider's / segmenter's device when they have one."""
    info: dict[str, Any] = {"requested": preferred}
    stack = getattr(vision, "stack", None)
    for part in ("decider", "segmenter", "embedder"):
        obj = getattr(stack, part, None) if stack is not None else None
        dev = getattr(obj, "device", None)
        if dev is not None:
            info[part] = str(dev)
            dt = getattr(obj, "dtype", None)
            if dt is not None:
                info[f"{part}_dtype"] = str(dt).replace("torch.", "")
    if getattr(vision, "backend", None) == "modal":
        info["remote"] = "modal GPU"
    torch = sys.modules.get("torch")  # never import torch just to describe it
    if torch is not None:
        try:
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
    info["cpus"] = os.cpu_count()
    return info


def _scrub(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: ("***" if any(s in k.lower() for s in _SECRET_KEYS) and v else _scrub(v)) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_scrub(x) for x in d]
    return d


def _jpeg(img, path: Path, side: int, quality: int = 85) -> str:
    im = img.convert("RGB")
    if max(im.size) > side:
        im = im.copy()
        im.thumbnail((side, side))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=quality)
    return str(path)


class _ErrorHandler(logging.Handler):
    """Every ERROR log record (log.exception included) also becomes an `error` event."""

    def __init__(self, rec: "Recorder"):
        super().__init__(level=logging.ERROR)
        self.rec = rec

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "_simon_recorded", False):
            return
        tb = "".join(traceback.format_exception(*record.exc_info)) if record.exc_info else None
        try:
            self.rec.event("error", where=record.name, message=record.getMessage(), traceback=tb, via="log")
        except Exception:  # noqa: BLE001 - logging must never raise
            pass


class Recorder:
    """Writes one session folder. All methods are cheap and never raise into the game."""

    enabled = True

    def __init__(self, root: str | Path | None = None, meta: dict | None = None, capture_log: bool = True):
        root = Path(root) if root else sessions_root()
        self.id = secrets.token_hex(3)
        self.started = time.time()
        self.dir = root / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.started))}-{self.id}"
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "crops").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.meta: dict[str, Any] = {"session_id": self.id, "started": round(self.started, 3),
                                     "started_iso": datetime.fromtimestamp(self.started).astimezone().isoformat(timespec="seconds"),
                                     "versions": versions(), "games": []}
        self.meta.update(_scrub(meta or {}))
        self._write_meta()
        self._handlers: list[logging.Handler] = []
        if capture_log:
            fh = logging.FileHandler(self.dir / "server.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
            fh.setLevel(logging.INFO)
            eh = _ErrorHandler(self)
            root_logger = logging.getLogger()
            for h in (fh, eh):
                root_logger.addHandler(h)
                self._handlers.append(h)
            if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET:
                root_logger.setLevel(logging.INFO)
        log.info("recording this session to %s (SIMON_RECORD=0 turns it off)", self.dir)

    # ---- files
    def _write_meta(self) -> None:
        tmp = self.dir / "session.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.dir / "session.json")

    def update_meta(self, **fields: Any) -> None:
        with self._lock:
            self.meta.update(_scrub(fields))
            self.meta["updated"] = round(time.time(), 3)
            self._write_meta()

    def _append(self, name: str, row: dict) -> dict:
        row = {"ts": round(time.time(), 3), **row}
        with self._lock, (self.dir / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return row

    def event(self, name: str, /, **fields: Any) -> dict:
        return self._append("events.jsonl", {"event": name, **fields})

    def feedback(self, **fields: Any) -> dict:
        return self._append("feedback.jsonl", fields)

    def error(self, where: str, exc: BaseException | None = None, **fields: Any) -> dict:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else traceback.format_exc()
        return self.event("error", where=where, message=f"{type(exc).__name__}: {exc}" if exc else None,
                          traceback=tb, **fields)

    def frame(self, img, name: str) -> str | None:
        try:
            return str(Path(_jpeg(img, self.dir / "frames" / name, FRAME_SIDE)).relative_to(self.dir))
        except Exception as e:  # noqa: BLE001
            log.warning("could not save frame %s: %s", name, e)
            return None

    def crop(self, img, name: str) -> str | None:
        try:
            return str(Path(_jpeg(img, self.dir / "crops" / name, CROP_SIDE)).relative_to(self.dir))
        except Exception as e:  # noqa: BLE001
            log.warning("could not save crop %s: %s", name, e)
            return None

    def game_started(self, game_id: str, **fields: Any) -> None:
        with self._lock:
            self.meta["games"].append({"game_id": game_id, "started": round(time.time(), 3), **_scrub(fields)})
            self._write_meta()

    def game_ended(self, game_id: str, **fields: Any) -> None:
        with self._lock:
            for g in self.meta["games"]:
                if g.get("game_id") == game_id:
                    g.update({"ended": round(time.time(), 3), **_scrub(fields)})
            self._write_meta()

    def close(self) -> None:
        self.update_meta(closed=round(time.time(), 3))
        root_logger = logging.getLogger()
        for h in self._handlers:
            root_logger.removeHandler(h)
            h.close()
        self._handlers = []


class NullRecorder:
    """SIMON_RECORD=0: same interface, does nothing."""

    enabled = False
    id = None
    dir = None

    def update_meta(self, **_: Any) -> None: ...
    def event(self, name: str, /, **_: Any) -> dict: return {}
    def feedback(self, **_: Any) -> dict: return {}
    def error(self, where: str, exc: BaseException | None = None, **_: Any) -> dict: return {}
    def frame(self, img, name: str) -> None: return None
    def crop(self, img, name: str) -> None: return None
    def game_started(self, game_id: str, **_: Any) -> None: ...
    def game_ended(self, game_id: str, **_: Any) -> None: ...
    def close(self) -> None: ...


def make_recorder(enabled: bool | None = None, root: str | Path | None = None, meta: dict | None = None,
                  capture_log: bool = True) -> Recorder | NullRecorder:
    if enabled is None:
        enabled = enabled_by_env()
    if not enabled:
        return NullRecorder()
    try:
        return Recorder(root, meta, capture_log)
    except Exception as e:  # noqa: BLE001 - a read-only disk must not stop the game
        log.warning("session recorder off: %s", e)
        return NullRecorder()


def list_sessions(root: str | Path | None = None) -> list[Path]:
    root = Path(root) if root else sessions_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "session.json").exists())


def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
