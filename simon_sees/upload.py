"""Get recorded playtest sessions off the laptop.

    python -m simon_sees upload     # new/changed session files -> a PRIVATE Hugging Face dataset repo
    python -m simon_sees bundle     # every session -> one zip you can send by hand

`upload` uses your own HF_TOKEN (needs write access) through huggingface_hub. The repo is created if it
does not exist, always with private=True, and the upload refuses to go to a repo that is public. Each
session folder remembers what it already sent (`.uploaded.json`), so running it again only sends what is new.
With SIMON_UPLOAD=hf the server also runs it by itself at the end of every game and after an end-of-game note.
"""

from __future__ import annotations

import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .recorder import list_sessions, sessions_root

log = logging.getLogger(__name__)

DEFAULT_REPO = "nice-bill/simon-sees-playtests"
MARKER = ".uploaded.json"
_SKIP_NAMES = {MARKER, "session.json.tmp"}

DATASET_CARD = """---
license: other
pretty_name: Simon Sees playtests
tags: [simon-sees, playtest, private]
---
# Simon Sees playtests (private)

Recorded play sessions of [Simon Sees](https://github.com/nice-bills/simon-sees): one folder per session under
`sessions/` with `session.json`, `events.jsonl` (every round: command, timings, identity scores, d1 answers,
verdicts, errors), `feedback.jsonl` (the players' own "Wrong call" corrections and notes), `frames/`, `crops/`
and `server.log`.

These are photos of real people who agreed to play. Keep this repository private.
"""


def default_repo() -> str:
    return os.environ.get("SIMON_HF_REPO") or DEFAULT_REPO


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def session_files(folder: Path) -> dict[str, list[float]]:
    """relative path -> [size, mtime] for every file worth sending."""
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.name not in _SKIP_NAMES and "__pycache__" not in p.parts:
            st = p.stat()
            out[p.relative_to(folder).as_posix()] = [st.st_size, round(st.st_mtime, 3)]
    return out


def changed_files(folder: Path, force: bool = False) -> list[str]:
    now = session_files(folder)
    if force or not (folder / MARKER).exists():
        return list(now)
    try:
        sent = json.loads((folder / MARKER).read_text()).get("files", {})
    except (OSError, json.JSONDecodeError):
        return list(now)
    return [rel for rel, sig in now.items() if sent.get(rel) != sig]


def _mark(folder: Path, repo_id: str, files: Iterable[str]) -> None:
    prev: dict[str, Any] = {}
    if (folder / MARKER).exists():
        try:
            prev = json.loads((folder / MARKER).read_text()).get("files", {})
        except (OSError, json.JSONDecodeError):
            prev = {}
    now = session_files(folder)
    prev.update({rel: now[rel] for rel in files if rel in now})
    (folder / MARKER).write_text(json.dumps({"repo": repo_id, "uploaded": round(time.time(), 3), "files": prev}, indent=1))


def upload_sessions(root: str | Path | None = None, repo_id: str | None = None, token: str | None = None,
                    sessions: Iterable[str | Path] | None = None, force: bool = False, allow_public: bool = False,
                    api: Any = None) -> dict:
    """Push new or changed session files to a private HF dataset repo. Returns a summary dict.
    `sessions` limits it to those folders (paths, or names under `root`)."""
    repo_id = repo_id or default_repo()
    token = token or hf_token()
    root = Path(root) if root else sessions_root()
    folders = [Path(s) if Path(s).is_dir() else root / str(s) for s in sessions] if sessions else list_sessions(root)
    folders = [f for f in folders if (f / "session.json").exists()]
    todo = [(f, changed_files(f, force)) for f in folders]
    todo = [(f, files) for f, files in todo if files]
    summary: dict[str, Any] = {"repo": repo_id, "url": f"https://huggingface.co/datasets/{repo_id}",
                               "uploaded": [], "up_to_date": [f.name for f in folders if f not in {t[0] for t in todo}]}
    if not todo:
        return summary

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise RuntimeError("upload needs huggingface_hub: pip install huggingface_hub") from e
        api = HfApi(token=token)
    from huggingface_hub import CommitOperationAdd

    if not token:
        log.info("HF_TOKEN is not set; trying the login saved by `hf auth login`")
    try:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        info = api.repo_info(repo_id, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"could not create or open the dataset {repo_id}: {type(e).__name__}: {str(e)[:300]}\n"
            "Check that HF_TOKEN is set and has WRITE access, and that the repo name starts with your own "
            "Hugging Face username (or pass --repo <you>/simon-sees-playtests).") from e
    if getattr(info, "private", True) is False and not allow_public:
        raise RuntimeError(f"{repo_id} is PUBLIC. These sessions contain photos of people, so nothing was uploaded. "
                           "Make the repo private on huggingface.co (Settings) or use another --repo.")
    card = []
    try:
        if not api.file_exists(repo_id, "README.md", repo_type="dataset"):
            card = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=DATASET_CARD.encode())]
    except Exception:  # noqa: BLE001 - the card is a nicety
        card = []
    for folder, files in todo:
        ops = card + [CommitOperationAdd(path_in_repo=f"sessions/{folder.name}/{rel}", path_or_fileobj=str(folder / rel))
                      for rel in files]
        card = []
        api.create_commit(repo_id, repo_type="dataset", operations=ops,
                          commit_message=f"playtest {folder.name}: {len(files)} file(s)")
        _mark(folder, repo_id, files)
        summary["uploaded"].append({"session": folder.name, "files": len(files)})
        log.info("uploaded %d file(s) of %s to %s", len(files), folder.name, repo_id)
    return summary


def make_bundle(root: str | Path | None = None, out: str | Path | None = None,
                sessions: Iterable[str | Path] | None = None) -> Path:
    """Zip every session (or the ones named) into one file to send by hand."""
    root = Path(root) if root else sessions_root()
    folders = [Path(s) if Path(s).is_dir() else root / str(s) for s in sessions] if sessions else list_sessions(root)
    folders = [f for f in folders if (f / "session.json").exists()]
    if not folders:
        raise RuntimeError(f"no recorded sessions in {root}/ yet: play a game first")
    out = Path(out) if out else Path(f"simon-sees-playtests-{time.strftime('%Y%m%d-%H%M%S')}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in folders:
            for rel in session_files(f):
                z.write(f / rel, f"sessions/{f.name}/{rel}")
    return out
