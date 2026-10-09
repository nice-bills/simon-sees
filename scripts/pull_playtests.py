"""Download the playtest sessions (agent / maintainer side).

    python scripts/pull_playtests.py                                  # HF dataset -> ./playtests/sessions/...
    python scripts/pull_playtests.py --repo someone/simon-sees-playtests --out playtests
    python scripts/pull_playtests.py --zip simon-sees-playtests-2026....zip   # a bundle sent by hand

Uses HF_TOKEN from the environment (read access to the private dataset is enough). Then:
    python scripts/playtest_report.py playtests --out playtests/report.md
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simon_sees.upload import default_repo  # noqa: E402


def pull_hf(repo_id: str, out: Path, token: str | None) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(out), token=token,
                             allow_patterns=["sessions/**", "README.md"])
    return Path(path)


def unzip(bundle: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as z:
        for name in z.namelist():
            target = (out / name).resolve()
            if not str(target).startswith(str(out.resolve())):  # no ../ escapes
                raise SystemExit(f"refusing to extract {name!r}")
        z.extractall(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="download recorded Simon Sees playtests")
    ap.add_argument("--repo", default=default_repo(), help="HF dataset repo id")
    ap.add_argument("--out", default="playtests")
    ap.add_argument("--zip", dest="bundle", help="unpack a bundle zip instead of downloading")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.bundle:
        unzip(Path(a.bundle), out)
    else:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            print("HF_TOKEN is not set (needed for a private dataset)", file=sys.stderr)
            return 1
        try:
            pull_hf(a.repo, out, token)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ in ("RepositoryNotFoundError", "GatedRepoError"):
                print(f"{a.repo}: not found or no access. Nothing uploaded yet, or this HF_TOKEN cannot read it "
                      "(the repo is private to whoever created it).", file=sys.stderr)
                return 1
            raise
    sessions = sorted(p.name for p in (out / "sessions").glob("*") if (p / "session.json").exists())
    print(f"{len(sessions)} session(s) in {out}/sessions" + "".join(f"\n  {s}" for s in sessions))
    print(f"\nnext: python scripts/playtest_report.py {out} --out {out}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
