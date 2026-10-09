"""Download the fine-tuned referee's LoRA from Tinker and make it servable without Tinker.

    # 1. the raw Tinker checkpoint (LoRA matrices + adapter_config.json), ~tens of MB
    python scripts/export_referee.py download tinker://<run>/sampler_weights/<step> --out models/referee-lora/tinker

    # 2. standard PEFT format (loads with peft / vLLM --lora-modules). Needs tinker-cookbook and the base
    #    model's safetensors (downloads Qwen/Qwen3.5-4B, ~9 GB, once)
    python scripts/export_referee.py peft models/referee-lora/tinker --out models/referee-lora/peft

    # 3. optional: merge into a full Qwen3.5-4B checkpoint (bf16, ~9 GB)
    python scripts/export_referee.py merge models/referee-lora/tinker --out models/referee-merged

Then play with the local weights instead of Tinker:
    SIMON_REFEREE_WEIGHTS=models/referee-lora/peft python -m simon_sees serve --decider local-qwen
models/ is git-ignored.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "Qwen/Qwen3.5-4B"


def ranged_get(url: str, chunk: int = 64 << 20) -> bytes:
    """GET in 64 MiB Range requests (some proxies cap a single response at 100 MiB; the archive is ~290 MB)."""
    buf = bytearray()
    while True:
        req = urllib.request.Request(url, headers={"Range": f"bytes={len(buf)}-{len(buf) + chunk - 1}"})
        with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310 - signed Tinker URL
            part = r.read()
            total = r.headers.get("Content-Range", "").rpartition("/")[2]
            status = r.status
        buf += part
        if status == 200 or not part or (total.isdigit() and len(buf) >= int(total)):
            return bytes(buf)


def download(tinker_path: str, out: Path) -> Path:
    import os

    import tinker

    key = os.environ.get("TINKER_API_KEY") or os.environ.get("SIMON_TINKER_API_KEY")
    rc = tinker.ServiceClient(**({"api_key": key} if key else {})).create_rest_client()
    url = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result().url
    out.mkdir(parents=True, exist_ok=True)
    data = ranged_get(url)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if m.issym() or m.islnk() or m.name.startswith("/") or ".." in Path(m.name).parts:
                raise RuntimeError(f"unsafe archive entry {m.name}")
        tf.extractall(out)  # noqa: S202 - validated above
    (out / "TINKER_PATH").write_text(tinker_path + "\n")
    print("downloaded", tinker_path, "->", out, sorted(p.name for p in out.iterdir()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("tinker_path")
    d.add_argument("--out", type=Path, default=Path("models/referee-lora/tinker"))
    p = sub.add_parser("peft")
    p.add_argument("adapter", type=Path)
    p.add_argument("--out", type=Path, default=Path("models/referee-lora/peft"))
    m = sub.add_parser("merge")
    m.add_argument("adapter", type=Path)
    m.add_argument("--out", type=Path, default=Path("models/referee-merged"))
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download(a.tinker_path, a.out)
        return
    try:
        from tinker_cookbook import weights
    except ImportError:
        sys.exit("needs tinker-cookbook: pip install git+https://github.com/thinking-machines-lab/tinker-cookbook")
    if a.cmd == "peft":
        weights.build_lora_adapter(base_model=BASE, adapter_path=str(a.adapter), output_path=str(a.out))
    else:
        weights.build_hf_model(base_model=BASE, adapter_path=str(a.adapter), output_path=str(a.out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
