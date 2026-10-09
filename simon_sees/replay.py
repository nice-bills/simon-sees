"""Dry-run input: replay a folder of images or a video file as if it were the phone's camera."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}


def fit(img: Image.Image, side: int = 384) -> Image.Image:
    """Downscale so the longer side is `side` px (the phone does the same before sending)."""
    img = img.convert("RGB")
    img.thumbnail((side, side), Image.Resampling.BILINEAR)
    return img


def iter_frames(source: str | Path, side: int = 384, fps: float = 3.0, loop: bool = True) -> Iterator[Image.Image]:
    src = Path(source)
    while True:
        if src.is_dir():
            files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXT)
            if not files:
                raise FileNotFoundError(f"no images in {src}")
            for p in files:
                yield fit(Image.open(p), side)
        elif src.suffix.lower() in VIDEO_EXT:
            yield from _video(src, side, fps)
        elif src.suffix.lower() in IMAGE_EXT:
            yield fit(Image.open(src), side)
        else:
            raise ValueError(f"don't know how to replay {src}")
        if not loop:
            return


def _video(path: Path, side: int, fps: float) -> Iterator[Image.Image]:
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError("replaying a video needs opencv: pip install opencv-python-headless "
                           "(or extract frames: ffmpeg -i in.mp4 -vf fps=3 frames/%04d.jpg)") from e
    cap = cv2.VideoCapture(str(path))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps))
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield fit(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), side)
        i += 1
    cap.release()
