"""FALLBACK motion check: plain pixel differencing, no AI.

Used only (a) when the decision model in use cannot take two images in one request, or
(b) as a logged second opinion next to the model's answer so the write-up can compare them.
It is deliberately dumb: wind in the trees or a passing dog will move pixels too, so it is
restricted to the player's own box (the union of where SAM/the detector found them in the BEFORE
and AFTER frames) when one is given.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image


def _gray(img: Image.Image, width: int = 96) -> np.ndarray:
    w, h = img.size
    small = img.convert("L").resize((width, max(1, round(h * width / w))), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.float32) / 255.0


def union_box(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def motion_score(before: Image.Image, after: Image.Image, box: Sequence[float] | None = None,
                 pixel_threshold: float = 0.12) -> float:
    """Fraction of pixels (inside `box`, in `before`'s pixel coordinates, if given) whose brightness
    changed by more than `pixel_threshold`, after normalising for a global exposure change. 0 = identical."""
    if after.size != before.size:
        after = after.resize(before.size, Image.Resampling.BILINEAR)
    if box is not None:
        w, h = before.size
        x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
        x1, y1 = min(w, max(x0 + 1, int(box[2]))), min(h, max(y0 + 1, int(box[3])))
        before, after = before.crop((x0, y0, x1, y1)), after.crop((x0, y0, x1, y1))
    a, b = _gray(before), _gray(after)
    b = b - float(np.median(b - a))  # the phone's auto-exposure shifts everything a little (median: robust to the mover)
    return float((np.abs(a - b) > pixel_threshold).mean())


def moved_probability(score: float, lo: float = 0.01, hi: float = 0.06) -> float:
    """Map a motion score to a pseudo-probability with a linear ramp. NOT calibrated: tune `lo`/`hi`
    on your own footage (see bench/)."""
    if score <= lo:
        return 0.0
    if score >= hi:
        return 1.0
    return (score - lo) / (hi - lo)
