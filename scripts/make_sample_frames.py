"""Draw a few cartoon 'park' frames (two stick figures) for a model-free loop check.

Legacy: the photo frames in samples/people are what the replay demo and the SAM/DINO bench use. These
drawings only show the game loop with `--segmenter mock --embedder mock --decider mock`; SAM 3 is not
expected to find people in them. Players are identified by their line-up photo, never by a description.

    python scripts/make_sample_frames.py samples/frames
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

W, H = 512, 384


def person(d: ImageDraw.ImageDraw, x: int, body: str, head_band: str | None, pose: str) -> None:
    ground = 330
    skin, pants = (230, 190, 160), (50, 60, 90)
    head_y = 150 if pose != "squat" else 200
    # legs
    hip = (x, head_y + 110)
    if pose == "one_foot":
        d.line([hip, (x - 8, ground)], fill=pants, width=12)
        d.line([hip, (x + 30, ground - 50)], fill=pants, width=12)
    elif pose == "squat":
        d.line([hip, (x - 30, ground - 30), (x - 20, ground)], fill=pants, width=12)
        d.line([hip, (x + 30, ground - 30), (x + 20, ground)], fill=pants, width=12)
    else:
        d.line([hip, (x - 15, ground)], fill=pants, width=12)
        d.line([hip, (x + 15, ground)], fill=pants, width=12)
    # torso
    d.rounded_rectangle([x - 26, head_y + 25, x + 26, head_y + 115], radius=10, fill=body)
    sh_l, sh_r = (x - 24, head_y + 35), (x + 24, head_y + 35)
    arms = {
        "arms_up": [(x - 30, head_y - 40), (x + 30, head_y - 40)],
        "hands_head": [(x - 8, head_y - 18), (x + 8, head_y - 18)],
        "airplane": [(x - 90, head_y + 35), (x + 90, head_y + 35)],
    }.get(pose, [(x - 34, head_y + 100), (x + 34, head_y + 100)])
    if pose == "hands_head":
        d.line([sh_l, (x - 45, head_y), arms[0]], fill=skin, width=10)
        d.line([sh_r, (x + 45, head_y), arms[1]], fill=skin, width=10)
    else:
        d.line([sh_l, arms[0]], fill=skin, width=10)
        d.line([sh_r, arms[1]], fill=skin, width=10)
    # head (+ an optional band, so the two figures are not identical)
    d.ellipse([x - 20, head_y - 20, x + 20, head_y + 22], fill=skin)
    if head_band:
        d.chord([x - 22, head_y - 26, x + 22, head_y + 10], 180, 360, fill=head_band)
        d.rectangle([x, head_y - 10, x + 32, head_y - 4], fill=head_band)


def scene(poses: tuple[str, str]) -> Image.Image:
    img = Image.new("RGB", (W, H), (150, 200, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 250, W, H], fill=(80, 150, 60))  # grass
    d.rectangle([420, 120, 445, 260], fill=(100, 70, 40))  # tree trunk
    d.ellipse([370, 40, 500, 160], fill=(40, 120, 50))
    person(d, 150, (200, 40, 40), None, poses[0])
    person(d, 320, (235, 235, 235), (30, 70, 200), poses[1])
    return img


FRAMES = [
    ("stand", "stand"), ("stand", "stand"), ("one_foot", "stand"), ("one_foot", "one_foot"),
    ("hands_head", "hands_head"), ("arms_up", "stand"), ("airplane", "airplane"), ("squat", "stand"),
    ("stand", "stand"), ("stand", "arms_up"),
]

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/frames")
    out.mkdir(parents=True, exist_ok=True)
    for i, poses in enumerate(FRAMES):
        scene(poses).save(out / f"{i:03d}_{poses[0]}_{poses[1]}.jpg", quality=85)
    print(f"wrote {len(FRAMES)} frames to {out}")
