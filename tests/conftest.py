"""Shared fakes: synthetic scenes of 'people' (two-tone blocks) with a mocked segmenter that reads the
block positions from the frame, the numpy MockEmbedder (pixel layout), and a scripted MockDecider.
No torch, no weights, no camera."""

from __future__ import annotations

from typing import Sequence

import pytest
from PIL import Image, ImageDraw

from simon_sees.engine import Frame
from simon_sees.pipeline import LocalVision, Stack
from simon_sees.vision import Detection, MockEmbedder, MockSegmenter

W, H = 640, 480
SLOTS = [(60, 140), (260, 340), (460, 540)]  # x ranges of the left / centre / right standing places

# three distinguishable "people": two-tone blocks (top, bottom). Only the mock embedder looks at pixel values.
PEOPLE = {
    "a": ((200, 40, 40), (30, 30, 90)),
    "b": ((240, 240, 240), (60, 60, 60)),
    "c": ((40, 160, 60), (200, 170, 120)),
}


def scene(layout: Sequence[str | None], pose: dict[str, int] | None = None) -> Image.Image:
    """layout[i] = who stands in slot i (None = empty). The person boxes ride along in img.info."""
    img = Image.new("RGB", (W, H), (110, 160, 90))
    d = ImageDraw.Draw(img)
    boxes = []
    for slot, who in enumerate(layout):
        if who is None:
            continue
        x0, x1 = SLOTS[slot]
        dy = (pose or {}).get(who, 0)
        top, bottom = PEOPLE[who]
        d.rectangle((x0, 100 + dy, x1, 260 + dy), fill=top)
        d.rectangle((x0, 260 + dy, x1, 420), fill=bottom)
        boxes.append((who, (float(x0), float(100 + dy), float(x1), 420.0)))
    img.info["boxes"] = boxes
    return img


def scene_segmenter() -> MockSegmenter:
    return MockSegmenter(script=lambda i, img: [Detection(b, 0.95) for _, b in img.info.get("boxes", [])])


class ScriptedIO:
    """GameIO that plays a list of scenes (the last one repeats)."""

    def __init__(self, scenes: Sequence[Image.Image]):
        self.scenes = list(scenes)
        self.i = 0
        self.said: list[tuple[str, str]] = []
        self.states: list[dict] = []

    async def say(self, text, mood="info"):
        self.said.append((text, mood))

    async def frames(self, n, gap_s):
        out = []
        for _ in range(n):
            out.append(Frame(self.scenes[min(self.i, len(self.scenes) - 1)]))
            self.i += 1
        return out

    async def update(self, state):
        self.states.append(state)

    @property
    def texts(self) -> list[str]:
        return [t for t, _ in self.said]


def vision_for(decider) -> LocalVision:
    return LocalVision(Stack(scene_segmenter(), MockEmbedder(), decider))


@pytest.fixture(autouse=True)
def _sessions_in_tmp(tmp_path, monkeypatch):
    """The playtest recorder is on by default: keep every test's sessions out of the repo."""
    monkeypatch.setenv("SIMON_SESSIONS", str(tmp_path / "sessions"))
    monkeypatch.delenv("SIMON_UPLOAD", raising=False)
    monkeypatch.delenv("SIMON_RECORD", raising=False)


@pytest.fixture
def fast():
    return dict(reaction_s=0, frame_gap_s=0, freeze_grace_s=0, freeze_window_s=0, rest_s=0)
