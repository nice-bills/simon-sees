"""The real-game test slice: the players in samples/people, found and cropped by the game's own pipeline.

    python scripts/build_game_slice.py [--segmenter sam3]

SAM 3 (or the FALLBACK detector) finds everyone in each samples/people frame; each player is cropped with
simon_sees.vision.crop_person exactly as a pose round does (mask isolation, neighbours painted out). The pose
each player holds is read from the frame's file name, left to right (e.g. 005_airplane_squat_airplane.jpg),
and every crop is asked every referee command: the matching command is a yes, all others a no.
Writes data/referee_game/*.jpg (small, committed) and data/referee_game_slice.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simon_sees.pose_rules import RULE_COMMANDS  # noqa: E402
from simon_sees.vision import VisionConfig, crop_person, make_segmenter  # noqa: E402

POSES = {  # file name word -> command id (None = standing / no pose)
    "onefoot": "one_foot", "hands-head": "hands_head", "arms-up": "arms_up", "airplane": "airplane",
    "squat": "squat", "stand": None, "lineup": None,
}


# Human check of every crop (done by looking at data/referee_game/*.jpg): where the file name's pose list is
# incomplete, the eye wins. None = ambiguous, the row is dropped.
HUMAN = {
    ("005_airplane_squat_airplane", 2): {"airplane": True},  # squatting AND arms out like wings
    ("006_centre-player-left-frame", 1): {"one_foot": True},  # left foot clearly lifted
    ("006_centre-player-left-frame", 2): {"one_foot": None},  # mid-step, can't tell
}


def poses_for(name: str, n: int) -> list[str | None]:
    words = name.split("_")[1:]
    if words and words[0] in ("lineup", "swapped-places", "centre-player-left-frame"):
        return [None] * n
    out = [POSES[w] for w in words if w in POSES]
    if len(out) != n:
        raise ValueError(f"{name}: {n} people found but {len(out)} poses in the name")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=ROOT / "samples" / "people")
    ap.add_argument("--segmenter", default="sam3")
    ap.add_argument("--side", type=int, default=640)
    args = ap.parse_args(argv)
    seg = None  # loaded only when a frame is not cached yet
    out_dir = ROOT / "data" / "referee_game"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    done_dir = out_dir / ".rows"  # per-frame results, so a killed run resumes where it stopped
    done_dir.mkdir(exist_ok=True)
    for f in sorted(args.frames.glob("*.jpg")):
        cached = done_dir / f"{f.stem}.json"
        if cached.exists():
            for r in json.loads(cached.read_text()):
                pid = int(r["id"].rsplit("-p", 1)[1].split("-")[0])
                label = HUMAN.get((f.stem, pid), {}).get(r["command"], r["pose"] == r["command"])
                if label is not None:
                    rows.append({**r, "label": label})
            continue
        frame_rows = []
        if seg is None:
            seg = make_segmenter(VisionConfig(segmenter=args.segmenter, device="cpu"))
        img = Image.open(f).convert("RGB")
        k = args.side / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)))
        dets = sorted(seg.segment([img])[0], key=lambda d: d.cx)
        poses = poses_for(f.stem, len(dets))
        for i, (d, pose) in enumerate(zip(dets, poses)):
            crop = crop_person(img, d, others=dets)
            rel = f"data/referee_game/{f.stem}_p{i + 1}.jpg"
            crop.save(ROOT / rel, quality=90)
            for cmd in RULE_COMMANDS:
                label = HUMAN.get((f.stem, i + 1), {}).get(cmd, pose == cmd)
                if label is None:
                    continue
                frame_rows.append({"id": f"game-{f.stem}-p{i + 1}-{cmd}", "split": "game", "command": cmd,
                             "label": label, "pose": pose or "standing", "crop": rel, "frame": f.name,
                             "segmenter": seg.name, "has_mask": d.mask is not None})
        cached.write_text(json.dumps(frame_rows))
        rows += frame_rows
        print(f.name, len(dets), poses, flush=True)
    with open(ROOT / "data" / "referee_game_slice.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("rows", len(rows))


if __name__ == "__main__":
    main()
