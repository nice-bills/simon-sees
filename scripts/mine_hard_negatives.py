"""Mine "near-miss" negatives for the referee (iteration 2) and write manifest-v2.jsonl.

    python scripts/mine_hard_negatives.py --coco-ann ../work/coco --data data/referee [--per-cmd 120] [--download]

The untuned and v1-tuned referees fail mostly on false positives: a skateboarder crouching or a baseball
player sliding read as "touching toes", someone sitting cross-legged as "squatting", a cyclist or runner as
"standing on one foot". The v1 negatives were mostly easy (upright, arms down). This script finds people whose
keypoints put them NEAR a pose but decisively fail one of its key conditions, and adds them as labelled
negatives (neg_kind="near"):

  touch_toes: torso bent over (> 35 deg) but legs deeply bent (< 110 deg) or both hands above the knees
  squat:      hips at ankle height (sitting on the ground / cross-legged), or kneeling, or chair height
  one_foot:   an ankle raised but the torso leaning (> 25 deg) or the standing leg bent (running, cycling, jumping)
  arms_up:    exactly one arm up (the other wrist below its shoulder)
  airplane:   exactly one arm out horizontal (the other hanging)
  hands_head: one hand at the head, the other below the shoulder

Only train/val images are used (same image hash split as build_referee_dataset.py; the human-verified test
photos are excluded), so TEST stays untouched.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from build_referee_dataset import (RARE_BELOW, VERIFIED, crowded, fetch, image_split, load,  # noqa: E402
                                   to_det, usable)
from simon_sees.pose_rules import RULE_COMMANDS, Pose, label_all  # noqa: E402
from simon_sees.vision import crop_person  # noqa: E402


def _knees(p):
    return [p.angle(f"{s}hip", f"{s}knee", f"{s}ank") for s in "lr"]


def near_touch_toes(p: Pose) -> bool:
    if not p.has("lsho", "rsho", "lhip", "rhip", "lknee", "rknee", "lank", "rank", "lwri", "rwri"):
        return False
    if p.torso_tilt() <= 35:
        return False
    wrists_low = min(p.above(w, k) for w, k in (("lwri", "lknee"), ("rwri", "rknee")))
    return min(_knees(p)) < 110 or wrists_low > 0.3


def near_squat(p: Pose) -> bool:
    if not p.has("lsho", "rsho", "lhip", "rhip", "lknee", "rknee", "lank", "rank"):
        return False
    knees = _knees(p)
    hip_over_ank = min(p.above(f"{s}hip", f"{s}ank") for s in "lr")
    knee_over_ank = min(p.above(f"{s}knee", f"{s}ank") for s in "lr")
    if max(knees) > 120:
        return False
    sitting_floor = hip_over_ank < 0.08
    kneeling = knee_over_ank < 0.15 and hip_over_ank > 0.3
    chair = hip_over_ank > 0.85 and p.upright(25)
    return sitting_floor or kneeling or chair


def near_one_foot(p: Pose) -> bool:
    if not p.has("lhip", "rhip", "lknee", "rknee", "lank", "rank", "lsho", "rsho"):
        return False
    lift = p.above("lank", "rank")
    if abs(lift) < 0.35:
        return False
    lo = "r" if lift > 0 else "l"
    stand = p.angle(f"{lo}hip", f"{lo}knee", f"{lo}ank")
    return p.torso_tilt() > 25 or stand < 140


def _wrist_vs_sho(p):
    return [p.above(f"{s}wri", f"{s}sho") for s in "lr"]


def near_arms_up(p: Pose) -> bool:
    if not p.has("nose", "lsho", "rsho", "lwri", "rwri"):
        return False
    up = [p.above(f"{s}wri", "nose") for s in "lr"]
    d = _wrist_vs_sho(p)
    return max(up) > 0.3 and min(d) < -0.2


def near_airplane(p: Pose) -> bool:
    if not p.has("lsho", "rsho", "lelb", "relb", "lwri", "rwri"):
        return False
    out = []
    for s in "lr":
        dx, dy = p[f"{s}wri"][0] - p[f"{s}sho"][0], p[f"{s}wri"][1] - p[f"{s}sho"][1]
        out.append(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6)) < 22 and abs(dx) / p.T > 0.75)
    return sum(out) == 1 and min(_wrist_vs_sho(p)) < -0.5


def near_hands_head(p: Pose) -> bool:
    if not p.has("nose", "lsho", "rsho", "lwri", "rwri"):
        return False
    up = [p.above(f"{s}wri", "nose") for s in "lr"]
    dx = [abs(p[f"{s}wri"][0] - p["nose"][0]) / p.T for s in "lr"]
    d = _wrist_vs_sho(p)
    one_at_head = any(0.0 < u < 0.7 and x < 0.5 for u, x in zip(up, dx))
    return one_at_head and min(d) < -0.3


NEAR = {"touch_toes": near_touch_toes, "squat": near_squat, "one_foot": near_one_foot, "arms_up": near_arms_up,
        "airplane": near_airplane, "hands_head": near_hands_head}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-ann", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "referee")
    ap.add_argument("--per-cmd", type=int, default=120, help="near-miss negatives per command (train)")
    ap.add_argument("--val-per-cmd", type=int, default=14)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args(argv)
    images, by_image, people = load(a.coco_ann)
    test_images = {json.loads(l)["image_id"] for l in open(VERIFIED)}
    base = [json.loads(l) for l in open(a.data / "manifest.jsonl")]
    have = {(r["command"], r["ann_id"]) for r in base}
    # same rare-image split as the main builder
    ppl = [(x, l) for x, l in people if not crowded(x, by_image[x["image_id"]])]
    n_pos = defaultdict(int)
    for x, l in ppl:
        for c, v in l.items():
            n_pos[c] += bool(v)
    rare = {c for c in RULE_COMMANDS if n_pos[c] < RARE_BELOW}
    rare_images = {x["image_id"] for x, l in ppl if any(l.get(c) for c in rare)}
    cands = defaultdict(list)
    for img_id, anns in by_image.items():
        if img_id in test_images:
            continue
        split = image_split(img_id, img_id in rare_images)
        if split == "test":
            continue
        for x in anns:
            if not usable(x) or crowded(x, anns):
                continue
            p = Pose(x["keypoints"])
            if p.T is None:
                continue
            labels = label_all(x["keypoints"])
            for cmd, fn in NEAR.items():
                if labels.get(cmd) is True or (cmd, x["id"]) in have:
                    continue
                try:
                    hit = fn(p)
                except (TypeError, ValueError, ZeroDivisionError):
                    hit = False
                if hit:
                    cands[(split, cmd)].append(x)
    rng = random.Random(a.seed)
    new = []
    for (split, cmd), lst in sorted(cands.items()):
        lst.sort(key=lambda x: x["id"])
        rng.shuffle(lst)
        k = a.per_cmd if split == "train" else a.val_per_cmd
        for x in lst[:k]:
            im = images[x["image_id"]]
            new.append({"id": f"{split}-{cmd}-{x['id']}", "split": split, "command": cmd, "label": False,
                        "rule_label": None, "drop": False, "neg_kind": "near", "ann_id": x["id"],
                        "image_id": x["image_id"], "coco_split": im["coco_split"], "file_name": im["file_name"],
                        "licence": im["licence"], "licence_url": im["licence_url"], "flickr_url": im.get("flickr_url"),
                        "crop": f"crops/{split}/{x['id']}.jpg"})
        print(split, cmd, "candidates", len(lst), "taken", min(k, len(lst)))
    out = a.data / "manifest-v2.jsonl"
    with open(out, "w") as f:
        for r in base + new:
            f.write(json.dumps(r) + "\n")
    print("wrote", out, len(base), "+", len(new))
    if not a.download:
        return
    cache = a.data / "coco_images"
    need = sorted({r["image_id"] for r in new})
    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda i: fetch(images[i], cache), need))
    anns = {x["id"]: x for lst in by_image.values() for x in lst}
    for r in new:
        dst = a.data / r["crop"]
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(cache / images[r["image_id"]]["file_name"]).convert("RGB")
        w, h = img.size
        x = anns[r["ann_id"]]
        others = [to_det(o, w, h) for o in by_image[r["image_id"]] if o["id"] != x["id"] and not o["iscrowd"]]
        crop_person(img, to_det(x, w, h), others=others).save(dst, quality=90)
    print("crops done")


if __name__ == "__main__":
    main()
