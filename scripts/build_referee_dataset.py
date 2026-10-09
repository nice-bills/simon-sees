"""Rebuild the referee fine-tuning set: COCO 2017 person keypoints -> rule labels -> game-style crops.

    python scripts/build_referee_dataset.py --coco-ann DIR --out data/referee [--download]

Steps (all deterministic, seed 0):
  1. read person_keypoints_{train,val}2017.json (CC-BY 4.0 annotations, https://cocodataset.org)
  2. label every clearly visible person with the explicit rules in simon_sees/pose_rules.py
     (confident positive / clear negative / dropped as ambiguous)
  3. split by IMAGE id (a hash), so no photo is in two splits; sample per command, balanced yes/no,
     half the negatives "hard" (people doing a different Simon pose)
  4. download only the images that were picked (images.cocodataset.org) and crop each person exactly the
     way the game crops a player (simon_sees.vision.crop_person: box + 12% margin, outside the COCO
     segmentation mask dimmed, other people painted grey, long side 384 px)
  5. write {out}/manifest.jsonl: one line per (person, command) example, with the image's own
     Flickr licence and URL.

The images are NOT committed: data/referee/ is git-ignored. The TEST split is not re-sampled: it is exactly
the human-checked examples in data/referee_test_verified.jsonl (each with its human label, or drop=true).
Optional label cleaning (--teacher preds.jsonl): keep a train/val example only when a large VLM
(Qwen3.6-35B-A3B, untuned, scored by bench/referee_eval.py) agrees with the rule label.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simon_sees.pose_rules import RULE_COMMANDS, label_all  # noqa: E402
from simon_sees.vision import Detection, crop_person  # noqa: E402

CAPS = {"train": (300, 150), "val": (14, 14), "test": (16, 16)}  # per command: max positives, min negatives
SPLIT_FRACTIONS = {"test": 0.07, "val": 0.06}


RARE_FRACTIONS = {"test": 0.24, "val": 0.12}  # images holding a positive of a rare pose (few in COCO)
RARE_BELOW = 200  # a command with fewer confident positives than this in all of COCO is "rare"


def image_split(image_id: int, rare: bool = False) -> str:
    """Split by IMAGE, so no photo is in two splits. Images that hold a positive of a rare pose use larger
    test/val fractions, so the test set has enough positives of every command to measure."""
    fr = RARE_FRACTIONS if rare else SPLIT_FRACTIONS
    h = int(hashlib.sha1(f"coco{image_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < fr["test"]:
        return "test"
    if h < fr["test"] + fr["val"]:
        return "val"
    return "train"


def usable(a: dict) -> bool:
    x, y, w, h = a["bbox"]
    return not a["iscrowd"] and a["num_keypoints"] >= 8 and h >= 120 and w >= 40


def load(ann_dir: Path) -> tuple[dict, dict, list]:
    images, by_image, people = {}, defaultdict(list), []
    for coco_split in ("train", "val"):
        d = json.load(open(ann_dir / f"person_keypoints_{coco_split}2017.json"))
        lic = {l["id"]: l for l in d["licenses"]}
        for im in d["images"]:
            im["coco_split"] = coco_split
            im["licence"] = lic[im["license"]]["name"]
            im["licence_url"] = lic[im["license"]]["url"]
            images[im["id"]] = im
        for a in d["annotations"]:
            by_image[a["image_id"]].append(a)
            if usable(a):
                labels = label_all(a["keypoints"])
                if labels:
                    people.append((a, labels))
    return images, by_image, people


def crowded(a: dict, others: list[dict]) -> bool:
    """Skip a person whose box is mostly covered by another (non-crowd) person's box."""
    x, y, w, h = a["bbox"]
    for o in others:
        if o["id"] == a["id"] or o.get("iscrowd"):
            continue
        ox, oy, ow, oh = o["bbox"]
        iw = max(0, min(x + w, ox + ow) - max(x, ox))
        ih = max(0, min(y + h, oy + oh) - max(y, oy))
        if iw * ih > 0.45 * w * h:
            return True
    return False


VERIFIED = Path(__file__).resolve().parents[1] / "data" / "referee_test_verified.jsonl"


def select(people: list, by_image: dict, seed: int = 0, teacher: dict | None = None) -> list[dict]:
    rng = random.Random(seed)
    pools: dict = defaultdict(lambda: defaultdict(list))  # split -> (cmd, kind) -> [ann]
    people = [(a, labels) for a, labels in people if not crowded(a, by_image[a["image_id"]])]
    n_pos = defaultdict(int)
    for a, labels in people:
        for c, v in labels.items():
            n_pos[c] += bool(v)
    rare_cmds = {c for c in RULE_COMMANDS if n_pos[c] < RARE_BELOW}
    rare_images = {a["image_id"] for a, labels in people if any(labels.get(c) for c in rare_cmds)}
    for a, labels in people:
        split = image_split(a["image_id"], a["image_id"] in rare_images)
        any_pos = {c for c, v in labels.items() if v}
        for cmd, v in labels.items():
            if v:
                pools[split][(cmd, "pos")].append(a)
            elif any_pos - {cmd}:
                pools[split][(cmd, "hard")].append(a)
            else:
                pools[split][(cmd, "neg")].append(a)
    def agrees(split, cmd, a, label):
        if teacher is None:
            return True
        p = teacher.get(f"{split}-{cmd}-{a['id']}")
        return p is not None and (p >= 0.5) == label

    out = []
    verified = [json.loads(l) for l in open(VERIFIED)] if VERIFIED.exists() else []
    if verified:
        for v in verified:
            out.append({"id": v["id"], "split": "test", "command": v["command"], "label": v["label"],
                        "rule_label": v.get("rule_label"), "drop": v.get("drop", False), "neg_kind": None,
                        "ann_id": v["ann_id"], "image_id": v["image_id"]})
    for split, (max_pos, min_neg) in CAPS.items():
        if split == "test" and verified:
            continue
        for cmd in RULE_COMMANDS:
            pools[split][(cmd, "pos")] = [a for a in pools[split][(cmd, "pos")] if agrees(split, cmd, a, True)]
            pools[split][(cmd, "hard")] = [a for a in pools[split][(cmd, "hard")] if agrees(split, cmd, a, False)]
            pools[split][(cmd, "neg")] = [a for a in pools[split][(cmd, "neg")] if agrees(split, cmd, a, False)]
            pos = sorted(pools[split][(cmd, "pos")], key=lambda a: a["id"])
            rng.shuffle(pos)
            pos = pos[:max_pos]
            n_neg = max(len(pos), min_neg)
            hard = sorted(pools[split][(cmd, "hard")], key=lambda a: a["id"])
            easy = sorted(pools[split][(cmd, "neg")], key=lambda a: a["id"])
            rng.shuffle(hard)
            rng.shuffle(easy)
            hard = hard[: n_neg // 2]
            neg = hard + easy[: n_neg - len(hard)]
            for a, label, kind in [(a, True, "pos") for a in pos] + [(a, False, "hard" if a in hard else "neg") for a in neg]:
                out.append({"id": f"{split}-{cmd}-{a['id']}", "split": split, "command": cmd, "label": label,
                            "neg_kind": None if label else kind, "ann_id": a["id"], "image_id": a["image_id"]})
    return out


def poly_mask(segs, w: int, h: int) -> np.ndarray | None:
    if not isinstance(segs, list) or not segs:
        return None
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in segs:
        if len(poly) >= 6:
            d.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
    return np.asarray(m, dtype=bool)


def fetch(im: dict, cache: Path) -> Path:
    p = cache / im["file_name"]
    if p.exists() and p.stat().st_size > 1000:
        return p
    url = f"http://images.cocodataset.org/{im['coco_split']}2017/{im['file_name']}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - fixed public host
                data = r.read()
            tmp = p.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(p)
            return p
        except Exception:  # noqa: BLE001
            # some sandboxes only let curl through their proxy
            import subprocess

            if subprocess.run(["curl", "-sf", "--max-time", "30", "-o", str(p), url]).returncode == 0:
                return p
            time.sleep(1 + attempt)
    raise RuntimeError(f"could not download {url}")


def to_det(a: dict, w: int, h: int) -> Detection:
    x, y, bw, bh = a["bbox"]
    return Detection((x, y, x + bw, y + bh), 1.0, poly_mask(a["segmentation"], w, h))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-ann", type=Path, required=True, help="dir with person_keypoints_{train,val}2017.json")
    ap.add_argument("--out", type=Path, default=Path("data/referee"))
    ap.add_argument("--images", type=Path, default=None, help="COCO image cache (default {out}/coco_images)")
    ap.add_argument("--download", action="store_true", help="download the picked images and write the crops")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit-seconds", type=float, default=0, help="stop downloading after N s (resumable)")
    ap.add_argument("--teacher", type=Path, default=None, help="teacher predictions (jsonl id,p) for label cleaning")
    ap.add_argument("--uncapped", action="store_true", help="write every confident example (for teacher scoring)")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    images, by_image, people = load(args.coco_ann)
    test_images = {json.loads(l)["image_id"] for l in open(VERIFIED)} if VERIFIED.exists() else set()
    people = [(a, l) for a, l in people if a["image_id"] not in test_images]  # test photos never train
    teacher = None
    if args.teacher:
        teacher = {r["id"]: r["p"] for r in map(json.loads, open(args.teacher)) if "id" in r}
    if args.uncapped:
        CAPS["train"] = (10**6, 150)
        CAPS["val"] = (10**6, 30)
    rows = select(people, by_image, teacher=teacher)
    for r in rows:
        im = images[r["image_id"]]
        r.update({"coco_split": im["coco_split"], "file_name": im["file_name"], "licence": im["licence"],
                  "licence_url": im["licence_url"], "flickr_url": im.get("flickr_url"),
                  "crop": f"crops/{r['split']}/{r['ann_id']}.jpg"})
    rows = [r for r in rows if not r.get("drop")] if args.uncapped else rows
    with open(args.out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        counts[(r["split"], r["command"])][0 if r["label"] else 1] += 1
    print(json.dumps({f"{s}/{c}": v for (s, c), v in sorted(counts.items())}, indent=0))
    print("examples", len(rows), "images", len({r["image_id"] for r in rows}))
    if not args.download:
        return

    cache = args.images or (args.out / "coco_images")
    cache.mkdir(parents=True, exist_ok=True)
    need = sorted({r["image_id"] for r in rows})
    t0 = time.time()
    todo = [i for i in need if not (cache / images[i]["file_name"]).exists()]
    print("to download", len(todo))
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, images[i], cache): i for i in todo}
        for k, fu in enumerate(cf.as_completed(futs)):
            fu.result()
            if args.limit_seconds and time.time() - t0 > args.limit_seconds:
                print("time limit, downloaded", k + 1)
                ex.shutdown(wait=False, cancel_futures=True)
                return
    anns = {a["id"]: a for lst in by_image.values() for a in lst}
    done = 0
    for r in rows:
        dst = args.out / r["crop"]
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        im = images[r["image_id"]]
        img = Image.open(cache / im["file_name"]).convert("RGB")
        w, h = img.size
        a = anns[r["ann_id"]]
        others = [to_det(o, w, h) for o in by_image[r["image_id"]] if o["id"] != a["id"] and not o["iscrowd"]]
        crop_person(img, to_det(a, w, h), others=others).save(dst, quality=90)
        done += 1
    print("crops written", done)


if __name__ == "__main__":
    main()
