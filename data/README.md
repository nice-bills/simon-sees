# Referee fine-tuning data

The fine-tuned referee (Qwen3.5-4B + LoRA, trained on Tinker) answers one question per player crop:
"Look at the person in this photo. Is that person ...? Answer with one word: yes or no." This folder documents
where its training and test examples come from. The images are not committed; they are rebuilt by
`scripts/build_referee_dataset.py` (only `data/referee_test_verified.jsonl`, the small real-game slice and this
README are in the repo).

## Source and licence

- **Annotations:** COCO 2017 person keypoints (`person_keypoints_train2017.json`, `person_keypoints_val2017.json`),
  © COCO Consortium, licensed **CC BY 4.0** (https://cocodataset.org/#termsofuse).
- **Images:** the COCO photos are Flickr images, each under its own Creative Commons licence (the COCO
  `licenses` table: CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA, CC BY-ND, CC BY-NC-ND, and a few "No known copyright
  restrictions" / US-government works). Every example in the manifest carries its image's licence name, licence URL
  and Flickr URL. Only the images that are actually used are downloaded, from `images.cocodataset.org`. They are
  used here to train and evaluate a research model; they are not redistributed by this repo.
- **Real-game slice:** `data/referee_game/` holds the player crops from `samples/people` (generated park photos
  made for this project), cut out by the game's own SAM 3 pipeline.

## Labels: explicit geometric rules

`simon_sees/pose_rules.py` turns a person's 17 keypoints into a label per command. Only keypoints marked visible
(v = 2) count. All distances are in torso lengths T (shoulder midpoint to hip midpoint), so the rules do not
depend on how large the person is. Each rule returns a confident **yes**, a clear **no**, or **ambiguous**
(dropped). Summary (see the code for the exact numbers):

| command | yes when | clear no when |
|---|---|---|
| `arms_up` (reach for the sky) | both wrists ≥ 0.3 T above the nose, elbows ≥ 0.1 T above the shoulders, arms straight (> 120°), torso upright | a wrist below its shoulder, or both hands near the head with bent elbows |
| `hands_head` | both wrists between nose height and 0.7 T above it, within 0.5 T of the nose sideways, elbows bent (< 120°) and pointing out to opposite sides | both wrists below the shoulders, arms straight up, or only one hand up |
| `airplane` | both arms within 22° of horizontal, reaching ≥ 0.75 T out, straight (> 145°), on opposite sides, torso upright | both arms hanging down, both arms up |
| `one_foot` | one ankle ≥ 0.45 T higher than the other, standing leg straight (> 160°) and under the hips, raised knee bent (< 130°), torso upright | ankles level (< 0.08 T) and both legs straight |
| `squat` | both knees < 95°, hips 0.15–0.65 T above the ankles (low but off the ground), knees ≥ 0.35 T above the ankles (not kneeling), torso < 60° from vertical | upright, legs straight, hips ≥ 1.3 T above the ankles |
| `touch_toes` | torso tilted > 45°, legs fairly straight (> 120°), a wrist within 0.8 T of an ankle and both wrists below the knees | upright with both wrists well above the knees |

People are also skipped when their box is mostly covered by another person's (the crop would be ambiguous), when
their box is under 120 px tall, or when fewer than 8 keypoints are labelled.

**Not covered:** `freeze` (judged by frame differencing), `sit_down` (ground versus chair is not visible in
keypoints), `touch_green` / `touch_ground` / `hug_tree` (they need the prop, not just the pose). `hand_mouth` and
`wave` had rules too, but the human check showed they do not work on COCO: a hand near the face is mostly a phone
call, a drink or a batting stance, and one raised hand is mostly a sport swing (1 in 16 and 3 in 16 rule positives
were real). They were dropped from training and testing.

## Crops

Each person is cropped exactly as the game crops a player (`simon_sees.vision.crop_person`): the box plus a 12 %
margin, everything outside the COCO segmentation mask dimmed, every other annotated person in the photo painted
flat grey (the same neighbour isolation the game now applies with SAM 3 masks), long side 384 px.

## Splits

Split by **image** (a hash of the COCO image id), so no photo is in two splits. Images that contain a positive of a
rare pose (fewer than 200 confident positives in all of COCO: `arms_up`, `hands_head`, `airplane`, `touch_toes`)
use larger test/val fractions so the test set has enough of them. Per command, train takes up to 300 positives and
at least 150 negatives (never fewer negatives than positives), half of the negatives "hard" (a person confidently doing a
different Simon pose).

**Label cleaning (train/val only):** a training example is kept only when an untuned large VLM
(Qwen3.6-35B-A3B on Tinker, same prompt) agrees with the rule label. This removes rule positives such as
cyclists and jumpers (`one_foot`), people sitting cross-legged (`squat`) and batting stances (`hands_head`) that
the human check found in the test candidates. The test labels never use the teacher; they are human.

## The test set: human-verified

`data/referee_test_verified.jsonl` is the test set. Every crop was looked at by eye (the person cropped as the
game would) and labelled against the command's question:

- 179 rule-labelled test candidates (6 commands): 15 rule labels were wrong and flipped (bicycle riders,
  jumping, tennis and batting swings, sitting cross-legged...), 19 ambiguous ones (mid-air, unclear) dropped.
- 83 extra candidates found with deliberately loose finders (to get more positives of the rare poses), labelled by
  hand: 31 yes, 31 no, 21 dropped.
- Result: **222 test examples, 81 yes / 141 no**, from 210 COCO images that are never used for training.

Plus the real-game slice (`data/referee_game_slice.jsonl`): every player crop from the 7 `samples/people` frames
asked every command.

## Rebuild

```bash
# 1. annotations (241 MB zip, CC BY 4.0)
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -j annotations_trainval2017.zip 'annotations/person_keypoints_*2017.json' -d coco/
# 2. candidate pool (every confident example), images + crops
python scripts/build_referee_dataset.py --coco-ann coco --out data/referee --uncapped --download
cp data/referee/manifest.jsonl data/referee/pool.jsonl
# 3. teacher agreement on the pool (Tinker, Qwen3.6-35B-A3B), then the final capped, cleaned manifest
for s in train val; do python bench/referee_eval.py --data data/referee --split $s \
    --model tinker-base:Qwen/Qwen3.6-35B-A3B --out data/referee/teacher-$s.jsonl; done
cat data/referee/teacher-*.jsonl > data/referee/teacher.jsonl
python scripts/build_referee_dataset.py --coco-ann coco --out data/referee --teacher data/referee/teacher.jsonl --download
# 4. the real-game slice (SAM 3 on CPU, ~20 s per frame)
python scripts/build_game_slice.py
```
