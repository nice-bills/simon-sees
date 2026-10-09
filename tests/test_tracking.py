"""The referee follows the player, wherever they stand (playtest bug: "it doesn't actually track the person,
it expects the person to be in the box").

A solo player is enrolled at the line-up, then walks across the frame, steps back, and changes pose (squat,
airplane, one foot). They must stay "seen", be judged on their own crop from the current frame, and the HUD
ring must follow them. Multi-player identity safety (no swaps) is checked alongside.

Two kinds of scene: the conftest-style two-tone blocks (fully controlled), and a real person cut out of
samples/people/ and pasted at different places / scales into an empty park (MockEmbedder, no model weights).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import ScriptedIO
from PIL import Image, ImageDraw, ImageOps

from simon_sees.commands import BY_ID, Call
from simon_sees.decider import MockDecider
from simon_sees.engine import Frame, Game, Settings, lineup, players_from_lineup
from simon_sees.identity import Identity, MatchConfig, assign
from simon_sees.pipeline import LocalVision, Stack, process_batch
from simon_sees.vision import Detection, MockEmbedder, MockSegmenter, keep_people

ROOT = Path(__file__).resolve().parents[1]
PEOPLE_DIR = ROOT / "samples" / "people"
BOXES = json.loads((ROOT / "scripts" / "fixtures" / "sample-boxes.json").read_text())  # 640-px-wide coords
FAST = dict(reaction_s=0, frame_gap_s=0, freeze_grace_s=0, freeze_window_s=0, rest_s=0)
W, H = 640, 480
GRASS = (110, 160, 90)


# --------------------------------------------------------------------------- block people that move and pose


def block_person(d: ImageDraw.ImageDraw, cx: float, pose: str = "stand", scale: float = 1.0, foot: float = 440,
                 colours=((200, 40, 40), (30, 30, 90))) -> tuple[float, float, float, float]:
    """A two-tone 'person' standing at x=cx. Poses change the silhouette (and so the mock embedding) a lot."""
    top, bottom = colours
    h, w = 300 * scale, 70 * scale
    if pose == "squat":
        h = 170 * scale
    y0 = foot - h
    x0, x1 = cx - w / 2, cx + w / 2
    d.rectangle((x0, y0, x1, y0 + h * 0.5), fill=top)
    d.rectangle((x0, y0 + h * 0.5, x1, foot), fill=bottom)
    if pose == "airplane":  # arms straight out: a wide bar of the top colour
        d.rectangle((cx - 2.6 * w, y0 + h * 0.18, cx + 2.6 * w, y0 + h * 0.26), fill=top)
        return (cx - 2.6 * w, y0, cx + 2.6 * w, foot)
    if pose == "one_foot":  # one leg lifted: the lower block is cut in half on one side
        d.rectangle((cx, foot - h * 0.22, x1, foot), fill=GRASS)
    if pose == "arms_up":
        d.rectangle((x0, y0 - 0.35 * h, x0 + w * 0.25, y0), fill=top)
        d.rectangle((x1 - w * 0.25, y0 - 0.35 * h, x1, y0), fill=top)
        return (x0, y0 - 0.35 * h, x1, foot)
    return (x0, y0, x1, foot)


def block_scene(*people: tuple) -> Image.Image:
    """people = (cx, pose, scale[, colours]). Boxes ride along in img.info like conftest.scene."""
    img = Image.new("RGB", (W, H), GRASS)
    d = ImageDraw.Draw(img)
    boxes = []
    for p in people:
        cx, pose, scale = p[:3]
        kw = {"colours": p[3]} if len(p) > 3 else {}
        boxes.append(("x", block_person(d, cx, pose, scale, **kw)))
    img.info["boxes"] = boxes
    return img


def segmenter_from_info(min_height: float = 0.12) -> MockSegmenter:
    """Reads the boxes the scene drew, then applies the real segmenters' size filter."""
    return MockSegmenter(script=lambda i, img: keep_people([Detection(b, 0.95) for _, b in img.info.get("boxes", [])],
                                                           img.height, 0.5, min_height, 8))


class PoseSensitiveEmbedder:
    """Behaves like the measured DINOv2-small box-crop embeddings (fast profile): the signature is dominated by
    the silhouette, so the SAME person in a different pose scores ~0.2-0.35 against their line-up crop (measured:
    stand vs airplane 0.28, stand vs squat 0.23), while the same pose anywhere in the frame scores ~1.
    70% one-hot silhouette bucket (aspect ratio of the crop), 30% the colour layout of MockEmbedder."""

    name = "pose-sensitive mock"

    def __init__(self):
        self.colour = MockEmbedder()

    def embed(self, crops):
        col = self.colour.embed(crops)
        out = []
        for c, v in zip(crops, col):
            bucket = int(np.clip(round(np.log(c.width / c.height) / 0.25) + 16, 0, 31))
            pose = np.zeros(32, np.float32)
            pose[bucket] = 1.0
            out.append(np.concatenate([np.sqrt(0.3) * v, np.sqrt(0.7) * pose]))
        return np.asarray(out, np.float32)


def stack(decider=None, embedder=None) -> Stack:
    return Stack(segmenter_from_info(), embedder or PoseSensitiveEmbedder(), decider)


# the walk: line-up in the middle, then all over the frame, in poses that change the silhouette
WALK = [
    ("lineup centre", (320, "stand", 1.0)),
    ("walks to the left edge", (70, "stand", 1.0)),
    ("squats at the left edge", (70, "squat", 1.0)),
    ("runs to the right edge, airplane", (560, "airplane", 0.8)),
    ("one foot, back in the middle", (330, "one_foot", 1.0)),
    ("arms up on the left", (150, "arms_up", 1.0)),
    ("steps far back (small)", (480, "stand", 0.45)),
    ("squats far back", (200, "squat", 0.55)),
]


def test_solo_player_is_followed_across_the_frame_and_through_pose_changes():
    st = stack()
    enrol = process_batch(st, {"kind": "enrol"}, [block_scene(WALK[0][1])])
    players, roster = players_from_lineup(enrol["people"])
    ident = roster.identities[players[0].id]
    for name, person in WALK[1:]:
        img = block_scene(person)
        res = process_batch(st, {"kind": "observe"}, [img], [ident])
        roster.apply(res["updates"])
        info = res["frames"][0]["players"]["0"]
        assert info["seen"], f"lost the solo player when they {name} (sim {info['sim']})"
        want = img.info["boxes"][0][1]
        assert info["box"] == pytest.approx([round(v, 1) for v in want], abs=0.2), name  # the box of THIS frame
        assert ident.last_box == pytest.approx(want, abs=0.2)


def test_solo_player_is_followed_even_with_a_passer_by_in_shot():
    passer = ((240, 220, 40), (240, 220, 40))  # a stranger in a yellow block, not enrolled
    st = stack(embedder=MockEmbedder())  # appearance-led signature: a stranger in the same pose looks different
    enrol = process_batch(st, {"kind": "enrol"}, [block_scene(WALK[0][1])])
    _, roster = players_from_lineup(enrol["people"])
    ident = roster.identities[0]
    for cx, pose, other in ((70, "stand", 520), (520, "squat", 120), (300, "one_foot", 560)):
        img = block_scene((cx, pose, 1.0), (other, "stand", 1.0, passer))
        res = process_batch(st, {"kind": "observe"}, [img], [ident])
        roster.apply(res["updates"])
        info = res["frames"][0]["players"]["0"]
        assert info["seen"], (cx, pose)
        assert (info["box"][0] + info["box"][2]) / 2 == pytest.approx(cx, abs=1)  # the player, not the stranger


def test_player_far_back_is_still_detected():
    """min_height used to be 0.25 of the frame: someone 6-8 m from a propped-up phone vanished."""
    small = Detection((300, 300, 330, 370), 0.9)  # 70 px of a 480 px frame = 0.146
    tiny = Detection((300, 340, 310, 370), 0.9)  # 30 px: a far-away passer-by, still dropped
    kept = keep_people([small, tiny], 480, 0.5, 0.12, 8)
    assert kept == [small]
    from simon_sees import vision
    import inspect
    for cls in (vision.Sam3Segmenter, vision.Sam31Segmenter, vision.FallbackDetector):
        assert inspect.signature(cls.__init__).parameters["min_height"].default == pytest.approx(0.12), cls


# --------------------------------------------------------------------------- the real-photo person


def _photo_person(photo: str, idx: int, pad: float = 0.04) -> Image.Image:
    x0, y0, x1, y1 = (v * 1.5 for v in BOXES[photo][idx])  # fixture boxes are 640 px wide, photos 960
    w, h = x1 - x0, y1 - y0
    return Image.open(PEOPLE_DIR / photo).convert("RGB").crop((int(x0 - pad * w), int(y0 - pad * h),
                                                                int(x1 + pad * w), int(y1 + pad * h)))


def _park() -> Image.Image:
    """An empty park: the left strip of a sample photo (nobody in it), mirrored and tiled."""
    strip = Image.open(PEOPLE_DIR / "000_lineup.jpg").convert("RGB").crop((0, 0, 260, 717))
    bg = Image.new("RGB", (960, 717))
    for i in range(4):
        bg.paste(strip if i % 2 == 0 else ImageOps.mirror(strip), (i * 260, 0))
    return bg


def _paste(bg: Image.Image, person: Image.Image, cx: float, scale: float = 1.0, foot: float = 690) -> Image.Image:
    im = bg.copy()
    if scale != 1.0:
        person = person.resize((max(1, int(person.width * scale)), max(1, int(person.height * scale))))
    x0, y0 = int(cx - person.width / 2), int(foot - person.height)
    im.paste(person, (x0, y0))
    im = im.resize((640, 478))
    k = 640 / 960
    im.info["boxes"] = [("p", (x0 * k, y0 * k, (x0 + person.width) * k, (y0 + person.height) * k))]
    return im


def test_real_person_photo_followed_around_the_park():
    bg = _park()
    stand = _photo_person("000_lineup.jpg", 0)
    one_foot = _photo_person("002_onefoot_hands-head_stand.jpg", 0)
    arms_up = _photo_person("003_arms-up_stand_arms-up.jpg", 0)
    st = stack(embedder=PoseSensitiveEmbedder())
    enrol = process_batch(st, {"kind": "enrol"}, [_paste(bg, stand, 480)])
    _, roster = players_from_lineup(enrol["people"])
    ident = roster.identities[0]
    for person, cx, scale in ((stand, 140, 1.0), (one_foot, 140, 1.0), (arms_up, 820, 1.0), (one_foot, 480, 1.0),
                              (stand, 650, 0.5), (arms_up, 200, 0.45)):
        img = _paste(bg, person, cx, scale)
        res = process_batch(st, {"kind": "observe"}, [img], [ident])
        roster.apply(res["updates"])
        info = res["frames"][0]["players"]["0"]
        assert info["seen"], (cx, scale, info)
        bx = info["box"]
        assert (bx[0] + bx[2]) / 2 == pytest.approx(cx * 640 / 960, abs=3)


# --------------------------------------------------------------------------- identity rules


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _orth(v, *others):
    v = np.asarray(v, dtype=np.float64)
    for o in others:
        v = v - (v @ o) * o
    return _unit(v)


def test_assign_solo_single_detection_is_the_player_wherever_it_is():
    rng = np.random.default_rng(1)
    me = Identity(0, [_unit(rng.normal(size=32))], (300, 100, 360, 420))
    far_away_pose = _unit(rng.normal(size=32))  # nothing like the line-up crop (cosine ~0)
    [m] = assign([me], far_away_pose[None], [(20, 200, 140, 420)])
    assert m.det == 0
    assert MatchConfig().threshold == 0.45  # the normal bar is unchanged; solo just doesn't need it


def _at(base, cos, rng, *avoid):
    """A unit vector with cosine `cos` to `base` and ~0 to everything in `avoid` (a pose change, a stranger)."""
    o = _orth(rng.normal(size=len(base)), base, *avoid)
    return _unit(cos * np.asarray(base, dtype=np.float64) + np.sqrt(1 - cos * cos) * o)


def test_assign_solo_with_a_stranger_needs_a_clear_winner():
    rng = np.random.default_rng(2)
    base = _unit(rng.normal(size=32))
    me = Identity(0, [base], (300, 100, 360, 420))
    mine = _at(base, 0.38, rng)  # squatted AND moved: the real DINOv2 cosine measured in the repro was 0.3785
    stranger = _at(base, 0.15, rng, mine)
    boxes = [(20, 100, 80, 420), (500, 200, 600, 420)]
    [m] = assign([me], np.stack([stranger, mine]), boxes)
    assert m.det == 1
    # two people who are both about as (un)like the player: not seen, never guessed
    [m] = assign([me], np.stack([_at(base, 0.36, rng), _at(base, 0.33, rng)]), boxes)
    assert m.det is None


def test_assign_two_players_count_rule_never_swaps_them():
    rng = np.random.default_rng(3)
    a = _unit(rng.normal(size=32))
    b = _orth(rng.normal(size=32), a)
    ida, idb = Identity(0, [a], (60, 100, 140, 420)), Identity(1, [b], (460, 100, 540, 420))
    # both moved (crossed over) AND changed pose: each is below the 0.45 bar but still most like themselves
    a2, b2 = _at(a, 0.38, rng, b), _at(b, 0.35, rng, a)
    ms = assign([ida, idb], np.stack([b2, a2]), [(250, 100, 330, 420), (560, 100, 630, 420)])
    assert {m.pid: m.det for m in ms} == {0: 1, 1: 0}
    # a detection that looks more like player 1 than like player 0 is never handed to player 0 on the count
    amb = _unit(0.3 * a + 0.5 * b + 0.8 * _orth(rng.normal(size=32), a, b))
    assert float(amb @ a) >= MatchConfig().count_floor
    ms = assign([ida, idb], np.stack([amb, b]), [(250, 100, 330, 420), (460, 100, 540, 420)])
    assert {m.pid: m.det for m in ms} == {0: None, 1: 1}


def test_multi_player_strangers_are_not_counted_in():
    rng = np.random.default_rng(4)
    a, b = _unit(rng.normal(size=32)), _unit(rng.normal(size=32))
    ida, idb = Identity(0, [a], (60, 100, 140, 420)), Identity(1, [b], (460, 100, 540, 420))
    s1 = _orth(rng.normal(size=32), a, b)  # nothing like either player
    s2 = _orth(rng.normal(size=32), a, b)
    # b left, two strangers walked in: more people than players, so no count rule
    ms = assign([ida, idb], np.stack([a, s1, s2]), [(60, 100, 140, 420), (250, 100, 330, 420), (560, 100, 620, 420)])
    assert {m.pid: m.det for m in ms} == {0: 0, 1: None}
    # b left and one stranger who looks nothing like b walked in: the head count matches, the floor still holds
    ms = assign([ida, idb], np.stack([a, s1]), [(60, 100, 140, 420), (250, 100, 330, 420)])
    assert {m.pid: m.det for m in ms} == {0: 0, 1: None}


# --------------------------------------------------------------------------- the game: judged, ring follows


def _solo_game(scenes, calls, decider=None, settings=None):
    async def go():
        io = ScriptedIO(scenes)
        d = decider or MockDecider(lambda name, meta: 0.9 if name == "did" else 0.05)
        vision = LocalVision(stack(d))
        res = await lineup(vision, io)
        players, roster = players_from_lineup(res["people"], ["Kofi"])

        class Picker:
            def next(self, n):
                return calls[min(n - 1, len(calls) - 1)]

        game = Game(vision, io, players, roster, settings or Settings(**FAST, max_rounds=len(calls)), picker=Picker())
        await game.run()
        return game, io, d
    return asyncio.run(go())


def test_solo_game_judges_the_player_every_round_while_they_move_and_pose():
    calls = [Call(BY_ID["squat"], True), Call(BY_ID["airplane"], True), Call(BY_ID["one_foot"], True),
             Call(BY_ID["freeze"], True)]
    scenes = [block_scene((320, "stand", 1.0)),           # line-up
              block_scene((320, "stand", 1.0)),           # locate
              block_scene((80, "squat", 1.0)), block_scene((90, "squat", 1.0)),        # round 1: 2 frames
              block_scene((540, "airplane", 0.8)), block_scene((545, "airplane", 0.8)),  # round 2
              block_scene((420, "one_foot", 0.5)), block_scene((420, "one_foot", 0.5)),  # round 3, far back
              block_scene((150, "stand", 1.0)), block_scene((150, "stand", 1.0))]     # round 4: freeze before/after
    game, io, d = _solo_game(scenes, calls)
    assert not any("lost you" in t.lower() or "can't see" in t.lower() for t in io.texts), io.texts
    judged = [r for batch in d.calls for r in batch]
    assert sum(1 for r in judged if r.meta.get("kind") == "pose") == 6  # 3 pose rounds x 2 frames, all judged
    assert game.players[0].alive and game.players[0].seen
    # the HUD ring is where the player is in the last judged frame, not where they lined up
    assert game.players[0].box == pytest.approx(list(scenes[-1].info["boxes"][0][1]), abs=0.2)


def test_lost_player_has_no_stale_ring():
    calls = [Call(BY_ID["squat"], True)]
    scenes = [block_scene((320, "stand", 1.0)), block_scene((320, "stand", 1.0)),
              Image.new("RGB", (W, H), GRASS), Image.new("RGB", (W, H), GRASS)]
    game, io, _ = _solo_game(scenes, calls)
    p = game.players[0]
    assert p.seen is False and p.box is None  # no ring left at the line-up spot telling them to stand there


def test_follow_moves_the_ring_between_looks():
    async def go():
        io = ScriptedIO([block_scene((320, "stand", 1.0))])
        vision = LocalVision(stack(MockDecider(lambda n, m: 0.9)))
        res = await lineup(vision, io)
        players, roster = players_from_lineup(res["people"])
        game = Game(vision, io, players, roster, Settings(**FAST, max_rounds=1))
        game.phase = "rest"
        moved = block_scene((90, "squat", 1.0))
        assert await game.follow(Frame(moved)) is True
        assert game.players[0].box == pytest.approx(list(moved.info["boxes"][0][1]), abs=0.2)
        assert game.view_frame.image is moved and game.view_seq == 1 and io.states[-1]["view"]["seq"] == 1
        game.phase = "watching"  # the judged look is about to happen: following waits
        assert await game.follow(Frame(block_scene((500, "stand", 1.0)))) is False
        assert game.players[0].box == pytest.approx(list(moved.info["boxes"][0][1]), abs=0.2)
    asyncio.run(go())
