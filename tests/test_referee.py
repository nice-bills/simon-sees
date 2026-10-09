"""The fine-tuned referee's pieces: the keypoint rules, the masked crops, and the tinker backend (mocked client)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from simon_sees.decider import Request, action_question
from simon_sees.pose_rules import RULE_COMMANDS, Pose, label_all
from simon_sees.vision import Detection, crop_person

# --------------------------------------------------------------------------- keypoint rules

# a standing person, 100 px torso, facing the camera. (x, y); y grows downwards.
STAND = {"nose": (100, 40), "leye": (105, 35), "reye": (95, 35), "lear": (110, 40), "rear": (90, 40),
         "lsho": (125, 80), "rsho": (75, 80), "lelb": (130, 130), "relb": (70, 130), "lwri": (132, 175),
         "rwri": (68, 175), "lhip": (115, 180), "rhip": (85, 180), "lknee": (115, 260), "rknee": (85, 260),
         "lank": (115, 340), "rank": (85, 340)}
NAMES = ["nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
         "lhip", "rhip", "lknee", "rknee", "lank", "rank"]


def kps(**moves) -> list[float]:
    pts = {**STAND, **moves}
    out: list[float] = []
    for n in NAMES:
        x, y = pts[n]
        out += [x, y, 2]
    return out


def test_standing_person_is_a_clear_no_for_every_pose():
    labels = label_all(kps())
    assert labels, "a clean standing pose should be labelled"
    assert not any(labels.values()), labels
    assert set(labels) <= set(RULE_COMMANDS)


def test_arms_up():
    up = kps(lelb=(130, 20), relb=(70, 20), lwri=(132, -40), rwri=(68, -40))
    assert label_all(up)["arms_up"] is True
    assert label_all(up).get("hands_head") is False  # straight up is not hands on head


def test_hands_on_head():
    hh = kps(lelb=(165, 40), relb=(35, 40), lwri=(115, 5), rwri=(85, 5))
    lab = label_all(hh)
    assert lab["hands_head"] is True
    assert lab.get("arms_up") is not True


def test_batting_stance_is_not_hands_on_head():
    # both hands together beside the head, both elbows on the same side
    bat = kps(lelb=(150, 70), relb=(140, 90), lwri=(125, 20), rwri=(120, 25))
    assert label_all(bat).get("hands_head") is not True


def test_airplane():
    plane = kps(lelb=(175, 80), relb=(25, 80), lwri=(225, 82), rwri=(-25, 82))
    assert label_all(plane)["airplane"] is True


def test_one_foot():
    lifted = kps(lknee=(150, 230), lank=(120, 280))
    assert label_all(lifted)["one_foot"] is True


def test_squat_vs_chair():
    squat = kps(lsho=(125, 180), rsho=(75, 180), lhip=(120, 280), rhip=(80, 280), lknee=(150, 250),
                rknee=(50, 250), lank=(120, 320), rank=(80, 320), nose=(100, 140), lelb=(130, 230),
                relb=(70, 230), lwri=(140, 250), rwri=(60, 250), leye=(105, 135), reye=(95, 135))
    assert label_all(squat)["squat"] is True


def test_touch_toes():
    bent = kps(lsho=(40, 200), rsho=(40, 190), nose=(20, 260), lhip=(130, 180), rhip=(130, 175),
               lknee=(132, 260), rknee=(128, 260), lank=(130, 340), rank=(126, 340),
               lelb=(60, 260), relb=(55, 255), lwri=(110, 330), rwri=(105, 325))
    assert label_all(bent)["touch_toes"] is True


def test_invisible_keypoints_are_never_labelled():
    k = kps()
    for i in range(len(k) // 3):
        k[3 * i + 2] = 1  # labelled but not visible
    assert label_all(k) == {}
    assert Pose(k).T is None


# --------------------------------------------------------------------------- masked crops


def _frame_with_two_people():
    img = Image.new("RGB", (300, 200), (40, 200, 40))
    a = np.asarray(img).copy()
    a[40:190, 60:120] = (220, 30, 30)  # player A (red)
    a[40:190, 128:188] = (30, 30, 220)  # neighbour B (blue), close enough to sit in A's crop margin
    img = Image.fromarray(a)
    ma = np.zeros((200, 300), bool)
    ma[40:190, 60:120] = True
    mb = np.zeros((200, 300), bool)
    mb[40:190, 128:188] = True
    return img, Detection((60, 40, 120, 190), 0.9, ma), Detection((128, 40, 188, 190), 0.9, mb)


def _blue(c: Image.Image) -> int:
    a = np.asarray(c).astype(int)
    return int(((a[..., 2] > 150) & (a[..., 0] < 100)).sum())


def test_crop_without_others_still_shows_the_neighbour_dimmed_and_with_others_hides_them():
    img, da, db = _frame_with_two_people()
    old = crop_person(img, da, pad=0.3, long_side=None, background="keep")
    assert _blue(old) > 50  # the defect: the neighbour is in the margin
    new = crop_person(img, da, pad=0.3, long_side=None, others=[da, db])
    assert _blue(new) == 0
    a = np.asarray(new).astype(int)
    assert ((a[..., 0] > 200) & (a[..., 1] < 60)).sum() > 0.9 * 60 * 150  # the player is untouched


def test_crop_with_boxes_only_paints_other_boxes_outside_own_box():
    img, da, db = _frame_with_two_people()
    da_box, db_box = Detection(da.box, 0.9), Detection(db.box, 0.9)
    new = crop_person(img, da_box, pad=0.3, long_side=None, others=[db_box])
    assert _blue(new) == 0
    assert ((np.asarray(new)[..., 0] > 200)).sum() > 0.9 * 60 * 150


def test_pipeline_crops_isolate_players():
    from simon_sees.pipeline import Stack, process_batch
    from simon_sees.vision import MockEmbedder

    img, da, db = _frame_with_two_people()

    class Seg:
        name, has_masks = "seg", True

        def segment(self, images):
            return [[da, db] for _ in images]

    out = process_batch(Stack(Seg(), MockEmbedder()), {"kind": "enrol"}, [img])
    import base64
    import io

    photo = Image.open(io.BytesIO(base64.b64decode(out["people"][0]["photo"]))).convert("RGB")
    assert _blue(photo) == 0


# --------------------------------------------------------------------------- tinker backend


class FakeTok:
    vocab = {"yes": 1, "no": 2, "Yes": 3, "No": 4, " yes": 5, " no": 6}

    def encode(self, s, add_special_tokens=False):
        if s in self.vocab:
            return [self.vocab[s]]
        if s in ("YES", "NO", " Yes", " No"):
            return [90, 91]
        return [7] * max(1, len(s) // 8)


class FakeSampler:
    """Returns yes/no top-k logprobs depending on the prompt length (a stand-in for the image content)."""

    def __init__(self):
        self.calls = []

    def sample(self, prompt, num_samples, sampling_params, topk_sample_logprobs=0, **kw):
        self.calls.append((prompt, sampling_params, topk_sample_logprobs))
        img = next(c for c in prompt.chunks if getattr(c, "type", "") == "image")
        p_yes = 0.9 if img.expected_tokens > 66 else 0.2
        seq = SimpleNamespace(tokens=[1], logprobs=[math.log(p_yes)],
                              topk_logprobs=[[(1, math.log(p_yes)), (2, math.log(1 - p_yes)), (9, -9.0)]])
        fut = SimpleNamespace(result=lambda: SimpleNamespace(sequences=[seq]))
        return fut


def test_p_yes_from_topk():
    from simon_sees.tinker_referee import p_yes_from_topk

    assert p_yes_from_topk([(1, math.log(0.6)), (2, math.log(0.2))], {1}, {2}) == pytest.approx(0.75)
    assert p_yes_from_topk([(9, -0.1)], {1}, {2}) == 0.5


def test_image_tokens_match_qwen_resize():
    from simon_sees.tinker_referee import image_tokens, smart_resize

    assert smart_resize(200, 384) == (192, 384)
    assert image_tokens(Image.new("RGB", (384, 200))) == 72
    assert image_tokens(Image.new("RGB", (100, 100))) == 64  # upscaled to the 65536-pixel minimum


def test_tinker_decider_batches_a_round_and_reads_p_yes():
    pytest.importorskip("tinker")
    from simon_sees.tinker_referee import TinkerDecider, TinkerScorer

    sampler = FakeSampler()
    scorer = TinkerScorer(client=sampler, tokenizer=FakeTok())
    d = TinkerDecider(model_path="tinker://run/sampler_weights/final", scorer=scorer)
    q = action_question("Is that person standing on one foot?", "one foot up", "both feet down")
    big, small = Image.new("RGB", (192, 384)), Image.new("RGB", (384, 96))  # 72 vs 64 image tokens
    res = d.decide([Request([big], {"did": q}, meta={"player": 0}), Request([small], {"did": q}, meta={"player": 1})])
    assert len(sampler.calls) == 2  # one request per player crop, sent together
    assert res[0].p("did") == pytest.approx(0.9, abs=1e-6)
    assert res[1].p("did") == pytest.approx(0.2, abs=1e-6)
    assert d.multi_image is False  # freeze stays on pixel differencing
    assert d.spans and d.spans[0]["span"] == "tinker.sample" and d.spans[0]["requests"] == 2
    prompt, params, k = sampler.calls[0]
    assert params.max_tokens == 1 and k == 20
    assert "tinker://run" in d.name


def test_training_datum_weights_only_the_answer():
    pytest.importorskip("tinker")
    from simon_sees.tinker_referee import image_tokens, training_datum

    img = Image.new("RGB", (192, 384))
    dt = training_datum(img, "Is that person squatting? Answer with one word: yes or no.", True, FakeTok())
    w = dt.loss_fn_inputs["weights"].data
    t = dt.loss_fn_inputs["target_tokens"].data
    assert len(w) == len(t) == dt.model_input.length
    assert sum(w) == 2  # "yes" + <|im_end|>
    assert t[w.index(1.0)] == FakeTok.vocab["yes"]
    assert dt.model_input.length > image_tokens(img)


def test_make_decider_tinker_falls_back_to_d1_factory(monkeypatch):
    from simon_sees import decider as dmod

    made = {}

    class FakeTD:
        name = "tinker"
        multi_image = False

        def __init__(self, *a, **kw):
            made["tinker"] = kw

    monkeypatch.setattr("simon_sees.tinker_referee.TinkerDecider", FakeTD)
    d = dmod.make_decider("tinker", "LiquidAI/d1-omni-600M", tinker_model="tinker://x/sampler_weights/y")
    assert isinstance(d, FakeTD) and made["tinker"]["model_path"] == "tinker://x/sampler_weights/y"


def test_dataset_split_is_by_image_and_deterministic():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_referee_dataset import image_split

    assert image_split(42) == image_split(42)
    splits = [image_split(i) for i in range(4000)]
    assert 0.03 < splits.count("test") / 4000 < 0.12
    assert splits.count("train") > splits.count("val") > 0


@pytest.mark.parametrize("env,expected", [
    ({"TINKER_API_KEY": "tml-main"}, "tml-main"),
    ({"SIMON_TINKER_API_KEY": "tml-alias"}, "tml-alias"),
    ({"TINKER_API_KEY": "tml-main", "SIMON_TINKER_API_KEY": "tml-alias"}, "tml-main"),
    ({}, None),
])
def test_tinker_scorer_reads_tinker_api_key_then_the_simon_alias(monkeypatch, env, expected):
    """The documented TINKER_API_KEY reaches tinker.ServiceClient; SIMON_TINKER_API_KEY is only a fallback."""
    from simon_sees import tinker_referee

    for k in ("TINKER_API_KEY", "SIMON_TINKER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    seen = {}

    class FakeServiceClient:
        def __init__(self, **kw):
            seen["kw"] = kw

        def create_sampling_client(self, **kw):
            seen["sampling"] = kw
            return FakeSampler()

    monkeypatch.setitem(sys.modules, "tinker", SimpleNamespace(ServiceClient=FakeServiceClient))
    tinker_referee.TinkerScorer(model_path="tinker://run/sampler_weights/final", tokenizer=FakeTok())
    assert seen["kw"] == ({"api_key": expected} if expected else {})
    assert seen["sampling"] == {"model_path": "tinker://run/sampler_weights/final"}
    assert tinker_referee.tinker_api_key() == expected
