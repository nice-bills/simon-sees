"""Game engine + identity tests with a mocked segmenter, embedder and decision model (no torch, no camera).

Scenes are synthetic: two-tone blocks standing in three places (conftest.scene). The mocked segmenter
reads the block boxes, the MockEmbedder turns each crop into a pixel-layout signature, and the scripted
MockDecider answers per player (by enrolled id), so every test controls exactly who did what."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from conftest import PEOPLE, ScriptedIO, scene, vision_for
from PIL import Image

from simon_sees.commands import BY_ID, COMMANDS, Call, CommandPicker
from simon_sees.decider import MockDecider
from simon_sees.decision_log import DecisionLog
from simon_sees.engine import Game, Player, Settings, combine, judge, lineup, out_line, players_from_lineup
from simon_sees.identity import Identity, MatchConfig, assign, brute_force_assignment, hungarian, order_left_to_right
from simon_sees.motion import motion_score, moved_probability
from simon_sees.pipeline import process_batch
from simon_sees.voice import parse_name

FAST = dict(reaction_s=0, frame_gap_s=0, freeze_grace_s=0, freeze_window_s=0, rest_s=0)
S = Settings(**FAST)
LINEUP = ["a", "b", "c"]  # left, centre, right at the line-up


class FixedPicker:
    """Plays a fixed list of calls, then repeats the last."""

    def __init__(self, calls):
        self.calls = list(calls)

    def next(self, round_no):
        return self.calls[min(round_no - 1, len(self.calls) - 1)]


def by_player(table, default_did=0.9, default_moved=0.05):
    """MockDecider script: {(pid, 'did'|'moved'): p}. pid is the line-up index (0 = leftmost)."""
    def f(name, meta):
        v = table.get((meta.get("player"), name))
        if isinstance(v, list):
            return v[meta.get("frame", 0)]
        return v if v is not None else (default_did if name == "did" else default_moved)
    return f


async def _setup(decider, scenes, names=None, settings=None, log=None, picker=None):
    io = ScriptedIO(scenes)
    vision = vision_for(decider)
    res = await lineup(vision, io)
    players, roster = players_from_lineup(res["people"], names or [])
    game = Game(vision, io, players, roster, settings or Settings(**FAST, max_rounds=3), log, picker=picker)
    return game, io, res


def play(decider, scenes, calls, names=None, max_rounds=3, log=None, **kw):
    async def go():
        game, io, _ = await _setup(decider, scenes, names, Settings(**FAST, max_rounds=max_rounds, **kw), log,
                                   FixedPicker(calls))
        winners = await game.run()
        return game, io, winners
    return asyncio.run(go())


# ------------------------------------------------------------------ rules


def test_judge_simon_says_disobey_is_out():
    call = Call(BY_ID["one_foot"], simon=True)
    assert judge(call, 0.1, True, S) == (True, "did_not_obey", False)
    assert judge(call, 0.9, True, S)[0] is False


def test_judge_trick_obey_is_out():
    call = Call(BY_ID["hands_head"], simon=False)
    assert judge(call, 0.9, True, S) == (True, "simon_didnt_say", False)
    assert judge(call, 0.1, True, S)[0] is False


def test_judge_uncertain_band_gives_benefit_of_doubt():
    out, _, unsure = judge(Call(BY_ID["one_foot"], simon=True), 0.42, True, S)
    assert not out and unsure
    out, _, unsure = judge(Call(BY_ID["one_foot"], simon=False), 0.58, True, S)
    assert not out and unsure


def test_judge_freeze():
    call = Call(BY_ID["freeze"], simon=True)
    assert judge(call, 0.9, True, S) == (True, "moved", False)
    assert judge(call, 0.05, True, S)[0] is False


def test_not_seen_is_never_out():
    assert judge(Call(BY_ID["one_foot"], simon=True), 0.0, False, S) == (False, "not_visible", True)


def test_combine():
    assert combine([0.2, 0.4]) == pytest.approx(0.3)
    assert combine([0.2, None, 0.4]) == pytest.approx(0.3)
    assert combine([0.2, 0.4], "max") == 0.4
    assert combine([0.2, 0.4], "min") == 0.2
    assert combine([]) == 0.0


def test_names_and_callouts():
    kofi, anon = Player(0, 1, "Kofi"), Player(1, 2)
    assert kofi.call_name == "Kofi" and kofi.display_name == "Kofi"
    assert anon.call_name == "Number 2" and anon.display_name == "Player 2"
    assert Player(2, 3, "Player 3").call_name == "Number 3"  # the placeholder is not a name
    assert out_line(kofi, Call(BY_ID["one_foot"], True), "did_not_obey") == "Kofi, you kept both feet down. You're out."
    assert out_line(anon, Call(BY_ID["wave"], False), "simon_didnt_say") == "Number 2, Simon didn't say! You're out."
    assert out_line(kofi, Call(BY_ID["freeze"], True), "moved") == "Kofi, you moved. You're out."


def test_no_clothing_words_in_any_question_or_line():
    from simon_sees.decider import action_question, freeze_question

    banned = ("wearing", "shirt", "cap", "jacket", "hoodie", "clothes", "colour", "color")
    for c in COMMANDS:
        for q in (action_question(c.check, c.true_hint, c.false_hint), freeze_question(c.check)):
            assert not any(w in q["instructions"].lower() for w in banned), q
        assert not any(w in c.out_phrase.lower() for w in banned)


# ------------------------------------------------------------------ commands


def test_call_text():
    assert Call(BY_ID["one_foot"], True).text == "Simon says stand on one foot."
    assert Call(BY_ID["one_foot"], False).text == "Stand on one foot!"


def test_picker_is_seeded_and_never_tricks_first_round_or_with_freeze():
    a = [CommandPicker(seed=3).next(r) for r in range(1, 40)]
    b = [CommandPicker(seed=3).next(r) for r in range(1, 40)]
    assert [(c.command.id, c.simon) for c in a] == [(c.command.id, c.simon) for c in b]
    picker = CommandPicker(seed=11, trick_rate=1.0)
    assert picker.next(1).simon
    for r in range(2, 60):
        c = picker.next(r)
        if c.command.kind == "freeze":
            assert c.simon


def test_picker_indoors_skips_outdoor_props():
    assert all(not c.outside_only for c in CommandPicker(seed=1, outdoors=False).pool)


def test_all_commands_have_out_phrase():
    assert all(c.out_phrase for c in COMMANDS)


# ------------------------------------------------------------------ identity: Hungarian + re-ID


@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2), (5, 5), (1, 3)])
def test_hungarian_matches_brute_force(shape, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_scipy(name, *a, **kw):  # exercise the pure-python fallback even if scipy is installed
        if name.startswith("scipy"):
            raise ImportError
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    rng = np.random.default_rng(sum(shape))
    for _ in range(20):
        cost = rng.random(shape)
        pairs = hungarian(cost)
        assert len(pairs) == min(shape)
        assert len({r for r, _ in pairs}) == len(pairs) and len({c for _, c in pairs}) == len(pairs)
        assert sum(cost[r, c] for r, c in pairs) == pytest.approx(brute_force_assignment(cost))


def test_enrolment_orders_players_left_to_right():
    # enrol a frame where the people stand c, a, b: Player 1 must be whoever is leftmost (c)
    io = ScriptedIO([scene(["c", "a", "b"])])
    vision = vision_for(MockDecider())
    res = asyncio.run(lineup(vision, io))
    xs = [p["box"][0] for p in res["people"]]
    assert xs == sorted(xs) and len(res["people"]) == 3
    assert all(p["photo"] and len(p["emb"]) > 0 for p in res["people"])  # every player gets their photo card
    assert order_left_to_right([(300, 0, 310, 1), (10, 0, 20, 1), (100, 0, 120, 1)]) == [1, 2, 0]
    players, roster = players_from_lineup(res["people"], ["Kofi", "", "Ama"])
    assert [p.number for p in players] == [1, 2, 3]
    assert [p.call_name for p in players] == ["Kofi", "Number 2", "Ama"]
    assert set(roster.identities) == {0, 1, 2}


def test_reid_follows_players_who_swap_places():
    vision = vision_for(None)
    enrol = vision.run({"kind": "enrol"}, [scene(LINEUP)])
    _, roster = players_from_lineup(enrol["people"])
    # same three people, positions permuted: a->right, b->left, c->centre
    res = vision.run({"kind": "observe"}, [scene(["b", "c", "a"])], list(roster.identities.values()))
    where = {int(pid): info["box"][0] for pid, info in res["frames"][0]["players"].items()}
    assert all(info["seen"] for info in res["frames"][0]["players"].values())
    assert where[0] == 460 and where[1] == 60 and where[2] == 260  # a (pid 0) is now on the right, etc.
    # and across two frames in one batch, swapping again in the second
    res = vision.run({"kind": "observe"}, [scene(["b", "c", "a"]), scene(["a", "b", "c"])],
                     list(roster.identities.values()))
    second = {int(pid): info["box"][0] for pid, info in res["frames"][1]["players"].items()}
    assert second == {0: 60, 1: 260, 2: 460}


def test_assign_rejects_strangers_below_threshold():
    ident = Identity(0, [np.array([1.0, 0.0])])
    ms = assign([ident], np.array([[0.0, 1.0]]), [(0, 0, 10, 10)], MatchConfig(threshold=0.5))
    assert ms[0].det is None
    ms = assign([ident], np.array([[0.9, 0.436]]), [(0, 0, 10, 10)], MatchConfig(threshold=0.5))
    assert ms[0].det == 0


def test_assign_keeps_a_player_through_a_big_pose_change_on_the_same_spot():
    # cosine only 0.3 (arms out like an airplane), but the detection is where the player just was
    ident = Identity(0, [np.array([1.0, 0.0])], last_box=(100, 50, 200, 400))
    emb = np.array([[0.3, 0.954]])
    assert assign([ident], emb, [(90, 40, 260, 400)], MatchConfig(threshold=0.45))[0].det == 0
    assert assign([ident], emb, [(400, 40, 500, 400)], MatchConfig(threshold=0.45))[0].det is None


def test_gallery_learns_confident_sightings_and_keeps_the_lineup_one():
    ident = Identity(0, [np.array([1.0, 0.0])], max_gallery=3)
    for k in range(5):
        ident.remember(np.array([0.9, 0.1 * k]), (k, 0, k + 1, 1))
    assert len(ident.gallery) == 3 and np.allclose(ident.gallery[0], [1.0, 0.0])
    assert ident.last_box == (4, 0, 5, 1)


# ------------------------------------------------------------------ full games


def test_two_player_game_eliminates_the_one_who_did_not_obey(tmp_path):
    d = MockDecider(by_player({(1, "did"): 0.05}))  # the centre player never does anything
    logp = tmp_path / "d.jsonl"
    game, io, winners = play(d, [scene(["a", "b", None])],
                             [Call(BY_ID["one_foot"], True)], names=["Kofi", "Ama"], max_rounds=5,
                             log=DecisionLog(logp))
    assert [p.display_name for p in winners] == ["Kofi"]
    assert game.round == 1
    assert "Ama, you kept both feet down. You're out." in io.texts
    assert any("Kofi wins" in t for t in io.texts)
    rows = [json.loads(line) for line in logp.read_text().splitlines()]
    assert {"game_start", "locate", "decision", "verdict", "game_end"} <= {r["event"] for r in rows}
    dec = [r for r in rows if r["event"] == "decision"]
    assert all("latency_ms" in r and "p_per_frame" in r for r in dec)
    assert all("Look at the person in this photo" in r["question"] for r in dec)


def test_unnamed_players_are_called_by_number():
    d = MockDecider(by_player({(1, "did"): 0.95}, default_did=0.03))
    game, io, winners = play(d, [scene(LINEUP)], [Call(BY_ID["hands_head"], False)], names=["Kofi"])
    assert "Number 2, Simon didn't say! You're out." in io.texts
    assert {p.display_name for p in winners} == {"Kofi", "Player 3"}


def test_one_request_per_player_per_frame_each_with_that_players_own_crop():
    d = MockDecider(by_player({}))
    game, io, _ = play(d, [scene(LINEUP)], [Call(BY_ID["arms_up"], True)], max_rounds=1, pose_frames=2)
    pose_batch = d.calls[-1]  # one decide() call for the whole round
    assert len(pose_batch) == 6  # 3 players x 2 frames
    assert all(len(r.images) == 1 and list(r.questions) == ["did"] for r in pose_batch)
    assert {(r.meta["player"], r.meta["frame"]) for r in pose_batch} == {(p, f) for p in range(3) for f in range(2)}
    assert all(r.images[0].size[1] > r.images[0].size[0] for r in pose_batch)  # a person-shaped crop, not the frame


def test_reid_in_game_after_swapping_places_judges_the_right_person():
    # line-up a b c; during the round they stand b c a. Player 1 (a) is the one who doesn't obey:
    # the elimination must go to Player 1 even though a is now on the RIGHT.
    def f(name, meta):
        return 0.05 if meta["player"] == 0 else 0.95
    d = MockDecider(f)
    game, io, winners = play(d, [scene(LINEUP), scene(LINEUP), scene(["b", "c", "a"])],
                             [Call(BY_ID["squat"], True)], names=["Kofi", "Ama", "Raj"], max_rounds=1)
    assert "Kofi, you didn't squat. You're out." in io.texts
    assert {p.display_name for p in winners} == {"Ama", "Raj"}
    # and the crop that was judged for Kofi really is a's pixels
    req = next(r for r in d.calls[-1] if r.meta["player"] == 0)
    top = np.asarray(req.images[0])[req.images[0].height // 3, req.images[0].width // 2]
    assert np.allclose(top, PEOPLE["a"][0], atol=40)


def test_lost_player_is_warned_never_eliminated():
    d = MockDecider(by_player({(1, "did"): 0.0}))
    game, io, winners = play(d, [scene(LINEUP), scene(LINEUP), scene(["a", None, "c"])],
                             [Call(BY_ID["one_foot"], True)], names=["Kofi", "Ama", "Raj"], max_rounds=1)
    assert all(p.alive for p in game.players)
    assert any(t.startswith("Heads up, Ama: I lost you.") for t in io.texts)
    ama = next(p for p in game.players if p.name == "Ama")
    assert ama.seen is False and ama.unseen_rounds == 1
    assert {r.meta["player"] for r in d.calls[-1]} == {0, 2}  # nobody judged a crop that isn't there


def test_missing_at_kickoff_is_asked_to_step_in(monkeypatch):
    monkeypatch.setattr("simon_sees.engine.asyncio.sleep", _no_sleep)
    d = MockDecider(by_player({}))
    game, io, _ = play(d, [scene(LINEUP), scene(["a", None, "c"]), scene(LINEUP)],
                       [Call(BY_ID["wave"], True)], names=["Kofi", "Ama", "Raj"], max_rounds=1)
    assert any(t == "I can't see Ama. Step into the picture." for t in io.texts)


async def _no_sleep(*_a, **_k):
    return None


def test_freeze_uses_each_players_before_and_after_crops():
    d = MockDecider(by_player({(0, "moved"): 0.92}))
    game, io, winners = play(d, [scene(LINEUP)], [Call(BY_ID["freeze"], True)], names=["Kofi", "Ama", "Raj"],
                             max_rounds=1)
    freeze_batch = d.calls[-1]
    assert len(freeze_batch) == 3 and all(len(r.images) == 2 for r in freeze_batch)
    assert all(r.images[0].size == r.images[1].size for r in freeze_batch)  # same box on both frames
    assert all(list(r.questions) == ["moved"] and "same person" in r.questions["moved"]["instructions"]
               for r in freeze_batch)
    assert "Kofi, you moved. You're out." in io.texts


def test_freeze_crops_follow_the_player_not_the_place():
    # BEFORE: a b c. AFTER: a and b swapped. Without re-ID, the pixels in each place changed and
    # everyone would "move"; with re-ID each pair of crops is the same person.
    d = MockDecider(by_player({}), multi_image=False)  # pixel FALLBACK decides
    game, io, winners = play(d, [scene(LINEUP), scene(LINEUP), scene(LINEUP), scene(["b", "a", "c"])],
                             [Call(BY_ID["freeze"], True)], max_rounds=1)
    # a and b really did walk (their union boxes span both places) -> out; c stood still -> safe
    assert [p.alive for p in game.players] == [False, False, True]


def test_freeze_pixel_fallback_without_multi_image():
    d = MockDecider(by_player({}), multi_image=False)
    game, io, _ = play(d, [scene(LINEUP)], [Call(BY_ID["freeze"], True)], max_rounds=1)
    assert all(p.alive for p in game.players)  # identical frames -> no motion
    assert not d.calls  # the decision model is never asked: pixel differencing inside each player's box decides


def test_freeze_pixel_fallback_catches_the_mover():
    d = MockDecider(by_player({}), multi_image=False)
    game, io, _ = play(d, [scene(LINEUP), scene(LINEUP), scene(LINEUP), scene(LINEUP, pose={"b": -60})],
                       [Call(BY_ID["freeze"], True)], names=["Kofi", "Ama", "Raj"], max_rounds=1)
    assert [p.name for p in game.players if not p.alive] == ["Ama"]


def test_everyone_caught_round_is_void():
    d = MockDecider(by_player({}, default_did=0.01))
    game, io, winners = play(d, [scene(["a", "b", None])], [Call(BY_ID["squat"], True)], max_rounds=2)
    assert len(winners) == 2
    assert any("doesn't count" in t for t in io.texts)


def test_solo_game_counts_rounds():
    d = MockDecider(by_player({}))
    game, io, _ = play(d, [scene([None, "a", None])], [Call(BY_ID["wave"], True)], max_rounds=3)
    assert game.round == 3
    assert any("survived 3 rounds" in t for t in io.texts)


def test_stop():
    async def go():
        game, io, _ = await _setup(MockDecider(by_player({})), [scene(LINEUP)],
                                   settings=Settings(**FAST, max_rounds=50), picker=FixedPicker([Call(BY_ID["wave"], True)]))
        task = asyncio.create_task(game.run())
        await asyncio.sleep(0)
        game.stop()
        await task
        return game
    assert asyncio.run(go()).round < 50


def test_save_frames(tmp_path):
    game, io, _ = play(MockDecider(by_player({})), [scene(LINEUP)], [Call(BY_ID["wave"], True)], max_rounds=1,
                       save_frames=str(tmp_path))
    assert list((tmp_path / game.id).glob("r01_wave_*.jpg"))


def test_state_carries_photos_and_numbers_not_clothing():
    game, io, _ = play(MockDecider(by_player({})), [scene(LINEUP)], [Call(BY_ID["wave"], True)], max_rounds=1,
                       names=["Kofi"])
    st = io.states[-1]
    assert [p["number"] for p in st["players"]] == [1, 2, 3]
    assert [p["display_name"] for p in st["players"]] == ["Kofi", "Player 2", "Player 3"]
    assert all(p["photo"].startswith("/api/photo/") for p in st["players"])
    assert "desc" not in st["players"][0] and "position" not in st["players"][0]
    assert st["models"]["segmenter"] == "mock" and st["models"]["embedder"] == "mock"


# ------------------------------------------------------------------ helpers


def test_motion_fallback():
    a = Image.new("RGB", (96, 72), (100, 100, 100))
    b = a.copy()
    assert motion_score(a, b) == 0.0
    b.paste((250, 250, 250), (0, 0, 40, 72))
    assert motion_score(a, b) > 0.3
    assert motion_score(a, b, (60, 0, 96, 72)) < motion_score(a, b, (0, 0, 60, 72))
    assert moved_probability(0.0) == 0.0 and moved_probability(1.0) == 1.0


@pytest.mark.parametrize("text,want", [
    ("I'm Kofi.", "Kofi"),
    ("hi, my name is ama mensah and I'm ready", "Ama Mensah"),
    ("Call me Yaw", "Yaw"),
    ("um, it's Abena", "Abena"),
])
def test_parse_name(text, want):
    assert parse_name(text) == want


def test_process_batch_reports_latency_and_models():
    vision = vision_for(MockDecider(by_player({})))
    enrol = vision.run({"kind": "enrol"}, [scene(LINEUP)])
    _, roster = players_from_lineup(enrol["people"])
    res = process_batch(vision.stack, {"kind": "pose", "check": "Is that person waving?"}, [scene(LINEUP)],
                        list(roster.identities.values()))
    assert set(res["latency_ms"]) >= {"segment", "embed", "match", "decide", "total"}
    assert res["models"] == {"segmenter": "mock", "embedder": "mock", "decider": "mock"}
    assert json.dumps(res)  # the Modal endpoint returns exactly this dict
