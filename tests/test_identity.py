"""Enrolment and re-identification, on hand-made embeddings (no models): who is who after the line-up."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import scene, scene_segmenter

from simon_sees.engine import players_from_lineup
from simon_sees.identity import Identity, MatchConfig, Roster, assign, continues_track, order_left_to_right
from simon_sees.pipeline import Stack, process_batch
from simon_sees.vision import MockEmbedder

RNG = np.random.default_rng(0)
D = 64


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


BASE = [unit(RNG.normal(size=D)) for _ in range(4)]  # three players + one stranger


def near(i, noise=0.25, seed=0):
    r = np.random.default_rng(seed)
    return unit(BASE[i] + noise * unit(r.normal(size=D)))


def box(slot):
    x0 = 60 + 200 * slot
    return (float(x0), 100.0, float(x0 + 80), 420.0)


def roster3():
    return [Identity(p, [BASE[p]], box(p)) for p in range(3)]


def test_enrol_numbers_left_to_right_whatever_order_the_segmenter_returns():
    boxes = [box(2), box(0), box(1)]
    assert order_left_to_right(boxes) == [1, 2, 0]
    stack = Stack(scene_segmenter(), MockEmbedder())
    r = process_batch(stack, {"kind": "enrol"}, [scene(["c", "a", "b"])])
    xs = [(p["box"][0] + p["box"][2]) / 2 for p in r["people"]]
    assert xs == sorted(xs) and len(r["people"]) == 3
    assert all(p["photo"] and len(p["emb"]) for p in r["people"])
    players, roster = players_from_lineup(r["people"], ["Kofi", "", "Ama"])
    assert [p.number for p in players] == [1, 2, 3]
    assert [p.display_name for p in players][0] == "Kofi" and players[1].display_name == "Player 2"
    assert sorted(roster.identities) == [0, 1, 2]


def test_rotation_swap_is_resolved_by_appearance_not_position():
    # everyone moved one place (a rotation): P1 is now on the right, P2 on the left, P3 in the middle
    embs = np.stack([near(1, seed=1), near(2, seed=2), near(0, seed=3)])
    ms = assign(roster3(), embs, [box(0), box(1), box(2)])
    assert {m.pid: m.det for m in ms} == {0: 2, 1: 0, 2: 1}


def test_tracking_bonus_cannot_outvote_a_clear_appearance_match():
    # two players swap exactly: each detection overlaps the OTHER player's last box completely
    idents = [Identity(0, [BASE[0]], box(0)), Identity(1, [BASE[1]], box(1))]
    embs = np.stack([near(1, 0.2, 4), near(0, 0.2, 5)])
    ms = assign(idents, embs, [box(0), box(1)], MatchConfig(track_bonus=0.15))
    assert {m.pid: m.det for m in ms} == {0: 1, 1: 0}


def test_player_who_left_is_unseen_and_the_others_stay_correct():
    embs = np.stack([near(0, seed=6), near(2, seed=7)])
    ms = {m.pid: m for m in assign(roster3(), embs, [box(0), box(2)])}
    assert ms[0].det == 0 and ms[2].det == 1 and ms[1].det is None


def test_a_stranger_walking_in_is_not_given_a_players_identity():
    idents = [Identity(0, [BASE[0]], box(0))]
    ms = assign(idents, np.stack([BASE[3]]), [box(2)])  # different person, different place
    assert ms[0].det is None


def test_stranger_does_not_steal_a_player_who_is_also_in_frame():
    embs = np.stack([BASE[3], near(0, seed=8)])
    ms = assign([Identity(0, [BASE[0]], box(0))], embs, [box(0), box(1)])
    assert ms[0].det == 1


def test_no_detections_means_nobody_is_seen():
    ms = assign(roster3(), np.zeros((0, D), np.float32), [])
    assert [m.det for m in ms] == [None, None, None]


def test_continues_track():
    assert continues_track(box(0), box(0))
    assert continues_track(box(0), (70.0, 120.0, 150.0, 420.0))
    assert not continues_track(box(0), box(2))
    assert not continues_track(None, box(0))


def test_identity_json_round_trip_and_roster_updates():
    ident = Identity(1, [BASE[1]], box(1))
    back = Identity.from_json(ident.to_json())
    assert back.pid == 1 and back.last_box == box(1)
    assert np.allclose(back.gallery[0], BASE[1], atol=1e-4)
    roster = Roster({1: back})
    roster.apply({"1": {"emb": near(1, seed=9).tolist(), "box": list(box(2))}})
    assert len(roster.identities[1].gallery) == 2 and roster.identities[1].last_box == box(2)
    roster.apply({"1": {"emb": None, "box": list(box(0))}, "7": {"emb": None, "box": list(box(1))}})
    assert len(roster.identities[1].gallery) == 2 and roster.identities[1].last_box == box(0)


@pytest.mark.parametrize("sightings", [1, 5, 12])
def test_gallery_is_capped_and_never_forgets_the_lineup_photo(sightings):
    ident = Identity(0, [BASE[0]], box(0), max_gallery=6)
    for k in range(sightings):
        ident.remember(near(0, seed=100 + k), box(0))
    assert len(ident.gallery) == min(1 + sightings, 6)
    assert np.array_equal(ident.gallery[0], BASE[0])


def test_observe_batch_reports_updates_for_the_roster():
    stack = Stack(scene_segmenter(), MockEmbedder())
    enrol = process_batch(stack, {"kind": "enrol"}, [scene(["a", "b", "c"])])
    _, roster = players_from_lineup(enrol["people"], [])
    r = process_batch(stack, {"kind": "observe"}, [scene(["b", "c", "a"])], list(roster.identities.values()))
    seen = r["frames"][0]["players"]
    order = sorted(((v["box"][0] + v["box"][2]) / 2, int(pid)) for pid, v in seen.items() if v["seen"])
    assert [pid for _, pid in order] == [1, 2, 0]
    roster.apply(r["updates"])
    assert roster.identities[0].last_box[0] == 460.0
