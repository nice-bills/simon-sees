"""The playtest loop: session recorder, the feedback endpoint, bundle, label export, upload (huggingface_hub
mocked), pull from a bundle, and the markdown report. No network, no models."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import ScriptedIO, scene, vision_for
from fastapi.testclient import TestClient
from PIL import Image

from simon_sees.decider import MockDecider
from simon_sees.engine import Game, Settings, lineup, players_from_lineup
from simon_sees.playtest_data import answer_from_verdict, load_session, standing_corrections
from simon_sees.recorder import NullRecorder, Recorder, list_sessions, make_recorder, read_jsonl
from simon_sees.upload import MARKER, changed_files, make_bundle, upload_sessions
from test_server import band_vision, fast_cfg, jpeg, until

ROOT = Path(__file__).resolve().parents[1]


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(rec, seed=4, rounds=4, decider=None) -> Game:
    """A full mocked game recorded by `rec` (three 'people', seeded mock decider)."""
    scenes = [scene(["a", "b", "c"])]
    vision = vision_for(decider or MockDecider(seed=seed))
    io = ScriptedIO(scenes)

    async def go():
        res = await lineup(vision, io)
        players, roster = players_from_lineup(res["people"], ["Kofi", "", "Ama"])
        g = Game(vision, io, players, roster, Settings(reaction_s=0, frame_gap_s=0, freeze_grace_s=0, freeze_window_s=0,
                                                       rest_s=0, max_rounds=rounds, seed=seed), recorder=rec)
        await g.run()
        return g

    return asyncio.run(go())


@pytest.fixture
def session(tmp_path) -> Recorder:
    rec = Recorder(tmp_path / "sessions", meta={"mode": "test", "seed": 4, "api_token": "hf_secret"}, capture_log=True)
    yield rec
    rec.close()


# --------------------------------------------------------------------------- recorder


def test_recorder_writes_a_complete_session(session):
    game = play(session)
    d = session.dir
    meta = json.loads((d / "session.json").read_text())
    assert meta["mode"] == "test" and meta["seed"] == 4 and "python" in meta["versions"]
    assert meta["api_token"] == "***"  # secrets never land on disk
    assert meta["games"][0]["game_id"] == game.id and meta["games"][0]["ended"]
    events = list(read_jsonl(d / "events.jsonl"))
    kinds = [e["event"] for e in events]
    assert kinds[0] == "game_start" and "locate" in kinds and kinds[-1] == "game_end"
    rounds = [e for e in events if e["event"] == "round"]
    assert len(rounds) == game.round >= 1
    r = rounds[0]
    assert r["round_id"] == f"{game.id}-r01" and r["command"] and isinstance(r["simon"], bool)
    assert r["trick"] == (not r["simon"]) and r["freeze"] == (r["kind"] == "freeze")
    assert set(r["timings_ms"]) >= {"segment", "embed", "match", "d1", "tts_call", "round_wall"}
    for p in r["players"]:
        assert {"pid", "sim_per_frame", "seen_per_frame", "p", "verdict", "reason", "crops"} <= set(p)
        assert p["verdict"] in ("out", "safe")
        for c in p["crops"]:
            assert c is None or (d / c).exists()
    for f in r["frames"]:
        im = Image.open(d / f)
        assert max(im.size) <= 640 and im.format == "JPEG"
    assert any((d / "crops").iterdir())


def test_recorder_captures_errors_with_tracebacks(session):
    try:
        raise ValueError("SAM fell over")
    except ValueError as e:
        session.error("vision", e, round=3)
    try:
        1 / 0
    except ZeroDivisionError:
        logging.getLogger("simon_sees.server").exception("game crashed")
    errs = [e for e in read_jsonl(session.dir / "events.jsonl") if e["event"] == "error"]
    assert errs[0]["message"] == "ValueError: SAM fell over" and "Traceback" in errs[0]["traceback"]
    assert errs[1]["via"] == "log" and "ZeroDivisionError" in errs[1]["traceback"]
    assert "game crashed" in (session.dir / "server.log").read_text()


def test_record_off_by_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMON_RECORD", "0")
    rec = make_recorder(root=tmp_path / "s")
    assert isinstance(rec, NullRecorder) and not (tmp_path / "s").exists()
    play(rec, rounds=2)  # the game runs the same without it


# --------------------------------------------------------------------------- feedback endpoint (and the server's session)


def _server_game(tmp_path, monkeypatch, **cfg_kw):
    monkeypatch.setattr("simon_sees.server.estimate_speech_s", lambda t: 0.0)
    cfg = fast_cfg(tmp_path, frame_timeout_s=0.05, sessions_dir=str(tmp_path / "sessions"), **cfg_kw)
    return create_and_play(cfg)


def create_and_play(cfg):
    from simon_sees.server import create_app

    app = create_app(cfg, vision=band_vision())
    client = TestClient(app)
    client.__enter__()
    ws = client.websocket_connect("/ws").__enter__()
    ws.receive_text()
    ws.send_bytes(jpeg())
    ws.send_text(json.dumps({"type": "lineup"}))
    until(ws, lambda m: m["type"] == "lineup" and m.get("status") == "done", feed=lambda: ws.send_bytes(jpeg()))
    ws.send_text(json.dumps({"type": "start", "names": {"0": "Kofi"}}))
    over, msgs = until(ws, lambda m: m["type"] == "state" and m.get("phase") == "over", feed=lambda: ws.send_bytes(jpeg()))
    return app, client, ws, over


def test_feedback_endpoint_links_corrections_to_rounds(tmp_path, monkeypatch):
    app, c, ws, over = _server_game(tmp_path, monkeypatch)
    try:
        rec = app.state.st["rec"]
        r = over["last_round"]
        d = r["decisions"][0]
        assert c.get("/api/info").json()["session"] == rec.dir.name
        res = c.post("/api/feedback", json={"kind": "wrong_call", "round_id": r["round_id"], "pid": d["pid"],
                                            "correct": "safe" if d["out"] else "out", "where": "verdict"})
        assert res.status_code == 200 and res.json()["recorded"] is True
        assert c.post("/api/feedback", json={"kind": "note", "text": "  the squat one was harsh  "}).status_code == 200
        # bad input is refused, not written
        assert c.post("/api/feedback", json={"kind": "wrong_call", "round_id": "nope-r01", "pid": 0, "correct": "out"}).status_code == 404
        assert c.post("/api/feedback", json={"kind": "wrong_call", "round_id": r["round_id"], "pid": 0, "correct": "maybe"}).status_code == 422
        assert c.post("/api/feedback", json={"kind": "note", "text": "  "}).status_code == 422
        assert c.post("/api/feedback", json={"kind": "rating"}).status_code == 422
        fb = list(read_jsonl(rec.dir / "feedback.jsonl"))
        assert len(fb) == 2
        assert fb[0]["round_id"] == r["round_id"] and fb[0]["given"] == ("out" if d["out"] else "safe")
        assert fb[0]["command"] == r["command"] and fb[0]["player"] == d["player"]
        assert fb[1] == fb[1] | {"kind": "note", "text": "the squat one was harsh"}
        rounds = [e for e in read_jsonl(rec.dir / "events.jsonl") if e["event"] == "round"]
        assert r["round_id"] in {e["round_id"] for e in rounds}
        assert any(e["event"] == "lineup" and e["frame"] for e in read_jsonl(rec.dir / "events.jsonl"))
        assert any(e["event"] == "say" for e in read_jsonl(rec.dir / "events.jsonl"))
    finally:
        ws.__exit__(None, None, None)
        c.__exit__(None, None, None)
    meta = json.loads((rec.dir / "session.json").read_text())
    assert meta["models"] == {"segmenter": "mock", "embedder": "mock", "decider": "mock"} and meta["closed"]


def test_retracted_corrections_drop_out():
    fb = [{"kind": "wrong_call", "round_id": "g-r01", "pid": 0, "correct": "safe"},
          {"kind": "wrong_call", "round_id": "g-r01", "pid": 1, "correct": "out"},
          {"kind": "wrong_call", "round_id": "g-r01", "pid": 0, "correct": "safe", "retracted": True}]
    assert list(standing_corrections(fb)) == [("g-r01", 1)]


def test_auto_upload_at_game_end(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("simon_sees.upload.upload_sessions", lambda *a, **k: calls.append(a) or {"uploaded": []})
    app, c, ws, _ = _server_game(tmp_path, monkeypatch, upload="hf", hf_repo="me/pt")
    try:
        t = app.state.st["uploading"]
        for _ in range(100):
            if calls:
                break
            import time

            time.sleep(0.05)
        assert t is not None and calls and calls[0][1] == "me/pt" and calls[0][3] == [app.state.st["rec"].dir]
    finally:
        ws.__exit__(None, None, None)
        c.__exit__(None, None, None)


# --------------------------------------------------------------------------- bundle, export, report, pull


def test_bundle_zips_sessions(session, tmp_path):
    play(session, rounds=2)
    (session.dir / MARKER).write_text("{}")
    out = make_bundle(session.dir.parent, tmp_path / "b.zip")
    names = zipfile.ZipFile(out).namelist()
    assert f"sessions/{session.dir.name}/events.jsonl" in names and f"sessions/{session.dir.name}/session.json" in names
    assert any("/frames/" in n for n in names) and not any(n.endswith(MARKER) for n in names)
    with pytest.raises(RuntimeError):
        make_bundle(tmp_path / "empty")


def test_export_labels_model_and_human(session, tmp_path):
    game = play(session, rounds=3)
    rounds = [e for e in read_jsonl(session.dir / "events.jsonl") if e["event"] == "round"]
    r = next(x for x in rounds if x["kind"] != "freeze")
    p = next(x for x in r["players"] if x["seen"])
    correct = "safe" if p["verdict"] == "out" else "out"
    session.feedback(kind="wrong_call", round_id=r["round_id"], pid=p["pid"], given=p["verdict"], correct=correct)
    ex = script("export_labels")
    out, counts = ex.export([str(session.dir.parent)], tmp_path / "data" / "labels.jsonl")
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert counts["human_correction"] >= 1 and counts["model"] >= 1
    human = [x for x in rows if x["source"] == "human_correction"]
    assert {x["round_id"] for x in human} == {r["round_id"]}
    want = answer_from_verdict(r["kind"], r["simon"], correct)
    assert all(x["label"] is want and x["answer"] == ("yes" if want else "no") for x in human)
    # the corrected player's model rows are replaced, not duplicated
    assert not [x for x in rows if x["source"] == "model" and x["round_id"] == r["round_id"] and x["pid"] == p["pid"]]
    for x in rows:
        assert {"image", "command", "label", "source"} <= set(x)
        assert (out.parent / x["image"]).exists()
    out2, c2 = ex.export([str(session.dir)], tmp_path / "h.jsonl", human_only=True)
    assert c2["model"] == 0 and c2["human_correction"] == counts["human_correction"]
    assert game.id


def test_answer_from_verdict_semantics():
    assert answer_from_verdict("pose", True, "out") is False      # Simon said, out: didn't do it
    assert answer_from_verdict("pose", False, "out") is True      # trick, out: did it
    assert answer_from_verdict("freeze", True, "out") is True     # freeze, out: moved
    assert answer_from_verdict("pose", True, "safe") is True


def test_report_and_pull_from_bundle(session, tmp_path):
    play(session, rounds=4)
    r = next(e for e in read_jsonl(session.dir / "events.jsonl") if e["event"] == "round")
    p = r["players"][0]
    session.feedback(kind="wrong_call", round_id=r["round_id"], pid=p["pid"], player=p["name"], given=p["verdict"],
                     correct="safe" if p["verdict"] == "out" else "out")
    session.feedback(kind="note", text="Simon couldn't see Ama in the shade", game_id="x")
    session.error("vision", RuntimeError("boom"))
    zip_path = make_bundle(session.dir.parent, tmp_path / "bundle.zip")
    pull = script("pull_playtests")
    assert pull.main(["--zip", str(zip_path), "--out", str(tmp_path / "pulled")]) == 0
    assert load_session(tmp_path / "pulled" / "sessions" / session.dir.name).rounds
    rep = script("playtest_report")
    md = rep.report(rep.load_sessions([tmp_path / "pulled"]))
    for heading in ("## Accuracy", "## Per-command error rates", "## Latency per stage", "## Identity",
                    "## Crashes and errors", "## The corrected calls", "## The players' notes"):
        assert heading in md
    assert "Simon couldn't see Ama in the shade" in md and "RuntimeError: boom" in md
    assert "1 standing 'Wrong call'" in md and "| DINO (embed) |" in md and "| mock |" in md


def test_swap_heuristic():
    rep = script("playtest_report")
    from simon_sees.playtest_data import Session

    A, B = [0, 0, 100, 300], [200, 0, 300, 300]
    ev = [{"event": "game_start"},
          {"event": "locate", "game_id": "g", "players": {"0": {"seen": True, "box": A}, "1": {"seen": True, "box": B}}},
          {"event": "round", "round_id": "g-r01", "players": [{"pid": 0, "box_per_frame": [B]}, {"pid": 1, "box_per_frame": [A]}]}]
    swaps = rep.suspected_swaps(Session(Path("."), {}, ev, []))
    assert len(swaps) == 1 and swaps[0]["pids"] == (0, 1)


# --------------------------------------------------------------------------- upload (huggingface_hub mocked)


class FakeApi:
    def __init__(self, private=True, readme=False):
        self.private, self.readme = private, readme
        self.repos, self.commits = [], []

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        self.repos.append(dict(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=exist_ok))

    def repo_info(self, repo_id, repo_type=None):
        return SimpleNamespace(private=self.private)

    def file_exists(self, repo_id, filename, repo_type=None):
        return self.readme

    def create_commit(self, repo_id, repo_type=None, operations=(), commit_message=""):
        self.commits.append(dict(repo_id=repo_id, repo_type=repo_type, paths=[o.path_in_repo for o in operations]))


def test_upload_creates_a_private_repo_and_only_sends_new_files(tmp_path):
    # no log capture here: a live server.log keeps changing, and would (rightly) be sent again
    session = Recorder(tmp_path / "up", capture_log=False)
    play(session, rounds=2)
    root = session.dir.parent
    api = FakeApi()
    res = upload_sessions(root, "me/simon-sees-playtests", token="hf_x", api=api)
    assert api.repos == [dict(repo_id="me/simon-sees-playtests", repo_type="dataset", private=True, exist_ok=True)]
    paths = api.commits[0]["paths"]
    assert "README.md" in paths and f"sessions/{session.dir.name}/events.jsonl" in paths
    assert api.commits[0]["repo_type"] == "dataset" and not any(MARKER in p for p in paths)
    assert res["uploaded"][0]["session"] == session.dir.name and (session.dir / MARKER).exists()
    # nothing new: no commit; then a note arrives and only feedback.jsonl goes up
    api2 = FakeApi(readme=True)
    assert upload_sessions(root, "me/simon-sees-playtests", token="hf_x", api=api2)["uploaded"] == []
    assert api2.commits == []
    session.feedback(kind="note", text="more please")
    assert changed_files(session.dir) == ["feedback.jsonl"]
    upload_sessions(root, "me/simon-sees-playtests", token="hf_x", api=api2)
    assert api2.commits[0]["paths"] == [f"sessions/{session.dir.name}/feedback.jsonl"]


def test_upload_refuses_a_public_repo(session):
    play(session, rounds=1)
    api = FakeApi(private=False)
    with pytest.raises(RuntimeError, match="PUBLIC"):
        upload_sessions(session.dir.parent, "me/pt", token="hf_x", api=api)
    assert api.commits == [] and not (session.dir / MARKER).exists()


def test_upload_cli_uses_hf_token(session, monkeypatch, capsys):
    play(session, rounds=1)
    seen = {}

    class Api(FakeApi):
        def __init__(self, token=None):
            super().__init__()
            seen["token"] = token
            seen["api"] = self

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    from simon_sees.__main__ import main

    assert main(["upload", "--sessions", str(session.dir.parent), "--repo", "me/pt"]) == 0
    assert seen["token"] == "hf_from_env" and seen["api"].commits
    assert "https://huggingface.co/datasets/me/pt" in capsys.readouterr().out
    assert list_sessions(session.dir.parent) == [session.dir]
