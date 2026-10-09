"""Server smoke tests: page served, line-up over the WebSocket returns photo cards, a full mocked game runs
(phone frames and replay), photos are served per player."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

from conftest import scene
from fastapi.testclient import TestClient

from simon_sees.decider import MockDecider
from simon_sees.engine import Settings
from simon_sees.pipeline import LocalVision, Stack
from simon_sees.server import Config, create_app
from simon_sees.vision import MockEmbedder, MockSegmenter

PEOPLE = Path(__file__).resolve().parents[1] / "samples" / "people"


def jpeg(img=None) -> bytes:
    buf = io.BytesIO()
    (img or scene(["a", "b", "c"])).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def fast_cfg(tmp_path, **kw) -> Config:
    s = Settings(reaction_s=0, frame_gap_s=0, freeze_grace_s=0, freeze_window_s=0, rest_s=0, max_rounds=3, seed=1)
    return Config(decider="mock", tts="browser", stt=False, log_path=str(tmp_path / "d.jsonl"), settings=s, **kw)


def band_vision(seed=3):
    return LocalVision(Stack(MockSegmenter(n=3), MockEmbedder(), MockDecider(seed=seed)))


def until(ws, pred, deadline=20, feed=None):
    end = time.time() + deadline
    msgs = []
    while time.time() < end:
        if feed:
            feed()
        m = json.loads(ws.receive_text())
        msgs.append(m)
        if pred(m):
            return m, msgs
    raise AssertionError(f"timed out; got {[x.get('type') for x in msgs]}")


def test_index_and_info(tmp_path):
    app = create_app(fast_cfg(tmp_path), vision=band_vision())
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "Line up" in r.text
        assert "Player description" not in r.text
        assert c.get("/static/app.js").status_code == 200
        info = c.get("/api/info").json()
        assert info["models"] == {"segmenter": "mock", "embedder": "mock", "decider": "mock"}
        assert info["backend"] == "local" and info["tts"] == "browser"


def test_start_without_lineup_is_refused(tmp_path):
    app = create_app(fast_cfg(tmp_path), vision=band_vision())
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "start"}))
        m, _ = until(ws, lambda m: m["type"] == "error")
        assert "line up first" in m["message"]


def test_lineup_then_game_with_phone_frames(tmp_path, monkeypatch):
    monkeypatch.setattr("simon_sees.server.estimate_speech_s", lambda t: 0.0)
    # frames arrive as JPEG bytes like a real phone's; the band segmenter stands in for SAM
    app = create_app(fast_cfg(tmp_path, frame_timeout_s=0.05), vision=band_vision())
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["type"] == "info"
        ws.send_bytes(jpeg())
        ws.send_text(json.dumps({"type": "lineup"}))
        m, _ = until(ws, lambda m: m["type"] == "lineup" and m.get("status") == "done", feed=lambda: ws.send_bytes(jpeg()))
        assert [p["number"] for p in m["players"]] == [1, 2, 3]
        photo = c.get(m["players"][0]["photo"])
        assert photo.status_code == 200 and photo.headers["content-type"] == "image/jpeg"
        ws.send_text(json.dumps({"type": "start", "names": {"0": "Kofi"}, "skip": [2]}))
        said = []

        def pred(msg):
            if msg["type"] == "say":
                said.append(msg["text"])
            return msg["type"] == "state" and msg.get("phase") == "over"

        over, msgs = until(ws, pred, feed=lambda: ws.send_bytes(jpeg()))
        assert said[0] == "Welcome to Simon Sees. Players: Kofi and Number 2. Only do it if Simon says. Let's look at you."
        assert [p["display_name"] for p in over["players"]] == ["Kofi", "Player 2"]
        assert c.get("/api/photo/0.jpg").status_code == 200
    rows = [json.loads(x) for x in (tmp_path / "d.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "lineup" and rows[-1]["event"] == "game_end"


def test_replay_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr("simon_sees.server.estimate_speech_s", lambda t: 0.0)
    cfg = fast_cfg(tmp_path, replay=str(PEOPLE), replay_fps=30)
    app = create_app(cfg, vision=band_vision(seed=5))
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["replay"] is True
        ws.send_text(json.dumps({"type": "lineup"}))
        m, _ = until(ws, lambda m: m["type"] == "lineup" and m.get("status") == "done")
        assert len(m["players"]) == 3
        ws.send_text(json.dumps({"type": "start", "names": {}}))
        until(ws, lambda m: m["type"] == "state" and m.get("phase") == "over")


def test_hud_rings_get_boxes_and_the_judged_frame(tmp_path, monkeypatch):
    """The phone draws a ring per player from state players[*].box over the frame at state["view"]["url"]."""
    monkeypatch.setattr("simon_sees.server.estimate_speech_s", lambda t: 0.0)
    cfg = fast_cfg(tmp_path, replay=str(PEOPLE), replay_fps=30)
    app = create_app(cfg, vision=band_vision(seed=5))
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.receive_text()
        before = c.get("/api/view/l0.jpg")  # replay lobby: the first replay frame, before anything was looked at
        assert before.status_code == 200 and before.headers["content-type"] == "image/jpeg"
        ws.send_text(json.dumps({"type": "lineup"}))
        m, _ = until(ws, lambda m: m["type"] == "lineup" and m.get("status") == "done")
        assert m["view"]["w"] > 0 and m["view"]["h"] > 0 and c.get(m["view"]["url"]).status_code == 200
        assert all(len(p["box"]) == 4 for p in m["players"])
        ws.send_text(json.dumps({"type": "start", "names": {}}))
        st, _ = until(ws, lambda m: m["type"] == "state" and (m.get("view") or {}).get("seq", 0) >= 1)
        assert all(p["box"] and len(p["box"]) == 4 for p in st["players"])
        w, h = st["view"]["w"], st["view"]["h"]
        assert all(0 <= p["box"][0] < p["box"][2] <= w and 0 <= p["box"][1] < p["box"][3] <= h for p in st["players"])
        r = c.get(st["view"]["url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        until(ws, lambda m: m["type"] == "state" and m.get("phase") == "over")
