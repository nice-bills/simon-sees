"""The server's remote-vision client against a mocked HTTP endpoint that speaks deploy/modal_app.py's API
(GET /info, POST /batch). The fake endpoint runs process_batch on the mocked stack, so a whole game can be
played through it."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import ScriptedIO, scene, scene_segmenter

from simon_sees.decider import MockDecider
from simon_sees.engine import Game, Settings, lineup, players_from_lineup
from simon_sees.identity import Identity
from simon_sees.pipeline import RemoteVision, Stack, from_b64, process_batch
from simon_sees.server import Config, build_vision
from simon_sees.vision import MockEmbedder


class FakeEndpoint:
    def __init__(self, require: str | None = "Bearer wk-1.ws-1", fail: int | None = None):
        self.stack = Stack(scene_segmenter(), MockEmbedder(), MockDecider(seed=3))
        self.calls: list[tuple[str, dict]] = []
        self.require, self.fail = require, fail
        ep = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _auth_ok(self):
                return ep.require is None or self.headers.get("Authorization") == ep.require

            def do_GET(self):
                ep.calls.append(("GET " + self.path, dict(self.headers)))
                if not self._auth_ok():
                    return self._reply(401, {"detail": "missing credentials"})
                self._reply(200, {"models": {"segmenter": "facebook/sam3.1", "embedder": "dinov3-test",
                                             "decider": "LiquidAI/d1-3B"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                ep.calls.append(("POST " + self.path, body))
                if not self._auth_ok():
                    return self._reply(401, {"detail": "missing credentials"})
                if ep.fail:
                    return self._reply(ep.fail, {"detail": "boom"})
                # the boxes ride along in img.info for the scene segmenter; JPEG loses that, so the
                # test sends them in the task (the real endpoint runs SAM instead)
                frames = [from_b64(f) for f in body["frames"]]
                for f, b in zip(frames, body["task"].pop("_boxes")):
                    f.info["boxes"] = [(w, tuple(x)) for w, x in b]
                idents = [Identity.from_json(d) for d in body["identities"]]
                out = process_batch(ep.stack, body["task"], frames, idents)
                out["models"] = {"segmenter": "facebook/sam3.1", "embedder": "dinov3-test", "decider": "LiquidAI/d1-3B"}
                self._reply(200, out)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


class BoxCarryingRemote(RemoteVision):
    """Adds the synthetic scenes' boxes to the task so the fake endpoint can 'segment' them."""

    def run(self, task, frames, identities=()):
        task = dict(task, _boxes=[[(w, list(b)) for w, b in f.info.get("boxes", [])] for f in frames])
        return super().run(task, frames, identities)


@pytest.fixture
def endpoint(monkeypatch):
    for k in ("NO_PROXY", "no_proxy"):  # a local HTTP proxy must not swallow 127.0.0.1
        monkeypatch.setenv(k, "127.0.0.1,localhost")
    ep = FakeEndpoint()
    yield ep
    ep.close()


def test_needs_a_url(monkeypatch):
    monkeypatch.delenv("SIMON_VISION_URL", raising=False)
    monkeypatch.delenv("SIMON_MODAL_URL", raising=False)
    with pytest.raises(RuntimeError, match="SIMON_VISION_URL"):
        RemoteVision()


def test_server_builds_the_remote_client_from_env(monkeypatch, endpoint):
    monkeypatch.setenv("SIMON_VISION_URL", endpoint.url + "/")
    monkeypatch.setenv("SIMON_VISION_TOKEN", "wk-1.ws-1")
    v = build_vision(Config(backend="modal"))
    assert isinstance(v, RemoteVision) and v.url == endpoint.url and v.backend == "modal"
    assert v.models["segmenter"] == "facebook/sam3.1"
    assert endpoint.calls[0][0] == "GET /info"
    assert endpoint.calls[0][1]["Authorization"] == "Bearer wk-1.ws-1"


def test_modal_key_secret_headers(endpoint):
    endpoint.require = None
    v = RemoteVision(endpoint.url, key="wk-2", secret="ws-2")
    v.models
    h = endpoint.calls[0][1]
    assert h["Modal-Key"] == "wk-2" and h["Modal-Secret"] == "ws-2" and "Authorization" not in h


def test_unreachable_info_degrades_but_batch_errors_are_loud(endpoint):
    v = BoxCarryingRemote(endpoint.url, token="wrong")
    assert "unreachable" in v.models["segmenter"]
    with pytest.raises(RuntimeError, match="HTTP 401"):
        v.run({"kind": "enrol"}, [scene(["a"])])


def test_enrol_and_reid_round_trip_over_http(endpoint):
    v = BoxCarryingRemote(endpoint.url, token="wk-1.ws-1")
    r = v.run({"kind": "enrol"}, [scene(["a", "b", "c"])])
    assert len(r["people"]) == 3 and "round_trip" in r["latency_ms"]
    _, roster = players_from_lineup(r["people"], [])
    r2 = v.run({"kind": "observe"}, [scene(["b", "c", "a"])], list(roster.identities.values()))
    body = endpoint.calls[-1][1]
    assert len(body["identities"]) == 3 and len(body["identities"][0]["gallery"][0]) > 0
    seen = r2["frames"][0]["players"]
    order = sorted(((p["box"][0] + p["box"][2]) / 2, int(k)) for k, p in seen.items() if p["seen"])
    assert [k for _, k in order] == [1, 2, 0]


def test_a_whole_game_through_the_remote_endpoint(endpoint, fast, tmp_path):
    v = BoxCarryingRemote(endpoint.url, token="wk-1.ws-1")
    io = ScriptedIO([scene(["a", "b", "c"])])

    async def go():
        res = await lineup(v, io)
        players, roster = players_from_lineup(res["people"], ["Kofi", "Ama", ""])
        game = Game(v, io, players, roster, Settings(max_rounds=3, seed=2, **fast))
        return await game.run()

    winners = asyncio.run(go())
    assert winners is not None
    kinds = {c[1]["task"]["kind"] for c in endpoint.calls if c[0] == "POST /batch"}
    assert "enrol" in kinds and kinds & {"pose", "freeze", "observe"}
