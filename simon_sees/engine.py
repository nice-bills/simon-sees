"""The game: line up, then Simon calls, waits a beat, looks, and decides who's out.

Players are not described by what they wear. At the line-up, SAM (text prompt "person") finds everyone
in front of the phone, DINOv3 embeds each person's crop, and each crop becomes that player's photo on the
team sheet ("Player 1..N" left to right, or the name they said/typed). Every round the same pipeline
finds the people again, matches them to the enrolled players (cosine similarity + Hungarian assignment +
box tracking, identity.py), and the decision model judges each player's OWN crop ("Is this person
standing on one foot?"; freeze = that player's BEFORE and AFTER crops). See pipeline.py.

The engine never touches the network, the camera or the speaker directly; it talks to a `GameIO` (the web
server, the dry-run replayer, or a test fake) and a vision backend (local or Modal). Rules are in
`judge()`, a pure function, so they can be tested without frames or models.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from . import tracing
from .commands import Call, CommandPicker
from .decision_log import DecisionLog
from .identity import Identity, Roster
from .motion import moved_probability, union_box
from .recorder import NullRecorder
from .vision import Detection, crop_person

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- data

DEFAULT_NAME = re.compile(r"^player\s*\d+$", re.I)


@dataclass
class Player:
    id: int  # stable id from the line-up (0-based, left to right)
    number: int  # their number on the team sheet (1-based, left to right at the line-up)
    name: str = ""  # what they said or typed; empty -> "Player N" on the sheet, "Number N" when Simon calls
    photo: str | None = None  # their line-up crop, JPEG base64 (served by the server, never in state broadcasts)
    alive: bool = True
    out_round: int | None = None
    out_reason: str = ""
    unseen_rounds: int = 0
    seen: bool = True
    last_p: float | None = None
    last_sim: float | None = None
    box: list[float] | None = None  # UI rings: where they were last seen, x0,y0,x1,y1 px of the view frame (state["view"])

    @property
    def named(self) -> bool:
        return bool(self.name.strip()) and not DEFAULT_NAME.match(self.name.strip())

    @property
    def display_name(self) -> str:
        return self.name.strip() if self.named else f"Player {self.number}"

    @property
    def call_name(self) -> str:
        """How Simon addresses them: 'Kofi', or 'Number 2' for someone who didn't give a name."""
        return self.name.strip() if self.named else f"Number {self.number}"


@dataclass
class Frame:
    image: Any  # PIL.Image
    ts: float = field(default_factory=time.time)


@dataclass
class Settings:
    max_rounds: int = 12
    reaction_s: float = 2.2      # beat between the call and the first look
    pose_frames: int = 2         # frames judged per pose/touch round
    frame_gap_s: float = 0.5
    freeze_grace_s: float = 0.9  # time to stop moving after "freeze"
    freeze_window_s: float = 1.6  # gap between the BEFORE and AFTER frames
    rest_s: float = 1.2          # "Relax" pause between rounds
    out_below: float = 0.35      # Simon said it: out if P(did it) is below this
    trick_out_above: float = 0.65  # Simon didn't say it: out if P(did it) is above this
    moved_above: float = 0.65    # freeze: out if P(moved) is above this
    aggregate: str = "mean"      # how per-frame probabilities combine: mean | max | min
    freeze_strategy: str = "pair"  # pair (d1 sees the player's before+after crops) | pixel (fallback, no AI)
    pixel_lo: float = 0.01
    pixel_hi: float = 0.06
    trick_rate: float = 0.25
    freeze_rate: float = 0.2
    outdoors: bool = True
    seed: int | None = None
    save_frames: str | None = None  # folder: keep judged frames locally to label later (bench/)


@dataclass
class Verdict:
    player_id: int
    out: bool
    reason: str  # ok | did_not_obey | simon_didnt_say | moved | not_visible
    p: float | None  # P(did the action) or P(moved)
    seen: bool
    uncertain: bool = False


class GameIO(Protocol):
    async def say(self, text: str, mood: str = "info") -> None: ...
    async def frames(self, n: int, gap_s: float) -> list[Frame]: ...
    async def update(self, state: dict) -> None: ...


class Vision(Protocol):
    backend: str
    models: dict

    def run(self, task: dict, frames: Sequence[Any], identities: Sequence[Identity] = ()) -> dict: ...


# --------------------------------------------------------------------------- rules


def combine(ps: Sequence[float | None], how: str = "mean") -> float:
    ps = [p for p in ps if p is not None]
    if not ps:
        return 0.0
    if how == "max":
        return max(ps)
    if how == "min":
        return min(ps)
    return sum(ps) / len(ps)


def judge(call: Call, p_action: float, seen: bool, s: Settings) -> tuple[bool, str, bool]:
    """(out, reason, uncertain) for one player. Uncertain answers never eliminate: a referee who
    is not sure gives the benefit of the doubt. Someone the cameras couldn't find is never out."""
    if not seen:
        return False, "not_visible", True
    if call.command.kind == "freeze":
        out = p_action >= s.moved_above
        return out, ("moved" if out else "ok"), (1 - s.moved_above) <= p_action < s.moved_above
    if call.simon:
        out = p_action < s.out_below
        return out, ("did_not_obey" if out else "ok"), s.out_below <= p_action < 0.5
    out = p_action >= s.trick_out_above
    return out, ("simon_didnt_say" if out else "ok"), 0.5 <= p_action < s.trick_out_above


def out_line(p: Player, call: Call, reason: str) -> str:
    if reason == "simon_didnt_say":
        return f"{p.call_name}, Simon didn't say! You're out."
    phrase = call.command.out_phrase or "didn't do it"
    return f"{p.call_name}, you {phrase}. You're out."


SAFE_LINES = ["Nice. Everyone's still in.", "Good. Nobody's out.", "Sharp. You all survive.", "Still all here. Impressive."]
TRICK_SAFE = ["Ha. I didn't say Simon says, and none of you fell for it.", "Nobody fell for it. Respect."]


def join_names(names: Sequence[str]) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


# --------------------------------------------------------------------------- line-up


def players_from_lineup(people: Sequence[dict], names: dict[int, str] | Sequence[str] | None = None,
                        keep: Sequence[int] | None = None) -> tuple[list[Player], Roster]:
    """Turn an `enrol` result (people already ordered left to right) into players + their identities.
    `names` maps line-up index -> name (or a list, in order); `keep` drops people who aren't playing."""
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}
    names = names or {}
    idx = [i for i in range(len(people)) if keep is None or i in keep]
    players, roster = [], Roster()
    for number, i in enumerate(idx, start=1):
        person = people[i]
        p = Player(id=i, number=number, name=(names.get(i) or "").strip()[:24], photo=person.get("photo"),
                   box=[float(v) for v in person["box"]] if person.get("box") else None)
        players.append(p)
        roster.identities[i] = Identity(i, [np.asarray(person["emb"], dtype=np.float32)],
                                        tuple(person["box"]) if person.get("box") else None)
    return players, roster


# --------------------------------------------------------------------------- game


class Game:
    def __init__(self, vision: Vision, io: GameIO, players: Sequence[Player], roster: Roster,
                 settings: Settings | None = None, decision_log: DecisionLog | None = None,
                 picker: CommandPicker | None = None, game_id: str | None = None, recorder=None):
        if not players:
            raise ValueError("need at least one player: line up in front of the camera first")
        self.s = settings or Settings()
        self.vision = vision
        self.io = io
        self.players = list(players)
        self.roster = roster
        self.log = decision_log or DecisionLog(None)
        self.picker = picker or CommandPicker(seed=self.s.seed, trick_rate=self.s.trick_rate,
                                              outdoors=self.s.outdoors, freeze_rate=self.s.freeze_rate)
        self.rng = random.Random(self.s.seed)
        self.id = game_id or uuid.uuid4().hex[:8]
        self.round = 0
        self.phase = "lobby"
        self.call: Call | None = None
        self.message = ""
        self.winners: list[Player] = []
        self.history: list[dict] = []
        self.rounds: dict[str, dict] = {}  # round_id -> last_round, so feedback can name a past round
        self.latencies: list[float] = []  # decision-model time per batch (ms)
        self.last_breakdown: dict = {}
        self.last_round: dict = {}
        self._stop = False
        self.rec = recorder or NullRecorder()  # playtest session recorder (recorder.py)
        self._ctx: dict = {}  # the last look of this round: frames + vision result, for the recorder
        self.view_frame: Frame | None = None  # UI: the frame the players' boxes refer to (served as /api/view/<seq>.jpg)
        self.view_seq = 0

    @property
    def round_id(self) -> str:
        """Links a round across events.jsonl, frames/, crops/ and the players' feedback."""
        return f"{self.id}-r{self.round:02d}"

    @property
    def models(self) -> dict:
        return dict(getattr(self.vision, "models", {}) or {})

    # ---- state for the UI
    def state(self) -> dict:
        m = self.models
        return {
            "game_id": self.id, "phase": self.phase, "round": self.round, "max_rounds": self.s.max_rounds,
            "call": self.call.text if self.call else None, "simon": self.call.simon if self.call else None,
            "message": self.message, "model": m.get("decider") or "?", "models": m,
            "backend": getattr(self.vision, "backend", "local"),
            "players": [{k: v for k, v in asdict(p).items() if k != "photo"}
                        | {"call_name": p.call_name, "display_name": p.display_name,
                           "photo": f"/api/photo/{p.id}.jpg" if p.photo else None} for p in self.players],
            "winners": [p.display_name for p in self.winners],
            "last_latency_ms": round(self.latencies[-1], 1) if self.latencies else None,
            "latency_breakdown": self.last_breakdown,
            "last_round": self.last_round,
            # UI rings: players[*].box are pixels of this frame (w x h); seq changes whenever the frame does
            "view": {"seq": self.view_seq, "w": self.view_frame.image.size[0], "h": self.view_frame.image.size[1],
                     "url": f"/api/view/{self.view_seq}.jpg"} if self.view_frame is not None else None,
        }

    async def _publish(self) -> None:
        await self.io.update(self.state())

    async def _say(self, text: str, mood: str = "info") -> None:
        self.message = text
        await self._publish()
        await self.io.say(text, mood)

    def stop(self) -> None:
        self._stop = True

    @property
    def alive(self) -> list[Player]:
        return [p for p in self.players if p.alive]

    def _player(self, pid: int | str) -> Player:
        return next(p for p in self.players if p.id == int(pid))

    async def _look(self, task: dict, frames: Sequence[Frame], players: Sequence[Player]) -> dict:
        idents = [self.roster.identities[p.id] for p in players]
        t0 = time.perf_counter()
        # to_thread copies contextvars, so Sentry spans nest under the round's transaction
        with tracing.vision_span(task.get("kind", "observe"), len(frames), len(players)):
            res = await asyncio.to_thread(self.vision.run, task, [f.image for f in frames], idents)
        self._ctx = {"frames": list(frames), "res": res, "look_ms": round(1e3 * (time.perf_counter() - t0), 1)}
        self.roster.apply(res.get("updates", {}))
        lat = res.get("latency_ms", {})
        self.last_breakdown = lat
        if task.get("kind") in ("pose", "freeze") and lat.get("decide"):
            self.latencies.append(float(lat["decide"]))
        for fr in res.get("frames", []):
            for pid, info in (fr.get("players") or {}).items():
                p = self._player(pid)
                p.last_sim = info.get("sim")
        try:  # UI rings: the last frame of the look, and each player's box in it (kept from before if not seen)
            last = (res.get("frames") or [{}])[-1].get("players") or {}
            for pid, info in last.items():
                if info.get("seen") and info.get("box"):
                    self._player(pid).box = [float(v) for v in info["box"]]
            if frames:
                self.view_frame, self.view_seq = frames[-1], self.view_seq + 1
        except Exception as e:  # noqa: BLE001 - the HUD must never break a round
            log.debug("view frame not updated: %s", e)
        return res

    def _save(self, frames: Sequence[Frame], tag: str) -> list[str]:
        if not self.s.save_frames:
            return []
        folder = Path(self.s.save_frames) / self.id
        folder.mkdir(parents=True, exist_ok=True)
        names = []
        for i, f in enumerate(frames):
            name = f"r{self.round:02d}_{tag}_{i}.jpg"
            f.image.convert("RGB").save(folder / name, quality=88)
            names.append(str(folder / name))
        return names

    @staticmethod
    def _seen(res: dict, pid: int) -> list[bool]:
        return [bool((fr.get("players") or {}).get(str(pid), {}).get("seen")) for fr in res.get("frames", [])]

    # ---- phases
    async def locate(self, attempts: int = 2) -> list[Player]:
        """Find every player once before the first call. Returns the ones it can't see."""
        missing: list[Player] = []
        for attempt in range(attempts):
            frames = await self.io.frames(1, 0)
            res = await self._look({"kind": "observe"}, frames, self.players)
            missing = []
            for p in self.players:
                p.seen = any(self._seen(res, p.id))
                if not p.seen:
                    missing.append(p)
                fr = res["frames"][0]["players"].get(str(p.id), {})
                self.log.write("locate", game_id=self.id, player=p.display_name, pid=p.id, seen=p.seen,
                               sim=fr.get("sim"), box=fr.get("box"), detections=res["frames"][0].get("detections"),
                               latency_ms=res.get("latency_ms"), models=res.get("models"), attempt=attempt,
                               frames=self._save(frames, f"locate{attempt}") if p is self.players[0] else None)
            self.rec.event("locate", game_id=self.id, attempt=attempt, latency_ms=res.get("latency_ms"),
                           detections=res["frames"][0].get("detections"),
                           players={str(p.id): {"name": p.display_name, "number": p.number,
                                                **(res["frames"][0]["players"].get(str(p.id), {}))} for p in self.players},
                           frame=self.rec.frame(frames[0].image, f"{self.id}-locate{attempt}.jpg"))
            await self._publish()
            if not missing:
                return []
            await self._say(f"I can't see {join_names([p.call_name for p in missing])}. Step into the picture.", "warn")
            await asyncio.sleep(2.0)
        return missing

    async def play_round(self) -> list[Player]:
        self.round += 1
        call = self.picker.next(self.round)
        self.call = call
        self.phase = "calling"
        alive = self.alive
        t_round = time.perf_counter()
        self._ctx = {}
        with tracing.round_span(self.id, self.round, call.text):
            await self._say(call.text, "call")
            tts_call_ms = getattr(self.io, "last_tts_ms", None)
            self.phase = "watching"
            await self._publish()
            evidence = await (self._watch_freeze(call, alive) if call.command.kind == "freeze"
                              else self._watch_pose(call, alive))
        self.phase = "judging"
        verdicts: dict[int, Verdict] = {}
        for p in alive:
            pa, seen = evidence[p.id]
            is_out, reason, unsure = judge(call, pa, seen, self.s)
            verdicts[p.id] = Verdict(p.id, is_out, reason, p=pa if seen else None, seen=seen, uncertain=unsure)
        outs = [p for p in alive if verdicts[p.id].out]
        voided = False
        if outs and len(outs) == len(alive) and len(alive) > 1:
            voided = True
            # everyone got caught at once: classic playground rule, the round doesn't count
            for v in verdicts.values():
                v.out, v.reason = False, "all_caught_round_void"
            outs = []
            await self._say("Everyone got caught. That one doesn't count.", "info")
        for p in alive:
            v = verdicts[p.id]
            p.last_p = v.p
            p.seen = v.seen
            p.unseen_rounds = p.unseen_rounds + 1 if v.reason == "not_visible" else 0
            self.log.write("verdict", game_id=self.id, round=self.round, command=call.command.id, simon=call.simon,
                           call=call.text, player=p.display_name, pid=p.id,
                           p_action=None if v.p is None else round(v.p, 4), seen=v.seen, out=v.out,
                           reason=v.reason, uncertain=v.uncertain, models=self.models,
                           thresholds={"out_below": self.s.out_below, "trick_out_above": self.s.trick_out_above,
                                       "moved_above": self.s.moved_above})
        for p in outs:
            p.alive, p.out_round, p.out_reason = False, self.round, verdicts[p.id].reason
        self._record_round(call, alive, verdicts, voided, t_round, tts_call_ms)
        self.last_round = {
            "round_id": self.round_id, "round": self.round, "call": call.text, "command": call.command.id, "kind": call.command.kind,
            "simon": call.simon, "latency_ms": round(self.latencies[-1], 1) if self.latencies else None,
            "breakdown": self.last_breakdown,
            "decisions": [{"player": p.display_name, "number": p.number, "pid": p.id,
                           "p": None if verdicts[p.id].p is None else round(verdicts[p.id].p, 3),
                           "out": verdicts[p.id].out, "reason": verdicts[p.id].reason,
                           "uncertain": verdicts[p.id].uncertain, "seen": verdicts[p.id].seen} for p in alive],
        }
        self.rounds[self.round_id] = self.last_round
        self.history.append({"round": self.round, "call": call.text, "out": [p.display_name for p in outs]})
        await self._publish()
        for p in outs:
            await self._say(out_line(p, call, verdicts[p.id].reason), "out")
        unseen = [p for p in alive if p.alive and verdicts[p.id].reason == "not_visible"]
        if unseen:
            await self._say(f"Heads up, {join_names([p.call_name for p in unseen])}: I lost you. "
                            "Get back in the picture.", "warn")
        if not outs and not unseen and verdicts and not voided:
            await self._say(self.rng.choice(SAFE_LINES if call.simon else TRICK_SAFE), "safe")
        return outs

    def _record_round(self, call: Call, alive: Sequence[Player], verdicts: dict[int, Verdict], voided: bool,
                      t_round: float, tts_call_ms: float | None) -> None:
        """One `round` event: the call, per-stage timings, and per player the identity match, d1's answer,
        the verdict and the crops the referee looked at. Never lets a recording problem stop the game."""
        if not self.rec.enabled:
            return
        try:
            rid, ctx = self.round_id, self._ctx or {}
            frames, res = ctx.get("frames", []), ctx.get("res", {})
            kind = call.command.kind
            frame_paths = [self.rec.frame(f.image, f"{rid}_f{i}.jpg") for i, f in enumerate(frames)]
            rows = []
            for p in alive:
                v = verdicts[p.id]
                per = [(fr.get("players") or {}).get(str(p.id), {}) for fr in res.get("frames", [])]
                d = res.get("decisions", {}).get(str(p.id)) or {}
                boxes = [pf.get("box") for pf in per]
                crops: list[str | None] = []
                if kind == "freeze":
                    if len(boxes) == 2 and all(boxes):
                        ub = union_box(tuple(boxes[0]), tuple(boxes[1]))
                        crops = [self.rec.crop(crop_person(f.image, Detection(tuple(ub), 1.0)), f"{rid}_p{p.id}_f{i}.jpg")
                                 for i, f in enumerate(frames)]
                else:
                    crops = [self.rec.crop(crop_person(f.image, Detection(tuple(b), 1.0)), f"{rid}_p{p.id}_f{i}.jpg")
                             if b else None for i, (f, b) in enumerate(zip(frames, boxes))]
                rows.append({
                    "pid": p.id, "number": p.number, "name": p.display_name,
                    "seen_per_frame": [bool(pf.get("seen")) for pf in per],
                    "sim_per_frame": [pf.get("sim") for pf in per], "box_per_frame": boxes,
                    "p_per_frame": d.get("p") if kind != "freeze" else None,
                    "p_moved": d.get("p_moved"), "pixel_motion": d.get("pixel_motion"),
                    "p": None if v.p is None else round(v.p, 4), "seen": v.seen,
                    "verdict": "out" if v.out else "safe", "out": v.out, "reason": v.reason, "uncertain": v.uncertain,
                    "crops": crops,
                })
            lat = dict(res.get("latency_ms") or {})
            self.rec.event(
                "round", round_id=rid, game_id=self.id, round=self.round, command=call.command.id, call=call.text,
                kind=kind, simon=call.simon, trick=not call.simon, freeze=kind == "freeze",
                question=res.get("question"), voided=voided,
                timings_ms={"segment": lat.get("segment"), "embed": lat.get("embed"), "match": lat.get("match"),
                            "d1": lat.get("decide"), "referee": lat.get("decide"), "tinker": lat.get("tinker"),
                            "vision_total": lat.get("total"),
                            "round_trip": lat.get("round_trip"), "look": ctx.get("look_ms"), "tts_call": tts_call_ms,
                            "round_wall": round(1e3 * (time.perf_counter() - t_round), 1)},
                detections=[fr.get("detections") for fr in res.get("frames", [])],
                frames=frame_paths, players=rows, models=self.models, spans=res.get("spans"),
                thresholds={"out_below": self.s.out_below, "trick_out_above": self.s.trick_out_above,
                            "moved_above": self.s.moved_above})
        except Exception as e:  # noqa: BLE001
            log.warning("recorder: round %s not recorded (%s)", self.round, e)

    async def _watch_pose(self, call: Call, alive: list[Player]) -> dict[int, tuple[float, bool]]:
        await asyncio.sleep(self.s.reaction_s)
        frames = await self.io.frames(self.s.pose_frames, self.s.frame_gap_s)
        saved = self._save(frames, call.command.id)
        cmd = call.command
        # one batch for all frames: SAM finds everyone, DINO says who's who, d1 judges each player's own crop
        res = await self._look({"kind": "pose", "check": cmd.check, "yes": cmd.true_hint, "no": cmd.false_hint},
                               frames, alive)
        out: dict[int, tuple[float, bool]] = {}
        for p in alive:
            per = (res.get("decisions", {}).get(str(p.id)) or {}).get("p") or [None] * len(frames)
            seen = self._seen(res, p.id)
            any_seen = any(seen) and any(x is not None for x in per)
            out[p.id] = (combine(per, self.s.aggregate), any_seen)
            self.log.write("decision", game_id=self.id, round=self.round, command=cmd.id, simon=call.simon,
                           player=p.display_name, pid=p.id, question=res.get("question"),
                           p_per_frame=per, seen_per_frame=seen,
                           sim_per_frame=[(fr.get("players") or {}).get(str(p.id), {}).get("sim")
                                          for fr in res.get("frames", [])],
                           latency_ms=res.get("latency_ms"), requests=res.get("requests"),
                           input_tokens=res.get("input_tokens"), models=res.get("models"), spans=res.get("spans"),
                           method="sam_dino_reid+referee_player_crop", frames=saved)
        return out

    async def _watch_freeze(self, call: Call, alive: list[Player]) -> dict[int, tuple[float, bool]]:
        await asyncio.sleep(self.s.freeze_grace_s)
        [before] = await self.io.frames(1, 0)
        await asyncio.sleep(self.s.freeze_window_s)
        [after] = await self.io.frames(1, 0)
        saved = self._save([before, after], "freeze")
        cmd = call.command
        use_model = self.s.freeze_strategy == "pair"
        res = await self._look({"kind": "freeze", "check": cmd.check, "yes": cmd.true_hint, "no": cmd.false_hint,
                                "use_model": use_model, "pixel_lo": self.s.pixel_lo, "pixel_hi": self.s.pixel_hi},
                               [before, after], alive)
        out: dict[int, tuple[float, bool]] = {}
        for p in alive:
            d = res.get("decisions", {}).get(str(p.id)) or {}
            seen = self._seen(res, p.id)
            both = all(seen) and len(seen) == 2
            if "p_moved" in d:
                pm, method = float(d["p_moved"]), "d1_player_before_after_crops"
            elif "pixel_motion" in d:
                # FALLBACK: no multi-image model (or --freeze pixel), so pixel differencing decides
                pm, method = moved_probability(float(d["pixel_motion"]), self.s.pixel_lo, self.s.pixel_hi), "FALLBACK_pixel_diff"
            else:
                pm, method = 0.0, "not_seen"
            out[p.id] = (pm, both)
            self.log.write("decision", game_id=self.id, round=self.round, command=cmd.id, simon=call.simon,
                           player=p.display_name, pid=p.id, question=res.get("question"),
                           p_moved=round(pm, 4), seen_per_frame=seen, pixel_motion=d.get("pixel_motion"),
                           latency_ms=res.get("latency_ms"), input_tokens=res.get("input_tokens"),
                           models=res.get("models"), method=method, frames=saved)
        return out

    async def run(self) -> list[Player]:
        self._stop = False
        self.phase = "intro"
        names = join_names([p.call_name for p in self.players])
        self.log.write("game_start", game_id=self.id, players=[p.display_name for p in self.players],
                       models=self.models, backend=getattr(self.vision, "backend", "local"), settings=asdict(self.s))
        self.rec.game_started(self.id, players=[p.display_name for p in self.players], models=self.models,
                              settings=asdict(self.s), seed=self.s.seed)
        self.rec.event("game_start", game_id=self.id, players=[{"pid": p.id, "number": p.number, "name": p.display_name}
                                                               for p in self.players], seed=self.s.seed)
        await self._say(f"Welcome to Simon Sees. Players: {names}. Only do it if Simon says. Let's look at you.", "info")
        self.phase = "locating"
        await self.locate()
        solo = len(self.players) == 1
        while not self._stop and self.round < self.s.max_rounds and (len(self.alive) >= (1 if solo else 2)):
            await self.play_round()
            if self._stop or not self.alive or (not solo and len(self.alive) < 2):
                break
            self.phase = "rest"
            await self._say("Relax.", "rest")
            await asyncio.sleep(self.s.rest_s)
        self.phase = "final"
        self.winners = self.alive
        if solo:
            p = self.players[0]
            survived = self.round - (1 if not p.alive else 0)
            await self._say(f"Game over. You survived {survived} round{'s' if survived != 1 else ''}.", "win")
        elif len(self.winners) == 1:
            await self._say(f"{self.winners[0].call_name} wins! Now go touch some grass, all of you.", "win")
        elif self.winners:
            await self._say(f"Time's up. Winners: {join_names([p.call_name for p in self.winners])}.", "win")
        else:
            await self._say("Nobody's left. Simon wins.", "win")
        lat = sorted(self.latencies)
        self.log.write("game_end", game_id=self.id, rounds=self.round, winners=[p.display_name for p in self.winners],
                       decisions=len(lat), median_decide_ms=round(lat[len(lat) // 2], 2) if lat else None,
                       history=self.history)
        self.rec.event("game_end", game_id=self.id, rounds=self.round, winners=[p.display_name for p in self.winners],
                       history=self.history)
        self.rec.game_ended(self.id, rounds=self.round, winners=[p.display_name for p in self.winners])
        self.phase = "over"
        await self._publish()
        return self.winners


async def lineup(vision: Vision, io: GameIO) -> dict:
    """Line-up: one fresh frame, everyone in it, left to right (with photos and embeddings)."""
    frames = await io.frames(1, 0)
    with tracing.vision_span("enrol", 1, 0):
        return await asyncio.to_thread(vision.run, {"kind": "enrol"}, [frames[0].image], [])
