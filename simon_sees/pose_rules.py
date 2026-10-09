"""Explicit geometric rules that turn COCO-style body keypoints into yes/no labels for Simon's pose commands.

Used to build the referee fine-tuning set (scripts/build_referee_dataset.py). Each rule returns

    True   a confident positive  (the pose clearly matches the command)
    False  a clear negative      (the pose clearly does NOT match)
    None   ambiguous: the example is dropped

so only clean examples reach training. Image coordinates: x to the right, y DOWN. Every distance is
measured in torso lengths T (shoulder midpoint to hip midpoint), so the rules do not depend on how big the
person is in the photo.

COCO keypoint order: 0 nose, 1 l-eye, 2 r-eye, 3 l-ear, 4 r-ear, 5 l-shoulder, 6 r-shoulder, 7 l-elbow,
8 r-elbow, 9 l-wrist, 10 r-wrist, 11 l-hip, 12 r-hip, 13 l-knee, 14 r-knee, 15 l-ankle, 16 r-ankle.
A keypoint counts only when it is labelled visible (v == 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

NAMES = ["nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
         "lhip", "rhip", "lknee", "rknee", "lank", "rank"]
IDX = {n: i for i, n in enumerate(NAMES)}

# the static pose commands of simon_sees/commands.py that keypoints can decide. Not covered:
#   freeze (judged by frame differencing), sit_down (ground vs chair is not visible in keypoints),
#   touch_green / touch_ground / hug_tree (need the prop, not just the pose).
#   hand_mouth and wave were tried and dropped: on COCO a hand near the face is mostly a phone call, a drink or a
#   batting stance, and one raised hand is mostly a sport swing. The human check of the test set found 1 in 16 and
#   3 in 16 rule positives were real, so their rules (kept below for reference) do not feed the dataset.
RULE_COMMANDS = ["one_foot", "hands_head", "arms_up", "touch_toes", "airplane", "squat"]
EXCLUDED_RULES = ["hand_mouth", "wave"]


class Pose:
    """Keypoints of one person, with torso-normalised helpers."""

    def __init__(self, kps: Sequence[float]):
        self.p: dict[str, tuple[float, float] | None] = {}
        for i, n in enumerate(NAMES):
            x, y, v = kps[3 * i], kps[3 * i + 1], kps[3 * i + 2]
            self.p[n] = (float(x), float(y)) if v == 2 else None
        ls, rs, lh, rh = self.p["lsho"], self.p["rsho"], self.p["lhip"], self.p["rhip"]
        self.T = None
        if ls and rs and lh and rh:
            self.msho = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            self.mhip = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            t = math.dist(self.msho, self.mhip)
            self.T = t if t > 1 else None

    def has(self, *names: str) -> bool:
        return self.T is not None and all(self.p[n] is not None for n in names)

    def __getitem__(self, n: str) -> tuple[float, float]:
        v = self.p[n]
        assert v is not None, n
        return v

    def d(self, a: str, b: str) -> float:
        return math.dist(self[a], self[b]) / self.T  # type: ignore[operator]

    def above(self, a: str, b: str) -> float:
        """How far a is ABOVE b, in torso lengths (negative = below)."""
        return (self[b][1] - self[a][1]) / self.T  # type: ignore[operator]

    def angle(self, a: str, b: str, c: str) -> float:
        """Angle at b (degrees) between b->a and b->c: 180 = straight limb."""
        ax, ay = self[a][0] - self[b][0], self[a][1] - self[b][1]
        cx, cy = self[c][0] - self[b][0], self[c][1] - self[b][1]
        na, nc = math.hypot(ax, ay), math.hypot(cx, cy)
        if na < 1e-6 or nc < 1e-6:
            return 180.0
        cos = max(-1.0, min(1.0, (ax * cx + ay * cy) / (na * nc)))
        return math.degrees(math.acos(cos))

    def torso_tilt(self) -> float:
        """Degrees the torso leans from vertical (0 = upright, 90 = horizontal, >90 = upside down)."""
        dx = self.msho[0] - self.mhip[0]
        dy = self.mhip[1] - self.msho[1]  # positive when the shoulders are above the hips
        return math.degrees(math.atan2(abs(dx), dy))

    def upright(self, max_tilt: float = 25.0) -> bool:
        return self.torso_tilt() <= max_tilt


# --------------------------------------------------------------------------- the rules


def _arm(side: str) -> tuple[str, str, str]:
    return (f"{side}sho", f"{side}elb", f"{side}wri")


def arms_up(p: Pose) -> bool | None:
    need = ("nose", "lsho", "rsho", "lelb", "relb", "lwri", "rwri")
    if not p.has(*need) or not p.upright(30):
        return None
    ups = [p.above(f"{s}wri", "nose") for s in "lr"]
    straight = [p.angle(*_arm(s)) for s in "lr"]
    elb_up = [p.above(f"{s}elb", f"{s}sho") for s in "lr"]
    if min(ups) > 0.3 and min(straight) > 120 and min(elb_up) > 0.1:
        return True
    # clearly not: at least one wrist below its shoulder, or both hands near the head with bent elbows
    if min(p.above(f"{s}wri", f"{s}sho") for s in "lr") < -0.15:
        return False
    if max(ups) < 0.35 and max(straight) < 110:
        return False
    return None


def hands_head(p: Pose) -> bool | None:
    need = ("nose", "lsho", "rsho", "lelb", "relb", "lwri", "rwri")
    if not p.has(*need) or not p.upright(30):
        return None
    wr = [f"{s}wri" for s in "lr"]
    up = [p.above(w, "nose") for w in wr]  # wrist height over the nose
    dx = [abs(p[w][0] - p["nose"][0]) / p.T for w in wr]  # type: ignore[operator]
    bent = [p.angle(*_arm(s)) for s in "lr"]
    elbows = [p[f"{s}elb"][0] - p["nose"][0] for s in "lr"]
    elbows_out = elbows[0] * elbows[1] < 0 and all(abs(e) > abs(p[w][0] - p["nose"][0]) for e, w in zip(elbows, wr))
    # both hands at head height, close to the head, elbows bent and pointing out to opposite sides
    # (rules out batting stances and racket swings, where both elbows sit on one side)
    if all(0.0 < u < 0.7 for u in up) and max(dx) < 0.5 and max(bent) < 120 and elbows_out:
        return True
    if max(p.above(w, f"{s}sho") for w, s in zip(wr, "lr")) < -0.1:  # both wrists below the shoulders
        return False
    if min(up) > 0.75 and min(bent) > 145:  # arms straight up above the head
        return False
    if min(p.above(w, f"{s}sho") for w, s in zip(wr, "lr")) < -0.3 and max(up) > 0.2:  # one hand up only
        return False
    return None


def airplane(p: Pose) -> bool | None:
    need = ("lsho", "rsho", "lelb", "relb", "lwri", "rwri")
    if not p.has(*need) or not p.upright(25):
        return None
    ok = []
    for s in "lr":
        sho, elb, wri = _arm(s)
        dx, dy = p[wri][0] - p[sho][0], p[wri][1] - p[sho][1]
        horiz = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))  # 0 = horizontal arm
        reach = abs(dx) / p.T  # type: ignore[operator]
        ok.append(horiz < 22 and reach > 0.75 and p.angle(sho, elb, wri) > 145)
    # the two wrists must be on opposite sides of the body
    outside = (p["lwri"][0] - p.msho[0]) * (p["rwri"][0] - p.msho[0]) < 0
    if all(ok) and outside:
        return True
    downs = [p.above(f"{s}wri", f"{s}sho") for s in "lr"]
    if max(downs) < -0.5:  # both arms hanging down
        return False
    if min(downs) > 0.6:  # both arms up
        return False
    if sum(ok) == 0 and min(downs) < -0.4:
        return False
    return None


def one_foot(p: Pose) -> bool | None:
    need = ("lhip", "rhip", "lknee", "rknee", "lank", "rank")
    if not p.has(*need) or not p.upright(20):
        return None
    lift = p.above("lank", "rank")  # >0: left ankle higher
    hi, lo = ("l", "r") if lift > 0 else ("r", "l")
    stand_knee = p.angle(f"{lo}hip", f"{lo}knee", f"{lo}ank")
    raised_knee = p.angle(f"{hi}hip", f"{hi}knee", f"{hi}ank")
    stance_under_hip = abs(p[f"{lo}ank"][0] - p.mhip[0]) / p.T < 0.45  # type: ignore[operator]
    if abs(lift) > 0.45 and stand_knee > 160 and raised_knee < 130 and stance_under_hip:
        return True
    if abs(lift) < 0.08 and min(stand_knee, raised_knee) > 160:
        return False
    return None


def squat(p: Pose) -> bool | None:
    need = ("lsho", "rsho", "lhip", "rhip", "lknee", "rknee", "lank", "rank")
    if not p.has(*need):
        return None
    knees = [p.angle(f"{s}hip", f"{s}knee", f"{s}ank") for s in "lr"]
    hip_over_ank = min(p.above(f"{s}hip", f"{s}ank") for s in "lr")
    knee_over_ank = min(p.above(f"{s}knee", f"{s}ank") for s in "lr")
    # a deep squat: knees sharply bent, hips low but still off the ground, shins standing (not kneeling),
    # shoulders above the hips. Chair sitting has the hips a full shin-length above the ankles.
    if max(knees) < 95 and 0.15 < hip_over_ank < 0.65 and knee_over_ank > 0.35 and p.torso_tilt() < 60:
        return True
    if p.upright(15) and min(knees) > 165 and hip_over_ank > 1.3:
        return False
    return None


def touch_toes(p: Pose) -> bool | None:
    need = ("lsho", "rsho", "lhip", "rhip", "lknee", "rknee", "lank", "rank", "lwri", "rwri")
    if not p.has(*need):
        return None
    knees = [p.angle(f"{s}hip", f"{s}knee", f"{s}ank") for s in "lr"]
    wrist_to_ank = min(min(p.d(w, a) for a in ("lank", "rank")) for w in ("lwri", "rwri"))
    wrists_low = min(p.above(w, k) for w, k in (("lwri", "lknee"), ("rwri", "rknee")))  # <0: below knees
    if p.torso_tilt() > 45 and min(knees) > 120 and wrist_to_ank < 0.8 and wrists_low < 0.0:
        return True
    if p.upright(15) and min(knees) > 160 and max(p.above(w, k) for w, k in (("lwri", "lknee"), ("rwri", "rknee"))) > 0.5 \
            and min(p.above(w, k) for w, k in (("lwri", "lknee"), ("rwri", "rknee"))) > 0.3:
        return False
    return None


def hand_mouth(p: Pose) -> bool | None:
    need = ("nose", "lsho", "rsho", "lwri", "rwri")
    if not p.has(*need) or not p.upright(30):
        return None
    if not (p.p["leye"] and p.p["reye"]):  # face toward the camera
        return None
    mouth = (p["nose"][0], p["nose"][1] + 0.12 * p.T)  # type: ignore[operator]
    dist = [math.dist(p[w], mouth) / p.T for w in ("lwri", "rwri")]  # type: ignore[operator]
    if min(dist) < 0.22:
        return True
    if min(dist) > 1.0:
        return False
    return None


def wave(p: Pose) -> bool | None:
    need = ("nose", "lsho", "rsho", "lelb", "relb", "lwri", "rwri")
    if not p.has(*need) or not p.upright(25):
        return None
    if not (p.p["leye"] and p.p["reye"]):  # facing the camera
        return None
    ups = {s: p.above(f"{s}wri", f"{s}sho") for s in "lr"}
    hi, lo = ("l", "r") if ups["l"] > ups["r"] else ("r", "l")
    head_clear = abs(p[f"{hi}wri"][0] - p["nose"][0]) / p.T > 0.35  # type: ignore[operator]  # not touching the head
    if ups[hi] > 0.45 and ups[lo] < -0.3 and p.above(f"{hi}elb", f"{hi}sho") > -0.25 and head_clear:
        return True
    if max(ups.values()) < -0.3:
        return False
    return None


RULES: dict[str, Callable[[Pose], bool | None]] = {
    "one_foot": one_foot, "hands_head": hands_head, "arms_up": arms_up, "touch_toes": touch_toes,
    "airplane": airplane, "squat": squat, "hand_mouth": hand_mouth, "wave": wave,
}


@dataclass
class Label:
    command: str
    value: bool


def label_all(kps: Sequence[float]) -> dict[str, bool]:
    """Every command whose rule is confident about this person: {command: True/False}."""
    p = Pose(kps)
    out = {}
    if p.T is None:
        return out
    for cid in RULE_COMMANDS:
        v = RULES[cid](p)
        if v is not None:
            out[cid] = v
    return out
