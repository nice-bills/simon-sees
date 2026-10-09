"""The things Simon can ask for, and how each one is judged from camera frames.

Every command is a plain description of a pose or action. The engine turns it into one
yes/no ("noul") question per player for the decision model, e.g.

    "Look at the person in this photo. Is that person standing on one foot?"   (asked about one player's own crop)

`kind` decides how the frames are judged:

- "pose":   a held body position, judged on frames taken after a reaction beat.
- "touch":  touching something (green, the ground, a tree...), judged like a pose.
- "freeze": hold still. Judged on a BEFORE and an AFTER frame: the model sees both
            images in one pass and is asked whether the player moved between them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Command:
    id: str
    spoken: str            # what Simon says after "Simon says", e.g. "stand on one foot"
    check: str             # the visual check, phrased about "that person"
    kind: str = "pose"     # pose | touch | freeze
    outside_only: bool = False  # needs real outdoor props (grass, a tree)
    true_hint: str = ""    # what counts as yes (criteria for the noul question)
    false_hint: str = ""   # what counts as no
    out_phrase: str = ""   # used in "Number 2, you ___. You're out."
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def trickable(self) -> bool:
        """Freeze is a bad trick: standing still is what everyone already does."""
        return self.kind != "freeze"


COMMANDS: list[Command] = [
    Command("one_foot", "stand on one foot", "Is that person standing on one foot, with the other foot lifted off the ground?",
            true_hint="one foot clearly lifted off the ground", false_hint="both feet on the ground",
            out_phrase="kept both feet down"),
    Command("hands_head", "put your hands on your head", "Does that person have both hands on top of their head?",
            true_hint="both hands resting on the head", false_hint="hands anywhere else",
            out_phrase="didn't put your hands on your head"),
    Command("arms_up", "reach for the sky", "Is that person holding both arms straight up above their head?",
            true_hint="both arms raised above the head", false_hint="arms down or to the side",
            out_phrase="didn't reach for the sky"),
    Command("touch_toes", "touch your toes", "Is that person bending forward with their hands at or near their feet?",
            true_hint="bent over, hands down at the feet or shins", false_hint="standing upright",
            out_phrase="didn't touch your toes"),
    Command("airplane", "be an airplane", "Is that person holding both arms straight out to the sides, like airplane wings?",
            true_hint="both arms stretched out horizontally", false_hint="arms down, up, or bent",
            out_phrase="aren't flying"),
    Command("squat", "squat down low", "Is that person squatting or crouching down low?",
            true_hint="knees deeply bent, body low", false_hint="standing up straight",
            out_phrase="didn't squat"),
    Command("hand_mouth", "cover your mouth", "Is that person covering their mouth with a hand?",
            true_hint="a hand over the mouth", false_hint="mouth visible, hands elsewhere",
            out_phrase="didn't cover your mouth"),
    Command("sit_down", "sit on the ground", "Is that person sitting on the ground?",
            true_hint="sitting on the ground or grass", false_hint="standing, kneeling or crouching",
            out_phrase="didn't sit down"),
    Command("wave", "wave at me", "Is that person waving a raised hand toward the camera?",
            true_hint="one hand raised and open, waving", false_hint="hands down",
            out_phrase="didn't wave"),
    Command("touch_green", "touch something green", "Is that person touching something green, like grass or leaves, with a hand?",
            kind="touch", outside_only=True, true_hint="a hand on grass, leaves or a green object",
            false_hint="hands not touching anything green", out_phrase="didn't touch anything green", tags=("touch-grass",)),
    Command("touch_ground", "touch the ground", "Is that person touching the ground with a hand?",
            kind="touch", true_hint="a hand on the ground", false_hint="no hand on the ground",
            out_phrase="didn't touch the ground", tags=("touch-grass",)),
    Command("hug_tree", "hug a tree", "Is that person hugging or holding onto a tree trunk?",
            kind="touch", outside_only=True, true_hint="arms around or a hand on a tree trunk",
            false_hint="not touching a tree", out_phrase="didn't hug a tree"),
    Command("freeze", "freeze", "Did that person move their body, arms, legs or head between the first photo and the second photo?",
            kind="freeze", true_hint="clearly moved: a limb, the head or the body is in a different position",
            false_hint="held the same pose in both photos", out_phrase="moved"),
]

BY_ID = {c.id: c for c in COMMANDS}

SIMON_PREFIX = "Simon says"


@dataclass(frozen=True)
class Call:
    """One round's instruction."""
    command: Command
    simon: bool  # True if "Simon says" was spoken

    @property
    def text(self) -> str:
        spoken = self.command.spoken
        return f"{SIMON_PREFIX} {spoken}." if self.simon else f"{spoken[0].upper()}{spoken[1:]}!"


class CommandPicker:
    """Chooses the next call. Seeded so dry runs and tests are reproducible."""

    def __init__(self, seed: int | None = None, trick_rate: float = 0.25, outdoors: bool = True,
                 allowed: list[str] | None = None, freeze_rate: float = 0.2):
        self.rng = random.Random(seed)
        self.trick_rate = trick_rate
        self.freeze_rate = freeze_rate
        pool = [c for c in COMMANDS if (outdoors or not c.outside_only)]
        if allowed:
            pool = [c for c in pool if c.id in allowed]
        if not pool:
            raise ValueError("no commands left to pick from")
        self.pool = pool
        self.last: str | None = None

    def next(self, round_no: int) -> Call:
        # never trick in the first round: people need to learn the rhythm first
        trick = round_no > 1 and self.rng.random() < self.trick_rate
        candidates = [c for c in self.pool if c.id != self.last]
        if trick:
            candidates = [c for c in candidates if c.trickable] or candidates
        freeze = [c for c in candidates if c.kind == "freeze"]
        if freeze and not trick and self.rng.random() < self.freeze_rate:
            cmd = freeze[0]
        else:
            others = [c for c in candidates if c.kind != "freeze"] or candidates
            cmd = self.rng.choice(others)
        self.last = cmd.id
        return Call(cmd, simon=not trick if cmd.trickable else True)
