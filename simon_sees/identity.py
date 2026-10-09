"""Who is who: enrolled players matched to the people found in a frame.

Every player has a small gallery of embeddings (DINOv3 of their own crop): the one from the line-up,
plus a few confident sightings added during the game so the signature follows lighting changes and
new poses. Each frame:

    similarity[p, d] = max cosine(gallery[p], det_embedding[d]) + a small bonus for overlapping
                       the box where player p was last seen (tracking, so neighbours don't swap)
    assignment       = Hungarian (min total cost) over 1 - similarity
    accepted         = assigned AND (cosine >= threshold
                                     OR cosine >= low_threshold AND the detection is where that player just was)
                       everyone else is "not seen this frame"

The second clause is tracking: a big pose change (arms out like an airplane, a deep squat) moves the
embedding a long way, but the person is still standing in the same spot.

Pure functions + a small dataclass, so the same code runs locally and on the Modal GPU.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

Box = tuple[float, float, float, float]


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment for a rectangular cost matrix (rows x cols). Returns (row, col) pairs,
    one per row when rows <= cols, else one per col. Uses scipy when installed; otherwise a plain
    O(n^3) Kuhn-Munkres (shortest augmenting path), which is plenty for a handful of players."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        r, c = linear_sum_assignment(cost)
        return [(int(a), int(b)) for a, b in zip(r, c)]
    except ImportError:
        pass
    transposed = cost.shape[0] > cost.shape[1]
    a = cost.T if transposed else cost
    n, m = a.shape  # n <= m
    INF = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    return sorted((c, r) if transposed else (r, c) for r, c in pairs)


def brute_force_assignment(cost: np.ndarray) -> float:
    """Reference minimum for tests."""
    n, m = cost.shape
    if n <= m:
        return min(sum(cost[i, j] for i, j in enumerate(perm)) for perm in itertools.permutations(range(m), n))
    return brute_force_assignment(cost.T)


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Identity:
    """One enrolled player as the matcher sees them (no names, no clothing: just pixels' signature)."""
    pid: int
    gallery: list[np.ndarray]  # L2-normalised embeddings; [0] is the line-up one and is never dropped
    last_box: Box | None = None
    max_gallery: int = 6

    def similarity(self, embs: np.ndarray) -> np.ndarray:
        g = np.stack(self.gallery)  # (k, d)
        return (embs @ g.T).max(axis=1) if len(embs) else np.zeros(0)

    def remember(self, emb: np.ndarray, box: Box | None) -> None:
        self.last_box = box
        self.gallery.append(np.asarray(emb, dtype=np.float32))
        if len(self.gallery) > self.max_gallery:
            del self.gallery[1]  # keep the line-up embedding, forget the oldest sighting

    def to_json(self) -> dict:
        return {"pid": self.pid, "gallery": [g.astype(np.float32).round(5).tolist() for g in self.gallery],
                "last_box": list(self.last_box) if self.last_box else None}

    @classmethod
    def from_json(cls, d: dict) -> "Identity":
        return cls(int(d["pid"]), [np.asarray(g, dtype=np.float32) for g in d["gallery"]],
                   tuple(d["last_box"]) if d.get("last_box") else None)


@dataclass
class Match:
    pid: int
    det: int | None  # index into the frame's detections, None = not seen
    sim: float  # cosine similarity of the assigned detection (or best one, when rejected)


@dataclass
class MatchConfig:
    threshold: float = 0.45  # minimum cosine to accept (DINO CLS of person crops: same person ~0.7-0.95)
    track_bonus: float = 0.15  # added for IoU with the last box (scaled by IoU)
    remember_above: float = 0.65  # confident sightings join the gallery
    margin: float = 0.0  # extra: accepted sim must beat the runner-up by this much (0 = off)
    low_threshold: float = 0.2  # ...or this much, when the detection continues the player's track
    track_iou: float = 0.3  # track continues: IoU with the last box, or the centre within half a box width


def continues_track(last: Box | None, box: Box, min_iou: float = 0.3) -> bool:
    if last is None:
        return False
    if iou(last, box) >= min_iou:
        return True
    w = max(1.0, last[2] - last[0])
    return abs((box[0] + box[2]) / 2 - (last[0] + last[2]) / 2) <= 0.5 * w


def assign(identities: Sequence[Identity], embs: np.ndarray, boxes: Sequence[Box],
           cfg: MatchConfig = MatchConfig()) -> list[Match]:
    """Match every identity to at most one detection, and every detection to at most one identity."""
    if not identities:
        return []
    if len(embs) == 0:
        return [Match(i.pid, None, 0.0) for i in identities]
    sims = np.stack([i.similarity(embs) for i in identities])  # (P, D)
    bonus = np.zeros_like(sims)
    for r, ident in enumerate(identities):
        if ident.last_box is not None:
            bonus[r] = [cfg.track_bonus * iou(ident.last_box, b) for b in boxes]
    pairs = dict(hungarian(1.0 - (sims + bonus)))
    out = []
    for r, ident in enumerate(identities):
        d = pairs.get(r)
        if d is None:
            out.append(Match(ident.pid, None, float(sims[r].max())))
            continue
        s = float(sims[r, d])
        ok = s >= cfg.threshold or (s >= cfg.low_threshold and continues_track(ident.last_box, boxes[d], cfg.track_iou))
        if ok and cfg.margin and sims.shape[1] > 1:
            others = np.delete(sims[r], d)
            ok = s - float(others.max()) >= cfg.margin or s >= cfg.remember_above
        out.append(Match(ident.pid, d if ok else None, s))
    return out


def order_left_to_right(boxes: Sequence[Box]) -> list[int]:
    """Enrolment numbering: Player 1 is the leftmost person in the picture."""
    return sorted(range(len(boxes)), key=lambda k: (boxes[k][0] + boxes[k][2]) / 2)


@dataclass
class Roster:
    identities: dict[int, Identity] = field(default_factory=dict)

    def snapshot(self, pids: Sequence[int] | None = None) -> list[dict]:
        return [self.identities[p].to_json() for p in (pids if pids is not None else self.identities)]

    def apply(self, updates: dict) -> None:
        """Updates from a batch result: {pid: {"emb": [...] | None, "box": [...]}}."""
        for pid, u in updates.items():
            ident = self.identities.get(int(pid))
            if ident is None:
                continue
            if u.get("emb") is not None:
                ident.remember(np.asarray(u["emb"], dtype=np.float32), tuple(u["box"]) if u.get("box") else None)
            elif u.get("box"):
                ident.last_box = tuple(u["box"])
