"""Every referee decision as one JSON line: what was asked, what the model answered, how long it took,
and what the game did about it. This file is the raw data for the write-up (see scripts/summarize_log.py)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class DecisionLog:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> dict:
        row = {"ts": round(time.time(), 3), "event": event, **fields}
        if self.path:
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row
