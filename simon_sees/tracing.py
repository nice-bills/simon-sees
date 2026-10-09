"""Optional Sentry Agent Tracing. Off unless SENTRY_DSN is set and sentry-sdk is installed.

Span layout (Sentry's gen_ai conventions):

    invoke_agent Simon Sees referee        op=gen_ai.invoke_agent   one per round
      evaluate LiquidAI/d1-3B              op=gen_ai.evaluate       one per decision batch

A d1 call IS an evaluation in Sentry's sense: a state plus noul/choice/score questions in, typed
answers out, zero output tokens. Frames never leave the device: only the question text and the
answers are attached, and only when SENTRY_SEND_PROMPTS=1.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any, Iterator, Sequence

log = logging.getLogger(__name__)
_sentry = None
_send_prompts = False
AGENT = "Simon Sees referee"


def init() -> bool:
    global _sentry, _send_prompts
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        log.warning("SENTRY_DSN is set but sentry-sdk is not installed: pip install sentry-sdk")
        return False
    _send_prompts = os.environ.get("SENTRY_SEND_PROMPTS", "0") == "1"
    sentry_sdk.init(dsn=dsn, traces_sample_rate=float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
                    send_default_pii=_send_prompts, environment=os.environ.get("SENTRY_ENV", "local"))
    _sentry = sentry_sdk
    log.info("Sentry tracing on")
    return True


def enabled() -> bool:
    return _sentry is not None


@contextlib.contextmanager
def round_span(game_id: str, round_no: int, call_text: str) -> Iterator[Any]:
    if not _sentry:
        yield None
        return
    with _sentry.start_transaction(op="gen_ai.invoke_agent", name=f"invoke_agent {AGENT}") as tx:
        tx.set_data("gen_ai.operation.name", "invoke_agent")
        tx.set_data("gen_ai.agent.name", AGENT)
        tx.set_tag("game_id", game_id)
        tx.set_tag("round", round_no)
        try:
            import sentry_sdk.ai

            sentry_sdk.ai.set_conversation_id(f"game-{game_id}")
        except Exception:  # noqa: BLE001 - older SDKs have no conversations
            pass
        if _send_prompts:
            tx.set_data("gen_ai.input.messages", json.dumps([{"role": "user", "parts": [{"type": "text", "content": call_text}]}]))
        yield tx


def decision_tracer(model_id: str, requests: Sequence, results: Sequence, ms: float) -> None:
    """Called by D1Decider after each batch: records one gen_ai.evaluate span (timed retroactively)."""
    if not _sentry:
        return
    import time

    from sentry_sdk import start_span

    end = time.time()
    with start_span(op="gen_ai.evaluate", name=f"evaluate {model_id}", start_timestamp=end - ms / 1e3) as span:
        span.set_data("gen_ai.operation.name", "evaluate")
        span.set_data("gen_ai.request.model", model_id)
        span.set_data("gen_ai.response.model", model_id)
        span.set_data("gen_ai.provider.name", "liquid")
        span.set_data("gen_ai.agent.name", AGENT)
        span.set_data("gen_ai.usage.input_tokens", sum(r.input_tokens for r in results))
        span.set_data("gen_ai.usage.output_tokens", 0)
        span.set_data("simon.requests", len(requests))
        span.set_data("simon.questions", sum(len(r.questions) for r in requests))
        span.set_data("simon.images", sum(len(r.images) for r in requests))
        span.set_data("simon.latency_ms", round(ms, 2))
        if _send_prompts:
            span.set_data("gen_ai.input.messages", json.dumps(
                [{"type": "evaluation", "state": f"{len(r.images)} camera frame(s)", "questions": r.questions} for r in requests]))
            span.set_data("gen_ai.output.messages", json.dumps(
                [{"type": "evaluation", "answers": r.answers} for r in results], default=float))


@contextlib.contextmanager
def vision_span(kind: str, n_frames: int, n_players: int) -> Iterator[Any]:
    """One span per vision batch (SAM -> DINO -> match -> d1). No images, just counts."""
    if not _sentry:
        yield None
        return
    from sentry_sdk import start_span

    with start_span(op="vision.batch", name=f"vision {kind}") as span:
        span.set_data("frames", n_frames)
        span.set_data("players", n_players)
        yield span
