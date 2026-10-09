"""Simon's voice (open TTS on the server) and ears (open STT so players can say their name at the line-up).

TTS, first that works:  Kokoro-82M (`pip install kokoro soundfile`, Apache-2.0 weights)
                        -> Piper (`pip install piper-tts`, set SIMON_PIPER_VOICE=/path/voice.onnx)
                        -> none: the phone speaks with the browser's speechSynthesis instead.
STT:                    faster-whisper (`pip install faster-whisper`, Whisper weights are MIT)
                        -> none: players type their name (or keep "Player N").
Both are optional; nothing here is imported unless enabled.
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import wave

log = logging.getLogger(__name__)


def _wav_bytes(samples, rate: int) -> bytes:
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / float(w.getframerate())


class Voice:
    """`synth(text) -> wav bytes | None`. `engine` is 'kokoro', 'piper', 'browser' or 'auto'."""

    def __init__(self, engine: str = "auto", kokoro_voice: str | None = None):
        self.engine = "browser"
        self._impl = None
        self._cache: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self.kokoro_voice = kokoro_voice or os.environ.get("SIMON_KOKORO_VOICE", "am_michael")
        order = ["kokoro", "piper"] if engine == "auto" else [engine]
        for name in order:
            if name == "browser":
                break
            try:
                self._impl = getattr(self, f"_load_{name}")()
                self.engine = name
                log.info("TTS: %s", name)
                break
            except Exception as e:  # noqa: BLE001 - optional dependency, fall through
                log.info("TTS %s unavailable (%s)", name, e)
        if self.engine == "browser":
            log.info("TTS: none on the server; the phone will use speechSynthesis")

    def _load_kokoro(self):
        from kokoro import KPipeline  # type: ignore

        pipe = KPipeline(lang_code="a")

        def synth(text: str) -> bytes:
            import numpy as np

            parts = [np.asarray(audio) for _, _, audio in pipe(text, voice=self.kokoro_voice)]
            return _wav_bytes(np.concatenate(parts) if parts else np.zeros(2400), 24000)

        return synth

    def _load_piper(self):
        from piper import PiperVoice  # type: ignore

        path = os.environ.get("SIMON_PIPER_VOICE")
        if not path:
            raise RuntimeError("set SIMON_PIPER_VOICE to a .onnx voice file")
        voice = PiperVoice.load(path)

        def synth(text: str) -> bytes:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                voice.synthesize_wav(text, w)
            return buf.getvalue()

        return synth

    @property
    def available(self) -> bool:
        return self._impl is not None

    def synth(self, text: str) -> bytes | None:
        if not self._impl:
            return None
        with self._lock:
            if text not in self._cache:
                self._cache[text] = self._impl(text)
            return self._cache[text]


def estimate_speech_s(text: str) -> float:
    """How long a phone takes to say `text` with speechSynthesis (about 2.7 words a second)."""
    return 0.4 + len(text.split()) / 2.7


# --------------------------------------------------------------------------- STT


class Ears:
    def __init__(self, enabled: bool = True, model: str | None = None):
        self._model = None
        self.engine = "typed"
        if not enabled:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore

            self._model = WhisperModel(model or os.environ.get("SIMON_WHISPER", "base.en"), device="auto",
                                       compute_type="int8")
            self.engine = "faster-whisper"
        except Exception as e:  # noqa: BLE001
            log.info("STT unavailable (%s); players type their name", e)

    @property
    def available(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: bytes) -> str:
        if not self._model:
            raise RuntimeError("speech recognition is not installed (pip install faster-whisper)")
        segments, _ = self._model.transcribe(io.BytesIO(audio), language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


_NAME_LEAD = re.compile(r"^(?:(?:hi|hey|hello|okay|ok|um|uh|so|yeah|yo|it'?s|i'?m|i am|me|this is|my name is|my name'?s|"
                        r"name'?s|call me|they call me|you can call me|i go by)\b[\s,]*)+", re.I)


def parse_name(text: str) -> str:
    """'Hi, I'm Kofi.' -> 'Kofi'; 'my name is ama mensah and I'm ready' -> 'Ama Mensah'. Two words at most:
    Simon will shout it."""
    t = text.strip().strip(".!?").strip()
    t = _NAME_LEAD.sub("", t, count=1)
    t = re.split(r"[.,;!?]| and\b| but\b", t, maxsplit=1, flags=re.I)[0]
    words = [w for w in re.findall(r"[^\W\d_][\w'-]*", t)][:2]
    return " ".join(w[:1].upper() + w[1:] for w in words)
