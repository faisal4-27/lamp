"""Local text-to-speech. Piper (neural, ~real-time x10 on CPU) with a pyttsx3
fallback (espeak on Linux, SAPI on Windows) if the Piper voice is missing."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import wave

import numpy as np

from ..config import ASSETS, Config
from .mixer import RATE

log = logging.getLogger("tts")


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(x) == 0:
        return x
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


class TTS:
    def __init__(self, cfg: Config):
        self.engine = "none"
        self._piper = None
        self._lock = threading.Lock()  # the reply and a song being rendered may synthesise at once
        path = ASSETS / "voices" / f"{cfg.piper_voice}.onnx"
        try:
            from piper import PiperVoice

            self._piper = PiperVoice.load(str(path))
            self.engine = "piper"
        except Exception as exc:  # missing model or runtime
            log.warning("piper unavailable (%s); falling back to pyttsx3", exc)
            self.engine = "pyttsx3"

    def synthesize(self, text: str) -> np.ndarray:
        with self._lock:
            return self._synthesize(text)

    def _synthesize(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        if self._piper is not None:
            parts, rate = [], RATE
            for chunk in self._piper.synthesize(text):
                parts.append(chunk.audio_float_array)
                rate = chunk.sample_rate
            audio = resample(np.concatenate(parts) if parts else np.zeros(1, np.float32), rate, RATE)
        else:
            audio = self._pyttsx3(text)
        log.debug("tts %.0f ms for %d chars", (time.perf_counter() - t0) * 1000, len(text))
        # Short fade avoids clicks at the edges.
        n = min(len(audio), 200)
        if n:
            audio[:n] *= np.linspace(0, 1, n)
            audio[-n:] *= np.linspace(1, 0, n)
        return audio

    def _pyttsx3(self, text: str) -> np.ndarray:
        import pyttsx3

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            eng = pyttsx3.init()
            eng.save_to_file(text, path)
            eng.runAndWait()
            with wave.open(path) as w:
                raw = w.readframes(w.getnframes())
                audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
                if w.getnchannels() == 2:
                    audio = audio[::2]
                return resample(audio, w.getframerate(), RATE)
        finally:
            os.remove(path)
