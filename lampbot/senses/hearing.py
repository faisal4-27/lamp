"""Microphone capture, voice-activity detection and local speech-to-text.

Audio never leaves the machine: WebRTC VAD segments utterances from a 16 kHz
stream and faster-whisper (int8, CPU) transcribes them. Only the resulting text
is used downstream.

The lamp is half-duplex: the brain closes the gate while the lamp itself is
speaking or playing sound, so it never transcribes its own voice. Barge-in would
need acoustic echo cancellation and is deliberately out of scope.
"""

from __future__ import annotations

import collections
import logging
import queue
import re
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd
import webrtcvad

from ..config import ASSETS, Config

log = logging.getLogger("hearing")
RATE = 16000
FRAME_MS = 30
FRAME = RATE * FRAME_MS // 1000
# Whisper is known to hallucinate these on silence or noise.
HALLUCINATIONS = {"", "you", "thank you", "thanks for watching", "bye", "uh", "um", "hmm"}


def _device(spec: str):
    if not spec:
        return None
    return int(spec) if spec.isdigit() else spec


class Hearing:
    def __init__(self, cfg: Config, on_event: Callable[[str, dict], None]):
        self.cfg = cfg
        self.on_event = on_event
        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.gate_open = False  # set by the brain: only listen when engaged and quiet
        self.level = 0.0
        self._frames: queue.Queue[bytes] = queue.Queue(maxsize=400)
        self._utterances: queue.Queue[tuple[np.ndarray, float]] = queue.Queue()
        self._stop = threading.Event()
        self._model = None
        self.ready = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._load_model, name="whisper-load", daemon=True).start()
        threading.Thread(target=self._segment, name="vad", daemon=True).start()
        threading.Thread(target=self._transcribe, name="stt", daemon=True).start()
        self._stream = sd.RawInputStream(samplerate=RATE, blocksize=FRAME, channels=1, dtype="int16",
                                         device=_device(self.cfg.mic_device), callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._stream.stop()
        except Exception:
            pass

    def _load_model(self) -> None:
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        kw = dict(device="cpu", compute_type="int8", cpu_threads=4, download_root=str(ASSETS / "whisper"))
        try:
            self._model = WhisperModel(self.cfg.whisper_model, local_files_only=True, **kw)
        except Exception:  # not fetched yet: download once
            self._model = WhisperModel(self.cfg.whisper_model, **kw)
        # Warm up so the first real utterance is not slow.
        list(self._model.transcribe(np.zeros(RATE, np.float32), language="en", beam_size=1)[0])
        log.info("whisper %s ready in %.1fs", self.cfg.whisper_model, time.perf_counter() - t0)
        self.ready.set()

    def _callback(self, indata, frames, t, status) -> None:
        try:
            self._frames.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _segment(self) -> None:
        pre_roll = collections.deque(maxlen=10)  # 300 ms of audio before speech onset
        voiced_run, silent_run = 0, 0
        speaking = False
        buf: list[bytes] = []
        end_frames = int(self.cfg.end_of_speech_s * 1000 / FRAME_MS)
        max_frames = int(self.cfg.max_utterance_s * 1000 / FRAME_MS)
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.5)
            except queue.Empty:
                continue
            pcm = np.frombuffer(frame, np.int16)
            rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768) ** 2)))
            self.level = 0.8 * self.level + 0.2 * min(1.0, rms * 12)
            if not self.gate_open:
                if speaking:
                    self.on_event("speech_cancelled", {})
                speaking, buf, voiced_run, silent_run = False, [], 0, 0
                pre_roll.clear()
                continue
            voiced = self.vad.is_speech(frame, RATE)
            if not speaking:
                pre_roll.append(frame)
                voiced_run = voiced_run + 1 if voiced else 0
                if voiced_run >= 4:  # ~120 ms of speech
                    speaking, buf, silent_run = True, list(pre_roll), 0
                    self.on_event("speech_start", {"t": time.monotonic()})
                continue
            buf.append(frame)
            silent_run = 0 if voiced else silent_run + 1
            if silent_run >= end_frames or len(buf) >= max_frames:
                t_end = time.monotonic()
                audio = np.frombuffer(b"".join(buf), np.int16).astype(np.float32) / 32768
                speaking, voiced_run = False, 0
                pre_roll.clear()
                if len(audio) > RATE * 0.4:
                    self._utterances.put((audio, t_end))
                    self.on_event("speech_end", {"t": t_end, "dur": len(audio) / RATE})
                else:
                    self.on_event("speech_cancelled", {})

    def _transcribe(self) -> None:
        while not self._stop.is_set():
            try:
                audio, t_end = self._utterances.get(timeout=0.5)
            except queue.Empty:
                continue
            self.ready.wait()
            t0 = time.perf_counter()
            segments, _ = self._model.transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                                 condition_on_previous_text=False, without_timestamps=True)
            text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()
            stt_ms = (time.perf_counter() - t0) * 1000
            norm = re.sub(r"[^a-z ]", "", text.lower()).strip()
            if norm in HALLUCINATIONS:
                self.on_event("speech_cancelled", {"text": text})
                continue
            self.on_event("transcript", {"text": text, "t_end": t_end, "stt_ms": stt_ms})
