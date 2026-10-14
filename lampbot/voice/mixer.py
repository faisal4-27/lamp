"""Single output stream that mixes voice, sound effects and music.

One stream (rather than one per sound) keeps latency predictable and lets the
character do things a real performer would: duck the music under speech, and
expose the live voice envelope so the head light can "talk" in sync.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

log = logging.getLogger("mixer")
RATE = 22050
BLOCK = 512


@dataclass(eq=False)  # identity comparison; samples are numpy arrays
class Voice:
    samples: np.ndarray
    channel: str
    gain: float = 1.0
    loop: bool = False
    pos: int = 0
    fade: float = 1.0
    fade_step: float = 0.0  # per-sample change; negative = fading out
    done: asyncio.Future | None = None
    loop_ref: asyncio.AbstractEventLoop | None = field(default=None, repr=False)


class Mixer:
    def __init__(self, device: str = "", music_volume: float = 0.35, sfx_volume: float = 0.6):
        self.device = int(device) if device.isdigit() else (device or None)
        self.gains = {"voice": 1.0, "sfx": sfx_volume, "music": music_volume}
        self._active: list[Voice] = []
        self._lock = threading.Lock()
        self._duck = 1.0
        self.voice_level = 0.0
        self.first_audio_t: float | None = None

    def start(self) -> None:
        self._stream = sd.OutputStream(samplerate=RATE, blocksize=BLOCK, channels=1, dtype="float32",
                                       device=self.device, callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()

    # ------------------------------------------------------------ control
    def play(self, samples: np.ndarray, channel: str, gain: float = 1.0, loop: bool = False,
             fade_in: float = 0.0) -> asyncio.Future:
        aloop = asyncio.get_running_loop()
        done = aloop.create_future()
        v = Voice(samples.astype(np.float32), channel, gain, loop, done=done, loop_ref=aloop)
        if fade_in > 0:
            v.fade, v.fade_step = 0.0, 1.0 / (fade_in * RATE)
        with self._lock:
            self._active.append(v)
        return done

    def stop_channel(self, channel: str, fade_out: float = 0.4) -> None:
        with self._lock:
            for v in self._active:
                if v.channel == channel:
                    v.fade_step = -1.0 / max(fade_out * RATE, 1)

    def busy(self, *channels: str) -> bool:
        with self._lock:
            return any(v.channel in channels for v in self._active)

    # ------------------------------------------------------------ audio thread
    def _callback(self, out, frames, t, status) -> None:
        mix = np.zeros(frames, np.float32)
        voice_bus = np.zeros(frames, np.float32)
        finished: list[Voice] = []
        with self._lock:
            speaking = any(v.channel == "voice" for v in self._active)
            target = 0.3 if speaking else 1.0
            self._duck += (target - self._duck) * 0.15
            for v in self._active:
                chunk = self._take(v, frames)
                if v.fade_step:
                    ramp = np.clip(v.fade + v.fade_step * np.arange(frames), 0, 1).astype(np.float32)
                    v.fade = float(ramp[-1])
                    chunk = chunk * ramp
                    if v.fade_step > 0 and v.fade >= 1:
                        v.fade_step = 0.0
                    elif v.fade_step < 0 and v.fade <= 0:
                        v.pos = len(v.samples)
                        v.loop = False
                g = v.gain * self.gains.get(v.channel, 1.0)
                if v.channel == "music":
                    g *= self._duck
                mix[: len(chunk)] += chunk * g
                if v.channel == "voice":
                    voice_bus[: len(chunk)] += chunk
                if v.pos >= len(v.samples) and not v.loop:
                    finished.append(v)
            for v in finished:
                self._active.remove(v)
        rms = float(np.sqrt(np.mean(voice_bus**2))) if speaking else 0.0
        self.voice_level = 0.6 * self.voice_level + 0.4 * min(1.0, rms * 6)
        out[:, 0] = np.tanh(mix)  # soft clip instead of hard wrap-around
        for v in finished:
            if v.done and v.loop_ref and not v.done.done():
                v.loop_ref.call_soon_threadsafe(lambda f=v.done: f.done() or f.set_result(True))

    @staticmethod
    def _take(v: Voice, frames: int) -> np.ndarray:
        end = v.pos + frames
        if end <= len(v.samples):
            chunk = v.samples[v.pos:end]
            v.pos = end
            return chunk
        chunk = v.samples[v.pos:]
        if v.loop:
            rest = frames - len(chunk)
            v.pos = rest % len(v.samples)
            reps = np.resize(v.samples, rest)  # wraps around the loop
            return np.concatenate([chunk, reps])
        v.pos = len(v.samples)
        return chunk
