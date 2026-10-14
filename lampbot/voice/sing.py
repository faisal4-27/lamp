"""Make the speaking voice sing: one word per melody note over a soft backing.

Piper has no pitch control, so each word is synthesised on its own, its median
pitch is measured (autocorrelation), it is resampled so that pitch lands on the
melody note, and then time-stretched (WSOLA, which keeps pitch) to fill the beat.
The result sounds like a small robot singing, which suits the character, and it
needs no singing-voice model.
"""

from __future__ import annotations

import re

import numpy as np

from .mixer import RATE
from .sounds import CHORDS, hz, pad, place, pluck

BPM = 100
# Kept near the voice's own register (~200 Hz) so resampling barely moves the formants.
MELODY = ["G3", "A3", "B3", "D4", "C4", "B3", "A3", "G3",
          "A3", "B3", "C4", "E4", "D4", "C4", "B3", "C4"]
PROGRESSION = ["C", "G", "Am", "F"]
DEFAULT_LYRICS = "I'm a little lamp, I glow for you\nshining on your desk the whole day through"
MAX_WORDS = 32


def f0(x: np.ndarray, lo: float = 80.0, hi: float = 400.0) -> float | None:
    """Median fundamental frequency of the voiced frames, or None if unvoiced."""
    n, hop = 1024, 256
    lags = np.arange(int(RATE / hi), int(RATE / lo))
    found = []
    for i in range(0, len(x) - n, hop):
        fr = x[i:i + n] * np.hanning(n)
        if np.sqrt(np.mean(fr**2)) < 0.02:
            continue
        ac = np.correlate(fr, fr, "full")[n - 1:]
        if ac[0] <= 0:
            continue
        k = int(lags[np.argmax(ac[lags])])
        if ac[k] / ac[0] > 0.45:  # clearly periodic
            a, b, c = ac[k - 1], ac[k], ac[k + 1]  # parabolic peak for sub-sample lag
            shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
            found.append(RATE / (k + float(np.clip(shift, -0.5, 0.5))))
    return float(np.median(found)) if found else None


def resample(x: np.ndarray, ratio: float) -> np.ndarray:
    """Play back `ratio` times faster (pitch x ratio, length / ratio)."""
    n = max(1, int(len(x) / ratio))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def stretch(x: np.ndarray, length: int, n: int = 1024, tol: int = 256) -> np.ndarray:
    """WSOLA time-stretch to `length` samples without changing pitch."""
    if len(x) < n * 2 or length <= 0:
        return np.resize(x, length) if len(x) else np.zeros(length, np.float32)
    hs = n // 2
    win = np.hanning(n).astype(np.float32)
    ha = (len(x) - n) / max(1, (length - n) / hs)
    xp = np.concatenate([np.zeros(tol, np.float32), x, np.zeros(n + tol, np.float32)])
    out = np.zeros(length + n, np.float32)
    norm = np.zeros(length + n, np.float32)
    prev = 0  # input position (in xp) of the last copied frame
    k = 0
    while k * hs < length:
        nominal = int(k * ha) + tol
        if k == 0:
            pos = nominal
        else:
            # Pick the offset whose frame best continues the previously copied one.
            natural = xp[prev + hs:prev + hs + n]
            best, pos = -np.inf, nominal
            for d in range(-tol, tol + 1, 16):
                c = nominal + d
                if c < 0 or c + n > len(xp):
                    continue
                score = float(np.dot(natural, xp[c:c + n]))
                if score > best:
                    best, pos = score, c
        frame = xp[pos:pos + n]
        if len(frame) < n:
            break
        out[k * hs:k * hs + n] += frame * win
        norm[k * hs:k * hs + n] += win
        prev = pos
        k += 1
    return (out[:length] / np.maximum(norm[:length], 1e-3)).astype(np.float32)


def trim(x: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    env = np.convolve(np.abs(x), np.ones(256) / 256, "same")
    idx = np.where(env > thresh)[0]
    return x[idx[0]:idx[-1] + 1] if len(idx) else x[:0]


def sing_note(word_audio: np.ndarray, note_hz: float, dur_s: float) -> np.ndarray:
    w = trim(word_audio)
    if len(w) < 200:
        return np.zeros(int(dur_s * RATE), np.float32)
    p = f0(w)
    if p:
        w = resample(w, float(np.clip(note_hz / p, 0.6, 1.8)))
        p = f0(w)  # short words measure noisily; one corrective pass fixes most of it
        if p:
            w = resample(w, float(np.clip(note_hz / p, 0.85, 1.18)))
    out = stretch(w, int(dur_s * RATE * 0.92))
    env = np.ones(len(out), np.float32)
    a, r = min(len(out), int(0.015 * RATE)), min(len(out), int(0.06 * RATE))
    env[:a] = np.linspace(0, 1, a)
    env[len(out) - r:] *= np.linspace(1, 0, r)
    peak = float(np.max(np.abs(out))) or 1.0
    return out * env * (0.8 / peak)


def words(lyrics: str) -> list[list[str]]:
    lines = [re.findall(r"[A-Za-z']+", ln) for ln in re.split(r"[\n.;!?]+", lyrics or "")]
    lines = [ln for ln in lines if ln] or words(DEFAULT_LYRICS)
    out, total = [], 0
    for ln in lines:
        ln = ln[:MAX_WORDS - total]
        if ln:
            out.append(ln)
            total += len(ln)
    return out


def render(lyrics: str, synth) -> tuple[np.ndarray, float]:
    """Return the sung song and its beat length. `synth(text) -> float32 audio at RATE`."""
    beat = 60.0 / BPM
    voice_events: list[tuple[float, np.ndarray]] = []
    t, i = 0.0, 0
    for line in words(lyrics):
        for j, word in enumerate(line):
            last = j == len(line) - 1
            dur = beat * (2 if last else 1)  # hold the end of each line
            voice_events.append((t, sing_note(synth(word), hz(MELODY[i % len(MELODY)]), dur)))
            t += dur
            i += 1
        t += beat  # breath between lines
    bars = int(np.ceil(t / (4 * beat)))
    total = bars * 4 * beat + 1.0
    out = np.zeros(int(total * RATE), np.float32)
    for at, clip in voice_events:
        place(out, clip, at)
    for b in range(bars):
        ch = [hz(n) for n in CHORDS[PROGRESSION[b % len(PROGRESSION)]]]
        place(out, pad(ch, 4 * beat + 0.3) * 1.4, b * 4 * beat)
        for k in range(4):
            place(out, pluck(ch[k % len(ch)] * 2, 0.4) * 0.5, b * 4 * beat + k * beat)
    return np.tanh(out * 1.1).astype(np.float32), beat
