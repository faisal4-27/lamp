"""Procedurally synthesised sound effects and music.

Everything is generated with numpy at start-up, so there are no audio assets to
license or ship, and every cue shares one tonal palette (C major, bell/pluck
timbres) which makes the character sound like one coherent thing.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .mixer import RATE

A4 = 440.0
TAIL = 2.0  # seconds of release rendered past a loop's end, then folded back
NOTE = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def hz(name: str) -> float:
    """'E5' -> frequency."""
    pitch, octave = name[:-1], int(name[-1])
    return A4 * 2 ** ((NOTE[pitch] + 12 * (octave + 1) - 69) / 12)


def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * RATE)) / RATE


def bell(f: float, dur: float = 0.9, bright: float = 1.0) -> np.ndarray:
    t = _t(dur)
    partials = [(1.0, 1.0, 3.0), (2.0, 0.5 * bright, 5.0), (3.01, 0.25 * bright, 7.0), (4.2, 0.12 * bright, 10.0)]
    out = sum(a * np.sin(2 * np.pi * f * m * t) * np.exp(-d * t) for m, a, d in partials)
    attack = np.minimum(1.0, t / 0.004)
    return (out * attack * 0.35).astype(np.float32)


def pluck(f: float, dur: float = 0.5) -> np.ndarray:
    t = _t(dur)
    out = sum((0.6 ** k) * np.sin(2 * np.pi * f * (k + 1) * t) * np.exp(-(4 + 3 * k) * t) for k in range(5))
    return (out * np.minimum(1.0, t / 0.003) * 0.3).astype(np.float32)


def pad(freqs: list[float], dur: float) -> np.ndarray:
    t = _t(dur)
    out = np.zeros_like(t)
    for f in freqs:
        for detune in (-0.6, 0.6):
            out += np.sin(2 * np.pi * (f + detune) * t) + 0.25 * np.sin(2 * np.pi * 2 * (f + detune) * t)
    env = np.minimum(1.0, t / 0.4) * np.minimum(1.0, (dur - t) / 0.5)
    return (out * env * 0.05).astype(np.float32)


def kick(dur: float = 0.3) -> np.ndarray:
    t = _t(dur)
    f = 45 + 90 * np.exp(-t * 30)
    return (np.sin(2 * np.pi * np.cumsum(f) / RATE) * np.exp(-t * 12) * 0.5).astype(np.float32)


def hat(dur: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(7)
    n = rng.standard_normal(int(dur * RATE))
    n = np.diff(n, prepend=0)  # crude high-pass
    return (n * np.exp(-_t(dur) * 80) * 0.05).astype(np.float32)


def glide(f0: float, f1: float, dur: float, vol: float = 0.25) -> np.ndarray:
    t = _t(dur)
    f = f0 * (f1 / f0) ** (t / dur)
    phase = 2 * np.pi * np.cumsum(f) / RATE
    env = np.sin(np.pi * t / dur) ** 0.7
    return ((np.sin(phase) + 0.2 * np.sin(2 * phase)) * env * vol).astype(np.float32)


def hum_voice(notes: list[tuple[float | None, float]]) -> np.ndarray:
    """Closed-mouth 'mmm': strong fundamental, few soft harmonics, legato glides,
    delayed vibrato, and a breath (None) wherever the tune rests."""
    n = int(sum(d for _, d in notes) * RATE)
    freq = np.zeros(n, np.float32)
    amp = np.zeros(n, np.float32)
    vib = np.zeros(n, np.float32)
    i, last = 0, notes[0][0] or 220.0
    for f, d in notes:
        k = int(d * RATE)
        t = np.arange(k) / RATE
        freq[i:i + k] = f or last
        if f:
            amp[i:i + k] = np.minimum(1.0, t / 0.09) * np.minimum(1.0, (d - t) / 0.12) * (1 - 0.15 * t / d)
            vib[i:i + k] = np.clip((t - 0.25) / 0.3, 0, 1)  # vibrato blooms after the onset
            last = f
        i += k
    glide = int(0.07 * RATE)  # portamento between notes
    freq = np.convolve(np.pad(freq, glide, mode="edge"), np.ones(glide) / glide, "same")[glide:-glide]
    t = np.arange(n) / RATE
    freq = freq * (1 + 0.005 * vib * np.sin(2 * np.pi * 5.2 * t))
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    out = sum(a * np.sin(m * phase) for m, a in [(1, 1.0), (2, 0.4), (3, 0.16), (4, 0.07), (5, 0.03)])
    return (out * amp * 0.22).astype(np.float32)


def place(track: np.ndarray, clip: np.ndarray, at: float) -> None:
    i = int(at * RATE)
    j = min(len(track), i + len(clip))
    if i < len(track):
        track[i:j] += clip[: j - i]


def sequence(events: list[tuple[float, np.ndarray]], tail: float = 0.6) -> np.ndarray:
    end = max(at + len(c) / RATE for at, c in events) + tail
    out = np.zeros(int(end * RATE), np.float32)
    for at, c in events:
        place(out, c, at)
    return out


# ------------------------------------------------------------------ effects
@lru_cache(maxsize=None)
def sfx(name: str) -> np.ndarray:
    if name == "notice":  # bright two-note "oh!"
        return sequence([(0.0, bell(hz("E5"), 0.8)), (0.11, bell(hz("B5"), 1.0))])
    if name == "listen":  # tiny cue that the mic is open
        return sequence([(0.0, pluck(hz("G5"), 0.25))], tail=0.05)
    if name == "think":
        return sequence([(0.0, pluck(hz("C5"), 0.3)), (0.12, pluck(hz("E5"), 0.3)),
                         (0.24, pluck(hz("D5"), 0.3))], tail=0.1) * 0.6
    if name == "success":
        notes = ["C6", "E6", "G6", "C7"]
        return sequence([(i * 0.07, bell(hz(n), 0.9, 1.2)) for i, n in enumerate(notes)])
    if name == "sad":
        return sequence([(0.0, glide(hz("E5"), hz("C5"), 0.35)), (0.32, glide(hz("D5"), hz("A4"), 0.55))])
    if name == "sleepy":
        return glide(hz("G5"), hz("C4"), 1.2, 0.18)
    if name == "curious":
        return glide(hz("C5"), hz("A5"), 0.35, 0.2)
    if name == "error":
        return sequence([(0.0, pluck(hz("D4"), 0.3)), (0.15, pluck(hz("C#4"), 0.4))])
    raise KeyError(name)


# ------------------------------------------------------------------ music
CHORDS = {
    "Cmaj7": ["C3", "E3", "G3", "B3"], "Am7": ["A2", "C3", "E3", "G3"],
    "Fmaj7": ["F2", "A2", "C3", "E3"], "G6": ["G2", "B2", "D3", "E3"],
    "C": ["C3", "E3", "G3"], "G": ["G2", "B2", "D3"], "Am": ["A2", "C3", "E3"], "F": ["F2", "A2", "C3"],
}


@lru_cache(maxsize=None)
def music(style: str) -> np.ndarray:
    """Seamless loops (except 'jingle')."""
    if style == "chill":  # 90 bpm lo-fi bed for when the lamp is content / idle with company
        beat = 60 / 90
        prog = ["Cmaj7", "Am7", "Fmaj7", "G6"]
        bar = 4 * beat
        out = np.zeros(int((4 * bar + TAIL) * RATE), np.float32)
        for b, ch in enumerate(prog):
            notes = [hz(n) for n in CHORDS[ch]]
            place(out, pad(notes, bar + 0.3), b * bar)
            arp = notes[1:] + [notes[0] * 2, notes[2] * 2]
            for i in range(8):
                place(out, pluck(arp[i % len(arp)] * 2, 0.45) * 0.7, b * bar + i * beat / 2)
            for k in range(4):
                if k % 2 == 0:
                    place(out, kick(), b * bar + k * beat)
                place(out, hat(), b * bar + k * beat + beat / 2)
        return _seamless(out, 4 * bar)
    if style == "happy":  # 120 bpm bouncy loop for dancing / celebrating
        beat = 60 / 120
        prog = ["C", "G", "Am", "F"]
        bar = 4 * beat
        melody = ["E5", "G5", "C6", "G5", "D5", "G5", "B5", "G5", "C5", "E5", "A5", "E5", "C5", "F5", "A5", "C6"]
        out = np.zeros(int((4 * bar + TAIL) * RATE), np.float32)
        for b, ch in enumerate(prog):
            notes = [hz(n) for n in CHORDS[ch]]
            for k in range(4):
                place(out, kick(0.25), b * bar + k * beat)
                place(out, hat(), b * bar + k * beat + beat / 2)
                place(out, pluck(notes[0], 0.3) * 1.2, b * bar + k * beat + beat / 2)
                place(out, bell(hz(melody[b * 4 + k]), 0.5, 0.8) * 0.8, b * bar + k * beat)
            place(out, pad(notes, bar + 0.2) * 0.8, b * bar)
        return _seamless(out, 4 * bar)
    if style == "lullaby":  # slow music box the lamp hums to itself when alone
        beat = 60 / 72
        tune = ["E5", "G5", "C6", "B5", "A5", "G5", "E5", "D5", "C5", "E5", "G5", "A5", "G5", "E5", "D5", "C5"]
        out = np.zeros(int((len(tune) * beat + TAIL) * RATE), np.float32)
        for i, n in enumerate(tune):
            place(out, bell(hz(n), 1.4, 1.6) * 0.7, i * beat)
            if i % 4 == 0:
                place(out, pad([hz(x) for x in CHORDS[["C", "Am", "F", "G"][i // 4]]], 4 * beat + 0.3) * 0.7, i * beat)
        return _seamless(out, len(tune) * beat)
    if style == "hum":  # the lamp humming to itself once nobody is paying attention
        beat = 60 / 76
        tune = [("E4", 1), ("G4", 1), ("C5", 2), ("B4", 1), ("A4", 1), ("G4", 2),
                ("E4", 1), ("D4", 1), ("C4", 2), ("D4", 1), ("E4", 1), ("D4", 3), (None, 1)]
        out = hum_voice([(None if n is None else hz(n), k * beat) for n, k in tune])
        bed = np.zeros_like(out)
        for i, ch in enumerate(["C", "G", "F", "G"]):
            place(bed, pad([hz(x) for x in CHORDS[ch]], 4 * beat + 0.3) * 0.35, i * 4 * beat)
        return _seamless(np.concatenate([out + bed[:len(out)], np.zeros(int(TAIL * RATE), np.float32)]),
                         len(out) / RATE)
    if style == "jingle":  # one-shot "ta-da" for a completed goal
        beat = 60 / 140
        notes = [("C5", 0), ("E5", 1), ("G5", 2), ("C6", 3), ("G5", 4.5), ("C6", 5)]
        ev = [(k * beat / 2, bell(hz(n), 0.8, 1.2)) for n, k in notes]
        ev += [(0.0, pad([hz(n) for n in CHORDS["C"]], 1.6))]
        return sequence(ev, tail=0.8)
    raise KeyError(style)


def _seamless(track: np.ndarray, length_s: float) -> np.ndarray:
    """Fold the release tail of the last bar back onto the start so the loop has no seam."""
    n = int(length_s * RATE)
    loop = track[:n].copy()
    tail = track[n:]
    loop[: len(tail)] += tail
    return np.tanh(loop * 1.2).astype(np.float32)
