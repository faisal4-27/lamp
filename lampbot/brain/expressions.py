"""The character's vocabulary of motion and light.

Gestures are keyframe lists built relative to the pose the lamp is currently
holding (usually "looking at you"), so a nod is a nod wherever the head points.
Timings are deliberately unhurried: the body would stretch anything faster than
its velocity limits anyway, and slow-in/slow-out reads as alive rather than robotic.
"""

from __future__ import annotations

from ..body.actuators import REST_POSE

# Light cues: (rgb, brightness, fade seconds, pulse Hz)
LIGHT = {
# Lit (bright) only while someone is looking at the lamp; dark whenever they look away.
    "asleep": ((1.0, 0.62, 0.35), 0.0, 1.0, 0.0),
    "idle": ((1.0, 0.78, 0.50), 0.0, 1.0, 0.0),
    "noticed": ((1.0, 0.85, 0.62), 0.0, 0.8, 0.0),
    "greet": ((1.0, 0.92, 0.75), 1.0, 0.15, 0.0),
    "listening": ((0.78, 0.92, 1.0), 1.0, 0.4, 0.45),
    "thinking": ((1.0, 0.70, 0.30), 0.9, 0.3, 1.3),
    "speaking": ((1.0, 0.90, 0.70), 0.9, 0.2, 0.0),
    "shine": ((1.0, 1.0, 0.95), 1.0, 0.4, 0.0),
}
MOOD_RGB = {
    "neutral": (1.0, 0.90, 0.70), "happy": (1.0, 0.85, 0.45), "curious": (0.72, 0.90, 1.0),
    "excited": (1.0, 0.60, 0.80), "thinking": (1.0, 0.70, 0.30), "sad": (0.45, 0.55, 1.0),
    "sleepy": (1.0, 0.60, 0.35),
}


def _rel(q0: dict[str, float], t: float, **offsets: float) -> dict:
    j = dict(q0)
    for k, v in offsets.items():
        name = {"base": "base_yaw_joint", "shoulder": "shoulder_pitch_joint", "elbow": "elbow_pitch_joint",
                "neck": "neck_yaw_joint", "head": "head_pitch_joint"}[k]
        j[name] = q0.get(name, 0.0) + v
    return {"joints": j, "t": t}


def nod(q: dict) -> list[dict]:
    return [_rel(q, 0.22, head=0.22), _rel(q, 0.22, head=-0.04), _rel(q, 0.22, head=0.16), _rel(q, 0.3)]


def shake_head(q: dict) -> list[dict]:
    return [_rel(q, 0.25, neck=0.28), _rel(q, 0.3, neck=-0.28), _rel(q, 0.3, neck=0.2), _rel(q, 0.3)]


def perk_up(q: dict) -> list[dict]:
    """Greeting: a little startle upwards, then settle facing the person."""
    return [_rel(q, 0.35, shoulder=0.18, elbow=-0.2, head=-0.25), _rel(q, 0.5, head=-0.05), _rel(q, 0.4)]


def wave_back(q: dict) -> list[dict]:
    """Waving back: head lifts, then wags side to side like a little hand."""
    frames = [_rel(q, 0.3, shoulder=0.1, head=-0.2)]
    for i in range(4):
        frames.append(_rel(q, 0.22, shoulder=0.1, head=-0.2, neck=0.3 * (-1) ** i))
    return frames + [_rel(q, 0.35)]


def sway(q: dict, beats: int, beat_s: float) -> list[dict]:
    """Gentle side-to-side sway in time with a song or hum."""
    frames = []
    for i in range(beats):
        s = 1 if i % 2 == 0 else -1
        frames.append(_rel(q, beat_s, base=0.08 * s, neck=0.12 * s, head=-0.06 + 0.04 * s))
    return frames + [_rel(q, 0.5)]


def think(q: dict) -> list[dict]:
    return [_rel(q, 0.6, neck=0.22, head=-0.18)]


def droop(q: dict) -> list[dict]:
    return [_rel(q, 1.2, shoulder=-0.05, head=0.45)]


def celebrate(q: dict) -> list[dict]:
    frames = []
    for i in range(3):
        frames += [_rel(q, 0.2, shoulder=0.12, elbow=-0.2, head=-0.2, base=0.15 * (-1) ** i),
                   _rel(q, 0.2, shoulder=-0.04, elbow=0.06)]
    return frames + [_rel(q, 0.4)]


def dance(q: dict, beats: int = 16, bpm: float = 120) -> list[dict]:
    beat = 60.0 / bpm
    frames = []
    for i in range(beats):
        s = 1 if i % 2 == 0 else -1
        # Amplitudes sized so each beat fits the joints' velocity limits at 120 bpm;
        # bigger swings get stretched by the body and fall off the music's beat.
        frames.append(_rel(q, beat, base=0.14 * s, neck=-0.14 * s, head=0.12 if i % 4 < 2 else -0.12,
                           shoulder=0.06 * s, elbow=-0.08 * s))
    return frames + [_rel(q, 0.5)]


def rest() -> list[dict]:
    return [{"joints": dict(REST_POSE), "t": 2.5}]


def look_around(q: dict, yaw: float, pitch: float) -> list[dict]:
    return [_rel(q, 1.4, base=yaw, head=pitch), {"joints": dict(_rel(q, 0, base=yaw, head=pitch)["joints"]), "t": 1.2}]


def hex_rgb(color: str) -> tuple[float, float, float] | None:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return None
    try:
        return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return None
