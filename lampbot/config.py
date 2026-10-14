"""Runtime configuration. Every value can be overridden with an environment variable
(LAMP_<NAME>), or in a .env file at the project root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
URDF_PATH = ROOT / "robot" / "dummy_lamp_5dof.urdf"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Config:
    # --- processes / protocol ---
    body_host: str = "127.0.0.1"
    body_port: int = 8765

    # --- camera / attention ---
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_hfov_deg: float = 70.0  # typical laptop webcam
    attention_hz: float = 15.0
    engage_after_s: float = 0.6  # sustained attention before engaging
    disengage_after_s: float = 2.0  # sustained inattention before disengaging
    max_yaw_ratio: float = 0.32  # |nose offset| / eye distance that still counts as "looking"
    min_face_px: int = 40  # ignore faces smaller than this (person too far)
    wave_window_s: float = 1.6  # a wave must happen within this window...
    wave_reversals: int = 3  # ...with at least this many side-to-side direction changes
    wave_cooldown_s: float = 5.0
    hum_after_s: float = 5.0  # start humming after this long without anyone looking

    # --- hearing ---
    mic_device: str = ""  # empty = system default
    whisper_model: str = "base.en"
    vad_aggressiveness: int = 2
    end_of_speech_s: float = 0.7
    max_utterance_s: float = 12.0

    # --- voice / audio out ---
    speaker_device: str = ""
    piper_voice: str = "en_US-amy-medium"
    music_volume: float = 0.35
    sfx_volume: float = 0.6

    # --- mind (cloud) ---
    model: str = "claude-opus-5"
    effort: str = "low"  # latency matters more than depth for live chat
    offline: bool = False  # force local-only behaviour (no cloud calls)
    observe_every_s: float = 25.0  # background scene observation while engaged
    image_width: int = 640  # frames are downscaled before upload

    # --- physical layout (lamp base frame, metres) ---
    # Where the laptop camera sits relative to the lamp base. The lamp faces -x.
    camera_pos: tuple = (-0.05, 0.30, 0.22)
    face_assumed_depth_m: float = 0.65
    object_assumed_depth_m: float = 0.45

    # --- logging ---
    run_dir: str = ""  # filled in at start-up

    extra: dict = field(default_factory=dict)


def load_config() -> Config:
    _load_dotenv()
    cfg = Config()
    for f in fields(cfg):
        raw = os.environ.get(f"LAMP_{f.name.upper()}")
        if raw is None or f.name == "extra":
            continue
        cur = getattr(cfg, f.name)
        if isinstance(cur, bool):
            val = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, tuple):
            val = tuple(float(x) for x in raw.split(","))
        else:
            val = raw
        setattr(cfg, f.name, val)
    return cfg
