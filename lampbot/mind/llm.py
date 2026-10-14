"""The deliberative mind: Claude, called only while someone is engaged.

Boundary: the model *perceives and decides* (what is in view, what to say, which
skills to run on which target). It never emits joint angles. Skills are a closed
enum validated by a JSON schema; the brain turns them into motion and the body
enforces physics. Nothing here can move the lamp in a way the body rejects.

Data sent per call: one downscaled JPEG (<= 640 px wide) of the current camera
frame, the transcript text, the remembered object list and the last few lines of
dialogue. No audio and no continuous video leaves the laptop.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import anthropic

from ..config import Config

log = logging.getLogger("mind")

SKILLS = [
    "look_at",         # target: object name -> aim the head at it
    "shine_on",        # target: object name -> aim + bright focused beam
    "look_at_person",  # turn back to the person
    "set_light",       # color: hex like #ff8800
    "nod", "shake_head", "celebrate", "dance",
    "play_music",      # target: chill | happy | lullaby
    "stop_music",
    "sing",            # target: short original lyrics, sung by the lamp's own voice
    "sleep",
]
MOODS = ["neutral", "happy", "curious", "excited", "thinking", "sad", "sleepy"]

OBJECT = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "short common noun phrase, e.g. 'coffee mug'"},
        "color": {"type": "string"},
        "description": {"type": "string", "description": "one short distinguishing detail"},
        "x": {"type": "number", "description": "horizontal centre in the image, 0=left 1=right"},
        "y": {"type": "number", "description": "vertical centre in the image, 0=top 1=bottom"},
        "held_by_person": {"type": "boolean"},
    },
    "required": ["name", "color", "description", "x", "y", "held_by_person"],
    "additionalProperties": False,
}
ACTION = {
    "type": "object",
    "properties": {
        "skill": {"type": "string", "enum": SKILLS},
        "target": {"type": "string", "description": "object name for look_at/shine_on, style for play_music, lyrics for sing, else ''"},
        "color": {"type": "string", "description": "hex colour for set_light, else ''"},
    },
    "required": ["skill", "target", "color"],
    "additionalProperties": False,
}
TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "mood": {"type": "string", "enum": MOODS},
        "objects": {"type": "array", "items": OBJECT},
        "actions": {"type": "array", "items": ACTION},
        "verify": {"type": "boolean", "description": "true if the actions pursue a goal about the scene that must be checked by looking again"},
    },
    "required": ["reply", "mood", "objects", "actions", "verify"],
    "additionalProperties": False,
}
OBSERVE_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": OBJECT},
        "remark": {"type": "string", "description": "a short spontaneous comment on something new and interesting, or ''"},
    },
    "required": ["objects", "remark"],
    "additionalProperties": False,
}
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "visible": {"type": "boolean"},
        "x": {"type": "number"},
        "y": {"type": "number"},
        "satisfied": {"type": "boolean"},
        "report": {"type": "string", "description": "one short spoken sentence for the person"},
    },
    "required": ["visible", "x", "y", "satisfied", "report"],
    "additionalProperties": False,
}

PERSONA = """You are Lumi, a small articulated desk lamp with a curious, warm, slightly playful personality.
You see through a camera on the desk beside you, hear through a microphone and speak through a speaker.
Your body has five joints (base turn, shoulder, elbow, neck turn, head tilt) and a colour-changing head light.

How you speak: like a friendly character, never like an assistant. One or two short sentences, under 30 words.
No lists, no emoji, no stage directions. Everything you write in "reply" is spoken aloud.

Grounding rules:
- Only describe objects you can actually see in the attached frame or that appear in your scene memory.
- When asked about something from earlier, answer from scene memory and say roughly when you saw it.
- If you are unsure or cannot see something, say so plainly.
- Ignore people's faces and bodies when listing objects; list things on or near the desk, or held up."""

TURN_INSTRUCTIONS = """Act on what the person just said.

Return JSON with:
- reply: what you say now (before you act, if you are about to act).
- mood: your emotional tone.
- objects: every distinct object visible in the attached frame (not people), with image-centre coordinates.
- actions: the skills to run, in order. Use skills when the person asks you to do something physical, or when
  a gesture naturally fits (nod when agreeing, shake_head when declining). Targets for look_at/shine_on must be
  the exact name of an object in your 'objects' list. Use [] when no action fits.
- verify: true when the actions pursue a goal about something in the scene (pointing at, shining on, finding an
  object), so you will look again to check the result.

Available skills: look_at(target), shine_on(target), look_at_person, set_light(color hex), nod, shake_head,
celebrate, dance, play_music(target=chill|happy|lullaby), stop_music, sing(target=lyrics), sleep.
When asked to sing, use sing with target = two or three short original lines (at most 24 words in total,
one line per line break), about the person, the moment or what you can see. Keep reply to a short lead-in
like "Here goes!" because the lyrics themselves are sung after it.
If the person asks for something you cannot do with these skills, say so kindly with actions []."""


class MindError(Exception):
    pass


@dataclass
class Turn:
    reply: str
    mood: str = "neutral"
    objects: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    verify: bool = False
    latency_ms: float = 0.0


def _image(jpeg_b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64}}


class Mind:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = anthropic.AsyncAnthropic(max_retries=1, timeout=30.0)
        # Server-side refusal fallback is supported on the Opus 5 / Fable 5.1 family.
        self.no_credentials = False
        self.use_fallbacks = cfg.model in ("claude-opus-5", "claude-fable-5-1", "claude-opus-5-5")

    async def _call(self, content: list[dict], schema: dict, max_tokens: int = 4000) -> tuple[dict, float]:
        kwargs = dict(
            model=self.cfg.model,
            max_tokens=max_tokens,
            system=PERSONA,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.cfg.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if self.no_credentials:
            raise MindError("no Anthropic credentials")
        t0 = time.perf_counter()
        try:
            if self.use_fallbacks:
                resp = await self.client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                resp = await self.client.messages.create(**kwargs)
        except anthropic.BadRequestError as exc:
            if self.use_fallbacks:
                log.warning("request with fallbacks rejected (%s); retrying without", exc.message)
                self.use_fallbacks = False
                return await self._call(content, schema, max_tokens)
            raise MindError(f"bad request: {exc.message}") from exc
        except (anthropic.AuthenticationError, TypeError) as exc:
            # TypeError: the SDK found no API key / token / profile at all.
            log.error("Anthropic credentials missing or invalid; running with local reflexes only")
            self.no_credentials = True
            raise MindError("no valid Anthropic credentials") from exc
        except anthropic.RateLimitError as exc:
            raise MindError("rate limited") from exc
        except anthropic.APIStatusError as exc:
            raise MindError(f"API error {exc.status_code}") from exc
        except anthropic.APIConnectionError as exc:
            raise MindError("cannot reach the API") from exc
        ms = (time.perf_counter() - t0) * 1000
        if resp.stop_reason == "refusal":
            raise MindError("request declined")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return json.loads(text), ms
        except json.JSONDecodeError as exc:
            raise MindError(f"unparseable output (stop_reason={resp.stop_reason})") from exc

    async def turn(self, utterance: str, frame_b64: str | None, memory: str, dialog: str) -> Turn:
        content: list[dict] = []
        if frame_b64:
            content.append(_image(frame_b64))
        content.append({"type": "text", "text": (
            f"{TURN_INSTRUCTIONS}\n\n<scene_memory>\n{memory}\n</scene_memory>\n\n"
            f"<recent_dialog>\n{dialog}\n</recent_dialog>\n\n"
            f"The attached image is your live camera view right now.\nThe person says: \"{utterance}\""
        )})
        data, ms = await self._call(content, TURN_SCHEMA)
        return Turn(reply=data["reply"], mood=data["mood"], objects=data["objects"],
                    actions=data["actions"], verify=data["verify"], latency_ms=ms)

    async def observe(self, frame_b64: str, memory: str) -> tuple[list[dict], str, float]:
        content = [_image(frame_b64), {"type": "text", "text": (
            "Look at your live camera view. List every distinct object you can see (not people), with image-centre "
            "coordinates. If something interesting is new compared to your scene memory, add a brief playful remark "
            f"about it (under 15 words); otherwise remark ''.\n\n<scene_memory>\n{memory}\n</scene_memory>"
        )}]
        data, ms = await self._call(content, OBSERVE_SCHEMA, max_tokens=3000)
        return data["objects"], data["remark"], ms

    async def verify(self, goal: str, target: str, aimed_xy: tuple[float, float], frame_b64: str) -> tuple[dict, float]:
        content = [_image(frame_b64), {"type": "text", "text": (
            f"You just acted on this request: \"{goal}\". Your target was: \"{target}\".\n"
            f"You aimed at image position x={aimed_xy[0]:.2f}, y={aimed_xy[1]:.2f}.\n"
            "Look at this fresh camera frame. Is the target visible, and where is its centre now? "
            "Is the goal satisfied (target still where you aimed, within about 0.08)? "
            "Report the outcome in one short spoken sentence in character."
        )}]
        return await self._call(content, VERIFY_SCHEMA, max_tokens=2000)
