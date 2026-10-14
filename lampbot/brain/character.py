"""The character: one state machine that owns every behaviour decision.

Senses publish events; only this class decides what the lamp does about them.
Reflexes (noticing, tracking, greeting cues, listening cues) are local and run in
milliseconds; deliberation (what to say, which skill to run) goes to the mind
and is masked by visible "thinking" behaviour.

    asleep ──face──▶ noticed ──attention / wave──▶ greeting ──▶ listening ◀──┐
       ▲                │ face lost                                │ speech   │
       └──── idle ◀─────┘                                          ▼          │
              ▲                                                thinking       │
              └──── disengaging ◀── attention lost ─────────  speaking/acting ┘

Nobody looking for `hum_after_s` -> it hums to itself (idle / noticed); anyone
looking or waving stops the hum and gets a "Hi there!".
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import random
import re
import time

import numpy as np

from ..config import ROOT, URDF_PATH, Config
from ..kinematics import LampModel
from ..mind.llm import Mind, MindError, Turn
from ..mind.memory import SceneMemory
from ..senses.hearing import Hearing
from ..senses.vision import AttentionState, Vision
from ..voice import sing as singing
from ..voice import sounds
from ..voice.mixer import Mixer
from ..voice.tts import TTS
from . import expressions as ex
from .aim import Aimer
from .body_link import BodyLink
from .metrics import Recorder

log = logging.getLogger("character")

STATE_COLOR = {
    "asleep": "#7a8194", "idle": "#c9b38a", "noticed": "#ffd27a", "greeting": "#ffe9b0",
    "listening": "#9fd3ff", "thinking": "#ffb35c", "speaking": "#fff0c8", "acting": "#c6a2ff",
    "disengaging": "#c9b38a",
}
GREETINGS = ["Hi there!"]
WELCOME_BACK = ["Hi there! Welcome back."]
FAREWELLS = ["Bye-bye!", "See you later!"]
HUM_MAX_S = 45.0  # hum for a while, then stay quiet until attention comes and goes again


class Character:
    def __init__(self, cfg: Config, recorder: Recorder):
        self.cfg = cfg
        self.rec = recorder
        self.loop = asyncio.get_running_loop()
        self.model = LampModel(URDF_PATH)
        self.aimer = Aimer(cfg, self.model)
        self.body = BodyLink(f"ws://{cfg.body_host}:{cfg.body_port}/ws?role=brain")
        self.memory = SceneMemory(ROOT / "data" / "memory.json")
        self.mind = Mind(cfg)
        self.mixer = Mixer(cfg.speaker_device, cfg.music_volume, cfg.sfx_volume)
        self.tts = TTS(cfg)
        self.events: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self.vision = Vision(cfg, lambda kind, st: self._post(kind, {"state": st}))
        self.hearing = Hearing(cfg, lambda kind, data: self._post(kind, data))
        self.body.on_operator = lambda text: self._post("typed", {"text": text})

        self.state = "idle"
        self.aim_q: dict[str, float] = {}  # pose the character is "holding" (gestures are relative to it)
        self.focus_until = 0.0  # while > now, attention stays on an object instead of the face
        self.focus_q: dict[str, float] | None = None
        self.last_face_t = time.monotonic()
        self.last_engaged_t = 0.0
        self.last_observe_t = 0.0
        self.engaged = False
        self.pending_disengage = False
        self.sleep_until = 0.0
        self.user_speaking = False
        self.light_override: tuple[float, float, float] | None = None
        self.music_style: str | None = None
        self.task: asyncio.Task | None = None
        self.gate_hold_until = 0.0
        self.turn_times: dict[str, float] = {}
        self.pending_text: str | None = None
        self.last_attention_t = time.monotonic()  # last moment someone was looking at the lamp
        self.last_greet_t = 0.0
        self.hum_started_t = 0.0
        self.hummed = False  # already hummed since attention was last lost
        self.song: asyncio.Future | None = None  # lyrics rendered while the lead-in is spoken

    def _post(self, kind: str, data: dict) -> None:
        self.loop.call_soon_threadsafe(self.events.put_nowait, (kind, data))

    # ================================================================ lifecycle
    async def run(self) -> None:
        asyncio.create_task(self.body.run())
        self.mixer.start()
        self.vision.start()
        self.hearing.start()
        await asyncio.wait_for(self.body.connected.wait(), timeout=30)
        self.aim_q = dict(self.body.q)
        self.body.hud("layout", camera_pos=list(self.cfg.camera_pos))
        self.body.hud("memory", objects=self.memory.hud())
        self._set_state("idle")
        self.rec.mark("start", model=self.cfg.model, tts=self.tts.engine, whisper=self.cfg.whisper_model)
        await asyncio.gather(self._event_loop(), self._control_loop(), self._hud_loop(), self._idle_loop())

    def _set_state(self, state: str) -> None:
        if state != self.state:
            log.info("state %s -> %s", self.state, state)
            self.rec.mark("state", state=state, prev=self.state)
        self.state = state
        self.body.hud("state", state=state.upper(), color=STATE_COLOR.get(state))
        if state in ex.LIGHT and not (self.light_override and state in ("listening", "speaking")):
            rgb, b, fade, pulse = ex.LIGHT[state]
            self._light(rgb, b, fade, pulse)
        elif self.light_override:
            self._light(self.light_override, 0.95, 0.4, 0.4 if state == "listening" else 0.0)

    def _light(self, rgb, brightness: float, fade: float = 0.3, pulse_hz: float = 0.0) -> None:
        """The head light is lit only while someone is looking at the lamp; otherwise it fades to off."""
        if self.engaged:
            self.body.light(rgb, brightness, fade, pulse_hz)
        else:
            self.body.light(rgb, 0.0, max(fade, 0.8), 0.0)

    def _spawn(self, coro) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = asyncio.create_task(self._guard(coro))

    async def _guard(self, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("behaviour failed")
            self._set_state("listening" if self.engaged else "idle")

    def _busy(self) -> bool:
        return self.task is not None and not self.task.done()

    # ================================================================ events
    async def _event_loop(self) -> None:
        while True:
            kind, data = await self.events.get()
            now = time.monotonic()
            if kind == "engaged":
                self.engaged = True
                self.pending_disengage = False
                self.hummed = False
                self.rec.mark("engaged", face=_face_dict(data["state"]))
                if now < self.sleep_until:
                    continue
                if self.state in ("idle", "noticed", "asleep", "disengaging"):
                    self._spawn(self.greet())  # pre-empts idle glances / dozing off
            elif kind == "disengaged":
                self.engaged = False
                self._light((0.0, 0.0, 0.0), 0.0, 0.8)
                self.last_attention_t = now
                self.hummed = False
                self.rec.mark("disengaged")
                if self._busy() and self.state in ("thinking", "speaking", "acting"):
                    self.pending_disengage = True  # finish the thought first, then let go
                else:
                    self._spawn(self.disengage())
            elif kind == "wave":
                self.rec.mark("wave", state=self.state)
                quiet = self.state == "listening" and not self._busy() and not self.user_speaking
                if self.state in ("idle", "noticed", "asleep", "disengaging") or quiet:
                    self.sleep_until = 0.0  # a wave is an explicit request for attention
                    self._spawn(self.wave_hello())
            elif kind == "face_seen":
                self.last_face_t = now
                if self.state in ("idle", "asleep") and now >= self.sleep_until:
                    self._spawn(self.notice())
            elif kind == "face_lost":
                pass  # idle loop handles the timeout so brief occlusions do not matter
            elif kind == "speech_start":
                self.user_speaking = True
                self.body.hud("attention", score=1.0, engaged=True)
            elif kind == "speech_end":
                self.user_speaking = False
                self.turn_times = {"speech_end": data["t"]}
                if self.state == "listening":
                    self._set_state("thinking")
                    self.body.motion(ex.think(self.aim_q), "think")
            elif kind == "speech_cancelled":
                self.user_speaking = False
                if self.state == "thinking" and not self._busy():
                    self._set_state("listening")
            elif kind == "typed":
                # Typed input skips the microphone but goes through the same turn pipeline.
                if not self._busy() or self.state in ("idle", "noticed", "asleep", "listening"):
                    self.turn_times = {"speech_end": now}
                    self._spawn(self.converse(data["text"], {"stt_ms": 0.0}))
                else:
                    self.pending_text = data["text"]  # handled when the lamp next listens
            elif kind == "transcript":
                text = data["text"]
                if self.state in ("listening", "thinking") and not self._busy():
                    self._spawn(self.converse(text, data))

    # ================================================================ behaviours
    async def notice(self) -> None:
        """Someone is around but not (yet) looking at the lamp."""
        was_asleep = self.state == "asleep"
        self._set_state("noticed")
        if was_asleep:
            self.mixer.stop_channel("music", 1.5)
            self.music_style = None
            await self.sfx("curious")

    async def greet(self) -> None:
        now = time.monotonic()
        recently = now - self.last_engaged_t < 90 and self.last_engaged_t > 0
        just_waved = now - self.last_greet_t < 8  # already said hi to the wave; don't say it twice
        self.last_engaged_t = self.last_greet_t = now
        self._set_state("greeting")
        self.rec.mark("greet", repeat=recently)
        self._stop_humming()
        q = self._face_pose() or self.aim_q
        self.aim_q = q
        motion = self.body.motion(ex.perk_up(q), "greet")
        if not just_waved:
            asyncio.create_task(self.sfx("notice"))
            await asyncio.sleep(0.5)
            await self.say(random.choice(WELCOME_BACK if recently else GREETINGS), mood="happy")
        await motion
        await self.listen()
        if time.monotonic() - self.last_observe_t > 8:
            asyncio.create_task(self.observe(comment=not recently))

    async def wave_hello(self) -> None:
        """Someone waved: wave back and say hi, even if they are not looking right at the lamp."""
        self.last_greet_t = time.monotonic()
        prev = self.state
        self._set_state("greeting")
        self._stop_humming()
        q = self._face_pose() or self.aim_q
        self.aim_q = q
        self._light(ex.LIGHT["greet"][0], 1.0, 0.1, 2.5)
        motion = self.body.motion(ex.wave_back(q), "wave")
        asyncio.create_task(self.sfx("notice"))
        await asyncio.sleep(0.3)
        await self.say("Hi there!", mood="happy")
        await motion
        if self.engaged:
            await self.listen()
        else:
            self._set_state("noticed" if self.vision.state.face is not None or prev == "noticed" else "idle")

    async def listen(self) -> None:
        if self.pending_disengage or not self.engaged:
            await self.disengage()
            return
        self._set_state("listening")
        if self.pending_text:
            text, self.pending_text = self.pending_text, None
            self._post("typed", {"text": text})
            return
        await self.sfx("listen")
        self.gate_hold_until = time.monotonic() + 0.25

    async def disengage(self) -> None:
        self.pending_disengage = False
        if self.state in ("idle", "asleep", "disengaging", "noticed"):
            return
        self._set_state("disengaging")
        self.hearing.gate_open = False
        await self.say(random.choice(FAREWELLS), mood="happy")
        if self.music_style == "happy":
            self.mixer.stop_channel("music", 2.0)
            self.music_style = None
        self.light_override = None
        await self.body.motion(ex.droop(self.aim_q), "disengage")
        self.focus_until = 0.0
        # Still in view but looking elsewhere: keep a polite eye on them.
        self._set_state("noticed" if self.vision.state.face is not None else "idle")

    async def converse(self, text: str, timing: dict) -> None:
        t = self.turn_times
        t["stt_ms"] = timing.get("stt_ms", 0.0)
        self.body.hud("dialog", role="user", text=text)
        self.memory.add_dialog("Person", text)
        self._set_state("thinking")
        asyncio.create_task(self.sfx("think", gain=0.5))
        frame = self._frame_b64()
        t0 = time.monotonic()
        try:
            if self.cfg.offline:
                raise MindError("offline mode")
            turn = await asyncio.wait_for(
                self.mind.turn(text, frame, self.memory.as_prompt(), self.memory.dialog_prompt()), timeout=30)
        except (MindError, asyncio.TimeoutError) as exc:
            log.warning("mind unavailable: %s", exc)
            turn = local_turn(text)
        t["mind_ms"] = (time.monotonic() - t0) * 1000
        if turn.objects:
            self.memory.update(turn.objects)
            self.last_observe_t = time.monotonic()
            self.body.hud("memory", objects=self.memory.hud())
        self.memory.add_dialog("Lumi", turn.reply)
        lyrics = next((a.get("target", "") for a in turn.actions if a["skill"] == "sing"), None)
        if lyrics is not None:
            self.song = asyncio.ensure_future(asyncio.to_thread(singing.render, lyrics, self.tts.synthesize))
        await self.say(turn.reply, mood=turn.mood, measure=True)
        if turn.actions:
            await self.act(turn, text)
        await self.listen()

    async def act(self, turn: Turn, goal: str) -> None:
        self._set_state("acting")
        self.rec.mark("plan", actions=turn.actions, verify=turn.verify)
        aimed: tuple[str, tuple[float, float], bool] | None = None
        for step in turn.actions[:6]:
            skill, target = step["skill"], step.get("target", "")
            self.rec.mark("skill", skill=skill, target=target)
            if skill in ("look_at", "shine_on"):
                obj = _find(turn.objects, target) or self._remembered(target)
                if obj is None:
                    await self.say(f"Hmm, I can't find the {target}.", mood="sad")
                    await self.sfx("error")
                    break
                await self.aim_at_object(obj, shine=skill == "shine_on")
                aimed = (target, (obj["x"], obj["y"]), skill == "shine_on")
            elif skill == "look_at_person":
                self.focus_until = 0.0
                q = self._face_pose()
                if q:
                    self.aim_q = q
                    await self.body.motion([{"joints": q, "t": 0.8}], "face")
            elif skill == "set_light":
                rgb = ex.hex_rgb(step.get("color", ""))
                if rgb:
                    self.light_override = rgb
                    self._light(rgb, 0.95, 0.6)
            elif skill == "nod":
                await self.body.motion(ex.nod(self.aim_q), "nod")
            elif skill == "shake_head":
                await self.body.motion(ex.shake_head(self.aim_q), "shake")
            elif skill == "celebrate":
                asyncio.create_task(self.sfx("success"))
                self._light((1.0, 0.75, 0.4), 1.0, 0.1, 3.0)
                await self.body.motion(ex.celebrate(self.aim_q), "celebrate")
            elif skill == "dance":
                self.play_music("happy")
                self._light((1.0, 0.55, 0.8), 1.0, 0.3, 2.0)
                await self.body.motion(ex.dance(self.aim_q), "dance")
                if self.music_style == "happy":  # the music belongs to the dance; end them together
                    self.mixer.stop_channel("music", 1.5)
                    self.music_style = None
            elif skill == "play_music":
                self.play_music(target if target in ("chill", "happy", "lullaby") else "chill")
            elif skill == "stop_music":
                self.mixer.stop_channel("music", 1.2)
                self.music_style = None
            elif skill == "sing":
                await self.sing_song(target)
            elif skill == "sleep":
                self.sleep_until = time.monotonic() + 20
                await self.go_to_sleep()
                return
        if aimed and turn.verify:
            await self.verify(goal, *aimed)

    async def verify(self, goal: str, target: str, aimed_xy: tuple[float, float], shine: bool) -> None:
        """Look again with a fresh frame; correct the aim if the target moved."""
        for attempt in range(2):
            await asyncio.sleep(0.6)  # let motion settle and the camera catch up
            frame = self._frame_b64()
            if self.cfg.offline or frame is None:
                return
            try:
                res, ms = await asyncio.wait_for(self.mind.verify(goal, target, aimed_xy, frame), timeout=25)
            except (MindError, asyncio.TimeoutError) as exc:
                log.warning("verify failed: %s", exc)
                return
            moved = math.hypot(res["x"] - aimed_xy[0], res["y"] - aimed_xy[1])
            self.rec.mark("verify", attempt=attempt, visible=res["visible"], moved=round(moved, 3),
                          satisfied=res["satisfied"], ms=round(ms))
            if not res["visible"]:
                await self.say(res["report"] or f"I lost sight of the {target}.", mood="sad")
                await self.sfx("sad")
                return
            # The code, not the model, decides whether the aim is good enough.
            if moved <= 0.08:
                asyncio.create_task(self.sfx("success"))
                await self.say(res["report"] or "Got it!", mood="happy")
                return
            obj = {"name": target, "x": res["x"], "y": res["y"], "held_by_person": False}
            self.memory.update([{**obj, "color": "", "description": ""}])
            await self.say("Oh, it moved. Let me follow it.", mood="curious")
            await self.aim_at_object(obj, shine=shine)
            aimed_xy = (res["x"], res["y"])
        await self.say("Close enough, I think!", mood="neutral")

    async def aim_at_object(self, obj: dict, shine: bool) -> None:
        p = self.aimer.object_point(obj["x"], obj["y"], obj.get("held_by_person", False))
        q, posture, err = self.aimer.pose_for(p, self.body.q)
        self.rec.mark("aim", target=obj.get("name"), point=[round(v, 3) for v in p], posture=posture,
                      err_deg=round(math.degrees(err), 1))
        self.body.hud("aim", point=p.tolist(), **{"from": self.aimer.emitter(q).tolist()})
        self.focus_q, self.focus_until = q, time.monotonic() + 12
        self.aim_q = q
        if shine:
            self._light(*ex.LIGHT["shine"][:3])
        await self.body.motion([{"joints": q, "t": 1.0}], "aim")

    async def observe(self, comment: bool) -> None:
        frame = self._frame_b64()
        if frame is None or self.cfg.offline:
            return
        self.last_observe_t = time.monotonic()
        try:
            objects, remark, ms = await asyncio.wait_for(self.mind.observe(frame, self.memory.as_prompt()), 25)
        except (MindError, asyncio.TimeoutError) as exc:
            log.warning("observe failed: %s", exc)
            return
        new = self.memory.update(objects)
        self.rec.mark("observe", n=len(objects), new=new, ms=round(ms))
        self.body.hud("memory", objects=self.memory.hud())
        if comment and remark and new and self.state == "listening" and not self.user_speaking and not self._busy():
            self.memory.add_dialog("Lumi", remark)
            self._spawn(self._remark(remark))

    async def _remark(self, text: str) -> None:
        await self.say(text, mood="curious")
        await self.listen()

    async def sing_song(self, lyrics: str) -> None:
        song, self.song = self.song, None
        if song is None:
            song = asyncio.ensure_future(asyncio.to_thread(singing.render, lyrics, self.tts.synthesize))
        audio, beat = await song
        lines = [" ".join(ln) for ln in singing.words(lyrics)]
        self.body.hud("dialog", role="lamp", text="♪ " + " / ".join(lines) + " ♪")
        self.memory.add_dialog("Lumi", "(sings) " + " / ".join(lines))
        self.rec.mark("sing", words=sum(len(ln.split()) for ln in lines), seconds=round(len(audio) / sounds.RATE, 1))
        self.mixer.stop_channel("music", 0.5)
        self.music_style = None
        motion = self.body.motion(ex.sway(self.aim_q, int(len(audio) / sounds.RATE / beat), beat), "sing")
        await self._perform(audio, ex.MOOD_RGB["excited"])
        await motion

    def _start_humming(self) -> None:
        self.play_music("hum")
        self.hum_started_t = time.monotonic()
        self.hummed = True

    def _stop_humming(self) -> None:
        if self.music_style in ("hum", "lullaby"):
            self.mixer.stop_channel("music", 0.8)
            self.music_style = None

    async def go_to_sleep(self) -> None:
        self._set_state("asleep")
        self.light_override = None
        await self.sfx("sleepy")
        await self.body.motion(ex.rest(), "sleep")
        self.aim_q = dict(self.body.q)
        self._start_humming()  # dozes off humming
        await asyncio.sleep(16)
        if self.state == "asleep":
            self.mixer.stop_channel("music", 4.0)
            self.music_style = None

    # ================================================================ outputs
    async def say(self, text: str, mood: str = "neutral", measure: bool = False) -> None:
        if not text:
            return
        prev = self.state
        self.body.hud("dialog", role="lamp", text=text)
        t0 = time.monotonic()
        audio = await asyncio.to_thread(self.tts.synthesize, text)
        tts_ms = (time.monotonic() - t0) * 1000
        self._set_state("speaking")
        rgb = self.light_override or ex.MOOD_RGB.get(mood, ex.MOOD_RGB["neutral"])
        if measure and "speech_end" in self.turn_times:
            t = self.turn_times
            total = (time.monotonic() - t["speech_end"]) * 1000
            self.rec.mark("turn_latency", total_ms=round(total), stt_ms=round(t.get("stt_ms", 0)),
                          mind_ms=round(t.get("mind_ms", 0)), tts_ms=round(tts_ms))
            self.body.hud("metrics", text=self._metrics_text(total))
        await self._perform(audio, rgb)
        if prev in ("acting", "greeting", "disengaging"):
            self._set_state(prev)  # callers in other states move on to their next state themselves

    async def _perform(self, audio: np.ndarray, rgb: tuple[float, float, float]) -> None:
        """Play on the voice channel (which closes the mic gate) with the head light following the voice."""
        done = self.mixer.play(audio, "voice")
        while not done.done():
            self._light(rgb, 0.7 + 0.3 * self.mixer.voice_level, 0.05)
            await asyncio.sleep(0.05)
        self.gate_hold_until = time.monotonic() + 0.3

    async def sfx(self, name: str, gain: float = 1.0) -> None:
        self.gate_hold_until = time.monotonic() + 1.0
        await self.mixer.play(sounds.sfx(name), "sfx", gain)
        self.gate_hold_until = time.monotonic() + 0.2

    def play_music(self, style: str) -> None:
        if self.music_style == style:
            return
        self.mixer.stop_channel("music", 0.8)
        self.mixer.play(sounds.music(style), "music", loop=True, fade_in=1.0)
        self.music_style = style
        self.rec.mark("music", style=style)

    # ================================================================ continuous loops
    async def _control_loop(self) -> None:
        """10 Hz: face tracking, breathing, and the microphone gate."""
        t = 0.0
        while True:
            await asyncio.sleep(0.1)
            t += 0.1
            now = time.monotonic()
            self.hearing.gate_open = (
                self.engaged and self.state in ("listening", "thinking") and now > self.gate_hold_until
                and not self.mixer.busy("voice", "sfx") and self.hearing.ready.is_set()
            )
            # Duck requested music while listening so it doesn't trip the VAD.
            self.mixer.gains["music"] = self.cfg.music_volume * (0.35 if self.state == "listening" else 1.0)
            breathe = {"shoulder_pitch_joint": 0.025 * math.sin(2 * math.pi * 0.22 * t),
                       "elbow_pitch_joint": -0.02 * math.sin(2 * math.pi * 0.22 * t + 0.6)}
            if self.state == "asleep":
                continue
            if now < self.focus_until and self.focus_q:
                q = dict(self.focus_q)
                speed = 0.4
            elif self.state in ("noticed", "greeting", "listening", "thinking", "speaking"):
                q = self._face_pose()
                if q is None:
                    continue
                self.aim_q = q
                speed = 0.3 if self.state == "noticed" else 0.55
            else:
                continue
            for k, v in breathe.items():
                q[k] = q.get(k, 0.0) + v
            self.body.track(q, speed)

    async def _idle_loop(self) -> None:
        """1 Hz: background observation, losing interest, dozing off, idle glances."""
        next_glance = time.monotonic() + 8
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            st = self.vision.state
            if st.face is not None:
                self.last_face_t = now
            if self.state == "noticed" and not self._busy() and now - self.last_face_t > 3:
                self._set_state("idle")
            # Engagement events are edge-triggered; re-check levels after a requested nap ends.
            if self.state in ("asleep", "noticed", "idle") and now >= self.sleep_until and not self._busy():
                if st.engaged:
                    self.engaged = True
                    self._spawn(self.greet())
                elif self.state == "asleep" and st.face is not None:
                    self._spawn(self.notice())
            if (self.state == "listening" and not self._busy() and not self.user_speaking
                    and now - self.last_observe_t > self.cfg.observe_every_s):
                asyncio.create_task(self.observe(comment=True))
            if self.state in ("idle", "noticed") and not self.engaged and not self._busy():
                if (self.music_style is None and not self.hummed and now >= self.sleep_until
                        and now - self.last_attention_t > self.cfg.hum_after_s):
                    self.rec.mark("hum", alone_s=round(now - self.last_attention_t))
                    self._start_humming()
                elif self.music_style == "hum" and now - self.hum_started_t > HUM_MAX_S:
                    self._stop_humming()
            if self.state == "idle" and not self._busy():
                if now - self.last_face_t > 60:
                    self._spawn(self.go_to_sleep())
                elif now > next_glance:
                    next_glance = now + random.uniform(7, 14)
                    base = dict(self.body.q)
                    self._spawn(self.body.motion(
                        ex.look_around(base, random.uniform(-0.6, 0.6), random.uniform(-0.1, 0.25)), "glance"))

    async def _hud_loop(self) -> None:
        n = 0
        while True:
            await asyncio.sleep(0.2)
            n += 1
            st = self.vision.state
            self.body.hud("attention", score=round(st.score, 2), engaged=st.engaged)
            jpeg = self.vision.jpeg(320, quality=60, annotate=True)
            if jpeg:
                self.body.hud("camera", jpeg=base64.b64encode(jpeg).decode())
            if n % 10 == 0:
                res = await asyncio.to_thread(self.rec.sample_resources)
                self._resources = res
                if n % 25 == 0:
                    self.body.hud("metrics", text=self._metrics_text())

    def _metrics_text(self, last_ms: float | None = None) -> str:
        r = getattr(self, "_resources", {})
        parts = [f"camera {self.vision.fps:.0f} fps · face {self.vision.detect_ms:.1f} ms"]
        if r:
            parts.append(f"brain {r.get('brain_cpu', 0):.0f}% CPU {r.get('brain_rss_mb', 0):.0f} MB"
                         + (f" · body {r.get('body_cpu', 0):.0f}% {r.get('body_rss_mb', 0):.0f} MB" if "body_cpu" in r else ""))
        if last_ms is not None:
            parts.append(f"last reply: {last_ms / 1000:.1f} s after you stopped talking")
        return "\n".join(parts)

    # ================================================================ helpers
    def _face_pose(self) -> dict[str, float] | None:
        face = self.vision.state.face
        if face is None:
            return None
        p = self.aimer.face_point(face.cx, face.cy, self.vision.face_depth_m(face))
        q, _, _ = self.aimer.pose_for(p, self.body.q, postures=("neutral", "tall"))
        return q

    def _frame_b64(self) -> str | None:
        jpeg = self.vision.jpeg(self.cfg.image_width, quality=80)
        return base64.b64encode(jpeg).decode() if jpeg else None

    def _remembered(self, name: str) -> dict | None:
        fact = self.memory.find(name)
        if fact is None:
            return None
        return {"name": fact.name, "x": fact.x, "y": fact.y, "held_by_person": fact.held_by_person}


def _find(objects: list[dict], name: str) -> dict | None:
    key = name.lower().strip()
    for o in objects:
        if o["name"].lower().strip() == key:
            return o
    for o in objects:
        n = o["name"].lower()
        if key and (key in n or n in key):
            return o
    return None


def _face_dict(st: AttentionState) -> dict:
    f = st.face
    return {} if f is None else {"yaw": round(f.yaw_ratio, 2), "pitch": round(f.pitch_ratio, 2), "px": round(f.size_px)}


def local_turn(text: str) -> Turn:
    """Offline fallback: keyword reflexes so the body still responds without the cloud."""
    t = text.lower()
    colors = {"red": "#ff3030", "green": "#30ff60", "blue": "#3070ff", "purple": "#a040ff",
              "pink": "#ff60c0", "orange": "#ff8c20", "white": "#ffffff", "yellow": "#ffe040"}
    for name, hexc in colors.items():
        if re.search(rf"\b{name}\b", t):
            return Turn(f"Going {name}!", "happy", actions=[{"skill": "set_light", "target": "", "color": hexc}])
    if re.search(r"\bsing(ing)?\b", t):
        return Turn("Here goes!", "excited", actions=[{"skill": "sing", "target": singing.DEFAULT_LYRICS, "color": ""}])
    if "dance" in t:
        return Turn("Ooh, let's dance!", "excited", actions=[{"skill": "dance", "target": "", "color": ""}])
    if "music" in t or "song" in t:
        return Turn("Here's a little tune.", "happy", actions=[{"skill": "play_music", "target": "chill", "color": ""}])
    if "stop" in t:
        return Turn("Okay, stopping.", "neutral", actions=[{"skill": "stop_music", "target": "", "color": ""}])
    if "sleep" in t or "good night" in t:
        return Turn("Goodnight!", "sleepy", actions=[{"skill": "sleep", "target": "", "color": ""}])
    if "yes" in t or "nod" in t:
        return Turn("Mm-hm!", "happy", actions=[{"skill": "nod", "target": "", "color": ""}])
    return Turn("I can't reach my thinking cloud right now, but I can still dance, sing, or change my light.",
                "sad", actions=[{"skill": "shake_head", "target": "", "color": ""}])
