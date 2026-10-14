"""Simulated lamp body: joint servos and the head light.

This is the stand-in for motor-controller firmware. It owns every physical
constraint, so nothing upstream (including a language model) can command an
unsafe move:
  * targets are clamped to the URDF soft limits;
  * timed motions are stretched so min-jerk peak velocity stays <= 80% of vmax;
  * every servo is velocity- and acceleration-limited regardless of mode;
  * if the brain goes quiet the body settles into a low-load rest pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..kinematics import LampModel

VEL_MARGIN = 0.8
# Heavier proximal joints get gentler acceleration (rad/s^2).
ACCEL = {
    "base_yaw_joint": 3.0,
    "shoulder_pitch_joint": 2.5,
    "elbow_pitch_joint": 3.5,
    "neck_yaw_joint": 6.0,
    "head_pitch_joint": 6.0,
}
REST_POSE = {
    "base_yaw_joint": 0.0,
    "shoulder_pitch_joint": 0.15,
    "elbow_pitch_joint": -0.55,
    "neck_yaw_joint": 0.0,
    "head_pitch_joint": 0.55,
}


def minjerk(s: float) -> tuple[float, float]:
    """Normalised min-jerk position and velocity (d/ds) at phase s in [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4


@dataclass
class Motion:
    id: str
    segments: list[tuple[dict[str, float], dict[str, float], float]]  # (start, end, duration)
    clamped: list[str]
    stretched: bool
    t: float = 0.0

    @property
    def duration(self) -> float:
        return sum(d for _, _, d in self.segments)

    def sample(self) -> tuple[dict[str, float], dict[str, float], bool]:
        t = self.t
        for start, end, dur in self.segments:
            if t <= dur:
                s = t / dur if dur > 0 else 1.0
                p, v = minjerk(s)
                pos = {n: start[n] + (end[n] - start[n]) * p for n in end}
                vel = {n: (end[n] - start[n]) * v / dur if dur > 0 else 0.0 for n in end}
                return pos, vel, False
            t -= dur
        last = self.segments[-1][1]
        return dict(last), {n: 0.0 for n in last}, True


@dataclass
class Light:
    rgb: list[float] = field(default_factory=lambda: [1.0, 0.85, 0.6])
    brightness: float = 0.15
    pulse_hz: float = 0.0
    _from: tuple = ((1.0, 0.85, 0.6), 0.15)
    _to: tuple = ((1.0, 0.85, 0.6), 0.15)
    _fade: float = 0.0
    _t: float = 0.0
    _phase: float = 0.0

    def set(self, rgb, brightness: float, fade: float, pulse_hz: float) -> None:
        self._from = (tuple(self.rgb), self.brightness)
        self._to = (tuple(min(max(c, 0.0), 1.0) for c in rgb), min(max(brightness, 0.0), 1.0))
        self._fade, self._t, self.pulse_hz = max(fade, 0.0), 0.0, max(pulse_hz, 0.0)

    def step(self, dt: float) -> dict:
        self._t += dt
        a = 1.0 if self._fade <= 0 else min(self._t / self._fade, 1.0)
        (r0, b0), (r1, b1) = self._from, self._to
        self.rgb = [x + (y - x) * a for x, y in zip(r0, r1)]
        self.brightness = b0 + (b1 - b0) * a
        self._phase += 2 * math.pi * self.pulse_hz * dt
        level = self.brightness
        if self.pulse_hz > 0:
            level *= 0.55 + 0.45 * (0.5 + 0.5 * math.sin(self._phase))
        return {"rgb": [round(c, 3) for c in self.rgb], "brightness": round(level, 3)}


class Body:
    def __init__(self, model: LampModel):
        self.model = model
        self.names = model.actuated
        self.q = dict(REST_POSE)
        self.qd = {n: 0.0 for n in self.names}
        self.track_target = dict(REST_POSE)
        self.track_speed = 0.5
        self.motion: Motion | None = None
        self.light = Light()
        self.light_out = self.light.step(0.0)
        self.t = 0.0
        self.events: list[dict] = []

    # ------------------------------------------------------------ commands
    def command_motion(self, mid: str, keyframes: list[dict], interrupt: bool = True) -> None:
        if self.motion and not interrupt:
            # Queue by appending segments to the running motion.
            start = dict(self.motion.segments[-1][1])
            base = self.motion
        else:
            if self.motion:
                self.events.append({"type": "motion_done", "id": self.motion.id, "status": "preempted",
                                    "clamped": self.motion.clamped, "stretched": self.motion.stretched,
                                    "duration": round(self.motion.t, 3)})
            start = dict(self.q)
            base = None
        segments, clamped_all, stretched = [], set(), False
        for kf in keyframes:
            target, clamped = self.model.clamp({**start, **kf.get("joints", {})})
            clamped_all.update(clamped)
            dur = max(float(kf.get("t", 0.5)), 0.05)
            # min-jerk peak velocity is 1.875 * distance / duration
            need = max(1.875 * abs(target[n] - start[n]) / (VEL_MARGIN * self.model.joints[n].vmax)
                       for n in self.names)
            if need > dur:
                dur, stretched = need, True
            segments.append((start, target, dur))
            start = target
        if base:
            base.segments += segments
            base.clamped = sorted(set(base.clamped) | clamped_all)
            base.stretched |= stretched
        else:
            self.motion = Motion(mid, segments, sorted(clamped_all), stretched)

    def command_track(self, joints: dict[str, float], speed: float) -> None:
        self.track_target, _ = self.model.clamp({**self.track_target, **joints})
        self.track_speed = min(max(speed, 0.05), 1.0)

    def command_stop(self) -> None:
        if self.motion:
            self.events.append({"type": "motion_done", "id": self.motion.id, "status": "preempted",
                                "clamped": self.motion.clamped, "stretched": self.motion.stretched,
                                "duration": round(self.motion.t, 3)})
        self.motion = None
        self.track_target = dict(self.q)

    def rest(self) -> None:
        self.command_motion("watchdog-rest", [{"joints": REST_POSE, "t": 2.5}])
        self.light.set([1.0, 0.75, 0.45], 0.06, 2.0, 0.0)

    # ------------------------------------------------------------ simulation
    def step(self, dt: float) -> None:
        self.t += dt
        if self.motion:
            self.motion.t += dt
            sp, ff, done = self.motion.sample()
            gain, vscale = 12.0, 1.0
            if done:
                m = self.motion
                self.events.append({"type": "motion_done", "id": m.id, "status": "completed",
                                    "clamped": m.clamped, "stretched": m.stretched,
                                    "duration": round(m.duration, 3)})
                self.track_target = dict(sp)
                self.motion = None
        else:
            sp, ff = self.track_target, {n: 0.0 for n in self.names}
            gain, vscale = 2.0 + 6.0 * self.track_speed, self.track_speed

        for n in self.names:
            j = self.model.joints[n]
            vmax = j.vmax * vscale
            v_cmd = ff.get(n, 0.0) + gain * (sp.get(n, self.q[n]) - self.q[n])
            v_cmd = min(max(v_cmd, -vmax), vmax)
            dv = min(max(v_cmd - self.qd[n], -ACCEL[n] * dt), ACCEL[n] * dt)
            self.qd[n] += dv
            self.q[n] += self.qd[n] * dt
            # Hard stop at the URDF hard limits (the soft limits are enforced on targets).
            if self.q[n] < j.lower or self.q[n] > j.upper:
                self.q[n] = min(max(self.q[n], j.lower), j.upper)
                self.qd[n] = 0.0
        self.light_out = self.light.step(dt)

    def state(self) -> dict:
        return {
            "type": "state",
            "t": round(self.t, 3),
            "q": {n: round(v, 4) for n, v in self.q.items()},
            "qd": {n: round(v, 3) for n, v in self.qd.items()},
            "light": self.light_out,
            "motion": self.motion.id if self.motion else None,
            "load": {n: round(v, 3) for n, v in self.model.gravity_load(self.q).items()},
        }
