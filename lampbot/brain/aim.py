"""From a point in the laptop camera image to a lamp pose.

Physical set-up assumed (configurable in Config): the lamp stands on the desk next
to the laptop, both facing the person. The lamp's base frame has z up and the lamp
faces -x. The laptop camera sits at `camera_pos` looking along -x too.

The camera is fixed, unlike the URDF's head camera, so we cannot close the loop
by re-centring the target in the image. Instead each image point is lifted to 3D:
  * faces: depth from inter-eye pixel distance (pinhole model, 63 mm IPD);
  * objects: the camera ray is intersected with the desk plane, or an assumed
    depth is used if the ray points above the horizon (object held up).
The head is then aimed at that 3D point with the kinematic look-at solver,
trying a few arm postures and keeping the one that aims best with low load.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import Config
from ..kinematics import LampModel

POSTURES = {
    "neutral": {"shoulder_pitch_joint": 0.30, "elbow_pitch_joint": -0.95},
    "tall": {"shoulder_pitch_joint": 0.55, "elbow_pitch_joint": -1.30},
    "lean_in": {"shoulder_pitch_joint": -0.20, "elbow_pitch_joint": -0.70},
}
DESK_Z = 0.03


class Aimer:
    def __init__(self, cfg: Config, model: LampModel):
        self.cfg = cfg
        self.model = model
        self.cam = np.array(cfg.camera_pos, float)
        self.tan_h = math.tan(math.radians(cfg.camera_hfov_deg) / 2)
        self.tan_v = self.tan_h * cfg.camera_height / cfg.camera_width
        # Camera basis in the lamp base frame (camera looks along -x).
        self.fwd = np.array([-1.0, 0.0, 0.0])
        self.right = np.array([0.0, 1.0, 0.0])
        self.up = np.array([0.0, 0.0, 1.0])

    def ray(self, u: float, v: float) -> np.ndarray:
        """Unnormalised ray with unit forward component for image point (u, v) in 0..1."""
        return self.fwd + self.tan_h * (2 * u - 1) * self.right - self.tan_v * (2 * v - 1) * self.up

    def face_point(self, u: float, v: float, depth: float) -> np.ndarray:
        return self.cam + depth * self.ray(u, v)

    def object_point(self, u: float, v: float, held: bool = False) -> np.ndarray:
        r = self.ray(u, v)
        if not held and r[2] < -0.05:
            s = (DESK_Z - self.cam[2]) / r[2]
            if 0.15 < s < 1.5:
                return self.cam + s * r
        return self.cam + self.cfg.object_assumed_depth_m * r

    def pose_for(self, target: np.ndarray, current: dict[str, float] | None = None,
                 postures: tuple[str, ...] = ("neutral", "tall", "lean_in")) -> tuple[dict[str, float], str, float]:
        best = None
        for name in postures:
            q0 = dict(current or {})
            q0.update(POSTURES[name])
            q, err = self.model.look_at(target, q0, iters=20)
            load = max(self.model.gravity_load(q).values())
            # Prefer accurate aim; break ties towards the least loaded posture.
            cost = err + 0.05 * load + (0.0 if name == "neutral" else 0.02)
            if best is None or cost < best[0]:
                best = (cost, q, name, err)
        _, q, name, err = best
        return q, name, err

    def emitter(self, q: dict[str, float]) -> np.ndarray:
        return self.model.head_ray(q)[0]
