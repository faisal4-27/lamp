"""URDF parsing, forward kinematics, static gravity load and a look-at solver.

Shared by the body (limits, load estimates) and the brain (aiming the head).
Pure numpy; the chain is tiny so everything is evaluated directly.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GRAVITY = np.array([0.0, 0.0, -9.81])


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def transform(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(*rpy)
    T[:3, 3] = xyz
    return T


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, C = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    T = np.eye(4)
    T[:3, :3] = [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]
    return T


def _vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.array(default, float)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray
    lower: float = 0.0
    upper: float = 0.0
    soft_lower: float = 0.0
    soft_upper: float = 0.0
    vmax: float = 0.0
    effort: float = 0.0


@dataclass
class Link:
    name: str
    mass: float = 0.0
    com: np.ndarray = field(default_factory=lambda: np.zeros(3))


class LampModel:
    def __init__(self, urdf_path: Path):
        root = ET.parse(urdf_path).getroot()
        self.links: dict[str, Link] = {}
        for el in root.findall("link"):
            link = Link(el.get("name"))
            inert = el.find("inertial")
            if inert is not None:
                link.mass = float(inert.find("mass").get("value"))
                o = inert.find("origin")
                link.com = _vec(o.get("xyz") if o is not None else None)
            self.links[link.name] = link

        self.joints: dict[str, Joint] = {}
        for el in root.findall("joint"):
            o = el.find("origin")
            j = Joint(
                name=el.get("name"),
                type=el.get("type"),
                parent=el.find("parent").get("link"),
                child=el.find("child").get("link"),
                origin=transform(_vec(o.get("xyz") if o is not None else None),
                                 _vec(o.get("rpy") if o is not None else None)),
                axis=_vec(el.find("axis").get("xyz") if el.find("axis") is not None else None, (1, 0, 0)),
            )
            lim, safe = el.find("limit"), el.find("safety_controller")
            if lim is not None:
                j.lower, j.upper = float(lim.get("lower")), float(lim.get("upper"))
                j.vmax, j.effort = float(lim.get("velocity")), float(lim.get("effort"))
                j.soft_lower, j.soft_upper = j.lower, j.upper
            if safe is not None:
                j.soft_lower = float(safe.get("soft_lower_limit"))
                j.soft_upper = float(safe.get("soft_upper_limit"))
            self.joints[j.name] = j

        children = {j.child for j in self.joints.values()}
        self.root = next(n for n in self.links if n not in children)
        # Topological order of joints from the root.
        self.order: list[Joint] = []
        frontier = [self.root]
        while frontier:
            parent = frontier.pop(0)
            for j in self.joints.values():
                if j.parent == parent:
                    self.order.append(j)
                    frontier.append(j.child)
        self.actuated = [j.name for j in self.order if j.type == "revolute"]

    # ---------------------------------------------------------------- limits
    def clamp(self, q: dict[str, float], soft: bool = True) -> tuple[dict[str, float], list[str]]:
        out, clamped = {}, []
        for name, v in q.items():
            j = self.joints[name]
            lo, hi = (j.soft_lower, j.soft_upper) if soft else (j.lower, j.upper)
            c = min(max(v, lo), hi)
            if abs(c - v) > 1e-6:
                clamped.append(name)
            out[name] = c
        return out, clamped

    def limits(self) -> dict[str, dict[str, float]]:
        return {n: {"lower": self.joints[n].soft_lower, "upper": self.joints[n].soft_upper,
                    "vmax": self.joints[n].vmax, "effort": self.joints[n].effort}
                for n in self.actuated}

    # ---------------------------------------------------------------- kinematics
    def fk(self, q: dict[str, float]) -> dict[str, np.ndarray]:
        """World (base_link) transform of every link."""
        T = {self.root: np.eye(4)}
        for j in self.order:
            M = T[j.parent] @ j.origin
            if j.type == "revolute":
                M = M @ axis_angle(j.axis, q.get(j.name, 0.0))
            T[j.child] = M
        return T

    def joint_frames(self, q: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World position and axis of each revolute joint."""
        T = self.fk(q)
        out = {}
        for j in self.order:
            if j.type != "revolute":
                continue
            M = T[j.parent] @ j.origin
            out[j.name] = (M[:3, 3], M[:3, :3] @ j.axis)
        return out

    def head_ray(self, q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """Light emitter position and pointing direction (lamp_head +x)."""
        T = self.fk(q)["light_emitter_link"]
        return T[:3, 3].copy(), T[:3, 0].copy()

    def gravity_load(self, q: dict[str, float]) -> dict[str, float]:
        """Static gravity torque on each joint as a fraction of its effort limit."""
        T = self.fk(q)
        frames = self.joint_frames(q)
        subtree = self._subtrees()
        load = {}
        for name, (p, axis) in frames.items():
            tau = 0.0
            for link_name in subtree[self.joints[name].child]:
                link = self.links[link_name]
                if link.mass <= 0:
                    continue
                com = T[link_name][:3, :3] @ link.com + T[link_name][:3, 3]
                tau += float(axis @ np.cross(com - p, link.mass * GRAVITY))
            load[name] = abs(tau) / self.joints[name].effort
        return load

    def _subtrees(self) -> dict[str, list[str]]:
        if hasattr(self, "_sub"):
            return self._sub
        kids: dict[str, list[str]] = {}
        for j in self.order:
            kids.setdefault(j.parent, []).append(j.child)

        def collect(link: str) -> list[str]:
            out = [link]
            for c in kids.get(link, []):
                out += collect(c)
            return out

        self._sub = {name: collect(name) for name in self.links}
        return self._sub

    # ---------------------------------------------------------------- aiming
    def look_at(self, target: np.ndarray, q0: dict[str, float],
                free=("base_yaw_joint", "neck_yaw_joint", "head_pitch_joint"),
                weights=None, iters: int = 30) -> tuple[dict[str, float], float]:
        """Damped least-squares aim of the head ray at `target` (base frame).

        Joints not in `free` keep their value from q0 (the chosen posture).
        `weights` regularise each free joint towards q0 so small corrections use the
        light neck/head joints and large turns recruit the heavier base.
        Returns (pose, residual angle in radians).
        """
        weights = weights or {"base_yaw_joint": 0.04, "neck_yaw_joint": 0.12, "head_pitch_joint": 0.05}
        q0 = {n: q0.get(n, 0.0) for n in self.actuated}
        q = dict(q0)
        names = list(free)

        def residual(qv: dict[str, float]) -> np.ndarray:
            p, d = self.head_ray(qv)
            want = target - p
            want /= np.linalg.norm(want) + 1e-9
            reg = [math.sqrt(weights.get(n, 0.0)) * (qv[n] - q0.get(n, 0.0)) for n in names]
            return np.concatenate([d - want, reg])

        for _ in range(iters):
            r = residual(q)
            J = np.zeros((len(r), len(names)))
            for i, n in enumerate(names):
                dq = dict(q)
                dq[n] += 1e-4
                J[:, i] = (residual(dq) - r) / 1e-4
            step = np.linalg.solve(J.T @ J + 1e-3 * np.eye(len(names)), -J.T @ r)
            for i, n in enumerate(names):
                q[n] += float(np.clip(step[i], -0.3, 0.3))
            q, _ = self.clamp(q)
            if np.linalg.norm(step) < 1e-4:
                break
        p, d = self.head_ray(q)
        want = (target - p) / (np.linalg.norm(target - p) + 1e-9)
        return q, float(math.acos(np.clip(d @ want, -1, 1)))
