"""Brain <-> body protocol (v1).

Transport: JSON text frames over one WebSocket, ws://<body>/ws?role=brain.
The body is authoritative for joint limits, velocity limits and safety; the brain
only ever *requests* motion. That is the same boundary a real lamp would have
between a host computer and its motor-controller firmware.

Brain -> body
  hello        {"type":"hello","role":"brain","protocol":1}
  motion       {"type":"motion","id":str,"keyframes":[{"joints":{name:rad},"t":sec}],
                "interrupt":bool}
               Timed keyframes, interpolated min-jerk. The body stretches any segment
               that would exceed 80% of a joint's velocity limit and clamps targets
               to the URDF soft limits. Joints not listed hold their position.
  track        {"type":"track","joints":{name:rad},"speed":0..1}
               Continuous set-point (face tracking, idle breathing). Ignored while a
               motion is running. Rate- and acceleration-limited by the body.
  light        {"type":"light","rgb":[r,g,b],"brightness":0..1,"fade":sec,"pulse_hz":float}
  stop         {"type":"stop"}  hold current pose, cancel motions.
  hud          {"type":"hud", ...}  opaque telemetry forwarded to viewers.

Body -> brain
  welcome      {"type":"welcome","protocol":1,"joints":{name:{lower,upper,vmax}}}
  state        {"type":"state","t":sec,"q":{name:rad},"qd":{name:rad/s},
                "light":{rgb,brightness},"motion":id|null,"load":{name:0..1}}  (30 Hz)
  operator     {"type":"operator","text":str}  typed input from a viewer (debug / demo aid)
  motion_done  {"type":"motion_done","id":str,"status":"completed"|"preempted",
                "clamped":[joint names], "stretched":bool, "duration":sec}
"""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 1

JOINTS = (
    "base_yaw_joint",
    "shoulder_pitch_joint",
    "elbow_pitch_joint",
    "neck_yaw_joint",
    "head_pitch_joint",
)

# Short aliases used throughout the brain code.
ALIAS = {
    "base": "base_yaw_joint",
    "shoulder": "shoulder_pitch_joint",
    "elbow": "elbow_pitch_joint",
    "neck": "neck_yaw_joint",
    "head": "head_pitch_joint",
}


def expand(joints: dict[str, float]) -> dict[str, float]:
    return {ALIAS.get(k, k): float(v) for k, v in joints.items()}


def encode(msg: dict[str, Any]) -> str:
    return json.dumps(msg, separators=(",", ":"))


def decode(raw: str | bytes) -> dict[str, Any]:
    msg = json.loads(raw)
    if not isinstance(msg, dict) or "type" not in msg:
        raise ValueError("message must be an object with a 'type'")
    return msg
