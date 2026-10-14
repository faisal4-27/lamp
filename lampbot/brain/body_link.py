"""Brain-side client of the body protocol (see lampbot/protocol.py)."""

from __future__ import annotations

import asyncio
import itertools
import logging

import aiohttp

from ..protocol import PROTOCOL_VERSION, decode, encode, expand

log = logging.getLogger("body-link")


class BodyLink:
    def __init__(self, url: str):
        self.url = url
        self.q: dict[str, float] = {}
        self.load: dict[str, float] = {}
        self.limits: dict[str, dict] = {}
        self.connected = asyncio.Event()
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._ids = itertools.count(1)
        self.on_operator = None  # callback(text) for typed input from the viewer

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.ws_connect(self.url, heartbeat=5.0) as ws:
                        self._ws = ws
                        await ws.send_str(encode({"type": "hello", "role": "brain", "protocol": PROTOCOL_VERSION}))
                        log.info("connected to body at %s", self.url)
                        async for raw in ws:
                            if raw.type == aiohttp.WSMsgType.TEXT:
                                self._on_message(decode(raw.data))
                except (aiohttp.ClientError, OSError) as exc:
                    log.warning("body unavailable (%s); retrying", exc)
                self._ws = None
                self.connected.clear()
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_result({"status": "disconnected"})
                self._pending.clear()
                await asyncio.sleep(1.0)

    def _on_message(self, msg: dict) -> None:
        kind = msg["type"]
        if kind == "state":
            self.q, self.load = msg["q"], msg.get("load", {})
            self.connected.set()
        elif kind == "welcome":
            if msg.get("protocol") != PROTOCOL_VERSION:
                log.error("protocol mismatch: body %s, brain %s", msg.get("protocol"), PROTOCOL_VERSION)
            self.limits = msg.get("joints", {})
        elif kind == "operator":
            if self.on_operator and msg.get("text"):
                self.on_operator(str(msg["text"]))
        elif kind == "motion_done":
            fut = self._pending.pop(msg["id"], None)
            if fut and not fut.done():
                fut.set_result(msg)

    def send(self, msg: dict) -> None:
        if self._ws is not None and not self._ws.closed:
            asyncio.ensure_future(self._ws.send_str(encode(msg)))

    # ------------------------------------------------------------ helpers
    def motion(self, keyframes: list[dict], name: str = "m", interrupt: bool = True) -> asyncio.Future:
        mid = f"{name}-{next(self._ids)}"
        fut = asyncio.get_running_loop().create_future()
        self._pending[mid] = fut
        frames = [{"joints": expand(k["joints"]), "t": k.get("t", 0.5)} for k in keyframes]
        self.send({"type": "motion", "id": mid, "keyframes": frames, "interrupt": interrupt})
        if self._ws is None:
            fut.set_result({"status": "disconnected"})
        return fut

    def track(self, joints: dict[str, float], speed: float = 0.5) -> None:
        self.send({"type": "track", "joints": expand(joints), "speed": speed})

    def light(self, rgb, brightness: float, fade: float = 0.3, pulse_hz: float = 0.0) -> None:
        self.send({"type": "light", "rgb": list(rgb), "brightness": brightness, "fade": fade, "pulse_hz": pulse_hz})

    def hud(self, topic: str, **data) -> None:
        self.send({"type": "hud", "topic": topic, **data})
