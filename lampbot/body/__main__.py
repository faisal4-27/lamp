"""Body process: simulated lamp + 3D viewer.

    python -m lampbot.body            # http://127.0.0.1:8765

One brain connects on /ws?role=brain; any number of browsers connect on
/ws?role=viewer and receive state + forwarded HUD telemetry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from aiohttp import WSMsgType, web

from ..config import ROOT, URDF_PATH, load_config
from ..kinematics import LampModel
from ..protocol import PROTOCOL_VERSION, decode, encode, expand
from .actuators import Body

log = logging.getLogger("body")
WEB = Path(__file__).parent / "web"
SIM_HZ = 200
STATE_HZ = 30
WATCHDOG_S = 3.0


class BodyServer:
    def __init__(self) -> None:
        self.model = LampModel(URDF_PATH)
        self.body = Body(self.model)
        self.brain: web.WebSocketResponse | None = None
        self.viewers: set[web.WebSocketResponse] = set()
        self.last_brain_msg = 0.0
        self.resting = True
        self.last_hud: dict[str, dict] = {}

    # ------------------------------------------------------------ messages
    def handle(self, msg: dict) -> None:
        kind = msg["type"]
        if kind == "motion":
            frames = [{"joints": expand(k.get("joints", {})), "t": k.get("t", 0.5)}
                      for k in msg.get("keyframes", [])]
            if frames:
                self.body.command_motion(str(msg.get("id", "motion")), frames, msg.get("interrupt", True))
        elif kind == "track":
            self.body.command_track(expand(msg.get("joints", {})), float(msg.get("speed", 0.5)))
        elif kind == "light":
            self.body.light.set(msg.get("rgb", [1, 1, 1]), float(msg.get("brightness", 0.5)),
                                float(msg.get("fade", 0.3)), float(msg.get("pulse_hz", 0.0)))
        elif kind == "stop":
            self.body.command_stop()
        elif kind == "hud":
            self.last_hud[msg.get("topic", "misc")] = msg
            self.broadcast_viewers(msg)
        elif kind == "operator":
            # Typed input from a viewer, forwarded to the brain as if it were heard.
            if self.brain and not self.brain.closed:
                asyncio.ensure_future(self.brain.send_str(encode(msg)))
        elif kind == "hello":
            pass
        else:
            log.warning("unknown message type %s", kind)

    def broadcast_viewers(self, msg: dict) -> None:
        raw = encode(msg)
        for ws in list(self.viewers):
            if not ws.closed:
                asyncio.ensure_future(ws.send_str(raw))

    # ------------------------------------------------------------ loops
    async def sim_loop(self) -> None:
        dt = 1.0 / SIM_HZ
        per_state = SIM_HZ // STATE_HZ
        tick = 0
        next_t = time.perf_counter()
        while True:
            self.body.step(dt)
            tick += 1
            if self.brain is None or time.monotonic() - self.last_brain_msg > WATCHDOG_S:
                if not self.resting:
                    log.info("brain silent: settling to rest pose")
                    self.body.rest()
                    self.resting = True
            if tick % per_state == 0:
                state = self.body.state()
                raw = encode(state)
                targets = list(self.viewers) + ([self.brain] if self.brain else [])
                for ws in targets:
                    if not ws.closed:
                        asyncio.ensure_future(ws.send_str(raw))
            for ev in self.body.events:
                if self.brain and not self.brain.closed:
                    asyncio.ensure_future(self.brain.send_str(encode(ev)))
                self.broadcast_viewers(ev)
            self.body.events.clear()
            next_t += dt
            await asyncio.sleep(max(0.0, next_t - time.perf_counter()))
            if time.perf_counter() - next_t > 0.25:  # fell behind (e.g. laptop suspended)
                next_t = time.perf_counter()

    # ------------------------------------------------------------ http
    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        role = request.query.get("role", "viewer")
        ws = web.WebSocketResponse(heartbeat=5.0, max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        await ws.send_str(encode({"type": "welcome", "protocol": PROTOCOL_VERSION,
                                  "joints": self.model.limits()}))
        if role == "brain":
            if self.brain and not self.brain.closed:
                await self.brain.close()
            self.brain = ws
            self.last_brain_msg = time.monotonic()
            self.resting = False
            log.info("brain connected")
        else:
            self.viewers.add(ws)
            for hud in self.last_hud.values():
                await ws.send_str(encode(hud))
        try:
            async for raw in ws:
                if raw.type != WSMsgType.TEXT:
                    continue
                try:
                    msg = decode(raw.data)
                except (ValueError, json.JSONDecodeError) as exc:
                    log.warning("bad message from %s: %s", role, exc)
                    continue
                if role == "brain":
                    self.last_brain_msg = time.monotonic()
                    self.resting = False
                self.handle(msg)
        finally:
            if ws is self.brain:
                self.brain = None
                log.info("brain disconnected")
            self.viewers.discard(ws)
        return ws

    async def index(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB / "index.html")


async def main() -> None:
    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    server = BodyServer()
    app = web.Application()
    app.router.add_get("/", server.index)
    app.router.add_get("/ws", server.ws_handler)
    app.router.add_static("/web", WEB)
    app.router.add_static("/robot", ROOT / "robot")
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, cfg.body_host, cfg.body_port).start()
    log.info("body up: viewer at http://%s:%d", cfg.body_host, cfg.body_port)
    await server.sim_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
