"""Dry-run the goal-directed loop without the cloud: a scripted mind claims a mug is
on the desk, the lamp shines on it, then 'looks again' and finds it moved, and corrects.

    python -m lampbot.body &      # or: python -m lampbot --body-only
    python scripts/dry_run.py

Exercises: plan -> skill executor -> aim (image -> 3D -> IK) -> body motion ->
verification -> correction, plus the voice / light / SFX cues along the way.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lampbot.brain.character import Character  # noqa: E402
from lampbot.brain.metrics import Recorder  # noqa: E402
from lampbot.config import ROOT, load_config  # noqa: E402
from lampbot.mind.llm import Turn  # noqa: E402
from lampbot.mind.memory import SceneMemory  # noqa: E402

MUG = {"name": "coffee mug", "color": "blue", "description": "white handle", "x": 0.30, "y": 0.78,
       "held_by_person": False}


class ScriptedMind:
    def __init__(self):
        self.checks = 0

    async def turn(self, utterance, frame, memory, dialog) -> Turn:
        await asyncio.sleep(0.8)
        return Turn(reply="Ooh, the blue mug? Coming right up.", mood="excited", objects=[MUG],
                    actions=[{"skill": "shine_on", "target": "coffee mug", "color": ""}], verify=True)

    async def observe(self, frame, memory):
        return [MUG], "", 10.0

    async def verify(self, goal, target, aimed_xy, frame):
        self.checks += 1
        await asyncio.sleep(0.5)
        if self.checks == 1:  # the person slid the mug to the right
            return {"visible": True, "x": 0.62, "y": 0.74, "satisfied": False, "report": ""}, 500.0
        return {"visible": True, "x": 0.63, "y": 0.74, "satisfied": True,
                "report": "There you go, your mug is in the spotlight!"}, 500.0


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config()
    run_dir = ROOT / "runs" / ("dryrun-" + time.strftime("%Y%m%d-%H%M%S"))
    ch = Character(cfg, Recorder(run_dir))
    ch.mind = ScriptedMind()
    ch.memory = SceneMemory(run_dir / "memory.json")  # keep the scripted mug out of the real memory
    asyncio.create_task(ch.run())
    await asyncio.sleep(6)
    ch.engaged = True
    ch._set_state("listening")
    ch._post("typed", {"text": "Shine your light on my mug"})
    await asyncio.sleep(14)
    print((run_dir / "events.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
