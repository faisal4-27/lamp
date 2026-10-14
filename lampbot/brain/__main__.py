"""Brain process: senses, character and mind.

    python -m lampbot.brain       (expects the body on 127.0.0.1:8765)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ..config import ROOT, load_config
from .character import Character
from .metrics import Recorder


async def main() -> None:
    cfg = load_config()
    run_dir = Path(cfg.run_dir) if cfg.run_dir else ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "brain.log", encoding="utf-8")],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.info("run directory: %s", run_dir)
    character = Character(cfg, Recorder(run_dir))
    await character.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
