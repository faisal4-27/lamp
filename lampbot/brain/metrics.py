"""Event log + resource sampling. Every run writes runs/<timestamp>/events.jsonl;
scripts/report.py turns it into the numbers for the technical note."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import psutil


class Recorder:
    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "events.jsonl"
        self._f = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.body_proc: psutil.Process | None = None
        self._body_tree: list[psutil.Process] = []
        pid = os.environ.get("LAMP_BODY_PID")
        if pid:
            try:
                self.body_proc = psutil.Process(int(pid))
                self.body_proc.cpu_percent(None)
            except psutil.Error:
                self.body_proc = None

    def mark(self, event: str, **data) -> None:
        row = {"t": round(time.time(), 3), "event": event, **data}
        with self._lock:
            self._f.write(json.dumps(row) + "\n")
            self._f.flush()

    def sample_resources(self) -> dict:
        out = {"brain_cpu": self.proc.cpu_percent(None) / psutil.cpu_count(),
               "brain_rss_mb": self.proc.memory_info().rss / 2**20}
        if self.body_proc:
            try:
                # Include children: on Windows a venv python.exe is a stub that spawns the real interpreter.
                if not self._body_tree:
                    self._body_tree = [self.body_proc] + self.body_proc.children(recursive=True)
                    for p in self._body_tree:
                        p.cpu_percent(None)
                procs = self._body_tree
                out["body_cpu"] = sum(p.cpu_percent(None) for p in procs) / psutil.cpu_count()
                out["body_rss_mb"] = sum(p.memory_info().rss for p in procs) / 2**20
            except psutil.Error:
                self.body_proc = None
        out = {k: round(v, 1) for k, v in out.items()}
        self.mark("resources", **out)
        return out
