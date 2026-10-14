"""Scene memory: structured facts about objects the lamp has seen.

Facts, not captions: each object keeps a name, colour, a short description, where
it was in view and when. That keeps recall grounded ("the blue mug, left of the
laptop, two minutes ago") and makes the memory inspectable in the viewer and on
disk (data/memory.json). Persisted across restarts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def where(x: float, y: float) -> str:
    h = "left" if x < 0.36 else "right" if x > 0.64 else "centre"
    v = "top" if y < 0.33 else "bottom" if y > 0.67 else "middle"
    return f"{v}-{h} of view"


def ago(seconds: float) -> str:
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    return f"{seconds / 3600:.1f} h ago"


@dataclass
class ObjectFact:
    name: str
    color: str = ""
    description: str = ""
    x: float = 0.5
    y: float = 0.5
    held_by_person: bool = False
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    times_seen: int = 1
    visible_now: bool = True

    def summary(self, now: float) -> str:
        bits = [self.color, self.description]
        held = "held up by the person" if self.held_by_person else where(self.x, self.y)
        return f"{', '.join(b for b in bits if b)} — {held}, {ago(now - self.last_seen)}"


class SceneMemory:
    def __init__(self, path: Path, max_objects: int = 40):
        self.path = path
        self.max_objects = max_objects
        self.objects: dict[str, ObjectFact] = {}
        self.dialog: list[tuple[str, str]] = []  # (speaker, text), most recent last
        self._load()

    def update(self, observed: list[dict], now: float | None = None) -> list[str]:
        """Merge one observation. Returns names of objects that are new."""
        now = now or time.time()
        new = []
        for o in self.objects.values():
            o.visible_now = False
        for item in observed:
            name = (item.get("name") or "").strip()
            if not name or _key(name) in ("person", "man", "woman", "face", "human"):
                continue
            k = _key(name)
            fact = self.objects.get(k)
            if fact is None:
                fact = ObjectFact(name=name, first_seen=now)
                self.objects[k] = fact
                new.append(name)
            else:
                fact.times_seen += 1
            fact.color = item.get("color") or fact.color
            fact.description = item.get("description") or fact.description
            fact.x, fact.y = float(item.get("x", fact.x)), float(item.get("y", fact.y))
            fact.held_by_person = bool(item.get("held_by_person", False))
            fact.last_seen, fact.visible_now = now, True
        if len(self.objects) > self.max_objects:
            for k in sorted(self.objects, key=lambda k: self.objects[k].last_seen)[: len(self.objects) - self.max_objects]:
                del self.objects[k]
        self._save()
        return new

    def find(self, name: str) -> ObjectFact | None:
        k = _key(name)
        if k in self.objects:
            return self.objects[k]
        for key, fact in self.objects.items():
            if k and (k in key or key in k):
                return fact
        return None

    def add_dialog(self, speaker: str, text: str) -> None:
        self.dialog.append((speaker, text))
        self.dialog = self.dialog[-12:]

    def as_prompt(self) -> str:
        now = time.time()
        if not self.objects:
            return "(nothing yet)"
        rows = sorted(self.objects.values(), key=lambda o: -o.last_seen)
        return "\n".join(f"- {o.name}: {o.summary(now)}{' [visible now]' if o.visible_now else ''}" for o in rows)

    def dialog_prompt(self) -> str:
        return "\n".join(f"{who}: {text}" for who, text in self.dialog) or "(no conversation yet)"

    def hud(self) -> list[dict]:
        now = time.time()
        rows = sorted(self.objects.values(), key=lambda o: -o.last_seen)
        return [{"name": o.name, "summary": o.summary(now)} for o in rows[:12]]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(o) for o in self.objects.values()], indent=1), encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            for row in json.loads(self.path.read_text(encoding="utf-8")):
                row["visible_now"] = False
                self.objects[_key(row["name"])] = ObjectFact(**row)
        except (ValueError, TypeError, KeyError):
            pass
