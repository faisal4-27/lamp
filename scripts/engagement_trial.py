"""Measure engagement reliability with a guided protocol (camera only, no brain needed).

    python scripts/engagement_trial.py --trials 20
    python scripts/engagement_trial.py --trials 20 --waves 10   # also wave / non-wave trials

Each trial beeps and shows an instruction (in the terminal and on a camera preview
window): LOOK at the laptop camera, LOOK AWAY (turn your head / look at your phone),
LEAVE the frame, WAVE, or DON'T wave. The preview never shows what the detector
thinks, so it cannot steer the participant. After a settle period
the detector's engaged flag is recorded against the instruction, together with
the time it took to change. Results go to runs/engagement-<time>/events.jsonl and
are summarised by scripts/report.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lampbot.config import ROOT, load_config  # noqa: E402
from lampbot.senses.vision import Vision  # noqa: E402

PROMPTS = {
    "look": ("LOOK at the camera", True),
    "away": ("LOOK AWAY (turn head or look down at your phone)", False),
    "leave": ("LEAVE the frame", False),
    "wave": ("WAVE at the camera", True),
    "busy": ("DON'T wave: talk with your hands, scratch your head, lean around", False),
}


SHORT = {"look": "LOOK at the camera", "away": "LOOK AWAY", "leave": "LEAVE the frame",
         "wave": "WAVE", "busy": "DON'T wave (move your hands)"}


def beep() -> None:
    """A short tone through the speakers (terminal bells are often muted, e.g. in VS Code)."""
    try:
        import sounddevice as sd

        t = np.arange(int(0.15 * 22050)) / 22050
        sd.play((0.3 * np.sin(2 * np.pi * 880 * t) * np.minimum(1, (0.15 - t) / 0.02)).astype(np.float32), 22050)
    except Exception:
        print("\a", end="")


def show(vision: Vision, lines: list[str]) -> None:
    frame, _ = vision.latest()
    if frame is None:
        frame = np.zeros((480, 640, 3), np.uint8)
    frame = cv2.flip(frame, 1)  # mirror view, like a selfie camera
    for k, text in enumerate(lines):
        y = 50 + 50 * k
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Lamp engagement test", frame)
    cv2.waitKey(30)


def wait(vision: Vision, seconds: float, lines: list[str], preview: bool) -> None:
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        if preview:
            show(vision, lines + [f"{left:.0f}s"])
        else:
            time.sleep(min(left, 0.1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--settle", type=float, default=4.0, help="seconds to wait before scoring each trial")
    ap.add_argument("--waves", type=int, default=0, help="also run this many wave / non-wave trials")
    ap.add_argument("--no-preview", action="store_true", help="don't open the camera preview window")
    args = ap.parse_args()

    cfg = load_config()
    run = ROOT / "runs" / ("engagement-" + time.strftime("%Y%m%d-%H%M%S"))
    run.mkdir(parents=True, exist_ok=True)
    out = (run / "events.jsonl").open("w", encoding="utf-8")
    changes: list[tuple[float, str]] = []
    vision = Vision(cfg, lambda kind, st: changes.append((time.monotonic(), kind)))
    vision.start()
    preview = not args.no_preview
    print("warming up the camera…")
    wait(vision, 3, ["Get ready..."], preview)

    plan = ["look", "away"] * (args.trials // 2) + ["leave"] * max(1, args.trials // 6)
    plan += ["wave", "busy"] * (args.waves // 2)
    random.shuffle(plan)
    for i, key in enumerate(plan, 1):
        text, truth = PROMPTS[key]
        print(f"\n[{i}/{len(plan)}] {text}")
        beep()
        t0 = time.monotonic()
        changes.clear()
        wait(vision, args.settle, [f"{i}/{len(plan)}  {SHORT[key]}"], preview)
        waves = [t - t0 for t, k in changes if k == "wave"]
        predicted = bool(waves) if key in ("wave", "busy") else vision.state.engaged
        flips = waves if key in ("wave", "busy") else [t - t0 for t, k in changes if k in ("engaged", "disengaged")]
        row = {"t": time.time(), "event": "wave_label" if key in ("wave", "busy") else "label", "condition": key, "truth": truth, "predicted": predicted,
               "switch_s": round(flips[0], 2) if flips else None, "flips": len(flips),
               "last_switch_s": round(flips[-1], 2) if flips else None, "settle_s": args.settle}
        out.write(json.dumps(row) + "\n")
        out.flush()
        word = ("wave", "no wave") if key in ("wave", "busy") else ("engaged", "not engaged")
        print(f"   expected {word[0] if truth else word[1]} -> got {word[0] if predicted else word[1]}"
              + (f" (switched after {flips[0]:.2f}s)" if flips else ""))
    vision.stop()
    out.close()
    if preview:
        cv2.destroyAllWindows()
    print(f"\nsaved {run / 'events.jsonl'}; summarise with: python scripts/report.py {run}")


if __name__ == "__main__":
    main()
