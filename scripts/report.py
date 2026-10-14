"""Summarise a run's events.jsonl into the numbers the technical note needs.

    python scripts/report.py                 # latest run
    python scripts/report.py runs/20261013-212713
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def row(name: str, values: list[float], unit: str = "ms") -> str:
    if not values:
        return f"  {name:<22} (no samples)"
    return (f"  {name:<22} n={len(values):<4} p50={pct(values, .5):8.0f}{unit}  p95={pct(values, .95):8.0f}{unit}"
            f"  max={max(values):8.0f}{unit}")


def main() -> None:
    if len(sys.argv) > 1:
        run = Path(sys.argv[1])
    else:
        runs = sorted((p for p in (ROOT / "runs").iterdir() if (p / "events.jsonl").exists()),
                      key=lambda p: (p / "events.jsonl").stat().st_mtime)
        run = runs[-1]
    events = [json.loads(line) for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    print(f"run: {run}")

    lat = [e for e in events if e["event"] == "turn_latency"]
    print("\nResponse latency (end of user speech -> first lamp audio):")
    print(row("total", [e["total_ms"] for e in lat]))
    print(row("speech-to-text", [e["stt_ms"] for e in lat if e["stt_ms"] > 0]))
    print(row("mind (cloud)", [e["mind_ms"] for e in lat]))
    print(row("text-to-speech", [e["tts_ms"] for e in lat]))

    eng = [e for e in events if e["event"] in ("engaged", "disengaged")]
    spans, start = [], None
    for e in eng:
        if e["event"] == "engaged":
            start = e["t"]
        elif start is not None:
            spans.append(e["t"] - start)
            start = None
    print("\nEngagement:")
    print(f"  engagements            {sum(e['event'] == 'engaged' for e in eng)}")
    if spans:
        print(f"  mean duration          {statistics.mean(spans):.1f}s")
        print(f"  short (<3 s) spans     {sum(s < 3 for s in spans)}   (candidate false engagements)")
    labels = [e for e in events if e["event"] == "label"]
    if labels:
        tp = sum(e["truth"] and e["predicted"] for e in labels)
        fp = sum(not e["truth"] and e["predicted"] for e in labels)
        fn = sum(e["truth"] and not e["predicted"] for e in labels)
        tn = sum(not e["truth"] and not e["predicted"] for e in labels)
        print(f"  labelled trials        TP={tp} FP={fp} FN={fn} TN={tn}  "
              f"precision={tp / max(tp + fp, 1):.2f} recall={tp / max(tp + fn, 1):.2f}")
        sw = [e["switch_s"] * 1000 for e in labels if e.get("switch_s") is not None]
        print(row("time to (dis)engage", sw))

    wl = [e for e in events if e["event"] == "wave_label"]
    if wl:
        hit = sum(e["truth"] and e["predicted"] for e in wl)
        false = sum(not e["truth"] and e["predicted"] for e in wl)
        print("\nWave detection:")
        print(f"  waves detected         {hit}/{sum(e['truth'] for e in wl)}   "
              f"false waves {false}/{sum(not e['truth'] for e in wl)}")
        print(row("time to detect", [e["switch_s"] * 1000 for e in wl if e["truth"] and e.get("switch_s") is not None]))
    counts = {k: sum(e["event"] == k for e in events) for k in ("wave", "hum", "sing")}
    if any(counts.values()):
        print(f"\nCharacter: waves answered {counts['wave']}  hums {counts['hum']}  songs {counts['sing']}")

    ver = [e for e in events if e["event"] == "verify"]
    if ver:
        print("\nGoal verification:")
        print(f"  checks                 {len(ver)}  corrections {sum(e['moved'] > 0.08 for e in ver)}")
        print(row("verify call", [e["ms"] for e in ver]))

    res = [e for e in events if e["event"] == "resources"]
    if res:
        print("\nResources (percent of all cores, resident memory):")
        for key, unit in (("brain_cpu", "%"), ("brain_rss_mb", "MB"), ("body_cpu", "%"), ("body_rss_mb", "MB")):
            vals = [e[key] for e in res if key in e]
            if vals:
                print(f"  {key:<22} mean={statistics.mean(vals):7.1f}{unit}  p95={pct(vals, .95):7.1f}{unit}  "
                      f"max={max(vals):7.1f}{unit}")


if __name__ == "__main__":
    main()
