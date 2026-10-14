"""Download the models and web assets the lamp needs, so runtime never hits the network
for anything except the (optional) Claude API.

    python scripts/fetch_assets.py            # everything
    python scripts/fetch_assets.py --skip-whisper

Everything lands in ./assets (git-ignored). Safe to re-run; existing files are kept.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
VENDOR = ROOT / "lampbot" / "body" / "web" / "vendor"

THREE = "https://cdn.jsdelivr.net/npm/three@0.170.0"
WEB_FILES = {
    VENDOR / "three.module.js": f"{THREE}/build/three.module.js",
    VENDOR / "STLLoader.js": f"{THREE}/examples/jsm/loaders/STLLoader.js",
    VENDOR / "OrbitControls.js": f"{THREE}/examples/jsm/controls/OrbitControls.js",
}
YUNET = (
    ASSETS / "face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx",
)
PIPER_VOICE = "en_US-amy-medium"


def fetch(dest: Path, url: str) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ok      {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch   {dest.relative_to(ROOT)}")
    req = urllib.request.Request(url, headers={"User-Agent": "lampbot-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--whisper-model", default="base.en")
    ap.add_argument("--voice", default=PIPER_VOICE)
    args = ap.parse_args()

    print("web viewer (three.js):")
    for dest, url in WEB_FILES.items():
        fetch(dest, url)

    print("face detector (YuNet):")
    fetch(*YUNET)

    print(f"voice (piper {args.voice}):")
    voices = ASSETS / "voices"
    lang, name, quality = args.voice.split("-")
    base = (f"https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            f"{lang.split('_')[0]}/{lang}/{name}/{quality}/{args.voice}")
    for ext in (".onnx", ".onnx.json"):
        fetch(voices / f"{args.voice}{ext}", f"{base}{ext}?download=true")

    if not args.skip_whisper:
        print(f"speech-to-text (faster-whisper {args.whisper_model}):")
        from faster_whisper import WhisperModel

        WhisperModel(args.whisper_model, device="cpu", compute_type="int8",
                     download_root=str(ASSETS / "whisper"))
        print("  ok")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
