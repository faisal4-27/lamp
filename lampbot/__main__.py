"""Launcher: start the body (simulator + viewer) and the brain as two processes.

    python -m lampbot                 # both, opens the viewer in a browser
    python -m lampbot --no-browser
    python -m lampbot --body-only     # just the simulated lamp (e.g. to jog it from the viewer)

Two processes mirror a real robot: the body would be firmware on the lamp, the
brain a host computer. If the brain crashes the body keeps running and settles
safely; the launcher restarts the brain.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser

from .config import load_config


def wait_port(host: str, port: int, timeout: float = 15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--body-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    body = subprocess.Popen([sys.executable, "-m", "lampbot.body"], env=env)
    if not wait_port(cfg.body_host, cfg.body_port):
        print("body did not start", file=sys.stderr)
        body.terminate()
        return 1
    url = f"http://{cfg.body_host}:{cfg.body_port}"
    print(f"viewer: {url}")
    if not args.no_browser:
        webbrowser.open(url)

    brain = None
    try:
        while True:
            if not args.body_only and (brain is None or brain.poll() is not None):
                if brain is not None:
                    print(f"brain exited with {brain.returncode}; restarting in 2 s", file=sys.stderr)
                    time.sleep(2)
                brain = subprocess.Popen([sys.executable, "-m", "lampbot.brain"],
                                         env=dict(env, LAMP_BODY_PID=str(body.pid)))
            if body.poll() is not None:
                print("body exited", file=sys.stderr)
                return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (brain, body):
            if p and p.poll() is None:
                p.terminate()
        for p in (brain, body):
            if p:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
