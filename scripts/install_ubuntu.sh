#!/usr/bin/env bash
# One-shot setup for a clean Ubuntu 24.04 LTS laptop (4 cores, 8 GB, no GPU).
#   ./scripts/install_ubuntu.sh            # install deps, venv, models
#   ./scripts/install_ubuntu.sh --service  # ...and install systemd user services
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> system packages"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3 python3-venv python3-dev \
  libportaudio2 \
  libgl1 libglib2.0-0 \
  espeak-ng \
  v4l-utils pipewire-pulse

echo "==> camera / audio permissions"
sudo usermod -aG video,audio "$USER" || true

echo "==> python environment"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "==> models and web assets (one-time download, ~250 MB)"
.venv/bin/python scripts/fetch_assets.py

[ -f .env ] || cp .env.example .env

if [[ "${1:-}" == "--service" ]]; then
  echo "==> systemd user services"
  mkdir -p ~/.config/systemd/user
  for unit in lamp-body lamp-brain; do
    sed "s#@ROOT@#$(pwd)#g" "deploy/systemd/$unit.service" > ~/.config/systemd/user/$unit.service
  done
  systemctl --user daemon-reload
  systemctl --user enable --now lamp-body.service lamp-brain.service
  # Keep the services running without an open login session (audio still needs the user's PipeWire).
  sudo loginctl enable-linger "$USER"
  echo "viewer: http://127.0.0.1:8765   logs: journalctl --user -u lamp-brain -f"
fi
echo "done. Put your ANTHROPIC_API_KEY in .env, then run: .venv/bin/python -m lampbot"
