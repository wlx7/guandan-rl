#!/usr/bin/env bash
# Linux / macOS installer: creates .venv, installs PyTorch and guandan-rl.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
if command -v nvidia-smi >/dev/null 2>&1; then pip install torch; else pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch; fi
WHL=$(ls guandan_rl-*.whl 2>/dev/null | head -1 || true)
if [ -n "$WHL" ]; then pip install "$WHL"; else pip install "git+https://github.com/wlx7/guandan-rl"; fi
python -m guandan info
echo "Done. Start with: GUANDAN_HOME=$PWD/guandan_data .venv/bin/python -m guandan serve"
