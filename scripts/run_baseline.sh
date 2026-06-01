#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
VENV_DIR="${VENV_DIR:-venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"

if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv "$VENV_DIR"
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import dotenv
import llama_cpp
PY
then
  "$PIP_BIN" install -r requirements.txt
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
fi

export BASELINE_PROVIDER="${BASELINE_PROVIDER:-local}"

echo "Starting Vinpearl tool chatbot at http://${HOST}:${PORT}"
exec "$PYTHON_BIN" src/app/web_app.py --host "$HOST" --port "$PORT"
