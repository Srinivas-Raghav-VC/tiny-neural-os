#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[1/4] marimo check --fix"
marimo check --fix notebooks/neural_computers_competition.py
marimo check notebooks/neural_computers_competition.py

echo "[2/4] regenerate competition session"
python3 scripts/create_sessions.py notebooks/neural_computers_competition.py

echo "[3/4] export html"
bash scripts/export_competition_notebook.sh

echo "[4/4] refresh README screenshots"
bash scripts/capture_readme_screenshots.sh

echo "Done. Public artifacts refreshed."
