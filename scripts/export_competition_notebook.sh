#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/rendered

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

marimo check notebooks/neural_computers_competition.py
marimo export html notebooks/neural_computers_competition.py \
  --no-include-code \
  -o outputs/rendered/neural_computers_competition.html \
  -f

echo "Exported: outputs/rendered/neural_computers_competition.html"
