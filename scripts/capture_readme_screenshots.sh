#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CHROMIUM_BIN="$(command -v chromium || command -v chromium-browser || true)"
if [[ -z "$CHROMIUM_BIN" ]]; then
  echo "Could not find chromium/chromium-browser on PATH." >&2
  exit 1
fi

if ! command -v convert >/dev/null 2>&1; then
  echo "ImageMagick 'convert' is required but not installed." >&2
  exit 1
fi

mkdir -p assets outputs/rendered

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

marimo check --fix notebooks/neural_computers_competition.py >/dev/null
marimo check notebooks/neural_computers_competition.py >/dev/null

marimo export html notebooks/neural_computers_competition.py \
  --no-include-code \
  -o outputs/rendered/neural_computers_competition.html \
  -f >/dev/null

TMP_TALL="$(mktemp /tmp/readme_tall.XXXXXX.png)"
TARGET_URL="file://$(pwd)/outputs/rendered/neural_computers_competition.html"

"$CHROMIUM_BIN" \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=7000 \
  --window-size=1440,8200 \
  --screenshot="$TMP_TALL" \
  "$TARGET_URL"

convert "$TMP_TALL" -crop 1440x1200+0+180 +repage assets/readme_hero.png
convert "$TMP_TALL" -crop 1440x2300+0+2000 +repage assets/readme_showdown.png
rm -f "$TMP_TALL"

echo "Wrote assets/readme_hero.png"
echo "Wrote assets/readme_showdown.png"
