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

marimo export html notebooks/neural_computers_competition.py \
  --no-include-code \
  -o outputs/rendered/neural_computers_competition.html \
  -f >/dev/null

TMP_TALL="assets/.readme_tmp_tall.png"

"$CHROMIUM_BIN" \
  --headless \
  --disable-gpu \
  --no-sandbox \
  --run-all-compositor-stages-before-draw \
  --virtual-time-budget=5000 \
  --window-size=1440,7000 \
  --screenshot="$TMP_TALL" \
  "file://$ROOT_DIR/outputs/rendered/neural_computers_competition.html" >/dev/null 2>&1

convert "$TMP_TALL" -crop 1440x1200+0+180 +repage assets/readme_hero.png
convert "$TMP_TALL" -crop 1440x2300+0+4300 +repage assets/readme_showdown.png
rm -f "$TMP_TALL"

echo "Wrote assets/readme_hero.png"
echo "Wrote assets/readme_showdown.png"
