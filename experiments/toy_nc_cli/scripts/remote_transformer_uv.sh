#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$HOME/alpha-proj}"
LOG_PATH="$ROOT_DIR/experiments/toy_nc_cli/results/transformer_remote.log"
META_PATH="$ROOT_DIR/experiments/toy_nc_cli/results/transformer_remote_meta.txt"

mkdir -p "$ROOT_DIR/experiments/toy_nc_cli/results"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

cd "$ROOT_DIR"
uv python install 3.12
uv venv --clear --python 3.12 .venv
source .venv/bin/activate
uv pip install -q numpy pandas matplotlib scikit-learn
uv pip install -q --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "whoami=$(whoami)"
  echo "python=$(python --version 2>&1)"
  echo "uv=$(uv --version 2>&1)"
  python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
PY
} | tee "$META_PATH"

python experiments/toy_nc_cli/scripts/train_transformer_baseline.py 2>&1 | tee "$LOG_PATH"
