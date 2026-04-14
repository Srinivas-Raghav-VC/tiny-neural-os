# Neural Computers mini notebook — verification log

## Environment
- Execution environment: Python virtual environment at `.venv`
- Installed core packages: marimo, numpy, matplotlib, pandas, scikit-learn

## Files created
- `notebooks/neural_computers_mini.py`
- `experiments/toy_nc_cli/src/toy_terminal.py`
- `experiments/toy_nc_cli/src/cell_model.py`
- `experiments/toy_nc_cli/src/studies.py`
- `experiments/toy_nc_cli/scripts/smoke_test.py`
- `requirements.txt`
- `outputs/rendered/neural_computers_mini.html`

## Checks run

### 1. Syntax check
Command:
```bash
. .venv/bin/activate && python -m py_compile notebooks/neural_computers_mini.py \
  experiments/toy_nc_cli/src/toy_terminal.py \
  experiments/toy_nc_cli/src/cell_model.py \
  experiments/toy_nc_cli/src/studies.py \
  experiments/toy_nc_cli/scripts/smoke_test.py
```
Result: passed.

### 2. Smoke test
Command:
```bash
. .venv/bin/activate && python experiments/toy_nc_cli/scripts/smoke_test.py
```
Result summary:
- determinism_ok: true
- roundtrip_ok: true
- copy_overfit char/changed/exact-line: 0.970 / 0.211 / 0.879
- learned_overfit char/changed/exact-line: 0.973 / 0.910 / 0.854
- heldout_copy char/changed/exact-line: 0.970 / 0.209 / 0.876
- heldout learned char/changed/exact-line: 0.970 / 0.812 / 0.827
- heldout arithmetic exact match: 0.0

Interpretation:
- The toy model beats the copy baseline strongly on **changed-cell accuracy**.
- It also slightly beats the copy baseline on **overall character accuracy** in the smoke test.
- Arithmetic exact match remains poor, which is consistent with the notebook's intended “rendering/control is easier than symbolic correctness” message.

### 3. marimo export check
Command:
```bash
. .venv/bin/activate && marimo export html notebooks/neural_computers_mini.py \
  -o outputs/rendered/neural_computers_mini.html -f
```
Result: succeeded and produced a rendered HTML artifact.

### 4. Fixed-study summaries used in the notebook
Command:
```bash
. .venv/bin/activate && python - <<'PY'
from pathlib import Path
import sys
ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT / 'experiments' / 'toy_nc_cli'))
from src.studies import conditioning_study_multiseed, noise_study_multiseed
print(conditioning_study_multiseed(18, 12, 96, 40, 8, (11, 17, 23)))
print(noise_study_multiseed(18, 12, 96, 40, 8, (13, 19, 29)))
PY
```
Observed headline pattern:
- Conditioning study: `none` is worst; `family` and `command` are better on changed-cell accuracy.
- Noise study: `clean→clean` is the strongest regime; cross-distribution transfer degrades performance.

### 5. Curiosity-first marimo V2 export check
Command:
```bash
. .venv/bin/activate && marimo export html notebooks/neural_computers_v2.py \
  -o outputs/rendered/neural_computers_v2.html -f --no-include-code
```
Result: succeeded and produced a rendered HTML artifact.

Observed V2 additions:
- hero / guided-story layout
- live toy terminal viewer with highlighted changes
- step-type breakdown chart showing typing vs Enter gap
- baseline showdown tabs on the same step
- screen-level error map and zoomed wrong-cell patch viewer
- experiment tabs and scoreboard cards
- failure gallery with real examples from the test set

### 6. Remote GPU GRU run
Commands used (summarized):
```bash
ssh srinivasr@10.10.0.215
# sync project
# run remote launcher inside tmux:
./experiments/toy_nc_cli/scripts/remote_gru_uv.sh ~/alpha-proj nc-gru
```
Result:
- remote run completed successfully on `NVIDIA A100-PCIE-40GB MIG 7g.40gb`
- fetched back:
  - `experiments/toy_nc_cli/results/gru_results.json`
  - `experiments/toy_nc_cli/results/gru_summary.csv`
  - `experiments/toy_nc_cli/results/gru_remote_meta.txt`
- debugged and fixed one rollout dtype bug in `experiments/toy_nc_cli/src/gru_model.py`

Headline finding:
- this first GRU baseline underperformed the current MLP baseline, especially on paraphrase Enter-step behavior
- remote pipeline is now working, but the GRU branch needs another iteration before it improves the notebook story

### 7. Remote GPU Transformer run
Command used (summarized):
```bash
./experiments/toy_nc_cli/scripts/remote_transformer_uv.sh ~/alpha-proj
```
Result:
- remote run completed successfully on the same VM/GPU stack
- fetched back:
  - `experiments/toy_nc_cli/results/transformer_results.json`
  - `experiments/toy_nc_cli/results/transformer_remote_meta.txt`
  - `experiments/toy_nc_cli/results/transformer_remote.log`

Headline finding:
- Transformer beat the GRU clearly
- Transformer did **not** beat the matched MLP overall
- Transformer was most interesting on some Enter-step settings, where it exceeded the MLP despite lower overall changed-cell accuracy

### 8. Real Mamba attempt on the GPU VM
Command used (summarized):
```bash
./experiments/toy_nc_cli/scripts/remote_mamba_uv.sh ~/alpha-proj
```
Result:
- `mamba-ssm` install/import path was attempted on the VM
- the package import failed due system-library mismatch:
  - `GLIBC_2.32 not found`
- logs and metadata fetched back:
  - `experiments/toy_nc_cli/results/mamba_remote_meta.txt`
  - `experiments/toy_nc_cli/results/mamba_remote.log`

Headline finding:
- a real Mamba path was attempted, but the current VM image is not compatible with the installed CUDA extension build
- this is an environment blocker, not a completed model result

### 9. Matched baseline comparison artifact
Created:
- `experiments/toy_nc_cli/results/mlp_matched_results.json`
- `experiments/toy_nc_cli/results/baseline_comparison.csv`
- `notes/sequence_baseline_comparison.md`

Matched comparison headline:
- MLP is still strongest overall
- GRU is weakest
- Transformer is the most interesting alternate baseline
- Mamba currently blocked by runtime compatibility

## Known caveats
- The MLP often reaches the max iteration count before formal convergence.
- Exact-line accuracy can lag behind changed-cell accuracy because small local mistakes can break an entire line match.
- The current notebook focuses on CLI only; GUI is intentionally deferred.
- `marimo check` still reports markdown-indentation warnings in `notebooks/neural_computers_v2.py`; these are cosmetic and did not block export.
