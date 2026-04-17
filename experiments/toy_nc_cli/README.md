# toy_nc_cli

Toy terminal benchmark code and baseline experiments used by the competition notebook.

## Core source files

- `src/toy_terminal.py` — terminal environment + episode generation
- `src/cell_model.py` — MLP-style cell baseline
- `src/transformer_model.py` — Transformer baseline
- `src/gru_model.py` — GRU baseline
- `src/mamba_model.py` — Mamba experiment scaffold
- `src/studies.py` — shared evaluation helpers

## Training / evaluation scripts

- `scripts/train_transformer_baseline.py`
- `scripts/train_gru_baseline.py`
- `scripts/train_mamba_baseline.py`
- `scripts/smoke_test.py`

Remote GPU helpers:
- `scripts/remote_transformer_uv.sh`
- `scripts/remote_gru_uv.sh`
- `scripts/remote_mamba_uv.sh`

## Benchmark artifacts consumed by notebook

- `results/baseline_comparison.csv`  ← primary table for notebook charts
- `results/mlp_matched_results.json`
- `results/transformer_results.json`
- `results/gru_results.json`

The notebook `notebooks/neural_computers_competition.py` reads these files directly.
Its explorer views are driven primarily by:

- overall changed-cell accuracy
- typing changed-cell accuracy
- Enter changed-cell accuracy
- plain character accuracy (included mainly as a cautionary easy metric)
