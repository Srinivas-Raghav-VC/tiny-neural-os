# toy_nc_cli experiments

This directory contains the toy terminal benchmark, model baselines, and saved experiment artifacts used by the notebook.

## Source code

- `src/toy_terminal.py` — toy terminal environment and episode generation
- `src/cell_model.py` — local MLP-style cell update baseline
- `src/transformer_model.py` — Transformer baseline
- `src/gru_model.py` — GRU baseline
- `src/mamba_model.py` — attempted Mamba integration wrapper
- `src/studies.py` — study helpers used across experiments

## Scripts

- `scripts/train_transformer_baseline.py`
- `scripts/train_gru_baseline.py`
- `scripts/train_mamba_baseline.py`
- `scripts/remote_transformer_uv.sh`
- `scripts/remote_gru_uv.sh`
- `scripts/remote_mamba_uv.sh`

## Key saved results

- `results/mlp_matched_results.json`
- `results/transformer_results.json`
- `results/gru_results.json`
- `results/baseline_comparison.csv`

These are the main result files consumed by the competition notebook.
