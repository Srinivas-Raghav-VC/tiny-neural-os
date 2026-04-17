# Marimo notebook assistant

This repository's primary public artifact is:

- `notebooks/neural_computers_competition.py`

It is a marimo notebook and should be edited as a marimo notebook first, not as a plain Python script.

## Project expectations

- Prefer a **clean, professional, public-facing** notebook.
- Optimize for **molab / GitHub-backed viewing**.
- Keep charts in **light mode**.
- Prefer **reactive controls** over static prose.
- Do not add cute metaphors, emoji-heavy styling, or unrelated decorative language.
- Evidence-backed claims only: numbers must trace to saved artifacts in `experiments/toy_nc_cli/results/`.

## Marimo fundamentals

- Cells execute reactively based on variable dependencies.
- Do not redeclare the same global variable in multiple cells.
- UI elements must be created in one cell and read via `.value` in downstream cells.
- The last expression in a cell is displayed automatically.
- Prefer marimo-native layouts (`mo.hstack`, `mo.vstack`, `mo.ui.tabs`, `mo.accordion`, `mo.stat`).

## Editing rules

- Keep imports centralized in the setup/import area.
- Do not import the same library twice.
- Do not define the same variable name in multiple cells.
- Prefer private local names with underscore prefixes inside cells when temporary variables are needed.
- If a visualization is meant to display, ensure the displayed object is the final rendered object for that cell.
- For matplotlib charts, prefer `mo.mpl.interactive(fig)` when interactivity materially helps.

## Verification workflow

After editing the notebook, run:

```bash
source .venv/bin/activate
marimo check notebooks/neural_computers_competition.py
marimo export html notebooks/neural_computers_competition.py --no-include-code -o outputs/rendered/neural_computers_competition.html -f
python3 scripts/create_sessions.py notebooks/neural_computers_competition.py
python3 scripts/validate_sessions.py
```

To refresh public-facing assets:

```bash
bash scripts/refresh_public_artifacts.sh
```

## Evidence files

Primary saved benchmark artifact:

- `experiments/toy_nc_cli/results/baseline_comparison.csv`

Related saved result files:

- `experiments/toy_nc_cli/results/mlp_matched_results.json`
- `experiments/toy_nc_cli/results/transformer_results.json`
- `experiments/toy_nc_cli/results/gru_results.json`

Do not invent or hardcode benchmark numbers that are not backed by these files.
