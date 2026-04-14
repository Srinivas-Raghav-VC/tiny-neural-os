# Neural Computers competition handoff

## Final notebook artifacts
- Source notebook: `notebooks/neural_computers_competition.py`
- Rendered HTML: `outputs/rendered/neural_computers_competition.html`

## Supporting experiments
- MLP matched results: `experiments/toy_nc_cli/results/mlp_matched_results.json`
- GRU results: `experiments/toy_nc_cli/results/gru_results.json`
- Transformer results: `experiments/toy_nc_cli/results/transformer_results.json`
- Combined comparison table: `experiments/toy_nc_cli/results/baseline_comparison.csv`
- Sequence comparison memo: `notes/sequence_baseline_comparison.md`

## Final story used in the notebook
1. Predicting the next screen from the current one is a toy version of the Neural Computers idea.
2. Typing/mechanics are easier than Enter/meaning.
3. The MLP is still strongest overall on this toy benchmark.
4. The Transformer is the interesting contrast model because it is relatively stronger on Enter-heavy semantic steps.
5. GRU is a negative result.
6. Mamba was attempted separately but is environment-blocked on the current VM due to a GLIBC mismatch.

## Why this version is the competition-facing one
This notebook is optimized for presentation and export:
- cleaner visual pacing
- fewer heavy live computations during render
- uses precomputed comparison results so the final HTML is stable and fast to load
- emphasizes the strongest audience-friendly scientific story instead of every experiment branch
- now includes an explicit benchmark/protocol section, an exact benchmark table, and claim text tightened to match the saved results
- removes unsupported hardcoded representative action numbers; charts now derive from the saved result files

## Notes
- The heavier exploratory version still exists at `notebooks/neural_computers_final.py`.
- The earlier curiosity-first redesign remains at `notebooks/neural_computers_v2.py`.
- The competition notebook is the recommended submission artifact.
- Latest polish pass completed after manual audit against:
  - `experiments/toy_nc_cli/results/mlp_matched_results.json`
  - `experiments/toy_nc_cli/results/transformer_results.json`
  - `experiments/toy_nc_cli/results/gru_results.json`
  - `experiments/toy_nc_cli/results/baseline_comparison.csv`

## Sources
- Neural Computers (2026): https://arxiv.org/abs/2604.06425
- Mamba-3 (2026): https://arxiv.org/abs/2603.15569
- Mamba repository: https://github.com/state-spaces/mamba
