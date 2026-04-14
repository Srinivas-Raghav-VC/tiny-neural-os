# Neural Computers toy project — immediate next steps

## What is already done
- Curiosity-first marimo notebook redesign completed in `notebooks/neural_computers_v2.py`
- Rendered HTML export completed in `outputs/rendered/neural_computers_v2.html`
- Remote GPU pipeline on `10.10.0.215` works with SSH + `uv` + `tmux`
- First GRU baseline completed remotely and fetched back
- `paper-finder` skill repo cloned into `third_party/paper-finder/`
- Project subagent `paper-finder` created at `.pi/agents/paper-finder.md`

## What is not done yet
- Tiny Transformer baseline
- Mamba / Mamba-style baseline
- Fair side-by-side comparison table across MLP / GRU / Transformer / Mamba
- Final notebook integration of any improved baselines
- Final competition-polish pass and export

## Recommended order
1. Implement tiny Transformer baseline.
2. Try real `mamba-ssm` on the GPU VM if install works cleanly.
3. If `mamba-ssm` is too fragile, implement a simple Mamba-style state-space baseline instead.
4. Run all baselines on matched splits and collect:
   - overall changed-cell accuracy
   - typing changed-cell accuracy
   - Enter changed-cell accuracy
   - exact-line accuracy
5. Only integrate baselines into the notebook if they improve or sharpen the story.
6. Finalize the notebook and export the submission artifact.

## Strategic note
The current GRU result is a useful negative result. It does not yet improve the notebook story, so the next baselines should be treated as pressure tests, not guaranteed upgrades.
