# Remote GRU baseline run

## VM
- Host: `10.10.0.215`
- User: `srinivasr`
- Run style: remote SSH + `uv` + `tmux`
- tmux session: `nc-gru`

## What I did
1. Verified SSH access.
2. Synced the project to `~/alpha-proj` on the VM.
3. Created a remote launcher script:
   - `experiments/toy_nc_cli/scripts/remote_gru_uv.sh`
4. Used `uv` on the VM to:
   - install Python 3.12
   - create `.venv`
   - install `numpy`, `pandas`, `matplotlib`, `scikit-learn`
   - install `torch`, `torchvision`, `torchaudio` from the CUDA 12.4 wheel index
5. Launched the GRU experiment in `tmux`.
6. Debugged one runtime failure (`torch.cuda.ShortTensor` passed into embedding) by patching `gru_model.py` to cast rollout inputs to `long`.
7. Re-launched and completed the run successfully.

## Remote environment observed
From `experiments/toy_nc_cli/results/gru_remote_meta.txt`:
- Python: `3.12.12`
- uv: `0.11.4`
- tmux: `3.0a`
- torch: `2.6.0+cu124`
- CUDA available: `True`
- Device: `NVIDIA A100-PCIE-40GB MIG 7g.40gb`

## Result artifacts
Local copies fetched back into the repo:
- `experiments/toy_nc_cli/results/gru_results.json`
- `experiments/toy_nc_cli/results/gru_summary.csv`
- `experiments/toy_nc_cli/results/gru_remote_meta.txt`

## Headline result
This first GRU baseline **did not beat** the existing MLP notebook baseline.

### GRU summary
- `standard_family`: changed-cell `0.194`, Enter-step `0.121`
- `standard_command`: changed-cell `0.218`, Enter-step `0.134`
- `paraphrase_family`: changed-cell `0.188`, Enter-step `0.170`
- `paraphrase_command`: changed-cell `0.195`, Enter-step `0.172`

### Comparison to the current notebook story
The current MLP paraphrase study is much stronger on overall changed-cell accuracy, and in the family-conditioned case it is also much stronger on Enter-step accuracy:
- MLP paraphrase `family` Enter-step mean: about `0.359`
- MLP paraphrase `command` Enter-step mean: about `0.182`
- GRU paraphrase `family` Enter-step: `0.170`
- GRU paraphrase `command` Enter-step: `0.172`

## Interpretation
This means the **first stateful baseline is currently underperforming**, not improving the story.
That is still useful:
- the remote pipeline works,
- the GRU code runs on GPU,
- and we now have a concrete negative result to debug rather than a vague hope.

## Most likely next fixes
1. Stop evaluating only fully autoregressive rollout for every metric; also score teacher-forced next-step prediction.
2. Increase training data and/or epochs.
3. Check whether the current GRU input representation is too lossy.
4. Add progress logging and per-epoch validation to catch undertraining earlier.
5. Compare against the MLP on a matched train/test budget and matched evaluation mode.
