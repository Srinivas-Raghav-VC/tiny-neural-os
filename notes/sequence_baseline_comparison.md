# Sequence baseline comparison

## Scope
Matched-budget comparison using the toy CLI task across four target settings:
- `standard_family`
- `standard_command`
- `paraphrase_family`
- `paraphrase_command`

Compared models:
- MLP (matched rerun on local machine)
- GRU (remote GPU run)
- Transformer (remote GPU run)
- Mamba (`mamba-ssm` attempt on remote GPU VM; install/import failed)

## Files
- `experiments/toy_nc_cli/results/mlp_matched_results.json`
- `experiments/toy_nc_cli/results/gru_results.json`
- `experiments/toy_nc_cli/results/transformer_results.json`
- `experiments/toy_nc_cli/results/baseline_comparison.csv`
- `experiments/toy_nc_cli/results/mamba_remote_meta.txt`
- `experiments/toy_nc_cli/results/mamba_remote.log`

## Main observations

### 1. MLP is still the strongest overall baseline
Across all four matched settings, the MLP had the highest overall changed-cell accuracy.

Examples:
- `standard_command`: MLP `0.947` vs Transformer `0.476` vs GRU `0.218`
- `paraphrase_command`: MLP `0.716` vs Transformer `0.496` vs GRU `0.195`

### 2. GRU underperformed badly
The first GRU baseline remained far behind both MLP and Transformer.

This means "add recurrence" did **not** automatically improve the task.

### 3. Transformer is interesting but not a clear winner
The Transformer did not beat the MLP overall, but it produced one genuinely interesting pattern:
- on some **Enter-step** settings, it did much better than the MLP
- but it often lost a lot on typing/local changed-cell accuracy

Examples:
- `standard_family` Enter-step changed accuracy:
  - Transformer `0.816`
  - MLP `0.484`
- `standard_command` Enter-step changed accuracy:
  - Transformer `0.578`
  - MLP `0.486`
- `paraphrase_family` Enter-step changed accuracy:
  - Transformer `0.472`
  - MLP `0.518`
- `paraphrase_command` Enter-step changed accuracy:
  - Transformer `0.421`
  - MLP `0.413`

So the Transformer looks more competitive on the "what happens after Enter" part, but worse on the total next-screen job.

### 4. Real Mamba was attempted and failed for environment reasons
The VM successfully installed PyTorch, but `mamba-ssm` was not importable because the packaged CUDA extension required `GLIBC_2.32`, which the VM image does not provide.

Observed error:
- `mamba_ssm_import=False: ... libc.so.6: version GLIBC_2.32 not found ...`

This means we did **try** a real Mamba path, but it is blocked by system-library compatibility on the current VM image.

## What this means for the notebook story
Right now the cleanest evidence-backed story is:
- **MLP** is still the best overall toy baseline.
- **GRU** did not help.
- **Transformer** is the most interesting alternate baseline because it seems to help on some Enter-step semantics, even though it loses overall.
- **Mamba** was attempted but is currently blocked by the VM/runtime stack, not by lack of effort.

## Recommended next decision
For the final notebook, the strongest grounded move is probably:
1. keep the MLP as the main baseline,
2. optionally add Transformer as a "what changes if the model sees the full sequence context?" comparison,
3. mention GRU as a negative result if useful,
4. mention Mamba as attempted but environment-blocked unless we change VM image or build stack.
