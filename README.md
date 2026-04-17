# tiny-neural-os

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Srinivas-Raghav-VC/tiny-neural-os/blob/main/notebooks/neural_computers_competition.py)

A research notebook that turns the **Neural Computers** idea into a small, reproducible terminal benchmark:

> Given the current terminal screen, can a model predict the next one?

It is built for **public sharing on molab** and for a competition-style review workflow.
The notebook now leans into marimo's reactive model instead of acting like a static report.

## What feels interactive now

- a live toy-terminal playground with step-by-step transitions
- side-by-side before/after terminal views
- a benchmark explorer with linked metric + setting controls
- tabbed analysis views for overview, pair comparison, and raw evidence
- interactive matplotlib viewers through `mo.mpl.interactive(...)`

## Snapshot

<p align="center">
  <img src="assets/readme_hero.png" alt="Notebook hero" width="100%" />
</p>

<p align="center">
  <img src="assets/readme_showdown.png" alt="Benchmark charts" width="100%" />
</p>

## Open this first

- **Notebook source (canonical):** `notebooks/neural_computers_competition.py`
- **Open in molab:**
  `https://molab.marimo.io/github/Srinivas-Raghav-VC/tiny-neural-os/blob/main/notebooks/neural_computers_competition.py`
- **Rendered HTML export:** `outputs/rendered/neural_computers_competition.html`

## Evidence-backed story in one view

From the saved benchmark runs in this repository:

- **MLP** is strongest on mean overall changed-cell accuracy.
- **Transformer** is the interesting contrast model: relatively stronger on Enter-heavy steps.
- **GRU** is a negative result in this setup.
- Transformer’s Enter edge is **not universal**: it shrinks strongly under paraphrase settings.

Primary benchmark file used by the notebook:
- `experiments/toy_nc_cli/results/baseline_comparison.csv`

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
marimo edit --watch notebooks/neural_computers_competition.py
```

## Rebuild public artifacts (end-to-end)

```bash
bash scripts/refresh_public_artifacts.sh
```

This runs:
1. `marimo check --fix`
2. session regeneration for the competition notebook
3. HTML export
4. README screenshot refresh

## Agent / editor scaffolding

For marimo-aware coding-agent workflows, the repo now includes:

- `CLAUDE.md` — project-specific marimo guidance
- `.claude/commands/marimo-check.md` — a slash-command starter for `marimo check --fix`

## Repository layout

- `notebooks/`
  - canonical notebook: `neural_computers_competition.py`
  - older exploratory variants kept as references
- `experiments/toy_nc_cli/`
  - toy terminal benchmark code, training scripts, and saved result files
- `outputs/`
  - rendered notebook and project writeups
- `assets/`
  - README screenshot assets
- `scripts/`
  - notebook export/session/screenshot automation

## Sources

- Neural Computers (2026): https://arxiv.org/abs/2604.06425
- marimo gallery: https://marimo.io/gallery/
- marimo gallery examples repository: https://github.com/marimo-team/gallery-examples
- marimo examples repository: https://github.com/marimo-team/examples
- marimo skills repository: https://github.com/marimo-team/skills
