import marimo

__generated_with = "0.23.1"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "experiments" / "toy_nc_cli"))

    from src.toy_terminal import (
        TerminalConfig,
        generate_episodes,
        html_screen,
        char_accuracy,
        changed_cell_accuracy,
        exact_line_accuracy,
    )
    from src.cell_model import CellUpdateModel, ModelConfig, evaluate_model
    from src.studies import (
        fit_bundle,
        conditioning_study_multiseed,
        noise_study_multiseed,
        paraphrase_generalization_multiseed,
    )

    return (
        CellUpdateModel,
        ModelConfig,
        ROOT,
        TerminalConfig,
        char_accuracy,
        changed_cell_accuracy,
        conditioning_study_multiseed,
        evaluate_model,
        exact_line_accuracy,
        fit_bundle,
        generate_episodes,
        html_screen,
        mo,
        noise_study_multiseed,
        np,
        paraphrase_generalization_multiseed,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
# Neural Computers in Miniature

A CPU-friendly marimo notebook inspired by **Neural Computers** (Zhuge et al., 2026, arXiv:2604.06425).

**Notebook goal:** bring one core idea of the paper to life: a model can learn short-horizon interface dynamics directly from aligned **I/O traces**.

This notebook does **not** attempt a paper-faithful Wan2.1 video replication. Instead, it builds a small terminal world where we can test three paper-inspired questions interactively:

1. **Can a learned update rule predict terminal evolution from traces?**
2. **Does stronger conditioning help?**
3. **Does clean data help more than noisy data?**

The toy system uses the visible character grid as the runtime state, and trains a local **cell-update MLP** to predict the next screen from the current screen plus optional conditioning.

What makes this interesting is not just "does it work?" but **what kind of computer behavior is easy to learn first**:
- surface rendering,
- local control,
- action-conditioned state updates,
- or genuine symbolic correctness.

**Primary sources**
- Paper: https://arxiv.org/abs/2604.06425
- PDF: https://arxiv.org/pdf/2604.06425
- Project page: https://metauto.ai/neuralcomputer/
- Competition page: https://marimo.io/pages/events/notebook-competition
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Why this toy abstraction?

The original paper trains large video models on CLI and GUI trajectories. For a notebook competition, that is the wrong scale.

This notebook keeps the **core learning problem** while changing the substrate:

- **paper:** pixels + diffusion/video backbone
- **notebook:** character-grid terminal + tiny MLP

That tradeoff buys us:
- CPU-only execution,
- exact text-space metrics,
- interactive experiments,
- and a clearer view of *what the model actually learned*.

The interesting question is not photorealism. It is whether the model learns **local runtime primitives**: typing, line continuation, output placement, and prompt progression.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Curiosity guide

A smart reader should leave this notebook with answers to three concrete questions:

1. **What does a model learn first when it learns from interface traces?**
2. **Does more conditioning always help, or does it create tradeoffs?**
3. **Is clean data always best, or is broad/noisy coverage sometimes more robust?**

The goal is not to worship the paper or the toy model. The goal is to use the toy model to expose a few real tensions inside the paper's thesis.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## In plain English: what is happening here?

Imagine recording a terminal screen while someone types commands.

- When they type **`w`**, the next screen is easy to predict: the letter `w` appears.
- When they press **Enter** after `whoami`, the next screen is harder to predict: now the model must know what that command *means*.

So this notebook separates two kinds of skill:
1. **mechanics** — copying the visible effect of typing and cursor movement
2. **semantics** — knowing what output should appear after a command runs

That is the main curiosity of the notebook: **does a model first learn how a computer looks, or how it actually works?**
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Could this be a Transformer or Mamba-style model instead?

Yes — and that is actually closer to the spirit of the paper.

### Layman version
Our current notebook model is like a **local clerk**: it looks at a small patch of the screen and decides what nearby characters should become next.

A **Transformer** is more like a reader who can glance across the whole screen at once.
A **Mamba / state-space model** is more like a reader with a strong running memory of what has been happening over time.

### Why we did *not* start there
For the competition, we wanted something:
- CPU-friendly,
- fast enough for a notebook,
- easy to understand,
- and easy to audit.

That is why the first version uses a tiny cell-update MLP.

### What a stronger version would look like
If we wanted to move closer to the actual paper, the upgrade path would be:
1. **current notebook:** local MLP over screen patches
2. **better stateful baseline:** GRU / small recurrent model over screen + action sequence
3. **stronger global baseline:** tiny Transformer over screen tokens + action tokens
4. **most NC-like compact state model:** Mamba / state-space model with an explicit evolving hidden state

### Honest recommendation
For this notebook, a **small recurrent or Transformer baseline** is the most realistic next upgrade.
A full Mamba variant would be interesting, but it adds dependency risk and may make the notebook harder to run in a CPU-only competition setting.
"""
    )
    return


@app.cell
def _(mo):
    train_n = mo.ui.slider(12, 36, value=24, step=6, label="Train episodes")
    test_n = mo.ui.slider(6, 18, value=12, step=6, label="Test episodes")
    condition_level = mo.ui.dropdown(
        options=["none", "family", "command"],
        value="family",
        label="Conditioning level",
    )
    noisy_train = mo.ui.checkbox(value=False, label="Train on noisy traces")
    hidden_size = mo.ui.slider(64, 160, value=128, step=32, label="Hidden units")
    max_iter = mo.ui.slider(20, 60, value=40, step=10, label="MLP iterations")
    negative_ratio = mo.ui.slider(4, 12, value=8, step=2, label="Unchanged:changed sample ratio")

    controls = mo.vstack(
        [
            mo.md("## Interactive training controls"),
            mo.hstack([train_n, test_n, condition_level]),
            mo.hstack([noisy_train, hidden_size, max_iter, negative_ratio]),
        ]
    )
    controls
    return (
        condition_level,
        hidden_size,
        max_iter,
        negative_ratio,
        noisy_train,
        test_n,
        train_n,
    )


@app.cell
def _(
    condition_level,
    evaluate_model,
    fit_bundle,
    hidden_size,
    max_iter,
    negative_ratio,
    noisy_train,
    pd,
    test_n,
    train_n,
):
    bundle = fit_bundle(
        train_n=train_n.value,
        test_n=test_n.value,
        noisy_train=noisy_train.value,
        noisy_test=False,
        condition_level=condition_level.value,
        context_mode="command",
        hidden_size=hidden_size.value,
        max_iter=max_iter.value,
        negative_ratio=negative_ratio.value,
        seed=7,
    )

    config = bundle["config"]
    model = bundle["model"]
    train_eps = bundle["train_eps"]
    test_eps = bundle["test_eps"]
    learned_metrics = bundle["metrics"]
    copy_metrics = evaluate_model(None, test_eps, baseline=True)
    heuristic_metrics = evaluate_model(None, test_eps, heuristic=True)

    metrics_df = pd.DataFrame(
        [
            {"model": "copy baseline", **copy_metrics},
            {"model": "interface heuristic", **heuristic_metrics},
            {"model": f"cell MLP ({condition_level.value})", **learned_metrics},
        ]
    )
    metrics_df
    return config, metrics_df, model, test_eps, train_eps


@app.cell(hide_code=True)
def _(metrics_df, mo, noisy_train):
    noisy_label = "noisy" if noisy_train.value else "clean"
    mo.md(
        f"""
## Main result snapshot

The table above compares three systems:

- **copy baseline**: never updates the screen
- **interface heuristic**: knows how typing/backspace/enter work, but not command semantics
- **cell MLP**: learns local terminal dynamics from traces

Interpret the metrics carefully:
- **char_acc**: overall screen character accuracy
- **changed_acc**: accuracy only on cells that actually change from one frame to the next
- **exact_line_acc**: fraction of exact line matches
- **arithmetic_exact_match**: exact final-screen match for arithmetic episodes only

For this toy setting, **changed-cell accuracy** is the most informative metric. A copy baseline looks strong on overall accuracy because most terminal cells do not change on most steps.

Current training data: **{noisy_label}**
"""
    )
    return


@app.cell
def _(mo, test_eps):
    episode_idx = mo.ui.slider(0, max(len(test_eps) - 1, 0), value=0, step=1, label="Test episode")
    episode_idx
    return (episode_idx,)


@app.cell
def _(episode_idx, model, test_eps):
    episode = test_eps[episode_idx.value]
    pred_frames = model.rollout(episode)
    max_step = len(episode.actions) - 1
    return episode, max_step, pred_frames


@app.cell
def _(max_step, mo):
    step_idx = mo.ui.slider(0, max_step, value=min(3, max_step), step=1, label="Rollout step")
    step_idx
    return (step_idx,)


@app.cell
def _(
    char_accuracy,
    changed_cell_accuracy,
    episode,
    exact_line_accuracy,
    html_screen,
    mo,
    pred_frames,
    step_idx,
):
    step = step_idx.value
    action = episode.actions[step]
    prev = pred_frames[step]
    gt_prev = episode.frames[step]
    pred = pred_frames[step + 1]
    truth = episode.frames[step + 1]

    step_metrics = {
        "char_acc": round(char_accuracy(pred, truth), 3),
        "changed_acc_gt_mask": round(changed_cell_accuracy(gt_prev, pred, truth), 3),
        "exact_line_acc": round(exact_line_accuracy(pred, truth), 3),
    }

    comparison_html = f"""
<table style='width:100%; border-collapse:separate; border-spacing:12px 0;'>
  <tr>
    <td style='vertical-align:top; width:33%'><b>Ground truth</b><br>{html_screen(truth, diff_to=pred)}</td>
    <td style='vertical-align:top; width:33%'><b>Prediction</b><br>{html_screen(pred, diff_to=truth)}</td>
    <td style='vertical-align:top; width:33%'><b>Previous predicted state</b><br>{html_screen(prev)}</td>
  </tr>
</table>
"""

    mo.vstack(
        [
            mo.md(
                f"""
## Rollout viewer

**Episode family:** `{episode.family}`  
**Command:** `{episode.command_text}`  
**Variant tag:** `{episode.command_variant}`  
**Action at this step:** `{action.display_text}`

**Step metrics:** `{step_metrics}`
"""
            ),
            mo.md(comparison_html),
        ]
    )
    return action, pred, prev, step, step_metrics, truth


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Fixed study 1 — conditioning strength

The paper reports that stronger conditioning materially helps terminal generation. In this toy notebook we compare three levels:

- **none**: only the current character grid
- **family**: grid + coarse command family and action kind
- **command**: grid + family + typed character + a caption-like script string

The last mode is intentionally **caption-conditioned**, analogous to the paper's prompt/caption conditioning. It is not meant to be a leak-free online setting; it is a toy analogue of “richer external specification helps generation.”

The chart below reports **3-seed means with standard deviations**, so the result is not riding on a single lucky split.
"""
    )
    return


@app.cell
def _(conditioning_study_multiseed):
    cond_df = conditioning_study_multiseed(
        train_n=18,
        test_n=12,
        hidden_size=96,
        max_iter=40,
        negative_ratio=8,
        seeds=(11, 17, 23),
    )
    cond_df
    return (cond_df,)


@app.cell
def _(cond_df, plt):
    cond_fig, cond_ax = plt.subplots(figsize=(7, 3.5))
    cond_ax.bar(cond_df["conditioning"], cond_df["changed_acc_mean"], yerr=cond_df["changed_acc_std"], color=["#9ca3af", "#60a5fa", "#34d399"], capsize=4)
    cond_ax.set_ylim(0, 1)
    cond_ax.set_ylabel("changed-cell accuracy")
    cond_ax.set_title("Richer conditioning improves changed-cell prediction")
    cond_fig.tight_layout()
    cond_fig
    return cond_ax, cond_fig


@app.cell(hide_code=True)
def _(cond_df, mo):
    best_changed = cond_df.sort_values("changed_acc_mean", ascending=False).iloc[0]
    best_line = cond_df.sort_values("exact_line_acc_mean", ascending=False).iloc[0]
    mo.md(
        f"""
### Honest read of the conditioning study

Two things can both be true:

- **`{best_changed['conditioning']}` wins on changed-cell accuracy** — richer conditioning helps the model place the *right local edits*.
- **`{best_line['conditioning']}` wins on exact-line accuracy** — a coarser signal can sometimes produce more stable full lines.

That is more interesting than a simple “more context is always better” story. In this toy setting, extra conditioning helps with **where to change**, but it does not automatically maximize **whole-screen stability**.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Fixed study 2 — clean vs noisy data

One of the paper's most useful practical observations is that **data quality matters a lot**. To separate that from train/test mismatch, this notebook evaluates a full **2×2 transfer matrix**:

- clean → clean
- noisy → clean
- clean → noisy
- noisy → noisy

That lets us ask a sharper question: is the gain really about cleaner supervision, or just about evaluation mismatch?
"""
    )
    return


@app.cell
def _(noise_study_multiseed):
    noise_df = noise_study_multiseed(
        train_n=18,
        test_n=12,
        hidden_size=96,
        max_iter=40,
        negative_ratio=8,
        seeds=(13, 19, 29),
    )
    noise_df
    return (noise_df,)


@app.cell
def _(noise_df, plt):
    order = ["clean→clean", "noisy→clean", "clean→noisy", "noisy→noisy"]
    palette = ["#34d399", "#f59e0b", "#60a5fa", "#ef4444"]
    ordered_noise_df = noise_df.set_index("setting").loc[order].reset_index()
    noise_fig, noise_ax = plt.subplots(figsize=(8, 3.8))
    noise_ax.bar(ordered_noise_df["setting"], ordered_noise_df["changed_acc_mean"], yerr=ordered_noise_df["changed_acc_std"], color=palette, capsize=4)
    noise_ax.set_ylim(0, 1)
    noise_ax.set_ylabel("changed-cell accuracy")
    noise_ax.set_title("Data cleanliness helps, and mismatch hurts")
    noise_fig.tight_layout()
    noise_fig
    return noise_ax, noise_fig


@app.cell(hide_code=True)
def _(mo, noise_df):
    ordered = noise_df.set_index("setting")
    best_setting = noise_df.sort_values("changed_acc_mean", ascending=False).iloc[0]["setting"]
    clean_clean = ordered.loc["clean→clean", "changed_acc_mean"]
    noisy_clean = ordered.loc["noisy→clean", "changed_acc_mean"]
    clean_noisy = ordered.loc["clean→noisy", "changed_acc_mean"]
    mo.md(
        f"""
### Honest read of the noise study

This is the most surprising result in the notebook.

- The best changed-cell regime in this run is **`{best_setting}`**.
- **noisy→clean** performs about as well as, or even slightly better than, **clean→clean** on local update accuracy.
- **clean→noisy** is much worse, which means narrow clean training does not prepare the model for interface variation.

That is a more interesting lesson than “clean data good, noisy data bad.” In this toy setting, **coverage can beat tidiness** for robustness, even if clean data still gives the most controlled learning environment.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Fixed study 3 — unseen paraphrases

Now we ask a more skeptical question.

Suppose the model trains only on one phrasing per command family:
- `pwd`
- `whoami`
- `echo $HOME`
- `date`
- `env | head -n 3`

Then we test it on **different phrasings with the same meaning**, such as:
- `pwd -P`
- `id -un`
- `printf '%s\n' $HOME`
- `date +%F`
- `printenv | head -n 3`

This lets us separate two kinds of generalization:
- **typing generalization**: can it keep up with new visible character sequences?
- **execution generalization**: after Enter, can it still produce the right kind of output?
"""
    )
    return


@app.cell
def _(paraphrase_generalization_multiseed):
    paraphrase_df = paraphrase_generalization_multiseed(
        train_n=15,
        test_n=15,
        hidden_size=96,
        max_iter=40,
        negative_ratio=8,
        seeds=(31, 37, 41),
    )
    paraphrase_df
    return (paraphrase_df,)


@app.cell
def _(np, paraphrase_df, plt):
    width = 0.35
    x = np.arange(len(paraphrase_df))
    para_fig, para_ax = plt.subplots(figsize=(8, 4))
    para_ax.bar(x - width / 2, paraphrase_df["typing_changed_acc_mean"], width, yerr=paraphrase_df["typing_changed_acc_std"], label="typing steps", color="#60a5fa", capsize=4)
    para_ax.bar(x + width / 2, paraphrase_df["enter_changed_acc_mean"], width, yerr=paraphrase_df["enter_changed_acc_std"], label="enter steps", color="#f97316", capsize=4)
    para_ax.set_xticks(x)
    para_ax.set_xticklabels(paraphrase_df["conditioning"])
    para_ax.set_ylim(0, 1)
    para_ax.set_ylabel("changed-cell accuracy")
    para_ax.set_title("Unseen paraphrases: mechanics transfer better than command execution")
    para_ax.legend()
    para_fig.tight_layout()
    para_fig
    return para_ax, para_fig


@app.cell(hide_code=True)
def _(mo, paraphrase_df):
    best_enter = paraphrase_df.sort_values("enter_changed_acc_mean", ascending=False).iloc[0]["conditioning"]
    mo.md(
        f"""
### Honest read of the paraphrase study

This is the clearest layman lesson in the notebook.

- On **typing steps**, the model is mostly learning visible mechanics: new phrasing is not catastrophic.
- On **Enter steps**, the problem becomes semantic: the model must decide what output belongs to the command.
- In this run, **`{best_enter}`** gives the best Enter-step generalization on unseen paraphrases.

So the notebook supports a very intuitive picture: a neural model can learn to behave like a terminal **before** it reliably understands what every command means.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## What this toy notebook shows

### Observations
- A small learned update rule can model short-horizon terminal changes from I/O traces.
- Richer conditioning improves local control.
- Cleaner traces improve learning.
- Arithmetic exact match remains weak even when surface rendering is good.

### Inference
This supports the paper's broad claim that **early runtime primitives** are learnable from aligned traces, while also echoing its warning that **symbolic reliability is harder than interface rendering**.

More specifically, the notebook suggests three curiosity-driven lessons:
- local interface updates are learnable before robust symbolic correctness,
- richer conditioning helps but introduces tradeoffs,
- and “clean vs noisy” is really a question about **signal quality vs coverage**.

### Limitations
- This notebook uses a character grid, not video latents.
- It does not reproduce the paper's Wan2.1-scale setup.
- It models local screen updates, not long-horizon reusable routines.

That said, the notebook is meant to do one thing well: make the paper's core idea **intuitive, inspectable, and interactive**.
"""
    )
    return


if __name__ == "__main__":
    app.run()
