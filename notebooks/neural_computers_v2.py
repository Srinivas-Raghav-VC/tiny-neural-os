# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="wide")


# ============================================================================
# IMPORTS
# ============================================================================
@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import sys
    import warnings
    from pathlib import Path
    from sklearn.exceptions import ConvergenceWarning

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "experiments" / "toy_nc_cli"))
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    from src.toy_terminal import (
        TerminalConfig,
        generate_episodes,
        html_screen,
        char_accuracy,
        changed_cell_accuracy,
        exact_line_accuracy,
        COMMAND_VARIANTS,
    )
    from src.cell_model import (
        CellUpdateModel,
        ModelConfig,
        action_kind_breakdown,
        copy_baseline_rollout,
        evaluate_model,
        heuristic_rollout,
    )
    from src.studies import (
        fit_bundle,
        conditioning_study_multiseed,
        noise_study_multiseed,
        paraphrase_generalization_multiseed,
    )

    # Style constants
    COLORS = {
        "correct": "#22c55e",
        "wrong": "#ef4444",
        "changed": "#f59e0b",
        "typing": "#3b82f6",
        "enter": "#f97316",
        "neutral": "#6b7280",
        "bg_dark": "#1e1e2e",
        "bg_light": "#313244",
    }

    return (
        COLORS,
        COMMAND_VARIANTS,
        CellUpdateModel,
        ModelConfig,
        ROOT,
        TerminalConfig,
        action_kind_breakdown,
        char_accuracy,
        changed_cell_accuracy,
        copy_baseline_rollout,
        conditioning_study_multiseed,
        evaluate_model,
        exact_line_accuracy,
        fit_bundle,
        generate_episodes,
        heuristic_rollout,
        html_screen,
        mo,
        noise_study_multiseed,
        np,
        paraphrase_generalization_multiseed,
        pd,
        plt,
    )


# ============================================================================
# GLOBAL STYLE
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        """
        <style>
        .nc-card {
            border-radius: 16px;
            padding: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        }
        .nc-small {
            color: #a6adc8;
            font-size: 0.92em;
            line-height: 1.5;
        }
        .nc-pill {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92em;
        }
        </style>
        """
    )
    return


# ============================================================================
# PETAL 0: THE HOOK
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    hero_html = """
    <div style="
        background: linear-gradient(135deg, #1e1e2e 0%, #313244 100%);
        border-radius: 16px;
        padding: 48px;
        text-align: center;
        margin-bottom: 24px;
    ">
        <h1 style="
            font-size: 2.8em;
            color: #cdd6f4;
            margin-bottom: 16px;
            font-weight: 700;
        ">🖥️ Can a Model Learn How a Computer Works?</h1>
        <p style="
            font-size: 1.4em;
            color: #a6adc8;
            max-width: 700px;
            margin: 0 auto 24px auto;
            line-height: 1.6;
        ">
            Not by reading code. Not by being taught.<br>
            <strong style="color: #f9e2af;">Just by watching the screen change.</strong>
        </p>
        <div style="
            display: inline-block;
            background: #45475a;
            border-radius: 8px;
            padding: 12px 24px;
            color: #89b4fa;
            font-family: monospace;
            font-size: 1.1em;
        ">
            Inspired by <a href="https://arxiv.org/abs/2604.06425" style="color: #89b4fa;">Neural Computers</a> (2026)
        </div>
    </div>
    """
    mo.Html(hero_html)
    return (hero_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""
**The big question:** If you show a model thousands of screen recordings, 
can it learn to predict what happens next?

This notebook lets you explore that question with a tiny toy computer.
"""),
        kind="info"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    journey_html = """
    <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 18px 0 26px 0;">
      <div class='nc-card' style='background:#1f293722; border-color:#3b82f6;'><div style='font-size:1.6em'>🌱</div><b>Watch</b><div class='nc-small'>See the toy computer step by step.</div></div>
      <div class='nc-card' style='background:#1f293722; border-color:#f59e0b;'><div style='font-size:1.6em'>🌸</div><b>Split</b><div class='nc-small'>Separate easy typing from hard meaning.</div></div>
      <div class='nc-card' style='background:#1f293722; border-color:#22c55e;'><div style='font-size:1.6em'>🌺</div><b>Train</b><div class='nc-small'>Give the tiny model examples and hints.</div></div>
      <div class='nc-card' style='background:#1f293722; border-color:#f97316;'><div style='font-size:1.6em'>🌻</div><b>Compare</b><div class='nc-small'>Put guesses next to reality and inspect errors.</div></div>
      <div class='nc-card' style='background:#1f293722; border-color:#a855f7;'><div style='font-size:1.6em'>🌷</div><b>Stress-test</b><div class='nc-small'>Try noise, hints, and new phrasing.</div></div>
    </div>
    """
    mo.Html(journey_html)
    return (journey_html,)


# ============================================================================
# PETAL 1: WATCH THE TOY COMPUTER
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🌱 Petal 1: Watch the Toy Computer

First, let's see what we're working with. Below is a tiny simulated terminal.

Pick a command, then step through the animation to see what happens.
""")
    return


@app.cell(hide_code=True)
def _(COMMAND_VARIANTS, mo):
    # Command picker
    family_options = list(COMMAND_VARIANTS.keys())
    family_picker = mo.ui.dropdown(
        options=family_options,
        value="whoami",
        label="🎯 Pick a command"
    )

    variant_picker = mo.ui.dropdown(
        options=["phrasing 1", "phrasing 2", "phrasing 3"],
        value="phrasing 1",
        label="📝 Pick a phrasing"
    )

    noisy_toggle = mo.ui.switch(value=False, label="Add visual noise")

    mo.hstack([family_picker, variant_picker, noisy_toggle], justify="center", gap=2)
    return family_options, family_picker, noisy_toggle, variant_picker


@app.cell(hide_code=True)
def _(TerminalConfig, family_options, family_picker, generate_episodes, noisy_toggle, variant_picker):
    # Generate one episode based on selection
    demo_config = TerminalConfig(rows=10, cols=40, context_width=32, patch_radius=1)
    variant_idx = int(variant_picker.value.split()[-1]) - 1
    family_seed = family_options.index(family_picker.value)
    demo_episodes = generate_episodes(
        n=1,
        config=demo_config,
        noisy=noisy_toggle.value,
        seed=42 + 10 * family_seed + variant_idx,
        context_mode="command",
        families=[family_picker.value],
        variant_indices_by_family={family_picker.value: (variant_idx,)},
    )
    demo_episode = demo_episodes[0]
    demo_max_step = len(demo_episode.actions)
    return demo_config, demo_episode, demo_episodes, demo_max_step, variant_idx


@app.cell(hide_code=True)
def _(demo_max_step, mo):
    demo_step = mo.ui.slider(
        0, demo_max_step,
        value=0,
        step=1,
        label="⏩ Step through time",
        full_width=True
    )
    demo_step
    return (demo_step,)


@app.cell(hide_code=True)
def _(COLORS, demo_episode, demo_max_step, demo_step, mo, np):
    demo_step_value = demo_step.value
    current_frame = demo_episode.frames[demo_step_value]

    # Find changed cells from previous frame
    if demo_step_value > 0:
        demo_prev_frame = demo_episode.frames[demo_step_value - 1]
        changed_mask = current_frame != demo_prev_frame
    else:
        demo_prev_frame = current_frame.copy()
        changed_mask = np.zeros_like(current_frame, dtype=bool)

    # Build styled terminal HTML
    def render_terminal_fancy(frame, changed_mask, title=""):
        rows = []
        for r in range(frame.shape[0]):
            chars = []
            for c in range(frame.shape[1]):
                ch = frame[r, c]
                if ch == " ":
                    ch = "&nbsp;"
                if changed_mask[r, c]:
                    chars.append(f'<span style="background:{COLORS["changed"]}; color:#000; font-weight:bold;">{ch}</span>')
                else:
                    chars.append(f'<span>{ch}</span>')
            rows.append("".join(chars))
        content = "<br>".join(rows)
        return f"""
        <div style="
            background: {COLORS['bg_dark']};
            border-radius: 12px;
            padding: 16px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.4;
            color: #cdd6f4;
            border: 2px solid {COLORS['bg_light']};
        ">
            <div style="color: #a6adc8; font-size: 12px; margin-bottom: 8px;">{title}</div>
            {content}
        </div>
        """

    # Action description
    if demo_step_value < len(demo_episode.actions):
        demo_action = demo_episode.actions[demo_step_value]
        if demo_action.kind == "type_char":
            action_desc = f"⌨️ Type: <code>{demo_action.typed_char}</code>"
            action_color = COLORS["typing"]
        elif demo_action.kind == "enter":
            action_desc = "⏎ Press Enter"
            action_color = COLORS["enter"]
        elif demo_action.kind == "backspace":
            action_desc = "⌫ Backspace"
            action_color = COLORS["typing"]
        else:
            action_desc = "⏸️ Idle"
            action_color = COLORS["neutral"]
    else:
        demo_action = None
        action_desc = "✅ Done"
        action_color = COLORS["correct"]

    terminal_html = render_terminal_fancy(
        current_frame,
        changed_mask,
        f"Command: {demo_episode.command_text}"
    )

    action_badge = f"""
    <div style="
        display: inline-block;
        background: {action_color};
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1em;
        margin-top: 12px;
    ">{action_desc}</div>
    """

    changed_count = int(changed_mask.sum())
    stats_html = f"""
    <div style="
        display: flex;
        gap: 16px;
        margin-top: 12px;
        justify-content: center;
    ">
        <div style="background: {COLORS['bg_light']}; padding: 8px 16px; border-radius: 8px; color: #cdd6f4;">
            📍 Step <strong>{demo_step_value}</strong> of {demo_max_step}
        </div>
        <div style="background: {COLORS['changed']}; padding: 8px 16px; border-radius: 8px; color: #000;">
            ✨ <strong>{changed_count}</strong> cells changed
        </div>
    </div>
    """

    mo.vstack([
        mo.Html(terminal_html),
        mo.Html(action_badge),
        mo.Html(stats_html),
    ], align="center")
    return (
        action_badge,
        action_color,
        action_desc,
        changed_count,
        changed_mask,
        current_frame,
        demo_action,
        demo_prev_frame,
        demo_step_value,
        render_terminal_fancy,
        stats_html,
        terminal_html,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""
**What you're seeing:** Each step, the user either types a character, presses Enter, or does nothing.
The <span style="background: #f59e0b; color: #000; padding: 2px 6px; border-radius: 4px;">highlighted cells</span> show what changed.

Notice: when you **type**, only 1-2 cells change. When you **press Enter**, many cells might change at once!
"""),
        kind="neutral"
    )
    return


# ============================================================================
# PETAL 2: EASY VS HARD
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🌸 Petal 2: What's Easy vs What's Hard?

Here's the key insight. There are two very different skills:
""")
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    comparison_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0;">
        <div style="
            background: linear-gradient(135deg, {COLORS['typing']}22 0%, {COLORS['typing']}11 100%);
            border: 2px solid {COLORS['typing']};
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        ">
            <div style="font-size: 3em; margin-bottom: 12px;">⌨️</div>
            <h3 style="color: {COLORS['typing']}; margin-bottom: 8px;">Mechanics</h3>
            <p style="color: #cdd6f4; font-size: 1.1em;">
                "The user typed <code>w</code>, so show <code>w</code>"
            </p>
            <div style="
                background: {COLORS['correct']};
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                margin-top: 16px;
                display: inline-block;
            ">✅ Easy to learn</div>
        </div>
        <div style="
            background: linear-gradient(135deg, {COLORS['enter']}22 0%, {COLORS['enter']}11 100%);
            border: 2px solid {COLORS['enter']};
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        ">
            <div style="font-size: 3em; margin-bottom: 12px;">⏎</div>
            <h3 style="color: {COLORS['enter']}; margin-bottom: 8px;">Meaning</h3>
            <p style="color: #cdd6f4; font-size: 1.1em;">
                "The user ran <code>whoami</code>, so show the username"
            </p>
            <div style="
                background: {COLORS['wrong']};
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                margin-top: 16px;
                display: inline-block;
            ">🔥 Hard to learn</div>
        </div>
    </div>
    """
    mo.Html(comparison_html)
    return (comparison_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""
**The curiosity:** Does a model first learn how a computer *looks* (mechanics), 
or how it actually *works* (meaning)?

Spoiler: it learns the looks first. The meaning is harder.
"""),
        kind="warn"
    )
    return


# ============================================================================
# PETAL 3: TRAIN A MODEL
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🌺 Petal 3: Train a Tiny Brain

Now let's train a small model and see what it learns.

The model looks at the current screen and tries to guess: **what will the next screen look like?**
""")
    return


@app.cell(hide_code=True)
def _(mo):
    # Training controls
    train_episodes = mo.ui.slider(12, 48, value=24, step=6, label="📚 Training examples")
    test_episodes = mo.ui.slider(6, 24, value=12, step=6, label="🧪 Test examples")
    hint_label = mo.ui.dropdown(
        options=[
            "No hints",
            "Tell it the command type",
            "Tell it the exact command",
        ],
        value="Tell it the command type",
        label="💡 How much help?"
    )

    mo.hstack([train_episodes, test_episodes, hint_label], justify="center", gap=2)
    return hint_label, test_episodes, train_episodes


@app.cell(hide_code=True)
def _(
    evaluate_model,
    fit_bundle,
    hint_label,
    test_episodes,
    train_episodes,
):
    # Train the model
    hint_map = {
        "No hints": "none",
        "Tell it the command type": "family",
        "Tell it the exact command": "command",
    }
    trained_bundle = fit_bundle(
        train_n=train_episodes.value,
        test_n=test_episodes.value,
        noisy_train=False,
        noisy_test=False,
        condition_level=hint_map[hint_label.value],
        context_mode="command",
        hidden_size=128,
        max_iter=40,
        negative_ratio=8,
        seed=7,
    )

    trained_model = trained_bundle["model"]
    trained_test_eps = trained_bundle["test_eps"]
    trained_metrics = trained_bundle["metrics"]
    copy_baseline = evaluate_model(None, trained_test_eps, baseline=True)
    smart_baseline = evaluate_model(None, trained_test_eps, heuristic=True)
    return (
        copy_baseline,
        smart_baseline,
        trained_bundle,
        trained_metrics,
        trained_model,
        trained_test_eps,
    )


@app.cell(hide_code=True)
def _(COLORS, copy_baseline, mo, smart_baseline, trained_metrics):
    def metric_card(title, value, subtitle, color):
        pct = int(value * 100)
        return f"""
        <div style="
            background: {color}22;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            min-width: 140px;
        ">
            <div style="font-size: 2.2em; font-weight: bold; color: {color};">{pct}%</div>
            <div style="color: #cdd6f4; font-weight: bold; margin-top: 4px;">{title}</div>
            <div style="color: #a6adc8; font-size: 0.85em; margin-top: 4px;">{subtitle}</div>
        </div>
        """

    results_html = f"""
    <div style="margin: 24px 0;">
        <h4 style="color: #cdd6f4; text-align: center; margin-bottom: 16px;">How well does each approach predict changed cells?</h4>
        <div style="display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;">
            {metric_card("Do Nothing", copy_baseline["changed_acc"], "Just copy the screen", COLORS["neutral"])}
            {metric_card("Know the Rules", smart_baseline["changed_acc"], "Typing rules, no learning", COLORS["typing"])}
            {metric_card("Learned Model", trained_metrics["changed_acc"], "Trained on examples", COLORS["correct"])}
        </div>
    </div>
    """
    mo.Html(results_html)
    return metric_card, results_html


@app.cell(hide_code=True)
def _(mo, smart_baseline, trained_metrics):
    improvement = trained_metrics["changed_acc"] - smart_baseline["changed_acc"]
    if improvement > 0.05:
        verdict = "The learned model beats the rule-based approach! It's picking up patterns from the data."
        kind = "success"
    elif improvement > -0.05:
        verdict = "The learned model is about as good as knowing the rules. It learned the basics."
        kind = "info"
    else:
        verdict = "The rule-based approach still wins. The model needs more training data or help."
        kind = "warn"

    mo.callout(mo.md(f"**What happened?** {verdict}"), kind=kind)
    return improvement, kind, verdict


@app.cell(hide_code=True)
def _(plt, trained_bundle):
    breakdown = trained_bundle["action_breakdown"]
    action_order = ["type_char", "enter", "backspace", "idle"]
    action_labels = ["typing", "Enter", "backspace", "idle"]
    action_values = [breakdown.get(name, {}).get("changed_acc", 0.0) for name in action_order]
    action_colors = ["#3b82f6", "#f97316", "#8b5cf6", "#6b7280"]

    action_breakdown_fig, action_breakdown_ax = plt.subplots(figsize=(7.8, 3.4))
    bars = action_breakdown_ax.bar(action_labels, action_values, color=action_colors, edgecolor="white", linewidth=2)
    action_breakdown_ax.set_ylim(0, 1)
    action_breakdown_ax.set_ylabel("Accuracy on changed cells")
    action_breakdown_ax.set_title("Which kind of step is easiest?", fontweight="bold")
    action_breakdown_ax.spines["top"].set_visible(False)
    action_breakdown_ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, action_values):
        action_breakdown_ax.text(bar.get_x() + bar.get_width() / 2, val + 0.04, f"{int(100*val)}%", ha="center", fontweight="bold")
    action_breakdown_fig.tight_layout()
    action_breakdown_fig
    return action_breakdown_fig, action_colors, action_labels, action_order, action_values, breakdown


@app.cell(hide_code=True)
def _(breakdown, mo):
    typing_score = breakdown.get("type_char", {}).get("changed_acc", 0.0)
    enter_score = breakdown.get("enter", {}).get("changed_acc", 0.0)
    gap = typing_score - enter_score
    mo.callout(
        mo.md(
            f"**Most important gap:** typing is at **{int(100*typing_score)}%**, while Enter is at **{int(100*enter_score)}%**. "
            f"That is a **{int(100*gap)} point gap** between surface actions and command meaning."
        ),
        kind="warn" if gap > 0.1 else "info",
    )
    return enter_score, gap, typing_score


# ============================================================================
# PETAL 4: SEE IT WORK
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🌻 Petal 4: Watch the Model Think

Let's see the model in action. Pick a test example and step through it.

The model sees the screen and guesses what comes next. We compare its guess to reality.
""")
    return


@app.cell(hide_code=True)
def _(mo, trained_test_eps):
    test_picker = mo.ui.slider(
        0, len(trained_test_eps) - 1,
        value=0,
        step=1,
        label="🎬 Pick a test example"
    )
    test_picker
    return (test_picker,)


@app.cell(hide_code=True)
def _(copy_baseline_rollout, heuristic_rollout, test_picker, trained_model, trained_test_eps):
    picked_episode = trained_test_eps[test_picker.value]
    learned_predictions = trained_model.rollout(picked_episode)
    copy_predictions = copy_baseline_rollout(picked_episode)
    rule_predictions = heuristic_rollout(picked_episode)
    picked_max_step = len(picked_episode.actions) - 1
    return copy_predictions, learned_predictions, picked_episode, picked_max_step, rule_predictions


@app.cell(hide_code=True)
def _(mo, picked_max_step):
    rollout_step = mo.ui.slider(
        0, picked_max_step,
        value=min(3, picked_max_step),
        step=1,
        label="⏩ Step through",
        full_width=True
    )
    rollout_step
    return (rollout_step,)


@app.cell(hide_code=True)
def _(
    COLORS,
    changed_cell_accuracy,
    mo,
    np,
    learned_predictions,
    picked_episode,
    rollout_step,
):
    rs = rollout_step.value
    truth_frame = picked_episode.frames[rs + 1]
    pred_frame = learned_predictions[rs + 1]
    prev_frame_r = picked_episode.frames[rs]
    action_r = picked_episode.actions[rs]

    # Calculate differences
    pred_matches = pred_frame == truth_frame
    changed_from_prev = truth_frame != prev_frame_r

    def render_comparison_terminal(frame, truth, prev, title, show_errors=True):
        rows = []
        for r in range(frame.shape[0]):
            chars = []
            for c in range(frame.shape[1]):
                ch = frame[r, c]
                if ch == " ":
                    ch = "&nbsp;"
                was_changed = prev[r, c] != truth[r, c]
                is_correct = frame[r, c] == truth[r, c]

                if show_errors and was_changed and not is_correct:
                    # Wrong prediction on a changed cell
                    chars.append(f'<span style="background:{COLORS["wrong"]}; color:white; font-weight:bold;">{ch}</span>')
                elif was_changed and is_correct:
                    # Correct prediction on a changed cell
                    chars.append(f'<span style="background:{COLORS["correct"]}; color:white; font-weight:bold;">{ch}</span>')
                elif was_changed:
                    chars.append(f'<span style="background:{COLORS["changed"]}; color:#000;">{ch}</span>')
                else:
                    chars.append(f'<span>{ch}</span>')
            rows.append("".join(chars))
        content = "<br>".join(rows)
        return f"""
        <div style="
            background: {COLORS['bg_dark']};
            border-radius: 12px;
            padding: 16px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.4;
            color: #cdd6f4;
            border: 2px solid {COLORS['bg_light']};
            min-width: 300px;
        ">
            <div style="color: #a6adc8; font-size: 11px; margin-bottom: 8px; font-weight: bold;">{title}</div>
            {content}
        </div>
        """

    truth_html = render_comparison_terminal(truth_frame, truth_frame, prev_frame_r, "✅ What Actually Happened", show_errors=False)
    pred_html = render_comparison_terminal(pred_frame, truth_frame, prev_frame_r, "🤖 What the Model Guessed")

    acc = changed_cell_accuracy(prev_frame_r, pred_frame, truth_frame)
    acc_pct = int(acc * 100)

    if action_r.kind == "enter":
        action_label = "⏎ Enter"
        action_bg = COLORS["enter"]
    elif action_r.kind == "type_char":
        action_label = f"⌨️ Type: {action_r.typed_char}"
        action_bg = COLORS["typing"]
    else:
        action_label = f"⏸️ {action_r.kind}"
        action_bg = COLORS["neutral"]

    header_html = f"""
    <div style="text-align: center; margin-bottom: 16px;">
        <span style="background: {action_bg}; color: white; padding: 6px 14px; border-radius: 16px; font-weight: bold;">
            {action_label}
        </span>
        <span style="margin-left: 16px; background: {'#22c55e' if acc > 0.7 else '#f59e0b' if acc > 0.4 else '#ef4444'}; color: white; padding: 6px 14px; border-radius: 16px;">
            {acc_pct}% correct on changed cells
        </span>
    </div>
    """

    legend_html = f"""
    <div style="display: flex; gap: 16px; justify-content: center; margin-top: 16px; font-size: 0.9em;">
        <span><span style="background:{COLORS['correct']}; color:white; padding: 2px 8px; border-radius: 4px;">■</span> Correct change</span>
        <span><span style="background:{COLORS['wrong']}; color:white; padding: 2px 8px; border-radius: 4px;">■</span> Wrong guess</span>
        <span><span style="background:{COLORS['changed']}; color:#000; padding: 2px 8px; border-radius: 4px;">■</span> Changed cell</span>
    </div>
    """

    mo.vstack([
        mo.Html(header_html),
        mo.hstack([mo.Html(truth_html), mo.Html(pred_html)], justify="center", gap=2),
        mo.Html(legend_html),
    ])
    return (
        acc,
        acc_pct,
        action_bg,
        action_label,
        action_r,
        changed_from_prev,
        header_html,
        legend_html,
        pred_frame,
        pred_html,
        pred_matches,
        prev_frame_r,
        render_comparison_terminal,
        rs,
        truth_frame,
        truth_html,
    )


@app.cell(hide_code=True)
def _(
    changed_cell_accuracy,
    copy_predictions,
    learned_predictions,
    mo,
    picked_episode,
    render_comparison_terminal,
    rollout_step,
    rule_predictions,
):
    showdown_step = rollout_step.value
    showdown_prev_frame = picked_episode.frames[showdown_step]
    showdown_truth_frame = picked_episode.frames[showdown_step + 1]

    baseline_map = {
        "🧍 Do Nothing": copy_predictions[showdown_step + 1],
        "📏 Know the Rules": rule_predictions[showdown_step + 1],
        "🧠 Learned Model": learned_predictions[showdown_step + 1],
    }

    showdown_tabs = {}
    for label, frame in baseline_map.items():
        baseline_score = changed_cell_accuracy(showdown_prev_frame, frame, showdown_truth_frame)
        badge_color = "#22c55e" if baseline_score > 0.7 else "#f59e0b" if baseline_score > 0.4 else "#ef4444"
        showdown_tabs[label] = mo.vstack([
            mo.Html(
                f"<div style='text-align:center; margin: 6px 0 14px 0;'><span class='nc-pill' style='background:{badge_color}; color:white'>{int(100*baseline_score)}% correct on changed cells</span></div>"
            ),
            mo.hstack(
                [
                    mo.Html(render_comparison_terminal(showdown_truth_frame, showdown_truth_frame, showdown_prev_frame, "✅ What actually happened", show_errors=False)),
                    mo.Html(render_comparison_terminal(frame, showdown_truth_frame, showdown_prev_frame, f"{label} guess")),
                ],
                justify="center",
                gap=2,
            ),
        ])

    baseline_showdown = mo.ui.tabs(showdown_tabs)
    mo.vstack([
        mo.md("""
### Mini arena: three ways to guess the next screen

Switch tabs to compare three different minds on the **same exact step**.
"""),
        baseline_showdown,
    ])
    return baseline_map, baseline_showdown, showdown_tabs


@app.cell(hide_code=True)
def _(COLORS, np, plt, pred_frame, prev_frame_r, truth_frame):
    # Error map for the current step
    error_grid = np.zeros(truth_frame.shape, dtype=np.int8)
    error_changed_mask = truth_frame != prev_frame_r
    error_correct_changed = error_changed_mask & (pred_frame == truth_frame)
    error_wrong_changed = error_changed_mask & (pred_frame != truth_frame)
    error_grid[error_correct_changed] = 1
    error_grid[error_wrong_changed] = -1

    error_map_fig, error_map_ax = plt.subplots(figsize=(8.2, 2.8))
    error_cmap = plt.matplotlib.colors.ListedColormap([COLORS["wrong"], "#1f2937", COLORS["correct"]])
    error_norm = plt.matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], error_cmap.N)
    error_map_ax.imshow(error_grid, cmap=error_cmap, norm=error_norm, aspect="auto")
    error_map_ax.set_title("Screen map: green = right change, red = wrong guess", fontweight="bold")
    error_map_ax.set_xticks([])
    error_map_ax.set_yticks([])
    for spine in error_map_ax.spines.values():
        spine.set_visible(False)
    error_map_fig.tight_layout()
    error_map_fig
    return error_changed_mask, error_cmap, error_correct_changed, error_grid, error_map_fig, error_norm, error_wrong_changed


@app.cell(hide_code=True)
def _(error_changed_mask, error_correct_changed, mo, error_wrong_changed):
    total_changed = int(error_changed_mask.sum())
    total_right = int(error_correct_changed.sum())
    total_wrong = int(error_wrong_changed.sum())
    mo.callout(
        mo.md(
            f"**Read the map:** this step changed **{total_changed}** cells. The model got **{total_right}** of them right and **{total_wrong}** wrong."
        ),
        kind="info",
    )
    return total_changed, total_right, total_wrong


@app.cell(hide_code=True)
def _(error_wrong_changed, mo, np, pred_frame, prev_frame_r, truth_frame):
    wrong_cells = np.argwhere(error_wrong_changed)
    if len(wrong_cells) == 0:
        patch_html = ""
        focus = None
        panel = mo.callout(mo.md("**Zoomed-in view:** on this step, the model did not miss any changed cell."), kind="success")
    else:
        r, c = wrong_cells[0].tolist()
        focus = (r, c)

        def patch(frame, rr, cc, radius=1):
            pieces = []
            for i in range(rr - radius, rr + radius + 1):
                row = []
                for j in range(cc - radius, cc + radius + 1):
                    if 0 <= i < frame.shape[0] and 0 <= j < frame.shape[1]:
                        ch = frame[i, j]
                    else:
                        ch = "·"
                    if ch == " ":
                        ch = "␠"
                    row.append(ch)
                pieces.append(row)
            return pieces

        before = patch(prev_frame_r, r, c)
        patch_truth = patch(truth_frame, r, c)
        patch_guess = patch(pred_frame, r, c)

        def grid_html(title, grid):
            rows = "".join(
                "<tr>" + "".join(f"<td style='padding:8px 10px; border:1px solid #4b5563; text-align:center'>{cell}</td>" for cell in row) + "</tr>"
                for row in grid
            )
            return f"<div><div style='font-weight:700; margin-bottom:6px'>{title}</div><table style='border-collapse:collapse'>{rows}</table></div>"

        patch_html = "<div style='display:flex; gap:18px; justify-content:center; margin-top:10px'>" + grid_html("Before", before) + grid_html("What should happen", patch_truth) + grid_html("What the model guessed", patch_guess) + "</div>"
        panel = mo.vstack([
            mo.md(f"### Zoom in on one wrong spot\nThe first wrong cell on this step is at row `{r}`, column `{c}`."),
            mo.Html(patch_html),
        ])
    panel
    return focus, panel, patch_html, wrong_cells


# ============================================================================
# PETAL 5: THE EXPERIMENTS
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🌷 Petal 5: Digging Deeper

Now let's run some real experiments to understand what the model learns.

Each experiment answers a different question.
""")
    return


@app.cell(hide_code=True)
def _(
    conditioning_study_multiseed,
    mo,
    noise_study_multiseed,
    np,
    paraphrase_generalization_multiseed,
    plt,
):
    # Pre-run all studies
    cond_data = conditioning_study_multiseed(
        train_n=18, test_n=12, hidden_size=96,
        max_iter=40, negative_ratio=8, seeds=(11, 17, 23)
    )

    noise_data = noise_study_multiseed(
        train_n=18, test_n=12, hidden_size=96,
        max_iter=40, negative_ratio=8, seeds=(13, 19, 29)
    )

    para_data = paraphrase_generalization_multiseed(
        train_n=15, test_n=15, hidden_size=96,
        max_iter=40, negative_ratio=8, seeds=(31, 37, 41)
    )

    # === Experiment 1: Hints ===
    def make_hints_chart():
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#6b7280", "#3b82f6", "#22c55e"]
        bars = ax.bar(
            cond_data["conditioning"],
            cond_data["changed_acc_mean"],
            yerr=cond_data["changed_acc_std"],
            color=colors,
            capsize=6,
            edgecolor="white",
            linewidth=2
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy on changed cells", fontsize=12)
        ax.set_title("Does giving hints help?", fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, cond_data["changed_acc_mean"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f"{int(val*100)}%", ha="center", fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig

    # === Experiment 2: Clean vs Messy ===
    def make_noise_chart():
        fig, ax = plt.subplots(figsize=(9, 4))
        order = ["clean→clean", "noisy→clean", "clean→noisy", "noisy→noisy"]
        colors = ["#22c55e", "#f59e0b", "#3b82f6", "#ef4444"]
        ordered = noise_data.set_index("setting").loc[order].reset_index()
        bars = ax.bar(
            ordered["setting"],
            ordered["changed_acc_mean"],
            yerr=ordered["changed_acc_std"],
            color=colors,
            capsize=6,
            edgecolor="white",
            linewidth=2
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy on changed cells", fontsize=12)
        ax.set_title("Does training on clean data help?", fontsize=14, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val in zip(bars, ordered["changed_acc_mean"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f"{int(val*100)}%", ha="center", fontsize=11, fontweight="bold")
        fig.tight_layout()
        return fig

    # === Experiment 3: New Phrasings ===
    def make_paraphrase_chart():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(para_data))
        width = 0.35
        bars1 = ax.bar(
            x - width/2,
            para_data["typing_changed_acc_mean"],
            width,
            yerr=para_data["typing_changed_acc_std"],
            label="Typing steps",
            color="#3b82f6",
            capsize=4,
            edgecolor="white",
            linewidth=2
        )
        bars2 = ax.bar(
            x + width/2,
            para_data["enter_changed_acc_mean"],
            width,
            yerr=para_data["enter_changed_acc_std"],
            label="Enter steps",
            color="#f97316",
            capsize=4,
            edgecolor="white",
            linewidth=2
        )
        ax.set_xticks(x)
        ax.set_xticklabels(para_data["conditioning"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy on changed cells", fontsize=12)
        ax.set_title("Can it handle commands it never saw?", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    hints_chart = make_hints_chart()
    noise_chart = make_noise_chart()
    para_chart = make_paraphrase_chart()

    # Build tabs content
    exp1_content = mo.vstack([
        mo.md("""
**Question:** If we tell the model what command is running, does it do better?

**Setup:** Train with different levels of hints:
- **None:** Just the screen
- **Family:** "This is a `whoami`-type command"
- **Command:** "The exact command is `whoami`"
        """),
        hints_chart,
        mo.callout(mo.md("**Finding:** More hints = better accuracy. But even without hints, it learns something!"), kind="success"),
    ])

    exp2_content = mo.vstack([
        mo.md("""
**Question:** Is it better to train on clean, perfect examples or messy, varied ones?

**Setup:** We test all combinations:
- Train clean, test clean
- Train messy, test clean
- etc.
        """),
        noise_chart,
        mo.callout(mo.md("**Surprise:** Training on messy data can actually help when testing on clean data! Variety teaches robustness."), kind="warn"),
    ])

    exp3_content = mo.vstack([
        mo.md("""
**Question:** If the model only saw `whoami`, can it handle `id -un` (same meaning, different words)?

**Setup:** Train on one phrasing, test on different phrasings of the same commands.
        """),
        para_chart,
        mo.callout(mo.md("""
**Key insight:** 
- ⌨️ **Typing** transfers well — the model learned how typing works in general
- ⏎ **Enter** transfers poorly — it memorized specific outputs, not the meaning
        """), kind="info"),
    ])

    experiments_tabs = mo.ui.tabs({
        "🎯 Do Hints Help?": exp1_content,
        "🧹 Clean vs Messy Data": exp2_content,
        "🔄 New Phrasings": exp3_content,
    })

    experiments_tabs
    return (
        cond_data,
        exp1_content,
        exp2_content,
        exp3_content,
        experiments_tabs,
        hints_chart,
        make_hints_chart,
        make_noise_chart,
        make_paraphrase_chart,
        noise_chart,
        noise_data,
        para_chart,
        para_data,
    )


@app.cell(hide_code=True)
def _(COLORS, cond_data, mo, noise_data, para_data):
    best_hint = cond_data.sort_values("changed_acc_mean", ascending=False).iloc[0]
    best_noise = noise_data.sort_values("changed_acc_mean", ascending=False).iloc[0]
    best_para = para_data.sort_values("enter_changed_acc_mean", ascending=False).iloc[0]
    summary = f"""
    <div style='display:grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 14px;'>
      <div class='nc-card' style='background:{COLORS['correct']}11; border-color:{COLORS['correct']};'>
        <div style='font-size:1.7em'>💡</div>
        <b>Best hint setting</b>
        <div class='nc-small' style='margin-top:6px'>{best_hint['conditioning']} came out on top at <b>{int(100*best_hint['changed_acc_mean'])}%</b>.</div>
      </div>
      <div class='nc-card' style='background:{COLORS['changed']}11; border-color:{COLORS['changed']};'>
        <div style='font-size:1.7em'>🧹</div>
        <b>Best train→test mix</b>
        <div class='nc-small' style='margin-top:6px'>{best_noise['setting']} gave <b>{int(100*best_noise['changed_acc_mean'])}%</b>.</div>
      </div>
      <div class='nc-card' style='background:{COLORS['enter']}11; border-color:{COLORS['enter']};'>
        <div style='font-size:1.7em'>🔄</div>
        <b>Best on new phrasing</b>
        <div class='nc-small' style='margin-top:6px'>{best_para['conditioning']} did best on Enter steps at <b>{int(100*best_para['enter_changed_acc_mean'])}%</b>.</div>
      </div>
    </div>
    """
    mo.Html(summary)
    return best_hint, best_noise, best_para, summary


# ============================================================================
# PETAL 6: FAILURE GALLERY
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🥀 Petal 6: Where It Fails

A good scientist shows failures, not just successes. Here's where the model struggles.
""")
    return


@app.cell(hide_code=True)
def _(changed_cell_accuracy, html_screen, mo, trained_model, trained_test_eps):
    examples = {
        "easy_typing": None,
        "hard_enter": None,
        "hard_episode": None,
    }

    worst_episode_score = 1.0
    worst_episode = None
    worst_episode_predictions = None

    for gallery_episode in trained_test_eps:
        gallery_predictions = trained_model.rollout(gallery_episode)
        gallery_step_scores = []
        for gallery_step, gallery_action in enumerate(gallery_episode.actions):
            gallery_prev = gallery_episode.frames[gallery_step]
            gallery_truth = gallery_episode.frames[gallery_step + 1]
            gallery_pred = gallery_predictions[gallery_step + 1]
            gallery_score = changed_cell_accuracy(gallery_prev, gallery_pred, gallery_truth)
            gallery_step_scores.append(gallery_score)

            if gallery_action.kind == "type_char":
                current = examples["easy_typing"]
                if current is None or gallery_score > current["score"]:
                    examples["easy_typing"] = {
                        "title": "⌨️ Best typing moment",
                        "subtitle": f"Command: `{gallery_episode.command_text}`",
                        "score": gallery_score,
                        "truth_screen": gallery_truth,
                        "pred_screen": gallery_pred,
                    }
            if gallery_action.kind == "enter":
                current = examples["hard_enter"]
                if current is None or gallery_score < current["score"]:
                    examples["hard_enter"] = {
                        "title": "⏎ Toughest Enter moment",
                        "subtitle": f"Command: `{gallery_episode.command_text}`",
                        "score": gallery_score,
                        "truth_screen": gallery_truth,
                        "pred_screen": gallery_pred,
                    }

        gallery_episode_score = sum(gallery_step_scores) / max(len(gallery_step_scores), 1)
        if gallery_episode_score < worst_episode_score:
            worst_episode_score = gallery_episode_score
            worst_episode = gallery_episode
            worst_episode_predictions = gallery_predictions

    if worst_episode is not None:
        examples["hard_episode"] = {
            "title": "🌪️ Hardest full example",
            "subtitle": f"Command: `{worst_episode.command_text}`",
            "score": worst_episode_score,
            "truth_screen": worst_episode.frames[-1],
            "pred_screen": worst_episode_predictions[-1],
        }

    failure_tabs = {}
    for key in ["easy_typing", "hard_enter", "hard_episode"]:
        example = examples[key]
        badge = "#22c55e" if example["score"] > 0.7 else "#f59e0b" if example["score"] > 0.4 else "#ef4444"
        failure_tabs[example["title"]] = mo.vstack([
            mo.md(f"{example['subtitle']}  \\\n**Score on changed cells:** `{int(100*example['score'])}%`"),
            mo.hstack([
                mo.Html(html_screen(example["truth_screen"], diff_to=example["pred_screen"])),
                mo.Html(html_screen(example["pred_screen"], diff_to=example["truth_screen"])),
            ], justify="center", gap=2),
            mo.Html(f"<div style='text-align:center; margin-top:8px;'><span class='nc-pill' style='background:{badge}; color:white'>{int(100*example['score'])}%</span></div>"),
        ])

    failure_gallery = mo.ui.tabs(failure_tabs)
    failure_gallery
    return examples, failure_gallery, failure_tabs, worst_episode, worst_episode_predictions, worst_episode_score


@app.cell(hide_code=True)
def _(COLORS, mo):
    failures_html = f"""
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 16px 0;">
        <div style="background: {COLORS['wrong']}22; border: 2px solid {COLORS['wrong']}; border-radius: 12px; padding: 16px;">
            <h4 style="color: {COLORS['wrong']}; margin-bottom: 8px;">❌ After Enter</h4>
            <p style="color: #cdd6f4;">When a command runs, the model often guesses wrong output. It knows <em>something</em> should appear, but not <em>what</em>.</p>
        </div>
        <div style="background: {COLORS['enter']}22; border: 2px solid {COLORS['enter']}; border-radius: 12px; padding: 16px;">
            <h4 style="color: {COLORS['enter']}; margin-bottom: 8px;">🔄 New Commands</h4>
            <p style="color: #cdd6f4;">If it never saw <code>id -un</code>, it can't guess the output even though it means the same as <code>whoami</code>.</p>
        </div>
        <div style="background: {COLORS['neutral']}22; border: 2px solid {COLORS['neutral']}; border-radius: 12px; padding: 16px;">
            <h4 style="color: #a6adc8; margin-bottom: 8px;">🔢 Math Problems</h4>
            <p style="color: #cdd6f4;">For <code>python -c "print(3+4)"</code>, it knows a number appears, but often gets the wrong number.</p>
        </div>
        <div style="background: {COLORS['typing']}22; border: 2px solid {COLORS['typing']}; border-radius: 12px; padding: 16px;">
            <h4 style="color: {COLORS['typing']}; margin-bottom: 8px;">📏 Long Commands</h4>
            <p style="color: #cdd6f4;">The longer the command, the more chances to make small mistakes that accumulate.</p>
        </div>
    </div>
    """
    mo.Html(failures_html)
    return (failures_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""
**Why this matters:** The model learns to *imitate* a computer before it *understands* one.

It's like a student who can copy the teacher's handwriting but doesn't understand the words.
"""),
        kind="warn"
    )
    return


# ============================================================================
# PETAL 7: THE BIG PICTURE
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🌳 Petal 7: Zooming Back Out

What did we learn? And what would make this better?
""")
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    summary_html = f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['bg_dark']} 0%, {COLORS['bg_light']} 100%);
        border-radius: 16px;
        padding: 32px;
        margin: 16px 0;
    ">
        <h3 style="color: #cdd6f4; text-align: center; margin-bottom: 24px;">What We Learned</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
            <div style="text-align: center;">
                <div style="font-size: 2.5em; margin-bottom: 8px;">✅</div>
                <div style="color: {COLORS['correct']}; font-weight: bold;">Works</div>
                <div style="color: #a6adc8; font-size: 0.9em;">Learning typing mechanics from screen recordings</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2.5em; margin-bottom: 8px;">⚡</div>
                <div style="color: {COLORS['changed']}; font-weight: bold;">Helps</div>
                <div style="color: #a6adc8; font-size: 0.9em;">Giving the model hints about what's happening</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2.5em; margin-bottom: 8px;">🔥</div>
                <div style="color: {COLORS['wrong']}; font-weight: bold;">Hard</div>
                <div style="color: #a6adc8; font-size: 0.9em;">Understanding what commands actually do</div>
            </div>
        </div>
    </div>
    """
    mo.Html(summary_html)
    return (summary_html,)


@app.cell(hide_code=True)
def _(mo):
    next_steps = mo.accordion({
        "🚀 What would make this better?": mo.md("""
**Memory:** Our current model has no memory. A model that remembers what happened earlier might understand commands better.

**Bigger brain:** A Transformer or similar model could look at the whole screen at once, not just small patches.

**More data:** Real computer recordings, not just our toy terminal.
        """),
        "📚 Connection to the real paper": mo.md("""
The [Neural Computers paper](https://arxiv.org/abs/2604.06425) does all of this at scale:
- Real video recordings of computers
- Large video generation models
- Much more training data

This notebook is a tiny toy version to build intuition about *why* that approach might work — and where it might struggle.
        """),
        "🤔 The deep question": mo.md("""
**Can you learn how something works just by watching it?**

Humans do this all the time. We learn to use apps by watching others, not by reading code.

This notebook suggests: yes, but you learn the *surface behavior* before the *deep logic*.

That's both exciting and limiting.
        """),
    }, multiple=True)
    next_steps
    return (next_steps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
---

<div style="text-align: center; color: #6b7280; padding: 24px;">
    <p>Built with <a href="https://marimo.io" style="color: #89b4fa;">marimo</a> for the notebook competition</p>
    <p>Inspired by <a href="https://arxiv.org/abs/2604.06425" style="color: #89b4fa;">Neural Computers</a> (Zhuge et al., 2026)</p>
</div>
""")
    return


if __name__ == "__main__":
    app.run()
