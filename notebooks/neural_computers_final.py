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
    import json
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
        "mlp": "#22c55e",
        "transformer": "#a855f7",
        "gru": "#ef4444",
        "accent": "#89b4fa",
    }

    plt.rcParams["figure.facecolor"] = "#1e1e2e"
    plt.rcParams["axes.facecolor"] = "#1e1e2e"
    plt.rcParams["axes.edgecolor"] = "#6b7280"
    plt.rcParams["axes.labelcolor"] = "#cdd6f4"
    plt.rcParams["text.color"] = "#cdd6f4"
    plt.rcParams["xtick.color"] = "#a6adc8"
    plt.rcParams["ytick.color"] = "#a6adc8"
    plt.rcParams["grid.color"] = "#45475a"
    plt.rcParams["legend.facecolor"] = "#313244"
    plt.rcParams["legend.edgecolor"] = "#6b7280"

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
        json,
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
        :root {
            --nc-bg: #1e1e2e;
            --nc-surface: #313244;
            --nc-text: #cdd6f4;
            --nc-muted: #a6adc8;
            --nc-accent: #89b4fa;
            --nc-green: #22c55e;
            --nc-red: #ef4444;
            --nc-orange: #f97316;
            --nc-yellow: #f59e0b;
            --nc-purple: #a855f7;
            --nc-blue: #3b82f6;
        }
        .nc-card {
            background: linear-gradient(145deg, var(--nc-surface), var(--nc-bg));
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.06);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .nc-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .nc-small {
            color: var(--nc-muted);
            font-size: 0.92em;
            line-height: 1.6;
        }
        .nc-pill {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9em;
            letter-spacing: 0.02em;
        }
        .nc-glow {
            box-shadow: 0 0 30px rgba(137, 180, 250, 0.15);
        }
        .nc-terminal {
            background: linear-gradient(180deg, #0d0d0d 0%, #1a1a2e 100%);
            border-radius: 12px;
            padding: 16px;
            font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            color: #e0e0e0;
            border: 1px solid #333;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
        }
        .nc-metric {
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(0,0,0,0.1));
        }
        .nc-metric-value {
            font-size: 2.8em;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 8px;
        }
        .nc-metric-label {
            font-size: 0.95em;
            font-weight: 600;
            color: var(--nc-text);
        }
        .nc-metric-sub {
            font-size: 0.82em;
            color: var(--nc-muted);
            margin-top: 4px;
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 20px rgba(137, 180, 250, 0.2); }
            50% { box-shadow: 0 0 40px rgba(137, 180, 250, 0.4); }
        }
        .nc-hero-glow {
            animation: pulse-glow 3s ease-in-out infinite;
        }
        </style>
        """
    )
    return


# ============================================================================
# HERO SECTION
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    hero_html = """
    <div class="nc-hero-glow" style="
        background: linear-gradient(135deg, #1e1e2e 0%, #2d1b4e 50%, #1e1e2e 100%);
        border-radius: 24px;
        padding: 56px 48px;
        text-align: center;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 30% 30%, rgba(168, 85, 247, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 70% 70%, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
            pointer-events: none;
        "></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 4em; margin-bottom: 16px;">🖥️</div>
            <h1 style="
                font-size: 2.6em;
                color: #cdd6f4;
                margin-bottom: 16px;
                font-weight: 800;
                letter-spacing: -0.02em;
            ">Can a Model Learn How a Computer Works?</h1>
            <p style="
                font-size: 1.35em;
                color: #a6adc8;
                max-width: 650px;
                margin: 0 auto 28px auto;
                line-height: 1.7;
            ">
                Not by reading code. Not by being taught.<br>
                <strong style="color: #f9e2af;">Just by watching the screen change.</strong>
            </p>
            <div style="
                display: inline-flex;
                gap: 12px;
                align-items: center;
                background: rgba(69, 71, 90, 0.6);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 14px 24px;
                border: 1px solid rgba(255,255,255,0.1);
            ">
                <span style="color: #89b4fa; font-size: 1.1em;">📄</span>
                <span style="color: #cdd6f4;">Inspired by</span>
                <a href="https://arxiv.org/abs/2604.06425" style="color: #89b4fa; font-weight: 600; text-decoration: none;">Neural Computers</a>
                <span style="color: #6b7280;">(2026)</span>
            </div>
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

This notebook lets you explore that question with a tiny toy computer — and see what different model architectures actually learn.
"""),
        kind="info"
    )
    return


# ============================================================================
# JOURNEY MAP
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    journey_html = """
    <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin: 24px 0 32px 0;">
        <div class='nc-card' style='border-left: 3px solid #3b82f6;'>
            <div style='font-size: 1.8em; margin-bottom: 8px;'>🌱</div>
            <div style='font-weight: 700; color: #cdd6f4; margin-bottom: 4px;'>Watch</div>
            <div class='nc-small'>See the toy computer step by step</div>
        </div>
        <div class='nc-card' style='border-left: 3px solid #f59e0b;'>
            <div style='font-size: 1.8em; margin-bottom: 8px;'>🌸</div>
            <div style='font-weight: 700; color: #cdd6f4; margin-bottom: 4px;'>Split</div>
            <div class='nc-small'>Separate easy typing from hard meaning</div>
        </div>
        <div class='nc-card' style='border-left: 3px solid #22c55e;'>
            <div style='font-size: 1.8em; margin-bottom: 8px;'>🌺</div>
            <div style='font-weight: 700; color: #cdd6f4; margin-bottom: 4px;'>Train</div>
            <div class='nc-small'>Give models examples and hints</div>
        </div>
        <div class='nc-card' style='border-left: 3px solid #a855f7;'>
            <div style='font-size: 1.8em; margin-bottom: 8px;'>⚔️</div>
            <div style='font-weight: 700; color: #cdd6f4; margin-bottom: 4px;'>Battle</div>
            <div class='nc-small'>MLP vs Transformer showdown</div>
        </div>
        <div class='nc-card' style='border-left: 3px solid #f97316;'>
            <div style='font-size: 1.8em; margin-bottom: 8px;'>🔬</div>
            <div style='font-weight: 700; color: #cdd6f4; margin-bottom: 4px;'>Experiment</div>
            <div class='nc-small'>Test hints, noise, new commands</div>
        </div>
        <div class='nc-card' style='border-left: 3px solid #ef4444;'>
            <div style='font-size: 1.8em; margin-bottom: 8px;'>🎯</div>
            <div style='font-weight: 700; color: #cdd6f4; margin-bottom: 4px;'>Discover</div>
            <div class='nc-small'>Find the surprising pattern</div>
        </div>
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
    demo_step = mo.ui.slider(0, demo_max_step, value=0, step=1, label="⏩ Step through time", full_width=True)
    demo_step
    return (demo_step,)


@app.cell(hide_code=True)
def _(COLORS, demo_episode, demo_max_step, demo_step, mo, np):
    demo_step_value = demo_step.value
    current_frame = demo_episode.frames[demo_step_value]
    if demo_step_value > 0:
        demo_prev_frame = demo_episode.frames[demo_step_value - 1]
        changed_mask = current_frame != demo_prev_frame
    else:
        demo_prev_frame = current_frame.copy()
        changed_mask = np.zeros_like(current_frame, dtype=bool)

    def render_terminal_fancy(frame, changed_mask, title=""):
        rows = []
        for r in range(frame.shape[0]):
            chars = []
            for c in range(frame.shape[1]):
                ch = frame[r, c]
                if ch == " ":
                    ch = "&nbsp;"
                if changed_mask[r, c]:
                    chars.append(f'<span style="background:{COLORS["changed"]}; color:#000; font-weight:bold; border-radius: 2px;">{ch}</span>')
                else:
                    chars.append(f'<span style="color: #b0b0b0;">{ch}</span>')
            rows.append("".join(chars))
        content = "<br>".join(rows)
        return f"""
        <div class="nc-terminal" style="min-width: 420px;">
            <div style="color: #6b7280; font-size: 11px; margin-bottom: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">{title}</div>
            <div style="background: #0a0a0f; padding: 12px; border-radius: 8px; border: 1px solid #222;">{content}</div>
        </div>
        """

    if demo_step_value < len(demo_episode.actions):
        demo_action = demo_episode.actions[demo_step_value]
        if demo_action.kind == "type_char":
            action_desc = f"⌨️ Type: <code style='background:#3b82f6; padding: 2px 8px; border-radius: 4px;'>{demo_action.typed_char}</code>"
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

    terminal_html = render_terminal_fancy(current_frame, changed_mask, f"Command: {demo_episode.command_text}")
    changed_count = int(changed_mask.sum())

    info_html = f"""
    <div style="display: flex; gap: 16px; margin-top: 16px; justify-content: center; flex-wrap: wrap;">
        <div style="background: {action_color}; color: white; padding: 10px 20px; border-radius: 24px; font-weight: bold; font-size: 1.05em; box-shadow: 0 4px 12px {action_color}44;">
            {action_desc}
        </div>
        <div style="background: {COLORS['bg_light']}; padding: 10px 20px; border-radius: 24px; color: #cdd6f4; border: 1px solid #45475a;">
            📍 Step <strong>{demo_step_value}</strong> / {demo_max_step}
        </div>
        <div style="background: {COLORS['changed']}22; padding: 10px 20px; border-radius: 24px; color: {COLORS['changed']}; border: 1px solid {COLORS['changed']}44;">
            ✨ <strong>{changed_count}</strong> cells changed
        </div>
    </div>
    """

    mo.vstack([mo.Html(terminal_html), mo.Html(info_html)], align="center")
    return (
        action_color,
        action_desc,
        changed_count,
        changed_mask,
        current_frame,
        demo_action,
        demo_prev_frame,
        demo_step_value,
        info_html,
        render_terminal_fancy,
        terminal_html,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md("""
**What you're seeing:** Each step, the user either types a character, presses Enter, or does nothing.
The <span style="background: #f59e0b; color: #000; padding: 2px 6px; border-radius: 4px; font-weight: bold;">highlighted cells</span> show what changed.

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
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 28px; margin: 28px 0;">
        <div class="nc-card" style="
            background: linear-gradient(145deg, {COLORS['typing']}15 0%, {COLORS['typing']}05 100%);
            border: 2px solid {COLORS['typing']}66;
            text-align: center;
            padding: 32px;
        ">
            <div style="font-size: 4em; margin-bottom: 16px;">⌨️</div>
            <h3 style="color: {COLORS['typing']}; margin-bottom: 12px; font-size: 1.4em;">Mechanics</h3>
            <p style="color: #cdd6f4; font-size: 1.15em; margin-bottom: 20px;">
                "The user typed <code style='background:#222; padding: 4px 8px; border-radius: 4px;'>w</code>, so show <code style='background:#222; padding: 4px 8px; border-radius: 4px;'>w</code>"
            </p>
            <div style="
                background: {COLORS['correct']};
                color: white;
                padding: 10px 24px;
                border-radius: 24px;
                display: inline-block;
                font-weight: bold;
                box-shadow: 0 4px 12px {COLORS['correct']}44;
            ">✅ Easy to learn</div>
        </div>
        <div class="nc-card" style="
            background: linear-gradient(145deg, {COLORS['enter']}15 0%, {COLORS['enter']}05 100%);
            border: 2px solid {COLORS['enter']}66;
            text-align: center;
            padding: 32px;
        ">
            <div style="font-size: 4em; margin-bottom: 16px;">⏎</div>
            <h3 style="color: {COLORS['enter']}; margin-bottom: 12px; font-size: 1.4em;">Meaning</h3>
            <p style="color: #cdd6f4; font-size: 1.15em; margin-bottom: 20px;">
                "The user ran <code style='background:#222; padding: 4px 8px; border-radius: 4px;'>whoami</code>, so show the username"
            </p>
            <div style="
                background: {COLORS['wrong']};
                color: white;
                padding: 10px 24px;
                border-radius: 24px;
                display: inline-block;
                font-weight: bold;
                box-shadow: 0 4px 12px {COLORS['wrong']}44;
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

**Spoiler:** It learns the looks first. The meaning is harder. But different architectures have different strengths...
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
    train_episodes = mo.ui.slider(6, 48, value=12, step=6, label="📚 Training examples")
    test_episodes = mo.ui.slider(4, 24, value=6, step=2, label="🧪 Test examples")
    hint_label = mo.ui.dropdown(
        options=["No hints", "Tell it the command type", "Tell it the exact command"],
        value="Tell it the command type",
        label="💡 How much help?"
    )
    mo.hstack([train_episodes, test_episodes, hint_label], justify="center", gap=2)
    return hint_label, test_episodes, train_episodes


@app.cell(hide_code=True)
def _(evaluate_model, fit_bundle, hint_label, test_episodes, train_episodes):
    hint_map = {"No hints": "none", "Tell it the command type": "family", "Tell it the exact command": "command"}
    trained_bundle = fit_bundle(
        train_n=train_episodes.value,
        test_n=test_episodes.value,
        noisy_train=False,
        noisy_test=False,
        condition_level=hint_map[hint_label.value],
        context_mode="command",
        hidden_size=96,
        max_iter=25,
        negative_ratio=6,
        seed=7,
    )
    trained_model = trained_bundle["model"]
    trained_test_eps = trained_bundle["test_eps"]
    trained_metrics = trained_bundle["metrics"]
    copy_baseline = evaluate_model(None, trained_test_eps, baseline=True)
    smart_baseline = evaluate_model(None, trained_test_eps, heuristic=True)
    return copy_baseline, smart_baseline, trained_bundle, trained_metrics, trained_model, trained_test_eps


@app.cell(hide_code=True)
def _(COLORS, copy_baseline, mo, smart_baseline, trained_metrics):
    def metric_card(title, value, subtitle, color, icon):
        pct = int(value * 100)
        return f"""
        <div class="nc-metric" style="border: 2px solid {color}44; min-width: 160px;">
            <div style="font-size: 1.6em; margin-bottom: 8px;">{icon}</div>
            <div class="nc-metric-value" style="color: {color};">{pct}%</div>
            <div class="nc-metric-label">{title}</div>
            <div class="nc-metric-sub">{subtitle}</div>
        </div>
        """

    results_html = f"""
    <div style="margin: 28px 0;">
        <h4 style="color: #cdd6f4; text-align: center; margin-bottom: 20px; font-size: 1.2em;">How well does each approach predict changed cells?</h4>
        <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
            {metric_card("Do Nothing", copy_baseline["changed_acc"], "Just copy the screen", COLORS["neutral"], "🧍")}
            {metric_card("Know Rules", smart_baseline["changed_acc"], "Typing rules only", COLORS["typing"], "📏")}
            {metric_card("Learned", trained_metrics["changed_acc"], "Trained on data", COLORS["correct"], "🧠")}
        </div>
    </div>
    """
    mo.Html(results_html)
    return metric_card, results_html


@app.cell(hide_code=True)
def _(COLORS, plt, trained_bundle):
    breakdown = trained_bundle["action_breakdown"]
    action_order = ["type_char", "enter", "backspace", "idle"]
    action_labels = ["Typing", "Enter", "Backspace", "Idle"]
    action_values = [breakdown.get(name, {}).get("changed_acc", 0.0) for name in action_order]
    action_colors_chart = [COLORS["typing"], COLORS["enter"], "#8b5cf6", COLORS["neutral"]]

    action_breakdown_fig, action_breakdown_ax = plt.subplots(figsize=(9, 4))
    action_bars = action_breakdown_ax.bar(action_labels, action_values, color=action_colors_chart, edgecolor="#1e1e2e", linewidth=3, width=0.65)
    action_breakdown_ax.set_ylim(0, 1.15)
    action_breakdown_ax.set_ylabel("Accuracy on changed cells", fontsize=12, fontweight="bold")
    action_breakdown_ax.set_title("Which kind of step is easiest?", fontsize=15, fontweight="bold", pad=15)
    action_breakdown_ax.spines["top"].set_visible(False)
    action_breakdown_ax.spines["right"].set_visible(False)
    action_breakdown_ax.axhline(y=0.5, color="#45475a", linestyle="--", alpha=0.7)
    for action_bar, action_val, action_col in zip(action_bars, action_values, action_colors_chart):
        action_breakdown_ax.text(action_bar.get_x() + action_bar.get_width() / 2, action_val + 0.05, f"{int(100*action_val)}%", ha="center", fontweight="bold", fontsize=13, color=action_col)
    action_breakdown_fig.tight_layout()
    action_breakdown_fig
    return action_breakdown_fig, action_colors_chart, action_labels, action_order, action_values, breakdown


@app.cell(hide_code=True)
def _(breakdown, mo):
    typing_score = breakdown.get("type_char", {}).get("changed_acc", 0.0)
    enter_score = breakdown.get("enter", {}).get("changed_acc", 0.0)
    gap = typing_score - enter_score
    mo.callout(
        mo.md(f"**The gap:** Typing is at **{int(100*typing_score)}%**, Enter is at **{int(100*enter_score)}%**. "
              f"That's a **{int(100*gap)} point gap** between surface actions and command meaning."),
        kind="warn" if gap > 0.1 else "info",
    )
    return enter_score, gap, typing_score


# ============================================================================
# PETAL 4: MODEL SHOWDOWN - THE KEY COMPARISON
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## ⚔️ Petal 4: The Architecture Showdown

Here's where it gets interesting. We trained **three different architectures** on the same task:

- **MLP** — A simple feedforward network that looks at each position independently
- **Transformer** — Sees the whole sequence and attends to everything
- **GRU** — A recurrent network that maintains hidden state

Which one wins? The answer surprised us.
""")
    return


@app.cell(hide_code=True)
def _(COLORS, ROOT, json, mo, np, pd, plt):
    # Load the pre-computed results from GPU experiments
    results_path = ROOT / "experiments" / "toy_nc_cli" / "results"
    
    try:
        mlp_data = json.loads((results_path / "mlp_matched_results.json").read_text())
        transformer_data = json.loads((results_path / "transformer_results.json").read_text())
        gru_data = json.loads((results_path / "gru_results.json").read_text())
        has_comparison_data = True
    except FileNotFoundError:
        has_comparison_data = False
        mlp_data = transformer_data = gru_data = {}

    if has_comparison_data:
        # Build comparison dataframe
        comparison_rows = []
        for setting in ["standard_family", "standard_command", "paraphrase_family", "paraphrase_command"]:
            comparison_rows.append({
                "setting": setting,
                "MLP": mlp_data[setting]["metrics"]["changed_acc"],
                "Transformer": transformer_data[setting]["metrics"]["changed_acc"],
                "GRU": gru_data[setting]["metrics"]["changed_acc"],
                "MLP_enter": mlp_data[setting]["action_breakdown"].get("enter", {}).get("changed_acc", 0),
                "Transformer_enter": transformer_data[setting]["action_breakdown"].get("enter", {}).get("changed_acc", 0),
                "GRU_enter": gru_data[setting]["action_breakdown"].get("enter", {}).get("changed_acc", 0),
            })
        comparison_df = pd.DataFrame(comparison_rows)

        # Create the main comparison chart
        showdown_fig, showdown_axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Overall accuracy
        ax1 = showdown_axes[0]
        x = np.arange(4)
        width = 0.25
        settings_labels = ["Standard\nFamily", "Standard\nCommand", "Paraphrase\nFamily", "Paraphrase\nCommand"]
        
        sd_bars1 = ax1.bar(x - width, comparison_df["MLP"], width, label="MLP", color=COLORS["mlp"], edgecolor="#1e1e2e", linewidth=2)
        sd_bars2 = ax1.bar(x, comparison_df["Transformer"], width, label="Transformer", color=COLORS["transformer"], edgecolor="#1e1e2e", linewidth=2)
        sd_bars3 = ax1.bar(x + width, comparison_df["GRU"], width, label="GRU", color=COLORS["gru"], edgecolor="#1e1e2e", linewidth=2)
        
        ax1.set_ylabel("Changed-cell Accuracy", fontsize=12, fontweight="bold")
        ax1.set_title("Overall Performance", fontsize=14, fontweight="bold", pad=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(settings_labels, fontsize=10)
        ax1.set_ylim(0, 1.1)
        ax1.legend(loc="upper right", fontsize=10)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.axhline(y=0.5, color="#45475a", linestyle="--", alpha=0.5)
        
        # Add value labels
        for sd_bars in [sd_bars1, sd_bars2, sd_bars3]:
            for sd_bar in sd_bars:
                sd_height = sd_bar.get_height()
                ax1.text(sd_bar.get_x() + sd_bar.get_width()/2, sd_height + 0.02, f'{int(sd_height*100)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Right: Enter-step accuracy (the interesting finding!)
        ax2 = showdown_axes[1]
        sd_bars1e = ax2.bar(x - width, comparison_df["MLP_enter"], width, label="MLP", color=COLORS["mlp"], edgecolor="#1e1e2e", linewidth=2)
        sd_bars2e = ax2.bar(x, comparison_df["Transformer_enter"], width, label="Transformer", color=COLORS["transformer"], edgecolor="#1e1e2e", linewidth=2)
        sd_bars3e = ax2.bar(x + width, comparison_df["GRU_enter"], width, label="GRU", color=COLORS["gru"], edgecolor="#1e1e2e", linewidth=2)
        
        ax2.set_ylabel("Enter-step Accuracy", fontsize=12, fontweight="bold")
        ax2.set_title("🎯 After Pressing Enter", fontsize=14, fontweight="bold", pad=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(settings_labels, fontsize=10)
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc="upper right", fontsize=10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.axhline(y=0.5, color="#45475a", linestyle="--", alpha=0.5)
        
        for sd_barse in [sd_bars1e, sd_bars2e, sd_bars3e]:
            for sd_bare in sd_barse:
                sd_heighte = sd_bare.get_height()
                ax2.text(sd_bare.get_x() + sd_bare.get_width()/2, sd_heighte + 0.02, f'{int(sd_heighte*100)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        showdown_fig.tight_layout()
        showdown_chart = showdown_fig
    else:
        showdown_chart = None
        comparison_df = None

    showdown_output = showdown_chart if showdown_chart else mo.callout(mo.md("**Note:** Pre-computed comparison data not found. Run the baseline scripts first."), kind="warn")
    return (
        comparison_df,
        comparison_rows,
        gru_data,
        has_comparison_data,
        mlp_data,
        results_path,
        settings_labels,
        showdown_chart,
        showdown_fig,
        showdown_output,
        transformer_data,
    )


@app.cell(hide_code=True)
def _(showdown_output):
    showdown_output
    return


@app.cell(hide_code=True)
def _(COLORS, has_comparison_data, mlp_data, mo, transformer_data):
    if not has_comparison_data:
        insight_output = mo.md("")
    else:
        mlp_enter = mlp_data["standard_family"]["action_breakdown"]["enter"]["changed_acc"]
        tf_enter = transformer_data["standard_family"]["action_breakdown"]["enter"]["changed_acc"]
        mlp_overall = mlp_data["standard_family"]["metrics"]["changed_acc"]
        tf_overall = transformer_data["standard_family"]["metrics"]["changed_acc"]
        
        insight_html = f"""
        <div style="
            background: linear-gradient(135deg, {COLORS['transformer']}15 0%, {COLORS['mlp']}15 100%);
            border-radius: 20px;
            padding: 28px;
            margin: 24px 0;
            border: 2px solid {COLORS['transformer']}44;
        ">
            <h3 style="color: #cdd6f4; text-align: center; margin-bottom: 20px;">🔍 The Surprising Discovery</h3>
            <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 20px; align-items: center;">
                <div style="text-align: center;">
                    <div style="font-size: 1.1em; color: #a6adc8; margin-bottom: 8px;">MLP wins overall</div>
                    <div style="font-size: 2.4em; font-weight: 800; color: {COLORS['mlp']};">{int(mlp_overall*100)}%</div>
                    <div style="color: #6b7280; font-size: 0.9em;">changed-cell accuracy</div>
                </div>
                <div style="font-size: 2em; color: #6b7280;">vs</div>
                <div style="text-align: center;">
                    <div style="font-size: 1.1em; color: #a6adc8; margin-bottom: 8px;">Transformer: {int(tf_overall*100)}%</div>
                    <div style="font-size: 2.4em; font-weight: 800; color: {COLORS['transformer']};">{int(tf_enter*100)}%</div>
                    <div style="color: {COLORS['transformer']}; font-size: 0.9em; font-weight: bold;">on Enter steps! 🎯</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 20px; padding-top: 20px; border-top: 1px solid #45475a;">
                <p style="color: #cdd6f4; font-size: 1.1em; margin: 0;">
                    The Transformer loses overall, but <strong style="color: {COLORS['transformer']};">massively outperforms on meaning-heavy steps</strong>.
                    <br><span style="color: #a6adc8;">MLP Enter accuracy: {int(mlp_enter*100)}% → Transformer: {int(tf_enter*100)}%</span>
                </p>
            </div>
        </div>
        """
        insight_output = mo.Html(insight_html)
    return (insight_output,)


@app.cell(hide_code=True)
def _(insight_output):
    insight_output
    return


@app.cell(hide_code=True)
def _(has_comparison_data, mo):
    comparison_callout = mo.callout(
        mo.md("""
**What this means:**

- 🟢 **MLP** is best at the easy parts (typing echoes, cursor movement)
- 🟣 **Transformer** struggles with easy parts but shines when **meaning matters** (after Enter)
- 🔴 **GRU** underperformed on everything — just adding recurrence doesn't help

The Transformer seems to learn something about *what commands do*, even though it's worse at the surface mechanics!
"""),
        kind="success"
    ) if has_comparison_data else mo.md("")
    return (comparison_callout,)


@app.cell(hide_code=True)
def _(comparison_callout):
    comparison_callout
    return


# ============================================================================
# PETAL 5: LIVE COMPARISON
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🔬 Petal 5: Watch Models Think Live

Let's see the MLP model in action on a test example. Step through and watch it predict.
""")
    return


@app.cell(hide_code=True)
def _(mo, trained_test_eps):
    test_picker = mo.ui.slider(0, len(trained_test_eps) - 1, value=0, step=1, label="🎬 Pick a test example")
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
    rollout_step = mo.ui.slider(0, picked_max_step, value=min(3, picked_max_step), step=1, label="⏩ Step through", full_width=True)
    rollout_step
    return (rollout_step,)


@app.cell(hide_code=True)
def _(COLORS, changed_cell_accuracy, learned_predictions, mo, np, picked_episode, rollout_step):
    rs = rollout_step.value
    truth_frame = picked_episode.frames[rs + 1]
    pred_frame = learned_predictions[rs + 1]
    prev_frame_r = picked_episode.frames[rs]
    action_r = picked_episode.actions[rs]
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
                    chars.append(f'<span style="background:{COLORS["wrong"]}; color:white; font-weight:bold; border-radius: 2px;">{ch}</span>')
                elif was_changed and is_correct:
                    chars.append(f'<span style="background:{COLORS["correct"]}; color:white; font-weight:bold; border-radius: 2px;">{ch}</span>')
                elif was_changed:
                    chars.append(f'<span style="background:{COLORS["changed"]}; color:#000; border-radius: 2px;">{ch}</span>')
                else:
                    chars.append(f'<span style="color: #888;">{ch}</span>')
            rows.append("".join(chars))
        content = "<br>".join(rows)
        return f"""
        <div class="nc-terminal" style="min-width: 360px;">
            <div style="color: #6b7280; font-size: 11px; margin-bottom: 10px; font-weight: 600;">{title}</div>
            <div style="background: #0a0a0f; padding: 12px; border-radius: 8px; border: 1px solid #222;">{content}</div>
        </div>
        """

    truth_html = render_comparison_terminal(truth_frame, truth_frame, prev_frame_r, "✅ GROUND TRUTH", show_errors=False)
    pred_html = render_comparison_terminal(pred_frame, truth_frame, prev_frame_r, "🧠 MODEL PREDICTION")
    acc = changed_cell_accuracy(prev_frame_r, pred_frame, truth_frame)
    acc_pct = int(acc * 100)

    if action_r.kind == "enter":
        action_label, action_bg = "⏎ Enter", COLORS["enter"]
    elif action_r.kind == "type_char":
        action_label, action_bg = f"⌨️ Type: {action_r.typed_char}", COLORS["typing"]
    else:
        action_label, action_bg = f"⏸️ {action_r.kind}", COLORS["neutral"]

    header_html = f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <span style="background: {action_bg}; color: white; padding: 8px 18px; border-radius: 20px; font-weight: bold; margin-right: 12px; box-shadow: 0 4px 12px {action_bg}44;">
            {action_label}
        </span>
        <span style="background: {'#22c55e' if acc > 0.7 else '#f59e0b' if acc > 0.4 else '#ef4444'}; color: white; padding: 8px 18px; border-radius: 20px; font-weight: bold;">
            {acc_pct}% correct on changed
        </span>
    </div>
    """

    legend_html = f"""
    <div style="display: flex; gap: 20px; justify-content: center; margin-top: 16px; font-size: 0.9em;">
        <span><span style="background:{COLORS['correct']}; color:white; padding: 3px 10px; border-radius: 4px;">■</span> Correct</span>
        <span><span style="background:{COLORS['wrong']}; color:white; padding: 3px 10px; border-radius: 4px;">■</span> Wrong</span>
        <span><span style="background:{COLORS['changed']}; color:#000; padding: 3px 10px; border-radius: 4px;">■</span> Changed</span>
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
def _(COLORS, np, plt, pred_frame, prev_frame_r, truth_frame):
    error_grid = np.zeros(truth_frame.shape, dtype=np.int8)
    error_changed_mask = truth_frame != prev_frame_r
    error_correct_changed = error_changed_mask & (pred_frame == truth_frame)
    error_wrong_changed = error_changed_mask & (pred_frame != truth_frame)
    error_grid[error_correct_changed] = 1
    error_grid[error_wrong_changed] = -1

    error_map_fig, error_map_ax = plt.subplots(figsize=(10, 3))
    error_cmap = plt.matplotlib.colors.ListedColormap([COLORS["wrong"], "#1e1e2e", COLORS["correct"]])
    error_norm = plt.matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], error_cmap.N)
    error_map_ax.imshow(error_grid, cmap=error_cmap, norm=error_norm, aspect="auto")
    error_map_ax.set_title("Error Map: 🟢 correct change  🔴 wrong guess  ⬛ unchanged", fontsize=12, fontweight="bold", pad=12)
    error_map_ax.set_xticks([])
    error_map_ax.set_yticks([])
    for spine in error_map_ax.spines.values():
        spine.set_visible(False)
    error_map_fig.tight_layout()
    error_map_fig
    return error_changed_mask, error_cmap, error_correct_changed, error_grid, error_map_fig, error_norm, error_wrong_changed


# ============================================================================
# PETAL 6: THE EXPERIMENTS
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🧪 Petal 6: Controlled Experiments

Let's run some rigorous experiments to understand what drives performance.
""")
    return


@app.cell(hide_code=True)
def _(conditioning_study_multiseed, mo, noise_study_multiseed, np, paraphrase_generalization_multiseed, plt):
    cond_data = conditioning_study_multiseed(train_n=12, test_n=8, hidden_size=64, max_iter=25, negative_ratio=6, seeds=(11, 17))
    noise_data = noise_study_multiseed(train_n=12, test_n=8, hidden_size=64, max_iter=25, negative_ratio=6, seeds=(13, 19))
    para_data = paraphrase_generalization_multiseed(train_n=10, test_n=8, hidden_size=64, max_iter=25, negative_ratio=6, seeds=(31, 37))

    def make_hints_chart():
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ["#6b7280", "#3b82f6", "#22c55e"]
        bars = ax.bar(cond_data["conditioning"], cond_data["changed_acc_mean"], yerr=cond_data["changed_acc_std"], color=colors, capsize=8, edgecolor="#1e1e2e", linewidth=3, width=0.6)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy on changed cells", fontsize=12, fontweight="bold")
        ax.set_title("Does giving hints help?", fontsize=15, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(y=0.5, color="#45475a", linestyle="--", alpha=0.7)
        for bar, val, col in zip(bars, cond_data["changed_acc_mean"], colors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{int(val*100)}%", ha="center", fontsize=13, fontweight="bold", color=col)
        fig.tight_layout()
        return fig

    def make_noise_chart():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        order = ["clean→clean", "noisy→clean", "clean→noisy", "noisy→noisy"]
        colors = ["#22c55e", "#f59e0b", "#3b82f6", "#ef4444"]
        ordered = noise_data.set_index("setting").loc[order].reset_index()
        bars = ax.bar(ordered["setting"], ordered["changed_acc_mean"], yerr=ordered["changed_acc_std"], color=colors, capsize=8, edgecolor="#1e1e2e", linewidth=3, width=0.6)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy on changed cells", fontsize=12, fontweight="bold")
        ax.set_title("Training Environment Study", fontsize=15, fontweight="bold", pad=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, val, col in zip(bars, ordered["changed_acc_mean"], colors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{int(val*100)}%", ha="center", fontsize=13, fontweight="bold", color=col)
        fig.tight_layout()
        return fig

    def make_paraphrase_chart():
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(para_data))
        width = 0.35
        bars1 = ax.bar(x - width/2, para_data["typing_changed_acc_mean"], width, yerr=para_data["typing_changed_acc_std"], label="Typing steps", color="#3b82f6", capsize=5, edgecolor="#1e1e2e", linewidth=2)
        bars2 = ax.bar(x + width/2, para_data["enter_changed_acc_mean"], width, yerr=para_data["enter_changed_acc_std"], label="Enter steps", color="#f97316", capsize=5, edgecolor="#1e1e2e", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(para_data["conditioning"], fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy on changed cells", fontsize=12, fontweight="bold")
        ax.set_title("Can it handle commands it never saw?", fontsize=15, fontweight="bold", pad=15)
        ax.legend(loc="upper right", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    hints_chart = make_hints_chart()
    noise_chart = make_noise_chart()
    para_chart = make_paraphrase_chart()

    exp1_content = mo.vstack([
        mo.md("""
**Question:** If we tell the model what command is running, does it do better?

**Setup:** Train with different levels of hints — none, command type, or exact command.
        """),
        hints_chart,
        mo.callout(mo.md("**Finding:** More hints = better accuracy. But even without hints, it learns something!"), kind="success"),
    ])

    exp2_content = mo.vstack([
        mo.md("""
**Question:** Is it better to train on clean, perfect examples or messy, varied ones?

**Setup:** We test all combinations of clean/noisy training and testing.
        """),
        noise_chart,
        mo.callout(mo.md("**Surprise:** Training on messy data can actually help! Variety teaches robustness."), kind="warn"),
    ])

    exp3_content = mo.vstack([
        mo.md("""
**Question:** If the model only saw `whoami`, can it handle `id -un` (same meaning, different words)?

**Setup:** Train on one phrasing per command, test on completely different phrasings.
        """),
        para_chart,
        mo.callout(mo.md("""
**Key insight:** 
- ⌨️ **Typing** transfers well — the model learned how typing works in general
- ⏎ **Enter** transfers poorly — it memorized specific outputs, not the meaning
        """), kind="info"),
    ])

    experiments_tabs = mo.ui.tabs({
        "🎯 Hints Study": exp1_content,
        "🧹 Clean vs Messy": exp2_content,
        "🔄 Paraphrase Test": exp3_content,
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


# ============================================================================
# PETAL 7: FAILURE GALLERY
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🥀 Petal 7: Where Models Fail

A good scientist shows failures, not just successes. Here's where even the best models struggle.
""")
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    failures_html = f"""
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0;">
        <div class="nc-card" style="border-left: 4px solid {COLORS['wrong']};">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 1.8em;">❌</span>
                <h4 style="color: {COLORS['wrong']}; margin: 0;">After Enter</h4>
            </div>
            <p style="color: #cdd6f4; margin: 0;">When a command runs, the model often guesses wrong output. It knows <em>something</em> should appear, but not <em>what</em>.</p>
        </div>
        <div class="nc-card" style="border-left: 4px solid {COLORS['enter']};">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 1.8em;">🔄</span>
                <h4 style="color: {COLORS['enter']}; margin: 0;">New Commands</h4>
            </div>
            <p style="color: #cdd6f4; margin: 0;">If it never saw <code>id -un</code>, it can't guess the output even though it means the same as <code>whoami</code>.</p>
        </div>
        <div class="nc-card" style="border-left: 4px solid {COLORS['neutral']};">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 1.8em;">🔢</span>
                <h4 style="color: #a6adc8; margin: 0;">Math Problems</h4>
            </div>
            <p style="color: #cdd6f4; margin: 0;">For <code>python -c "print(3+4)"</code>, it knows a number appears, but often gets the wrong number.</p>
        </div>
        <div class="nc-card" style="border-left: 4px solid {COLORS['typing']};">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 1.8em;">📏</span>
                <h4 style="color: {COLORS['typing']}; margin: 0;">Long Commands</h4>
            </div>
            <p style="color: #cdd6f4; margin: 0;">The longer the command, the more chances to make small mistakes that accumulate.</p>
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
# PETAL 8: THE BIG PICTURE
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
## 🌳 Petal 8: What We Learned

Let's zoom out and see the big picture.
""")
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    final_summary_html = f"""
    <div style="
        background: linear-gradient(135deg, {COLORS['bg_dark']} 0%, #2d1b4e 50%, {COLORS['bg_dark']} 100%);
        border-radius: 24px;
        padding: 40px;
        margin: 24px 0;
        position: relative;
        overflow: hidden;
    ">
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(circle at 20% 80%, rgba(34, 197, 94, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(168, 85, 247, 0.1) 0%, transparent 50%); pointer-events: none;"></div>
        <div style="position: relative; z-index: 1;">
            <h3 style="color: #cdd6f4; text-align: center; margin-bottom: 32px; font-size: 1.5em;">Key Discoveries</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 3em; margin-bottom: 12px;">🏆</div>
                    <div style="color: {COLORS['mlp']}; font-weight: bold; font-size: 1.2em; margin-bottom: 8px;">MLP Wins Overall</div>
                    <div style="color: #a6adc8; font-size: 0.95em;">Simple feedforward networks excel at the easy parts</div>
                </div>
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 3em; margin-bottom: 12px;">🎯</div>
                    <div style="color: {COLORS['transformer']}; font-weight: bold; font-size: 1.2em; margin-bottom: 8px;">Transformer Wins Meaning</div>
                    <div style="color: #a6adc8; font-size: 0.95em;">Attention helps with what commands <em>do</em></div>
                </div>
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 3em; margin-bottom: 12px;">🔥</div>
                    <div style="color: {COLORS['wrong']}; font-weight: bold; font-size: 1.2em; margin-bottom: 8px;">Enter is Hard</div>
                    <div style="color: #a6adc8; font-size: 0.95em;">All models struggle with semantic prediction</div>
                </div>
            </div>
        </div>
    </div>
    """
    mo.Html(final_summary_html)
    return (final_summary_html,)


@app.cell(hide_code=True)
def _(mo):
    takeaways = mo.accordion({
        "🤔 The Deep Question": mo.md("""
**Can you learn how something works just by watching it?**

Humans do this all the time. We learn to use apps by watching others, not by reading code.

This notebook suggests: **yes, but you learn the *surface behavior* before the *deep logic*.**

That's both exciting and limiting.
        """),
        "📊 Architecture Lesson": mo.md("""
Different architectures have different strengths:

- **MLP**: Best at local, predictable patterns (typing echoes)
- **Transformer**: Better at global semantics (command meaning)
- **GRU**: Didn't help on this task — memory alone isn't enough

The winner depends on what you're measuring!
        """),
        "📚 Connection to Neural Computers": mo.md("""
The [Neural Computers paper](https://arxiv.org/abs/2604.06425) does all of this at scale:
- Real video recordings of computers
- Large video generation models
- Much more training data

This notebook is a tiny toy version to build intuition about *why* that approach might work — and where it might struggle.
        """),
        "🚀 What Would Make This Better": mo.md("""
- **Memory**: Our models have no explicit memory. A model that remembers earlier context might understand commands better.
- **Bigger models**: More capacity could help close the Enter-step gap.
- **Real data**: Move from toy terminals to actual screen recordings.
        """),
    }, multiple=True)
    takeaways
    return (takeaways,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
---

<div style="text-align: center; color: #6b7280; padding: 32px;">
    <p style="margin-bottom: 12px;">
        <span style="font-size: 1.2em;">Built with</span> 
        <a href="https://marimo.io" style="color: #89b4fa; font-weight: bold; text-decoration: none;">marimo</a>
    </p>
    <p style="margin-bottom: 12px;">
        Inspired by <a href="https://arxiv.org/abs/2604.06425" style="color: #89b4fa; text-decoration: none;">Neural Computers</a> (Zhuge et al., 2026)
    </p>
    <p style="color: #45475a; font-size: 0.9em;">
        🧪 MLP vs Transformer experiments run on NVIDIA A100 GPU
    </p>
</div>
""")
    return


if __name__ == "__main__":
    app.run()
