# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///
"""
Neural Computers competition notebook.
Presentation-focused and export-friendly: it uses precomputed experiment results.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "experiments" / "toy_nc_cli"))

    from src.toy_terminal import COMMAND_VARIANTS, TerminalConfig, generate_episodes, html_screen

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
        "muted": "#a6adc8",
        "text": "#cdd6f4",
        "accent": "#89b4fa",
    }

    plt.rcParams.update(
        {
            "figure.facecolor": "#1e1e2e",
            "axes.facecolor": "#1e1e2e",
            "axes.edgecolor": "#6b7280",
            "axes.labelcolor": "#cdd6f4",
            "text.color": "#cdd6f4",
            "xtick.color": "#a6adc8",
            "ytick.color": "#a6adc8",
            "grid.color": "#45475a",
            "legend.facecolor": "#313244",
            "legend.edgecolor": "#6b7280",
        }
    )

    return COLORS, COMMAND_VARIANTS, ROOT, TerminalConfig, generate_episodes, html_screen, json, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    style_html_comp = """
    <style>
    .nc-card {
        background: linear-gradient(145deg, #313244, #1e1e2e);
        border-radius: 18px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.28);
    }
    .nc-small {
        color: #a6adc8;
        font-size: 0.93em;
        line-height: 1.6;
    }
    .nc-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95em;
        overflow: hidden;
        border-radius: 14px;
    }
    .nc-table th, .nc-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #45475a;
        text-align: center;
    }
    .nc-table th {
        background: #313244;
        color: #cdd6f4;
        font-weight: 700;
    }
    .nc-table td:first-child, .nc-table th:first-child {
        text-align: left;
    }
    .nc-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9em;
    }
    .nc-hero {
        box-shadow: 0 12px 36px rgba(0,0,0,0.22);
    }
    </style>
    """
    mo.Html(style_html_comp)
    return


# ============================================================================
# HERO
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    hero_html_comp = """
    <div class="nc-hero" style="background: linear-gradient(135deg, #1e1e2e 0%, #2d1b4e 50%, #1e1e2e 100%); border-radius: 24px; padding: 56px 48px; text-align: center; margin-bottom: 32px; position: relative; overflow: hidden;">
        <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle at 30% 30%, rgba(168, 85, 247, 0.10) 0%, transparent 50%), radial-gradient(circle at 70% 70%, rgba(59, 130, 246, 0.10) 0%, transparent 50%); pointer-events: none;"></div>
        <div style="position: relative; z-index: 1;">
            <div style="display:inline-block; margin-bottom: 18px; padding: 8px 14px; border: 1px solid rgba(255,255,255,0.12); border-radius: 999px; color: #cdd6f4; font-size: 0.9em; letter-spacing: 0.04em; text-transform: uppercase; background: rgba(255,255,255,0.04);">Interactive research notebook</div>
            <h1 style="font-size: 2.7em; color: #cdd6f4; margin-bottom: 16px; font-weight: 800; letter-spacing: -0.02em;">Can a Model Learn How a Computer Works?</h1>
            <p style="font-size: 1.35em; color: #a6adc8; max-width: 680px; margin: 0 auto 28px auto; line-height: 1.7;">
                Not by reading code. Not by being taught.<br>
                <strong style="color: #f9e2af;">Just by watching the screen change.</strong>
            </p>
            <div style="display: inline-flex; gap: 12px; align-items: center; background: rgba(69, 71, 90, 0.6); backdrop-filter: blur(10px); border-radius: 12px; padding: 14px 24px; border: 1px solid rgba(255,255,255,0.1);">
                <span style="color: #cdd6f4;">Inspired by</span>
                <a href="https://arxiv.org/abs/2604.06425" style="color: #89b4fa; font-weight: 600; text-decoration: none;">Neural Computers</a>
                <span style="color: #6b7280;">(2026)</span>
            </div>
        </div>
    </div>
    """
    mo.Html(hero_html_comp)
    return


@app.cell(hide_code=True)
def _(mo):
    intro_callout_comp = mo.callout(
        mo.md(
            """
**The big question:** If you show a model lots of screen recordings, can it learn to predict what happens next?

This notebook explores that question with a toy terminal benchmark and a fair comparison between three model families.
"""
        ),
        kind="info",
    )
    intro_callout_comp
    return


@app.cell(hide_code=True)
def _(mo):
    journey_html_comp = """
    <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin: 24px 0 32px 0;">
        <div class='nc-card' style='border-left: 3px solid #3b82f6;'><div style='font-size: 0.82em; letter-spacing:0.06em; text-transform:uppercase; color:#89b4fa; margin-bottom: 8px;'>01</div><div style='font-weight: 700; color: #cdd6f4;'>Task</div><div class='nc-small'>Predict the next screen from the current one.</div></div>
        <div class='nc-card' style='border-left: 3px solid #f59e0b;'><div style='font-size: 0.82em; letter-spacing:0.06em; text-transform:uppercase; color:#f59e0b; margin-bottom: 8px;'>02</div><div style='font-weight: 700; color: #cdd6f4;'>Split</div><div class='nc-small'>Typing is easy. Meaning after Enter is hard.</div></div>
        <div class='nc-card' style='border-left: 3px solid #22c55e;'><div style='font-size: 0.82em; letter-spacing:0.06em; text-transform:uppercase; color:#22c55e; margin-bottom: 8px;'>03</div><div style='font-weight: 700; color: #cdd6f4;'>Metric</div><div class='nc-small'>We score changed cells, not easy unchanged background.</div></div>
        <div class='nc-card' style='border-left: 3px solid #a855f7;'><div style='font-size: 0.82em; letter-spacing:0.06em; text-transform:uppercase; color:#a855f7; margin-bottom: 8px;'>04</div><div style='font-weight: 700; color: #cdd6f4;'>Comparison</div><div class='nc-small'>MLP vs Transformer vs GRU on the same settings.</div></div>
        <div class='nc-card' style='border-left: 3px solid #f97316;'><div style='font-size: 0.82em; letter-spacing:0.06em; text-transform:uppercase; color:#f97316; margin-bottom: 8px;'>05</div><div style='font-weight: 700; color: #cdd6f4;'>Finding</div><div class='nc-small'>Different architectures win at different parts of the job.</div></div>
    </div>
    """
    mo.Html(journey_html_comp)
    return


# ============================================================================
# WHAT EXACTLY WAS MEASURED?
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    protocol_md_comp = mo.md(
        """
## What exactly was measured?

Before looking at the winner, here is the benchmark in plain English.
"""
    )
    protocol_md_comp
    return


@app.cell(hide_code=True)
def _(mo):
    protocol_html_comp = """
    <div style="display:grid; grid-template-columns: 1.2fr 1fr 1fr 1fr; gap: 16px; margin: 20px 0 24px 0;">
        <div class="nc-card" style="border-left: 4px solid #89b4fa;">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:#89b4fa; margin-bottom:8px;">01</div>
            <div style="font-weight:700; color:#cdd6f4; margin-bottom:6px;">One sample</div>
            <div class="nc-small">A terminal episode: current screen → next screen after one action.</div>
        </div>
        <div class="nc-card" style="border-left: 4px solid #22c55e;">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:#22c55e; margin-bottom:8px;">02</div>
            <div style="font-weight:700; color:#cdd6f4; margin-bottom:6px;">Main metric</div>
            <div class="nc-small"><b>Changed-cell accuracy</b>: only score the cells that actually changed.</div>
        </div>
        <div class="nc-card" style="border-left: 4px solid #a855f7;">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:#a855f7; margin-bottom:8px;">03</div>
            <div style="font-weight:700; color:#cdd6f4; margin-bottom:6px;">Four settings</div>
            <div class="nc-small">Standard family, standard command, paraphrase family, paraphrase command.</div>
        </div>
        <div class="nc-card" style="border-left: 4px solid #f97316;">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:#f97316; margin-bottom:8px;">04</div>
            <div style="font-weight:700; color:#cdd6f4; margin-bottom:6px;">Fair comparison</div>
            <div class="nc-small">Same toy task, same evaluation settings, matched saved result files.</div>
        </div>
    </div>
    """
    mo.Html(protocol_html_comp)
    return


@app.cell(hide_code=True)
def _(mo):
    metric_note_comp = mo.callout(
        mo.md(
            """
**Why changed-cell accuracy?** Most of the screen never changes. A model can look impressive on ordinary character accuracy while still failing the actual prediction problem.
"""
        ),
        kind="warn",
    )
    metric_note_comp
    return


# ============================================================================
# WATCH THE TOY COMPUTER
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    toy_watch_heading_comp = mo.md(
        """
## Watch the toy computer

Before the benchmark numbers, here is the task itself as a tiny interactive terminal.
Pick a command and scrub through time.
"""
    )
    toy_watch_heading_comp
    return


@app.cell(hide_code=True)
def _(COMMAND_VARIANTS, mo):
    toy_family_picker_comp = mo.ui.dropdown(
        options=list(COMMAND_VARIANTS.keys()),
        value="whoami",
        label="Command family",
    )
    toy_noise_toggle_comp = mo.ui.switch(value=False, label="Add noise")
    toy_controls_row_comp = mo.hstack(
        [toy_family_picker_comp, toy_noise_toggle_comp],
        justify="center",
        gap=2,
    )
    toy_controls_row_comp
    return toy_family_picker_comp, toy_noise_toggle_comp


@app.cell(hide_code=True)
def _(COMMAND_VARIANTS, mo, toy_family_picker_comp):
    toy_variant_options_comp = [
        f"phrasing {toy_variant_index_comp + 1}"
        for toy_variant_index_comp in range(len(COMMAND_VARIANTS[toy_family_picker_comp.value]))
    ]
    toy_variant_picker_comp = mo.ui.dropdown(
        options=toy_variant_options_comp,
        value=toy_variant_options_comp[0],
        label="Command phrasing",
    )
    toy_variant_picker_comp
    return toy_variant_picker_comp, toy_variant_options_comp


@app.cell(hide_code=True)
def _(TerminalConfig, generate_episodes, toy_family_picker_comp, toy_noise_toggle_comp, toy_variant_picker_comp):
    toy_demo_config_comp = TerminalConfig(rows=10, cols=40, context_width=32, patch_radius=1)
    toy_variant_idx_comp = int(toy_variant_picker_comp.value.split()[-1]) - 1
    toy_demo_episode_comp = generate_episodes(
        n=1,
        config=toy_demo_config_comp,
        noisy=toy_noise_toggle_comp.value,
        seed=77 + toy_variant_idx_comp,
        context_mode="command",
        families=[toy_family_picker_comp.value],
        variant_indices_by_family={toy_family_picker_comp.value: (toy_variant_idx_comp,)},
    )[0]
    toy_demo_max_step_comp = len(toy_demo_episode_comp.actions)
    return toy_demo_config_comp, toy_demo_episode_comp, toy_demo_max_step_comp


@app.cell(hide_code=True)
def _(mo, toy_demo_max_step_comp):
    toy_step_slider_comp = mo.ui.slider(
        0,
        toy_demo_max_step_comp,
        value=0,
        step=1,
        label="Step through time",
        full_width=True,
    )
    toy_step_slider_comp
    return (toy_step_slider_comp,)


@app.cell(hide_code=True)
def _(COLORS, mo, np, toy_demo_episode_comp, toy_demo_max_step_comp, toy_step_slider_comp):
    toy_step_value_comp = toy_step_slider_comp.value
    toy_frame_comp = toy_demo_episode_comp.frames[toy_step_value_comp]
    if toy_step_value_comp == 0:
        toy_prev_frame_comp = toy_frame_comp.copy()
        toy_changed_mask_comp = np.zeros_like(toy_frame_comp, dtype=bool)
        toy_action_label_comp = "Start state"
        toy_action_color_comp = COLORS["neutral"]
    else:
        toy_prev_frame_comp = toy_demo_episode_comp.frames[toy_step_value_comp - 1]
        toy_changed_mask_comp = toy_frame_comp != toy_prev_frame_comp
        toy_action_comp = toy_demo_episode_comp.actions[toy_step_value_comp - 1]
        if toy_action_comp.kind == "type_char":
            toy_action_label_comp = f"Type {toy_action_comp.typed_char!r}"
            toy_action_color_comp = COLORS["typing"]
        elif toy_action_comp.kind == "enter":
            toy_action_label_comp = "Press Enter"
            toy_action_color_comp = COLORS["enter"]
        elif toy_action_comp.kind == "backspace":
            toy_action_label_comp = "Backspace"
            toy_action_color_comp = COLORS["typing"]
        else:
            toy_action_label_comp = "Idle"
            toy_action_color_comp = COLORS["neutral"]

    def toy_render_terminal_comp(toy_frame_render_comp, toy_changed_render_comp, toy_title_render_comp):
        toy_lines_render_comp = []
        for toy_row_index_comp in range(toy_frame_render_comp.shape[0]):
            toy_chars_render_comp = []
            for toy_col_index_comp in range(toy_frame_render_comp.shape[1]):
                toy_char_comp = toy_frame_render_comp[toy_row_index_comp, toy_col_index_comp]
                toy_char_html_comp = "&nbsp;" if toy_char_comp == " " else toy_char_comp
                if toy_changed_render_comp[toy_row_index_comp, toy_col_index_comp]:
                    toy_chars_render_comp.append(
                        f'<span style="background:{COLORS["changed"]}; color:#111; font-weight:700; border-radius:2px;">{toy_char_html_comp}</span>'
                    )
                else:
                    toy_chars_render_comp.append(f'<span style="color:#d7dae2;">{toy_char_html_comp}</span>')
            toy_lines_render_comp.append("".join(toy_chars_render_comp))
        toy_body_render_comp = "<br>".join(toy_lines_render_comp)
        return f"""
        <div style="background: linear-gradient(180deg, #0d0d0d 0%, #151521 100%); border-radius: 14px; padding: 16px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #e5e7eb; border: 1px solid #313244; min-width: 460px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);">
            <div style="color:#8a90a8; font-size:11px; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;">{toy_title_render_comp}</div>
            {toy_body_render_comp}
        </div>
        """

    toy_changed_count_comp = int(toy_changed_mask_comp.sum())
    toy_demo_panel_comp = mo.vstack(
        [
            mo.Html(
                toy_render_terminal_comp(
                    toy_frame_comp,
                    toy_changed_mask_comp,
                    f"Command: {toy_demo_episode_comp.command_text}",
                )
            ),
            mo.Html(
                f"""
                <div style="display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin-top: 10px;">
                    <span class="nc-pill" style="background:{toy_action_color_comp}; color:white;">{toy_action_label_comp}</span>
                    <span class="nc-pill" style="background:#313244; color:#cdd6f4;">Step {toy_step_value_comp}/{toy_demo_max_step_comp}</span>
                    <span class="nc-pill" style="background:{COLORS['changed']}22; color:{COLORS['changed']}; border:1px solid {COLORS['changed']}44;">{toy_changed_count_comp} changed cells</span>
                </div>
                """
            ),
        ],
        align="center",
    )
    toy_demo_panel_comp
    return toy_action_color_comp, toy_action_label_comp, toy_changed_count_comp


@app.cell(hide_code=True)
def _(mo):
    toy_watch_note_comp = mo.callout(
        mo.md(
            """
**Notice the asymmetry:** typing usually changes one tiny patch, but pressing Enter can change many cells at once.
That is why the benchmark separates easy mechanics from harder command meaning.
"""
        ),
        kind="info",
    )
    toy_watch_note_comp
    return


# ============================================================================
# EASY VS HARD
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    challenge_md_comp = mo.md(
        """
## The core challenge: easy vs. hard

The toy computer asks models to learn two very different skills.
"""
    )
    challenge_md_comp
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    challenge_html_comp = f"""
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 28px; margin: 24px 0;">
        <div class="nc-card" style="background: linear-gradient(145deg, {COLORS['typing']}15 0%, {COLORS['typing']}05 100%); border: 2px solid {COLORS['typing']}66; text-align:center; padding:32px;">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:{COLORS['typing']}; margin-bottom:16px;">Local update</div>
            <h3 style="color:{COLORS['typing']}; margin-bottom:12px; font-size:1.4em;">Mechanics</h3>
            <p style="color:#cdd6f4; font-size:1.12em; margin-bottom:20px;">“The user typed <code style='background:#222; padding:4px 8px; border-radius:4px;'>w</code>, so show <code style='background:#222; padding:4px 8px; border-radius:4px;'>w</code>.”</p>
            <div style="background:{COLORS['correct']}; color:white; padding:10px 24px; border-radius:24px; display:inline-block; font-weight:bold;">Easier</div>
        </div>
        <div class="nc-card" style="background: linear-gradient(145deg, {COLORS['enter']}15 0%, {COLORS['enter']}05 100%); border: 2px solid {COLORS['enter']}66; text-align:center; padding:32px;">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:{COLORS['enter']}; margin-bottom:16px;">Semantic update</div>
            <h3 style="color:{COLORS['enter']}; margin-bottom:12px; font-size:1.4em;">Meaning</h3>
            <p style="color:#cdd6f4; font-size:1.12em; margin-bottom:20px;">“The user ran <code style='background:#222; padding:4px 8px; border-radius:4px;'>whoami</code>, so show the username.”</p>
            <div style="background:{COLORS['wrong']}; color:white; padding:10px 24px; border-radius:24px; display:inline-block; font-weight:bold;">Harder</div>
        </div>
    </div>
    """
    mo.Html(challenge_html_comp)
    return


@app.cell(hide_code=True)
def _(mo):
    challenge_callout_comp = mo.callout(
        mo.md(
            """
**The scientific question:** Does a model first learn how a computer *looks*, or how it actually *works*?

This benchmark suggests: surface mechanics come first. Command meaning is the harder part.
"""
        ),
        kind="info",
    )
    challenge_callout_comp
    return


# ============================================================================
# HOW TO IMPLEMENT THE IDEA
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    impl_heading_comp = mo.md(
        """
## How you would implement the core idea yourself

At a high level, this toy benchmark turns the paper’s idea into a simple loop:
current screen in → next screen out.
"""
    )
    impl_heading_comp
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    impl_flow_html_comp = f"""
    <div style="display:grid; grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr; gap: 10px; align-items:center; margin: 20px 0 26px 0;">
        <div class="nc-card" style="text-align:center; border-left:4px solid {COLORS['accent']};">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:{COLORS['accent']}; margin-bottom:8px;">Input</div>
            <div style="font-weight:700; color:#cdd6f4; margin-bottom:6px;">1. Current screen</div>
            <div class="nc-small">What the terminal looks like now.</div>
        </div>
        <div style="text-align:center; color:#a6adc8; font-size:1.5em;">→</div>
        <div class="nc-card" style="text-align:center; border-left:4px solid {COLORS['typing']};">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:{COLORS['typing']}; margin-bottom:8px;">Context</div>
            <div style="font-weight:700; color:#cdd6f4; margin-bottom:6px;">2. Optional hint</div>
            <div class="nc-small">Command family or exact command.</div>
        </div>
        <div style="text-align:center; color:#a6adc8; font-size:1.5em;">→</div>
        <div class="nc-card" style="text-align:center; border-left:4px solid {COLORS['transformer']};">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:{COLORS['transformer']}; margin-bottom:8px;">Prediction</div>
            <div style="font-weight:700; color:#cdd6f4; margin-bottom:6px;">3. Model</div>
            <div class="nc-small">Predict the full next screen.</div>
        </div>
        <div style="text-align:center; color:#a6adc8; font-size:1.5em;">→</div>
        <div class="nc-card" style="text-align:center; border-left:4px solid {COLORS['changed']};">
            <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:{COLORS['changed']}; margin-bottom:8px;">Evaluation</div>
            <div style="font-weight:700; color:#cdd6f4; margin-bottom:6px;">4. Score</div>
            <div class="nc-small">Measure accuracy on cells that changed.</div>
        </div>
    </div>
    """
    mo.Html(impl_flow_html_comp)
    return


@app.cell(hide_code=True)
def _(mo):
    impl_tabs_comp = mo.ui.tabs(
        {
            "Data": mo.md(
                """
An episode is a sequence of terminal frames.

Each step contains:
- the current screen
- the action that happened (`type_char`, `enter`, `idle`, ...)
- the next screen

That makes the supervised learning target very concrete: **predict the next frame**.
"""
            ),
            "Model": mo.md(
                """
The simplest baseline is local:
- look at the current screen
- maybe include a hint about the command
- predict the next screen

Then compare that local baseline against sequence models like a Transformer or GRU.
"""
            ),
            "Metric": mo.md(
                """
Do **not** rely only on plain character accuracy.

Most of the screen usually stays the same, so a model can look good while missing the important part.

The better metric here is:
- **changed-cell accuracy** = accuracy restricted to cells that actually changed
"""
            ),
            "Minimal recipe": mo.md(
                """
```python
for episode in episodes:
    for t in range(len(frames) - 1):
        x = current_screen[t], optional_hint[t]
        y = next_screen[t + 1]
        loss = screen_prediction_loss(model(x), y)
```

Then evaluate on changed-cell accuracy, typing-only steps, and Enter-only steps.
"""
            ),
        }
    )
    impl_tabs_comp
    return


# ============================================================================
# LOAD PRECOMPUTED RESULTS
# ============================================================================
@app.cell(hide_code=True)
def _(ROOT, json, pd):
    comp_results_path = ROOT / "experiments" / "toy_nc_cli" / "results"
    comp_setting_keys = ["standard_family", "standard_command", "paraphrase_family", "paraphrase_command"]
    comp_setting_labels = {
        "standard_family": "Standard / family hint",
        "standard_command": "Standard / exact command",
        "paraphrase_family": "Paraphrase / family hint",
        "paraphrase_command": "Paraphrase / exact command",
    }
    comp_model_labels = {"mlp": "MLP", "transformer": "Transformer", "gru": "GRU"}
    comp_has_data = True
    comp_error_text = ""

    try:
        comp_mlp_json = json.loads((comp_results_path / "mlp_matched_results.json").read_text())
        comp_transformer_json = json.loads((comp_results_path / "transformer_results.json").read_text())
        comp_gru_json = json.loads((comp_results_path / "gru_results.json").read_text())
        comp_csv_df = pd.read_csv(comp_results_path / "baseline_comparison.csv")
    except Exception as comp_exc:
        comp_has_data = False
        comp_error_text = str(comp_exc)
        comp_mlp_json = {}
        comp_transformer_json = {}
        comp_gru_json = {}
        comp_csv_df = pd.DataFrame()

    if comp_has_data:
        comp_overall_map = {
            comp_model_name: [
                float(
                    comp_csv_df.loc[
                        (comp_csv_df["model"] == comp_model_name)
                        & (comp_csv_df["setting"] == comp_setting_key),
                        "changed_acc",
                    ].iloc[0]
                )
                for comp_setting_key in comp_setting_keys
            ]
            for comp_model_name in ["mlp", "transformer", "gru"]
        }
        comp_typing_map = {
            comp_model_name: [
                float(
                    comp_csv_df.loc[
                        (comp_csv_df["model"] == comp_model_name)
                        & (comp_csv_df["setting"] == comp_setting_key),
                        "typing_changed_acc",
                    ].iloc[0]
                )
                for comp_setting_key in comp_setting_keys
            ]
            for comp_model_name in ["mlp", "transformer", "gru"]
        }
        comp_enter_map = {
            comp_model_name: [
                float(
                    comp_csv_df.loc[
                        (comp_csv_df["model"] == comp_model_name)
                        & (comp_csv_df["setting"] == comp_setting_key),
                        "enter_changed_acc",
                    ].iloc[0]
                )
                for comp_setting_key in comp_setting_keys
            ]
            for comp_model_name in ["mlp", "transformer", "gru"]
        }
        comp_char_map = {
            comp_model_name: [
                float(
                    comp_csv_df.loc[
                        (comp_csv_df["model"] == comp_model_name)
                        & (comp_csv_df["setting"] == comp_setting_key),
                        "char_acc",
                    ].iloc[0]
                )
                for comp_setting_key in comp_setting_keys
            ]
            for comp_model_name in ["mlp", "transformer", "gru"]
        }

        comp_profile_means = {
            comp_model_name: {
                "overall": float(comp_csv_df.loc[comp_csv_df["model"] == comp_model_name, "changed_acc"].mean()),
                "typing": float(comp_csv_df.loc[comp_csv_df["model"] == comp_model_name, "typing_changed_acc"].mean()),
                "enter": float(comp_csv_df.loc[comp_csv_df["model"] == comp_model_name, "enter_changed_acc"].mean()),
                "char": float(comp_csv_df.loc[comp_csv_df["model"] == comp_model_name, "char_acc"].mean()),
            }
            for comp_model_name in ["mlp", "transformer", "gru"]
        }
        comp_profile_stds = {
            comp_model_name: {
                "overall": float(comp_csv_df.loc[comp_csv_df["model"] == comp_model_name, "changed_acc"].std()),
                "typing": float(comp_csv_df.loc[comp_csv_df["model"] == comp_model_name, "typing_changed_acc"].std()),
                "enter": float(comp_csv_df.loc[comp_csv_df["model"] == comp_model_name, "enter_changed_acc"].std()),
                "char": float(comp_csv_df.loc[comp_csv_df["model"] == comp_model_name, "char_acc"].std()),
            }
            for comp_model_name in ["mlp", "transformer", "gru"]
        }
        comp_transformer_enter_delta = [
            comp_tf_enter_val - comp_mlp_enter_val
            for comp_tf_enter_val, comp_mlp_enter_val in zip(
                comp_enter_map["transformer"], comp_enter_map["mlp"]
            )
        ]
        comp_transformer_enter_win_count = sum(comp_delta_val > 0 for comp_delta_val in comp_transformer_enter_delta)
        comp_mlp_overall_win_count = sum(
            comp_overall_map["mlp"][comp_index_val] > comp_overall_map["transformer"][comp_index_val]
            and comp_overall_map["mlp"][comp_index_val] > comp_overall_map["gru"][comp_index_val]
            for comp_index_val in range(len(comp_setting_keys))
        )

        comp_table_rows_html = ""
        for comp_setting_row in comp_setting_keys:
            comp_mlp_overall_row = float(
                comp_csv_df.loc[
                    (comp_csv_df["model"] == "mlp") & (comp_csv_df["setting"] == comp_setting_row),
                    "changed_acc",
                ].iloc[0]
            )
            comp_tf_overall_row = float(
                comp_csv_df.loc[
                    (comp_csv_df["model"] == "transformer") & (comp_csv_df["setting"] == comp_setting_row),
                    "changed_acc",
                ].iloc[0]
            )
            comp_gru_overall_row = float(
                comp_csv_df.loc[
                    (comp_csv_df["model"] == "gru") & (comp_csv_df["setting"] == comp_setting_row),
                    "changed_acc",
                ].iloc[0]
            )
            comp_mlp_enter_row = float(
                comp_csv_df.loc[
                    (comp_csv_df["model"] == "mlp") & (comp_csv_df["setting"] == comp_setting_row),
                    "enter_changed_acc",
                ].iloc[0]
            )
            comp_tf_enter_row = float(
                comp_csv_df.loc[
                    (comp_csv_df["model"] == "transformer") & (comp_csv_df["setting"] == comp_setting_row),
                    "enter_changed_acc",
                ].iloc[0]
            )
            comp_gru_enter_row = float(
                comp_csv_df.loc[
                    (comp_csv_df["model"] == "gru") & (comp_csv_df["setting"] == comp_setting_row),
                    "enter_changed_acc",
                ].iloc[0]
            )
            comp_overall_best_row = max(comp_mlp_overall_row, comp_tf_overall_row, comp_gru_overall_row)
            comp_enter_best_row = max(comp_mlp_enter_row, comp_tf_enter_row, comp_gru_enter_row)

            def comp_cell_style(comp_value_row, comp_best_row):
                if comp_value_row == comp_best_row:
                    return "background: rgba(34,197,94,0.18); color: #d9f99d; font-weight: 700;"
                return "color: #cdd6f4;"

            comp_table_rows_html += f"""
            <tr>
                <td style="color:#cdd6f4; font-weight:600;">{comp_setting_labels[comp_setting_row]}</td>
                <td style="{comp_cell_style(comp_mlp_overall_row, comp_overall_best_row)}">{100*comp_mlp_overall_row:.1f}%</td>
                <td style="{comp_cell_style(comp_tf_overall_row, comp_overall_best_row)}">{100*comp_tf_overall_row:.1f}%</td>
                <td style="{comp_cell_style(comp_gru_overall_row, comp_overall_best_row)}">{100*comp_gru_overall_row:.1f}%</td>
                <td style="{comp_cell_style(comp_mlp_enter_row, comp_enter_best_row)}">{100*comp_mlp_enter_row:.1f}%</td>
                <td style="{comp_cell_style(comp_tf_enter_row, comp_enter_best_row)}">{100*comp_tf_enter_row:.1f}%</td>
                <td style="{comp_cell_style(comp_gru_enter_row, comp_enter_best_row)}">{100*comp_gru_enter_row:.1f}%</td>
            </tr>
            """
    else:
        comp_overall_map = {}
        comp_typing_map = {}
        comp_enter_map = {}
        comp_char_map = {}
        comp_profile_means = {}
        comp_profile_stds = {}
        comp_transformer_enter_delta = []
        comp_transformer_enter_win_count = 0
        comp_mlp_overall_win_count = 0
        comp_table_rows_html = ""

    return (
        comp_char_map,
        comp_csv_df,
        comp_enter_map,
        comp_error_text,
        comp_gru_json,
        comp_has_data,
        comp_mlp_json,
        comp_mlp_overall_win_count,
        comp_model_labels,
        comp_overall_map,
        comp_profile_means,
        comp_profile_stds,
        comp_results_path,
        comp_setting_keys,
        comp_setting_labels,
        comp_table_rows_html,
        comp_transformer_enter_delta,
        comp_transformer_enter_win_count,
        comp_transformer_json,
        comp_typing_map,
    )


@app.cell(hide_code=True)
def _(comp_error_text, comp_has_data, comp_results_path, mo):
    data_status_output_comp = (
        mo.callout(
            mo.md(
                f"**Loaded benchmark files** from `{comp_results_path}`. The charts below use saved experiment results, not hand-entered demo values."
            ),
            kind="success",
        )
        if comp_has_data
        else mo.callout(
            mo.md(
                f"**Result files are missing or unreadable.** The notebook cannot make evidence-backed claims until the saved experiment files are present. Error: `{comp_error_text}`"
            ),
            kind="danger",
        )
    )
    data_status_output_comp
    return


# ============================================================================
# SHOWDOWN
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    showdown_heading_comp = mo.md(
        """
## The architecture showdown

Now for the main comparison: same toy task, same four evaluation settings, three different model families.
"""
    )
    showdown_heading_comp
    return


@app.cell(hide_code=True)
def _(COLORS, comp_has_data, comp_overall_map, comp_enter_map, np, plt, mo):
    if not comp_has_data:
        showdown_pair_output_comp = mo.md("")
    else:
        showdown_pair_fig = plt.figure(figsize=(14, 5))
        showdown_pair_axes = showdown_pair_fig.subplots(1, 2)
        showdown_pair_x = np.arange(4)
        showdown_pair_width = 0.25
        showdown_pair_labels = [
            "Standard\nfamily",
            "Standard\ncommand",
            "Paraphrase\nfamily",
            "Paraphrase\ncommand",
        ]

        showdown_pair_ax_left = showdown_pair_axes[0]
        showdown_pair_ax_left_mlp_bars = showdown_pair_ax_left.bar(
            showdown_pair_x - showdown_pair_width,
            comp_overall_map["mlp"],
            showdown_pair_width,
            label="MLP",
            color=COLORS["mlp"],
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        showdown_pair_ax_left_tf_bars = showdown_pair_ax_left.bar(
            showdown_pair_x,
            comp_overall_map["transformer"],
            showdown_pair_width,
            label="Transformer",
            color=COLORS["transformer"],
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        showdown_pair_ax_left_gru_bars = showdown_pair_ax_left.bar(
            showdown_pair_x + showdown_pair_width,
            comp_overall_map["gru"],
            showdown_pair_width,
            label="GRU",
            color=COLORS["gru"],
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        showdown_pair_ax_left.set_ylabel("Changed-cell accuracy", fontsize=12, fontweight="bold")
        showdown_pair_ax_left.set_title("Overall next-screen prediction", fontsize=14, fontweight="bold", pad=12)
        showdown_pair_ax_left.set_xticks(showdown_pair_x)
        showdown_pair_ax_left.set_xticklabels(showdown_pair_labels, fontsize=10)
        showdown_pair_ax_left.set_ylim(0, 1.08)
        showdown_pair_ax_left.spines["top"].set_visible(False)
        showdown_pair_ax_left.spines["right"].set_visible(False)
        showdown_pair_ax_left.axhline(y=0.5, color="#45475a", linestyle="--", alpha=0.5)
        showdown_pair_ax_left.legend(loc="upper right", fontsize=10)

        for showdown_pair_bar_left in list(showdown_pair_ax_left_mlp_bars) + list(showdown_pair_ax_left_tf_bars) + list(showdown_pair_ax_left_gru_bars):
            showdown_pair_height_left = showdown_pair_bar_left.get_height()
            showdown_pair_ax_left.text(
                showdown_pair_bar_left.get_x() + showdown_pair_bar_left.get_width() / 2,
                showdown_pair_height_left + 0.02,
                f"{100*showdown_pair_height_left:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        showdown_pair_ax_right = showdown_pair_axes[1]
        showdown_pair_ax_right_mlp_bars = showdown_pair_ax_right.bar(
            showdown_pair_x - showdown_pair_width,
            comp_enter_map["mlp"],
            showdown_pair_width,
            label="MLP",
            color=COLORS["mlp"],
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        showdown_pair_ax_right_tf_bars = showdown_pair_ax_right.bar(
            showdown_pair_x,
            comp_enter_map["transformer"],
            showdown_pair_width,
            label="Transformer",
            color=COLORS["transformer"],
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        showdown_pair_ax_right_gru_bars = showdown_pair_ax_right.bar(
            showdown_pair_x + showdown_pair_width,
            comp_enter_map["gru"],
            showdown_pair_width,
            label="GRU",
            color=COLORS["gru"],
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        showdown_pair_ax_right.set_ylabel("Enter-step changed-cell accuracy", fontsize=12, fontweight="bold")
        showdown_pair_ax_right.set_title("What happens after pressing Enter?", fontsize=14, fontweight="bold", pad=12)
        showdown_pair_ax_right.set_xticks(showdown_pair_x)
        showdown_pair_ax_right.set_xticklabels(showdown_pair_labels, fontsize=10)
        showdown_pair_ax_right.set_ylim(0, 1.08)
        showdown_pair_ax_right.spines["top"].set_visible(False)
        showdown_pair_ax_right.spines["right"].set_visible(False)
        showdown_pair_ax_right.axhline(y=0.5, color="#45475a", linestyle="--", alpha=0.5)
        showdown_pair_ax_right.legend(loc="upper right", fontsize=10)

        for showdown_pair_bar_right in list(showdown_pair_ax_right_mlp_bars) + list(showdown_pair_ax_right_tf_bars) + list(showdown_pair_ax_right_gru_bars):
            showdown_pair_height_right = showdown_pair_bar_right.get_height()
            showdown_pair_ax_right.text(
                showdown_pair_bar_right.get_x() + showdown_pair_bar_right.get_width() / 2,
                showdown_pair_height_right + 0.02,
                f"{100*showdown_pair_height_right:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        showdown_pair_fig.tight_layout()
        showdown_pair_output_comp = showdown_pair_fig

    showdown_pair_output_comp
    return


@app.cell(hide_code=True)
def _(COLORS, comp_has_data, comp_mlp_overall_win_count, comp_profile_means, comp_transformer_enter_win_count, mo):
    if not comp_has_data:
        discovery_output_comp = mo.md("")
    else:
        discovery_output_html_comp = f"""
        <div style="background: linear-gradient(135deg, {COLORS['transformer']}15 0%, {COLORS['mlp']}15 100%); border-radius: 20px; padding: 28px; margin: 24px 0; border: 2px solid {COLORS['transformer']}44;">
            <h3 style="color:#cdd6f4; text-align:center; margin-bottom:20px;">Main comparison outcome</h3>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-bottom: 18px;">
                <div class="nc-card" style="border-left:4px solid {COLORS['mlp']}; text-align:center;">
                    <div style="font-size:0.95em; color:#a6adc8; margin-bottom:6px;">Mean overall changed-cell accuracy</div>
                    <div style="font-size:2.2em; font-weight:800; color:{COLORS['mlp']};">{100*comp_profile_means['mlp']['overall']:.1f}%</div>
                    <div class="nc-small">MLP is best overall in <b>{comp_mlp_overall_win_count}/4</b> settings.</div>
                </div>
                <div class="nc-card" style="border-left:4px solid {COLORS['transformer']}; text-align:center;">
                    <div style="font-size:0.95em; color:#a6adc8; margin-bottom:6px;">Mean Enter-step accuracy</div>
                    <div style="font-size:2.2em; font-weight:800; color:{COLORS['transformer']};">{100*comp_profile_means['transformer']['enter']:.1f}%</div>
                    <div class="nc-small">Transformer beats MLP on Enter in <b>{comp_transformer_enter_win_count}/4</b> settings.</div>
                </div>
                <div class="nc-card" style="border-left:4px solid {COLORS['gru']}; text-align:center;">
                    <div style="font-size:0.95em; color:#a6adc8; margin-bottom:6px;">Mean overall changed-cell accuracy</div>
                    <div style="font-size:2.2em; font-weight:800; color:{COLORS['gru']};">{100*comp_profile_means['gru']['overall']:.1f}%</div>
                    <div class="nc-small">GRU was a negative result on this benchmark.</div>
                </div>
            </div>
            <div style="text-align:center; padding-top:16px; border-top:1px solid #45475a; color:#cdd6f4; line-height:1.7;">
                <b>The honest story:</b> the MLP is still the strongest toy baseline overall, but the Transformer is more competitive on the meaning-heavy Enter step.
                <br><span style="color:#a6adc8;">That advantage is real, but it is not universal: it is strongest in the standard setting and much smaller under paraphrase.</span>
            </div>
        </div>
        """
        discovery_output_comp = mo.Html(discovery_output_html_comp)

    discovery_output_comp
    return


@app.cell(hide_code=True)
def _(COLORS, comp_has_data, comp_transformer_enter_delta, comp_setting_keys, comp_setting_labels, plt, mo, np):
    if not comp_has_data:
        gap_chart_output_comp = mo.md("")
    else:
        gap_chart_fig_comp = plt.figure(figsize=(9, 3.8))
        gap_chart_ax_comp = gap_chart_fig_comp.subplots()
        gap_chart_x_comp = np.arange(len(comp_setting_keys))
        gap_chart_colors_comp = [COLORS["transformer"] if gap_val_comp > 0 else COLORS["wrong"] for gap_val_comp in comp_transformer_enter_delta]
        gap_chart_bars_comp = gap_chart_ax_comp.bar(
            gap_chart_x_comp,
            comp_transformer_enter_delta,
            color=gap_chart_colors_comp,
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        gap_chart_ax_comp.axhline(y=0, color="#a6adc8", linewidth=1.5)
        gap_chart_ax_comp.set_xticks(gap_chart_x_comp)
        gap_chart_ax_comp.set_xticklabels(
            [
                "Standard\nfamily",
                "Standard\ncommand",
                "Paraphrase\nfamily",
                "Paraphrase\ncommand",
            ],
            fontsize=10,
        )
        gap_chart_ax_comp.set_ylabel("Transformer Enter gain over MLP", fontsize=11, fontweight="bold")
        gap_chart_ax_comp.set_title("Where the Transformer helps on Enter", fontsize=14, fontweight="bold", pad=12)
        gap_chart_ax_comp.spines["top"].set_visible(False)
        gap_chart_ax_comp.spines["right"].set_visible(False)
        for gap_chart_bar_comp in gap_chart_bars_comp:
            gap_chart_height_comp = gap_chart_bar_comp.get_height()
            gap_chart_ax_comp.text(
                gap_chart_bar_comp.get_x() + gap_chart_bar_comp.get_width() / 2,
                gap_chart_height_comp + (0.01 if gap_chart_height_comp >= 0 else -0.04),
                f"{100*gap_chart_height_comp:+.1f}",
                ha="center",
                va="bottom" if gap_chart_height_comp >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )
        gap_chart_fig_comp.tight_layout()
        gap_chart_output_comp = gap_chart_fig_comp

    gap_chart_output_comp
    return


@app.cell(hide_code=True)
def _(mo):
    gap_note_comp = mo.callout(
        mo.md(
            """
**Read this chart carefully:** positive bars mean the Transformer is better than the MLP on Enter. The gain is huge in the easiest standard-family setting, modest in standard-command, and nearly disappears under paraphrase.
"""
        ),
        kind="info",
    )
    gap_note_comp
    return


# ============================================================================
# MODEL PROFILES
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    profile_heading_comp = mo.md(
        """
## Model profile

Below is a more honest summary than a single winner badge: mean performance across the four settings.
"""
    )
    profile_heading_comp
    return


@app.cell(hide_code=True)
def _(COLORS, comp_has_data, comp_profile_means, comp_profile_stds, np, plt, mo):
    if not comp_has_data:
        profile_output_comp = mo.md("")
    else:
        profile_fig_comp = plt.figure(figsize=(10, 4.8))
        profile_ax_comp = profile_fig_comp.subplots()
        profile_categories_comp = ["Overall", "Typing", "Enter"]
        profile_x_comp = np.arange(len(profile_categories_comp))
        profile_width_comp = 0.25

        profile_mlp_means_comp = [
            comp_profile_means["mlp"]["overall"],
            comp_profile_means["mlp"]["typing"],
            comp_profile_means["mlp"]["enter"],
        ]
        profile_transformer_means_comp = [
            comp_profile_means["transformer"]["overall"],
            comp_profile_means["transformer"]["typing"],
            comp_profile_means["transformer"]["enter"],
        ]
        profile_gru_means_comp = [
            comp_profile_means["gru"]["overall"],
            comp_profile_means["gru"]["typing"],
            comp_profile_means["gru"]["enter"],
        ]
        profile_mlp_stds_comp = [
            comp_profile_stds["mlp"]["overall"],
            comp_profile_stds["mlp"]["typing"],
            comp_profile_stds["mlp"]["enter"],
        ]
        profile_transformer_stds_comp = [
            comp_profile_stds["transformer"]["overall"],
            comp_profile_stds["transformer"]["typing"],
            comp_profile_stds["transformer"]["enter"],
        ]
        profile_gru_stds_comp = [
            comp_profile_stds["gru"]["overall"],
            comp_profile_stds["gru"]["typing"],
            comp_profile_stds["gru"]["enter"],
        ]

        profile_mlp_bars_comp = profile_ax_comp.bar(
            profile_x_comp - profile_width_comp,
            profile_mlp_means_comp,
            profile_width_comp,
            yerr=profile_mlp_stds_comp,
            capsize=6,
            label="MLP",
            color=COLORS["mlp"],
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        profile_transformer_bars_comp = profile_ax_comp.bar(
            profile_x_comp,
            profile_transformer_means_comp,
            profile_width_comp,
            yerr=profile_transformer_stds_comp,
            capsize=6,
            label="Transformer",
            color=COLORS["transformer"],
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        profile_gru_bars_comp = profile_ax_comp.bar(
            profile_x_comp + profile_width_comp,
            profile_gru_means_comp,
            profile_width_comp,
            yerr=profile_gru_stds_comp,
            capsize=6,
            label="GRU",
            color=COLORS["gru"],
            edgecolor="#1e1e2e",
            linewidth=2,
        )
        profile_ax_comp.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        profile_ax_comp.set_xticks(profile_x_comp)
        profile_ax_comp.set_xticklabels(profile_categories_comp, fontsize=11)
        profile_ax_comp.set_ylim(0, 1.08)
        profile_ax_comp.set_title("Mean accuracy across the four saved settings", fontsize=14, fontweight="bold", pad=12)
        profile_ax_comp.spines["top"].set_visible(False)
        profile_ax_comp.spines["right"].set_visible(False)
        profile_ax_comp.axhline(y=0.5, color="#45475a", linestyle="--", alpha=0.5)
        profile_ax_comp.legend(loc="upper right")

        for profile_bar_comp in list(profile_mlp_bars_comp) + list(profile_transformer_bars_comp) + list(profile_gru_bars_comp):
            profile_height_comp = profile_bar_comp.get_height()
            profile_ax_comp.text(
                profile_bar_comp.get_x() + profile_bar_comp.get_width() / 2,
                profile_height_comp + 0.03,
                f"{100*profile_height_comp:.1f}%",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

        profile_fig_comp.tight_layout()
        profile_output_comp = profile_fig_comp

    profile_output_comp
    return


@app.cell(hide_code=True)
def _(comp_has_data, comp_profile_means, mo):
    metric_trap_output_comp = (
        mo.callout(
            mo.md(
                f"""
**Metric trap:** all three models look similar on plain character accuracy — MLP {100*comp_profile_means['mlp']['char']:.1f}%, Transformer {100*comp_profile_means['transformer']['char']:.1f}%, GRU {100*comp_profile_means['gru']['char']:.1f}%.

But changed-cell accuracy tells the real story, because it ignores the easy unchanged background.
"""
            ),
            kind="warn",
        )
        if comp_has_data
        else mo.md("")
    )
    metric_trap_output_comp
    return


# ============================================================================
# TRADEOFF MAP
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    tradeoff_heading_comp = mo.md(
        """
## Mechanics vs. meaning

This view compresses the whole story into one picture:
- farther right = better at typing/mechanics
- higher up = better at Enter/meaning
- larger bubble = better overall changed-cell accuracy
"""
    )
    tradeoff_heading_comp
    return


@app.cell(hide_code=True)
def _(COLORS, comp_has_data, comp_model_labels, comp_profile_means, comp_profile_stds, plt, mo):
    if not comp_has_data:
        tradeoff_output_comp = mo.md("")
    else:
        tradeoff_fig_comp = plt.figure(figsize=(8.8, 6.0))
        tradeoff_ax_comp = tradeoff_fig_comp.subplots()
        tradeoff_ax_comp.set_xlim(0, 1.05)
        tradeoff_ax_comp.set_ylim(0, 1.0)
        tradeoff_ax_comp.set_xlabel("Typing accuracy (mechanics)", fontsize=12, fontweight="bold")
        tradeoff_ax_comp.set_ylabel("Enter accuracy (meaning)", fontsize=12, fontweight="bold")
        tradeoff_ax_comp.set_title("Where each architecture lives", fontsize=14, fontweight="bold", pad=12)
        tradeoff_ax_comp.spines["top"].set_visible(False)
        tradeoff_ax_comp.spines["right"].set_visible(False)
        tradeoff_ax_comp.grid(alpha=0.25)
        tradeoff_ax_comp.axvline(0.5, color="#45475a", linestyle="--", linewidth=1)
        tradeoff_ax_comp.axhline(0.5, color="#45475a", linestyle="--", linewidth=1)

        for tradeoff_model_key_comp, tradeoff_color_key_comp in [
            ("mlp", "mlp"),
            ("transformer", "transformer"),
            ("gru", "gru"),
        ]:
            tradeoff_x_comp = comp_profile_means[tradeoff_model_key_comp]["typing"]
            tradeoff_y_comp = comp_profile_means[tradeoff_model_key_comp]["enter"]
            tradeoff_size_comp = 2600 * comp_profile_means[tradeoff_model_key_comp]["overall"]
            tradeoff_ax_comp.scatter(
                [tradeoff_x_comp],
                [tradeoff_y_comp],
                s=tradeoff_size_comp,
                color=COLORS[tradeoff_color_key_comp],
                alpha=0.82,
                edgecolors="#e5e7eb",
                linewidths=1.5,
                zorder=3,
            )
            tradeoff_ax_comp.errorbar(
                tradeoff_x_comp,
                tradeoff_y_comp,
                xerr=comp_profile_stds[tradeoff_model_key_comp]["typing"],
                yerr=comp_profile_stds[tradeoff_model_key_comp]["enter"],
                fmt="none",
                ecolor="#cdd6f4",
                elinewidth=1,
                alpha=0.7,
                zorder=2,
            )
            tradeoff_ax_comp.text(
                tradeoff_x_comp,
                tradeoff_y_comp,
                f"{comp_model_labels[tradeoff_model_key_comp]}\n{100*comp_profile_means[tradeoff_model_key_comp]['overall']:.1f}% overall",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#111827",
                zorder=4,
            )

        tradeoff_ax_comp.text(0.97, 0.06, "easy mechanics →", ha="right", color="#7dd3fc", fontsize=9)
        tradeoff_ax_comp.text(0.05, 0.95, "↑ harder semantic success", va="top", color="#f9a8d4", fontsize=9)
        tradeoff_fig_comp.tight_layout()
        tradeoff_output_comp = tradeoff_fig_comp

    tradeoff_output_comp
    return


@app.cell(hide_code=True)
def _(mo):
    tradeoff_note_comp = mo.callout(
        mo.md(
            """
**Why this chart matters:** the MLP sits farther right because it is strongest at local mechanics. The Transformer sits higher because it is relatively stronger on Enter. The GRU stays in the lower-left corner.
"""
        ),
        kind="success",
    )
    tradeoff_note_comp
    return


# ============================================================================
# EXACT RESULTS TABLE
# ============================================================================
@app.cell(hide_code=True)
def _(comp_has_data, comp_table_rows_html, mo):
    if not comp_has_data:
        exact_table_output_comp = mo.md("")
    else:
        exact_table_html_comp = f"""
        <div class="nc-card" style="padding: 0; overflow: hidden; margin-top: 12px;">
            <table class="nc-table">
                <thead>
                    <tr>
                        <th rowspan="2">Setting</th>
                        <th colspan="3">Overall changed-cell accuracy</th>
                        <th colspan="3">Enter-step changed-cell accuracy</th>
                    </tr>
                    <tr>
                        <th>MLP</th>
                        <th>Transformer</th>
                        <th>GRU</th>
                        <th>MLP</th>
                        <th>Transformer</th>
                        <th>GRU</th>
                    </tr>
                </thead>
                <tbody>
                    {comp_table_rows_html}
                </tbody>
            </table>
        </div>
        """
        exact_table_output_comp = mo.accordion(
            {
                "See the exact benchmark numbers": mo.vstack(
                    [
                        mo.md("Best cell in each metric block is highlighted. Numbers come directly from the saved JSON/CSV results."),
                        mo.Html(exact_table_html_comp),
                    ]
                )
            },
            multiple=True,
        )

    exact_table_output_comp
    return


# ============================================================================
# TAKEAWAYS
# ============================================================================
@app.cell(hide_code=True)
def _(mo):
    takeaway_heading_comp = mo.md(
        """
## Takeaway

Here is the clearest evidence-backed story this toy benchmark supports.
"""
    )
    takeaway_heading_comp
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    takeaway_html_comp = f"""
    <div style="background: linear-gradient(135deg, {COLORS['bg_dark']} 0%, #2d1b4e 50%, {COLORS['bg_dark']} 100%); border-radius: 24px; padding: 40px; margin: 24px 0; position: relative; overflow: hidden;">
        <div style="position:absolute; inset:0; background: radial-gradient(circle at 20% 80%, rgba(34,197,94,0.10) 0%, transparent 45%), radial-gradient(circle at 80% 20%, rgba(168,85,247,0.10) 0%, transparent 45%);"></div>
        <div style="position:relative; z-index:1; display:grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
            <div style="text-align:center; padding: 18px;">
                <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:{COLORS['mlp']}; margin-bottom:12px;">01</div>
                <div style="color:{COLORS['mlp']}; font-weight:800; font-size:1.2em; margin-bottom:8px;">Best overall model</div>
                <div style="color:#a6adc8; font-size:0.95em;">The MLP is still the strongest baseline on the main benchmark metric.</div>
            </div>
            <div style="text-align:center; padding: 18px;">
                <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:{COLORS['transformer']}; margin-bottom:12px;">02</div>
                <div style="color:{COLORS['transformer']}; font-weight:800; font-size:1.2em; margin-bottom:8px;">Most interesting contrast</div>
                <div style="color:#a6adc8; font-size:0.95em;">The Transformer is relatively stronger on meaning-heavy Enter steps.</div>
            </div>
            <div style="text-align:center; padding: 18px;">
                <div style="font-size:0.82em; letter-spacing:0.06em; text-transform:uppercase; color:{COLORS['wrong']}; margin-bottom:12px;">03</div>
                <div style="color:{COLORS['wrong']}; font-weight:800; font-size:1.2em; margin-bottom:8px;">Important limit</div>
                <div style="color:#a6adc8; font-size:0.95em;">That Transformer gain shrinks sharply under paraphrase, so the toy task still does not show deep semantic generalization.</div>
            </div>
        </div>
    </div>
    """
    mo.Html(takeaway_html_comp)
    return


@app.cell(hide_code=True)
def _(mo):
    takeaway_accordion_comp = mo.accordion(
        {
            "What this says about model design": mo.md(
                """
- **MLP** is a strong baseline when the task is mostly local and mechanical.
- **Transformer** becomes interesting when the task depends more on global context or command semantics.
- **GRU** did not help here, so “add recurrence” is not enough by itself.
                """
            ),
            "What this says about the Neural Computers idea": mo.md(
                """
The full Neural Computers paper works at a much larger scale: real computer recordings, video generation models, and much more data.

This toy notebook does **not** replicate that paper. Instead, it builds intuition for a smaller question: what can a model learn from screen transitions alone, and where does that break?
                """
            ),
            "What to try next": mo.md(
                """
1. Use a stronger semantic task than fixed toy commands.
2. Add true held-out semantic generalization, not just phrasing variants.
3. Test a hybrid story: an MLP-style local mechanism plus a Transformer-style global context path.
                """
            ),
        },
        multiple=True,
    )
    takeaway_accordion_comp
    return


@app.cell(hide_code=True)
def _(mo):
    footer_md_comp = mo.md(
        """
---
<div style="text-align:center; color:#6b7280; padding: 32px;">
    <p style="margin-bottom: 10px;">Built with <a href="https://marimo.io" style="color:#89b4fa;">marimo</a></p>
    <p style="margin-bottom: 10px;">Inspired by <a href="https://arxiv.org/abs/2604.06425" style="color:#89b4fa;">Neural Computers</a> (Zhuge et al., 2026)</p>
    <p style="font-size:0.9em; color:#45475a;">This competition notebook uses saved local experiment artifacts for reproducible presentation.</p>
</div>
"""
    )
    footer_md_comp
    return


if __name__ == "__main__":
    app.run()
