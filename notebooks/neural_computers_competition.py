# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///
"""Competition notebook: tiny-neural-os.

A gallery-style, evidence-first notebook inspired by Neural Computers (2026).
Uses precomputed local benchmark files for reproducible presentation.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import html
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "experiments" / "toy_nc_cli" / "src"))

    from toy_terminal import COMMAND_VARIANTS, TerminalConfig, generate_episodes

    SETTING_ORDER = [
        "standard_family",
        "standard_command",
        "paraphrase_family",
        "paraphrase_command",
    ]

    SETTING_LABELS = {
        "standard_family": "Standard · family hint",
        "standard_command": "Standard · exact command",
        "paraphrase_family": "Paraphrase · family hint",
        "paraphrase_command": "Paraphrase · exact command",
    }

    MODEL_LABELS = {
        "mlp": "MLP",
        "transformer": "Transformer",
        "gru": "GRU",
    }
    MODEL_LABEL_TO_KEY = {label: key for key, label in MODEL_LABELS.items()}
    SETTING_LABEL_TO_KEY = {label: key for key, label in SETTING_LABELS.items()}

    MODEL_COLORS = {
        "MLP": "#22c55e",
        "Transformer": "#7c3aed",
        "GRU": "#ef4444",
    }

    METRIC_OPTIONS = {
        "Overall changed-cell accuracy": "changed_acc",
        "Enter changed-cell accuracy": "enter_changed_acc",
        "Typing changed-cell accuracy": "typing_changed_acc",
        "Character accuracy (easy metric)": "char_acc",
    }

    def pct(value: float) -> str:
        return f"{100 * value:.1f}%"

    def load_baseline_csv() -> tuple[bool, pd.DataFrame, str, Path]:
        results_dir = ROOT / "experiments" / "toy_nc_cli" / "results"
        csv_path = results_dir / "baseline_comparison.csv"
        try:
            df = pd.read_csv(csv_path)
            required_cols = {
                "model",
                "setting",
                "char_acc",
                "changed_acc",
                "typing_changed_acc",
                "enter_changed_acc",
            }
            missing = sorted(required_cols - set(df.columns))
            if missing:
                return False, pd.DataFrame(), f"Missing CSV columns: {missing}", results_dir
            return True, df, "", results_dir
        except Exception as exc:  # pragma: no cover - presentation fallback
            return False, pd.DataFrame(), str(exc), results_dir

    def terminal_html(frame: np.ndarray, changed_mask: np.ndarray, title: str) -> str:
        rows = []
        for r in range(frame.shape[0]):
            cells = []
            for c in range(frame.shape[1]):
                ch = frame[r, c]
                safe = "&nbsp;" if ch == " " else html.escape(ch)
                if changed_mask[r, c]:
                    cells.append(
                        f"<span style='background:#fef3c7;color:#111827;border-radius:2px;padding:0 1px'>{safe}</span>"
                    )
                else:
                    cells.append(f"<span>{safe}</span>")
            rows.append("".join(cells))

        return f"""
        <div class='nc-shell'>
            <div class='nc-shell-title'>{title}</div>
            {'<br>'.join(rows)}
        </div>
        """

    def metric_map_for_setting(
        benchmark_df: pd.DataFrame,
        setting_key: str,
        metric_key: str,
    ) -> dict[str, float]:
        subset = benchmark_df[benchmark_df["setting"] == setting_key].set_index("model")
        return {
            model_key: float(subset.loc[model_key, metric_key])
            for model_key in ["mlp", "transformer", "gru"]
        }

    def trim_code(text: str, max_lines: int = 160) -> str:
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text
        return "\n".join(lines[:max_lines] + ["", "# ... truncated for notebook display ..."])


@app.cell(hide_code=True)
def _():
    mo.Html(
        """
        <style>
            :root {
                --nc-ink: #0f172a;
            }

            .nc-wrap {
                max-width: 1120px;
                margin: 0 auto;
                padding: 0 28px;
            }

            .nc-hero {
                width: 100vw;
                margin-left: calc(50% - 50vw);
                margin-bottom: 2rem;
                background:
                    radial-gradient(1100px 480px at 10% 0%, rgba(59,130,246,0.24), transparent 55%),
                    radial-gradient(900px 440px at 90% 0%, rgba(124,58,237,0.2), transparent 55%),
                    #0b1220;
                color: #e2e8f0;
                border-bottom: 1px solid rgba(148,163,184,0.25);
            }

            .nc-hero-grid {
                display: grid;
                grid-template-columns: 1.1fr 0.9fr;
                gap: 2.4rem;
                align-items: center;
                padding: 68px 0 58px;
            }

            .nc-kicker {
                display: inline-block;
                letter-spacing: 0.07em;
                text-transform: uppercase;
                font-size: 0.75rem;
                color: #93c5fd;
                margin-bottom: 0.95rem;
                font-weight: 700;
            }

            .nc-hero h1 {
                color: #f8fafc;
                font-size: clamp(2rem, 4vw, 3.35rem);
                line-height: 1.06;
                margin: 0 0 1rem;
                letter-spacing: -0.025em;
            }

            .nc-hero p {
                color: #cbd5e1;
                font-size: 1.08rem;
                line-height: 1.7;
                max-width: 60ch;
                margin: 0;
            }

            .nc-meta {
                margin-top: 1.2rem;
                color: #bfdbfe;
                font-size: 0.95rem;
            }

            .nc-hero-shell {
                border: 1px solid rgba(148, 163, 184, 0.3);
                border-radius: 14px;
                background: #0f172a;
                color: #e2e8f0;
                padding: 16px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                line-height: 1.45;
                font-size: 0.92rem;
                box-shadow: 0 22px 52px rgba(2, 6, 23, 0.45);
            }

            .nc-shell {
                border: 1px solid #1e293b;
                border-radius: 14px;
                background: #0f172a;
                color: #e2e8f0;
                padding: 14px;
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                line-height: 1.24;
                font-size: 0.86rem;
                overflow-x: auto;
            }

            .nc-shell-title {
                color: #93c5fd;
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 10px;
            }

            @media (max-width: 920px) {
                .nc-hero-grid {
                    grid-template-columns: 1fr;
                    padding: 56px 0 48px;
                }

                .nc-wrap {
                    padding: 0 20px;
                }
            }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.Html(
        """
        <section class="nc-hero">
            <div class="nc-wrap nc-hero-grid">
                <div>
                    <div class="nc-kicker">tiny-neural-os · marimo research notebook</div>
                    <h1>Can a model learn how a computer works from screen transitions alone?</h1>
                    <p>
                        This notebook turns the core idea from <em>Neural Computers</em> (2026)
                        into a compact, reproducible benchmark: predict the next terminal screen
                        from the current one, then separate mechanical typing from command meaning.
                    </p>
                    <div class="nc-meta">
                        Source paper:
                        <a href="https://arxiv.org/abs/2604.06425" style="color:#93c5fd; text-decoration:none; font-weight:600;">
                            Neural Computers (arXiv:2604.06425)
                        </a>
                    </div>
                </div>
                <pre class="nc-hero-shell">$ whoami
    researcher
    $ pwd
    /home/research
    $ python -c "print(19*3)"
    57
    $ █</pre>
            </div>
        </section>
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### What this notebook does

    - Builds intuition with a live toy terminal.
    - Uses **saved benchmark files** (no hand-entered demo numbers).
    - Compares MLP, Transformer, and GRU under matched settings.
    - Separates easy mechanics (typing) from harder semantic updates (Enter).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1) Watch the toy computer

    Use this interactive toy terminal to see why the task is asymmetric: typing often changes a tiny local patch, while Enter can trigger larger, meaning-dependent updates.
    """)
    return


@app.cell(hide_code=True)
def _():
    family_picker = mo.ui.dropdown(
        options=list(COMMAND_VARIANTS.keys()),
        value="whoami",
        label="Command family",
    )
    noise_toggle = mo.ui.switch(value=False, label="Add output noise")
    return family_picker, noise_toggle


@app.cell(hide_code=True)
def _(family_picker):
    phrasing_options = [
        f"Phrasing {i + 1}" for i in range(len(COMMAND_VARIANTS[family_picker.value]))
    ]
    phrasing_picker = mo.ui.dropdown(
        options=phrasing_options,
        value=phrasing_options[0],
        label="Command phrasing",
    )
    return (phrasing_picker,)


@app.cell(hide_code=True)
def _(family_picker, noise_toggle, phrasing_picker):
    controls = mo.hstack(
        [family_picker, phrasing_picker, noise_toggle],
        widths=[2.4, 2.4, 1.2],
        gap=1,
        align="end",
    )
    controls
    return


@app.cell(hide_code=True)
def _(family_picker, noise_toggle, phrasing_picker):
    toy_config = TerminalConfig(rows=10, cols=40, context_width=32, patch_radius=1)
    variant_idx = int(phrasing_picker.value.split()[-1]) - 1

    episode = generate_episodes(
        n=1,
        config=toy_config,
        noisy=noise_toggle.value,
        seed=77 + variant_idx,
        context_mode="command",
        families=[family_picker.value],
        variant_indices_by_family={family_picker.value: (variant_idx,)},
    )[0]

    max_step = len(episode.actions)
    return episode, max_step


@app.cell(hide_code=True)
def _(max_step):
    step_slider = mo.ui.slider(
        0,
        max_step,
        value=0,
        step=1,
        label="Step through time",
        full_width=True,
    )
    return (step_slider,)


@app.cell(hide_code=True)
def _(step_slider):
    step_slider
    return


@app.cell(hide_code=True)
def _(episode, step_slider):
    step_value = step_slider.value
    frame = episode.frames[step_value]

    if step_value == 0:
        prev_frame = frame.copy()
        changed_mask = np.zeros_like(frame, dtype=bool)
        action_label = "Start state"
        action_group = "Idle"
    else:
        prev_frame = episode.frames[step_value - 1]
        changed_mask = frame != prev_frame
        action = episode.actions[step_value - 1]
        if action.kind == "type_char":
            action_label = f"Type {action.typed_char!r}"
            action_group = "Local mechanics"
        elif action.kind == "enter":
            action_label = "Press Enter"
            action_group = "Meaning-heavy update"
        elif action.kind == "backspace":
            action_label = "Backspace"
            action_group = "Local mechanics"
        else:
            action_label = action.kind
            action_group = "Other"

    changed_count = int(changed_mask.sum())
    empty_mask = np.zeros_like(frame, dtype=bool)

    previous_panel = mo.Html(
        terminal_html(
            prev_frame,
            empty_mask,
            "Before step",
        )
    )
    current_panel = mo.Html(
        terminal_html(
            frame,
            changed_mask,
            f"After step · {episode.command_text}",
        )
    )

    terminal_compare = mo.hstack(
        [previous_panel, current_panel],
        widths=[1, 1],
        gap=1,
    )
    stats = mo.hstack(
        [
            mo.stat(label="Current action", value=action_label, caption=action_group),
            mo.stat(label="Changed cells", value=str(changed_count), caption="Cells highlighted on the right"),
            mo.stat(label="Step", value=f"{step_value}/{len(episode.actions)}", caption=f"Command: {episode.command_text}"),
        ],
        widths="equal",
        gap=1,
    )

    toy_panel_view = mo.vstack([terminal_compare, stats], gap=1)
    toy_panel_view
    return action_group, action_label, changed_count


@app.cell(hide_code=True)
def _(action_group, action_label, changed_count):
    if action_group == "Meaning-heavy update":
        note = mo.callout(
            mo.md(
                f"**{action_label}** changes **{changed_count} cells** here. This is the harder regime: one action fans out into a larger screen update, so the model needs more than local typing mechanics."
            ),
            kind="warn",
        )
    elif action_group == "Local mechanics":
        note = mo.callout(
            mo.md(
                f"**{action_label}** changes **{changed_count} cells** here. This is the easier regime: the update is local and mostly mechanical, which is why simple baselines can already look strong."
            ),
            kind="info",
        )
    else:
        note = mo.callout(
            mo.md(
                "**Why this matters:** ordinary character accuracy can look high even when a model misses the important transition. That is why the benchmark centers changed-cell accuracy."
            ),
            kind="info",
        )

    note
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2) Benchmark protocol (plain English)

    | Item | Protocol |
    |---|---|
    | Task | Predict the next terminal frame from the current frame. |
    | Main metric | **Changed-cell accuracy** (only cells that changed are scored). |
    | Settings | Standard vs. paraphrase commands, each with family vs. exact command hints. |
    | Models compared | MLP, Transformer, GRU on the same saved benchmark settings. |
    """)
    return


@app.cell(hide_code=True)
def _():
    has_results, results_df, results_error, results_dir = load_baseline_csv()
    return has_results, results_df, results_dir, results_error


@app.cell(hide_code=True)
def _(has_results, results_dir, results_error):
    status_view = (
        mo.callout(
            mo.md(
                f"Loaded benchmark CSV from `{results_dir}`. All plots below are computed from this saved file."
            ),
            kind="success",
        )
        if has_results
        else mo.callout(
            mo.md(
                f"Could not load benchmark files from `{results_dir}`. Error: `{results_error}`"
            ),
            kind="danger",
        )
    )
    status_view
    return


@app.cell
def _(has_results, results_df):
    if not has_results:
        long_df = pd.DataFrame()
        profile_df = pd.DataFrame()
        enter_delta_df = pd.DataFrame()
        summary_row = {}
        exact_table_df = pd.DataFrame()
    else:
        long_df = results_df.copy()
        long_df["setting_label"] = long_df["setting"].map(SETTING_LABELS)
        long_df["setting_rank"] = long_df["setting"].map(
            {setting: idx for idx, setting in enumerate(SETTING_ORDER)}
        )
        long_df["model_label"] = long_df["model"].map(MODEL_LABELS)

        profile_df = (
            long_df.groupby(["model", "model_label"], as_index=False)
            .agg(
                overall_mean=("changed_acc", "mean"),
                typing_mean=("typing_changed_acc", "mean"),
                enter_mean=("enter_changed_acc", "mean"),
                char_mean=("char_acc", "mean"),
            )
            .sort_values("overall_mean", ascending=False)
        )

        overall_pivot = (
            long_df.pivot(index="setting", columns="model", values="changed_acc")
            .reindex(SETTING_ORDER)
        )
        enter_pivot = (
            long_df.pivot(index="setting", columns="model", values="enter_changed_acc")
            .reindex(SETTING_ORDER)
        )

        enter_delta_df = pd.DataFrame(
            {
                "setting": SETTING_ORDER,
                "setting_label": [SETTING_LABELS[s] for s in SETTING_ORDER],
                "delta": (enter_pivot["transformer"] - enter_pivot["mlp"]).values,
            }
        )
        enter_delta_df["direction"] = np.where(
            enter_delta_df["delta"] >= 0,
            "Transformer ahead",
            "MLP ahead",
        )

        mlp_overall_wins = int(
            (
                (overall_pivot["mlp"] > overall_pivot["transformer"])
                & (overall_pivot["mlp"] > overall_pivot["gru"])
            ).sum()
        )
        transformer_enter_wins = int((enter_pivot["transformer"] > enter_pivot["mlp"]).sum())

        summary_row = {
            "mlp_overall": float(
                profile_df.loc[profile_df["model"] == "mlp", "overall_mean"].iloc[0]
            ),
            "transformer_enter": float(
                profile_df.loc[profile_df["model"] == "transformer", "enter_mean"].iloc[0]
            ),
            "gru_overall": float(
                profile_df.loc[profile_df["model"] == "gru", "overall_mean"].iloc[0]
            ),
            "mlp_overall_wins": mlp_overall_wins,
            "transformer_enter_wins": transformer_enter_wins,
        }

        rows = []
        for setting in SETTING_ORDER:
            this_overall = {
                model: float(overall_pivot.loc[setting, model])
                for model in ["mlp", "transformer", "gru"]
            }
            this_enter = {
                model: float(enter_pivot.loc[setting, model])
                for model in ["mlp", "transformer", "gru"]
            }
            best_overall = max(this_overall.values())
            best_enter = max(this_enter.values())

            def fmt(v: float, best: float) -> str:
                return f"{100*v:.1f}%{' ★' if np.isclose(v, best) else ''}"

            rows.append(
                {
                    "Setting": SETTING_LABELS[setting],
                    "Overall · MLP": fmt(this_overall["mlp"], best_overall),
                    "Overall · Transformer": fmt(this_overall["transformer"], best_overall),
                    "Overall · GRU": fmt(this_overall["gru"], best_overall),
                    "Enter · MLP": fmt(this_enter["mlp"], best_enter),
                    "Enter · Transformer": fmt(this_enter["transformer"], best_enter),
                    "Enter · GRU": fmt(this_enter["gru"], best_enter),
                }
            )

        exact_table_df = pd.DataFrame(rows)
    return exact_table_df, long_df, profile_df, summary_row


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 3) Benchmark explorer

    This part is the real notebook workspace: change one control, and the benchmark views update together.
    """)
    return


@app.cell(hide_code=True)
def _():
    metric_picker = mo.ui.dropdown(
        options=list(METRIC_OPTIONS.keys()),
        value="Overall changed-cell accuracy",
        label="Main metric",
    )
    focus_setting_picker = mo.ui.dropdown(
        options=[SETTING_LABELS[s] for s in SETTING_ORDER],
        value=SETTING_LABELS["standard_family"],
        label="Focus setting",
    )
    benchmark_controls = mo.hstack(
        [metric_picker, focus_setting_picker],
        widths=[2, 2],
        gap=1,
        align="end",
    )
    benchmark_controls
    return focus_setting_picker, metric_picker


@app.cell(hide_code=True)
def _(has_results, long_df, metric_picker):
    benchmark_overview_view = mo.md("")

    if has_results:
        _metric_col = METRIC_OPTIONS[metric_picker.value]
        _x = np.arange(len(SETTING_ORDER))
        _width = 0.24

        _model_specs = [
            ("mlp", "MLP", MODEL_COLORS["MLP"], -_width),
            ("transformer", "Transformer", MODEL_COLORS["Transformer"], 0.0),
            ("gru", "GRU", MODEL_COLORS["GRU"], _width),
        ]

        _fig, _ax = plt.subplots(figsize=(12.6, 4.9))

        for _model_key, _model_label, _model_color, _offset in _model_specs:
            _vals = [
                float(
                    long_df.loc[
                        (long_df["model"] == _model_key) & (long_df["setting"] == _setting_key),
                        _metric_col,
                    ].iloc[0]
                )
                for _setting_key in SETTING_ORDER
            ]
            _bars = _ax.bar(
                _x + _offset,
                _vals,
                _width,
                label=_model_label,
                color=_model_color,
                edgecolor="#ffffff",
                linewidth=1.8,
            )
            for _bar in _bars:
                _h = _bar.get_height()
                _ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _h + 0.016,
                    f"{100*_h:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#0f172a",
                )

        _ax.set_ylim(0, 1.08)
        _ax.set_ylabel(metric_picker.value, fontweight="bold")
        _ax.set_xticks(_x)
        _ax.set_xticklabels(
            [
                "Standard\nfamily",
                "Standard\ncommand",
                "Paraphrase\nfamily",
                "Paraphrase\ncommand",
            ],
            fontsize=10,
        )
        _ax.grid(axis="y", color="#e5e7eb", linewidth=1)
        _ax.set_axisbelow(True)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.spines["left"].set_color("#cbd5e1")
        _ax.spines["bottom"].set_color("#cbd5e1")
        _ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        _fig.tight_layout()

        benchmark_overview_view = mo.mpl.interactive(_fig)
    return (benchmark_overview_view,)


@app.cell(hide_code=True)
def _(has_results, summary_row):
    benchmark_summary_view = mo.md("")

    if has_results:
        benchmark_summary_view = mo.hstack(
            [
                mo.stat(
                    label="MLP mean overall",
                    value=pct(summary_row["mlp_overall"]),
                    caption=f"Best overall in {summary_row['mlp_overall_wins']}/4 settings",
                ),
                mo.stat(
                    label="Transformer mean Enter",
                    value=pct(summary_row["transformer_enter"]),
                    caption=f"Beats MLP on Enter in {summary_row['transformer_enter_wins']}/4 settings",
                ),
                mo.stat(
                    label="GRU mean overall",
                    value=pct(summary_row["gru_overall"]),
                    caption="Negative result in this benchmark",
                ),
            ],
            widths="equal",
            gap=1,
        )
    return (benchmark_summary_view,)


@app.cell(hide_code=True)
def _(focus_setting_picker, has_results, long_df, metric_picker):
    focused_setting_view = mo.md("")

    if has_results:
        _setting_key = SETTING_LABEL_TO_KEY[focus_setting_picker.value]
        _metric_key = METRIC_OPTIONS[metric_picker.value]
        _metric_values = metric_map_for_setting(long_df, _setting_key, _metric_key)
        _overall_values = metric_map_for_setting(long_df, _setting_key, "changed_acc")
        _enter_values = metric_map_for_setting(long_df, _setting_key, "enter_changed_acc")
        _typing_values = metric_map_for_setting(long_df, _setting_key, "typing_changed_acc")

        _best_metric_model = max(_metric_values, key=_metric_values.get)
        _best_overall_model = max(_overall_values, key=_overall_values.get)
        _best_enter_model = max(_enter_values, key=_enter_values.get)

        _story = mo.callout(
            mo.md(
                f"""
    **{focus_setting_picker.value}.** On the selected main metric (**{metric_picker.value}**), **{MODEL_LABELS[_best_metric_model]}** leads at **{pct(_metric_values[_best_metric_model])}**.
    Overall, this setting is won by **{MODEL_LABELS[_best_overall_model]}** at **{pct(_overall_values[_best_overall_model])}**.
    On Enter-only steps, **{MODEL_LABELS[_best_enter_model]}** is best at **{pct(_enter_values[_best_enter_model])}**.
    """
            ),
            kind="info",
        )

        _cards = mo.hstack(
            [
                mo.stat(
                    label="MLP",
                    value=pct(_overall_values["mlp"]),
                    caption=f"Enter {pct(_enter_values['mlp'])} · Typing {pct(_typing_values['mlp'])}",
                ),
                mo.stat(
                    label="Transformer",
                    value=pct(_overall_values["transformer"]),
                    caption=(
                        f"Enter {pct(_enter_values['transformer'])} · "
                        f"Typing {pct(_typing_values['transformer'])}"
                    ),
                ),
                mo.stat(
                    label="GRU",
                    value=pct(_overall_values["gru"]),
                    caption=f"Enter {pct(_enter_values['gru'])} · Typing {pct(_typing_values['gru'])}",
                ),
            ],
            widths="equal",
            gap=1,
        )

        focused_setting_view = mo.vstack([_story, _cards], gap=1)
    return (focused_setting_view,)


@app.cell(hide_code=True)
def _():
    reference_model_picker = mo.ui.dropdown(
        options=list(MODEL_LABEL_TO_KEY.keys()),
        value="MLP",
        label="Reference model",
    )
    comparison_model_picker = mo.ui.dropdown(
        options=list(MODEL_LABEL_TO_KEY.keys()),
        value="Transformer",
        label="Comparison model",
    )
    return comparison_model_picker, reference_model_picker


@app.cell(hide_code=True)
def _(
    comparison_model_picker,
    has_results,
    long_df,
    metric_picker,
    reference_model_picker,
):
    pair_delta_view = mo.md("")

    _pair_controls = mo.hstack(
        [reference_model_picker, comparison_model_picker],
        widths=[1, 1],
        gap=1,
        align="end",
    )

    if has_results:
        if reference_model_picker.value == comparison_model_picker.value:
            _note = mo.callout(
                mo.md("Choose two different models to compare their gain across settings."),
                kind="warn",
            )
            pair_delta_view = mo.vstack([_pair_controls, _note])
        else:
            _left_key = MODEL_LABEL_TO_KEY[reference_model_picker.value]
            _right_key = MODEL_LABEL_TO_KEY[comparison_model_picker.value]
            _metric_key = METRIC_OPTIONS[metric_picker.value]
            _left_vals = [
                float(
                    long_df.loc[
                        (long_df["model"] == _left_key) & (long_df["setting"] == _setting_key),
                        _metric_key,
                    ].iloc[0]
                )
                for _setting_key in SETTING_ORDER
            ]
            _right_vals = [
                float(
                    long_df.loc[
                        (long_df["model"] == _right_key) & (long_df["setting"] == _setting_key),
                        _metric_key,
                    ].iloc[0]
                )
                for _setting_key in SETTING_ORDER
            ]
            _delta = np.array(_right_vals) - np.array(_left_vals)
            _colors = ["#7c3aed" if _val >= 0 else "#dc2626" for _val in _delta]

            _fig, _ax = plt.subplots(figsize=(12.2, 3.8))
            _bars = _ax.bar(_x := np.arange(len(SETTING_ORDER)), _delta, color=_colors, edgecolor="#ffffff", linewidth=1.8)
            _ax.axhline(0.0, color="#64748b", linewidth=1.2)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(
                [
                    "Standard\nfamily",
                    "Standard\ncommand",
                    "Paraphrase\nfamily",
                    "Paraphrase\ncommand",
                ],
                fontsize=10,
            )
            _ax.set_ylabel(
                f"{comparison_model_picker.value} gain over {reference_model_picker.value}",
                fontweight="bold",
            )
            _ax.grid(axis="y", color="#e5e7eb", linewidth=1)
            _ax.set_axisbelow(True)
            _ax.spines["top"].set_visible(False)
            _ax.spines["right"].set_visible(False)
            _ax.spines["left"].set_color("#cbd5e1")
            _ax.spines["bottom"].set_color("#cbd5e1")

            for _bar, _val in zip(_bars, _delta):
                _ax.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _val + (0.012 if _val >= 0 else -0.02),
                    f"{100*_val:+.1f}",
                    ha="center",
                    va="bottom" if _val >= 0 else "top",
                    fontsize=9,
                    fontweight="bold",
                )

            _fig.tight_layout()
            _pair_story = mo.callout(
                mo.md(
                    f"The chart below shows **{comparison_model_picker.value} − {reference_model_picker.value}** on **{metric_picker.value}**. Positive bars mean the comparison model is ahead in that setting."
                ),
                kind="info",
            )
            pair_delta_view = mo.vstack([_pair_controls, _pair_story, mo.mpl.interactive(_fig)], gap=1)
    return (pair_delta_view,)


@app.cell(hide_code=True)
def _(has_results, profile_df):
    tradeoff_view = mo.md("")

    if has_results:
        _fig, _ax = plt.subplots(figsize=(8.4, 6.2))

        for _, _row in profile_df.iterrows():
            _label = _row["model_label"]
            _x = float(_row["typing_mean"])
            _y = float(_row["enter_mean"])
            _overall = float(_row["overall_mean"])

            _ax.scatter(
                _x,
                _y,
                s=2600 * _overall,
                color=MODEL_COLORS[_label],
                edgecolor="#0f172a",
                linewidth=1.0,
                alpha=0.82,
                zorder=3,
            )
            _ax.text(
                _x,
                _y,
                f"{_label}\n{100*_overall:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#0f172a",
                zorder=4,
            )

        _ax.set_xlim(0.0, 1.0)
        _ax.set_ylim(0.0, 1.0)
        _ax.set_xlabel("Typing accuracy (mechanics)", fontweight="bold")
        _ax.set_ylabel("Enter accuracy (meaning)", fontweight="bold")
        _ax.grid(True, color="#e5e7eb", linewidth=1)
        _ax.set_axisbelow(True)
        _ax.axvline(0.5, color="#cbd5e1", linestyle="--", linewidth=1)
        _ax.axhline(0.5, color="#cbd5e1", linestyle="--", linewidth=1)
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.spines["left"].set_color("#cbd5e1")
        _ax.spines["bottom"].set_color("#cbd5e1")
        _fig.tight_layout()

        _tradeoff_note = mo.callout(
            mo.md(
                "Farther right means better typing mechanics. Higher means stronger Enter behavior. Larger bubbles mean stronger mean overall changed-cell accuracy."
            ),
            kind="info",
        )
        tradeoff_view = mo.vstack([_tradeoff_note, mo.mpl.interactive(_fig)], gap=1)
    return (tradeoff_view,)


@app.cell(hide_code=True)
def _(exact_table_df, has_results):
    exact_table_element = mo.ui.table(exact_table_df, selection="single", page_size=4) if has_results else mo.md("")
    return (exact_table_element,)


@app.cell(hide_code=True)
def _(exact_table_element, has_results):
    evidence_summary_view = mo.md("")

    if has_results:
        _selected_rows = exact_table_element.value
        if len(_selected_rows) == 0:
            evidence_summary_view = mo.callout(
                mo.md("Select a row in the exact table to read one setting in detail."),
                kind="info",
            )
        else:
            _row = _selected_rows.iloc[0]
            evidence_summary_view = mo.callout(
                mo.md(
                    f"**{_row['Setting']}** — overall best: {_row['Overall · MLP']}, {_row['Overall · Transformer']}, {_row['Overall · GRU']}. Enter best: {_row['Enter · MLP']}, {_row['Enter · Transformer']}, {_row['Enter · GRU']}."
                ),
                kind="success",
            )
    return (evidence_summary_view,)


@app.cell(hide_code=True)
def _(has_results, long_df):
    raw_rows_view = mo.md("")

    if has_results:
        _raw_df = long_df[
            [
                "setting_label",
                "model_label",
                "changed_acc",
                "typing_changed_acc",
                "enter_changed_acc",
                "char_acc",
            ]
        ].rename(
            columns={
                "setting_label": "Setting",
                "model_label": "Model",
                "changed_acc": "Overall",
                "typing_changed_acc": "Typing",
                "enter_changed_acc": "Enter",
                "char_acc": "Character",
            }
        )
        raw_rows_view = mo.ui.dataframe(_raw_df)
    return (raw_rows_view,)


@app.cell(hide_code=True)
def _(
    benchmark_overview_view,
    benchmark_summary_view,
    evidence_summary_view,
    exact_table_element,
    focused_setting_view,
    pair_delta_view,
    raw_rows_view,
    tradeoff_view,
):
    benchmark_tabs_view = mo.ui.tabs(
        {
            "Overview": mo.vstack(
                [benchmark_overview_view, benchmark_summary_view, focused_setting_view],
                gap=1.2,
            ),
            "Pair comparison": mo.vstack([pair_delta_view, tradeoff_view], gap=1.2),
            "Evidence": mo.vstack(
                [
                    evidence_summary_view,
                    exact_table_element,
                    mo.accordion({"Long benchmark rows": raw_rows_view}, multiple=True),
                ],
                gap=1.0,
            ),
        },
        value="Overview",
    )
    benchmark_tabs_view
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 4) Reproducibility and extension

    This notebook stays CPU-friendly by loading saved benchmark artifacts, but the training and evaluation code is part of the project and is surfaced below.
    """)
    return


@app.cell(hide_code=True)
def _():
    repro_paths = {
        "smoke_script": ROOT / "experiments" / "toy_nc_cli" / "scripts" / "smoke_test.py",
        "studies": ROOT / "experiments" / "toy_nc_cli" / "src" / "studies.py",
        "transformer_train": ROOT / "experiments" / "toy_nc_cli" / "scripts" / "train_transformer_baseline.py",
        "gru_train": ROOT / "experiments" / "toy_nc_cli" / "scripts" / "train_gru_baseline.py",
        "remote_transformer": ROOT / "experiments" / "toy_nc_cli" / "scripts" / "remote_transformer_uv.sh",
        "remote_gru": ROOT / "experiments" / "toy_nc_cli" / "scripts" / "remote_gru_uv.sh",
        "smoke_result": ROOT / "experiments" / "toy_nc_cli" / "results" / "smoke_test.json",
    }
    repro_code = {
        name: path.read_text(encoding="utf-8")
        for name, path in repro_paths.items()
        if path.exists() and path.suffix in {".py", ".sh"}
    }
    smoke_result = (
        json.loads(repro_paths["smoke_result"].read_text(encoding="utf-8"))
        if repro_paths["smoke_result"].exists()
        else {}
    )
    return repro_code, repro_paths, smoke_result


@app.cell(hide_code=True)
def _():
    smoke_run_button = mo.ui.run_button(label="Run local smoke test")
    smoke_run_button
    return (smoke_run_button,)


@app.cell(hide_code=True)
def _(repro_code, repro_paths, smoke_result, smoke_run_button):
    smoke_view = mo.md("")

    _smoke_output = ""
    _latest_smoke = smoke_result

    if smoke_run_button.value:
        import subprocess

        _proc = subprocess.run(
            ["python3", "experiments/toy_nc_cli/scripts/smoke_test.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        _smoke_output = (_proc.stdout or "") + ("\n" + _proc.stderr if _proc.stderr else "")
        if repro_paths["smoke_result"].exists():
            _latest_smoke = json.loads(repro_paths["smoke_result"].read_text(encoding="utf-8"))

    if _latest_smoke:
        _stats = mo.hstack(
            [
                mo.stat(label="Determinism", value="OK" if _latest_smoke["determinism_ok"] else "FAIL"),
                mo.stat(label="Encode/decode", value="OK" if _latest_smoke["roundtrip_ok"] else "FAIL"),
                mo.stat(
                    label="Held-out changed-cell acc",
                    value=pct(float(_latest_smoke["heldout"]["changed_acc"])),
                    caption="small local smoke run",
                ),
                mo.stat(
                    label="Copy baseline on held-out",
                    value=pct(float(_latest_smoke["heldout_copy"]["changed_acc"])),
                    caption="why learning matters",
                ),
            ],
            widths="equal",
            gap=1,
        )
        _story = mo.callout(
            mo.md(
                f"The smoke test is a **CPU-scale sanity check**, not the canonical competition benchmark. On this saved run, the learned local model reaches **{pct(float(_latest_smoke['heldout']['changed_acc']))}** changed-cell accuracy on held-out episodes, versus **{pct(float(_latest_smoke['heldout_copy']['changed_acc']))}** for the copy baseline."
            ),
            kind="info",
        )
        _stdout_block = mo.accordion(
            {
                "Show smoke-test code and latest stdout": mo.vstack(
                    [
                        mo.md(f"```python\n{repro_code['smoke_script']}\n```") if 'smoke_script' in repro_code else mo.md(""),
                        mo.md(f"```text\n{_smoke_output.strip() or json.dumps(_latest_smoke, indent=2)}\n```") if (_smoke_output or _latest_smoke) else mo.md(""),
                    ]
                )
            },
            multiple=True,
        )
        smoke_view = mo.vstack([smoke_run_button, _stats, _story, _stdout_block], gap=1)
    else:
        smoke_view = mo.vstack(
            [
                smoke_run_button,
                mo.callout(mo.md("No smoke-test artifact is available yet."), kind="warn"),
            ]
        )
    return (smoke_view,)


@app.cell(hide_code=True)
def _():
    extension_note_view = mo.callout(
        mo.md(
            """
    **Why this is more than a plain summary:** the notebook adds its own extension to the paper inspiration — a toy terminal benchmark that splits local mechanics from meaning-heavy Enter steps, includes paraphrase settings, and compares matched baselines. That extension provides insight even without reproducing the full paper.
    """
        ),
        kind="success",
    )
    return (extension_note_view,)


@app.cell(hide_code=True)
def _(repro_code):
    code_tabs_view = mo.ui.tabs(
        {
            "MLP study code": mo.md(f"```python\n{trim_code(repro_code['studies'])}\n```") if 'studies' in repro_code else mo.md(""),
            "Transformer training": mo.md(f"```python\n{trim_code(repro_code['transformer_train'])}\n```") if 'transformer_train' in repro_code else mo.md(""),
            "GRU training": mo.md(f"```python\n{trim_code(repro_code['gru_train'])}\n```") if 'gru_train' in repro_code else mo.md(""),
            "Remote GPU launchers": mo.vstack(
                [
                    mo.md(f"```bash\n{repro_code['remote_transformer']}\n```") if 'remote_transformer' in repro_code else mo.md(""),
                    mo.md(f"```bash\n{repro_code['remote_gru']}\n```") if 'remote_gru' in repro_code else mo.md(""),
                ]
            ),
        },
        value="MLP study code",
        lazy=True,
    )
    return (code_tabs_view,)


@app.cell(hide_code=True)
def _(code_tabs_view, extension_note_view, smoke_view):
    repro_tabs_view = mo.ui.tabs(
        {
            "Scope + extension": extension_note_view,
            "Local smoke test": smoke_view,
            "Training code": code_tabs_view,
        },
        value="Scope + extension",
        lazy=True,
    )
    repro_tabs_view
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 5) Takeaways

    1. **Best overall baseline:** MLP remains strongest on changed-cell accuracy.
    2. **Most interesting contrast:** Transformer is relatively stronger on Enter-heavy updates.
    3. **Critical limitation:** that Transformer gain shrinks sharply under paraphrase.

    ### What this means for next iterations

    - Keep MLP-style local mechanics as a strong baseline path.
    - Add a stronger semantic test than fixed toy command families.
    - Evaluate hybrid models that combine local patch updates with global context.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ---

    Built with [marimo](https://marimo.io/) · notebook path: `notebooks/neural_computers_competition.py`
    Data source for charts: `experiments/toy_nc_cli/results/baseline_comparison.csv`
    """)
    return


if __name__ == "__main__":
    app.run()
