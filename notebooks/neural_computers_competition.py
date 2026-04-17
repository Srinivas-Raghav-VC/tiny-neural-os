# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo",
#   "altair",
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///
"""Tiny Neural OS: A marimo notebook inspired by Neural Computers (arXiv:2604.06425).

This notebook implements a toy terminal benchmark from scratch, trains baseline models,
and visualizes results with Altair — all in one reproducible, interactive artifact.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Tiny Neural OS

    **Can a model learn how a computer works from screen transitions alone?**

    This notebook implements the core idea from [Neural Computers (arXiv:2604.06425)](https://arxiv.org/abs/2604.06425):
    predict the next terminal screen from the current one, then measure whether the model
    captures mechanical typing versus semantic command execution.

    ---

    ## What we build

    1. **Toy terminal simulator** — generates command episodes with typed input and output
    2. **MLP baseline** — per-cell classifier using local patch features
    3. **Evaluation protocol** — separates typing accuracy from Enter accuracy
    4. **Interactive visualizations** — explore the benchmark with Altair charts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 1. The Toy Terminal

    We simulate a simple terminal where a user types commands and sees output.
    Each episode is a sequence of frames (screen states) and actions (keystrokes).
    """)
    return


@app.cell
def _():
    import html as html_lib
    import random
    from dataclasses import dataclass, field
    from typing import Literal

    import numpy as np
    import pandas as pd

    # Character vocabulary for the terminal
    PRINTABLE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    CURSOR = "█"
    PAD = "·"
    VOCAB = list(PRINTABLE) + [CURSOR, PAD]
    CHAR_TO_IDX = {ch: i for i, ch in enumerate(VOCAB)}

    @dataclass
    class TerminalConfig:
        rows: int = 10
        cols: int = 40
        context_width: int = 32
        patch_radius: int = 1

    @dataclass
    class Action:
        kind: Literal["type_char", "backspace", "enter", "idle"]
        typed_char: str = ""
        command_family: str = ""
        command_text: str = ""
        hint_text: str = ""
        noisy: bool = False

    @dataclass
    class Episode:
        family: str
        command_text: str
        frames: list[np.ndarray] = field(default_factory=list)
        actions: list[Action] = field(default_factory=list)

    return (
        Action,
        CHAR_TO_IDX,
        CURSOR,
        Episode,
        PAD,
        TerminalConfig,
        VOCAB,
        html_lib,
        np,
        pd,
        random,
    )


@app.cell
def _(CURSOR, PAD, np):
    # Command variants for the toy terminal
    COMMAND_VARIANTS = {
        "pwd": [
            {"cmd": "pwd", "output": "/home/researcher"},
            {"cmd": "echo $PWD", "output": "/home/researcher"},
            {"cmd": "printenv PWD", "output": "/home/researcher"},
        ],
        "whoami": [
            {"cmd": "whoami", "output": "researcher"},
            {"cmd": "echo $USER", "output": "researcher"},
            {"cmd": "id -un", "output": "researcher"},
        ],
        "date": [
            {"cmd": "date +%Y", "output": "2026"},
            {"cmd": "date '+%Y'", "output": "2026"},
            {"cmd": "echo $(date +%Y)", "output": "2026"},
        ],
        "echo_home": [
            {"cmd": "echo $HOME", "output": "/home/researcher"},
            {"cmd": "printenv HOME", "output": "/home/researcher"},
            {"cmd": "echo ~", "output": "/home/researcher"},
        ],
        "python_arith": [
            {"cmd": "python -c 'print(7*8)'", "output": "56"},
            {"cmd": "python3 -c 'print(7*8)'", "output": "56"},
            {"cmd": "python -c \"print(7*8)\"", "output": "56"},
        ],
    }

    FAMILIES = list(COMMAND_VARIANTS.keys())

    def make_blank_screen(rows: int, cols: int) -> np.ndarray:
        screen = np.full((rows, cols), PAD, dtype="<U1")
        screen[0, 2] = CURSOR
        screen[0, 0] = "$"
        screen[0, 1] = " "
        return screen

    def find_cursor(screen: np.ndarray) -> tuple[int, int]:
        positions = np.argwhere(screen == CURSOR)
        if len(positions) == 0:
            return (0, 2)
        return tuple(positions[0])

    return COMMAND_VARIANTS, FAMILIES, find_cursor, make_blank_screen


@app.cell
def _(
    Action,
    COMMAND_VARIANTS,
    CURSOR,
    Episode,
    PAD,
    TerminalConfig,
    find_cursor,
    make_blank_screen,
    random,
):
    def generate_episode(
        config: TerminalConfig,
        family: str,
        variant_idx: int = 0,
        noisy: bool = False,
        rng: random.Random | None = None,
    ) -> Episode:
        """Generate a single terminal episode."""
        rng = rng or random.Random()
        variant = COMMAND_VARIANTS[family][variant_idx % len(COMMAND_VARIANTS[family])]
        cmd_text = variant["cmd"]
        output_text = variant["output"]

        if noisy and rng.random() < 0.3:
            output_text = output_text + rng.choice(["!", "?", "..."])

        screen = make_blank_screen(config.rows, config.cols)
        frames = [screen.copy()]
        actions = []

        # Type the command
        for ch in cmd_text:
            r, c = find_cursor(screen)
            screen[r, c] = ch
            if c + 1 < config.cols:
                screen[r, c + 1] = CURSOR
            actions.append(
                Action(
                    kind="type_char",
                    typed_char=ch,
                    command_family=family,
                    command_text=cmd_text,
                    noisy=noisy,
                )
            )
            frames.append(screen.copy())

        # Press Enter
        r, c = find_cursor(screen)
        screen[r, c] = PAD

        # Write output on next line
        out_row = r + 1
        if out_row < config.rows:
            for i, ch in enumerate(output_text[: config.cols]):
                screen[out_row, i] = ch

        # New prompt
        prompt_row = out_row + 1
        if prompt_row < config.rows:
            screen[prompt_row, 0] = "$"
            screen[prompt_row, 1] = " "
            screen[prompt_row, 2] = CURSOR

        actions.append(
            Action(
                kind="enter",
                typed_char="",
                command_family=family,
                command_text=cmd_text,
                noisy=noisy,
            )
        )
        frames.append(screen.copy())

        return Episode(family=family, command_text=cmd_text, frames=frames, actions=actions)

    def generate_episodes(
        n: int,
        config: TerminalConfig,
        families: list[str] | None = None,
        noisy: bool = False,
        seed: int = 42,
    ) -> list[Episode]:
        """Generate multiple episodes."""
        rng = random.Random(seed)
        families = families or list(COMMAND_VARIANTS.keys())
        episodes = []
        for _ in range(n):
            family = rng.choice(families)
            variant_idx = rng.randint(0, len(COMMAND_VARIANTS[family]) - 1)
            ep = generate_episode(config, family, variant_idx, noisy, rng)
            episodes.append(ep)
        return episodes

    return (generate_episodes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Interactive Terminal Playground

    Use the controls below to explore how the terminal evolves step by step.
    Notice how **typing** changes only a small local patch, while **Enter** triggers
    a larger, meaning-dependent screen update.
    """)
    return


@app.cell
def _(COMMAND_VARIANTS, mo):
    family_picker = mo.ui.dropdown(
        options=list(COMMAND_VARIANTS.keys()),
        value="whoami",
        label="Command family",
    )
    variant_picker = mo.ui.dropdown(
        options=["Variant 1", "Variant 2", "Variant 3"],
        value="Variant 1",
        label="Phrasing variant",
    )
    noise_toggle = mo.ui.switch(value=False, label="Add noise")
    return family_picker, noise_toggle, variant_picker


@app.cell
def _(family_picker, noise_toggle, variant_picker):
    _variant_idx = int(variant_picker.value.split()[-1]) - 1

    from dataclasses import dataclass as _dc

    @_dc
    class _TC:
        rows: int = 10
        cols: int = 40
        context_width: int = 32
        patch_radius: int = 1

    _config = _TC()

    # Import locally to avoid cell dependency issues
    import random as _random

    _rng = _random.Random(42 + _variant_idx)

    # Inline episode generation for this cell
    from typing import Literal as _Lit

    _CURSOR = "█"
    _PAD = "·"

    _VARIANTS = {
        "pwd": [
            {"cmd": "pwd", "output": "/home/researcher"},
            {"cmd": "echo $PWD", "output": "/home/researcher"},
            {"cmd": "printenv PWD", "output": "/home/researcher"},
        ],
        "whoami": [
            {"cmd": "whoami", "output": "researcher"},
            {"cmd": "echo $USER", "output": "researcher"},
            {"cmd": "id -un", "output": "researcher"},
        ],
        "date": [
            {"cmd": "date +%Y", "output": "2026"},
            {"cmd": "date '+%Y'", "output": "2026"},
            {"cmd": "echo $(date +%Y)", "output": "2026"},
        ],
        "echo_home": [
            {"cmd": "echo $HOME", "output": "/home/researcher"},
            {"cmd": "printenv HOME", "output": "/home/researcher"},
            {"cmd": "echo ~", "output": "/home/researcher"},
        ],
        "python_arith": [
            {"cmd": "python -c 'print(7*8)'", "output": "56"},
            {"cmd": "python3 -c 'print(7*8)'", "output": "56"},
            {"cmd": "python -c \"print(7*8)\"", "output": "56"},
        ],
    }

    import numpy as _np

    def _make_screen(rows, cols):
        s = _np.full((rows, cols), _PAD, dtype="<U1")
        s[0, 0], s[0, 1], s[0, 2] = "$", " ", _CURSOR
        return s

    def _find_cursor(s):
        pos = _np.argwhere(s == _CURSOR)
        return tuple(pos[0]) if len(pos) > 0 else (0, 2)

    _family = family_picker.value
    _var = _VARIANTS[_family][_variant_idx % len(_VARIANTS[_family])]
    _cmd, _out = _var["cmd"], _var["output"]

    if noise_toggle.value and _rng.random() < 0.3:
        _out = _out + _rng.choice(["!", "?", "..."])

    _screen = _make_screen(_config.rows, _config.cols)
    _frames = [_screen.copy()]
    _action_labels = ["Start"]

    for _ch in _cmd:
        _r, _c = _find_cursor(_screen)
        _screen[_r, _c] = _ch
        if _c + 1 < _config.cols:
            _screen[_r, _c + 1] = _CURSOR
        _frames.append(_screen.copy())
        _action_labels.append(f"Type '{_ch}'")

    _r, _c = _find_cursor(_screen)
    _screen[_r, _c] = _PAD
    _out_row = _r + 1
    if _out_row < _config.rows:
        for _i, _ch in enumerate(_out[: _config.cols]):
            _screen[_out_row, _i] = _ch
    _prompt_row = _out_row + 1
    if _prompt_row < _config.rows:
        _screen[_prompt_row, 0], _screen[_prompt_row, 1], _screen[_prompt_row, 2] = "$", " ", _CURSOR
    _frames.append(_screen.copy())
    _action_labels.append("Enter")

    playground_frames = _frames
    playground_actions = _action_labels
    playground_cmd = _cmd
    playground_max_step = len(_frames) - 1
    return (
        playground_actions,
        playground_cmd,
        playground_frames,
        playground_max_step,
    )


@app.cell
def _(mo, playground_max_step):
    step_slider = mo.ui.slider(
        start=0,
        stop=playground_max_step,
        value=0,
        label="Step",
        full_width=True,
    )
    return (step_slider,)


@app.cell
def _(family_picker, mo, noise_toggle, step_slider, variant_picker):
    mo.hstack(
        [family_picker, variant_picker, noise_toggle, step_slider],
        widths=[1.5, 1.5, 1, 3],
        gap=1,
        align="end",
    )
    return


@app.cell
def _(
    html_lib,
    mo,
    np,
    playground_actions,
    playground_cmd,
    playground_frames,
    step_slider,
):
    _step = step_slider.value
    _frame = playground_frames[_step]
    _action = playground_actions[_step]

    if _step > 0:
        _prev = playground_frames[_step - 1]
        _changed = _frame != _prev
    else:
        _changed = np.zeros_like(_frame, dtype=bool)

    _changed_count = int(_changed.sum())
    _is_enter = _action == "Enter"

    def _render_terminal(frame, changed_mask, title):
        rows_html = []
        for r in range(frame.shape[0]):
            cells = []
            for c in range(frame.shape[1]):
                ch = frame[r, c]
                safe = "&nbsp;" if ch == " " else html_lib.escape(ch)
                if changed_mask[r, c]:
                    cells.append(f"<span style='background:#fef3c7;color:#111;border-radius:2px;padding:0 1px'>{safe}</span>")
                else:
                    cells.append(f"<span>{safe}</span>")
            rows_html.append("".join(cells))
        return f"""
        <div style='border:1px solid #1e293b;border-radius:10px;background:#0f172a;color:#e2e8f0;
                    padding:12px;font-family:monospace;line-height:1.3;font-size:0.85rem'>
            <div style='color:#93c5fd;font-size:0.7rem;text-transform:uppercase;margin-bottom:8px'>{title}</div>
            {'<br>'.join(rows_html)}
        </div>
        """

    _empty_mask = np.zeros_like(_frame, dtype=bool)
    _before_html = _render_terminal(playground_frames[max(0, _step - 1)], _empty_mask, "Before")
    _after_html = _render_terminal(_frame, _changed, "After")

    _terminal_view = mo.hstack(
        [mo.Html(_before_html), mo.Html(_after_html)],
        widths=[1, 1],
        gap=1,
    )

    _action_kind = "Meaning-heavy (Enter)" if _is_enter else "Local mechanics"
    _stats_view = mo.hstack(
        [
            mo.stat(label="Action", value=_action, caption=_action_kind),
            mo.stat(label="Changed cells", value=str(_changed_count)),
            mo.stat(label="Command", value=playground_cmd),
        ],
        widths="equal",
        gap=1,
    )

    _callout = mo.callout(
        mo.md(
            f"**{_action}** changes **{_changed_count}** cells. "
            + ("This is the **harder** regime: the model needs semantic understanding."
               if _is_enter else "This is **easy**: local patch prediction suffices.")
        ),
        kind="warn" if _is_enter else "info",
    )

    mo.vstack([_terminal_view, _stats_view, _callout], gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 2. The MLP Baseline

    We train a simple MLP classifier that predicts each cell's next character
    based on a local patch around it plus the action being taken.

    **Key insight:** This model is strong on typing (local changes) but weaker on Enter
    (where distant cells change based on command semantics).
    """)
    return


@app.cell
def _(CHAR_TO_IDX, FAMILIES, PAD, VOCAB, np):
    from sklearn.neural_network import MLPClassifier

    ACTION_KINDS = ["idle", "type_char", "backspace", "enter"]
    ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_KINDS)}
    FAMILY_TO_IDX = {name: i for i, name in enumerate(FAMILIES)}

    def one_hot(idx: int, size: int) -> np.ndarray:
        arr = np.zeros(size, dtype=np.float32)
        arr[idx] = 1.0
        return arr

    def patch_encoding(frame: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
        rows, cols = frame.shape
        patch_chars = []
        for rr in range(row - radius, row + radius + 1):
            for cc in range(col - radius, col + radius + 1):
                if 0 <= rr < rows and 0 <= cc < cols:
                    patch_chars.append(frame[rr, cc])
                else:
                    patch_chars.append(PAD)
        out = np.zeros((len(patch_chars), len(VOCAB)), dtype=np.float32)
        for i, ch in enumerate(patch_chars):
            out[i, CHAR_TO_IDX.get(ch, CHAR_TO_IDX[PAD])] = 1.0
        return out.ravel()

    def encode_cell(frame, row, col, action, config, condition_level):
        """Encode features for a single cell."""
        parts = [
            patch_encoding(frame, row, col, config.patch_radius),
            np.array([row / max(frame.shape[0] - 1, 1), col / max(frame.shape[1] - 1, 1)], dtype=np.float32),
        ]
        if condition_level in ("family", "command"):
            parts.append(one_hot(ACTION_TO_IDX.get(action.kind, 0), len(ACTION_KINDS)))
            parts.append(one_hot(FAMILY_TO_IDX.get(action.command_family, 0), len(FAMILIES)))
        if condition_level == "command":
            typed_idx = CHAR_TO_IDX.get(action.typed_char, CHAR_TO_IDX[PAD])
            parts.append(one_hot(typed_idx, len(VOCAB)))
        return np.concatenate(parts)

    def build_dataset(episodes, config, condition_level, negative_ratio=8, seed=42):
        """Build training dataset from episodes."""
        rng = np.random.RandomState(seed)
        X_list, y_list = [], []

        for ep in episodes:
            for t, action in enumerate(ep.actions):
                before = ep.frames[t]
                after = ep.frames[t + 1]
                changed_mask = before != after

                for r in range(before.shape[0]):
                    for c in range(before.shape[1]):
                        if changed_mask[r, c]:
                            feat = encode_cell(before, r, c, action, config, condition_level)
                            label = CHAR_TO_IDX.get(after[r, c], CHAR_TO_IDX[PAD])
                            X_list.append(feat)
                            y_list.append(label)

                # Sample negatives
                unchanged = np.argwhere(~changed_mask)
                n_neg = min(len(unchanged), int(changed_mask.sum()) * negative_ratio)
                if n_neg > 0:
                    neg_idx = rng.choice(len(unchanged), size=n_neg, replace=False)
                    for idx in neg_idx:
                        r, c = unchanged[idx]
                        feat = encode_cell(before, r, c, action, config, condition_level)
                        label = CHAR_TO_IDX.get(after[r, c], CHAR_TO_IDX[PAD])
                        X_list.append(feat)
                        y_list.append(label)

        return np.array(X_list, dtype=np.float32), np.array(y_list)

    return MLPClassifier, build_dataset, encode_cell


@app.cell
def _(CHAR_TO_IDX, PAD, encode_cell, np):
    def evaluate_model(model, episodes, config, condition_level):
        """Evaluate model on episodes, returning per-action-type accuracy."""
        results = {"typing": {"correct": 0, "total": 0}, "enter": {"correct": 0, "total": 0}}

        for ep in episodes:
            for t, action in enumerate(ep.actions):
                before = ep.frames[t]
                after = ep.frames[t + 1]
                changed_mask = before != after
                action_type = "enter" if action.kind == "enter" else "typing"

                if not changed_mask.any():
                    continue

                # Predict
                features = []
                positions = []
                for r in range(before.shape[0]):
                    for c in range(before.shape[1]):
                        if changed_mask[r, c]:
                            features.append(encode_cell(before, r, c, action, config, condition_level))
                            positions.append((r, c))

                if not features:
                    continue

                preds = model.predict(np.array(features))
                for (r, c), pred_idx in zip(positions, preds):
                    true_char = after[r, c]
                    true_idx = CHAR_TO_IDX.get(true_char, CHAR_TO_IDX[PAD])
                    results[action_type]["total"] += 1
                    if pred_idx == true_idx:
                        results[action_type]["correct"] += 1

        typing_acc = results["typing"]["correct"] / max(results["typing"]["total"], 1)
        enter_acc = results["enter"]["correct"] / max(results["enter"]["total"], 1)
        total_correct = results["typing"]["correct"] + results["enter"]["correct"]
        total_all = results["typing"]["total"] + results["enter"]["total"]
        overall_acc = total_correct / max(total_all, 1)

        return {
            "typing_acc": typing_acc,
            "enter_acc": enter_acc,
            "overall_acc": overall_acc,
            "typing_total": results["typing"]["total"],
            "enter_total": results["enter"]["total"],
        }

    return (evaluate_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Training the baseline

    We train on a small dataset and evaluate on held-out episodes.
    The training happens live when you run this cell.
    """)
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train MLP Baseline", kind="success")
    train_size_slider = mo.ui.slider(start=20, stop=100, step=10, value=40, label="Training episodes")
    condition_picker = mo.ui.dropdown(
        options=["none", "family", "command"],
        value="command",
        label="Conditioning level",
    )
    mo.hstack([train_button, train_size_slider, condition_picker], gap=1, align="end")
    return condition_picker, train_button, train_size_slider


@app.cell
def _(
    MLPClassifier,
    TerminalConfig,
    build_dataset,
    condition_picker,
    evaluate_model,
    generate_episodes,
    mo,
    train_button,
    train_size_slider,
):
    training_result = None

    if train_button.value:
        _config = TerminalConfig(rows=10, cols=40, context_width=32, patch_radius=1)
        _condition = condition_picker.value
        _n_train = train_size_slider.value
        _n_test = 20

        with mo.status.spinner("Generating episodes..."):
            _train_eps = generate_episodes(_n_train, _config, seed=42)
            _test_eps = generate_episodes(_n_test, _config, seed=123)

        with mo.status.spinner("Building dataset..."):
            _X_train, _y_train = build_dataset(_train_eps, _config, _condition, negative_ratio=8, seed=42)

        with mo.status.spinner("Training MLP..."):
            _model = MLPClassifier(
                hidden_layer_sizes=(128,),
                max_iter=80,
                learning_rate_init=1e-3,
                batch_size=256,
                random_state=42,
                verbose=False,
            )
            _model.fit(_X_train, _y_train)

        with mo.status.spinner("Evaluating..."):
            _metrics = evaluate_model(_model, _test_eps, _config, _condition)

        training_result = {
            "model": _model,
            "config": _config,
            "condition": _condition,
            "train_size": _n_train,
            "test_size": _n_test,
            "dataset_size": len(_X_train),
            "metrics": _metrics,
        }
    return (training_result,)


@app.cell
def _(mo, training_result):
    if training_result is None:
        training_view = mo.callout(mo.md("Click **Train MLP Baseline** above to train the model."), kind="info")
    else:
        _m = training_result["metrics"]
        training_view = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.stat(
                            label="Overall accuracy",
                            value=f"{100 * _m['overall_acc']:.1f}%",
                            caption="Changed cells only",
                        ),
                        mo.stat(
                            label="Typing accuracy",
                            value=f"{100 * _m['typing_acc']:.1f}%",
                            caption=f"n={_m['typing_total']}",
                        ),
                        mo.stat(
                            label="Enter accuracy",
                            value=f"{100 * _m['enter_acc']:.1f}%",
                            caption=f"n={_m['enter_total']}",
                        ),
                    ],
                    widths="equal",
                    gap=1,
                ),
                mo.callout(
                    mo.md(
                        f"**Training:** {training_result['train_size']} episodes, "
                        f"{training_result['dataset_size']} samples. "
                        f"**Conditioning:** {training_result['condition']}. "
                        f"**Test:** {training_result['test_size']} episodes."
                    ),
                    kind="success",
                ),
            ],
            gap=1,
        )
    training_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 3. Benchmark Results

    Below we show pre-computed benchmark results comparing **MLP**, **Transformer**, and **GRU**
    across four settings:

    | Setting | Description |
    |---------|-------------|
    | Standard · Family | Train and test on same commands, model sees family hint |
    | Standard · Command | Train and test on same commands, model sees exact command |
    | Paraphrase · Family | Train on variant 1, test on variants 2-3, family hint only |
    | Paraphrase · Command | Train on variant 1, test on variants 2-3, exact command hint |
    """)
    return


@app.cell
def _(pd):
    # Pre-computed benchmark results (from remote GPU runs)
    benchmark_data = pd.DataFrame([
        {"model": "MLP", "setting": "Standard · Family", "overall": 0.681, "typing": 0.896, "enter": 0.656},
        {"model": "MLP", "setting": "Standard · Command", "overall": 0.947, "typing": 0.903, "enter": 0.998},
        {"model": "MLP", "setting": "Paraphrase · Family", "overall": 0.637, "typing": 0.898, "enter": 0.610},
        {"model": "MLP", "setting": "Paraphrase · Command", "overall": 0.716, "typing": 0.739, "enter": 0.718},
        {"model": "Transformer", "setting": "Standard · Family", "overall": 0.599, "typing": 0.792, "enter": 0.619},
        {"model": "Transformer", "setting": "Standard · Command", "overall": 0.880, "typing": 0.868, "enter": 0.982},
        {"model": "Transformer", "setting": "Paraphrase · Family", "overall": 0.543, "typing": 0.659, "enter": 0.627},
        {"model": "Transformer", "setting": "Paraphrase · Command", "overall": 0.639, "typing": 0.655, "enter": 0.747},
        {"model": "GRU", "setting": "Standard · Family", "overall": 0.541, "typing": 0.825, "enter": 0.452},
        {"model": "GRU", "setting": "Standard · Command", "overall": 0.732, "typing": 0.820, "enter": 0.786},
        {"model": "GRU", "setting": "Paraphrase · Family", "overall": 0.478, "typing": 0.620, "enter": 0.558},
        {"model": "GRU", "setting": "Paraphrase · Command", "overall": 0.549, "typing": 0.612, "enter": 0.640},
    ])
    return (benchmark_data,)


@app.cell
def _(mo):
    metric_picker = mo.ui.dropdown(
        options=["Overall changed-cell acc", "Typing acc", "Enter acc"],
        value="Overall changed-cell acc",
        label="Metric to display",
    )
    return (metric_picker,)


@app.cell
def _(metric_picker, mo):
    mo.hstack([metric_picker], justify="start")
    return


@app.cell
def _(benchmark_data, metric_picker, mo):
    import altair as alt

    _metric_map = {
        "Overall changed-cell acc": "overall",
        "Typing acc": "typing",
        "Enter acc": "enter",
    }
    _metric_col = _metric_map[metric_picker.value]

    _chart_data = benchmark_data.copy()
    _chart_data["value"] = _chart_data[_metric_col]
    _chart_data["value_pct"] = (_chart_data["value"] * 100).round(1).astype(str) + "%"

    _setting_order = [
        "Standard · Family",
        "Standard · Command",
        "Paraphrase · Family",
        "Paraphrase · Command",
    ]

    _color_scale = alt.Scale(
        domain=["MLP", "Transformer", "GRU"],
        range=["#22c55e", "#7c3aed", "#ef4444"],
    )

    _bars = (
        alt.Chart(_chart_data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("setting:N", sort=_setting_order, title=None, axis=alt.Axis(labelAngle=-15)),
            xOffset="model:N",
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=metric_picker.value),
            color=alt.Color("model:N", scale=_color_scale, legend=alt.Legend(title="Model", orient="top")),
            tooltip=["model", "setting", "value_pct"],
        )
    )

    _text = (
        alt.Chart(_chart_data)
        .mark_text(dy=-8, fontSize=10, fontWeight="bold")
        .encode(
            x=alt.X("setting:N", sort=_setting_order),
            xOffset="model:N",
            y=alt.Y("value:Q"),
            text="value_pct:N",
            color=alt.value("#0f172a"),
        )
    )

    _chart = (
        (_bars + _text)
        .properties(width=700, height=340, title=f"Baseline Comparison: {metric_picker.value}")
        .configure_axis(grid=True, gridColor="#e5e7eb")
        .configure_view(strokeWidth=0)
    )

    mo.ui.altair_chart(_chart, chart_selection=False)
    return (alt,)


@app.cell(hide_code=True)
def _(benchmark_data, mo):
    # Summary stats
    _mlp_overall = benchmark_data[benchmark_data["model"] == "MLP"]["overall"].mean()
    _transformer_enter = benchmark_data[benchmark_data["model"] == "Transformer"]["enter"].mean()
    _gru_overall = benchmark_data[benchmark_data["model"] == "GRU"]["overall"].mean()

    mo.hstack(
        [
            mo.stat(label="MLP mean overall", value=f"{100 * _mlp_overall:.1f}%", caption="Best overall baseline"),
            mo.stat(label="Transformer mean Enter", value=f"{100 * _transformer_enter:.1f}%", caption="Stronger on Enter steps"),
            mo.stat(label="GRU mean overall", value=f"{100 * _gru_overall:.1f}%", caption="Underperforms here"),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Typing vs Enter Tradeoff

    The scatter plot below shows each model's average **typing accuracy** (x-axis)
    versus **Enter accuracy** (y-axis). Bubble size reflects overall accuracy.
    """)
    return


@app.cell
def _(alt, benchmark_data, mo):
    _profile = (
        benchmark_data.groupby("model")
        .agg(typing_mean=("typing", "mean"), enter_mean=("enter", "mean"), overall_mean=("overall", "mean"))
        .reset_index()
    )

    _color_scale = alt.Scale(
        domain=["MLP", "Transformer", "GRU"],
        range=["#22c55e", "#7c3aed", "#ef4444"],
    )

    _scatter = (
        alt.Chart(_profile)
        .mark_circle(opacity=0.85, stroke="#0f172a", strokeWidth=1)
        .encode(
            x=alt.X("typing_mean:Q", scale=alt.Scale(domain=[0.5, 1]), title="Typing accuracy (mechanics)"),
            y=alt.Y("enter_mean:Q", scale=alt.Scale(domain=[0.4, 1]), title="Enter accuracy (meaning)"),
            size=alt.Size("overall_mean:Q", scale=alt.Scale(range=[800, 2500]), legend=None),
            color=alt.Color("model:N", scale=_color_scale, legend=alt.Legend(title="Model")),
            tooltip=["model", "typing_mean", "enter_mean", "overall_mean"],
        )
    )

    _text = (
        alt.Chart(_profile)
        .mark_text(dy=-25, fontSize=11, fontWeight="bold")
        .encode(
            x="typing_mean:Q",
            y="enter_mean:Q",
            text="model:N",
            color=alt.value("#0f172a"),
        )
    )

    _chart = (
        (_scatter + _text)
        .properties(width=500, height=380, title="Typing vs Enter Tradeoff")
        .configure_axis(grid=True, gridColor="#e5e7eb")
        .configure_view(strokeWidth=0)
    )

    mo.ui.altair_chart(_chart, chart_selection=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Raw Benchmark Data

    Explore the full benchmark table below. Click column headers to sort.
    """)
    return


@app.cell
def _(benchmark_data, mo):
    _display = benchmark_data.copy()
    _display["overall"] = (_display["overall"] * 100).round(1).astype(str) + "%"
    _display["typing"] = (_display["typing"] * 100).round(1).astype(str) + "%"
    _display["enter"] = (_display["enter"] * 100).round(1).astype(str) + "%"
    _display.columns = ["Model", "Setting", "Overall", "Typing", "Enter"]

    mo.ui.table(_display, selection=None, page_size=12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 4. Key Takeaways

    1. **MLP is the strongest overall baseline** — it wins changed-cell accuracy across most settings
    2. **Transformer shows relative strength on Enter** — but the gain is modest and shrinks under paraphrase
    3. **GRU underperforms** — a negative result in this benchmark
    4. **Typing vs Enter split is diagnostic** — it reveals whether a model relies on local mechanics or captures command semantics

    ### Extension: What this notebook adds

    This notebook goes beyond summarizing the paper:

    - **Custom benchmark** with typing/Enter split and paraphrase generalization
    - **Matched baseline comparison** (MLP, Transformer, GRU on identical data)
    - **Interactive playground** to build intuition about the task
    - **Live training** to experiment with the MLP baseline yourself

    ---

    ## 5. Reproducibility

    The training scripts and full model implementations are in `experiments/toy_nc_cli/`.
    The benchmark CSV used for the visualizations above is at `experiments/toy_nc_cli/results/baseline_comparison.csv`.

    ```bash
    # Run the smoke test locally
    python experiments/toy_nc_cli/scripts/smoke_test.py

    # Train Transformer on GPU
    python experiments/toy_nc_cli/scripts/train_transformer_baseline.py

    # Train GRU on GPU
    python experiments/toy_nc_cli/scripts/train_gru_baseline.py
    ```

    ---

    Built with [marimo](https://marimo.io/) · Inspired by [Neural Computers (arXiv:2604.06425)](https://arxiv.org/abs/2604.06425)
    """)
    return


if __name__ == "__main__":
    app.run()
