from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal
import html
import random
import re
import string

import numpy as np

CURSOR = "█"
PAD = " "

PromptStyle = Literal["clean", "noisy"]
ContextMode = Literal["none", "family", "command", "hinted"]


def pad_line(text: str, cols: int) -> str:
    text = text[:cols]
    return text + PAD * (cols - len(text))


@dataclass(frozen=True)
class TerminalConfig:
    rows: int = 10
    cols: int = 40
    context_width: int = 24
    patch_radius: int = 1


@dataclass
class Action:
    kind: str
    display_text: str
    command_family: str
    typed_char: str = ""
    command_text: str = ""
    hint_text: str = ""
    noisy: bool = False


@dataclass
class Episode:
    family: str
    command_text: str
    command_variant: str
    hint_text: str
    noisy: bool
    actions: list[Action]
    frames: list[np.ndarray]


class TerminalState:
    def __init__(self, config: TerminalConfig, prompt: str):
        self.config = config
        self.prompt = prompt
        self.history: list[str] = []
        self.current_input = ""
        self.show_cursor = True

    def clone(self) -> "TerminalState":
        other = TerminalState(self.config, self.prompt)
        other.history = list(self.history)
        other.current_input = self.current_input
        other.show_cursor = self.show_cursor
        return other

    def type_char(self, ch: str) -> None:
        self.current_input += ch

    def backspace(self) -> None:
        self.current_input = self.current_input[:-1]

    def enter(self, outputs: Iterable[str], next_prompt: str | None = None) -> None:
        self.history.append(self.prompt + self.current_input)
        for line in outputs:
            self.history.append(line)
        self.prompt = self.prompt if next_prompt is None else next_prompt
        self.current_input = ""

    def render(self) -> np.ndarray:
        active = self.prompt + self.current_input + (CURSOR if self.show_cursor else "")
        lines = self.history + [active]
        visible = lines[-self.config.rows :]
        visible = [pad_line(line, self.config.cols) for line in visible]
        while len(visible) < self.config.rows:
            visible.insert(0, PAD * self.config.cols)
        return np.array([list(line) for line in visible], dtype="<U1")


def html_screen(screen: np.ndarray, diff_to: np.ndarray | None = None) -> str:
    lines = []
    for r in range(screen.shape[0]):
        parts = []
        for c in range(screen.shape[1]):
            ch = screen[r, c]
            safe = html.escape(ch)
            if ch == PAD:
                safe = "&nbsp;"
            if diff_to is not None and diff_to[r, c] != ch:
                parts.append(f'<span style="background:#5c1f24;color:#ffefef">{safe}</span>')
            elif ch == CURSOR:
                parts.append(f'<span style="background:#f9e2af;color:#111">{safe}</span>')
            else:
                parts.append(safe)
        lines.append("".join(parts))
    body = "<br>".join(lines)
    return (
        "<div style='background:#111827;color:#e5e7eb;padding:12px;"
        "border-radius:8px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
        "line-height:1.2;white-space:pre;'>"
        + body
        + "</div>"
    )


def screen_to_text(screen: np.ndarray) -> str:
    return "\n".join("".join(row.tolist()).rstrip() for row in screen)


def char_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    return float((pred == truth).mean())


def changed_cell_accuracy(prev: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> float:
    mask = prev != truth
    if not mask.any():
        return 1.0
    return float((pred[mask] == truth[mask]).mean())


def exact_line_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_lines = ["".join(row.tolist()).rstrip() for row in pred]
    truth_lines = ["".join(row.tolist()).rstrip() for row in truth]
    matches = sum(int(p == t) for p, t in zip(pred_lines, truth_lines))
    return matches / len(truth_lines)


def build_vocabulary() -> list[str]:
    chars = set(PAD + CURSOR)
    chars.update(string.ascii_letters)
    chars.update(string.digits)
    chars.update(string.punctuation)
    return sorted(chars)


VOCAB = build_vocabulary()
CHAR_TO_IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX_TO_CHAR = {i: ch for ch, i in CHAR_TO_IDX.items()}


def encode_screen(screen: np.ndarray) -> np.ndarray:
    out = np.zeros(screen.shape, dtype=np.int16)
    for ch, idx in CHAR_TO_IDX.items():
        out[screen == ch] = idx
    return out


def decode_screen(encoded: np.ndarray) -> np.ndarray:
    flat = [IDX_TO_CHAR[int(x)] for x in encoded.reshape(-1)]
    return np.array(flat, dtype="<U1").reshape(encoded.shape)


def safe_arithmetic_eval(expr: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([+\-*])\s*(\d+)\s*", expr)
    if not match:
        raise ValueError(f"Unsupported arithmetic expression: {expr!r}")
    a, op, b = match.groups()
    left = int(a)
    right = int(b)
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    raise ValueError(op)


def command_family_to_outputs(
    family: str,
    payload: str,
    rng: random.Random,
    noisy: bool,
) -> list[str]:
    if family == "pwd":
        paths = ["/home/research", "/mnt/data/demo", "/workspace/project"]
        return [paths[rng.randrange(len(paths))] if noisy else "/home/research"]
    if family == "whoami":
        users = ["researcher", "demo", "student", "alpha"]
        return [users[rng.randrange(len(users))] if noisy else "researcher"]
    if family == "echo_home":
        homes = ["/home/researcher", "/Users/demo", "/workspace/home"]
        return [homes[rng.randrange(len(homes))] if noisy else "/home/researcher"]
    if family == "date":
        if noisy:
            return [rng.choice([
                "2026-04-13 11:59:00 PST",
                "2026-04-14 09:12:33 PST",
                "2026-04-19 18:04:51 PST",
            ])]
        return ["2026-04-13 11:59:00 PST"]
    if family == "env_head":
        variants = [
            ["USER=researcher", "HOME=/home/researcher", "SHELL=/bin/bash"],
            ["USER=student", "HOME=/workspace/home", "SHELL=/bin/zsh"],
            ["USER=alpha", "HOME=/mnt/data/demo", "SHELL=/bin/bash"],
        ]
        return variants[rng.randrange(len(variants))] if noisy else variants[0]
    if family == "python_arith":
        result = str(safe_arithmetic_eval(payload))
        return [result]
    raise ValueError(f"Unknown family: {family}")


def sample_wrong_char(correct: str, rng: random.Random) -> str:
    pool = [ch for ch in "abcdefghijklmnopqrstuvwxyz0123456789+-*/$" if ch != correct]
    return rng.choice(pool)


FAMILIES = ["pwd", "whoami", "echo_home", "date", "env_head", "python_arith"]

COMMAND_VARIANTS = {
    "pwd": ["pwd", "pwd -P", "printf '%s\\n' $PWD"],
    "whoami": ["whoami", "id -un", "printf '%s\\n' $USER"],
    "echo_home": ["echo $HOME", "printf '%s\\n' $HOME", "python -c \"import os; print(os.environ['HOME'])\""],
    "date": ["date", "date +%F", "date +%Y-%m-%d"],
    "env_head": ["env | head -n 3", "printenv | head -n 3", "set | head -n 3"],
}


def sample_command(
    family: str,
    rng: random.Random,
    variant_indices: tuple[int, ...] | None = None,
) -> tuple[str, str, str]:
    if family in COMMAND_VARIANTS:
        variants = COMMAND_VARIANTS[family]
        if variant_indices is None:
            allowed = list(range(len(variants)))
        else:
            allowed = [idx for idx in variant_indices if 0 <= idx < len(variants)]
            if not allowed:
                allowed = list(range(len(variants)))
        variant_id = rng.choice(allowed)
        cmd = variants[variant_id]
        return cmd, cmd, f"v{variant_id}"
    if family == "python_arith":
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        op = rng.choice(["+", "-", "*"])
        expr = f"{a}{op}{b}"
        return expr, expr, "generated"
    raise ValueError(family)


def prompt_for(noisy: bool, rng: random.Random) -> str:
    if not noisy:
        return "research@nc:~$ "
    prompts = [
        "research@nc:~$ ",
        "demo@mini:~/proj$ ",
        "student@box:/tmp$ ",
        "(venv) alpha@lab:~$ ",
    ]
    return rng.choice(prompts)


def make_episode(
    config: TerminalConfig,
    family: str,
    noisy: bool,
    seed: int,
    context_mode: ContextMode = "command",
    variant_indices_by_family: dict[str, tuple[int, ...]] | None = None,
) -> Episode:
    rng = random.Random(seed)
    variant_indices = None if variant_indices_by_family is None else variant_indices_by_family.get(family)
    command_text, payload, command_variant = sample_command(family, rng, variant_indices=variant_indices)
    hint_text = ""
    display_command = command_text
    if family == "python_arith" and context_mode == "hinted":
        hint_text = f"answer={safe_arithmetic_eval(payload)}"
        display_command = f"{command_text}  # {hint_text}"

    prompt = prompt_for(noisy, rng)
    state = TerminalState(config, prompt)
    frames = [state.render()]
    actions: list[Action] = []

    typo_pos = None
    if noisy and rng.random() < 0.35 and family != "python_arith" and len(display_command) >= 2:
        typo_pos = rng.randrange(len(display_command))

    for i, ch in enumerate(display_command):
        if typo_pos is not None and i == typo_pos:
            wrong = sample_wrong_char(ch, rng)
            state.type_char(wrong)
            actions.append(
                Action(
                    kind="type_char",
                    display_text=f"Type '{wrong}' (typo)",
                    command_family=family,
                    typed_char=wrong,
                    command_text=command_text,
                    hint_text=hint_text,
                    noisy=noisy,
                )
            )
            frames.append(state.render())
            state.backspace()
            actions.append(
                Action(
                    kind="backspace",
                    display_text="Backspace",
                    command_family=family,
                    command_text=command_text,
                    hint_text=hint_text,
                    noisy=noisy,
                )
            )
            frames.append(state.render())

        state.type_char(ch)
        actions.append(
            Action(
                kind="type_char",
                display_text=f"Type '{ch}'",
                command_family=family,
                typed_char=ch,
                command_text=command_text,
                hint_text=hint_text,
                noisy=noisy,
            )
        )
        frames.append(state.render())
        if noisy and rng.random() < 0.15:
            actions.append(
                Action(
                    kind="idle",
                    display_text="Idle",
                    command_family=family,
                    command_text=command_text,
                    hint_text=hint_text,
                    noisy=noisy,
                )
            )
            frames.append(state.render())

    outputs = command_family_to_outputs(family, payload, rng, noisy)
    state.enter(outputs)
    actions.append(
        Action(
            kind="enter",
            display_text="Enter",
            command_family=family,
            command_text=command_text,
            hint_text=hint_text,
            noisy=noisy,
        )
    )
    frames.append(state.render())
    actions.append(
        Action(
            kind="idle",
            display_text="Idle",
            command_family=family,
            command_text=command_text,
            hint_text=hint_text,
            noisy=noisy,
        )
    )
    frames.append(state.render())

    return Episode(
        family=family,
        command_text=command_text,
        command_variant=command_variant,
        hint_text=hint_text,
        noisy=noisy,
        actions=actions,
        frames=frames,
    )


def generate_episodes(
    n: int,
    config: TerminalConfig,
    noisy: bool,
    seed: int,
    context_mode: ContextMode = "command",
    families: list[str] | None = None,
    variant_indices_by_family: dict[str, tuple[int, ...]] | None = None,
) -> list[Episode]:
    rng = random.Random(seed)
    families = FAMILIES if families is None else families
    episodes = []
    while len(episodes) < n:
        family_cycle = list(families)
        rng.shuffle(family_cycle)
        for family in family_cycle:
            if len(episodes) >= n:
                break
            episodes.append(
                make_episode(
                    config=config,
                    family=family,
                    noisy=noisy,
                    seed=rng.randint(0, 10_000_000),
                    context_mode=context_mode,
                    variant_indices_by_family=variant_indices_by_family,
                )
            )
    return episodes
