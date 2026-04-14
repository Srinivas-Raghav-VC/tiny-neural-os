from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random
from typing import Literal

import numpy as np
from sklearn.neural_network import MLPClassifier

from .toy_terminal import (
    Action,
    Episode,
    TerminalConfig,
    TerminalState,
    VOCAB,
    CHAR_TO_IDX,
    CURSOR,
    PAD,
    FAMILIES,
    char_accuracy,
    changed_cell_accuracy,
    exact_line_accuracy,
)

ConditionLevel = Literal["none", "family", "command", "hinted"]

ACTION_KINDS = ["idle", "type_char", "backspace", "enter"]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_KINDS)}
FAMILY_TO_IDX = {name: i for i, name in enumerate(FAMILIES)}


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 128
    max_iter: int = 60
    negative_ratio: int = 12
    patch_radius: int = 1
    random_state: int = 0


@dataclass
class RolloutMetrics:
    char_acc: float
    changed_acc: float
    exact_line_acc: float


class CellUpdateModel:
    def __init__(self, config: TerminalConfig, model_config: ModelConfig, condition_level: ConditionLevel):
        self.config = config
        self.model_config = model_config
        self.condition_level = condition_level
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(model_config.hidden_size,),
            max_iter=model_config.max_iter,
            learning_rate_init=1e-3,
            batch_size=256,
            random_state=model_config.random_state,
            verbose=False,
        )

    def fit(self, episodes: list[Episode]) -> "CellUpdateModel":
        X, y = build_cell_dataset(
            episodes=episodes,
            config=self.config,
            condition_level=self.condition_level,
            negative_ratio=self.model_config.negative_ratio,
            random_state=self.model_config.random_state,
        )
        self.classifier.fit(X, y)
        return self

    def predict_next_frame(self, frame: np.ndarray, action: Action) -> np.ndarray:
        rows, cols = frame.shape
        features = [
            encode_cell_features(
                frame=frame,
                row=r,
                col=c,
                action=action,
                config=self.config,
                condition_level=self.condition_level,
            )
            for r in range(rows)
            for c in range(cols)
        ]
        pred = self.classifier.predict(np.asarray(features, dtype=np.float32))
        chars = np.array([VOCAB[int(idx)] for idx in pred], dtype="<U1").reshape(rows, cols)
        return chars

    def rollout(self, episode: Episode) -> list[np.ndarray]:
        preds = [episode.frames[0].copy()]
        current = episode.frames[0].copy()
        for action in episode.actions:
            current = self.predict_next_frame(current, action)
            preds.append(current)
        return preds


def one_hot(index: int, size: int) -> np.ndarray:
    arr = np.zeros(size, dtype=np.float32)
    arr[index] = 1.0
    return arr


@lru_cache(maxsize=4096)
def context_encoding(text: str, width: int) -> np.ndarray:
    text = text[:width].ljust(width)
    out = np.zeros((width, len(VOCAB)), dtype=np.float32)
    for i, ch in enumerate(text):
        out[i, CHAR_TO_IDX.get(ch, CHAR_TO_IDX[PAD])] = 1.0
    return out.reshape(-1)


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
    return out.reshape(-1)


def action_kind_encoding(action: Action) -> np.ndarray:
    return one_hot(ACTION_TO_IDX[action.kind], len(ACTION_KINDS))


def typed_char_encoding(action: Action) -> np.ndarray:
    typed = action.typed_char if action.typed_char in CHAR_TO_IDX else PAD
    return one_hot(CHAR_TO_IDX.get(typed, CHAR_TO_IDX[PAD]), len(VOCAB))


def family_encoding(action: Action) -> np.ndarray:
    return one_hot(FAMILY_TO_IDX[action.command_family], len(FAMILIES))


def context_text_for(action: Action, condition_level: ConditionLevel) -> str:
    if condition_level == "none":
        return ""
    if condition_level == "family":
        return action.command_family
    if condition_level == "command":
        return action.command_text
    hinted = action.command_text
    if action.hint_text:
        hinted += " # " + action.hint_text
    return hinted


def encode_cell_features(
    frame: np.ndarray,
    row: int,
    col: int,
    action: Action,
    config: TerminalConfig,
    condition_level: ConditionLevel,
) -> np.ndarray:
    parts = [
        patch_encoding(frame, row, col, config.patch_radius),
        np.asarray([row / max(frame.shape[0] - 1, 1), col / max(frame.shape[1] - 1, 1)], dtype=np.float32),
        np.asarray([1.0 if action.noisy else 0.0], dtype=np.float32),
    ]
    if condition_level in {"family", "command", "hinted"}:
        parts.append(action_kind_encoding(action))
        parts.append(family_encoding(action))
    if condition_level in {"command", "hinted"}:
        parts.append(typed_char_encoding(action))
        parts.append(context_encoding(context_text_for(action, condition_level), config.context_width))
    return np.concatenate(parts).astype(np.float32)


def build_cell_dataset(
    episodes: list[Episode],
    config: TerminalConfig,
    condition_level: ConditionLevel,
    negative_ratio: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(random_state)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for episode in episodes:
        for step, action in enumerate(episode.actions):
            prev = episode.frames[step]
            nxt = episode.frames[step + 1]
            changed = np.argwhere(prev != nxt)
            unchanged = np.argwhere(prev == nxt)
            keep = [tuple(x) for x in changed.tolist()]
            if len(changed) == 0:
                sample_size = min(len(unchanged), 16)
            else:
                sample_size = min(len(unchanged), negative_ratio * len(changed))
            if sample_size:
                picks = rng.sample(range(len(unchanged)), sample_size)
                keep.extend(tuple(unchanged[i].tolist()) for i in picks)
            for row, col in keep:
                features.append(
                    encode_cell_features(
                        frame=prev,
                        row=row,
                        col=col,
                        action=action,
                        config=config,
                        condition_level=condition_level,
                    )
                )
                labels.append(CHAR_TO_IDX[nxt[row, col]])
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int16)


def _prompt_from_initial_frame(frame: np.ndarray) -> str:
    non_empty = ["".join(row.tolist()).rstrip() for row in frame if "".join(row.tolist()).strip()]
    if not non_empty:
        return "$ "
    return non_empty[-1].replace(CURSOR, "")


def copy_baseline_rollout(episode: Episode) -> list[np.ndarray]:
    preds = [episode.frames[0].copy()]
    current = episode.frames[0].copy()
    for _action in episode.actions:
        preds.append(current.copy())
    return preds


def heuristic_rollout(episode: Episode, placeholder_output: str = "<output>") -> list[np.ndarray]:
    prompt = _prompt_from_initial_frame(episode.frames[0])
    state = TerminalState(TerminalConfig(rows=episode.frames[0].shape[0], cols=episode.frames[0].shape[1]), prompt)
    preds = [state.render()]
    heuristic_outputs = {
        "pwd": ["/path/to/project"],
        "whoami": ["researcher"],
        "echo_home": ["/home/researcher"],
        "date": ["2026-04-13"],
        "env_head": ["USER=...", "HOME=...", "SHELL=..."],
        "python_arith": ["42"],
    }
    for action in episode.actions:
        if action.kind == "type_char":
            state.type_char(action.typed_char)
        elif action.kind == "backspace":
            state.backspace()
        elif action.kind == "enter":
            state.enter(heuristic_outputs.get(action.command_family, [placeholder_output]))
        preds.append(state.render())
    return preds


def summarize_rollout(pred_frames: list[np.ndarray], episode: Episode) -> RolloutMetrics:
    char_scores = []
    changed_scores = []
    line_scores = []
    for prev, pred, truth in zip(episode.frames[:-1], pred_frames[1:], episode.frames[1:]):
        char_scores.append(char_accuracy(pred, truth))
        changed_scores.append(changed_cell_accuracy(prev, pred, truth))
        line_scores.append(exact_line_accuracy(pred, truth))
    return RolloutMetrics(
        char_acc=float(np.mean(char_scores)),
        changed_acc=float(np.mean(changed_scores)),
        exact_line_acc=float(np.mean(line_scores)),
    )


def evaluate_model(
    model: CellUpdateModel | None,
    episodes: list[Episode],
    baseline: bool = False,
    heuristic: bool = False,
) -> dict[str, float]:
    metrics = []
    for episode in episodes:
        if baseline:
            pred_frames = copy_baseline_rollout(episode)
        elif heuristic:
            pred_frames = heuristic_rollout(episode)
        else:
            assert model is not None
            pred_frames = model.rollout(episode)
        metrics.append(summarize_rollout(pred_frames, episode))
    return {
        "char_acc": float(np.mean([m.char_acc for m in metrics])),
        "changed_acc": float(np.mean([m.changed_acc for m in metrics])),
        "exact_line_acc": float(np.mean([m.exact_line_acc for m in metrics])),
    }


def arithmetic_exact_match(model: CellUpdateModel, episodes: list[Episode]) -> float:
    values = []
    for episode in episodes:
        if episode.family != "python_arith":
            continue
        pred_frames = model.rollout(episode)
        pred_final = "\n".join("".join(row.tolist()).rstrip() for row in pred_frames[-1])
        truth_final = "\n".join("".join(row.tolist()).rstrip() for row in episode.frames[-1])
        values.append(float(pred_final == truth_final))
    if not values:
        return float("nan")
    return float(np.mean(values))


def action_kind_breakdown(
    model: CellUpdateModel | None,
    episodes: list[Episode],
    baseline: bool = False,
    heuristic: bool = False,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[tuple[float, float, float]]] = {}
    for episode in episodes:
        if baseline:
            pred_frames = copy_baseline_rollout(episode)
        elif heuristic:
            pred_frames = heuristic_rollout(episode)
        else:
            assert model is not None
            pred_frames = model.rollout(episode)
        for step, action in enumerate(episode.actions):
            prev = episode.frames[step]
            pred = pred_frames[step + 1]
            truth = episode.frames[step + 1]
            buckets.setdefault(action.kind, []).append(
                (
                    char_accuracy(pred, truth),
                    changed_cell_accuracy(prev, pred, truth),
                    exact_line_accuracy(pred, truth),
                )
            )
    out: dict[str, dict[str, float]] = {}
    for kind, values in buckets.items():
        arr = np.asarray(values, dtype=np.float32)
        out[kind] = {
            "char_acc": float(arr[:, 0].mean()),
            "changed_acc": float(arr[:, 1].mean()),
            "exact_line_acc": float(arr[:, 2].mean()),
            "count": float(len(values)),
        }
    return out
