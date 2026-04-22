"""Generate a lightweight cache of model predictions for the notebook's hero gallery.

Trains the notebook's default MLP once, then runs an autoregressive rollout on
one canonical variant per command family, and saves the TRUE and PREDICTED
final screens as plain strings. The notebook reads this JSON at the very top to
show a visual "the model works" gallery before the reader has to click Train.

Output: experiments/toy_nc_cli/results/gallery_cache.json
"""

from __future__ import annotations

import io
import json
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------
# Notebook primitives — kept in lockstep with notebooks/neural_computers_competition.py
# ---------------------------------------------------------------------------
PRINTABLE = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
)
CURSOR = "\u2588"
PAD = "\u00b7"
VOCAB = list(PRINTABLE) + [CURSOR, PAD]
CHAR_TO_IDX = {ch: i for i, ch in enumerate(VOCAB)}

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
    ],
    "echo_home": [
        {"cmd": "echo $HOME", "output": "/home/researcher"},
        {"cmd": "printenv HOME", "output": "/home/researcher"},
    ],
    "python_arith": [
        {"cmd": "python -c 'print(7*8)'", "output": "56"},
        {"cmd": "python3 -c 'print(7*8)'", "output": "56"},
    ],
}
FAMILIES = list(COMMAND_VARIANTS.keys())
ACTION_KINDS = ["idle", "type_char", "backspace", "enter"]
ACTION_TO_IDX = {k: i for i, k in enumerate(ACTION_KINDS)}
FAMILY_TO_IDX = {k: i for i, k in enumerate(FAMILIES)}


def make_blank_screen(rows: int = 10, cols: int = 40) -> np.ndarray:
    s = np.full((rows, cols), PAD, dtype="<U1")
    s[0, 0] = "$"
    s[0, 1] = " "
    s[0, 2] = CURSOR
    return s


def find_cursor(screen: np.ndarray) -> tuple[int, int]:
    pos = np.argwhere(screen == CURSOR)
    return tuple(pos[0]) if len(pos) else (0, 2)


class Action:
    def __init__(self, kind: str, typed_char: str = "", command_family: str = "", command_text: str = ""):
        self.kind = kind
        self.typed_char = typed_char
        self.command_family = command_family
        self.command_text = command_text


class Episode:
    def __init__(self, family: str, command_text: str, frames, actions):
        self.family = family
        self.command_text = command_text
        self.frames = frames
        self.actions = actions


def generate_episode(family: str, variant: dict) -> Episode:
    """Mirrors notebook's generate_episode (deterministic given variant)."""
    rows, cols = 10, 40
    screen = make_blank_screen(rows, cols)
    frames = [screen.copy()]
    actions = []
    for ch in variant["cmd"]:
        r, c = find_cursor(screen)
        screen[r, c] = ch
        if c + 1 < cols:
            screen[r, c + 1] = CURSOR
        actions.append(Action("type_char", ch, family, variant["cmd"]))
        frames.append(screen.copy())
    r, c = find_cursor(screen)
    screen[r, c] = PAD
    out_row = r + 1
    if out_row < rows:
        for i, ch in enumerate(variant["output"][:cols]):
            screen[out_row, i] = ch
    pr = out_row + 1
    if pr < rows:
        screen[pr, 0] = "$"
        screen[pr, 1] = " "
        screen[pr, 2] = CURSOR
    actions.append(Action("enter", "", family, variant["cmd"]))
    frames.append(screen.copy())
    return Episode(family, variant["cmd"], frames, actions)


def generate_episodes(n: int, seed: int = 42) -> list[Episode]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        fam = rng.choice(FAMILIES)
        var = rng.choice(COMMAND_VARIANTS[fam])
        out.append(generate_episode(fam, var))
    return out


def one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    v[idx] = 1.0
    return v


def encode_patch(screen: np.ndarray, row: int, col: int, radius: int = 1) -> np.ndarray:
    rows, cols = screen.shape
    patch = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = row + dr, col + dc
            patch.append(screen[r, c] if 0 <= r < rows and 0 <= c < cols else PAD)
    enc = np.zeros((len(patch), len(VOCAB)), dtype=np.float32)
    for i, ch in enumerate(patch):
        enc[i, CHAR_TO_IDX.get(ch, CHAR_TO_IDX[PAD])] = 1.0
    return enc.ravel()


def encode_cell_features(screen: np.ndarray, row: int, col: int, action: Action, conditioning: str = "full") -> np.ndarray:
    feats = [encode_patch(screen, row, col, 1)]
    nr = row / max(screen.shape[0] - 1, 1)
    nc = col / max(screen.shape[1] - 1, 1)
    feats.append(np.array([nr, nc], dtype=np.float32))
    if conditioning in ("family", "full"):
        feats.append(one_hot(ACTION_TO_IDX.get(action.kind, 0), len(ACTION_KINDS)))
        feats.append(one_hot(FAMILY_TO_IDX.get(action.command_family, 0), len(FAMILIES)))
    if conditioning == "full":
        feats.append(one_hot(CHAR_TO_IDX.get(action.typed_char, CHAR_TO_IDX[PAD]), len(VOCAB)))
    return np.concatenate(feats)


def build_dataset(episodes: list[Episode], conditioning: str = "full", neg_ratio: int = 8, seed: int = 42):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for ep in episodes:
        for t, action in enumerate(ep.actions):
            before, after = ep.frames[t], ep.frames[t + 1]
            mask = before != after
            for r, c in np.argwhere(mask):
                X.append(encode_cell_features(before, r, c, action, conditioning))
                y.append(CHAR_TO_IDX.get(after[r, c], CHAR_TO_IDX[PAD]))
            unchanged = np.argwhere(~mask)
            n_neg = min(len(unchanged), mask.sum() * neg_ratio)
            if n_neg > 0:
                idx = rng.choice(len(unchanged), size=n_neg, replace=False)
                for i in idx:
                    r, c = unchanged[i]
                    X.append(encode_cell_features(before, r, c, action, conditioning))
                    y.append(CHAR_TO_IDX.get(after[r, c], CHAR_TO_IDX[PAD]))
    return np.asarray(X, dtype=np.float32), np.asarray(y)


def screen_to_strings(screen: np.ndarray) -> list[str]:
    """Convert an (R, C) char grid to a list of R strings."""
    return ["".join(row.tolist()) for row in screen]


def autoregressive_rollout(model: MLPClassifier, ep: Episode, conditioning: str):
    """Feed the model's own predictions back as input.

    Returns (final_pred_frame, per_step_ar_acc) where per_step_ar_acc is the
    fraction of cells matching the ground-truth next frame at each step.
    """
    rows_n, cols_n = ep.frames[0].shape
    all_positions = [(r, c) for r in range(rows_n) for c in range(cols_n)]
    pred_frame = ep.frames[0].copy()
    per_step = []
    for t, action in enumerate(ep.actions):
        X = np.asarray(
            [encode_cell_features(pred_frame, r, c, action, conditioning) for r, c in all_positions],
            dtype=np.float32,
        )
        cls = model.predict(X)
        nxt = pred_frame.copy()
        for (r, c), k in zip(all_positions, cls):
            nxt[r, c] = VOCAB[int(k)]
        pred_frame = nxt
        true_next = ep.frames[t + 1]
        per_step.append({
            "step": t + 1,
            "action": action.kind,
            "ar_acc": round(float((nxt == true_next).mean()), 4),
        })
    return pred_frame, per_step


def main() -> int:
    # Same defaults the notebook uses when you click Train with no changes.
    conditioning = "full"
    n_train = 60
    n_epochs = 25
    hidden = 128
    neg_ratio = 8
    lr = 2e-3

    print(f"[1/4] generating {n_train} training episodes...")
    train_eps = generate_episodes(n_train, seed=42)

    print(f"[2/4] building dataset (conditioning={conditioning}, neg_ratio={neg_ratio})...")
    X, y = build_dataset(train_eps, conditioning=conditioning, neg_ratio=neg_ratio, seed=42)
    print(f"    X={X.shape} y={y.shape}")

    print(f"[3/4] training MLP (hidden={hidden}, epochs={n_epochs}, lr={lr})...")
    model = MLPClassifier(
        hidden_layer_sizes=(hidden,),
        learning_rate_init=lr,
        batch_size=256,
        random_state=42,
    )
    all_classes = np.arange(len(VOCAB))
    for epoch in range(n_epochs):
        model.partial_fit(X, y, classes=all_classes)
    print(f"    final_loss={float(model.loss_):.4f}")

    print("[4/4] running AR rollout per family and caching...")
    entries = []
    for family, variants in COMMAND_VARIANTS.items():
        variant = variants[0]
        ep = generate_episode(family, variant)
        true_final = ep.frames[-1]
        pred_final, per_step_ar = autoregressive_rollout(model, ep, conditioning)

        diff_mask = pred_final != true_final
        n_wrong = int(diff_mask.sum())
        changed_mask = ep.frames[0] != true_final
        n_changed = int(changed_mask.sum())
        if n_changed > 0:
            n_changed_correct = int(((pred_final == true_final) & changed_mask).sum())
            changed_acc = n_changed_correct / n_changed
        else:
            changed_acc = 1.0
            n_changed_correct = 0

        # Auto-derived insights the notebook renders above the screens.
        pred_text = "".join("".join(row) for row in pred_final)
        expected = variant["output"]
        output_present = expected in pred_text
        # Longest prefix of the expected output that appears anywhere in the prediction;
        # rough proxy for "did the model learn the semantic mapping at all?".
        output_prefix_len = 0
        for L in range(len(expected), 0, -1):
            if expected[:L] in pred_text:
                output_prefix_len = L
                break
        cmd_text = variant["cmd"]
        first_line_str = "".join(pred_final[0].tolist())
        cmd_in_first_line = cmd_text in first_line_str.replace(PAD, "").replace(CURSOR, "")

        entries.append({
            "family": family,
            "cmd": cmd_text,
            "output": expected,
            "n_actions": len(ep.actions),
            "n_wrong": n_wrong,
            "n_total": int(true_final.size),
            "n_changed": n_changed,
            "n_changed_correct": n_changed_correct,
            "changed_acc": round(float(changed_acc), 4),
            "true_final": screen_to_strings(true_final),
            "pred_final": screen_to_strings(pred_final),
            "changed_mask": changed_mask.astype(np.int8).tolist(),
            "diff_mask": diff_mask.astype(np.int8).tolist(),
            "per_step_ar": per_step_ar,
            "insights": {
                "output_present_verbatim": bool(output_present),
                "output_prefix_chars_found": output_prefix_len,
                "expected_output_length": len(expected),
                "command_in_first_line": bool(cmd_in_first_line),
            },
        })
        print(
            f"    {family:12s} cmd={cmd_text!r:40s} "
            f"changed_acc={100 * changed_acc:5.1f}% ({n_changed_correct}/{n_changed}) "
            f"output_present={'Y' if output_present else '.'} "
            f"prefix={output_prefix_len}/{len(expected)}"
        )

    out_path = Path("experiments/toy_nc_cli/results/gallery_cache.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/generate_gallery_cache.py",
        "hyperparameters": {
            "conditioning": conditioning,
            "n_train_episodes": n_train,
            "n_epochs": n_epochs,
            "hidden": hidden,
            "neg_ratio": neg_ratio,
            "learning_rate": lr,
            "random_state": 42,
        },
        "final_loss": round(float(model.loss_), 4),
        "entries": entries,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
