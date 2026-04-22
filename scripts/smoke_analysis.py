"""End-to-end smoke test that exercises the analysis-cell code paths.

Trains a small MLP the same way the notebook does, then runs the core
logic of each new analysis visualization against the trained model to
catch any runtime errors before they land in front of a user.
"""
from __future__ import annotations

import io
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

# Re-declare the notebook's primitives locally. Keep in sync with the notebook.
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
    ],
    "whoami": [
        {"cmd": "whoami", "output": "researcher"},
        {"cmd": "echo $USER", "output": "researcher"},
    ],
    "date": [{"cmd": "date +%Y", "output": "2026"}],
}
FAMILIES = list(COMMAND_VARIANTS.keys())
ACTION_KINDS = ["idle", "type_char", "backspace", "enter"]
ACTION_TO_IDX = {k: i for i, k in enumerate(ACTION_KINDS)}
FAMILY_TO_IDX = {k: i for i, k in enumerate(FAMILIES)}


def make_blank_screen(rows=10, cols=40):
    s = np.full((rows, cols), PAD, dtype="<U1")
    s[0, 0] = "$"
    s[0, 1] = " "
    s[0, 2] = CURSOR
    return s


def find_cursor(screen):
    pos = np.argwhere(screen == CURSOR)
    return tuple(pos[0]) if len(pos) else (0, 2)


class Action:
    def __init__(self, kind, typed_char="", command_family="", command_text=""):
        self.kind = kind
        self.typed_char = typed_char
        self.command_family = command_family
        self.command_text = command_text


class Episode:
    def __init__(self, family, command_text, frames, actions):
        self.family = family
        self.command_text = command_text
        self.frames = frames
        self.actions = actions


def generate_episode(family, variant):
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


def generate_episodes(n, seed=42):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        fam = rng.choice(FAMILIES)
        var = rng.choice(COMMAND_VARIANTS[fam])
        out.append(generate_episode(fam, var))
    return out


def one_hot(idx, size):
    v = np.zeros(size, dtype=np.float32)
    v[idx] = 1.0
    return v


def encode_patch(screen, row, col, radius=1):
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


def encode_cell_features(screen, row, col, action, conditioning="full"):
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


def build_dataset(episodes, conditioning="full", neg_ratio=8, seed=42):
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


def main():
    print("[1/6] generating episodes...")
    train_eps = generate_episodes(20, seed=42)
    test_eps = generate_episodes(10, seed=999)

    print("[2/6] building dataset...")
    conditioning = "full"
    X, y = build_dataset(train_eps, conditioning=conditioning, neg_ratio=6, seed=42)
    print(f"    X={X.shape} y={y.shape}")

    print("[3/6] training MLP via partial_fit with snapshots (evolution tab)...")
    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        learning_rate_init=0.002,
        batch_size=256,
        random_state=42,
    )
    all_classes = np.arange(len(VOCAB))
    n_epochs = 8

    ref_ep = generate_episode("whoami", COMMAND_VARIANTS["whoami"][0])
    ref_t = len(ref_ep.actions) - 1
    ref_action = ref_ep.actions[ref_t]
    ref_before = ref_ep.frames[ref_t]
    ref_after = ref_ep.frames[ref_t + 1]
    ref_positions = np.argwhere(ref_before != ref_after)
    ref_X = np.asarray(
        [encode_cell_features(ref_before, int(r), int(c), ref_action, conditioning) for r, c in ref_positions],
        dtype=np.float32,
    )
    ref_true = np.asarray(
        [CHAR_TO_IDX.get(ref_after[int(r), int(c)], CHAR_TO_IDX[PAD]) for r, c in ref_positions]
    )

    loss_curve = []
    snaps = []
    for epoch in range(n_epochs):
        model.partial_fit(X, y, classes=all_classes)
        loss_curve.append(float(model.loss_))
        preds = model.predict(ref_X)
        acc = float(np.mean(preds == ref_true))
        snaps.append({
            "epoch": epoch + 1,
            "accuracy": acc,
            "loss": float(model.loss_),
            "predicted_chars": [VOCAB[int(p)] for p in preds],
        })
    model.loss_curve_ = loss_curve
    assert len(snaps) == n_epochs
    assert 0.0 <= snaps[-1]["accuracy"] <= 1.0
    assert snaps[0]["accuracy"] <= snaps[-1]["accuracy"] + 1e-6 or snaps[-1]["accuracy"] > 0.3
    print(
        "    epoch_accs:",
        [f"{s['accuracy']:.2f}" for s in snaps],
        "| final_loss=",
        f"{loss_curve[-1]:.3f}",
    )

    print("[4/6] per-position error heatmap logic...")
    rows_n, cols_n = 10, 40
    correct = np.zeros((rows_n, cols_n), dtype=np.int32)
    total = np.zeros((rows_n, cols_n), dtype=np.int32)
    for ep in test_eps:
        for t, action in enumerate(ep.actions):
            before, after = ep.frames[t], ep.frames[t + 1]
            mask = before != after
            positions = np.argwhere(mask)
            if len(positions) == 0:
                continue
            Xb = np.asarray(
                [encode_cell_features(before, r, c, action, conditioning) for r, c in positions],
                dtype=np.float32,
            )
            preds = model.predict(Xb)
            for (r, c), p in zip(positions, preds):
                true = CHAR_TO_IDX.get(after[r, c], CHAR_TO_IDX[PAD])
                total[r, c] += 1
                if p == true:
                    correct[r, c] += 1
    rows_acc = [
        correct[r].sum() / max(total[r].sum(), 1) for r in range(rows_n)
    ]
    print("    row accuracies:", [f"{a:.2f}" for a in rows_acc])

    print("[5/6] probe logic (predict_proba top-5) + anatomy computations...")
    ep = test_eps[0]
    t = 2
    action = ep.actions[t]
    before, after = ep.frames[t], ep.frames[t + 1]
    positions = np.argwhere(before != after)
    r, c = positions[0]
    feats = encode_cell_features(before, int(r), int(c), action, conditioning)
    proba = model.predict_proba(feats.reshape(1, -1))[0]
    classes = np.asarray(model.classes_, dtype=int)
    order = np.argsort(-proba)[:5]
    top = [(VOCAB[int(classes[i])], float(proba[i])) for i in order]
    print(f"    true={after[r, c]!r} top5={top}")
    assert abs(proba.sum() - 1.0) < 1e-4

    # Anatomy — stage 2: feature-segment decomposition
    seg_spec = [
        ("patch", 0, 873),
        ("position", 873, 875),
        ("action", 875, 879),
        ("family", 879, 884),
        ("typed_char", 884, 981),
    ]
    seg_stats = {
        name: int((feats[lo:hi] != 0).sum())
        for name, lo, hi in seg_spec
        if hi <= len(feats)
    }
    print(f"    segment non-zero counts: {seg_stats}")
    assert sum(seg_stats.values()) > 0, "all segments zero — encoding broken"

    # Anatomy — stage 3: manual forward pass for hidden-layer activation
    W1 = model.coefs_[0]
    b1 = model.intercepts_[0]
    assert W1.shape[0] == len(feats), "coef_[0] shape mismatch"
    z1 = feats @ W1 + b1
    h1 = np.maximum(z1, 0.0)
    active = float((h1 > 0).mean())
    print(f"    hidden-layer: {h1.shape[0]} units, {100 * active:.0f}% active")

    # Anatomy — stage 5: feature ablation
    baseline_top = int(np.argmax(proba))
    baseline_prob = float(proba[baseline_top])
    ablations = {
        "patch": (0, 873),
        "action_family": (875, 884),
        "typed_char": (884, 981),
    }
    for name, (lo, hi) in ablations.items():
        x_ab = feats.copy()
        x_ab[lo:hi] = 0.0
        p_ab = model.predict_proba(x_ab.reshape(1, -1))[0]
        top_ab = int(np.argmax(p_ab))
        print(
            f"    ablate {name}: baseline top={VOCAB[int(classes[baseline_top])]!r}"
            f" ({100 * baseline_prob:.0f}%) → {VOCAB[int(classes[top_ab])]!r} ({100 * p_ab[top_ab]:.0f}%)"
        )

    print("[6/6] autoregressive rollout logic...")
    ep = generate_episode("whoami", COMMAND_VARIANTS["whoami"][0])
    true_frames = ep.frames
    pred_frames = [true_frames[0].copy()]
    all_positions = [(r, c) for r in range(rows_n) for c in range(cols_n)]
    per_step = []
    for t, action in enumerate(ep.actions):
        current = pred_frames[-1]
        Xb = np.asarray(
            [encode_cell_features(current, r, c, action, conditioning) for r, c in all_positions],
            dtype=np.float32,
        )
        cls = model.predict(Xb)
        nxt = current.copy()
        for (r, c), k in zip(all_positions, cls):
            nxt[r, c] = VOCAB[int(k)]
        pred_frames.append(nxt)
        per_step.append({
            "step": t + 1,
            "action": action.kind,
            "ar_acc": float((nxt == true_frames[t + 1]).sum()) / (rows_n * cols_n),
        })
    df = pd.DataFrame(per_step)
    print(df.to_string(index=False))
    assert df["ar_acc"].iloc[0] > 0.5, "first step should be easy"
    print("\nALL ANALYSIS CELL LOGIC OK")


if __name__ == "__main__":
    main()
