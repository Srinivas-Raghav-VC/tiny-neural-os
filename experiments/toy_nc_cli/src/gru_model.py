from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from .toy_terminal import (
    Episode,
    CHAR_TO_IDX,
    VOCAB,
    encode_screen,
    decode_screen,
    PAD,
    char_accuracy,
    changed_cell_accuracy,
    exact_line_accuracy,
)

ConditionLevel = Literal["none", "family", "command"]
ACTION_KINDS = ["idle", "type_char", "backspace", "enter"]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_KINDS)}


@dataclass(frozen=True)
class GRUConfig:
    char_emb_dim: int = 8
    screen_proj_dim: int = 128
    action_emb_dim: int = 8
    family_emb_dim: int = 8
    command_emb_dim: int = 8
    hidden_dim: int = 192
    context_width: int = 32
    batch_size: int = 16
    epochs: int = 30
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def command_text_to_ids(text: str, width: int) -> np.ndarray:
    text = text[:width].ljust(width)
    return np.array([CHAR_TO_IDX.get(ch, CHAR_TO_IDX[PAD]) for ch in text], dtype=np.int64)


class EpisodeSequenceDataset(Dataset):
    def __init__(self, episodes: list[Episode], config: GRUConfig, family_to_idx: dict[str, int]):
        self.records = [self._encode_episode(ep, config, family_to_idx) for ep in episodes]

    @staticmethod
    def _encode_episode(episode: Episode, config: GRUConfig, family_to_idx: dict[str, int]) -> dict:
        frames = np.stack([encode_screen(frame) for frame in episode.frames], axis=0)
        T = len(episode.actions)
        action_kind = np.zeros((T,), dtype=np.int64)
        typed_char = np.zeros((T,), dtype=np.int64)
        family_ids = np.full((T,), family_to_idx.get(episode.family, 0), dtype=np.int64)
        cmd_ids = np.repeat(command_text_to_ids(episode.command_text, config.context_width)[None, :], T, axis=0)
        noisy = np.full((T, 1), float(episode.noisy), dtype=np.float32)
        for i, action in enumerate(episode.actions):
            action_kind[i] = ACTION_TO_IDX[action.kind]
            typed_char[i] = CHAR_TO_IDX.get(action.typed_char or PAD, CHAR_TO_IDX[PAD])
        return {
            "frames": frames,
            "action_kind": action_kind,
            "typed_char": typed_char,
            "family_id": family_ids,
            "command_ids": cmd_ids,
            "noisy": noisy,
            "length": T,
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_records(records: list[dict]) -> dict:
    B = len(records)
    max_T = max(r["length"] for r in records)
    rows, cols = records[0]["frames"].shape[1:]
    context_width = records[0]["command_ids"].shape[1]

    screen = np.zeros((B, max_T, rows, cols), dtype=np.int64)
    target = np.zeros((B, max_T, rows, cols), dtype=np.int64)
    action_kind = np.zeros((B, max_T), dtype=np.int64)
    typed_char = np.zeros((B, max_T), dtype=np.int64)
    family_id = np.zeros((B, max_T), dtype=np.int64)
    command_ids = np.zeros((B, max_T, context_width), dtype=np.int64)
    noisy = np.zeros((B, max_T, 1), dtype=np.float32)
    mask = np.zeros((B, max_T), dtype=np.bool_)

    for i, r in enumerate(records):
        T = r["length"]
        screen[i, :T] = r["frames"][:-1]
        target[i, :T] = r["frames"][1:]
        action_kind[i, :T] = r["action_kind"]
        typed_char[i, :T] = r["typed_char"]
        family_id[i, :T] = r["family_id"]
        command_ids[i, :T] = r["command_ids"]
        noisy[i, :T] = r["noisy"]
        mask[i, :T] = True

    return {
        "screen": torch.from_numpy(screen),
        "target": torch.from_numpy(target),
        "action_kind": torch.from_numpy(action_kind),
        "typed_char": torch.from_numpy(typed_char),
        "family_id": torch.from_numpy(family_id),
        "command_ids": torch.from_numpy(command_ids),
        "noisy": torch.from_numpy(noisy),
        "mask": torch.from_numpy(mask),
    }


class ScreenGRU(nn.Module):
    def __init__(self, rows: int, cols: int, num_families: int, config: GRUConfig, condition_level: ConditionLevel):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.config = config
        self.condition_level = condition_level
        self.vocab_size = len(VOCAB)

        self.char_emb = nn.Embedding(self.vocab_size, config.char_emb_dim)
        self.screen_proj = nn.Linear(rows * cols * config.char_emb_dim, config.screen_proj_dim)
        self.action_emb = nn.Embedding(len(ACTION_KINDS), config.action_emb_dim)
        self.typed_emb = nn.Embedding(self.vocab_size, config.action_emb_dim)
        self.family_emb = nn.Embedding(max(num_families, 1), config.family_emb_dim)
        self.command_emb = nn.Embedding(self.vocab_size, config.command_emb_dim)

        input_dim = config.screen_proj_dim + 1
        if condition_level in {"family", "command"}:
            input_dim += config.action_emb_dim + config.family_emb_dim
        if condition_level == "command":
            input_dim += config.action_emb_dim + config.command_emb_dim

        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.gru = nn.GRU(config.hidden_dim, config.hidden_dim, batch_first=True)
        self.decoder = nn.Linear(config.hidden_dim, rows * cols * self.vocab_size)

    def _encode_inputs(self, batch: dict) -> torch.Tensor:
        screen = batch["screen"]
        B, T, R, C = screen.shape
        screen_emb = self.char_emb(screen).reshape(B, T, R * C * self.config.char_emb_dim)
        screen_vec = self.screen_proj(screen_emb)
        parts = [screen_vec, batch["noisy"]]
        if self.condition_level in {"family", "command"}:
            parts.append(self.action_emb(batch["action_kind"]))
            parts.append(self.family_emb(batch["family_id"]))
        if self.condition_level == "command":
            parts.append(self.typed_emb(batch["typed_char"]))
            parts.append(self.command_emb(batch["command_ids"]).mean(dim=2))
        return self.input_proj(torch.cat(parts, dim=-1))

    def forward(self, batch: dict) -> torch.Tensor:
        x = self._encode_inputs(batch)
        h, _ = self.gru(x)
        return self.decoder(h).reshape(h.shape[0], h.shape[1], self.rows, self.cols, self.vocab_size)

    def step(self, batch: dict, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._encode_inputs(batch)
        h, hidden = self.gru(x, hidden)
        logits = self.decoder(h[:, 0]).reshape(batch["screen"].shape[0], self.rows, self.cols, self.vocab_size)
        return logits, hidden


class GRUBaseline:
    def __init__(self, rows: int, cols: int, families: list[str], config: GRUConfig, condition_level: ConditionLevel):
        self.rows = rows
        self.cols = cols
        self.family_to_idx = {name: i for i, name in enumerate(sorted(set(families)))}
        self.config = config
        self.condition_level = condition_level
        self.model = ScreenGRU(rows, cols, len(self.family_to_idx), config, condition_level).to(config.device)

    def _device_batch(self, batch: dict) -> dict:
        return {k: v.to(self.config.device) for k, v in batch.items()}

    def fit(self, episodes: list[Episode]) -> list[float]:
        set_seed(self.config.seed)
        ds = EpisodeSequenceDataset(episodes, self.config, self.family_to_idx)
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True, collate_fn=collate_records)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        losses: list[float] = []
        self.model.train()
        for _ in range(self.config.epochs):
            running = 0.0
            count = 0
            for batch in loader:
                batch = self._device_batch(batch)
                logits = self.model(batch)
                mask = batch["mask"]
                logits_flat = logits[mask].reshape(-1, logits.shape[-1])
                target_flat = batch["target"][mask].reshape(-1)
                loss = F.cross_entropy(logits_flat, target_flat)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                running += float(loss.item())
                count += 1
            losses.append(running / max(count, 1))
        return losses

    def rollout(self, episode: Episode) -> list[np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            current = torch.from_numpy(encode_screen(episode.frames[0]).astype(np.int64))[None].to(self.config.device)
            family_id = torch.tensor([[self.family_to_idx.get(episode.family, 0)]], dtype=torch.long, device=self.config.device)
            command_ids = torch.from_numpy(command_text_to_ids(episode.command_text, self.config.context_width))[None].long().to(self.config.device)
            noisy = torch.tensor([[[float(episode.noisy)]]], dtype=torch.float32, device=self.config.device)
            hidden = None
            preds = [episode.frames[0].copy()]
            for action in episode.actions:
                batch = {
                    "screen": current[:, None, :, :],
                    "action_kind": torch.tensor([[ACTION_TO_IDX[action.kind]]], dtype=torch.long, device=self.config.device),
                    "typed_char": torch.tensor([[CHAR_TO_IDX.get(action.typed_char or PAD, CHAR_TO_IDX[PAD])]], dtype=torch.long, device=self.config.device),
                    "family_id": family_id,
                    "command_ids": command_ids[:, None, :],
                    "noisy": noisy,
                }
                logits, hidden = self.model.step(batch, hidden)
                pred_tokens = logits.argmax(dim=-1)[0].detach().cpu().numpy().astype(np.int64)
                pred_screen = decode_screen(pred_tokens)
                preds.append(pred_screen)
                current = torch.from_numpy(pred_tokens)[None].to(self.config.device)
        return preds


def summarize_rollout(pred_frames: list[np.ndarray], episode: Episode) -> dict[str, float]:
    char_scores = []
    changed_scores = []
    line_scores = []
    for prev, pred, truth in zip(episode.frames[:-1], pred_frames[1:], episode.frames[1:]):
        char_scores.append(char_accuracy(pred, truth))
        changed_scores.append(changed_cell_accuracy(prev, pred, truth))
        line_scores.append(exact_line_accuracy(pred, truth))
    return {
        "char_acc": float(np.mean(char_scores)),
        "changed_acc": float(np.mean(changed_scores)),
        "exact_line_acc": float(np.mean(line_scores)),
    }


def evaluate_gru(model: GRUBaseline, episodes: list[Episode]) -> dict[str, float]:
    vals = [summarize_rollout(model.rollout(ep), ep) for ep in episodes]
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}


def action_kind_breakdown_gru(model: GRUBaseline, episodes: list[Episode]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[tuple[float, float, float]]] = {}
    for episode in episodes:
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
    out = {}
    for kind, vals in buckets.items():
        arr = np.asarray(vals, dtype=np.float32)
        out[kind] = {
            "char_acc": float(arr[:, 0].mean()),
            "changed_acc": float(arr[:, 1].mean()),
            "exact_line_acc": float(arr[:, 2].mean()),
            "count": float(len(vals)),
        }
    return out
