from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .gru_model import (
    EpisodeSequenceDataset,
    collate_records,
    summarize_rollout,
    set_seed,
)
from .toy_terminal import (
    Episode,
    CHAR_TO_IDX,
    VOCAB,
    encode_screen,
    decode_screen,
    PAD,
)

ConditionLevel = Literal["none", "family", "command"]
ACTION_KINDS = ["idle", "type_char", "backspace", "enter"]
ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_KINDS)}


@dataclass(frozen=True)
class TransformerConfig:
    char_emb_dim: int = 8
    screen_proj_dim: int = 128
    action_emb_dim: int = 8
    family_emb_dim: int = 8
    command_emb_dim: int = 8
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    ff_mult: int = 4
    max_steps: int = 64
    context_width: int = 32
    batch_size: int = 16
    epochs: int = 35
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def command_text_to_ids(text: str, width: int) -> np.ndarray:
    text = text[:width].ljust(width)
    return np.array([CHAR_TO_IDX.get(ch, CHAR_TO_IDX[PAD]) for ch in text], dtype=np.int64)


class ScreenTransformer(nn.Module):
    def __init__(self, rows: int, cols: int, num_families: int, config: TransformerConfig, condition_level: ConditionLevel):
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
        self.pos_emb = nn.Embedding(config.max_steps, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * config.ff_mult,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.out_norm = nn.LayerNorm(config.hidden_dim)
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
        x = self.input_proj(torch.cat(parts, dim=-1))
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_emb(positions)[None, :, :]
        return x

    def forward(self, batch: dict) -> torch.Tensor:
        x = self._encode_inputs(batch)
        T = x.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(x, mask=causal_mask)
        h = self.out_norm(h)
        return self.decoder(h).reshape(h.shape[0], h.shape[1], self.rows, self.cols, self.vocab_size)


class TransformerBaseline:
    def __init__(self, rows: int, cols: int, families: list[str], config: TransformerConfig, condition_level: ConditionLevel):
        self.rows = rows
        self.cols = cols
        self.family_to_idx = {name: i for i, name in enumerate(sorted(set(families)))}
        self.config = config
        self.condition_level = condition_level
        self.model = ScreenTransformer(rows, cols, len(self.family_to_idx), config, condition_level).to(config.device)

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
            preds = [episode.frames[0].copy()]
            current_frames = [encode_screen(episode.frames[0]).astype(np.int64)]
            family_value = self.family_to_idx.get(episode.family, 0)
            command_ids = command_text_to_ids(episode.command_text, self.config.context_width)
            for step, action in enumerate(episode.actions):
                seq_len = len(current_frames)
                batch = {
                    "screen": torch.from_numpy(np.stack(current_frames, axis=0))[None].long().to(self.config.device),
                    "action_kind": torch.tensor([[ACTION_TO_IDX[a.kind] for a in episode.actions[:seq_len]]], dtype=torch.long, device=self.config.device),
                    "typed_char": torch.tensor([[CHAR_TO_IDX.get(a.typed_char or PAD, CHAR_TO_IDX[PAD]) for a in episode.actions[:seq_len]]], dtype=torch.long, device=self.config.device),
                    "family_id": torch.tensor([[family_value] * seq_len], dtype=torch.long, device=self.config.device),
                    "command_ids": torch.from_numpy(np.repeat(command_ids[None, :], seq_len, axis=0))[None].long().to(self.config.device),
                    "noisy": torch.tensor([[[float(episode.noisy)]] * seq_len], dtype=torch.float32, device=self.config.device),
                }
                logits = self.model(batch)
                pred_tokens = logits[0, -1].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
                pred_screen = decode_screen(pred_tokens)
                preds.append(pred_screen)
                current_frames.append(pred_tokens)
        return preds


def evaluate_transformer(model: TransformerBaseline, episodes: list[Episode]) -> dict[str, float]:
    vals = [summarize_rollout(model.rollout(ep), ep) for ep in episodes]
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}


def action_kind_breakdown_transformer(model: TransformerBaseline, episodes: list[Episode]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[tuple[float, float, float]]] = {}
    for episode in episodes:
        pred_frames = model.rollout(episode)
        for step, action in enumerate(episode.actions):
            prev = episode.frames[step]
            pred = pred_frames[step + 1]
            truth = episode.frames[step + 1]
            from .toy_terminal import char_accuracy, changed_cell_accuracy, exact_line_accuracy
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
