from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
from torch import nn

from open_gen.model.image_encoders import ImageTokenEncoder
from open_gen.model.model_config import OpenGen1Config


@dataclass
class TokenSequence:
    tokens: torch.Tensor
    pad_mask: torch.Tensor
    block_ids: torch.Tensor
    positions: torch.Tensor


def _stable_hash(value: str) -> int:
    return int(hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest(), 16)


def _sinusoidal_embedding(values: torch.Tensor, dim: int, *, base: float = 10000.0) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros((*values.shape, dim), dtype=torch.float32)
    if dim <= 0:
        raise ValueError("dim must be positive")
    values = values.to(dtype=torch.float32)
    half_dim = dim // 2
    if half_dim == 0:
        return values.unsqueeze(-1).new_zeros((*values.shape, dim))
    freq_idx = torch.arange(half_dim, device=values.device, dtype=torch.float32)
    freqs = torch.exp(-math.log(base) * freq_idx / max(half_dim - 1, 1))
    angles = values.unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < dim:
        pad = emb.new_zeros((*emb.shape[:-1], dim - emb.shape[-1]))
        emb = torch.cat([emb, pad], dim=-1)
    return emb


def _image_local_position_embedding(num_tokens: int, dim: int, device: torch.device) -> torch.Tensor:
    if num_tokens == 0:
        return torch.zeros((0, dim), device=device, dtype=torch.float32)
    side = math.isqrt(num_tokens)
    prefix_tokens = 0
    grid_tokens = num_tokens
    if side * side != num_tokens:
        side = math.isqrt(num_tokens - 1) if num_tokens > 1 else 0
        if side * side == num_tokens - 1:
            prefix_tokens = 1
            grid_tokens = num_tokens - 1
        else:
            positions = torch.arange(num_tokens, device=device, dtype=torch.float32)
            return _sinusoidal_embedding(positions, dim)

    rows = torch.arange(side, device=device, dtype=torch.float32).repeat_interleave(side)
    cols = torch.arange(side, device=device, dtype=torch.float32).repeat(side)
    row_dim = dim // 2
    col_dim = dim - row_dim
    grid_emb = torch.cat(
        [
            _sinusoidal_embedding(rows, row_dim),
            _sinusoidal_embedding(cols, col_dim),
        ],
        dim=-1,
    )
    if prefix_tokens == 0:
        return grid_emb
    prefix_emb = torch.zeros((prefix_tokens, dim), device=device, dtype=torch.float32)
    return torch.cat([prefix_emb, grid_emb], dim=0)


class ContextTokenizer(nn.Module):
    def __init__(self, width: int, vocab_size: int, max_text_tokens: int) -> None:
        super().__init__()
        self.width = width
        self.max_text_tokens = max_text_tokens
        self.embedding = nn.Embedding(vocab_size, width)
        self.field_embedding = nn.Embedding(3, width)
        self.numeric_projs = nn.ModuleList([nn.LazyLinear(width) for _ in range(3)])

    def _string_tokens(self, text: str, field_idx: int) -> torch.Tensor:
        words = text.strip().split()
        if not words:
            words = ["<empty>"]
        words = words[: self.max_text_tokens]
        ids = [(_stable_hash(f"{field_idx}:{word}") % self.embedding.num_embeddings) for word in words]
        ids_t = torch.tensor(ids, device=self.embedding.weight.device)
        return self.embedding(ids_t) + self.field_embedding.weight[field_idx]

    def _numeric_tokens(self, value: Any, field_idx: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.embedding.weight.device).flatten()
        return self.numeric_projs[field_idx](tensor.unsqueeze(0)) + self.field_embedding.weight[field_idx]

    def forward(self, context: dict[str, list[Any | None]]) -> tuple[torch.Tensor, torch.Tensor]:
        field_order = ["goal", "embodiment", "action_repr"]
        batch_size = len(context[field_order[0]])
        sample_tokens: list[torch.Tensor] = []
        sample_masks: list[torch.Tensor] = []
        for batch_idx in range(batch_size):
            pieces: list[torch.Tensor] = []
            for field_idx, field_name in enumerate(field_order):
                value = context[field_name][batch_idx]
                if value is None:
                    continue
                if isinstance(value, str):
                    pieces.append(self._string_tokens(value, field_idx))
                else:
                    pieces.append(self._numeric_tokens(value, field_idx))
            if pieces:
                tokens = torch.cat([piece if piece.ndim == 2 else piece.unsqueeze(0) for piece in pieces], dim=0)
                mask = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
            else:
                tokens = torch.zeros((0, self.width), dtype=torch.float32, device=self.embedding.weight.device)
                mask = torch.zeros((0,), dtype=torch.bool, device=self.embedding.weight.device)
            sample_tokens.append(tokens)
            sample_masks.append(mask)
        max_tokens = max((tokens.shape[0] for tokens in sample_tokens), default=0)
        batch_tokens = torch.zeros((batch_size, max_tokens, self.width), dtype=torch.float32, device=self.embedding.weight.device)
        batch_masks = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=self.embedding.weight.device)
        for idx, (tokens, mask) in enumerate(zip(sample_tokens, sample_masks, strict=True)):
            batch_tokens[idx, : tokens.shape[0]] = tokens
            batch_masks[idx, : mask.shape[0]] = mask
        return batch_tokens, batch_masks


class MultimodalTokenizer(nn.Module):
    def __init__(self, config: OpenGen1Config) -> None:
        super().__init__()
        width = config.trunk.width
        self.config = config
        self.context_tokenizer = ContextTokenizer(width, config.context.vocab_size, config.context.max_text_tokens)
        self.image_encoder = ImageTokenEncoder(config.image_encoder)
        self.image_proj = nn.Linear(self.image_encoder.out_dim, width, bias=False)
        self.proprio_proj = nn.Linear(config.proprio_dim, width, bias=False) if config.proprio_dim > 0 else None
        self.force_proj = nn.Linear(config.force_dim, width, bias=False) if config.force_dim > 0 else None
        self.action_proj = nn.Linear(config.action_dim, width, bias=False)
        self.modality_embedding = nn.Embedding(6, width)

    def _encode_images(self, image_batch: torch.Tensor) -> torch.Tensor:
        if image_batch.shape[1] == 0:
            return torch.zeros(
                (image_batch.shape[0], 0, 0, self.config.trunk.width),
                device=image_batch.device,
                dtype=torch.float32,
            )
        batch, steps = image_batch.shape[:2]
        flat = image_batch.reshape(batch * steps, *image_batch.shape[2:])
        tokens = self.image_proj(self.image_encoder(flat))
        local_pos = _image_local_position_embedding(tokens.shape[1], self.config.trunk.width, tokens.device).to(tokens.dtype)
        tokens = tokens + local_pos.unsqueeze(0)
        return tokens.reshape(batch, steps, tokens.shape[1], tokens.shape[2])

    def _positions(self, batch: int, length: int, device: torch.device, start: int = 0) -> torch.Tensor:
        return torch.arange(start, start + length, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)

    def tokenize_context(self, context: dict[str, list[Any | None]], *, start_position: int = 0) -> TokenSequence:
        tokens, mask = self.context_tokenizer(context)
        batch = tokens.shape[0]
        positions = self._positions(batch, tokens.shape[1], tokens.device, start=start_position)
        block_ids = torch.zeros((batch, tokens.shape[1]), dtype=torch.long, device=tokens.device)
        return TokenSequence(tokens=tokens, pad_mask=mask, block_ids=block_ids, positions=positions)

    def tokenize_observations(
        self,
        observations: dict[str, torch.Tensor],
        *,
        start_block: int = 1,
        start_position: int = 0,
        past_action: dict[str, torch.Tensor] | None = None,
    ) -> TokenSequence:
        left_tokens = self._encode_images(observations["left_wrist_image"]) + self.modality_embedding.weight[0]
        right_tokens = self._encode_images(observations["right_wrist_image"]) + self.modality_embedding.weight[1]
        head_tokens = self._encode_images(observations["head_image"]) + self.modality_embedding.weight[2]

        batch, steps = observations["mask"].shape
        all_tokens: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        all_blocks: list[torch.Tensor] = []
        current_block = start_block

        for step in range(steps):
            step_tokens = [left_tokens[:, step], right_tokens[:, step]]
            step_masks = [
                observations["mask"][:, step : step + 1].expand(-1, left_tokens.shape[2]),
                observations["mask"][:, step : step + 1].expand(-1, right_tokens.shape[2]),
            ]
            if head_tokens.shape[2] > 0:
                step_tokens.append(head_tokens[:, step])
                step_masks.append(observations["head_mask"][:, step : step + 1].expand(-1, head_tokens.shape[2]))
            if self.proprio_proj is not None and observations["proprio"].shape[-1] > 0:
                step_tokens.append(self.proprio_proj(observations["proprio"][:, step]).unsqueeze(1) + self.modality_embedding.weight[3])
                step_masks.append(observations["proprio_mask"][:, step : step + 1])
            if self.force_proj is not None and observations["force"].shape[-1] > 0:
                step_tokens.append(self.force_proj(observations["force"][:, step]).unsqueeze(1) + self.modality_embedding.weight[4])
                step_masks.append(observations["force_mask"][:, step : step + 1])

            tokens = torch.cat(step_tokens, dim=1)
            mask = torch.cat(step_masks, dim=1)
            blocks = torch.full((batch, tokens.shape[1]), current_block, dtype=torch.long, device=tokens.device)
            all_tokens.append(tokens)
            all_masks.append(mask)
            all_blocks.append(blocks)
            current_block += 1

            if past_action is not None and past_action["value"].shape[1] > step:
                action_token = self.action_proj(past_action["value"][:, step]).unsqueeze(1) + self.modality_embedding.weight[5]
                action_mask = past_action["mask"][:, step : step + 1]
                action_block = torch.full((batch, 1), current_block, dtype=torch.long, device=tokens.device)
                all_tokens.append(action_token)
                all_masks.append(action_mask)
                all_blocks.append(action_block)
                current_block += 1

        if all_tokens:
            tokens = torch.cat(all_tokens, dim=1)
            mask = torch.cat(all_masks, dim=1)
            block_ids = torch.cat(all_blocks, dim=1)
            tokens = tokens + _sinusoidal_embedding(block_ids, self.config.trunk.width).to(tokens.dtype)
            positions = (block_ids - start_block + start_position).to(dtype=torch.long)
        else:
            device = observations["mask"].device
            tokens = torch.zeros((batch, 0, self.config.trunk.width), dtype=torch.float32, device=device)
            mask = torch.zeros((batch, 0), dtype=torch.bool, device=device)
            block_ids = torch.zeros((batch, 0), dtype=torch.long, device=device)
            positions = torch.zeros((batch, 0), dtype=torch.long, device=device)
        return TokenSequence(tokens=tokens, pad_mask=mask, block_ids=block_ids, positions=positions)
