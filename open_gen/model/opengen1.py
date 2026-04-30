from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from open_gen.model.flow_action_expert import FlowActionExpert
from open_gen.model.model_config import OpenGen1Config
from open_gen.model.modules import BlockCausalTransformer, KVCache, masked_mean
from open_gen.model.tokenizers import MultimodalTokenizer, TokenSequence


@dataclass
class StreamingState:
    caches: list[KVCache | None]
    memory_tokens: torch.Tensor
    memory_mask: torch.Tensor
    next_block: int
    next_position: int


class OpenGen1Model(nn.Module):
    def __init__(self, config: OpenGen1Config) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = MultimodalTokenizer(config)
        self.trunk = BlockCausalTransformer(
            dim=config.trunk.width,
            depth=config.trunk.depth,
            num_heads=config.trunk.num_heads,
            mlp_ratio=config.trunk.mlp_ratio,
            rope_base=config.trunk.rope_base,
            dropout=config.trunk.dropout,
        )
        self.action_expert = FlowActionExpert(
            width=config.action_expert.width,
            action_dim=config.action_dim,
            depth=config.action_expert.depth,
            num_heads=config.action_expert.num_heads,
            mlp_ratio=config.action_expert.mlp_ratio,
            flow_steps=config.action_expert.flow_steps,
            time_inject=config.action_expert.time_inject,
        )
        if config.action_expert.width != config.trunk.width:
            self.memory_adapter = nn.Linear(config.trunk.width, config.action_expert.width, bias=False)
        else:
            self.memory_adapter = nn.Identity()
        self.world_latent_head = nn.Linear(config.trunk.width, config.world_model.latent_dim, bias=False)
        self.future_proprio_head = nn.Linear(config.trunk.width, config.proprio_dim, bias=False) if config.proprio_dim > 0 else None
        self.future_force_head = nn.Linear(config.trunk.width, config.force_dim, bias=False) if config.force_dim > 0 else None

    def _encode_sequence(self, sequence: TokenSequence) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.trunk(
            sequence.tokens,
            block_ids=sequence.block_ids,
            positions=sequence.positions,
            pad_mask=sequence.pad_mask,
        )
        return hidden, sequence.pad_mask

    def _build_policy_sequence(self, batch: dict[str, Any]) -> TokenSequence:
        context = self.tokenizer.tokenize_context(batch["context"])
        observations = self.tokenizer.tokenize_observations(
            batch["observations"],
            start_block=1,
            start_position=context.positions.shape[1],
        )
        return self._concat_sequences(context, observations)

    def _build_world_model_sequence(self, batch: dict[str, Any]) -> TokenSequence:
        context = self.tokenizer.tokenize_context(batch["context"])
        observations = self.tokenizer.tokenize_observations(
            batch["observations"],
            start_block=1,
            start_position=context.positions.shape[1],
            past_action=batch["past_action"],
        )
        return self._concat_sequences(context, observations)

    def _concat_sequences(self, left: TokenSequence, right: TokenSequence) -> TokenSequence:
        tokens = torch.cat([left.tokens, right.tokens], dim=1)
        pad_mask = torch.cat([left.pad_mask, right.pad_mask], dim=1)
        block_ids = torch.cat([left.block_ids, right.block_ids], dim=1)
        positions = torch.cat([left.positions, right.positions], dim=1)
        return TokenSequence(tokens=tokens, pad_mask=pad_mask, block_ids=block_ids, positions=positions)

    def _encode_future_target(self, batch: dict[str, Any]) -> torch.Tensor | None:
        if not batch["future"]["mask"].any():
            return None
        future_context = {"goal": [None] * batch["future"]["mask"].shape[0], "embodiment": [None] * batch["future"]["mask"].shape[0], "action_repr": [None] * batch["future"]["mask"].shape[0]}
        context = self.tokenizer.tokenize_context(future_context)
        observations = self.tokenizer.tokenize_observations(batch["future"], start_block=1, start_position=0)
        sequence = self._concat_sequences(context, observations)
        hidden, mask = self._encode_sequence(sequence)
        return self.world_latent_head(masked_mean(hidden, mask)).detach()

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        policy_sequence = self._build_policy_sequence(batch)
        policy_hidden, policy_mask = self._encode_sequence(policy_sequence)
        memory = self.memory_adapter(policy_hidden)
        _, action_loss = self.action_expert(
            memory=memory,
            memory_mask=policy_mask,
            actions=batch["action_target"]["value"],
            action_mask=batch["action_target"]["mask"],
        )
        total_loss = action_loss
        outputs: dict[str, torch.Tensor] = {
            "loss": action_loss,
            "action_loss": action_loss.detach(),
        }

        if self.config.world_model.enabled:
            future_target = self._encode_future_target(batch)
            if future_target is not None and batch["past_action"]["mask"].any():
                wm_sequence = self._build_world_model_sequence(batch)
                wm_hidden, wm_mask = self._encode_sequence(wm_sequence)
                summary = masked_mean(wm_hidden, wm_mask)
                future_pred = self.world_latent_head(summary)
                future_latent_loss = torch.mean((future_pred - future_target) ** 2)
                total_loss = total_loss + self.config.world_model.future_latent_weight * future_latent_loss
                outputs["future_latent_loss"] = future_latent_loss.detach()

                if self.future_proprio_head is not None and batch["future"]["proprio_mask"].any():
                    pred = self.future_proprio_head(summary).unsqueeze(1)
                    target = batch["future"]["proprio"][:, :1]
                    mask = batch["future"]["proprio_mask"][:, :1].unsqueeze(-1)
                    proprio_loss = (((pred - target) ** 2) * mask).sum() / torch.clamp(mask.sum() * target.shape[-1], min=1)
                    total_loss = total_loss + self.config.world_model.proprio_weight * proprio_loss
                    outputs["future_proprio_loss"] = proprio_loss.detach()
                if self.future_force_head is not None and batch["future"]["force_mask"].any():
                    pred = self.future_force_head(summary).unsqueeze(1)
                    target = batch["future"]["force"][:, :1]
                    mask = batch["future"]["force_mask"][:, :1].unsqueeze(-1)
                    force_loss = (((pred - target) ** 2) * mask).sum() / torch.clamp(mask.sum() * target.shape[-1], min=1)
                    total_loss = total_loss + self.config.world_model.force_weight * force_loss
                    outputs["future_force_loss"] = force_loss.detach()

        outputs["loss"] = total_loss
        return outputs

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Any], action_horizon: int | None = None) -> torch.Tensor:
        policy_sequence = self._build_policy_sequence(batch)
        policy_hidden, policy_mask = self._encode_sequence(policy_sequence)
        return self.action_expert.sample(self.memory_adapter(policy_hidden), policy_mask, action_horizon or batch["action_target"]["value"].shape[1])

    @torch.no_grad()
    def append_observation(
        self,
        state: StreamingState | None,
        batch: dict[str, Any],
        *,
        include_context: bool = False,
    ) -> StreamingState:
        if state is None:
            caches = [None] * len(self.trunk.blocks)
            memory_tokens = torch.zeros((batch["observations"]["mask"].shape[0], 0, self.config.trunk.width), device=batch["observations"]["mask"].device)
            memory_mask = torch.zeros((batch["observations"]["mask"].shape[0], 0), dtype=torch.bool, device=batch["observations"]["mask"].device)
            next_block = 1
            next_position = 0
        else:
            caches = state.caches
            memory_tokens = state.memory_tokens
            memory_mask = state.memory_mask
            next_block = state.next_block
            next_position = state.next_position

        sequence_parts: list[TokenSequence] = []
        if include_context:
            context = self.tokenizer.tokenize_context(batch["context"], start_position=next_position)
            next_position += context.tokens.shape[1]
            sequence_parts.append(context)
        observations = self.tokenizer.tokenize_observations(
            batch["observations"],
            start_block=next_block,
            start_position=next_position,
        )
        if observations.block_ids.numel() > 0:
            num_blocks = int(observations.block_ids.max().item()) - next_block + 1
            next_block = int(observations.block_ids.max().item()) + 1
            next_position += num_blocks
        sequence_parts.append(observations)
        sequence = sequence_parts[0] if len(sequence_parts) == 1 else self._concat_sequences(sequence_parts[0], sequence_parts[1])
        hidden, next_caches = self.trunk(
            sequence.tokens,
            block_ids=sequence.block_ids,
            positions=sequence.positions,
            pad_mask=sequence.pad_mask,
            caches=caches,
            update_cache=True,
        )
        memory_tokens = torch.cat([memory_tokens, hidden], dim=1)
        memory_mask = torch.cat([memory_mask, sequence.pad_mask], dim=1)
        return StreamingState(
            caches=next_caches,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            next_block=next_block,
            next_position=next_position,
        )

    @torch.no_grad()
    def decode_from_state(self, state: StreamingState, action_horizon: int) -> torch.Tensor:
        memory = self.memory_adapter(state.memory_tokens)
        return self.action_expert.sample(memory, state.memory_mask, action_horizon)
