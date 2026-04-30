from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from open_gen.model.modules import CrossAttentionBlock


def posemb_sincos(time: torch.Tensor, dim: int, *, min_period: float = 4e-3, max_period: float = 4.0) -> torch.Tensor:
    """Sine-cosine positional embedding tuned for scalar t in [0, 1].

    Period range matches openpi (`min_period=4e-3`, `max_period=4.0`), so the
    embedding is sensitive across the unit interval rather than wasting
    capacity on much longer periods like the standard transformer scaling.
    """
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    half_dim = dim // 2
    fraction = torch.linspace(0.0, 1.0, half_dim, device=time.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    angles = time.to(torch.float32).unsqueeze(-1) * (2 * math.pi / period).unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(dtype=time.dtype)


class FlowActionExpert(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        action_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        flow_steps: int,
        time_inject: str = "adarms",
    ) -> None:
        super().__init__()
        if time_inject not in {"add", "adarms"}:
            raise ValueError(f"Unsupported time_inject mode: {time_inject}")
        self.width = width
        self.action_dim = action_dim
        self.flow_steps = flow_steps
        self.time_inject = time_inject
        self.action_in = nn.Linear(action_dim, width, bias=False)
        self.action_out = nn.Linear(width, action_dim, bias=False)
        self.time_in = nn.Linear(width, width, bias=False)
        self.time_out = nn.Linear(width, width, bias=False)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=width,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_base=10000.0,
                    adarms=(time_inject == "adarms"),
                    cond_dim=width if time_inject == "adarms" else None,
                )
                for _ in range(depth)
            ]
        )

    def _sample_time(self, batch: int, device: torch.device) -> torch.Tensor:
        dist = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device))
        return dist.sample((batch,)).clamp_(1e-3, 1.0)

    def _time_features(self, time: torch.Tensor) -> torch.Tensor:
        time_emb = posemb_sincos(time, self.width)
        if self.time_inject == "adarms":
            return F.silu(self.time_out(F.silu(self.time_in(time_emb))))
        return self.time_out(F.silu(self.time_in(time_emb)))

    def _denoise(
        self,
        memory: torch.Tensor | None,
        memory_mask: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        memory_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        time_features = self._time_features(time)
        if self.time_inject == "adarms":
            x = self.action_in(noisy_actions)
            adarms_cond = time_features
        else:
            x = self.action_in(noisy_actions) + time_features.unsqueeze(1)
            adarms_cond = None
        positions = torch.arange(x.shape[1], device=x.device, dtype=torch.long).unsqueeze(0).expand(x.shape[0], -1)
        self_mask = torch.ones((x.shape[0], x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device)
        self_mask = self_mask & action_mask[:, :, None] & action_mask[:, None, :]
        rope_cos, rope_sin = self.blocks[0].self_attn.rope(positions, dtype=x.dtype)
        for idx, block in enumerate(self.blocks):
            x = block(
                x,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                self_mask=self_mask,
                memory=memory,
                memory_mask=memory_mask,
                memory_kv=memory_kvs[idx] if memory_kvs is not None else None,
                adarms_cond=adarms_cond,
            )
        return self.action_out(x)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time = self._sample_time(actions.shape[0], actions.device)
        noise = torch.randn_like(actions)
        time_view = time[:, None, None]
        noisy_actions = time_view * noise + (1.0 - time_view) * actions
        target_vector_field = noise - actions
        pred_vector_field = self._denoise(memory, memory_mask, noisy_actions, time, action_mask)
        diff = (pred_vector_field - target_vector_field) ** 2
        loss = (diff * action_mask.unsqueeze(-1)).sum() / torch.clamp(action_mask.sum() * actions.shape[-1], min=1)
        return pred_vector_field, loss

    @torch.no_grad()
    def sample(self, memory: torch.Tensor, memory_mask: torch.Tensor, action_horizon: int) -> torch.Tensor:
        actions = torch.randn(memory.shape[0], action_horizon, self.action_dim, device=memory.device, dtype=memory.dtype)
        action_mask = torch.ones(memory.shape[0], action_horizon, dtype=torch.bool, device=memory.device)
        # Memory is constant across denoising steps, so each block's K/V from cross-attention
        # is computed once and reused across every Euler step.
        memory_kvs = [block.cross_attn.compute_kv(memory) for block in self.blocks]
        step_size = 1.0 / self.flow_steps
        for step in range(self.flow_steps, 0, -1):
            time = torch.full((memory.shape[0],), step / self.flow_steps, device=memory.device, dtype=memory.dtype)
            velocity = self._denoise(
                memory=None,
                memory_mask=memory_mask,
                noisy_actions=actions,
                time=time,
                action_mask=action_mask,
                memory_kvs=memory_kvs,
            )
            actions = actions - step_size * velocity
        return actions
