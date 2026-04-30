from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def enable_sdpa_kernels() -> None:
    if not torch.cuda.is_available() or not hasattr(torch.backends, "cuda"):
        return
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class AdaRMSNorm(nn.Module):
    """RMSNorm with AdaLN-Zero modulation, matching openpi pi05's adaRMS.

    Produces normalized * (1 + scale) + shift and a gate term, all driven by
    a per-batch conditioning vector. The modulation projection is zero-init,
    so blocks start as identity (gate=0) and the model has to learn nonzero
    gates during training.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(cond_dim, 3 * dim, bias=True)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x * rms
        modulation = self.modulation(cond).unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        return normed * (1 + scale) + shift, gate


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return x + y
    return x + gate * y


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.cache_dtype: torch.dtype | None = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(positions, self.inv_freq.to(device))
        cos = torch.stack([torch.cos(angles), torch.cos(angles)], dim=-1).reshape(seq_len, self.dim)
        sin = torch.stack([torch.sin(angles), torch.sin(angles)], dim=-1).reshape(seq_len, self.dim)
        self.cos_cached = cos.to(dtype=dtype)
        self.sin_cached = sin.to(dtype=dtype)
        self.cache_dtype = dtype

    def forward(self, positions: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        max_position = int(positions.max().item()) + 1 if positions.numel() > 0 else 0
        needs_refresh = (
            self.cos_cached.numel() == 0
            or self.cos_cached.device != positions.device
            or self.cos_cached.shape[0] < max_position
            or self.cache_dtype != dtype
        )
        if needs_refresh:
            self._build_cache(max_position, positions.device, dtype)
        flat_positions = positions.reshape(-1)
        cos = self.cos_cached.index_select(0, flat_positions).view(*positions.shape, self.dim)
        sin = self.sin_cached.index_select(0, flat_positions).view(*positions.shape, self.dim)
        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)
    return (x * cos.unsqueeze(1)) + (x_rot * sin.unsqueeze(1))


@dataclass
class KVCache:
    key: torch.Tensor
    value: torch.Tensor


def build_block_causal_mask(query_blocks: torch.Tensor, key_blocks: torch.Tensor) -> torch.Tensor:
    return key_blocks[:, None, :] <= query_blocks[:, :, None]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_base: float, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
        update_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary(q, rope_cos, rope_sin)
        k = apply_rotary(k, rope_cos, rope_sin)

        if kv_cache is not None:
            key = torch.cat([kv_cache.key, k], dim=2)
            value = torch.cat([kv_cache.value, v], dim=2)
        else:
            key, value = k, v

        attn = F.scaled_dot_product_attention(
            q,
            key,
            value,
            attn_mask=attn_mask[:, None, :, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = attn.transpose(1, 2).contiguous().view(batch, tokens, dim)
        next_cache = KVCache(key.detach(), value.detach()) if update_cache else None
        return self.out(out), next_cache


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def compute_kv(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.kv(memory).view(memory.shape[0], memory.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return kv[0].contiguous(), kv[1].contiguous()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None,
        mask: torch.Tensor,
        *,
        kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q = self.q(x).view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        if kv is None:
            assert memory is not None
            k, v = self.compute_kv(memory)
        else:
            k, v = kv
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = attn.transpose(1, 2).contiguous().view(batch, tokens, dim)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, rope_base: float, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, rope_base, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
        update_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, next_cache = self.attn(
            self.norm1(x),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
            update_cache=update_cache,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, next_cache


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        rope_base: float,
        dropout: float = 0.0,
        *,
        adarms: bool = False,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.adarms = adarms
        if adarms:
            if cond_dim is None:
                raise ValueError("cond_dim must be provided when adarms=True")
            self.self_norm = AdaRMSNorm(dim, cond_dim)
            self.cross_norm = AdaRMSNorm(dim, cond_dim)
            self.mlp_norm = AdaRMSNorm(dim, cond_dim)
        else:
            self.self_norm = RMSNorm(dim)
            self.cross_norm = RMSNorm(dim)
            self.mlp_norm = RMSNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, rope_base, dropout)
        self.cross_attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.mlp = SwiGLU(dim, hidden)

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        self_mask: torch.Tensor,
        memory: torch.Tensor | None,
        memory_mask: torch.Tensor,
        memory_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        adarms_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.adarms:
            normed_self, gate_self = self.self_norm(x, adarms_cond)
            self_out, _ = self.self_attn(normed_self, rope_cos=rope_cos, rope_sin=rope_sin, attn_mask=self_mask)
            x = gated_residual(x, self_out, gate_self)

            normed_cross, gate_cross = self.cross_norm(x, adarms_cond)
            cross_out = self.cross_attn(normed_cross, memory, memory_mask, kv=memory_kv)
            x = gated_residual(x, cross_out, gate_cross)

            normed_mlp, gate_mlp = self.mlp_norm(x, adarms_cond)
            x = gated_residual(x, self.mlp(normed_mlp), gate_mlp)
            return x

        self_out, _ = self.self_attn(self.self_norm(x), rope_cos=rope_cos, rope_sin=rope_sin, attn_mask=self_mask)
        x = x + self_out
        x = x + self.cross_attn(self.cross_norm(x), memory, memory_mask, kv=memory_kv)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class BlockCausalTransformer(nn.Module):
    def __init__(self, *, dim: int, depth: int, num_heads: int, mlp_ratio: float, rope_base: float, dropout: float) -> None:
        super().__init__()
        enable_sdpa_kernels()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, rope_base=rope_base, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        pad_mask: torch.Tensor,
        caches: list[KVCache | None] | None = None,
        update_cache: bool = False,
    ) -> tuple[torch.Tensor, list[KVCache | None]]:
        if caches is None:
            caches = [None] * len(self.blocks)
        if len(caches) != len(self.blocks):
            raise ValueError("Cache depth mismatch")
        if any(cache is not None for cache in caches):
            past_len = next(cache.key.shape[2] for cache in caches if cache is not None)
            prefix_mask = torch.ones(tokens.shape[0], tokens.shape[1], past_len, dtype=torch.bool, device=tokens.device)
            local_mask = build_block_causal_mask(block_ids, block_ids)
            current_key_mask = pad_mask[:, None, :].expand(-1, tokens.shape[1], -1)
            attn_mask = torch.cat([prefix_mask, local_mask & current_key_mask], dim=-1)
        else:
            attn_mask = build_block_causal_mask(block_ids, block_ids)
            attn_mask = attn_mask & pad_mask[:, None, :]
        attn_mask = attn_mask & pad_mask[:, :, None]

        next_caches: list[KVCache | None] = []
        x = tokens
        rope_cos, rope_sin = self.blocks[0].attn.rope(positions, dtype=x.dtype)
        for block, cache in zip(self.blocks, caches, strict=True):
            x, next_cache = block(
                x,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                attn_mask=attn_mask,
                kv_cache=cache,
                update_cache=update_cache,
            )
            next_caches.append(next_cache if update_cache else cache)
        return self.norm(x), next_caches


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    denom = torch.clamp(weights.sum(dim=1), min=1.0)
    return (x * weights).sum(dim=1) / denom
