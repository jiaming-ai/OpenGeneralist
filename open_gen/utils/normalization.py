from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class NormStats:
    mean: list[float]
    std: list[float]
    count: int
    q01: list[float] | None = None
    q99: list[float] | None = None

    @classmethod
    def from_tensors(
        cls,
        mean: torch.Tensor,
        std: torch.Tensor,
        count: int,
        *,
        q01: torch.Tensor | None = None,
        q99: torch.Tensor | None = None,
    ) -> "NormStats":
        return cls(
            mean=mean.tolist(),
            std=std.tolist(),
            count=count,
            q01=q01.tolist() if q01 is not None else None,
            q99=q99.tolist() if q99 is not None else None,
        )

    def mean_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(self.mean, dtype=torch.float32, device=device)

    def std_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(self.std, dtype=torch.float32, device=device)

    def q01_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        if self.q01 is None:
            raise ValueError("NormStats has no q01; recompute with track_quantiles=True")
        return torch.tensor(self.q01, dtype=torch.float32, device=device)

    def q99_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        if self.q99 is None:
            raise ValueError("NormStats has no q99; recompute with track_quantiles=True")
        return torch.tensor(self.q99, dtype=torch.float32, device=device)


class RunningStats:
    """Tracks per-dimension mean/std and (optionally) q01/q99 for normalization.

    Quantiles are computed over the buffered samples at finalize time. For
    development-scale datasets this is fine; for very large datasets, swap
    in a streaming quantile estimator later.
    """

    def __init__(self, dim: int, *, track_quantiles: bool = False) -> None:
        self.dim = dim
        self.count = 0
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.sum_sq = torch.zeros(dim, dtype=torch.float64)
        self._track_quantiles = track_quantiles
        self._samples: list[torch.Tensor] = []

    def update(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        values = values.detach().to(torch.float64)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if mask is not None:
            mask = mask.detach().to(torch.bool).reshape(-1)
            values = values.reshape(mask.shape[0], -1)[mask]
        else:
            values = values.reshape(-1, values.shape[-1])
        if values.numel() == 0:
            return
        self.count += values.shape[0]
        self.sum += values.sum(dim=0)
        self.sum_sq += (values * values).sum(dim=0)
        if self._track_quantiles:
            self._samples.append(values.to(torch.float32).cpu())

    def finalize(self) -> NormStats:
        if self.count == 0:
            mean = torch.zeros(self.dim, dtype=torch.float32)
            std = torch.ones(self.dim, dtype=torch.float32)
            q01 = torch.zeros(self.dim, dtype=torch.float32) if self._track_quantiles else None
            q99 = torch.ones(self.dim, dtype=torch.float32) if self._track_quantiles else None
            return NormStats.from_tensors(mean, std, 0, q01=q01, q99=q99)
        mean = self.sum / self.count
        var = torch.clamp(self.sum_sq / self.count - mean * mean, min=1e-8)
        std = torch.sqrt(var)
        q01 = q99 = None
        if self._track_quantiles:
            samples = torch.cat(self._samples, dim=0)
            q01 = torch.quantile(samples, 0.01, dim=0)
            q99 = torch.quantile(samples, 0.99, dim=0)
        return NormStats.from_tensors(
            mean.to(torch.float32),
            std.to(torch.float32),
            self.count,
            q01=q01,
            q99=q99,
        )


def save_stats(path: str | Path, stats: dict[str, NormStats]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: asdict(value) for key, value in stats.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_stats(path: str | Path) -> dict[str, NormStats]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: NormStats(**value) for key, value in payload.items()}


def zscore_normalize(x: torch.Tensor, stats: NormStats | None) -> torch.Tensor:
    if stats is None:
        return x
    mean = stats.mean_tensor(device=x.device)
    std = stats.std_tensor(device=x.device)
    dim = min(x.shape[-1], mean.shape[-1])
    y = x.clone()
    y[..., :dim] = (y[..., :dim] - mean[:dim]) / (std[:dim] + 1e-6)
    return y


def zscore_unnormalize(x: torch.Tensor, stats: NormStats | None) -> torch.Tensor:
    if stats is None:
        return x
    mean = stats.mean_tensor(device=x.device)
    std = stats.std_tensor(device=x.device)
    dim = min(x.shape[-1], mean.shape[-1])
    y = x.clone()
    y[..., :dim] = y[..., :dim] * (std[:dim] + 1e-6) + mean[:dim]
    return y


def quantile_normalize(x: torch.Tensor, stats: NormStats | None) -> torch.Tensor:
    if stats is None or stats.q01 is None or stats.q99 is None:
        return x
    q01 = stats.q01_tensor(device=x.device)
    q99 = stats.q99_tensor(device=x.device)
    dim = min(x.shape[-1], q01.shape[-1])
    y = x.clone()
    y[..., :dim] = (y[..., :dim] - q01[:dim]) / (q99[:dim] - q01[:dim] + 1e-6) * 2.0 - 1.0
    return y


def quantile_unnormalize(x: torch.Tensor, stats: NormStats | None) -> torch.Tensor:
    if stats is None or stats.q01 is None or stats.q99 is None:
        return x
    q01 = stats.q01_tensor(device=x.device)
    q99 = stats.q99_tensor(device=x.device)
    dim = min(x.shape[-1], q01.shape[-1])
    y = x.clone()
    y[..., :dim] = (y[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6) + q01[:dim]
    return y


def normalize(x: torch.Tensor, stats: NormStats | None, *, mode: str) -> torch.Tensor:
    if mode == "zscore":
        return zscore_normalize(x, stats)
    if mode == "quantile":
        return quantile_normalize(x, stats)
    raise ValueError(f"Unsupported norm mode: {mode}")


def unnormalize(x: torch.Tensor, stats: NormStats | None, *, mode: str) -> torch.Tensor:
    if mode == "zscore":
        return zscore_unnormalize(x, stats)
    if mode == "quantile":
        return quantile_unnormalize(x, stats)
    raise ValueError(f"Unsupported norm mode: {mode}")
