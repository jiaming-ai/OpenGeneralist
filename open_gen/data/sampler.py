from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from open_gen.data.curriculum import CurriculumStage


@dataclass
class BucketSpec:
    min_T: int
    max_T: int


def make_buckets(boundaries: list[int]) -> list[BucketSpec]:
    if not boundaries:
        raise ValueError("bucket_boundaries cannot be empty")
    sorted_boundaries = sorted(set(boundaries))
    if sorted_boundaries[0] < 1:
        raise ValueError("bucket_boundaries must be positive integers")
    buckets: list[BucketSpec] = []
    prev = 0
    for upper in sorted_boundaries:
        buckets.append(BucketSpec(min_T=prev + 1, max_T=upper))
        prev = upper
    return buckets


class BucketedTokenBudgetBatchSampler(Sampler[list[tuple[int, int]]]):
    """Yield batches of (sample_index, num_past_frames) tuples.

    Each batch comes from a single bucket so the resulting tensors share a
    shape after collation. Per-step batch size is derived from a token budget
    so that compute and memory stay roughly constant across buckets. When
    `world_size > 1`, all ranks see the same bucket on the same step and only
    differ in which sample slice they get; this keeps DDP shape-balanced and
    loss reduction unbiased without per-rank token weighting.
    """

    def __init__(
        self,
        *,
        native_lengths: list[int],
        bucket_boundaries: list[int],
        tokens_per_frame: int,
        context_tokens: int,
        stage: CurriculumStage,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        self._native_lengths = list(native_lengths)
        self._buckets = make_buckets(list(bucket_boundaries))
        self._tokens_per_frame = max(1, int(tokens_per_frame))
        self._context_tokens = max(0, int(context_tokens))
        self._world_size = world_size
        self._rank = rank
        self._seed = seed
        self._drop_last = drop_last
        self._epoch = 0
        self._stage = stage
        self._rebuild_plan()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._rebuild_plan()

    def set_stage(self, stage: CurriculumStage) -> None:
        self._stage = stage
        self._rebuild_plan()

    def _effective_range(self, bucket: BucketSpec) -> tuple[int, int] | None:
        eff_min = max(bucket.min_T, self._stage.min_past_frames)
        eff_max = min(bucket.max_T, self._stage.max_past_frames)
        if eff_min > eff_max:
            return None
        return eff_min, eff_max

    def _global_batch_size(self, eff_max: int) -> int:
        per_step_tokens = self._context_tokens + eff_max * self._tokens_per_frame
        per_rank = max(1, self._stage.token_budget // max(1, per_step_tokens))
        return per_rank * self._world_size

    def _rebuild_plan(self) -> None:
        rng = np.random.default_rng(self._seed + self._epoch + 1)
        plan: list[tuple[int, int, int, np.ndarray]] = []
        for bucket_idx, bucket in enumerate(self._buckets):
            effective = self._effective_range(bucket)
            if effective is None:
                continue
            eff_min, eff_max = effective
            eligible = np.array(
                [idx for idx, length in enumerate(self._native_lengths) if length >= eff_min],
                dtype=np.int64,
            )
            if eligible.size == 0:
                continue
            global_bs = self._global_batch_size(eff_max)
            shuffled = eligible.copy()
            rng.shuffle(shuffled)
            num_full = shuffled.size // global_bs
            for batch_idx in range(num_full):
                start = batch_idx * global_bs
                indices = shuffled[start : start + global_bs]
                plan.append((bucket_idx, eff_min, eff_max, indices))
            remainder = shuffled.size - num_full * global_bs
            if remainder > 0 and not self._drop_last:
                tail = shuffled[num_full * global_bs :]
                pad_needed = global_bs - remainder
                fill = rng.choice(eligible, size=pad_needed, replace=True)
                plan.append((bucket_idx, eff_min, eff_max, np.concatenate([tail, fill])))
        rng.shuffle(plan)
        self._plan = plan

    def _draw_batch_T(self, eff_min: int, eff_max: int, rng: np.random.Generator) -> int:
        if eff_max <= eff_min:
            return eff_max
        if self._stage.frame_sampling == "max":
            return eff_max
        if self._stage.frame_sampling == "uniform":
            return int(rng.integers(low=eff_min, high=eff_max + 1))
        if self._stage.frame_sampling == "geometric":
            choices = np.arange(eff_min, eff_max + 1)
            weights = 0.5 ** np.arange(choices.size, dtype=np.float64)
            weights /= weights.sum()
            return int(rng.choice(choices, p=weights))
        raise ValueError(f"unknown frame_sampling {self._stage.frame_sampling}")

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        rng = np.random.default_rng(self._seed + self._epoch + 9973)
        for _, eff_min, eff_max, indices in self._plan:
            T_target = self._draw_batch_T(eff_min, eff_max, rng)
            global_bs = indices.size
            per_rank = global_bs // self._world_size
            start = self._rank * per_rank
            my_indices = indices[start : start + per_rank]
            yield [(int(idx), T_target) for idx in my_indices]

    def __len__(self) -> int:
        return len(self._plan)

    def current_target_lengths(self) -> list[int]:
        rng = np.random.default_rng(self._seed + self._epoch + 9973)
        return [self._draw_batch_T(eff_min, eff_max, rng) for _, eff_min, eff_max, _ in self._plan]
