from __future__ import annotations

from typing import Any, Iterator

from torch.utils.data import Dataset

from open_gen.data.collate import MultimodalCollator
from open_gen.data.sampler import BucketedTokenBudgetBatchSampler


class BucketedDataLoader:
    """Lightweight loader that pairs a bucketed batch sampler with a collator.

    Per batch, the sampler emits a list of ``(idx, T_target)`` tuples sharing
    the same ``T_target``. We set ``collator.pad_to_obs_steps`` to that target
    so every batch in the same bucket has identical observation-step
    dimension, regardless of which samples land in which rank's slice.

    This is intentionally synchronous (single-process). For distributed
    training each rank runs its own loader; the underlying sampler ensures
    they pick the same bucket on each step and only differ in their slice.
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        batch_sampler: BucketedTokenBudgetBatchSampler,
        collate_fn: MultimodalCollator,
    ) -> None:
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for batch_keys in self.batch_sampler:
            if not batch_keys:
                continue
            T_target = batch_keys[0][1]
            self.collate_fn.set_pad_to_obs_steps(T_target)
            items = [self.dataset[key] for key in batch_keys]
            yield self.collate_fn(items)
