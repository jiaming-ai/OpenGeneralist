from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "gloo"
    initialized: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def init_distributed(device_hint: str = "cpu") -> DistEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return DistEnv(rank=0, world_size=1, local_rank=0, backend="gloo", initialized=False)
    backend = "nccl" if (device_hint.startswith("cuda") and torch.cuda.is_available()) else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return DistEnv(rank=rank, world_size=world_size, local_rank=local_rank, backend=backend, initialized=True)


def finalize_distributed(env: DistEnv) -> None:
    if env.initialized and dist.is_initialized():
        dist.destroy_process_group()
