from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from open_gen.data.adapter import CanonicalTrajectoryAdapter, TrajectoryFieldMapping
from open_gen.data.schema import CanonicalTrajectory

DatasetKey = int | tuple[int, int]


def _resolve_key(key: DatasetKey) -> tuple[int, int | None]:
    if isinstance(key, tuple):
        index, num_past_frames = key
        return int(index), int(num_past_frames)
    return int(key), None


def _maybe_truncate(trajectory: CanonicalTrajectory, num_past_frames: int | None) -> CanonicalTrajectory:
    if num_past_frames is None:
        return trajectory
    return trajectory.truncate(num_past_frames)


class MappedTrajectoryDataset(Dataset[CanonicalTrajectory]):
    def __init__(self, samples: list[dict[str, Any]], adapter: CanonicalTrajectoryAdapter) -> None:
        self.samples = samples
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key: DatasetKey) -> CanonicalTrajectory:
        index, num_past_frames = _resolve_key(key)
        return _maybe_truncate(self.adapter.adapt(self.samples[index]), num_past_frames)

    def native_lengths(self) -> list[int]:
        return [int(self.adapter.adapt(sample).native_length()) for sample in self.samples]


class JsonlTrajectoryDataset(Dataset[CanonicalTrajectory]):
    def __init__(self, path: str | Path, adapter: CanonicalTrajectoryAdapter) -> None:
        self.path = Path(path)
        self.adapter = adapter
        self.lines = self.path.read_text(encoding="utf-8").splitlines()

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, key: DatasetKey) -> CanonicalTrajectory:
        index, num_past_frames = _resolve_key(key)
        return _maybe_truncate(self.adapter.adapt(json.loads(self.lines[index])), num_past_frames)

    def native_lengths(self) -> list[int]:
        return [int(self.adapter.adapt(json.loads(line)).native_length()) for line in self.lines]


class TorchTrajectoryDataset(Dataset[CanonicalTrajectory]):
    def __init__(self, path: str | Path, adapter: CanonicalTrajectoryAdapter) -> None:
        self.samples = torch.load(path, map_location="cpu", weights_only=False)
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key: DatasetKey) -> CanonicalTrajectory:
        index, num_past_frames = _resolve_key(key)
        return _maybe_truncate(self.adapter.adapt(self.samples[index]), num_past_frames)

    def native_lengths(self) -> list[int]:
        return [int(self.adapter.adapt(sample).native_length()) for sample in self.samples]


@dataclass
class SyntheticDatasetConfig:
    num_samples: int = 64
    sequence_length: int = 4
    action_horizon: int = 4
    image_size: int = 96
    action_dim: int = 20
    proprio_dim: int = 20
    force_dim: int = 6
    include_head: bool = True
    include_proprio: bool = True
    include_force: bool = True
    include_past_action: bool = True
    include_future_targets: bool = True
    seed: int = 7


class SyntheticRobotDataset(Dataset[CanonicalTrajectory]):
    def __init__(self, config: SyntheticDatasetConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def __len__(self) -> int:
        return self.config.num_samples

    def _image_sequence(self, length: int) -> np.ndarray:
        return self.rng.integers(
            low=0,
            high=255,
            size=(length, self.config.image_size, self.config.image_size, 3),
            dtype=np.uint8,
        )

    def __getitem__(self, key: DatasetKey) -> CanonicalTrajectory:
        index, num_past_frames = _resolve_key(key)
        _ = index
        cfg = self.config
        action = self.rng.normal(size=(cfg.action_horizon, cfg.action_dim)).astype(np.float32)
        proprio = self.rng.normal(size=(cfg.sequence_length, cfg.proprio_dim)).astype(np.float32) if cfg.include_proprio else None
        force = self.rng.normal(size=(cfg.sequence_length, cfg.force_dim)).astype(np.float32) if cfg.include_force else None
        past_action = (
            self.rng.normal(size=(cfg.sequence_length, cfg.action_dim)).astype(np.float32) if cfg.include_past_action else None
        )
        future_proprio = self.rng.normal(size=(1, cfg.proprio_dim)).astype(np.float32) if cfg.include_future_targets and cfg.include_proprio else None
        future_force = self.rng.normal(size=(1, cfg.force_dim)).astype(np.float32) if cfg.include_future_targets and cfg.include_force else None
        trajectory = CanonicalTrajectory(
            goal="pick up block",
            embodiment="dual_arm_mobile",
            action_repr="joint_position_20d",
            left_wrist_image=self._image_sequence(cfg.sequence_length),
            right_wrist_image=self._image_sequence(cfg.sequence_length),
            head_image=self._image_sequence(cfg.sequence_length) if cfg.include_head else None,
            proprio=proprio,
            force=force,
            past_action=past_action,
            action=action,
            future_left_wrist_image=self._image_sequence(1) if cfg.include_future_targets else None,
            future_right_wrist_image=self._image_sequence(1) if cfg.include_future_targets else None,
            future_head_image=self._image_sequence(1) if cfg.include_future_targets and cfg.include_head else None,
            future_proprio=future_proprio,
            future_force=future_force,
        ).validate(action_dim=cfg.action_dim, proprio_dim=cfg.proprio_dim, force_dim=cfg.force_dim)
        return _maybe_truncate(trajectory, num_past_frames)

    def native_lengths(self) -> list[int]:
        return [int(self.config.sequence_length)] * self.config.num_samples


def build_dataset(
    data_cfg: dict[str, Any], dims: dict[str, int], split: str
) -> Dataset[CanonicalTrajectory]:
    dataset_type = data_cfg["type"]
    shared = data_cfg.get("shared", {}) or {}
    split_cfg = {**shared, **(data_cfg.get(split, {}) or {})}

    if dataset_type == "synthetic":
        return SyntheticRobotDataset(
            SyntheticDatasetConfig(
                action_dim=dims["action"],
                proprio_dim=dims["proprio"],
                force_dim=dims["force"],
                **split_cfg,
            )
        )

    adapter = CanonicalTrajectoryAdapter(
        TrajectoryFieldMapping(fields=data_cfg["mapping"], defaults=data_cfg.get("defaults", {})),
        action_dim=dims["action"],
        proprio_dim=dims["proprio"],
        force_dim=dims["force"],
    )
    if dataset_type == "jsonl":
        return JsonlTrajectoryDataset(split_cfg["path"], adapter)
    if dataset_type == "torch":
        return TorchTrajectoryDataset(split_cfg["path"], adapter)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")
