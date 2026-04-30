from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_array(value: Any, dtype: np.dtype | None = np.float32) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(dtype) if dtype is not None and value.dtype != dtype else value
    return np.asarray(value, dtype=dtype)


def ensure_time_major(value: np.ndarray | None, *, ndim: int) -> np.ndarray | None:
    if value is None:
        return None
    if value.ndim == ndim - 1:
        return value[None, ...]
    if value.ndim != ndim:
        raise ValueError(f"Expected array with {ndim - 1} or {ndim} dims, got shape {value.shape}")
    return value


def pad_last_dim(value: np.ndarray, size: int) -> np.ndarray:
    if value.shape[-1] == size:
        return value.astype(np.float32)
    if value.shape[-1] > size:
        return value[..., :size].astype(np.float32)
    pad_shape = (*value.shape[:-1], size - value.shape[-1])
    return np.concatenate([value, np.zeros(pad_shape, dtype=np.float32)], axis=-1)


@dataclass
class CanonicalTrajectory:
    left_wrist_image: np.ndarray
    right_wrist_image: np.ndarray
    action: np.ndarray
    goal: str | np.ndarray | None = None
    embodiment: str | np.ndarray | None = None
    action_repr: str | np.ndarray | None = None
    head_image: np.ndarray | None = None
    proprio: np.ndarray | None = None
    force: np.ndarray | None = None
    past_action: np.ndarray | None = None
    future_left_wrist_image: np.ndarray | None = None
    future_right_wrist_image: np.ndarray | None = None
    future_head_image: np.ndarray | None = None
    future_proprio: np.ndarray | None = None
    future_force: np.ndarray | None = None

    def native_length(self) -> int:
        return int(self.left_wrist_image.shape[0])

    def truncate(self, num_past_frames: int) -> "CanonicalTrajectory":
        if num_past_frames <= 0:
            raise ValueError("num_past_frames must be positive")
        length = self.native_length()
        keep = min(num_past_frames, length)
        start = length - keep

        def _tail(value: np.ndarray | None) -> np.ndarray | None:
            return value[start:] if value is not None else None

        self.left_wrist_image = _tail(self.left_wrist_image)
        self.right_wrist_image = _tail(self.right_wrist_image)
        self.head_image = _tail(self.head_image)
        self.proprio = _tail(self.proprio)
        self.force = _tail(self.force)
        self.past_action = _tail(self.past_action)
        return self

    def validate(self, *, action_dim: int, proprio_dim: int, force_dim: int) -> "CanonicalTrajectory":
        self.left_wrist_image = ensure_time_major(_as_array(self.left_wrist_image, np.uint8), ndim=4)
        self.right_wrist_image = ensure_time_major(_as_array(self.right_wrist_image, np.uint8), ndim=4)
        self.action = ensure_time_major(pad_last_dim(_as_array(self.action), action_dim), ndim=2)
        self.head_image = ensure_time_major(_as_array(self.head_image, np.uint8), ndim=4)
        self.proprio = (
            ensure_time_major(pad_last_dim(_as_array(self.proprio), proprio_dim), ndim=2) if self.proprio is not None else None
        )
        self.force = ensure_time_major(pad_last_dim(_as_array(self.force), force_dim), ndim=2) if self.force is not None else None
        self.past_action = (
            ensure_time_major(pad_last_dim(_as_array(self.past_action), action_dim), ndim=2)
            if self.past_action is not None
            else None
        )
        self.future_left_wrist_image = ensure_time_major(_as_array(self.future_left_wrist_image, np.uint8), ndim=4)
        self.future_right_wrist_image = ensure_time_major(_as_array(self.future_right_wrist_image, np.uint8), ndim=4)
        self.future_head_image = ensure_time_major(_as_array(self.future_head_image, np.uint8), ndim=4)
        self.future_proprio = (
            ensure_time_major(pad_last_dim(_as_array(self.future_proprio), proprio_dim), ndim=2)
            if self.future_proprio is not None
            else None
        )
        self.future_force = (
            ensure_time_major(pad_last_dim(_as_array(self.future_force), force_dim), ndim=2)
            if self.future_force is not None
            else None
        )
        return self


CANONICAL_FIELDS = {
    "goal",
    "embodiment",
    "action_repr",
    "left_wrist_image",
    "right_wrist_image",
    "head_image",
    "proprio",
    "force",
    "past_action",
    "action",
    "future_left_wrist_image",
    "future_right_wrist_image",
    "future_head_image",
    "future_proprio",
    "future_force",
}
