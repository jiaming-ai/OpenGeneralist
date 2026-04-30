from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from open_gen.data.schema import CanonicalTrajectory
from open_gen.data.transforms import AugmentationConfig, build_image_transform
from open_gen.utils.normalization import NormStats, normalize as norm_apply


@dataclass
class BatchNorms:
    action: NormStats | None = None
    proprio: NormStats | None = None
    force: NormStats | None = None


def _stack_strings(values: list[str | torch.Tensor | None]) -> list[str | torch.Tensor | None]:
    return values


def _first_non_none(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    for tensor in tensors:
        if tensor is not None:
            return tensor
    return None


def _pad_tensor_sequence(
    tensors: list[torch.Tensor | None],
    *,
    pad_shape: tuple[int, ...],
    dtype: torch.dtype,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_len is None:
        max_len = max((tensor.shape[0] for tensor in tensors if tensor is not None), default=0)
    batch = torch.zeros((len(tensors), max_len, *pad_shape), dtype=dtype)
    mask = torch.zeros((len(tensors), max_len), dtype=torch.bool)
    for idx, tensor in enumerate(tensors):
        if tensor is None:
            continue
        length = tensor.shape[0]
        batch[idx, :length] = tensor
        mask[idx, :length] = True
    return batch, mask


class MultimodalCollator:
    def __init__(
        self,
        *,
        image_size: int,
        encoder_type: str,
        train: bool,
        norms: BatchNorms | None = None,
        augmentation: AugmentationConfig | None = None,
        pad_to_obs_steps: int | None = None,
        norm_mode: str = "zscore",
    ) -> None:
        if norm_mode not in {"zscore", "quantile"}:
            raise ValueError(f"Unsupported norm_mode: {norm_mode}")
        self.norms = norms or BatchNorms()
        self.norm_mode = norm_mode
        aug = augmentation or AugmentationConfig()
        self.wrist_transform = build_image_transform(
            image_size=image_size,
            encoder_type=encoder_type,
            train=train,
            augmentation=aug,
            view="wrist",
        )
        self.non_wrist_transform = build_image_transform(
            image_size=image_size,
            encoder_type=encoder_type,
            train=train,
            augmentation=aug,
            view="non_wrist",
        )
        self.pad_to_obs_steps = pad_to_obs_steps

    def set_pad_to_obs_steps(self, value: int | None) -> None:
        self.pad_to_obs_steps = value

    def _image_sequence(self, image_seq: Any, *, view: str) -> torch.Tensor:
        transform = self.wrist_transform if view == "wrist" else self.non_wrist_transform
        return torch.stack([transform(frame) for frame in image_seq], dim=0)

    def __call__(self, items: list[CanonicalTrajectory]) -> dict[str, Any]:
        left_images, right_images, head_images = [], [], []
        proprios, forces, past_actions = [], [], []
        future_left, future_right, future_head = [], [], []
        future_proprio, future_force = [], []
        actions = []

        for item in items:
            left_images.append(self._image_sequence(item.left_wrist_image, view="wrist"))
            right_images.append(self._image_sequence(item.right_wrist_image, view="wrist"))
            head_images.append(self._image_sequence(item.head_image, view="non_wrist") if item.head_image is not None else None)
            proprios.append(torch.from_numpy(item.proprio) if item.proprio is not None else None)
            forces.append(torch.from_numpy(item.force) if item.force is not None else None)
            past_actions.append(torch.from_numpy(item.past_action) if item.past_action is not None else None)
            actions.append(torch.from_numpy(item.action))
            future_left.append(self._image_sequence(item.future_left_wrist_image, view="wrist") if item.future_left_wrist_image is not None else None)
            future_right.append(self._image_sequence(item.future_right_wrist_image, view="wrist") if item.future_right_wrist_image is not None else None)
            future_head.append(self._image_sequence(item.future_head_image, view="non_wrist") if item.future_head_image is not None else None)
            future_proprio.append(torch.from_numpy(item.future_proprio) if item.future_proprio is not None else None)
            future_force.append(torch.from_numpy(item.future_force) if item.future_force is not None else None)

        observed_steps = max(image.shape[0] for image in left_images)
        obs_steps = max(observed_steps, self.pad_to_obs_steps) if self.pad_to_obs_steps is not None else observed_steps
        left_batch, obs_mask = _pad_tensor_sequence(left_images, pad_shape=left_images[0].shape[1:], dtype=torch.float32, max_len=obs_steps)
        right_batch, _ = _pad_tensor_sequence(right_images, pad_shape=right_images[0].shape[1:], dtype=torch.float32, max_len=obs_steps)
        head_batch, head_mask = _pad_tensor_sequence(
            head_images,
            pad_shape=left_images[0].shape[1:],
            dtype=torch.float32,
            max_len=obs_steps,
        )
        action_steps = max(action.shape[0] for action in actions)
        action_batch, action_mask = _pad_tensor_sequence(actions, pad_shape=(actions[0].shape[-1],), dtype=torch.float32)
        proprio_ref = _first_non_none(proprios)
        proprio_batch, proprio_mask = _pad_tensor_sequence(
            proprios,
            pad_shape=(proprio_ref.shape[-1],) if proprio_ref is not None else (0,),
            dtype=torch.float32,
            max_len=obs_steps,
        )
        force_ref = _first_non_none(forces)
        force_batch, force_mask = _pad_tensor_sequence(
            forces,
            pad_shape=(force_ref.shape[-1],) if force_ref is not None else (0,),
            dtype=torch.float32,
            max_len=obs_steps,
        )
        past_action_batch, past_action_mask = _pad_tensor_sequence(
            past_actions,
            pad_shape=(actions[0].shape[-1],),
            dtype=torch.float32,
            max_len=obs_steps,
        )
        future_steps = max((tensor.shape[0] for tensor in future_left if tensor is not None), default=0)
        future_left_batch, future_obs_mask = _pad_tensor_sequence(
            future_left,
            pad_shape=left_images[0].shape[1:],
            dtype=torch.float32,
            max_len=future_steps,
        )
        future_right_batch, _ = _pad_tensor_sequence(
            future_right,
            pad_shape=left_images[0].shape[1:],
            dtype=torch.float32,
            max_len=future_steps,
        )
        future_head_batch, future_head_mask = _pad_tensor_sequence(
            future_head,
            pad_shape=left_images[0].shape[1:],
            dtype=torch.float32,
            max_len=future_steps,
        )
        future_proprio_ref = _first_non_none(future_proprio)
        future_proprio_batch, future_proprio_mask = _pad_tensor_sequence(
            future_proprio,
            pad_shape=(future_proprio_ref.shape[-1],) if future_proprio_ref is not None else (0,),
            dtype=torch.float32,
            max_len=future_steps,
        )
        future_force_ref = _first_non_none(future_force)
        future_force_batch, future_force_mask = _pad_tensor_sequence(
            future_force,
            pad_shape=(future_force_ref.shape[-1],) if future_force_ref is not None else (0,),
            dtype=torch.float32,
            max_len=future_steps,
        )

        action_batch = norm_apply(action_batch, self.norms.action, mode=self.norm_mode)
        if proprio_batch.shape[-1] > 0:
            proprio_batch = norm_apply(proprio_batch, self.norms.proprio, mode=self.norm_mode)
        if force_batch.shape[-1] > 0:
            force_batch = norm_apply(force_batch, self.norms.force, mode=self.norm_mode)
        if past_action_batch.shape[-1] > 0:
            past_action_batch = norm_apply(past_action_batch, self.norms.action, mode=self.norm_mode)
        if future_proprio_batch.shape[-1] > 0:
            future_proprio_batch = norm_apply(future_proprio_batch, self.norms.proprio, mode=self.norm_mode)
        if future_force_batch.shape[-1] > 0:
            future_force_batch = norm_apply(future_force_batch, self.norms.force, mode=self.norm_mode)

        return {
            "context": {
                "goal": _stack_strings([item.goal for item in items]),
                "embodiment": _stack_strings([item.embodiment for item in items]),
                "action_repr": _stack_strings([item.action_repr for item in items]),
            },
            "observations": {
                "left_wrist_image": left_batch,
                "right_wrist_image": right_batch,
                "head_image": head_batch,
                "proprio": proprio_batch,
                "force": force_batch,
                "mask": obs_mask,
                "head_mask": head_mask,
                "proprio_mask": proprio_mask,
                "force_mask": force_mask,
            },
            "past_action": {
                "value": past_action_batch,
                "mask": past_action_mask,
            },
            "action_target": {
                "value": action_batch,
                "mask": action_mask,
            },
            "future": {
                "left_wrist_image": future_left_batch,
                "right_wrist_image": future_right_batch,
                "head_image": future_head_batch,
                "proprio": future_proprio_batch,
                "force": future_force_batch,
                "mask": future_obs_mask,
                "head_mask": future_head_mask,
                "proprio_mask": future_proprio_mask,
                "force_mask": future_force_mask,
            },
        }
