from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def encoder_image_stats(encoder_type: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if encoder_type.startswith("siglip"):
        return SIGLIP_MEAN, SIGLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


@dataclass
class AugmentationConfig:
    """Per-view image augmentation knobs.
    """

    enabled: bool = True
    resized_crop_scale_min: float = 0.95
    rotation_degrees: float = 5.0
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.0


def build_image_transform(
    *,
    image_size: int,
    encoder_type: str,
    train: bool,
    augmentation: AugmentationConfig,
    view: str = "non_wrist",
) -> Callable[[np.ndarray], torch.Tensor]:
    if view not in {"wrist", "non_wrist"}:
        raise ValueError(f"Unsupported view: {view}")
    mean, std = encoder_image_stats(encoder_type)
    is_wrist = view == "wrist"
    do_geom = train and augmentation.enabled and not is_wrist
    do_color = train and augmentation.enabled

    ops: list[torch.nn.Module] = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if do_geom:
        ops.append(
            v2.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(augmentation.resized_crop_scale_min, 1.0),
                antialias=True,
            )
        )
        if augmentation.rotation_degrees > 0:
            ops.append(v2.RandomRotation(degrees=augmentation.rotation_degrees))
    else:
        ops.append(v2.Resize((image_size, image_size), antialias=True))

    if do_color:
        ops.append(
            v2.ColorJitter(
                brightness=augmentation.brightness,
                contrast=augmentation.contrast,
                saturation=augmentation.saturation,
                hue=augmentation.hue,
            )
        )

    ops.append(v2.Normalize(mean=mean, std=std))
    transform = v2.Compose(ops)

    def apply(image: np.ndarray) -> torch.Tensor:
        return transform(image)

    return apply
