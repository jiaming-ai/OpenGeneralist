from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImageEncoderConfig:
    type: str = "resnet18"
    pretrained: bool = True
    trainable: bool = False
    image_size: int = 224
    output_tokens: int | None = None


@dataclass
class ContextConfig:
    vocab_size: int = 8192
    max_text_tokens: int = 16


@dataclass
class TrunkConfig:
    width: int = 1024
    depth: int = 18
    num_heads: int = 16
    mlp_ratio: float = 4.0
    rope_base: float = 10000.0
    dropout: float = 0.0


@dataclass
class ActionExpertConfig:
    depth: int = 6
    width: int = 1024
    num_heads: int = 16
    mlp_ratio: float = 4.0
    flow_steps: int = 8
    time_inject: str = "adarms"


@dataclass
class WorldModelConfig:
    enabled: bool = True
    latent_dim: int = 1024
    future_latent_weight: float = 1.0
    proprio_weight: float = 0.2
    force_weight: float = 0.2


@dataclass
class OpenGen1Config:
    action_dim: int = 20
    proprio_dim: int = 20
    force_dim: int = 6
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    action_expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)


def _make_dataclass(cls: type[Any], payload: dict[str, Any]) -> Any:
    return cls(**payload)


def config_from_dict(payload: dict[str, Any]) -> OpenGen1Config:
    return OpenGen1Config(
        action_dim=payload["action_dim"],
        proprio_dim=payload["proprio_dim"],
        force_dim=payload["force_dim"],
        image_encoder=_make_dataclass(ImageEncoderConfig, payload["image_encoder"]),
        context=_make_dataclass(ContextConfig, payload["context"]),
        trunk=_make_dataclass(TrunkConfig, payload["trunk"]),
        action_expert=_make_dataclass(ActionExpertConfig, payload["action_expert"]),
        world_model=_make_dataclass(WorldModelConfig, payload["world_model"]),
    )
