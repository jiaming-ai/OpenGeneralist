from __future__ import annotations

import timm
import torch
from torch import nn
from torchvision import models

from open_gen.model.model_config import ImageEncoderConfig


class ResNetTokenEncoder(nn.Module):
    def __init__(self, name: str, pretrained: bool) -> None:
        super().__init__()
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        elif name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            backbone = models.resnet34(weights=weights)
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet encoder: {name}")
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        self.out_dim = backbone.fc.in_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.body(images)
        return features.flatten(2).transpose(1, 2)


class SiglipTokenEncoder(nn.Module):
    def __init__(self, name: str, pretrained: bool) -> None:
        super().__init__()
        model_name = "vit_base_patch16_siglip_224" if name == "siglip" else name
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="")
        self.out_dim = self.backbone.num_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(images)
        if features.ndim == 4:
            return features.flatten(2).transpose(1, 2)
        if features.ndim == 3:
            return features
        raise ValueError(f"Unexpected SigLIP feature shape: {features.shape}")


class ImageTokenEncoder(nn.Module):
    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self.config = config
        encoder_type = self.config.type
        if encoder_type.startswith("resnet"):
            self.backbone = ResNetTokenEncoder(encoder_type, self.config.pretrained)
        elif encoder_type.startswith("siglip"):
            self.backbone = SiglipTokenEncoder(encoder_type, self.config.pretrained)
        else:
            raise ValueError(f"Unsupported image encoder: {encoder_type}")
        if not self.config.trainable:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def out_dim(self) -> int:
        return self.backbone.out_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.config.trainable:
            self.backbone.eval()
            with torch.no_grad():
                return self.backbone(images)
        return self.backbone(images)
