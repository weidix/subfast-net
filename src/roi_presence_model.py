from __future__ import annotations

import torch
from torch import nn

from .model import DepthwiseBlock


class LocalContrastEnhancement(nn.Module):
    def __init__(self, kernel_size: int = 7, eps: float = 1e-4) -> None:
        super().__init__()
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer greater than 1")
        self.background = nn.AvgPool2d(
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        self.eps = eps

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        local_background = self.background(feature_map)
        local_difference = (feature_map - local_background).abs()
        local_scale = self.background(local_difference).clamp_min(self.eps)
        local_contrast = local_difference / local_scale
        return torch.cat([feature_map, local_contrast], dim=1)


class LocalTextnessPresenceHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int | None = None, topk_ratio: float = 0.05) -> None:
        super().__init__()
        if topk_ratio <= 0.0 or topk_ratio > 1.0:
            raise ValueError("topk_ratio must be in (0, 1]")
        hidden_dim = hidden_dim or feature_dim
        self.topk_ratio = topk_ratio
        self.local = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.textness = nn.Conv2d(hidden_dim, 1, 1)
        self.flatten = nn.Flatten(start_dim=1)

    def textness_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.textness(self.local(feature_map))

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        textness = self.flatten(self.textness_map(feature_map))
        k = max(1, int(textness.shape[1] * self.topk_ratio + 0.999999))
        return textness.topk(k, dim=1).values.mean(dim=1)


class RoiPresenceModel(nn.Module):
    def __init__(self, width: int = 32, *, presence_topk_ratio: float = 0.05) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            DepthwiseBlock(width),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            DepthwiseBlock(width * 2),
            DepthwiseBlock(width * 2),
        )
        feature_dim = width * 2
        self.presence_contrast = LocalContrastEnhancement()
        self.presence_head = LocalTextnessPresenceHead(
            feature_dim * 2,
            hidden_dim=feature_dim,
            topk_ratio=presence_topk_ratio,
        )

    def encode_map(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward_with_presence_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        presence_feature = self.presence_contrast(self.encode_map(images))
        textness_map = self.presence_head.textness_map(presence_feature)
        textness = self.presence_head.flatten(textness_map)
        k = max(1, int(textness.shape[1] * self.presence_head.topk_ratio + 0.999999))
        return textness.topk(k, dim=1).values.mean(dim=1), textness_map

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.presence_head(self.presence_contrast(self.encode_map(images)))
