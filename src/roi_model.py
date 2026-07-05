from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import DepthwiseBlock


class HybridLiteEmbeddingHead(nn.Module):
    def __init__(self, feature_dim: int, embedding_dim: int, sequence_channels: int) -> None:
        super().__init__()
        if sequence_channels > 64:
            raise ValueError("embedding_sequence_channels must be <= 64")
        if sequence_channels <= 0:
            raise ValueError("embedding_sequence_channels must be positive")
        self.sequence_projection = nn.Conv1d(feature_dim, sequence_channels, 1)
        self.sequence_block = nn.Sequential(
            nn.Conv1d(sequence_channels, sequence_channels, 3, padding=1, groups=sequence_channels, bias=False),
            nn.BatchNorm1d(sequence_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(sequence_channels, sequence_channels, 1, bias=False),
            nn.BatchNorm1d(sequence_channels),
            nn.SiLU(inplace=True),
        )
        self.attention = nn.Conv1d(sequence_channels, 1, 1)
        self.fusion = nn.Linear(embedding_dim + sequence_channels, embedding_dim)
        self._initialize_fusion(embedding_dim)

    def _initialize_fusion(self, embedding_dim: int) -> None:
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        with torch.no_grad():
            self.fusion.weight[:, :embedding_dim].copy_(torch.eye(embedding_dim))
            self.fusion.weight[:, embedding_dim:].normal_(mean=0.0, std=0.01)

    def forward(self, gap_feature: torch.Tensor, feature_map: torch.Tensor | None) -> torch.Tensor:
        if feature_map is None:
            seq_feature = gap_feature.new_zeros((gap_feature.shape[0], self.sequence_projection.out_channels))
        else:
            sequence = feature_map.mean(dim=2)
            sequence = self.sequence_projection(sequence)
            sequence = self.sequence_block(sequence)
            weights = torch.softmax(self.attention(sequence), dim=2)
            seq_feature = (sequence * weights).sum(dim=2)
        return self.fusion(torch.cat([gap_feature, seq_feature], dim=1))


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

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        textness_map = self.textness_map(feature_map)
        textness = self.flatten(textness_map)
        k = max(1, int(textness.shape[1] * self.topk_ratio + 0.999999))
        return textness.topk(k, dim=1).values.mean(dim=1)

    def textness_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.textness(self.local(feature_map))


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


class RoiPresenceEmbeddingModel(nn.Module):
    def __init__(
        self,
        width: int = 32,
        embedding_dim: int = 128,
        *,
        embedding_head_type: str = "gap",
        embedding_sequence_channels: int = 16,
        presence_topk_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        if embedding_head_type not in {"gap", "hybrid_lite"}:
            raise ValueError(f"unsupported embedding_head_type: {embedding_head_type}")
        self.embedding_head_type = embedding_head_type
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.presence_contrast = LocalContrastEnhancement()
        self.presence_head = LocalTextnessPresenceHead(feature_dim * 2, hidden_dim=feature_dim, topk_ratio=presence_topk_ratio)
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim, embedding_dim),
        )
        self.hybrid_embedding_head = (
            HybridLiteEmbeddingHead(feature_dim, embedding_dim, embedding_sequence_channels)
            if embedding_head_type == "hybrid_lite"
            else None
        )

    def encode_map(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.pool(self.encode_map(images)))

    def _pool_features(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.pool(feature_map))

    def forward_presence(self, features_or_images: torch.Tensor) -> torch.Tensor:
        if features_or_images.ndim != 4:
            raise ValueError("presence head requires ROI images or a feature map")
        feature_map = self.encode_map(features_or_images) if features_or_images.shape[1] == 3 else features_or_images
        return self.presence_head(self.presence_contrast(feature_map))

    def forward_embedding(self, features_or_images: torch.Tensor) -> torch.Tensor:
        feature_map = self.encode_map(features_or_images) if features_or_images.ndim == 4 else None
        features = self._pool_features(feature_map) if feature_map is not None else features_or_images
        gap_feature = self.embedding_head(features)
        if self.hybrid_embedding_head is not None:
            gap_feature = self.hybrid_embedding_head(gap_feature, feature_map)
        return F.normalize(gap_feature, p=2, dim=1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.encode_map(images)
        features = self._pool_features(feature_map)
        presence_logit = self.presence_head(self.presence_contrast(feature_map))
        gap_feature = self.embedding_head(features)
        if self.hybrid_embedding_head is not None:
            gap_feature = self.hybrid_embedding_head(gap_feature, feature_map)
        return presence_logit, F.normalize(gap_feature, p=2, dim=1)

    def forward_with_presence_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.encode_map(images)
        features = self._pool_features(feature_map)
        presence_feature = self.presence_contrast(feature_map)
        textness_map = self.presence_head.textness_map(presence_feature)
        presence_logit = self.presence_head.flatten(textness_map)
        k = max(1, int(presence_logit.shape[1] * self.presence_head.topk_ratio + 0.999999))
        presence_logit = presence_logit.topk(k, dim=1).values.mean(dim=1)
        gap_feature = self.embedding_head(features)
        if self.hybrid_embedding_head is not None:
            gap_feature = self.hybrid_embedding_head(gap_feature, feature_map)
        return presence_logit, F.normalize(gap_feature, p=2, dim=1), textness_map
