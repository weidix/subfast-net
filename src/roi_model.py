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


class RoiPresenceEmbeddingModel(nn.Module):
    def __init__(
        self,
        width: int = 32,
        embedding_dim: int = 128,
        *,
        embedding_head_type: str = "gap",
        embedding_sequence_channels: int = 16,
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
        self.presence_head = nn.Linear(feature_dim, 1)
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
        if features_or_images.ndim == 4:
            feature_map = self.encode_map(features_or_images)
            features = self._pool_features(feature_map)
        else:
            features = features_or_images
        return self.presence_head(features).squeeze(1)

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
        presence_logit = self.presence_head(features).squeeze(1)
        gap_feature = self.embedding_head(features)
        if self.hybrid_embedding_head is not None:
            gap_feature = self.hybrid_embedding_head(gap_feature, feature_map)
        return presence_logit, F.normalize(gap_feature, p=2, dim=1)
