from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import DepthwiseBlock


class LocalContrastResidual(nn.Module):
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
        residual = feature_map - self.background(feature_map)
        scale = self.background(residual.abs()).clamp_min(self.eps)
        return residual / scale


class MaskedAttentionEmbeddingHead(nn.Module):
    """Encodes the ordered subtitle width under a supervised attention map.

    Global average pooling lets background pixels dominate the descriptor, which
    breaks same-segment pairs whose background changes and discards character
    order. Height pooling keeps one token per horizontal region; the fixed-width
    sequence and its learned positions are then encoded without width pooling.
    """

    def __init__(self, feature_dim: int, embedding_dim: int, width_tokens: int = 32) -> None:
        super().__init__()
        if width_tokens <= 0:
            raise ValueError("width_tokens must be positive")
        self.contrast = LocalContrastResidual()
        stacked_dim = feature_dim * 2
        self.width_tokens = width_tokens
        self.attention = nn.Sequential(
            nn.Conv2d(stacked_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
        )
        self.token_projection = nn.Conv1d(stacked_dim, feature_dim, 1, bias=False)
        self.position_embedding = nn.Parameter(torch.empty(1, feature_dim, width_tokens))
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim, bias=False),
            nn.GroupNorm(1, feature_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dim * width_tokens, embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.cat([feature_map, self.contrast(feature_map)], dim=1)
        attention_logits = self.attention(stacked)
        weights = torch.sigmoid(attention_logits)
        height_denominator = weights.sum(dim=2).clamp_min(1e-4)
        width_sequence = (stacked * weights).sum(dim=2) / height_denominator
        width_sequence = width_sequence * weights.mean(dim=2)
        width_sequence = F.adaptive_avg_pool1d(width_sequence, self.width_tokens)
        width_sequence = self.token_projection(width_sequence) + self.position_embedding
        width_sequence = width_sequence + self.sequence_encoder(width_sequence)
        embedding = self.projection(width_sequence.flatten(start_dim=1))
        return embedding, attention_logits


class MaskedGlobalEmbeddingHead(nn.Module):
    """Legacy masked-global head kept for old checkpoint reproduction."""

    def __init__(self, feature_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.contrast = LocalContrastResidual()
        stacked_dim = feature_dim * 2
        self.attention = nn.Sequential(
            nn.Conv2d(stacked_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(stacked_dim, stacked_dim),
            nn.SiLU(inplace=True),
            nn.Linear(stacked_dim, embedding_dim),
        )

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.cat([feature_map, self.contrast(feature_map)], dim=1)
        attention_logits = self.attention(stacked)
        weights = torch.sigmoid(attention_logits)
        denominator = weights.sum(dim=(2, 3)).clamp_min(1e-4)
        pooled = (stacked * weights).sum(dim=(2, 3)) / denominator
        return self.projection(pooled), attention_logits


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
        embedding_dim: int = 256,
        *,
        presence_topk_ratio: float = 0.05,
        embedding_width_tokens: int = 32,
        embedding_aggregation: str = "width_tokens",
    ) -> None:
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.presence_contrast = LocalContrastEnhancement()
        self.presence_head = LocalTextnessPresenceHead(feature_dim * 2, hidden_dim=feature_dim, topk_ratio=presence_topk_ratio)
        if embedding_aggregation == "width_tokens":
            self.embedding_head = MaskedAttentionEmbeddingHead(
                feature_dim,
                embedding_dim,
                width_tokens=embedding_width_tokens,
            )
        elif embedding_aggregation == "masked_global":
            self.embedding_head = MaskedGlobalEmbeddingHead(feature_dim, embedding_dim)
        else:
            raise ValueError(f"unsupported embedding aggregation: {embedding_aggregation}")

    def encode_map(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.pool(self.encode_map(images)))

    def forward_presence(self, features_or_images: torch.Tensor) -> torch.Tensor:
        if features_or_images.ndim != 4:
            raise ValueError("presence head requires ROI images or a feature map")
        feature_map = self.encode_map(features_or_images) if features_or_images.shape[1] == 3 else features_or_images
        return self.presence_head(self.presence_contrast(feature_map))

    def forward_embedding(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_embedding_with_attention(images)[0]

    def forward_embedding_with_attention(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding, attention_logits = self.embedding_head(self.encode_map(images))
        return F.normalize(embedding, p=2, dim=1), attention_logits

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.encode_map(images)
        presence_logit = self.presence_head(self.presence_contrast(feature_map))
        embedding, _ = self.embedding_head(feature_map)
        return presence_logit, F.normalize(embedding, p=2, dim=1)

    def forward_with_presence_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_map = self.encode_map(images)
        presence_feature = self.presence_contrast(feature_map)
        textness_map = self.presence_head.textness_map(presence_feature)
        presence_logit = self.presence_head.flatten(textness_map)
        k = max(1, int(presence_logit.shape[1] * self.presence_head.topk_ratio + 0.999999))
        presence_logit = presence_logit.topk(k, dim=1).values.mean(dim=1)
        embedding, _ = self.embedding_head(feature_map)
        return presence_logit, F.normalize(embedding, p=2, dim=1), textness_map
