from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Choose stable per-sample normalization groups with at least four channels each."""
    groups = min(max_groups, max(1, channels // 4))
    while channels % groups:
        groups -= 1
    return groups


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_group_count(channels), channels)


class StableDepthwiseBlock(nn.Module):
    """A batch-composition-independent residual block for small ROI batches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.depthwise_norm = _group_norm(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.pointwise_norm = _group_norm(channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(feature_map)
        residual = self.activation(self.depthwise_norm(residual))
        residual = self.pointwise_norm(self.pointwise(residual))
        return self.activation(feature_map + residual)


class BoundedLocalContrast(nn.Module):
    """Extract local contrast without stored statistics or unbounded division spikes."""

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
        local_mean = self.background(feature_map)
        local_second_moment = self.background(feature_map.square())
        local_variance = (local_second_moment - local_mean.square()).clamp_min(0.0)
        standardized = (feature_map - local_mean) * torch.rsqrt(local_variance + self.eps)
        bounded_contrast = torch.tanh(standardized * 0.5)
        return torch.cat([feature_map, bounded_contrast], dim=1)


class CoherentEvidencePooling(nn.Module):
    """Pool spatially coherent evidence instead of selecting isolated top-k peaks."""

    def __init__(self, *, kernel_size: int = 3, temperature: float = 0.5) -> None:
        super().__init__()
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer greater than 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.support = nn.AvgPool2d(
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        self.temperature = float(temperature)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        region_logits: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            coherent_logits = self.support(region_logits).flatten(start_dim=1)
            valid_locations = torch.ones_like(coherent_logits, dtype=torch.bool)
        else:
            valid = F.interpolate(valid_mask, size=region_logits.shape[-2:], mode="area")
            valid = (valid > 0.5).to(region_logits.dtype)
            supported_valid = self.support(valid)
            coherent_logits = self.support(region_logits * valid) / supported_valid.clamp_min(1e-6)
            coherent_logits = coherent_logits.flatten(start_dim=1)
            valid_locations = valid.flatten(start_dim=1) > 0.0
        location_count = valid_locations.sum(dim=1).clamp_min(1)
        masked_logits = coherent_logits.masked_fill(~valid_locations, torch.finfo(coherent_logits.dtype).min)
        evidence = self.temperature * (
            torch.logsumexp(masked_logits / self.temperature, dim=1) - location_count.log()
        )
        positive_scale = F.softplus(self.log_scale) + 0.5
        return positive_scale * evidence + self.bias


class SubtitleRegionPresenceHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int | None = None,
        evidence_kernel_size: int = 3,
        evidence_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.local = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1, bias=False),
            _group_norm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.evidence_pool = CoherentEvidencePooling(
            kernel_size=evidence_kernel_size,
            temperature=evidence_temperature,
        )

    def region_logits(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.local(feature_map)

    def forward(
        self,
        feature_map: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.evidence_pool(self.region_logits(feature_map), valid_mask)


class RoiPresenceModel(nn.Module):
    architecture_version = 2

    def __init__(
        self,
        width: int = 32,
        *,
        evidence_kernel_size: int = 3,
        evidence_temperature: float = 0.5,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            _group_norm(width),
            nn.SiLU(inplace=True),
            StableDepthwiseBlock(width),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            _group_norm(width * 2),
            nn.SiLU(inplace=True),
            StableDepthwiseBlock(width * 2),
            StableDepthwiseBlock(width * 2),
        )
        feature_dim = width * 2
        self.local_contrast = BoundedLocalContrast()
        self.presence_head = SubtitleRegionPresenceHead(
            feature_dim * 2,
            hidden_dim=feature_dim,
            evidence_kernel_size=evidence_kernel_size,
            evidence_temperature=evidence_temperature,
        )

    def encode_map(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    @staticmethod
    def input_valid_mask(images: torch.Tensor) -> torch.Tensor:
        """Recover the exact zero padding emitted by the V2 letterbox preprocessor."""
        return (images.abs().amax(dim=1, keepdim=True) > 1e-8).to(images.dtype)

    def forward_with_presence_map(
        self,
        images: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_mask = valid_mask if valid_mask is not None else self.input_valid_mask(images)
        presence_feature = self.local_contrast(self.encode_map(images))
        region_logits = self.presence_head.region_logits(presence_feature)
        return self.presence_head.evidence_pool(region_logits, valid_mask), region_logits

    def forward(
        self,
        images: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = valid_mask if valid_mask is not None else self.input_valid_mask(images)
        return self.presence_head(self.local_contrast(self.encode_map(images)), valid_mask)
