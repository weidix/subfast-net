from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class Downsample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(input_channels), input_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class EvidencePooling(nn.Module):
    """Learned scalar evidence directly from the full-frame dense logits."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.support = nn.AvgPool2d(
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, region_logits: torch.Tensor) -> torch.Tensor:
        coherent = self.support(region_logits)
        scale = F.softplus(self.log_scale) + 0.5
        return scale * coherent.amax(dim=(1, 2, 3)) + self.bias


class FramePresenceModel(nn.Module):
    """A full-frame RGB subtitle-presence model with dense auxiliary evidence."""

    architecture_version = 1
    feature_stride = 2

    def __init__(self, *, width: int = 24, evidence_kernel_size: int = 5) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if evidence_kernel_size <= 1 or evidence_kernel_size % 2 == 0:
            raise ValueError("evidence_kernel_size must be an odd integer greater than 1")
        mid_width = width + width // 2
        high_width = width * 2
        context_width = width * 3
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(inplace=True),
            ResidualDepthwiseBlock(width),
        )
        self.down_1 = nn.Sequential(Downsample(width, mid_width), ResidualDepthwiseBlock(mid_width))
        self.down_2 = nn.Sequential(Downsample(mid_width, high_width), ResidualDepthwiseBlock(high_width))
        self.down_3 = nn.Sequential(
            Downsample(high_width, context_width),
            ResidualDepthwiseBlock(context_width, kernel_size=5),
            ResidualDepthwiseBlock(context_width, kernel_size=5),
            ResidualDepthwiseBlock(context_width, kernel_size=5),
            ResidualDepthwiseBlock(context_width, kernel_size=5),
        )
        self.up_2 = nn.Sequential(
            nn.Conv2d(context_width, high_width, 1, bias=False),
            ResidualDepthwiseBlock(high_width),
        )
        self.up_1 = nn.Sequential(
            nn.Conv2d(high_width, mid_width, 1, bias=False),
            ResidualDepthwiseBlock(mid_width),
        )
        self.up_0 = nn.Sequential(
            nn.Conv2d(mid_width, width, 1, bias=False),
            ResidualDepthwiseBlock(width),
        )
        self.region_head = nn.Conv2d(width, 1, 3, padding=1)
        self.evidence_pool = EvidencePooling(evidence_kernel_size)
        self.global_head = nn.Linear(context_width, 1)

    def encode_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.stem(images)
        mid = self.down_1(local)
        high = self.down_2(mid)
        context = self.down_3(high)
        high = high + F.interpolate(self.up_2(context), size=high.shape[-2:], mode="bilinear", align_corners=False)
        mid = mid + F.interpolate(self.up_1(high), size=mid.shape[-2:], mode="bilinear", align_corners=False)
        local = local + F.interpolate(self.up_0(mid), size=local.shape[-2:], mode="bilinear", align_corners=False)
        return local, context

    def forward_with_presence_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map, context = self.encode_map(images)
        region_logits = self.region_head(feature_map)
        local_evidence = self.evidence_pool(region_logits)
        global_evidence = self.global_head(F.adaptive_avg_pool2d(context, 1).flatten(1)).squeeze(1)
        return local_evidence + global_evidence, region_logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        presence_logits, _ = self.forward_with_presence_map(images)
        return presence_logits


__all__ = ["FramePresenceModel"]
