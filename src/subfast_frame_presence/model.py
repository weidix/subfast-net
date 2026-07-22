from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _conv_block(
    input_channels: int,
    output_channels: int,
    *,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True),
    )


class FramePresenceModel(nn.Module):
    """Compact full-frame RGB subtitle-presence model with dense supervision."""

    architecture_version = 3
    feature_stride = 4

    def __init__(self, *, width: int = 24) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        region_width = width + width // 2
        context_width = width * 2
        self.stem = _conv_block(3, width, kernel_size=5, stride=2)
        self.detail = _conv_block(width, width, kernel_size=3, stride=2)
        self.region_encoder = _conv_block(width, region_width, kernel_size=5, stride=2)
        self.context = _conv_block(region_width, context_width, kernel_size=3, stride=2)
        self.context_refine = _conv_block(
            context_width,
            context_width,
            kernel_size=3,
            dilation=3,
        )
        self.context_projection = nn.Conv2d(context_width, region_width, 1, bias=False)
        self.detail_projection = nn.Conv2d(region_width, width, 1, bias=False)
        self.detail_refine = _conv_block(width, width, kernel_size=3)
        self.region_head = nn.Conv2d(width, 1, 1)
        self.global_head = nn.Linear(context_width, 1)

    def encode_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        detail = self.detail(self.stem(images))
        region = self.region_encoder(detail)
        context = self.context_refine(self.context(region))
        region = region + F.interpolate(
            self.context_projection(context),
            size=region.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail = detail + F.interpolate(
            self.detail_projection(region),
            size=detail.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.detail_refine(detail), context

    def forward_with_presence_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        region_features, context = self.encode_map(images)
        region_logits = self.region_head(region_features)
        coherent_logits = F.avg_pool2d(
            region_logits,
            5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )
        local_evidence = coherent_logits.amax(dim=(1, 2, 3))
        global_evidence = self.global_head(F.adaptive_avg_pool2d(context, 1).flatten(1)).squeeze(1)
        return local_evidence + global_evidence, region_logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        presence_logits, _ = self.forward_with_presence_map(images)
        return presence_logits


__all__ = ["FramePresenceModel"]
