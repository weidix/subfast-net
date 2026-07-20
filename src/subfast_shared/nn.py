"""Reusable neural-network building blocks."""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseBlock(nn.Module):
    """Residual depthwise-separable convolution block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


__all__ = ["DepthwiseBlock"]
