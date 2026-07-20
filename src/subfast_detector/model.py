from __future__ import annotations

import torch
from torch import nn

from subfast_shared.nn import DepthwiseBlock


class SubtitleDetector(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
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
        self.head = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.stem(x))
        return torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
