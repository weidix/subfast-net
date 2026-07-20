"""Shared image-normalization constants for subtitle vision models."""

from __future__ import annotations

import torch


IMAGENET_MEAN_VALUES = (0.485, 0.456, 0.406)
IMAGENET_STD_VALUES = (0.229, 0.224, 0.225)
IMAGENET_MEAN = torch.tensor(IMAGENET_MEAN_VALUES, dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor(IMAGENET_STD_VALUES, dtype=torch.float32).view(3, 1, 1)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_MEAN_VALUES",
    "IMAGENET_STD",
    "IMAGENET_STD_VALUES",
]
