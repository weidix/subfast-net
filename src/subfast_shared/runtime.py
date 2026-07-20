"""Shared training runtime helpers."""

from __future__ import annotations

import torch


def choose_device(requested: str) -> torch.device:
    """Resolve an explicit device or choose the best available accelerator."""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


__all__ = ["choose_device"]
