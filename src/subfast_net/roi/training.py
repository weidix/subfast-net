"""Shared runtime helpers for the ROI training families.

The presence, embedding, and matcher trainers intentionally keep separate
models and objectives, but they use the same small set of data and runtime
utilities.  Keeping those utilities here prevents one trainer from importing
another trainer just to reuse a parser or a progress helper.
"""

from __future__ import annotations

import argparse
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed the random generators used by ROI data loaders and models."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize_device(device: torch.device) -> None:
    """Wait for asynchronous device work before recording a benchmark time."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def parse_roi_size(value: str) -> tuple[int, int]:
    """Parse the public ``WIDTHxHEIGHT`` ROI resize option."""

    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI resize dimensions must be positive")
    return width, height


def model_parameter_count(model: torch.nn.Module) -> int:
    """Return the number of trainable and frozen parameters in ``model``."""

    return sum(parameter.numel() for parameter in model.parameters())


def format_dataset_summary(
    name: str,
    dataset: Any,
    *,
    include_diagnostics: bool = False,
) -> str:
    """Format the common ROI dataset summary used by training logs.

    Presence-only datasets expose a few additional diagnostic counters.  They
    are included only when requested so the embedding and matcher logs retain
    their compact, stable format.
    """

    summary = dataset.summary
    roots = ", ".join(f"{root}={count}" for root, count in sorted(summary.roots.items()))
    text = (
        f"{name}: samples={summary.total} positive={summary.positive} empty={summary.empty} "
        f"positive_ratio={summary.positive_ratio:.3f} empty_ratio={summary.empty_ratio:.3f}"
    )
    if hasattr(summary, "positive_segments"):
        text += (
            f" positive_segments={summary.positive_segments}"
            f" repeated_positive_segments={summary.repeated_positive_segments}"
            f" same_segment_pairs={summary.same_segment_pairs}"
        )
    if include_diagnostics:
        text += (
            f" text_distractor_negatives={dataset.text_distractor_negatives}"
            f" positive_without_region={dataset.positive_without_region}"
            f" positive_without_donor={dataset.positive_without_donor}"
        )
    return f"{text} roi_size={summary.roi_size[0]}x{summary.roi_size[1]} roots=[{roots}]"
