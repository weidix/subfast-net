from __future__ import annotations

import torch
from torch.nn import functional as F

from .stream_model import StreamingSegmentModelOutput


def streaming_detection_loss(
    output: StreamingSegmentModelOutput,
    presence_targets: torch.Tensor,
    boundary_event_targets: torch.Tensor,
    mask: torch.Tensor,
    presence_positive_weight: float | torch.Tensor,
    boundary_event_positive_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train causal subtitle presence and start/end boundary events jointly."""

    if presence_targets.shape != output.presence_logits.shape:
        raise ValueError("presence targets must match the presence-logit shape")
    if boundary_event_targets.shape != (*output.presence_logits.shape, 2):
        raise ValueError("boundary event targets must have shape [batch,time,2]")
    if output.boundary_event_logits.shape != boundary_event_targets.shape:
        raise ValueError("boundary event logits must have shape [batch,time,2]")
    if mask.shape != output.presence_logits.shape:
        raise ValueError("mask must match the presence-logit shape")

    device = output.presence_logits.device
    dtype = output.presence_logits.dtype
    presence_targets = presence_targets.to(device=device, dtype=dtype)
    boundary_event_targets = boundary_event_targets.to(device=device, dtype=dtype)
    valid_weight = mask.to(device=device, dtype=dtype)
    _validate_targets(presence_targets, "presence targets")
    _validate_targets(boundary_event_targets, "boundary event targets")
    if not bool(torch.isfinite(valid_weight).all()) or bool((valid_weight < 0).any()):
        raise ValueError("mask weights must be finite and non-negative")

    positive_weight = torch.as_tensor(
        presence_positive_weight,
        device=device,
        dtype=dtype,
    )
    if (
        positive_weight.numel() != 1
        or not bool(torch.isfinite(positive_weight))
        or bool(positive_weight <= 0)
    ):
        raise ValueError("presence_positive_weight must be finite and positive")
    event_positive_weights = boundary_event_positive_weights.to(
        device=device,
        dtype=dtype,
    )
    if (
        event_positive_weights.shape != (2,)
        or not bool(torch.isfinite(event_positive_weights).all())
        or bool((event_positive_weights <= 0).any())
    ):
        raise ValueError(
            "boundary event weights must contain two finite positive values"
        )

    presence_losses = F.binary_cross_entropy_with_logits(
        output.presence_logits,
        presence_targets,
        reduction="none",
        pos_weight=positive_weight,
    )
    presence_focal_weight = (
        (presence_targets - torch.sigmoid(output.presence_logits)).abs().square()
    )
    valid_mass = valid_weight.sum().clamp_min(1.0)
    presence_loss = (
        presence_losses * presence_focal_weight * valid_weight
    ).sum() / valid_mass

    event_losses = F.binary_cross_entropy_with_logits(
        output.boundary_event_logits,
        boundary_event_targets,
        reduction="none",
        pos_weight=event_positive_weights,
    )
    event_focal_weight = (
        (boundary_event_targets - torch.sigmoid(output.boundary_event_logits))
        .abs()
        .square()
    )
    event_channel_loss = (
        event_losses * event_focal_weight * valid_weight.unsqueeze(-1)
    ).sum(dim=(0, 1)) / valid_mass

    total = presence_loss + event_channel_loss.sum()
    return total, {
        "presence_loss": float(presence_loss.detach()),
        "start_event_loss": float(event_channel_loss[0].detach()),
        "end_event_loss": float(event_channel_loss[1].detach()),
    }


def _validate_targets(targets: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(targets).all()) or bool(
        ((targets < 0) | (targets > 1)).any()
    ):
        raise ValueError(f"{name} must be finite and within [0,1]")
