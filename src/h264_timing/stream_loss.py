from __future__ import annotations

import math

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
    segment_anchor_targets: torch.Tensor | None = None,
    segment_anchor_positive_weight: float | torch.Tensor | None = None,
    segment_boundary_weight: float = 2.0,
    segment_loss_weight: float = 1.0,
    negative_weight: float = 1.0,
    boundary_event_loss_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
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
    if sample_weight is not None:
        if sample_weight.shape != mask.shape:
            raise ValueError("sample weights must match the mask shape")
        sample_weight = sample_weight.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(sample_weight).all()) or bool(
            (sample_weight < 0).any()
        ):
            raise ValueError("sample weights must be finite and non-negative")
        valid_weight = valid_weight * sample_weight
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
    if not math.isfinite(float(negative_weight)) or negative_weight <= 0.0:
        raise ValueError("negative weight must be finite and positive")
    if (
        not math.isfinite(float(boundary_event_loss_weight))
        or boundary_event_loss_weight < 0.0
    ):
        raise ValueError(
            "boundary event loss weight must be finite and non-negative"
        )
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
    presence_class_weight = torch.where(
        presence_targets > 0.5,
        torch.ones_like(presence_targets),
        torch.as_tensor(negative_weight, device=device, dtype=dtype),
    )
    valid_mass = valid_weight.sum().clamp_min(1.0)
    presence_loss = (
        presence_losses * presence_focal_weight * presence_class_weight * valid_weight
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
    event_class_weight = torch.where(
        boundary_event_targets > 0.5,
        torch.ones_like(boundary_event_targets),
        torch.as_tensor(negative_weight, device=device, dtype=dtype),
    )
    event_channel_loss = (
        event_losses
        * event_focal_weight
        * event_class_weight
        * valid_weight.unsqueeze(-1)
    ).sum(dim=(0, 1)) / valid_mass

    total = presence_loss + boundary_event_loss_weight * event_channel_loss.sum()
    components = {
        "presence_loss": float(presence_loss.detach()),
        "start_event_loss": float(event_channel_loss[0].detach()),
        "end_event_loss": float(event_channel_loss[1].detach()),
    }
    if segment_anchor_targets is not None:
        if segment_anchor_targets.shape != (*output.presence_logits.shape, 3):
            raise ValueError(
                "segment anchor targets must have shape [batch,time,3]"
            )
        if segment_anchor_positive_weight is None:
            raise ValueError(
                "segment anchor positive weight is required with anchor targets"
            )
        if not math.isfinite(float(segment_boundary_weight)) or segment_boundary_weight < 0.0:
            raise ValueError("segment boundary weight must be finite and non-negative")
        if not math.isfinite(float(segment_loss_weight)) or segment_loss_weight < 0.0:
            raise ValueError("segment loss weight must be finite and non-negative")
        anchor_targets = segment_anchor_targets.to(device=device, dtype=dtype)
        _validate_targets(anchor_targets[..., 0], "segment anchor targets")
        if not bool(torch.isfinite(anchor_targets[..., 1:]).all()):
            raise ValueError("segment anchor offsets must be finite")
        anchor_weight = torch.as_tensor(
            segment_anchor_positive_weight, device=device, dtype=dtype
        )
        if (
            anchor_weight.numel() != 1
            or not bool(torch.isfinite(anchor_weight))
            or bool(anchor_weight <= 0.0)
        ):
            raise ValueError("segment anchor positive weight must be finite and positive")
        anchor_logits = output.segment_anchor_logits
        anchor_targets_binary = anchor_targets[..., 0]
        anchor_losses = F.binary_cross_entropy_with_logits(
            anchor_logits,
            anchor_targets_binary,
            reduction="none",
            pos_weight=anchor_weight,
        )
        anchor_focal_weight = (
            anchor_targets_binary - torch.sigmoid(anchor_logits)
        ).abs().square()
        anchor_loss = (
            anchor_losses * anchor_focal_weight * valid_weight
        ).sum() / valid_mass

        positive_mask = valid_weight * anchor_targets_binary
        positive_mass = positive_mask.sum().clamp_min(1.0)
        start_losses = F.smooth_l1_loss(
            output.segment_start_offsets_seconds,
            anchor_targets[..., 1],
            reduction="none",
            beta=0.02,
        )
        end_losses = F.smooth_l1_loss(
            output.segment_end_offsets_seconds,
            anchor_targets[..., 2],
            reduction="none",
            beta=0.02,
        )
        start_loss = (start_losses * positive_mask).sum() / positive_mass
        end_loss = (end_losses * positive_mask).sum() / positive_mass
        predicted_start = output.segment_start_offsets_seconds
        predicted_end = output.segment_end_offsets_seconds
        target_start = anchor_targets[..., 1]
        target_end = anchor_targets[..., 2]
        intersection = (
            torch.minimum(predicted_end, target_end)
            - torch.maximum(predicted_start, target_start)
        ).clamp_min(0.0)
        predicted_duration = (predicted_end - predicted_start).clamp_min(1e-6)
        target_duration = (target_end - target_start).clamp_min(1e-6)
        union = predicted_duration + target_duration - intersection
        iou_loss = (((1.0 - intersection / union.clamp_min(1e-6)) * positive_mask).sum() / positive_mass)
        total = total + segment_loss_weight * (
            anchor_loss
            + segment_boundary_weight * (start_loss + end_loss + iou_loss)
        )
        components.update(
            {
                "segment_anchor_loss": float(anchor_loss.detach()),
                "segment_start_loss": float(start_loss.detach()),
                "segment_end_loss": float(end_loss.detach()),
                "segment_anchor_iou_loss": float(iou_loss.detach()),
            }
        )
    return total, components


def _validate_targets(targets: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(targets).all()) or bool(
        ((targets < 0) | (targets > 1)).any()
    ):
        raise ValueError(f"{name} must be finite and within [0,1]")
