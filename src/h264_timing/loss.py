from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import SegmentModelOutput


def segment_detection_loss(
    output: SegmentModelOutput,
    targets: torch.Tensor,
    boundary_event_targets: torch.Tensor,
    mask: torch.Tensor,
    regression_mask: torch.Tensor,
    *,
    proposal_positive_weight: float | torch.Tensor,
    boundary_event_positive_weights: torch.Tensor,
    regression_score_threshold: float = 0.05,
    boundary_weight: float = 2.0,
    iou_weight: float = 1.0,
    boundary_event_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train proposal confidence and both boundaries of each proposed segment jointly."""
    expected_shape = (*output.score_logits.shape, 3)
    if targets.shape != expected_shape:
        raise ValueError(f"targets must have shape {expected_shape}")
    if boundary_event_targets.shape != (*output.score_logits.shape, 2):
        raise ValueError("boundary event targets must have shape [batch,time,2]")
    if mask.shape != output.score_logits.shape:
        raise ValueError("mask must match the proposal score shape")
    if regression_mask.shape != output.score_logits.shape:
        raise ValueError("regression_mask must match the proposal score shape")
    if not 0.0 < regression_score_threshold <= 1.0:
        raise ValueError("regression_score_threshold must be in (0,1]")
    positive_weight = torch.as_tensor(
        proposal_positive_weight,
        device=output.score_logits.device,
        dtype=output.score_logits.dtype,
    )
    if positive_weight.numel() != 1 or not bool(torch.isfinite(positive_weight)) or bool(
        positive_weight <= 0
    ):
        raise ValueError("proposal_positive_weight must be finite and positive")
    event_positive_weights = boundary_event_positive_weights.to(
        device=output.score_logits.device,
        dtype=output.score_logits.dtype,
    )
    if (
        event_positive_weights.shape != (2,)
        or not bool(torch.isfinite(event_positive_weights).all())
        or bool((event_positive_weights <= 0).any())
    ):
        raise ValueError("boundary event weights must contain two finite positive values")

    valid_weight = mask.to(dtype=output.score_logits.dtype)
    proposal_targets = targets[..., 0]
    proposal_losses = F.binary_cross_entropy_with_logits(
        output.score_logits,
        proposal_targets,
        reduction="none",
        pos_weight=positive_weight,
    )
    focal_weight = (
        proposal_targets - torch.sigmoid(output.score_logits)
    ).abs().square()
    proposal_losses = proposal_losses * focal_weight
    proposal_loss = (proposal_losses * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)

    event_losses = F.binary_cross_entropy_with_logits(
        output.boundary_event_logits,
        boundary_event_targets,
        reduction="none",
        pos_weight=event_positive_weights,
    )
    event_focal_weight = (
        boundary_event_targets - torch.sigmoid(output.boundary_event_logits)
    ).abs().square()
    event_losses = event_losses * event_focal_weight
    event_valid = valid_weight.unsqueeze(-1)
    event_channel_loss = (event_losses * event_valid).sum(dim=(0, 1)) / (
        valid_weight.sum().clamp_min(1.0)
    )

    target_duration = (targets[..., 2] - targets[..., 1]).clamp_min(1e-3)
    regression_weight = (
        valid_weight
        * regression_mask.to(valid_weight.dtype)
        * proposal_targets
        / target_duration
        * (proposal_targets >= regression_score_threshold).to(valid_weight.dtype)
    )
    regression_mass = regression_weight.sum().clamp_min(1.0)
    start_losses = F.smooth_l1_loss(
        output.start_offsets_seconds,
        targets[..., 1],
        reduction="none",
        beta=0.05,
    )
    end_losses = F.smooth_l1_loss(
        output.end_offsets_seconds,
        targets[..., 2],
        reduction="none",
        beta=0.05,
    )
    start_loss = (start_losses * regression_weight).sum() / regression_mass
    end_loss = (end_losses * regression_weight).sum() / regression_mass

    predicted_start = output.start_offsets_seconds
    predicted_end = output.end_offsets_seconds
    target_start = targets[..., 1]
    target_end = targets[..., 2]
    intersection = (
        torch.minimum(predicted_end, target_end)
        - torch.maximum(predicted_start, target_start)
    ).clamp_min(0.0)
    predicted_duration = (predicted_end - predicted_start).clamp_min(1e-6)
    target_segment_duration = (target_end - target_start).clamp_min(1e-6)
    union = predicted_duration + target_segment_duration - intersection
    temporal_iou = intersection / union.clamp_min(1e-6)
    temporal_iou_loss = ((1.0 - temporal_iou) * regression_weight).sum() / regression_mass

    total = (
        proposal_loss
        + boundary_weight * (start_loss + end_loss)
        + iou_weight * temporal_iou_loss
        + boundary_event_weight * event_channel_loss.sum()
    )
    return total, {
        "proposal_loss": float(proposal_loss.detach()),
        "start_boundary_loss": float(start_loss.detach()),
        "end_boundary_loss": float(end_loss.detach()),
        "temporal_iou_loss": float(temporal_iou_loss.detach()),
        "start_event_loss": float(event_channel_loss[0].detach()),
        "end_event_loss": float(event_channel_loss[1].detach()),
    }
