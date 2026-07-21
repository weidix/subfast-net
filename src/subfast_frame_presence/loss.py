from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class FramePresenceLoss:
    total: torch.Tensor
    presence_bce: torch.Tensor
    presence_margin: torch.Tensor
    region_bce: torch.Tensor
    region_dice: torch.Tensor
    region_projection: torch.Tensor
    region_boundary: torch.Tensor
    region_area: torch.Tensor
    region_edge: torch.Tensor


def _bounding_envelope(target: torch.Tensor) -> torch.Tensor:
    """Fill the outer target bounds without introducing a CPU synchronization."""
    occupied_x = target.amax(dim=2, keepdim=True)
    occupied_y = target.amax(dim=3, keepdim=True)
    x_from_left = torch.cummax(occupied_x, dim=3).values
    x_from_right = torch.flip(
        torch.cummax(torch.flip(occupied_x, dims=(3,)), dim=3).values,
        dims=(3,),
    )
    y_from_top = torch.cummax(occupied_y, dim=2).values
    y_from_bottom = torch.flip(
        torch.cummax(torch.flip(occupied_y, dims=(2,)), dim=2).values,
        dims=(2,),
    )
    return x_from_left * x_from_right * y_from_top * y_from_bottom


def _presence_margin_loss(
    logits: torch.Tensor,
    presence: torch.Tensor,
    *,
    positive_margin: float,
    negative_margin: float,
    hard_fraction: float,
) -> torch.Tensor:
    positive = presence > 0.5
    negative = ~positive
    losses: list[torch.Tensor] = []
    if bool(positive.any()):
        values = F.relu(positive_margin - logits[positive])
        count = max(1, round(values.numel() * hard_fraction))
        losses.append(values.topk(count).values.mean())
    if bool(negative.any()):
        values = F.relu(logits[negative] - negative_margin)
        count = max(1, round(values.numel() * hard_fraction))
        losses.append(values.topk(count).values.mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _balanced_presence_bce(logits: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    pixel_loss = F.binary_cross_entropy_with_logits(logits, presence, reduction="none")
    positive = presence > 0.5
    losses: list[torch.Tensor] = []
    if bool(positive.any()):
        losses.append(pixel_loss[positive].mean())
    if bool((~positive).any()):
        losses.append(pixel_loss[~positive].mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _balanced_region_bce(
    region_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_margin: float,
    negative_margin: float,
) -> torch.Tensor:
    positive_pixel_loss = F.softplus(positive_margin - region_logits)
    negative_pixel_loss = F.softplus(region_logits - negative_margin)
    positive_mass = target.sum(dim=(1, 2, 3))
    positive_loss = (positive_pixel_loss * target).sum(dim=(1, 2, 3)) / positive_mass.clamp_min(1.0)
    valid_mask = (region_logits.detach() > -11.0).to(region_logits.dtype)
    negative_mask = (1.0 - target) * valid_mask
    negative_loss = (negative_pixel_loss * negative_mask).sum(dim=(1, 2, 3)) / negative_mask.sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)
    per_sample = torch.where(positive_mass > 0.0, 0.5 * (positive_loss + negative_loss), negative_loss)
    return per_sample.mean()


def _positive_dice(region_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target.flatten(start_dim=1).sum(dim=1) > 0.0
    if not bool(positive.any()):
        return region_logits.sum() * 0.0
    valid_mask = (region_logits[positive].detach() > -11.0).to(region_logits.dtype)
    probability = torch.sigmoid(region_logits[positive]) * valid_mask
    selected_target = target[positive]
    intersection = (probability * selected_target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + selected_target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _projection_loss(region_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = target.flatten(start_dim=1).sum(dim=1) > 0.0
    if not bool(positive.any()):
        return region_logits.sum() * 0.0
    valid_mask = (region_logits[positive].detach() > -11.0).to(region_logits.dtype)
    probability = torch.sigmoid(region_logits[positive]) * valid_mask
    selected_target = target[positive]
    probability_x = probability.amax(dim=2)
    probability_y = probability.amax(dim=3)
    target_x = selected_target.amax(dim=2)
    target_y = selected_target.amax(dim=3)
    return 0.5 * (
        F.binary_cross_entropy(probability_x, target_x)
        + F.binary_cross_entropy(probability_y, target_y)
    )


def _boundary_margin_loss(
    region_logits: torch.Tensor,
    envelope_target: torch.Tensor,
    *,
    margin: float,
    hard_fraction: float,
) -> torch.Tensor:
    positive = envelope_target.flatten(start_dim=1).sum(dim=1) > 0.0
    if not bool(positive.any()):
        return region_logits.sum() * 0.0
    selected_target = envelope_target[positive]
    complement = F.pad(1.0 - selected_target, (1, 1, 1, 1), value=1.0)
    interior = 1.0 - F.max_pool2d(complement, kernel_size=3, stride=1)
    boundary = (selected_target - interior).clamp_min(0.0)
    violations = F.relu(margin - region_logits[positive]) * boundary
    per_sample = violations.sum(dim=(1, 2, 3)) / boundary.sum(dim=(1, 2, 3)).clamp_min(1.0)
    count = max(1, round(per_sample.numel() * hard_fraction))
    return per_sample.topk(count).values.mean()


def _envelope_boundary(envelope_target: torch.Tensor) -> torch.Tensor:
    complement = F.pad(1.0 - envelope_target, (1, 1, 1, 1), value=1.0)
    interior = 1.0 - F.max_pool2d(complement, kernel_size=3, stride=1)
    return (envelope_target - interior).clamp_min(0.0)


def _sampled_boundary(envelope_target: torch.Tensor) -> torch.Tensor:
    boundary = _envelope_boundary(envelope_target)
    height, width = boundary.shape[-2:]
    y = torch.arange(height, device=boundary.device).view(height, 1)
    x = torch.arange(width, device=boundary.device).view(1, width)
    sampling_mask = ((x + y) % 2 == 0).to(boundary.dtype)
    return boundary * sampling_mask


def _soft_area_loss(
    region_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    soft_area_limit: float,
    hard_fraction: float,
    activation_threshold: float,
    temperature: float,
) -> torch.Tensor:
    positive = target.flatten(start_dim=1).sum(dim=1) > 0.0
    if not bool(positive.any()):
        return region_logits.sum() * 0.0
    valid_mask = (region_logits[positive].detach() > -11.0).to(region_logits.dtype)
    threshold_logit = torch.logit(region_logits.new_tensor(activation_threshold))
    soft_active = torch.sigmoid((region_logits[positive] - threshold_logit) * temperature)
    selected_target = target[positive]
    overflow_mass = (soft_active * valid_mask * (1.0 - selected_target)).sum(dim=(1, 2, 3))
    target_mass = selected_target.sum(dim=(1, 2, 3)).clamp_min(1.0)
    overflow_limit = max(0.0, soft_area_limit - 1.0)
    violations = F.relu(overflow_mass / target_mass - overflow_limit)
    count = max(1, round(violations.numel() * hard_fraction))
    return violations.topk(count).values.mean()


def frame_presence_loss(
    presence_logits: torch.Tensor,
    region_logits: torch.Tensor,
    compact_region_logits: torch.Tensor,
    presence: torch.Tensor,
    region_target: torch.Tensor,
    *,
    presence_margin_weight: float,
    presence_hard_fraction: float,
    positive_margin: float,
    negative_margin: float,
    region_loss_weight: float,
    region_bce_weight: float,
    region_positive_margin: float,
    region_negative_margin: float,
    dice_weight: float,
    projection_weight: float,
    boundary_weight: float,
    boundary_margin: float,
    boundary_hard_fraction: float,
    area_weight: float,
    soft_area_limit: float,
    area_hard_fraction: float,
    area_activation_threshold: float,
    area_temperature: float,
    edge_weight: float,
) -> FramePresenceLoss:
    presence_bce = _balanced_presence_bce(presence_logits, presence)
    presence_margin = _presence_margin_loss(
        presence_logits,
        presence,
        positive_margin=positive_margin,
        negative_margin=negative_margin,
        hard_fraction=presence_hard_fraction,
    )
    envelope_target = _bounding_envelope(region_target)
    region_bce = _balanced_region_bce(
        compact_region_logits,
        envelope_target,
        positive_margin=region_positive_margin,
        negative_margin=region_negative_margin,
    )
    region_dice = _positive_dice(compact_region_logits, envelope_target)
    region_projection = _projection_loss(compact_region_logits, envelope_target)
    region_boundary = _boundary_margin_loss(
        compact_region_logits,
        envelope_target,
        margin=boundary_margin,
        hard_fraction=boundary_hard_fraction,
    )
    region_area = _soft_area_loss(
        compact_region_logits,
        region_target,
        soft_area_limit=soft_area_limit,
        hard_fraction=area_hard_fraction,
        activation_threshold=area_activation_threshold,
        temperature=area_temperature,
    )
    region_edge = _balanced_region_bce(
        region_logits,
        _sampled_boundary(envelope_target),
        positive_margin=region_positive_margin,
        negative_margin=region_negative_margin,
    )
    region_total = (
        region_bce_weight * region_bce
        + dice_weight * region_dice
        + projection_weight * region_projection
        + boundary_weight * region_boundary
        + area_weight * region_area
        + edge_weight * region_edge
    )
    total = (
        presence_bce
        + presence_margin_weight * presence_margin
        + region_loss_weight * region_total
    )
    return FramePresenceLoss(
        total=total,
        presence_bce=presence_bce,
        presence_margin=presence_margin,
        region_bce=region_bce,
        region_dice=region_dice,
        region_projection=region_projection,
        region_boundary=region_boundary,
        region_area=region_area,
        region_edge=region_edge,
    )
