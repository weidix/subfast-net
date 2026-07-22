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


def _balanced_binary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets > 0.5
    terms: list[torch.Tensor] = []
    if bool(positive.any()):
        terms.append(losses[positive].mean())
    if bool((~positive).any()):
        terms.append(losses[~positive].mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _presence_margin_loss(
    logits: torch.Tensor,
    presence: torch.Tensor,
    *,
    positive_margin: float,
    negative_margin: float,
) -> torch.Tensor:
    positive = presence > 0.5
    terms: list[torch.Tensor] = []
    if bool(positive.any()):
        violations = F.softplus(positive_margin - logits[positive])
        terms.append(violations.topk(max(1, (violations.numel() + 3) // 4)).values.mean())
    if bool((~positive).any()):
        violations = F.softplus(logits[~positive] - negative_margin)
        terms.append(violations.topk(max(1, (violations.numel() + 3) // 4)).values.mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _region_bce(
    region_logits: torch.Tensor,
    region_targets: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(region_logits, region_targets, reduction="none")
    positive_mask = (region_targets > 0.5).to(losses.dtype) * supervision_mask
    negative_mask = (region_targets <= 0.5).to(losses.dtype) * supervision_mask
    terms: list[torch.Tensor] = []
    if bool(positive_mask.any()):
        terms.append((losses * positive_mask).sum() / positive_mask.sum().clamp_min(1.0))
    if bool(negative_mask.any()):
        terms.append((losses * negative_mask).sum() / negative_mask.sum().clamp_min(1.0))
    return torch.stack(terms).mean() if terms else region_logits.sum() * 0.0


def _region_dice(
    region_logits: torch.Tensor,
    region_targets: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> torch.Tensor:
    positive = region_targets.flatten(start_dim=1).amax(dim=1) > 0.5
    if not bool(positive.any()):
        return region_logits.sum() * 0.0
    probability = torch.sigmoid(region_logits[positive]) * supervision_mask[positive]
    target = region_targets[positive] * supervision_mask[positive]
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def frame_presence_loss(
    presence_logits: torch.Tensor,
    region_logits: torch.Tensor,
    presence: torch.Tensor,
    subtitle_masks: torch.Tensor,
    supervision_masks: torch.Tensor,
    *,
    region_loss_weight: float,
    region_dice_weight: float,
    margin_loss_weight: float,
    positive_logit_margin: float,
    negative_logit_margin: float,
) -> FramePresenceLoss:
    region_targets = F.interpolate(
        subtitle_masks,
        size=region_logits.shape[-2:],
        mode="area",
    ).clamp(0.0, 1.0)
    supervision_mask = F.interpolate(
        supervision_masks,
        size=region_logits.shape[-2:],
        mode="area",
    ).gt(0.5).to(region_logits.dtype)
    presence_bce = _balanced_binary_loss(presence_logits, presence)
    presence_margin = _presence_margin_loss(
        presence_logits,
        presence,
        positive_margin=positive_logit_margin,
        negative_margin=negative_logit_margin,
    )
    region_bce = _region_bce(region_logits, region_targets, supervision_mask)
    region_dice = _region_dice(region_logits, region_targets, supervision_mask)
    total = (
        presence_bce
        + margin_loss_weight * presence_margin
        + region_loss_weight * (region_bce + region_dice_weight * region_dice)
    )
    return FramePresenceLoss(
        total=total,
        presence_bce=presence_bce,
        presence_margin=presence_margin,
        region_bce=region_bce,
        region_dice=region_dice,
    )


__all__ = ["FramePresenceLoss", "frame_presence_loss"]
