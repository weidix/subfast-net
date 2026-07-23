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


@dataclass(frozen=True)
class FramePresenceLossInput:
    presence_logits: torch.Tensor
    region_logits: torch.Tensor
    presence: torch.Tensor
    subtitle_masks: torch.Tensor
    supervision_masks: torch.Tensor


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


def _region_terms(
    inputs: list[FramePresenceLossInput],
) -> tuple[torch.Tensor, torch.Tensor]:
    reference = inputs[0].region_logits
    positive_bce_sum = reference.sum() * 0.0
    negative_bce_sum = reference.sum() * 0.0
    positive_pixel_count = reference.new_zeros(())
    negative_pixel_count = reference.new_zeros(())
    dice_losses: list[torch.Tensor] = []
    for item in inputs:
        region_targets = F.interpolate(
            item.subtitle_masks,
            size=item.region_logits.shape[-2:],
            mode="area",
        ).clamp(0.0, 1.0)
        supervision_mask = F.interpolate(
            item.supervision_masks,
            size=item.region_logits.shape[-2:],
            mode="area",
        ).gt(0.5).to(item.region_logits.dtype)
        losses = F.binary_cross_entropy_with_logits(item.region_logits, region_targets, reduction="none")
        positive_mask = (region_targets > 0.5).to(losses.dtype) * supervision_mask
        negative_mask = (region_targets <= 0.5).to(losses.dtype) * supervision_mask
        positive_bce_sum = positive_bce_sum + (losses * positive_mask).sum()
        negative_bce_sum = negative_bce_sum + (losses * negative_mask).sum()
        positive_pixel_count = positive_pixel_count + positive_mask.sum()
        negative_pixel_count = negative_pixel_count + negative_mask.sum()

        positive_samples = region_targets.flatten(start_dim=1).amax(dim=1) > 0.5
        if bool(positive_samples.any()):
            probability = torch.sigmoid(item.region_logits[positive_samples]) * supervision_mask[positive_samples]
            target = region_targets[positive_samples] * supervision_mask[positive_samples]
            intersection = (probability * target).sum(dim=(1, 2, 3))
            denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice_losses.append(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))

    bce_terms: list[torch.Tensor] = []
    if bool(positive_pixel_count > 0):
        bce_terms.append(positive_bce_sum / positive_pixel_count)
    if bool(negative_pixel_count > 0):
        bce_terms.append(negative_bce_sum / negative_pixel_count)
    region_bce = torch.stack(bce_terms).mean() if bce_terms else reference.sum() * 0.0
    region_dice = torch.cat(dice_losses).mean() if dice_losses else reference.sum() * 0.0
    return region_bce, region_dice


def frame_presence_macro_loss(
    inputs: list[FramePresenceLossInput],
    *,
    region_loss_weight: float,
    region_dice_weight: float,
    margin_loss_weight: float,
    positive_logit_margin: float,
    negative_logit_margin: float,
) -> FramePresenceLoss:
    if not inputs:
        raise ValueError("a logical macro batch must contain at least one execution micro batch")
    presence_logits = torch.cat([item.presence_logits for item in inputs])
    presence = torch.cat([item.presence for item in inputs])
    presence_bce = _balanced_binary_loss(presence_logits, presence)
    presence_margin = _presence_margin_loss(
        presence_logits,
        presence,
        positive_margin=positive_logit_margin,
        negative_margin=negative_logit_margin,
    )
    region_bce, region_dice = _region_terms(inputs)
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


def frame_presence_loss(
    presence_logits: torch.Tensor,
    region_logits: torch.Tensor,
    presence: torch.Tensor,
    subtitle_masks: torch.Tensor,
    supervision_masks: torch.Tensor,
    **kwargs: float,
) -> FramePresenceLoss:
    return frame_presence_macro_loss(
        [
            FramePresenceLossInput(
                presence_logits=presence_logits,
                region_logits=region_logits,
                presence=presence,
                subtitle_masks=subtitle_masks,
                supervision_masks=supervision_masks,
            )
        ],
        **kwargs,
    )


__all__ = [
    "FramePresenceLoss",
    "FramePresenceLossInput",
    "frame_presence_loss",
    "frame_presence_macro_loss",
]
