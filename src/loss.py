from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    region_bce: torch.Tensor
    kernel_bce: torch.Tensor
    region_dice: torch.Tensor
    kernel_dice: torch.Tensor


def balanced_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0.5
    if not torch.any(valid):
        return logits.sum() * 0.0
    positives = (targets > 0.5) & valid
    negatives = (targets <= 0.5) & valid
    positive_count = positives.sum().clamp_min(1)
    negative_count = negatives.sum()
    positive_weight = (negative_count / positive_count).clamp_min(1.0).detach()
    weights = torch.where(positives, positive_weight, torch.ones_like(targets))
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights * mask
    return loss.sum() / valid.sum().clamp_min(1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits) * mask
    targets = targets * mask
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + 1.0) / (probs.sum() + targets.sum() + 1.0)


def detection_loss(logits: torch.Tensor, regions: torch.Tensor, kernels: torch.Tensor, training_masks: torch.Tensor) -> LossBreakdown:
    region_logits = logits[:, 0:1]
    kernel_logits = logits[:, 1:2]
    region_bce = balanced_bce_with_logits(region_logits, regions, training_masks)
    kernel_bce = balanced_bce_with_logits(kernel_logits, kernels, training_masks)
    region_dice = dice_loss(region_logits, regions, training_masks)
    kernel_dice = dice_loss(kernel_logits, kernels, training_masks)
    total = region_bce + kernel_bce + region_dice + kernel_dice
    return LossBreakdown(total=total, region_bce=region_bce, kernel_bce=kernel_bce, region_dice=region_dice, kernel_dice=kernel_dice)

