from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import torch
from torch.nn import functional as F

_SHORT_SUBTITLE_MAX_CHARS = 2


def normalize_presence_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(
        char
        for char in normalized
        if not char.isspace() and not unicodedata.category(char).startswith(("P", "S"))
    )


def short_positive_mask(presence: torch.Tensor, ocr_texts: list[str]) -> torch.Tensor:
    return torch.tensor(
        [
            bool(is_positive) and 0 < len(normalize_presence_text(text)) <= _SHORT_SUBTITLE_MAX_CHARS
            for is_positive, text in zip((presence > 0.5).detach().cpu().tolist(), ocr_texts, strict=True)
        ],
        dtype=torch.bool,
        device=presence.device,
    )


def presence_importance_weights(
    presence: torch.Tensor,
    *,
    sampled_positive_prior: float,
    target_positive_prior: float,
) -> torch.Tensor | None:
    """Undo an optional balanced sampler's prior shift in the scalar BCE."""
    if not 0.0 < sampled_positive_prior < 1.0:
        return None
    if not 0.0 < target_positive_prior < 1.0:
        return None
    if abs(sampled_positive_prior - target_positive_prior) < 1e-9:
        return None
    positive_weight = target_positive_prior / sampled_positive_prior
    negative_weight = (1.0 - target_positive_prior) / (1.0 - sampled_positive_prior)
    return torch.where(
        presence > 0.5,
        presence.new_tensor(positive_weight),
        presence.new_tensor(negative_weight),
    )


def resize_candidate_masks(subtitle_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        subtitle_masks,
        size=size,
        mode="area",
    ).clamp(0.0, 1.0)


def subtitle_region_targets(
    subtitle_masks: torch.Tensor,
    presence: torch.Tensor,
    size: tuple[int, int],
    valid_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    candidates = resize_candidate_masks(subtitle_masks, size)
    target = candidates * (presence > 0.5).to(candidates.dtype).view(-1, 1, 1, 1)
    if valid_masks is not None:
        valid = F.interpolate(valid_masks, size=size, mode="area") > 0.5
        target = target * valid.to(target.dtype)
    return target


def positive_with_region_mask(subtitle_masks: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    has_region = subtitle_masks.flatten(start_dim=1).amax(dim=1) > 0.0
    return (presence > 0.5) & has_region


@dataclass(frozen=True)
class RegionLoss:
    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor
    projection: torch.Tensor


def _balanced_map_bce(
    region_logits: torch.Tensor,
    target: torch.Tensor,
    candidate_masks: torch.Tensor,
    presence: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    text_distractor_weight: float,
) -> torch.Tensor:
    pixel_loss = F.binary_cross_entropy_with_logits(region_logits, target, reduction="none")
    positive_mass = target.sum(dim=(1, 2, 3))
    positive_loss = (pixel_loss * target).sum(dim=(1, 2, 3)) / positive_mass.clamp_min(1.0)

    distractor = candidate_masks * (presence <= 0.5).to(target.dtype).view(-1, 1, 1, 1)
    negative_weight = valid_mask * (1.0 - target) * (1.0 + text_distractor_weight * distractor)
    negative_loss = (pixel_loss * negative_weight).sum(dim=(1, 2, 3)) / negative_weight.sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)
    per_sample = torch.where(positive_mass > 0.0, 0.5 * (positive_loss + negative_loss), negative_loss)
    return per_sample.mean()


def _positive_dice_loss(
    region_logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    positive = target.flatten(start_dim=1).sum(dim=1) > 0.0
    if not bool(positive.any()):
        return region_logits.sum() * 0.0
    probability = torch.sigmoid(region_logits[positive]) * valid_mask[positive]
    selected_target = target[positive]
    intersection = (probability * selected_target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + selected_target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _projection_loss(
    region_logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    positive = target.flatten(start_dim=1).sum(dim=1) > 0.0
    if not bool(positive.any()):
        return region_logits.sum() * 0.0
    selected_valid = valid_mask[positive]
    probability = torch.sigmoid(region_logits[positive]) * selected_valid
    selected_target = target[positive]
    probability_x = probability.amax(dim=2)
    probability_y = probability.amax(dim=3)
    target_x = selected_target.amax(dim=2)
    target_y = selected_target.amax(dim=3)
    valid_x = selected_valid.amax(dim=2)
    valid_y = selected_valid.amax(dim=3)
    loss_x = F.binary_cross_entropy(probability_x, target_x, reduction="none")
    loss_y = F.binary_cross_entropy(probability_y, target_y, reduction="none")
    return 0.5 * (
        (loss_x * valid_x).sum() / valid_x.sum().clamp_min(1.0)
        + (loss_y * valid_y).sum() / valid_y.sum().clamp_min(1.0)
    )


def subtitle_region_loss(
    region_logits: torch.Tensor,
    subtitle_masks: torch.Tensor,
    presence: torch.Tensor,
    valid_masks: torch.Tensor | None = None,
    *,
    dice_weight: float,
    projection_weight: float,
    text_distractor_weight: float,
) -> RegionLoss:
    candidate_masks = resize_candidate_masks(subtitle_masks, region_logits.shape[-2:])
    valid_mask = (
        (F.interpolate(valid_masks, size=region_logits.shape[-2:], mode="area") > 0.5).to(
            region_logits.dtype
        )
        if valid_masks is not None
        else torch.ones_like(candidate_masks)
    )
    candidate_masks = candidate_masks * valid_mask
    target = candidate_masks * (presence > 0.5).to(region_logits.dtype).view(-1, 1, 1, 1)
    bce = _balanced_map_bce(
        region_logits,
        target,
        candidate_masks,
        presence,
        valid_mask,
        text_distractor_weight=text_distractor_weight,
    )
    dice = _positive_dice_loss(region_logits, target, valid_mask)
    projection = _projection_loss(region_logits, target, valid_mask)
    return RegionLoss(
        total=bce + dice_weight * dice + projection_weight * projection,
        bce=bce,
        dice=dice,
        projection=projection,
    )


def feathered_region_mask(subtitle_masks: torch.Tensor, *, dilation: int = 5) -> torch.Tensor:
    if dilation <= 0 or dilation % 2 == 0:
        raise ValueError("dilation must be a positive odd integer")
    hard_mask = (subtitle_masks > 0.0).to(subtitle_masks.dtype)
    expanded = F.max_pool2d(hard_mask, dilation, stride=1, padding=dilation // 2)
    return F.avg_pool2d(expanded, 3, stride=1, padding=1).clamp(0.0, 1.0)


def composite_valid_mask(
    subtitle_masks: torch.Tensor,
    inside_valid_mask: torch.Tensor,
    outside_valid_mask: torch.Tensor,
) -> torch.Tensor:
    alpha = feathered_region_mask(subtitle_masks).to(
        device=inside_valid_mask.device,
        dtype=inside_valid_mask.dtype,
    )
    blended = alpha * inside_valid_mask + (1.0 - alpha) * outside_valid_mask
    return (blended > 0.5).to(inside_valid_mask.dtype)


def erase_subtitle_regions(
    images: torch.Tensor,
    subtitle_masks: torch.Tensor,
    *,
    donor_images: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace annotated regions while leaving the rest of each ROI untouched."""
    if images.shape[0] != subtitle_masks.shape[0]:
        raise ValueError("images and subtitle_masks must have the same batch size")
    alpha = feathered_region_mask(subtitle_masks).to(device=images.device, dtype=images.dtype)
    if donor_images is None:
        vertical_shift = max(1, images.shape[-2] // 3)
        replacement = 0.5 * (
            torch.roll(images, shifts=vertical_shift, dims=-2)
            + torch.roll(images, shifts=-vertical_shift, dims=-2)
        )
    else:
        if donor_images.shape != images.shape:
            raise ValueError("donor_images must have the same shape as images")
        replacement = donor_images
    return images * (1.0 - alpha) + replacement * alpha


def transplant_subtitle_regions(
    positive_images: torch.Tensor,
    subtitle_masks: torch.Tensor,
    background_images: torch.Tensor,
) -> torch.Tensor:
    """Place the annotated foreground on an unrelated empty-ROI background."""
    if positive_images.shape != background_images.shape:
        raise ValueError("positive_images and background_images must have the same shape")
    alpha = feathered_region_mask(subtitle_masks).to(
        device=positive_images.device,
        dtype=positive_images.dtype,
    )
    return positive_images * alpha + background_images * (1.0 - alpha)


@dataclass(frozen=True)
class CounterfactualLoss:
    total: torch.Tensor
    erased_bce: torch.Tensor
    necessity: torch.Tensor
    sufficiency: torch.Tensor
    seam_control_bce: torch.Tensor


def counterfactual_presence_loss(
    original_logits: torch.Tensor,
    erased_logits: torch.Tensor,
    transplanted_logits: torch.Tensor | None,
    seam_control_logits: torch.Tensor | None,
    *,
    margin: float,
) -> CounterfactualLoss:
    if not original_logits.numel():
        zero = original_logits.sum() * 0.0
        return CounterfactualLoss(zero, zero, zero, zero, zero)
    erased_bce = F.binary_cross_entropy_with_logits(erased_logits, torch.zeros_like(erased_logits))
    necessity = F.relu(margin - (original_logits - erased_logits)).mean()
    sufficiency = (
        F.smooth_l1_loss(transplanted_logits, original_logits.detach())
        if transplanted_logits is not None
        else original_logits.sum() * 0.0
    )
    seam_control_bce = (
        F.binary_cross_entropy_with_logits(seam_control_logits, torch.zeros_like(seam_control_logits))
        if seam_control_logits is not None
        else original_logits.sum() * 0.0
    )
    return CounterfactualLoss(
        total=erased_bce + necessity + sufficiency + seam_control_bce,
        erased_bce=erased_bce,
        necessity=necessity,
        sufficiency=sufficiency,
        seam_control_bce=seam_control_bce,
    )
