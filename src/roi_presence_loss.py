from __future__ import annotations

import unicodedata

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


def presence_loss_weights(
    presence: torch.Tensor,
    ocr_texts: list[str],
    weight: float,
) -> torch.Tensor | None:
    if weight == 1.0:
        return None
    weights = torch.ones_like(presence)
    mask = short_positive_mask(presence, ocr_texts)
    if bool(mask.any()):
        weights[mask] = weight
    return weights


def short_positive_mask_loss(
    textness_map: torch.Tensor,
    subtitle_masks: torch.Tensor,
    presence: torch.Tensor,
    ocr_texts: list[str],
    weight: float,
) -> torch.Tensor:
    if weight <= 0.0:
        return textness_map.sum() * 0.0
    mask = short_positive_mask(presence, ocr_texts)
    if not bool(mask.any()):
        return textness_map.sum() * 0.0
    target = F.interpolate(
        subtitle_masks.to(device=textness_map.device, dtype=textness_map.dtype),
        size=textness_map.shape[-2:],
        mode="area",
    ).clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(textness_map[mask], target[mask]) * weight
