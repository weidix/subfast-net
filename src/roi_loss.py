from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .roi_pairs import select_embedding_pairs


@dataclass(frozen=True)
class RoiLossBreakdown:
    total: torch.Tensor
    presence_loss: torch.Tensor
    embedding_loss: torch.Tensor
    embedding_pairs: int
    embedding_local_positive_pairs: int
    embedding_local_negative_pairs: int
    embedding_ocr_negative_pairs: int
    embedding_skipped_pairs: int
    embedding_bce_loss: torch.Tensor
    positive_consistency_loss: torch.Tensor


def metric_embedding_loss(
    embedding: torch.Tensor,
    presence: torch.Tensor,
    segment_ids: list[str],
    *,
    alpha: float = 1.0,
    roots: list[str],
    video_ids: list[str | None],
    frame_indices: list[int | None],
    ocr_texts: list[str],
    frame_window: int,
    ocr_negative_enabled: bool,
    ocr_negative_max_similarity: float,
    positive_consistency_beta: float = 0.0,
    positive_consistency_margin: float = 0.75,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, int, int, int, int, int, torch.Tensor, torch.Tensor]:
    selection = select_embedding_pairs(
        presence=presence,
        segment_ids=segment_ids,
        roots=roots,
        video_ids=video_ids,
        frame_indices=frame_indices,
        ocr_texts=ocr_texts,
        frame_window=frame_window,
        ocr_negative_enabled=ocr_negative_enabled,
        ocr_negative_max_similarity=ocr_negative_max_similarity,
    )
    if not selection.pairs:
        zero = embedding.sum() * 0.0
        return zero, 0, 0, 0, 0, selection.skipped_pairs, zero, zero
    left = torch.tensor([pair.i for pair in selection.pairs], dtype=torch.long, device=embedding.device)
    right = torch.tensor([pair.j for pair in selection.pairs], dtype=torch.long, device=embedding.device)
    targets = torch.tensor([1.0 if pair.same else 0.0 for pair in selection.pairs], dtype=embedding.dtype, device=embedding.device)
    similarities = (embedding[left] * embedding[right]).sum(dim=1)
    logits = similarities / max(temperature, 1e-6)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets) * alpha
    positive_similarities = similarities[targets > 0.5]
    if positive_similarities.numel() == 0 or positive_consistency_beta <= 0.0:
        consistency_loss = embedding.sum() * 0.0
    else:
        consistency_loss = F.relu(positive_consistency_margin - positive_similarities).pow(2).mean()
    embedding_loss = bce_loss + positive_consistency_beta * consistency_loss
    return (
        embedding_loss,
        selection.embedding_pairs,
        selection.local_positive_pairs,
        selection.local_negative_pairs,
        selection.ocr_negative_pairs,
        selection.skipped_pairs,
        bce_loss,
        consistency_loss,
    )


def roi_presence_embedding_loss(
    presence_logit: torch.Tensor,
    embedding: torch.Tensor,
    presence: torch.Tensor,
    segment_ids: list[str],
    *,
    presence_loss_weights: torch.Tensor | None = None,
    roots: list[str],
    video_ids: list[str | None],
    frame_indices: list[int | None],
    ocr_texts: list[str],
    embedding_loss_weight: float,
    embedding_loss_alpha: float = 1.0,
    embedding_pair_frame_window: int,
    embedding_ocr_negative_enabled: bool,
    embedding_ocr_negative_max_similarity: float,
    embedding_positive_consistency_beta: float = 0.0,
    embedding_positive_consistency_margin: float = 0.75,
    embedding_temperature: float = 0.1,
) -> RoiLossBreakdown:
    presence_loss = F.binary_cross_entropy_with_logits(presence_logit, presence, weight=presence_loss_weights)
    (
        embedding_loss,
        embedding_pairs,
        embedding_local_positive_pairs,
        embedding_local_negative_pairs,
        embedding_ocr_negative_pairs,
        embedding_skipped_pairs,
        embedding_bce_loss,
        positive_consistency_loss,
    ) = metric_embedding_loss(
        embedding,
        presence,
        segment_ids,
        roots=roots,
        video_ids=video_ids,
        frame_indices=frame_indices,
        ocr_texts=ocr_texts,
        alpha=embedding_loss_alpha,
        frame_window=embedding_pair_frame_window,
        ocr_negative_enabled=embedding_ocr_negative_enabled,
        ocr_negative_max_similarity=embedding_ocr_negative_max_similarity,
        positive_consistency_beta=embedding_positive_consistency_beta,
        positive_consistency_margin=embedding_positive_consistency_margin,
        temperature=embedding_temperature,
    )
    total = presence_loss + embedding_loss_weight * embedding_loss
    return RoiLossBreakdown(
        total=total,
        presence_loss=presence_loss,
        embedding_loss=embedding_loss,
        embedding_pairs=embedding_pairs,
        embedding_local_positive_pairs=embedding_local_positive_pairs,
        embedding_local_negative_pairs=embedding_local_negative_pairs,
        embedding_ocr_negative_pairs=embedding_ocr_negative_pairs,
        embedding_skipped_pairs=embedding_skipped_pairs,
        embedding_bce_loss=embedding_bce_loss,
        positive_consistency_loss=positive_consistency_loss,
    )
