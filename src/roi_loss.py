from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .roi_pairs import (
    EmbeddingPair,
    EmbeddingPairSelection,
    is_same_subtitle_text,
    max_ocr_negative_pairs,
    normalize_ocr_text,
    normalized_ocr_text_similarity_at_most,
    select_embedding_pairs,
)


@dataclass(frozen=True)
class _EmbeddingMemoryNegative:
    segment_id: str
    video_id: str | None
    normalized_ocr: str
    embedding: torch.Tensor


class EmbeddingPairMemory:
    def __init__(self, *, ocr_negative_max_similarity: float, ocr_negative_ratio: float) -> None:
        self.ocr_negative_max_similarity = ocr_negative_max_similarity
        self.ocr_negative_ratio = ocr_negative_ratio
        self._embeddings: dict[tuple[str, str], torch.Tensor] = {}
        self._local_negative_index: dict[tuple[str, str, str], list[_EmbeddingMemoryNegative]] = {}
        self._ocr_negative_bank: dict[str, list[_EmbeddingMemoryNegative]] = {}

    def _rebuild_local_negative_index(self, root: str) -> None:
        for key in [key for key in self._local_negative_index if key[0] == root]:
            del self._local_negative_index[key]
        for entry in self._ocr_negative_bank.get(root, []):
            if entry.video_id is None:
                continue
            self._local_negative_index.setdefault((root, entry.video_id, entry.segment_id), []).append(entry)

    def loss_and_update(
        self,
        embedding: torch.Tensor,
        presence: torch.Tensor,
        segment_ids: list[str],
        roots: list[str],
        video_ids: list[str | None],
        ocr_texts: list[str],
        adjacent_segment_ids: list[set[str] | frozenset[str] | list[str] | tuple[str, ...]],
    ) -> tuple[torch.Tensor, int]:
        current_indices: list[int] = []
        previous_embeddings: list[torch.Tensor] = []
        targets: list[float] = []
        updates: list[tuple[tuple[str, str], torch.Tensor]] = []
        for index, is_positive in enumerate((presence.detach() > 0.5).tolist()):
            if not is_positive:
                continue
            key = (roots[index], segment_ids[index])
            previous = self._embeddings.get(key)
            if previous is not None:
                current_indices.append(index)
                previous_embeddings.append(previous)
                targets.append(1.0)
            updates.append((key, embedding[index].detach()))
            video_id = video_ids[index]
            normalized_ocr = normalize_ocr_text(ocr_texts[index])
            local_negative_embeddings: list[torch.Tensor] = []
            ocr_negative_embeddings: list[torch.Tensor] = []
            adjacent_ids = adjacent_segment_ids[index]
            if video_id is not None:
                for adjacent_segment_id in adjacent_ids:
                    if adjacent_segment_id == segment_ids[index]:
                        continue
                    for previous in self._local_negative_index.get((roots[index], video_id, adjacent_segment_id), []):
                        if is_same_subtitle_text(normalized_ocr, previous.normalized_ocr):
                            continue
                        local_negative_embeddings.append(previous.embedding)
            for previous in self._ocr_negative_bank.get(roots[index], []):
                if previous.segment_id == segment_ids[index]:
                    continue
                if (
                    video_id is not None
                    and previous.video_id == video_id
                    and previous.segment_id in adjacent_ids
                ):
                    continue
                if normalized_ocr_text_similarity_at_most(
                    normalized_ocr,
                    previous.normalized_ocr,
                    self.ocr_negative_max_similarity,
                ):
                    ocr_negative_embeddings.append(previous.embedding)
            ocr_limit = max_ocr_negative_pairs(
                local_negative_pairs=len(local_negative_embeddings),
                positive_pairs=1,
                ocr_negative_ratio=self.ocr_negative_ratio,
            )
            for previous_embedding in local_negative_embeddings:
                current_indices.append(index)
                previous_embeddings.append(previous_embedding)
                targets.append(0.0)
            for previous_embedding in ocr_negative_embeddings[:ocr_limit]:
                current_indices.append(index)
                previous_embeddings.append(previous_embedding)
                targets.append(0.0)
        for key, value in updates:
            self._embeddings[key] = value
        for index, is_positive in enumerate((presence.detach() > 0.5).tolist()):
            if is_positive:
                root = roots[index]
                bank = self._ocr_negative_bank.setdefault(root, [])
                entry = _EmbeddingMemoryNegative(
                    segment_id=segment_ids[index],
                    video_id=video_ids[index],
                    normalized_ocr=normalize_ocr_text(ocr_texts[index]),
                    embedding=embedding[index].detach(),
                )
                bank.append(entry)
                if entry.video_id is not None:
                    self._local_negative_index.setdefault((root, entry.video_id, entry.segment_id), []).append(entry)
                if len(bank) > 512:
                    del bank[:-512]
                    self._rebuild_local_negative_index(root)
        if not current_indices:
            return embedding.sum() * 0.0, 0
        # Calculate every selected pair in one tensor operation. Performing one
        # GPU dot product per pair makes the memory loss dispatch thousands of
        # tiny kernels per batch even though the selected pairs are independent.
        left = embedding[torch.tensor(current_indices, dtype=torch.long, device=embedding.device)]
        right = torch.stack(previous_embeddings)
        scores = (left * right).sum(dim=1)
        target = torch.tensor(targets, dtype=scores.dtype, device=scores.device)
        return embedding_margin_loss(scores, target), len(current_indices)


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
    embedding_candidate_positive_pairs: int
    embedding_candidate_negative_pairs: int
    embedding_selected_positive_pairs: int
    embedding_selected_negative_pairs: int
    embedding_margin_loss: torch.Tensor
    positive_consistency_loss: torch.Tensor
    supervised_contrastive_loss: torch.Tensor


@dataclass(frozen=True)
class BalancedEmbeddingPairs:
    indices: torch.Tensor
    candidate_positive_pairs: int
    candidate_negative_pairs: int
    selected_positive_pairs: int
    selected_negative_pairs: int


def balance_embedding_pairs(
    similarities: torch.Tensor,
    targets: torch.Tensor,
    *,
    negative_ratio: float,
) -> BalancedEmbeddingPairs:
    if not 0.0 <= negative_ratio <= 1.0:
        raise ValueError("negative_ratio must be in [0, 1]")
    positive_indices = torch.nonzero(targets > 0.5, as_tuple=False).flatten()
    negative_indices = torch.nonzero(targets <= 0.5, as_tuple=False).flatten()
    positive_count = int(positive_indices.numel())
    negative_count = int(negative_indices.numel())
    if negative_count == 0:
        selected_negative = negative_indices
    else:
        ordered_negative = negative_indices[torch.argsort(similarities[negative_indices], descending=True)]
        if positive_count == 0:
            selected_negative = ordered_negative[:1]
        elif negative_ratio >= 1.0:
            selected_negative = ordered_negative
        elif negative_ratio <= 0.0:
            selected_negative = ordered_negative[:0]
        else:
            limit = round(positive_count * negative_ratio / (1.0 - negative_ratio))
            selected_negative = ordered_negative[:limit]
    indices = torch.cat((positive_indices, selected_negative))
    return BalancedEmbeddingPairs(
        indices=indices,
        candidate_positive_pairs=positive_count,
        candidate_negative_pairs=negative_count,
        selected_positive_pairs=positive_count,
        selected_negative_pairs=int(selected_negative.numel()),
    )


def embedding_margin_loss(
    similarities: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_margin: float = 0.9,
    negative_margin: float = 0.1,
    gamma_positive: float = 20.0,
    gamma_negative: float = 40.0,
    hard_negative_weight: float = 2.0,
) -> torch.Tensor:
    positive_violation = F.relu(positive_margin - similarities)
    negative_violation = F.relu(similarities - negative_margin)
    positive_mask = targets > 0.5
    negative_mask = ~positive_mask
    hard_positive_mask = positive_mask & (positive_violation > 0.0)
    hard_negative_mask = negative_mask & (negative_violation > 0.0)
    positive_loss = (
        torch.logsumexp(gamma_positive * positive_violation[hard_positive_mask], dim=0) / gamma_positive
        if bool(hard_positive_mask.any())
        else similarities.sum() * 0.0
    )
    negative_loss = (
        torch.logsumexp(gamma_negative * negative_violation[hard_negative_mask], dim=0) / gamma_negative
        if bool(hard_negative_mask.any())
        else similarities.sum() * 0.0
    )
    return positive_loss + hard_negative_weight * negative_loss


def supervised_contrastive_embedding_loss(
    embedding: torch.Tensor,
    presence: torch.Tensor,
    segment_ids: list[str],
    roots: list[str],
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    positive_indices = [index for index, is_present in enumerate((presence.detach() > 0.5).tolist()) if is_present]
    if len(positive_indices) < 2:
        return embedding.sum() * 0.0
    index_tensor = torch.tensor(positive_indices, dtype=torch.long, device=embedding.device)
    scoped_embedding = embedding[index_tensor]
    labels = [(roots[index], segment_ids[index]) for index in positive_indices]
    label_equal = [
        [left == right for right in labels]
        for left in labels
    ]
    positive_mask = torch.tensor(label_equal, dtype=torch.bool, device=embedding.device)
    self_mask = torch.eye(len(positive_indices), dtype=torch.bool, device=embedding.device)
    positive_mask = positive_mask & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not bool(valid_anchor.any()):
        return embedding.sum() * 0.0
    logits = scoped_embedding @ scoped_embedding.T / temperature
    logits = logits - logits.detach().max(dim=1, keepdim=True).values
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = (log_prob * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -positive_log_prob[valid_anchor].mean()


def metric_embedding_loss(
    embedding: torch.Tensor,
    presence: torch.Tensor,
    segment_ids: list[str],
    *,
    alpha: float = 1.0,
    roots: list[str],
    video_ids: list[str | None],
    ocr_texts: list[str],
    adjacent_segment_ids: list[set[str] | frozenset[str] | list[str] | tuple[str, ...]],
    ocr_negative_enabled: bool,
    ocr_negative_max_similarity: float,
    ocr_negative_ratio: float,
    positive_consistency_beta: float = 0.0,
    positive_consistency_margin: float = 0.75,
    temperature: float = 0.1,
    embedding_negative_ratio: float = 0.5,
    embedding_supcon_weight: float = 0.5,
    embedding_tail_gamma_positive: float = 20.0,
    embedding_tail_gamma_negative: float = 40.0,
    embedding_tail_hard_negative_weight: float = 2.0,
    explicit_pairs: Sequence[EmbeddingPair] | None = None,
) -> tuple[torch.Tensor, int, int, int, int, int, int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    if explicit_pairs is None:
        selection = select_embedding_pairs(
            presence=presence,
            segment_ids=segment_ids,
            roots=roots,
            video_ids=video_ids,
            ocr_texts=ocr_texts,
            adjacent_segment_ids=adjacent_segment_ids,
            ocr_negative_enabled=ocr_negative_enabled,
            ocr_negative_max_similarity=ocr_negative_max_similarity,
            ocr_negative_ratio=ocr_negative_ratio,
        )
    else:
        selection = EmbeddingPairSelection(
            pairs=list(explicit_pairs),
            local_positive_pairs=sum(1 for pair in explicit_pairs if pair.same and pair.source == "local"),
            local_negative_pairs=sum(1 for pair in explicit_pairs if not pair.same and pair.source == "local"),
            ocr_negative_pairs=sum(1 for pair in explicit_pairs if not pair.same and pair.source == "ocr"),
            skipped_pairs=0,
        )
    if not selection.pairs:
        zero = embedding.sum() * 0.0
        supcon_loss = (
            zero
            if explicit_pairs is not None
            else supervised_contrastive_embedding_loss(
                embedding,
                presence,
                segment_ids,
                roots,
                temperature=temperature,
            )
        )
        return supcon_loss * embedding_supcon_weight, 0, 0, 0, 0, selection.skipped_pairs, 0, 0, 0, 0, zero, zero, supcon_loss
    left = torch.tensor([pair.i for pair in selection.pairs], dtype=torch.long, device=embedding.device)
    right = torch.tensor([pair.j for pair in selection.pairs], dtype=torch.long, device=embedding.device)
    targets = torch.tensor([1.0 if pair.same else 0.0 for pair in selection.pairs], dtype=embedding.dtype, device=embedding.device)
    similarities = (embedding[left] * embedding[right]).sum(dim=1)
    if explicit_pairs is None:
        balanced = balance_embedding_pairs(similarities, targets, negative_ratio=embedding_negative_ratio)
        selected_similarities = similarities[balanced.indices]
        selected_targets = targets[balanced.indices]
        candidate_positive_pairs = balanced.candidate_positive_pairs
        candidate_negative_pairs = balanced.candidate_negative_pairs
        selected_positive_pairs = balanced.selected_positive_pairs
        selected_negative_pairs = balanced.selected_negative_pairs
    else:
        selected_similarities = similarities
        selected_targets = targets
        candidate_positive_pairs = int((targets > 0.5).sum().item())
        candidate_negative_pairs = int((targets <= 0.5).sum().item())
        selected_positive_pairs = candidate_positive_pairs
        selected_negative_pairs = candidate_negative_pairs
    margin_loss = (
        embedding_margin_loss(
            selected_similarities,
            selected_targets,
            gamma_positive=embedding_tail_gamma_positive,
            gamma_negative=embedding_tail_gamma_negative,
            hard_negative_weight=embedding_tail_hard_negative_weight,
        )
        * alpha
    )
    positive_similarities = selected_similarities[selected_targets > 0.5]
    if positive_similarities.numel() == 0 or positive_consistency_beta <= 0.0:
        consistency_loss = embedding.sum() * 0.0
    else:
        consistency_loss = F.relu(positive_consistency_margin - positive_similarities).pow(2).mean()
    supcon_loss = (
        embedding.sum() * 0.0
        if explicit_pairs is not None
        else supervised_contrastive_embedding_loss(
            embedding,
            presence,
            segment_ids,
            roots,
            temperature=temperature,
        )
    )
    embedding_loss = margin_loss + positive_consistency_beta * consistency_loss + embedding_supcon_weight * supcon_loss
    return (
        embedding_loss,
        int(selected_similarities.numel()),
        selection.local_positive_pairs,
        selection.local_negative_pairs,
        selection.ocr_negative_pairs,
        selection.skipped_pairs,
        candidate_positive_pairs,
        candidate_negative_pairs,
        selected_positive_pairs,
        selected_negative_pairs,
        margin_loss,
        consistency_loss,
        supcon_loss,
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
    ocr_texts: list[str],
    embedding_loss_weight: float,
    embedding_loss_alpha: float = 1.0,
    adjacent_segment_ids: list[set[str] | frozenset[str] | list[str] | tuple[str, ...]],
    embedding_ocr_negative_enabled: bool,
    embedding_ocr_negative_max_similarity: float,
    embedding_ocr_negative_ratio: float,
    embedding_positive_consistency_beta: float = 0.0,
    embedding_positive_consistency_margin: float = 0.75,
    embedding_temperature: float = 0.1,
    embedding_negative_ratio: float = 0.5,
    embedding_supcon_weight: float = 0.5,
    embedding_tail_gamma_positive: float = 20.0,
    embedding_tail_gamma_negative: float = 40.0,
    embedding_tail_hard_negative_weight: float = 2.0,
    presence_loss_enabled: bool = True,
    embedding_loss_enabled: bool = True,
    explicit_embedding_pairs: Sequence[EmbeddingPair] | None = None,
) -> RoiLossBreakdown:
    presence_loss = (
        F.binary_cross_entropy_with_logits(presence_logit, presence, weight=presence_loss_weights)
        if presence_loss_enabled
        else presence_logit.sum() * 0.0
    )
    if embedding_loss_enabled:
        (
            embedding_loss,
            embedding_pairs,
            embedding_local_positive_pairs,
            embedding_local_negative_pairs,
            embedding_ocr_negative_pairs,
            embedding_skipped_pairs,
            embedding_candidate_positive_pairs,
            embedding_candidate_negative_pairs,
            embedding_selected_positive_pairs,
            embedding_selected_negative_pairs,
            pair_margin_loss,
            positive_consistency_loss,
            supervised_contrastive_loss,
        ) = metric_embedding_loss(
            embedding,
            presence,
            segment_ids,
            roots=roots,
            video_ids=video_ids,
            ocr_texts=ocr_texts,
            alpha=embedding_loss_alpha,
            adjacent_segment_ids=adjacent_segment_ids,
            ocr_negative_enabled=embedding_ocr_negative_enabled,
            ocr_negative_max_similarity=embedding_ocr_negative_max_similarity,
            ocr_negative_ratio=embedding_ocr_negative_ratio,
            positive_consistency_beta=embedding_positive_consistency_beta,
            positive_consistency_margin=embedding_positive_consistency_margin,
            temperature=embedding_temperature,
            embedding_negative_ratio=embedding_negative_ratio,
            embedding_supcon_weight=embedding_supcon_weight,
            embedding_tail_gamma_positive=embedding_tail_gamma_positive,
            embedding_tail_gamma_negative=embedding_tail_gamma_negative,
            embedding_tail_hard_negative_weight=embedding_tail_hard_negative_weight,
            explicit_pairs=explicit_embedding_pairs,
        )
    else:
        embedding_loss = embedding.sum() * 0.0
        pair_margin_loss = embedding_loss
        positive_consistency_loss = embedding_loss
        supervised_contrastive_loss = embedding_loss
        embedding_pairs = 0
        embedding_local_positive_pairs = 0
        embedding_local_negative_pairs = 0
        embedding_ocr_negative_pairs = 0
        embedding_skipped_pairs = 0
        embedding_candidate_positive_pairs = 0
        embedding_candidate_negative_pairs = 0
        embedding_selected_positive_pairs = 0
        embedding_selected_negative_pairs = 0
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
        embedding_candidate_positive_pairs=embedding_candidate_positive_pairs,
        embedding_candidate_negative_pairs=embedding_candidate_negative_pairs,
        embedding_selected_positive_pairs=embedding_selected_positive_pairs,
        embedding_selected_negative_pairs=embedding_selected_negative_pairs,
        embedding_margin_loss=pair_margin_loss,
        positive_consistency_loss=positive_consistency_loss,
        supervised_contrastive_loss=supervised_contrastive_loss,
    )
