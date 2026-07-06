from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .roi_pairs import normalize_ocr_text, normalized_ocr_text_similarity_at_most, select_embedding_pairs


class _LocalPairIndex:
    def __init__(self) -> None:
        self.frames: list[int] = []
        self.entries: dict[int, list[tuple[int, str, torch.Tensor]]] = {}
        self.next_order = 0

    def add(self, frame_index: int, segment_id: str, embedding: torch.Tensor) -> None:
        if frame_index not in self.entries:
            insort(self.frames, frame_index)
            self.entries[frame_index] = []
        self.entries[frame_index].append((self.next_order, segment_id, embedding))
        self.next_order += 1

    def within(self, frame_index: int, frame_window: int) -> list[tuple[str, torch.Tensor]]:
        start = bisect_left(self.frames, frame_index - frame_window)
        end = bisect_right(self.frames, frame_index + frame_window)
        matches = [entry for frame in self.frames[start:end] for entry in self.entries[frame]]
        matches.sort(key=lambda entry: entry[0])
        return [(segment_id, embedding) for _, segment_id, embedding in matches]


class EmbeddingPairMemory:
    def __init__(self, frame_window: int) -> None:
        self.frame_window = frame_window
        self._embeddings: dict[tuple[str, str], torch.Tensor] = {}
        self._local: dict[tuple[str, str], _LocalPairIndex] = {}
        self._negative_bank: dict[str, list[tuple[str, str | None, int | None, str, torch.Tensor]]] = {}

    def loss_and_update(
        self,
        embedding: torch.Tensor,
        presence: torch.Tensor,
        segment_ids: list[str],
        roots: list[str],
        video_ids: list[str | None],
        frame_indices: list[int | None],
        ocr_texts: list[str],
    ) -> tuple[torch.Tensor, int]:
        current_indices: list[int] = []
        previous_embeddings: list[torch.Tensor] = []
        targets: list[float] = []
        updates: list[tuple[tuple[str, str], torch.Tensor]] = []
        local_updates: list[tuple[tuple[str, str], int, str, torch.Tensor]] = []
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
            frame_index = frame_indices[index]
            normalized_ocr = normalize_ocr_text(ocr_texts[index])
            if video_id is not None and frame_index is not None:
                local_key = (roots[index], video_id)
                local_index = self._local.get(local_key)
                for previous_segment, previous_embedding in (
                    local_index.within(frame_index, self.frame_window) if local_index is not None else []
                ):
                    if previous_segment != segment_ids[index]:
                        current_indices.append(index)
                        previous_embeddings.append(previous_embedding)
                        targets.append(0.0)
                local_updates.append((local_key, frame_index, segment_ids[index], embedding[index].detach()))
            for previous_segment, previous_video, previous_frame, previous_ocr, previous_embedding in self._negative_bank.get(roots[index], []):
                if previous_segment == segment_ids[index]:
                    continue
                is_local = (
                    video_id is not None
                    and video_id == previous_video
                    and frame_index is not None
                    and previous_frame is not None
                    and abs(frame_index - previous_frame) <= self.frame_window
                )
                if not is_local:
                    if normalized_ocr_text_similarity_at_most(normalized_ocr, previous_ocr, 0.2):
                        current_indices.append(index)
                        previous_embeddings.append(previous_embedding)
                        targets.append(0.0)
        for key, value in updates:
            self._embeddings[key] = value
        for key, frame_index, segment_id, value in local_updates:
            local_index = self._local.get(key)
            if local_index is None:
                local_index = _LocalPairIndex()
                self._local[key] = local_index
            local_index.add(frame_index, segment_id, value)
        for index, is_positive in enumerate((presence.detach() > 0.5).tolist()):
            if is_positive:
                bank = self._negative_bank.setdefault(roots[index], [])
                bank.append(
                    (
                        segment_ids[index],
                        video_ids[index],
                        frame_indices[index],
                        normalize_ocr_text(ocr_texts[index]),
                        embedding[index].detach(),
                    )
                )
                del bank[:-512]
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
) -> torch.Tensor:
    positive_violation = F.relu(positive_margin - similarities)
    negative_violation = F.relu(similarities - negative_margin)
    positive_mask = targets > 0.5
    negative_mask = ~positive_mask
    hard_positive_mask = positive_mask & (positive_violation > 0.0)
    hard_negative_mask = negative_mask & (negative_violation > 0.0)
    positive_loss = (positive_violation.square() * hard_positive_mask).sum() / hard_positive_mask.sum().clamp_min(1)
    negative_loss = (negative_violation.square() * hard_negative_mask).sum() / hard_negative_mask.sum().clamp_min(1)
    return positive_loss + negative_loss


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
    embedding_negative_ratio: float = 0.5,
) -> tuple[torch.Tensor, int, int, int, int, int, int, int, int, int, torch.Tensor, torch.Tensor]:
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
        return zero, 0, 0, 0, 0, selection.skipped_pairs, 0, 0, 0, 0, zero, zero
    left = torch.tensor([pair.i for pair in selection.pairs], dtype=torch.long, device=embedding.device)
    right = torch.tensor([pair.j for pair in selection.pairs], dtype=torch.long, device=embedding.device)
    targets = torch.tensor([1.0 if pair.same else 0.0 for pair in selection.pairs], dtype=embedding.dtype, device=embedding.device)
    similarities = (embedding[left] * embedding[right]).sum(dim=1)
    balanced = balance_embedding_pairs(similarities, targets, negative_ratio=embedding_negative_ratio)
    selected_similarities = similarities[balanced.indices]
    selected_targets = targets[balanced.indices]
    margin_loss = embedding_margin_loss(selected_similarities, selected_targets) * alpha
    positive_similarities = selected_similarities[selected_targets > 0.5]
    if positive_similarities.numel() == 0 or positive_consistency_beta <= 0.0:
        consistency_loss = embedding.sum() * 0.0
    else:
        consistency_loss = F.relu(positive_consistency_margin - positive_similarities).pow(2).mean()
    embedding_loss = margin_loss + positive_consistency_beta * consistency_loss
    return (
        embedding_loss,
        int(balanced.indices.numel()),
        selection.local_positive_pairs,
        selection.local_negative_pairs,
        selection.ocr_negative_pairs,
        selection.skipped_pairs,
        balanced.candidate_positive_pairs,
        balanced.candidate_negative_pairs,
        balanced.selected_positive_pairs,
        balanced.selected_negative_pairs,
        margin_loss,
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
    embedding_negative_ratio: float = 0.5,
) -> RoiLossBreakdown:
    presence_loss = F.binary_cross_entropy_with_logits(presence_logit, presence, weight=presence_loss_weights)
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
        embedding_negative_ratio=embedding_negative_ratio,
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
        embedding_candidate_positive_pairs=embedding_candidate_positive_pairs,
        embedding_candidate_negative_pairs=embedding_candidate_negative_pairs,
        embedding_selected_positive_pairs=embedding_selected_positive_pairs,
        embedding_selected_negative_pairs=embedding_selected_negative_pairs,
        embedding_margin_loss=pair_margin_loss,
        positive_consistency_loss=positive_consistency_loss,
    )
