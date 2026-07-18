from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import LocalContrastResidual


class LocalAlignmentEmbeddingHead(nn.Module):
    """Keeps the subtitle ROI as an ordered sequence of normalized width tokens."""

    def __init__(
        self,
        feature_dim: int,
        *,
        width_tokens: int = 32,
        valid_token_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        if width_tokens <= 0:
            raise ValueError("width_tokens must be positive")
        if not 0.0 <= valid_token_threshold <= 1.0:
            raise ValueError("valid_token_threshold must be in [0, 1]")
        self.contrast = LocalContrastResidual()
        stacked_dim = feature_dim * 2
        self.width_tokens = width_tokens
        self.valid_token_threshold = valid_token_threshold
        self.attention = nn.Sequential(
            nn.Conv2d(stacked_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 1),
        )
        self.token_projection = nn.Conv1d(stacked_dim, feature_dim, 1, bias=False)
        self.position_embedding = nn.Parameter(torch.empty(1, feature_dim, width_tokens))
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 3, padding=1, groups=feature_dim, bias=False),
            nn.GroupNorm(1, feature_dim),
            nn.SiLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.token_norm = nn.LayerNorm(feature_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.cat([feature_map, self.contrast(feature_map)], dim=1)
        attention_logits = self.attention(stacked)
        weights = torch.sigmoid(attention_logits)
        height_denominator = weights.sum(dim=2).clamp_min(1e-4)
        width_sequence = (stacked * weights).sum(dim=2) / height_denominator
        coverage = weights.mean(dim=2)
        width_sequence = width_sequence * coverage
        width_sequence = F.adaptive_avg_pool1d(width_sequence, self.width_tokens)
        coverage = F.adaptive_avg_pool1d(coverage, self.width_tokens).squeeze(1)
        tokens = self.token_projection(width_sequence) + self.position_embedding
        tokens = tokens + self.sequence_encoder(tokens)
        tokens = self.token_norm(tokens.transpose(1, 2))
        tokens = F.normalize(tokens, p=2, dim=-1)
        valid = coverage >= self.valid_token_threshold
        return tokens * valid.unsqueeze(-1), attention_logits


def _directed_local_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    bandwidth: int,
    position_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_count = query.shape[1]
    positions = torch.arange(token_count, device=query.device)
    distance = (positions[:, None] - positions[None, :]).abs()
    allowed = distance <= bandwidth
    similarity = query @ key.transpose(1, 2)
    similarity = similarity - position_penalty * distance.to(similarity.dtype)
    query_valid = query.square().sum(dim=-1) > 0.0
    key_valid = key.square().sum(dim=-1) > 0.0
    candidates = allowed.unsqueeze(0) & key_valid.unsqueeze(1)
    floor = torch.finfo(similarity.dtype).min
    best = similarity.masked_fill(~candidates, floor).max(dim=-1).values
    matched = query_valid & candidates.any(dim=-1)
    best = torch.where(matched, best, torch.full_like(best, -1.0))
    return best, matched, query_valid


def _aggregate_local_scores(
    scores: torch.Tensor,
    matched: torch.Tensor,
    query_valid: torch.Tensor,
    *,
    bottom_ratio: float,
    mean_weight: float,
    bottom_weight: float,
    unmatched_penalty: float,
) -> torch.Tensor:
    results: list[torch.Tensor] = []
    for row_scores, row_matched, row_query_valid in zip(scores, matched, query_valid, strict=True):
        valid_scores = row_scores[row_matched]
        if valid_scores.numel() == 0:
            results.append(row_scores.sum() * 0.0 - 1.0 - unmatched_penalty)
            continue
        bottom_count = max(1, math.ceil(valid_scores.numel() * bottom_ratio))
        bottom = valid_scores.topk(bottom_count, largest=False).values.mean()
        query_count = int(row_query_valid.sum().item())
        unmatched_fraction = 1.0 - float(valid_scores.numel()) / max(1, query_count)
        results.append(
            mean_weight * valid_scores.mean()
            + bottom_weight * bottom
            - unmatched_penalty * unmatched_fraction
        )
    return torch.stack(results)


def local_alignment_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bandwidth: int = 3,
    position_penalty: float = 0.02,
    bottom_ratio: float = 0.20,
    mean_weight: float = 0.60,
    bottom_weight: float = 0.40,
    unmatched_penalty: float = 0.10,
) -> torch.Tensor:
    """Banded, bidirectional MaxSim with bottom-token aggregation."""

    if left.ndim != 3 or right.shape != left.shape:
        raise ValueError("local alignment inputs must both have shape [batch, tokens, dimensions]")
    if bandwidth < 0 or not 0.0 < bottom_ratio <= 1.0:
        raise ValueError("bandwidth must be non-negative and bottom_ratio must be in (0, 1]")
    left_scores, left_matched, left_valid = _directed_local_scores(
        left, right, bandwidth=bandwidth, position_penalty=position_penalty
    )
    right_scores, right_matched, right_valid = _directed_local_scores(
        right, left, bandwidth=bandwidth, position_penalty=position_penalty
    )
    left_similarity = _aggregate_local_scores(
        left_scores,
        left_matched,
        left_valid,
        bottom_ratio=bottom_ratio,
        mean_weight=mean_weight,
        bottom_weight=bottom_weight,
        unmatched_penalty=unmatched_penalty,
    )
    right_similarity = _aggregate_local_scores(
        right_scores,
        right_matched,
        right_valid,
        bottom_ratio=bottom_ratio,
        mean_weight=mean_weight,
        bottom_weight=bottom_weight,
        unmatched_penalty=unmatched_penalty,
    )
    return 0.5 * (left_similarity + right_similarity)


def pair_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim == 2:
        return (left * right).sum(dim=-1)
    return local_alignment_similarity(left, right)


def extreme_gap_loss(
    similarities: torch.Tensor,
    targets: torch.Tensor,
    *,
    tail_ratio: float = 0.10,
    temperature: float = 0.05,
    margin: float = 0.15,
) -> torch.Tensor:
    positives = similarities[targets > 0.5]
    negatives = similarities[targets <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return similarities.sum() * 0.0
    positive_count = max(1, math.ceil(positives.numel() * tail_ratio))
    negative_count = max(1, math.ceil(negatives.numel() * tail_ratio))
    positive_bottom = positives.topk(positive_count, largest=False).values
    negative_top = negatives.topk(negative_count, largest=True).values
    max_negative_soft = temperature * torch.logsumexp(negative_top / temperature, dim=0)
    min_positive_soft = -temperature * torch.logsumexp(-positive_bottom / temperature, dim=0)
    return F.softplus(max_negative_soft - min_positive_soft - margin)
