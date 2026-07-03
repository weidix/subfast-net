from __future__ import annotations

import torch

from .roi_pairs import select_embedding_pairs


def presence_metrics(presence_logit: torch.Tensor, presence: torch.Tensor) -> dict[str, float]:
    pred = torch.sigmoid(presence_logit) >= 0.5
    target = presence > 0.5
    tp = int((pred & target).sum().detach().cpu())
    fp = int((pred & ~target).sum().detach().cpu())
    fn = int((~pred & target).sum().detach().cpu())
    tn = int((~pred & ~target).sum().detach().cpu())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "presence_accuracy": accuracy,
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "presence_tp": float(tp),
        "presence_fp": float(fp),
        "presence_fn": float(fn),
        "presence_tn": float(tn),
    }


def similarity_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _roc_auc(scores: list[float], targets: list[bool]) -> float:
    positive_count = sum(1 for target in targets if target)
    negative_count = len(targets) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.0
    ranked = sorted(zip(scores, targets), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        rank_sum += average_rank * sum(1 for _, target in ranked[index:end] if target)
        rank += end - index
        index = end
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def embedding_metrics(
    embedding: torch.Tensor,
    presence: torch.Tensor,
    segment_ids: list[str],
    *,
    roots: list[str],
    video_ids: list[str | None],
    frame_indices: list[int | None],
    ocr_texts: list[str],
    frame_window: int,
    ocr_negative_enabled: bool,
    ocr_negative_max_similarity: float,
    threshold: float,
) -> dict[str, float]:
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
        return {
            "embedding_pair_accuracy": 0.0,
            "embedding_same_similarity": 0.0,
            "embedding_diff_similarity": 0.0,
            "embedding_pairs": 0.0,
            "embedding_local_positive_pairs": 0.0,
            "embedding_local_negative_pairs": 0.0,
            "embedding_ocr_negative_pairs": 0.0,
            "embedding_skipped_pairs": float(selection.skipped_pairs),
            "embedding_roc_auc": 0.0,
            "embedding_positive_p10": 0.0,
            "embedding_positive_p05": 0.0,
            "embedding_positive_p25": 0.0,
            "embedding_positive_p50": 0.0,
            "embedding_negative_p90": 0.0,
            "embedding_negative_p95": 0.0,
            "embedding_negative_p99": 0.0,
            "embedding_false_positive_pairs": 0.0,
            "embedding_false_negative_pairs": 0.0,
        }
    same_values: list[float] = []
    diff_values: list[float] = []
    scores: list[float] = []
    targets: list[bool] = []
    correct = 0
    false_positive = 0
    false_negative = 0
    total = 0
    for pair in selection.pairs:
        score = float((embedding[pair.i] * embedding[pair.j]).sum().detach().cpu())
        prediction = score >= threshold
        if pair.same:
            same_values.append(score)
        else:
            diff_values.append(score)
        false_positive += int(prediction and not pair.same)
        false_negative += int((not prediction) and pair.same)
        scores.append(score)
        targets.append(pair.same)
        correct += int(prediction == pair.same)
        total += 1
    return {
        "embedding_pair_accuracy": correct / total if total else 0.0,
        "embedding_same_similarity": sum(same_values) / len(same_values) if same_values else 0.0,
        "embedding_diff_similarity": sum(diff_values) / len(diff_values) if diff_values else 0.0,
        "embedding_pairs": float(total),
        "embedding_local_positive_pairs": float(selection.local_positive_pairs),
        "embedding_local_negative_pairs": float(selection.local_negative_pairs),
        "embedding_ocr_negative_pairs": float(selection.ocr_negative_pairs),
        "embedding_skipped_pairs": float(selection.skipped_pairs),
        "embedding_roc_auc": _roc_auc(scores, targets),
        "embedding_positive_p05": similarity_percentile(same_values, 0.05),
        "embedding_positive_p10": similarity_percentile(same_values, 0.10),
        "embedding_positive_p25": similarity_percentile(same_values, 0.25),
        "embedding_positive_p50": similarity_percentile(same_values, 0.50),
        "embedding_negative_p90": similarity_percentile(diff_values, 0.90),
        "embedding_negative_p95": similarity_percentile(diff_values, 0.95),
        "embedding_negative_p99": similarity_percentile(diff_values, 0.99),
        "embedding_false_positive_pairs": float(false_positive),
        "embedding_false_negative_pairs": float(false_negative),
    }
