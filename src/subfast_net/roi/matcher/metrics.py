from __future__ import annotations

from ..embedding.metrics import similarity_percentile
from ..pairs import RoiPairSelection


def _roc_auc(scores: list[float], targets: list[bool]) -> float:
    positive_count = sum(targets)
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
        rank_sum += average_rank * sum(target for _, target in ranked[index:end])
        rank += end - index
        index = end
    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def _best_f1_threshold(scores: list[float], targets: list[bool]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    total_positive = sum(targets)
    if total_positive == 0:
        return max(scores) + 1e-6, 0.0
    best_threshold = max(scores) + 1e-6
    best_f1 = 0.0
    true_positive = 0
    false_positive = 0
    ordered = sorted(zip(scores, targets), key=lambda item: item[0], reverse=True)
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == threshold:
            if ordered[index][1]:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        false_negative = total_positive - true_positive
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


def pair_score_metrics(
    scores: list[float],
    selection: RoiPairSelection,
    *,
    threshold: float,
) -> dict[str, float]:
    if len(scores) != len(selection.pairs):
        raise ValueError("pair score count does not match pair selection")
    targets = [pair.same for pair in selection.pairs]
    positive_scores = [score for score, target in zip(scores, targets, strict=True) if target]
    negative_scores = [score for score, target in zip(scores, targets, strict=True) if not target]
    predictions = [score >= threshold for score in scores]
    false_positive = sum(prediction and not target for prediction, target in zip(predictions, targets, strict=True))
    false_negative = sum(not prediction and target for prediction, target in zip(predictions, targets, strict=True))
    correct = sum(prediction == target for prediction, target in zip(predictions, targets, strict=True))
    min_positive = min(positive_scores) if positive_scores else 0.0
    max_negative = max(negative_scores) if negative_scores else 0.0
    gap = min_positive - max_negative if positive_scores and negative_scores else 0.0
    best_threshold, best_f1 = _best_f1_threshold(scores, targets)
    return {
        "pair_accuracy": correct / len(scores) if scores else 0.0,
        "pair_same_score": sum(positive_scores) / len(positive_scores) if positive_scores else 0.0,
        "pair_diff_score": sum(negative_scores) / len(negative_scores) if negative_scores else 0.0,
        "pair_count": float(len(scores)),
        "pair_local_positive_count": float(selection.local_positive_pairs),
        "pair_local_negative_count": float(selection.local_negative_pairs),
        "pair_ocr_negative_count": float(selection.ocr_negative_pairs),
        "pair_skipped_count": float(selection.skipped_pairs),
        "pair_roc_auc": _roc_auc(scores, targets),
        "pair_positive_p05": similarity_percentile(positive_scores, 0.05),
        "pair_positive_p10": similarity_percentile(positive_scores, 0.10),
        "pair_positive_p50": similarity_percentile(positive_scores, 0.50),
        "pair_negative_p90": similarity_percentile(negative_scores, 0.90),
        "pair_negative_p95": similarity_percentile(negative_scores, 0.95),
        "pair_negative_p99": similarity_percentile(negative_scores, 0.99),
        "pair_false_positive_count": float(false_positive),
        "pair_false_negative_count": float(false_negative),
        "pair_min_positive_score": min_positive,
        "pair_max_negative_score": max_negative,
        "pair_gap": gap,
        "pair_best_f1_threshold": best_threshold,
        "pair_best_f1": best_f1,
        "pair_zero_error_threshold_exists": float(gap > 0.0),
    }
