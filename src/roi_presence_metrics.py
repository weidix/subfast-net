from __future__ import annotations

import torch

from .roi_presence_loss import short_positive_mask


def _percentile(values: list[float], percentile: float) -> float:
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


def presence_metrics(logits: torch.Tensor, presence: torch.Tensor) -> dict[str, float]:
    scores = torch.sigmoid(logits)
    predicted = scores >= 0.5
    target = presence > 0.5
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    score_values = scores.detach().cpu().tolist()
    target_values = target.detach().cpu().tolist()
    positive_scores = [
        score for score, is_positive in zip(score_values, target_values, strict=True) if is_positive
    ]
    negative_scores = [
        score for score, is_positive in zip(score_values, target_values, strict=True) if not is_positive
    ]
    min_positive = min(positive_scores) if positive_scores else 0.0
    max_negative = max(negative_scores) if negative_scores else 0.0
    gap = min_positive - max_negative if positive_scores and negative_scores else 0.0
    best_threshold, best_f1 = _best_f1_threshold(score_values, target_values)
    return {
        "presence_accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "presence_tp": float(tp),
        "presence_fp": float(fp),
        "presence_fn": float(fn),
        "presence_tn": float(tn),
        "presence_positive_score": sum(positive_scores) / len(positive_scores) if positive_scores else 0.0,
        "presence_negative_score": sum(negative_scores) / len(negative_scores) if negative_scores else 0.0,
        "presence_roc_auc": _roc_auc(score_values, target_values),
        "presence_positive_p05": _percentile(positive_scores, 0.05),
        "presence_positive_p10": _percentile(positive_scores, 0.10),
        "presence_positive_p50": _percentile(positive_scores, 0.50),
        "presence_negative_p90": _percentile(negative_scores, 0.90),
        "presence_negative_p95": _percentile(negative_scores, 0.95),
        "presence_negative_p99": _percentile(negative_scores, 0.99),
        "presence_min_positive_score": min_positive,
        "presence_max_negative_score": max_negative,
        "presence_gap": gap,
        "presence_best_f1_threshold": best_threshold,
        "presence_best_f1": best_f1,
        "presence_zero_error_threshold_exists": float(gap > 0.0),
    }


def scoped_presence_metrics(
    logits: torch.Tensor,
    presence: torch.Tensor,
    ocr_texts: list[str],
) -> dict[str, float]:
    short_positive = short_positive_mask(presence, ocr_texts).cpu()
    positive = presence > 0.5
    negative = ~positive

    def metric_for(mask: torch.Tensor, prefix: str) -> dict[str, float]:
        if not bool(mask.any()):
            return {f"{prefix}_presence_f1": 0.0, f"{prefix}_presence_accuracy": 0.0}
        metrics = presence_metrics(logits[mask], presence[mask])
        return {
            f"{prefix}_presence_f1": metrics["presence_f1"],
            f"{prefix}_presence_accuracy": metrics["presence_accuracy"],
        }

    metrics = {"global_presence_f1": presence_metrics(logits, presence)["presence_f1"]}
    metrics.update(metric_for(negative | (positive & ~short_positive), "normal"))
    metrics.update(metric_for(negative | short_positive, "short"))
    return metrics


def checkpoint_score(metrics: dict[str, float]) -> float:
    return sum(
        float(metrics[name])
        for name in ("global_presence_f1", "normal_presence_f1", "short_presence_f1")
    ) / 3.0


def checkpoint_rank(metrics: dict[str, float]) -> tuple[float, float]:
    """Keep F1 as the primary target and use validation separation as its tie-break."""
    return checkpoint_score(metrics), float(metrics.get("presence_gap", 0.0))
