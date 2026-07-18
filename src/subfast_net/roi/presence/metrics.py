from __future__ import annotations

import math

import torch

from .loss import short_positive_mask


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


def _tail_mean(values: list[float], fraction: float, *, largest: bool) -> float:
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    ordered = sorted(values, reverse=largest)
    return sum(ordered[:count]) / count


def _expected_calibration_error(scores: torch.Tensor, target: torch.Tensor, bins: int = 10) -> float:
    error = scores.new_zeros(())
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (scores >= lower) & (scores < upper if index + 1 < bins else scores <= upper)
        if bool(selected.any()):
            confidence = scores[selected].mean()
            accuracy = target[selected].to(scores.dtype).mean()
            error = error + selected.to(scores.dtype).mean() * (confidence - accuracy).abs()
    return float(error)


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


def presence_metrics(
    logits: torch.Tensor,
    presence: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    scores = torch.sigmoid(logits)
    predicted = scores >= threshold
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
    positive_tail = _tail_mean(positive_scores, 0.01, largest=False)
    negative_tail = _tail_mean(negative_scores, 0.01, largest=True)
    best_threshold, best_f1 = _best_f1_threshold(score_values, target_values)
    clamped_scores = scores.clamp(1e-7, 1.0 - 1e-7)
    target_float = target.to(scores.dtype)
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
        "presence_robust_gap": _percentile(positive_scores, 0.05) - _percentile(negative_scores, 0.95),
        "presence_positive_lower_tail_mean_1pct": positive_tail,
        "presence_negative_upper_tail_mean_1pct": negative_tail,
        "presence_tail_gap_1pct": positive_tail - negative_tail,
        "presence_best_f1_threshold": best_threshold,
        "presence_best_f1": best_f1,
        "presence_zero_error_threshold_exists": float(gap > 0.0),
        "presence_decision_threshold": threshold,
        "presence_brier": float((scores - target_float).square().mean()),
        "presence_nll": float(
            -(target_float * clamped_scores.log() + (1.0 - target_float) * (1.0 - clamped_scores).log()).mean()
        ),
        "presence_ece": _expected_calibration_error(scores, target),
    }


def scoped_presence_metrics(
    logits: torch.Tensor,
    presence: torch.Tensor,
    ocr_texts: list[str],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    short_positive = short_positive_mask(presence, ocr_texts).cpu()
    positive = presence > 0.5
    negative = ~positive

    def metric_for(mask: torch.Tensor, prefix: str) -> dict[str, float]:
        if not bool(mask.any()):
            return {f"{prefix}_presence_f1": 0.0, f"{prefix}_presence_accuracy": 0.0}
        metrics = presence_metrics(logits[mask], presence[mask], threshold=threshold)
        return {
            f"{prefix}_presence_f1": metrics["presence_f1"],
            f"{prefix}_presence_accuracy": metrics["presence_accuracy"],
        }

    metrics = {"global_presence_f1": presence_metrics(logits, presence, threshold=threshold)["presence_f1"]}
    metrics.update(metric_for(negative | (positive & ~short_positive), "normal"))
    metrics.update(metric_for(negative | short_positive, "short"))
    return metrics


def segment_presence_metrics(
    logits: torch.Tensor,
    presence: torch.Tensor,
    segment_ids: list[str],
    roots: list[str],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, (root, segment_id) in enumerate(zip(roots, segment_ids, strict=True)):
        grouped.setdefault((root, segment_id), []).append(index)
    segment_logits: list[torch.Tensor] = []
    segment_targets: list[torch.Tensor] = []
    for indices in grouped.values():
        selected = torch.tensor(indices, device=logits.device)
        target = presence[selected]
        if bool((target > 0.5).any()):
            segment_logits.append(logits[selected].amin())
            segment_targets.append(target.new_tensor(1.0))
        else:
            segment_logits.append(logits[selected].amax())
            segment_targets.append(target.new_tensor(0.0))
    metrics = presence_metrics(
        torch.stack(segment_logits),
        torch.stack(segment_targets),
        threshold=threshold,
    )
    return {
        "segment_presence_f1": metrics["presence_f1"],
        "segment_presence_recall": metrics["presence_recall"],
        "segment_presence_accuracy": metrics["presence_accuracy"],
        "segment_presence_positive_lower_tail_mean_1pct": metrics[
            "presence_positive_lower_tail_mean_1pct"
        ],
        "segment_presence_tail_gap_1pct": metrics["presence_tail_gap_1pct"],
        "segment_count": float(len(grouped)),
    }


def region_localization_metrics(
    region_logits: torch.Tensor,
    region_targets: torch.Tensor,
    valid_masks: torch.Tensor | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    probability = torch.sigmoid(region_logits)
    if valid_masks is None:
        valid_masks = torch.ones_like(region_targets)
    records: list[dict[str, float]] = []
    for sample_probability, sample_target, sample_valid in zip(
        probability,
        region_targets,
        valid_masks,
        strict=True,
    ):
        valid = sample_valid > 0.5
        target = (sample_target > 0.25) & valid
        if not bool(target.any()):
            records.append({})
            continue
        predicted = (sample_probability >= 0.5) & valid
        intersection = float((predicted & target).sum())
        union = float((predicted | target).sum())
        predicted_count = float(predicted.sum())
        target_count = float(target.sum())
        inside = float(sample_probability[target].mean())
        outside_mask = valid & ~target
        outside = float(sample_probability[outside_mask].mean()) if bool(outside_mask.any()) else 0.0
        valid_probability = sample_probability.masked_fill(~valid, -1.0)
        maximum_index = int(valid_probability.flatten().argmax())
        pointing = float(target.flatten()[maximum_index])
        records.append(
            {
                "region_iou": intersection / union if union else 0.0,
                "region_dice": 2.0 * intersection / max(1.0, predicted_count + target_count),
                "region_inside_score": inside,
                "region_outside_score": outside,
                "region_contrast": inside - outside,
                "region_pointing": pointing,
            }
        )

    localized = [record for record in records if record]

    def mean(name: str) -> float:
        return sum(record[name] for record in localized) / len(localized) if localized else 0.0

    metrics = {
        "region_localizable_positive_count": float(len(localized)),
        "region_iou": mean("region_iou"),
        "region_dice": mean("region_dice"),
        "region_inside_score": mean("region_inside_score"),
        "region_outside_score": mean("region_outside_score"),
        "region_contrast": mean("region_contrast"),
        "region_pointing_accuracy": mean("region_pointing"),
    }
    return metrics, records


def text_distractor_metrics(
    logits: torch.Tensor,
    presence: torch.Tensor,
    candidate_masks: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[dict[str, float], torch.Tensor]:
    has_candidate = candidate_masks.flatten(start_dim=1).amax(dim=1) > 0.0
    distractor = (presence <= 0.5) & has_candidate
    count = int(distractor.sum())
    if not count:
        return {
            "text_distractor_count": 0.0,
            "text_distractor_fpr": 0.0,
            "text_distractor_score_p95": 0.0,
            "subtitle_specificity_evaluable": 0.0,
        }, distractor
    scores = torch.sigmoid(logits[distractor])
    return {
        "text_distractor_count": float(count),
        "text_distractor_fpr": float((scores >= threshold).to(torch.float32).mean()),
        "text_distractor_score_p95": _percentile(scores.detach().cpu().tolist(), 0.95),
        "subtitle_specificity_evaluable": 1.0,
    }, distractor


def checkpoint_score(metrics: dict[str, float]) -> float:
    classification = sum(
        float(metrics[name])
        for name in ("global_presence_f1", "normal_presence_f1", "short_presence_f1")
    ) / 3.0
    evidence = (
        float(metrics.get("region_pointing_accuracy", 0.0))
        + float(metrics.get("counterfactual_erased_flip_rate", 0.0))
    ) / 2.0
    return 0.75 * classification + 0.25 * evidence


def checkpoint_rank(metrics: dict[str, float]) -> tuple[float, ...]:
    """Prefer fixed-threshold quality, then localized and causally necessary evidence."""
    scoped_f1 = [
        float(metrics[name])
        for name in ("global_presence_f1", "normal_presence_f1", "short_presence_f1")
    ]
    scoped_f1.append(float(metrics.get("segment_presence_f1", scoped_f1[0])))
    hard_text_quality = (
        1.0 - float(metrics.get("text_distractor_fpr", 0.0))
        if float(metrics.get("subtitle_specificity_evaluable", 0.0))
        else 0.0
    )
    return (
        min(scoped_f1),
        sum(scoped_f1) / len(scoped_f1),
        hard_text_quality,
        float(metrics.get("region_pointing_accuracy", 0.0)),
        float(metrics.get("counterfactual_score_drop_lower_tail_1pct", 0.0)),
        float(metrics.get("presence_tail_gap_1pct", 0.0)),
        -float(metrics.get("presence_brier", 1.0)),
    )
