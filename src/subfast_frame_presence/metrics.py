from __future__ import annotations

import math

import torch


def _tail_mean(values: list[float], fraction: float, *, largest: bool) -> float:
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    ordered = sorted(values, reverse=largest)
    return sum(ordered[:count]) / count


def presence_metrics(
    logits: torch.Tensor,
    presence: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float]:
    scores = torch.sigmoid(logits.detach().cpu())
    target = presence.detach().cpu() > 0.5
    predicted = scores >= threshold
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive_scores = scores[target].tolist()
    negative_scores = scores[~target].tolist()
    minimum_positive = min(positive_scores) if positive_scores else 0.0
    maximum_negative = max(negative_scores) if negative_scores else 0.0
    gap = (
        minimum_positive - maximum_negative
        if positive_scores and negative_scores
        else 0.0
    )
    positive_tail = _tail_mean(positive_scores, 0.01, largest=False)
    negative_tail = _tail_mean(negative_scores, 0.01, largest=True)
    return {
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "presence_tp": float(tp),
        "presence_fp": float(fp),
        "presence_fn": float(fn),
        "presence_tn": float(tn),
        "presence_min_positive_score": minimum_positive,
        "presence_max_negative_score": maximum_negative,
        "presence_gap": gap,
        "presence_positive_lower_tail_mean_1pct": positive_tail,
        "presence_negative_upper_tail_mean_1pct": negative_tail,
        "presence_tail_gap_1pct": positive_tail - negative_tail,
        "presence_decision_threshold": threshold,
    }


def _binary_bounds(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    rows, columns = torch.nonzero(mask, as_tuple=True)
    if not rows.numel():
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def _box_area(bounds: tuple[int, int, int, int]) -> int:
    return max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])


def _box_intersection(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )


def region_metrics(
    region_logits: torch.Tensor,
    region_targets: torch.Tensor,
    presence: torch.Tensor,
    *,
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    probabilities = torch.sigmoid(region_logits.detach().cpu())
    targets = region_targets.detach().cpu() > 0.5
    positive = presence.detach().cpu() > 0.5
    records: list[dict[str, float]] = []
    positive_records: list[dict[str, float]] = []
    negative_activation: list[float] = []
    for probability, target, is_positive in zip(
        probabilities,
        targets,
        positive.tolist(),
        strict=True,
    ):
        probability = probability.squeeze(0)
        target = target.squeeze(0)
        predicted = probability >= threshold
        if not is_positive:
            activation = float(predicted.to(torch.float32).mean())
            negative_activation.append(activation)
            records.append(
                {
                    "region_active_cells": float(predicted.sum()),
                    "region_full_frame_activation_ratio": activation,
                }
            )
            continue
        target_count = int(target.sum())
        predicted_count = int(predicted.sum())
        intersection = int((predicted & target).sum())
        overflow = int((predicted & ~target).sum())
        missed = target_count - intersection
        union = target_count + overflow
        target_bounds = _binary_bounds(target)
        predicted_bounds = _binary_bounds(predicted)
        if target_bounds is None:
            raise ValueError("positive validation sample has an empty region target")
        containment = 0.0
        bbox_overflow_ratio = float("inf")
        bbox_area_ratio = float("inf")
        bbox_full_frame_ratio = 1.0
        if predicted_bounds is not None:
            containment = float(
                predicted_bounds[0] <= target_bounds[0]
                and predicted_bounds[1] <= target_bounds[1]
                and predicted_bounds[2] >= target_bounds[2]
                and predicted_bounds[3] >= target_bounds[3]
            )
            target_box_area = max(1, _box_area(target_bounds))
            predicted_box_area = _box_area(predicted_bounds)
            box_intersection = _box_intersection(predicted_bounds, target_bounds)
            bbox_overflow_ratio = (predicted_box_area - box_intersection) / target_box_area
            bbox_area_ratio = predicted_box_area / target_box_area
            bbox_full_frame_ratio = predicted_box_area / target.numel()
        record = {
            "region_target_cells": float(target_count),
            "region_active_cells": float(predicted_count),
            "region_target_recall": intersection / max(1, target_count),
            "region_miss_ratio": missed / max(1, target_count),
            "region_overflow_ratio": overflow / max(1, target_count),
            "region_heatmap_area_ratio": predicted_count / max(1, target_count),
            "region_iou": intersection / max(1, union),
            "region_bbox_containment": containment,
            "region_bbox_overflow_ratio": bbox_overflow_ratio,
            "region_bbox_area_ratio": bbox_area_ratio,
            "region_bbox_full_frame_ratio": bbox_full_frame_ratio,
            "region_area_limit_pass": float(predicted_count <= target_count),
            "region_extra_area_limit_pass": float(overflow <= target_count),
        }
        records.append(record)
        positive_records.append(record)

    def mean(name: str) -> float:
        return (
            sum(record[name] for record in positive_records) / len(positive_records)
            if positive_records
            else 0.0
        )

    def maximum(name: str) -> float:
        return max((record[name] for record in positive_records), default=0.0)

    def minimum(name: str) -> float:
        return min((record[name] for record in positive_records), default=0.0)

    return {
        "region_positive_count": float(len(positive_records)),
        "region_target_recall": mean("region_target_recall"),
        "region_target_recall_min": minimum("region_target_recall"),
        "region_iou": mean("region_iou"),
        "region_overflow_ratio": mean("region_overflow_ratio"),
        "region_overflow_ratio_max": maximum("region_overflow_ratio"),
        "region_heatmap_area_ratio": mean("region_heatmap_area_ratio"),
        "region_heatmap_area_ratio_max": maximum("region_heatmap_area_ratio"),
        "region_bbox_containment_rate": mean("region_bbox_containment"),
        "region_bbox_overflow_ratio": mean("region_bbox_overflow_ratio"),
        "region_bbox_overflow_ratio_max": maximum("region_bbox_overflow_ratio"),
        "region_bbox_full_frame_ratio": mean("region_bbox_full_frame_ratio"),
        "region_bbox_full_frame_ratio_max": maximum("region_bbox_full_frame_ratio"),
        "region_area_limit_pass_rate": mean("region_area_limit_pass"),
        "region_extra_area_limit_pass_rate": mean("region_extra_area_limit_pass"),
        "negative_region_activation_ratio": (
            sum(negative_activation) / len(negative_activation)
            if negative_activation
            else 0.0
        ),
    }, records


def checkpoint_rank(metrics: dict[str, float]) -> tuple[float, ...]:
    area_pass = float(metrics["region_area_limit_pass_rate"])
    containment = float(metrics["region_bbox_containment_rate"])
    maximum_overflow = float(metrics["region_overflow_ratio_max"])
    return (
        float(metrics["presence_f1"] == 1.0 and metrics["presence_recall"] == 1.0),
        round(float(metrics["presence_f1"]), 6),
        round(float(metrics["presence_recall"]), 6),
        float(metrics["presence_gap"] >= 0.8),
        min(round(float(metrics["presence_gap"]), 6), 0.8),
        float(containment == 1.0 and area_pass == 1.0 and maximum_overflow <= 1.0),
        min(containment, area_pass),
        containment,
        area_pass,
        -maximum_overflow,
        -float(metrics["region_bbox_full_frame_ratio_max"]),
        -float(metrics["region_overflow_ratio"]),
        -float(metrics["val_loss"]),
    )
