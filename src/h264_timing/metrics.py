from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .labels import SubtitleInterval


DEFAULT_FRAME_TOLERANCE_SECONDS = 1.0 / 30.0


@dataclass(frozen=True)
class IntervalMetricSample:
    """One video's intervals and timing contract for aggregate evaluation."""

    predicted: Sequence[SubtitleInterval]
    target: Sequence[SubtitleInterval]
    video_duration_seconds: float
    frame_tolerance_seconds: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.video_duration_seconds)
            or self.video_duration_seconds < 0.0
        ):
            raise ValueError("video_duration_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.frame_tolerance_seconds)
            or self.frame_tolerance_seconds < 0.0
        ):
            raise ValueError("frame_tolerance_seconds must be finite and non-negative")


@dataclass(frozen=True)
class _MatchScore:
    count: int = 0
    iou_sum: float = 0.0
    boundary_error_sum: float = 0.0

    def key(self) -> tuple[int, float, float]:
        return self.count, self.iou_sum, -self.boundary_error_sum


def interval_iou(left: SubtitleInterval, right: SubtitleInterval) -> float:
    intersection = max(
        0.0,
        min(left.end_seconds, right.end_seconds)
        - max(left.start_seconds, right.start_seconds),
    )
    union = max(left.end_seconds, right.end_seconds) - min(
        left.start_seconds, right.start_seconds
    )
    return intersection / union if union > 0.0 else 0.0


def _prf(true_positive: int, predicted: int, target: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else float(target == 0)
    recall = true_positive / target if target else float(predicted == 0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return precision, recall, f1


def _ordered_interval_matches(
    predicted: Sequence[SubtitleInterval],
    target: Sequence[SubtitleInterval],
    *,
    minimum_iou: float | None = None,
    boundary_tolerance_seconds: float | None = None,
) -> list[tuple[int, int, float]]:
    if (minimum_iou is None) == (boundary_tolerance_seconds is None):
        raise ValueError("exactly one interval matching constraint must be provided")

    predicted_order = sorted(
        range(len(predicted)),
        key=lambda index: (
            predicted[index].start_seconds,
            predicted[index].end_seconds,
            index,
        ),
    )
    target_order = sorted(
        range(len(target)),
        key=lambda index: (
            target[index].start_seconds,
            target[index].end_seconds,
            index,
        ),
    )
    predicted_count = len(predicted_order)
    target_count = len(target_order)
    # 1 skips a prediction, 2 skips a target, and 3 matches the current pair.
    actions = np.zeros((predicted_count, target_count), dtype=np.uint8)
    following_scores = [_MatchScore() for _ in range(target_count + 1)]

    for predicted_offset in range(predicted_count - 1, -1, -1):
        predicted_interval = predicted[predicted_order[predicted_offset]]
        current_scores = [_MatchScore() for _ in range(target_count + 1)]
        for target_offset in range(target_count - 1, -1, -1):
            target_interval = target[target_order[target_offset]]
            best_score = following_scores[target_offset]
            best_action = 1

            skip_target_score = current_scores[target_offset + 1]
            if skip_target_score.key() > best_score.key():
                best_score = skip_target_score
                best_action = 2

            iou = interval_iou(predicted_interval, target_interval)
            start_error = abs(
                predicted_interval.start_seconds - target_interval.start_seconds
            )
            end_error = abs(
                predicted_interval.end_seconds - target_interval.end_seconds
            )
            if minimum_iou is not None:
                accepted = iou >= minimum_iou
            elif boundary_tolerance_seconds is not None:
                accepted = (
                    start_error <= boundary_tolerance_seconds
                    and end_error <= boundary_tolerance_seconds
                )
            else:
                raise RuntimeError("interval matching constraint disappeared")
            if accepted:
                following = following_scores[target_offset + 1]
                match_score = _MatchScore(
                    count=following.count + 1,
                    iou_sum=following.iou_sum + iou,
                    boundary_error_sum=(
                        following.boundary_error_sum + start_error + end_error
                    ),
                )
                if match_score.key() >= best_score.key():
                    best_score = match_score
                    best_action = 3

            current_scores[target_offset] = best_score
            actions[predicted_offset, target_offset] = best_action
        following_scores = current_scores

    matches: list[tuple[int, int, float]] = []
    predicted_offset = 0
    target_offset = 0
    while predicted_offset < predicted_count and target_offset < target_count:
        action = int(actions[predicted_offset, target_offset])
        if action == 1:
            predicted_offset += 1
        elif action == 2:
            target_offset += 1
        else:
            predicted_index = predicted_order[predicted_offset]
            target_index = target_order[target_offset]
            matches.append(
                (
                    predicted_index,
                    target_index,
                    interval_iou(predicted[predicted_index], target[target_index]),
                )
            )
            predicted_offset += 1
            target_offset += 1
    return matches


def match_intervals(
    predicted: Sequence[SubtitleInterval],
    target: Sequence[SubtitleInterval],
    *,
    minimum_iou: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Return ordered one-to-one IoU matches in original-list coordinates."""
    if not math.isfinite(minimum_iou) or not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be finite and in [0,1]")
    return _ordered_interval_matches(predicted, target, minimum_iou=minimum_iou)


def match_intervals_by_boundaries(
    predicted: Sequence[SubtitleInterval],
    target: Sequence[SubtitleInterval],
    *,
    tolerance_seconds: float,
) -> list[tuple[int, int, float]]:
    """Match only pairs whose start and end errors are both within tolerance."""
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0.0:
        raise ValueError("tolerance_seconds must be finite and non-negative")
    return _ordered_interval_matches(
        predicted,
        target,
        boundary_tolerance_seconds=tolerance_seconds,
    )


def interval_metrics(
    predicted: Sequence[SubtitleInterval],
    target: Sequence[SubtitleInterval],
    *,
    video_duration_seconds: float,
    frame_tolerance_seconds: float = DEFAULT_FRAME_TOLERANCE_SECONDS,
) -> dict[str, float]:
    return aggregate_interval_metrics(
        [
            IntervalMetricSample(
                predicted=predicted,
                target=target,
                video_duration_seconds=video_duration_seconds,
                frame_tolerance_seconds=frame_tolerance_seconds,
            )
        ]
    )


def _error_summary(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    errors = np.asarray(values, dtype=np.float64)
    return (
        float(errors.mean()),
        float(np.percentile(errors, 95.0)),
        float(errors.max()),
    )


def _add_segment_metrics(
    metrics: dict[str, float],
    *,
    suffix: str,
    matched: int,
    predicted: int,
    target: int,
) -> None:
    precision, recall, f1 = _prf(matched, predicted, target)
    metrics[f"segment_precision_{suffix}"] = precision
    metrics[f"segment_recall_{suffix}"] = recall
    metrics[f"segment_f1_{suffix}"] = f1


def aggregate_interval_metrics(
    samples: Iterable[IntervalMetricSample],
) -> dict[str, float]:
    """Aggregate IoU and strict paired-boundary metrics across videos."""
    predicted_count = 0
    target_count = 0
    matched_count = 0
    duration_seconds = 0.0
    iou_sum = 0.0
    start_errors: list[float] = []
    end_errors: list[float] = []
    paired_counts = {
        "1frame": [0, 0, 0],
        "100ms": [0, 0, 0],
        "250ms": [0, 0, 0],
    }
    for sample in samples:
        predicted = sample.predicted
        target = sample.target
        matches = match_intervals(predicted, target)
        predicted_count += len(predicted)
        target_count += len(target)
        matched_count += len(matches)
        duration_seconds += sample.video_duration_seconds
        for predicted_index, target_index, iou in matches:
            iou_sum += iou
            start_errors.append(
                abs(
                    predicted[predicted_index].start_seconds
                    - target[target_index].start_seconds
                )
            )
            end_errors.append(
                abs(
                    predicted[predicted_index].end_seconds
                    - target[target_index].end_seconds
                )
            )

        for suffix, tolerance in (
            ("1frame", sample.frame_tolerance_seconds),
            ("100ms", 0.10),
            ("250ms", 0.25),
        ):
            strict_matches = match_intervals_by_boundaries(
                predicted,
                target,
                tolerance_seconds=tolerance,
            )
            counts = paired_counts[suffix]
            counts[0] += len(strict_matches)
            counts[1] += len(predicted)
            counts[2] += len(target)

    precision, recall, f1 = _prf(matched_count, predicted_count, target_count)
    false_count = max(0, predicted_count - matched_count)
    missed_count = max(0, target_count - matched_count)
    start_mae, start_p95, start_max = _error_summary(start_errors)
    end_mae, end_p95, end_max = _error_summary(end_errors)
    (
        strict_1frame_matched,
        strict_1frame_predicted,
        strict_1frame_target,
    ) = paired_counts["1frame"]
    metrics = {
        "interval_precision_iou50": precision,
        "interval_recall_iou50": recall,
        "interval_f1_iou50": f1,
        "matched_interval_mean_iou": iou_sum / matched_count if matched_count else 0.0,
        "false_intervals_per_minute": false_count
        / max(duration_seconds / 60.0, 1e-6),
        "matched_interval_count": float(matched_count),
        "missed_interval_count": float(missed_count),
        "false_interval_count": float(false_count),
        "missed_segment_1frame_count": float(
            max(0, strict_1frame_target - strict_1frame_matched)
        ),
        "false_segment_1frame_count": float(
            max(0, strict_1frame_predicted - strict_1frame_matched)
        ),
        "predicted_interval_count": float(predicted_count),
        "target_interval_count": float(target_count),
        "start_mae_seconds": start_mae,
        "start_error_p95_seconds": start_p95,
        "start_error_max_seconds": start_max,
        "end_mae_seconds": end_mae,
        "end_error_p95_seconds": end_p95,
        "end_error_max_seconds": end_max,
    }
    for suffix, counts in paired_counts.items():
        strict_matched, strict_predicted, strict_target = counts
        _add_segment_metrics(
            metrics,
            suffix=suffix,
            matched=strict_matched,
            predicted=strict_predicted,
            target=strict_target,
        )
    return metrics
