from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .labels import SubtitleInterval


@dataclass(frozen=True)
class SegmentSelectionConfig:
    score_threshold: float = 0.10
    nms_iou_threshold: float = 0.70
    minimum_duration_seconds: float = 0.20
    maximum_duration_seconds: float = 8.00
    peak_radius_frames: int = 2
    boundary_event_threshold: float = 0.20
    start_boundary_refinement_seconds: float = 0.60
    end_boundary_refinement_seconds: float = 1.20
    end_event_relative_threshold: float = 0.80
    boundary_event_peak_radius_frames: int = 2
    require_boundary_events: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.score_threshold) or not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score threshold must be in [0,1]")
        if (
            not math.isfinite(self.boundary_event_threshold)
            or not 0.0 <= self.boundary_event_threshold <= 1.0
        ):
            raise ValueError("boundary event threshold must be in [0,1]")
        if (
            not math.isfinite(self.start_boundary_refinement_seconds)
            or self.start_boundary_refinement_seconds < 0.0
        ):
            raise ValueError("start boundary refinement radius must be non-negative")
        if (
            not math.isfinite(self.end_boundary_refinement_seconds)
            or self.end_boundary_refinement_seconds < 0.0
        ):
            raise ValueError("end boundary refinement radius must be non-negative")
        if (
            not math.isfinite(self.end_event_relative_threshold)
            or not 0.0 < self.end_event_relative_threshold <= 1.0
        ):
            raise ValueError("end event relative threshold must be in (0,1]")
        if (
            not math.isfinite(self.nms_iou_threshold)
            or not 0.0 < self.nms_iou_threshold <= 1.0
        ):
            raise ValueError("NMS IoU threshold must be in (0,1]")
        if (
            not math.isfinite(self.minimum_duration_seconds)
            or self.minimum_duration_seconds < 0.0
        ):
            raise ValueError("minimum duration must be finite and non-negative")
        if (
            not math.isfinite(self.maximum_duration_seconds)
            or self.maximum_duration_seconds <= self.minimum_duration_seconds
        ):
            raise ValueError("maximum duration must be greater than minimum duration")
        if self.peak_radius_frames < 0:
            raise ValueError("peak radius must be non-negative")
        if self.boundary_event_peak_radius_frames < 0:
            raise ValueError("boundary event peak radius must be non-negative")
        if not isinstance(self.require_boundary_events, bool):
            raise ValueError("require_boundary_events must be boolean")
        if self.require_boundary_events and (
            self.start_boundary_refinement_seconds == 0.0
            or self.end_boundary_refinement_seconds == 0.0
        ):
            raise ValueError(
                "both boundary refinement radii must be positive when events are required"
            )

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentPrediction:
    start_seconds: float
    end_seconds: float
    confidence: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.start_seconds, self.end_seconds, self.confidence)
        ):
            raise ValueError("segment prediction values must be finite")
        if self.start_seconds < 0.0 or self.end_seconds <= self.start_seconds:
            raise ValueError("segment prediction must have start < end")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("segment confidence must be in [0,1]")

    def to_interval(self) -> SubtitleInterval:
        return SubtitleInterval(self.start_seconds, self.end_seconds)


def _local_peak_indices(
    scores: np.ndarray, *, threshold: float, radius: int
) -> np.ndarray:
    if scores.ndim != 1:
        raise ValueError("proposal scores must be one-dimensional")
    if len(scores) == 0:
        return np.empty((0,), dtype=np.int64)
    if radius == 0:
        return np.flatnonzero(scores >= threshold)
    peaks: list[int] = []
    index = 0
    while index < len(scores):
        if scores[index] < threshold:
            index += 1
            continue
        left = max(0, index - radius)
        right = min(len(scores), index + radius + 1)
        local_maximum = float(scores[left:right].max())
        if scores[index] < local_maximum:
            index += 1
            continue
        plateau_end = index + 1
        while plateau_end < len(scores) and scores[plateau_end] == scores[index]:
            plateau_end += 1
        peaks.append((index + plateau_end - 1) // 2)
        index = plateau_end
    return np.asarray(peaks, dtype=np.int64)


def select_segments(
    proposals: np.ndarray,
    timestamps: np.ndarray,
    *,
    config: SegmentSelectionConfig = SegmentSelectionConfig(),
) -> list[SegmentPrediction]:
    """Select already-regressed segments; no presence runs or boundary pairing are used."""
    if proposals.ndim != 2 or proposals.shape[1] != 5:
        raise ValueError("proposals must have shape [anchors,5]")
    if timestamps.shape != (len(proposals),):
        raise ValueError("timestamps must match proposal anchors")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0.0):
        raise ValueError("timestamps must be finite and sorted")
    if not np.isfinite(proposals).all():
        raise ValueError("segment proposals must be finite")
    if len(proposals) == 0:
        return []
    proposals, boundary_event_support = _refined_proposals_and_support(
        proposals,
        timestamps,
        event_threshold=config.boundary_event_threshold,
        start_radius_seconds=config.start_boundary_refinement_seconds,
        end_radius_seconds=config.end_boundary_refinement_seconds,
        end_relative_threshold=config.end_event_relative_threshold,
        event_peak_radius_frames=config.boundary_event_peak_radius_frames,
    )
    scores = proposals[:, 0]
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("proposal confidence must be in [0,1]")
    peak_scores = scores
    if config.require_boundary_events and (
        config.start_boundary_refinement_seconds > 0.0
        or config.end_boundary_refinement_seconds > 0.0
    ):
        peak_scores = np.where(boundary_event_support.all(axis=1), scores, -np.inf)
    peak_indices = _local_peak_indices(
        peak_scores,
        threshold=config.score_threshold,
        radius=config.peak_radius_frames,
    )
    candidate_rows: list[tuple[float, float, float]] = []
    for index in peak_indices:
        score, start_seconds, end_seconds = (
            float(value) for value in proposals[int(index), :3]
        )
        duration = end_seconds - start_seconds
        if (
            start_seconds < 0.0
            or duration < config.minimum_duration_seconds
            or duration > config.maximum_duration_seconds
        ):
            continue
        candidate_rows.append((score, start_seconds, end_seconds))
    if not candidate_rows:
        return []
    candidates = np.asarray(candidate_rows, dtype=np.float64)
    order = np.lexsort((candidates[:, 2], candidates[:, 1], -candidates[:, 0]))
    selected_indices: list[int] = []
    while order.size:
        selected_index = int(order[0])
        selected_indices.append(selected_index)
        remaining = order[1:]
        if not remaining.size:
            break
        selected_start = candidates[selected_index, 1]
        selected_end = candidates[selected_index, 2]
        intersections = np.maximum(
            0.0,
            np.minimum(selected_end, candidates[remaining, 2])
            - np.maximum(selected_start, candidates[remaining, 1]),
        )
        unions = np.maximum(selected_end, candidates[remaining, 2]) - np.minimum(
            selected_start, candidates[remaining, 1]
        )
        ious = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions > 0.0,
        )
        order = remaining[ious < config.nms_iou_threshold]
    selected = [
        SegmentPrediction(
            start_seconds=float(candidates[index, 1]),
            end_seconds=float(candidates[index, 2]),
            confidence=float(candidates[index, 0]),
        )
        for index in selected_indices
    ]
    return sorted(selected, key=lambda item: (item.start_seconds, item.end_seconds))


def refine_segment_boundaries(
    proposals: np.ndarray,
    timestamps: np.ndarray,
    *,
    event_threshold: float,
    start_radius_seconds: float,
    end_radius_seconds: float,
    end_relative_threshold: float,
    event_peak_radius_frames: int = 2,
) -> np.ndarray:
    """Align already-regressed segment edges to nearby model boundary events."""
    refined, _ = _refined_proposals_and_support(
        proposals,
        timestamps,
        event_threshold=event_threshold,
        start_radius_seconds=start_radius_seconds,
        end_radius_seconds=end_radius_seconds,
        end_relative_threshold=end_relative_threshold,
        event_peak_radius_frames=event_peak_radius_frames,
    )
    return refined


def _refined_proposals_and_support(
    proposals: np.ndarray,
    timestamps: np.ndarray,
    *,
    event_threshold: float,
    start_radius_seconds: float,
    end_radius_seconds: float,
    end_relative_threshold: float,
    event_peak_radius_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    refined = proposals.copy()
    support = np.zeros((len(proposals), 2), dtype=np.bool_)
    if not len(proposals):
        return refined, support
    start_event_scores = proposals[:, 3]
    if start_radius_seconds > 0.0:
        for index, boundary_time in enumerate(proposals[:, 1]):
            left = int(
                np.searchsorted(
                    timestamps,
                    boundary_time - start_radius_seconds,
                    side="left",
                )
            )
            right = int(
                np.searchsorted(
                    timestamps,
                    boundary_time + start_radius_seconds,
                    side="right",
                )
            )
            if right <= left:
                continue
            event_index = left + int(np.argmax(start_event_scores[left:right]))
            if start_event_scores[event_index] >= event_threshold:
                refined[index, 1] = timestamps[event_index]
                support[index, 0] = True

    end_event_scores = proposals[:, 4]
    end_event_peaks = _local_peak_indices(
        end_event_scores,
        threshold=event_threshold,
        radius=event_peak_radius_frames,
    )
    if end_radius_seconds <= 0.0 or not len(end_event_peaks):
        return refined, support
    end_peak_times = timestamps[end_event_peaks]
    for index, boundary_time in enumerate(proposals[:, 2]):
        candidate_mask = (
            (np.abs(end_peak_times - boundary_time) <= end_radius_seconds)
            & (end_peak_times >= timestamps[index])
        )
        candidates = end_event_peaks[candidate_mask]
        if not len(candidates):
            continue
        strongest_score = float(end_event_scores[candidates].max())
        minimum_score = max(
            event_threshold,
            end_relative_threshold * strongest_score,
        )
        eligible = candidates[end_event_scores[candidates] >= minimum_score]
        if len(eligible):
            refined[index, 2] = timestamps[int(eligible[0])]
            support[index, 1] = True
    return refined, support


def write_segment_predictions(path: Path, segments: list[SegmentPrediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "start_seconds",
                "end_seconds",
                "duration_seconds",
                "confidence",
                "label",
            ]
        )
        for segment in segments:
            writer.writerow(
                [
                    f"{segment.start_seconds:.6f}",
                    f"{segment.end_seconds:.6f}",
                    f"{segment.end_seconds - segment.start_seconds:.6f}",
                    f"{segment.confidence:.6f}",
                    "subtitle",
                ]
            )
