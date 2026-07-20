from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from subtitle_timing_core.postprocess import SegmentPrediction


def _validate_probability(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be in [0,1]")


def _validate_non_negative(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class StreamingDecoderConfig:
    """Thresholds for causal subtitle-segment decoding."""

    score_threshold: float = 0.10
    presence_on_threshold: float = 0.50
    presence_off_threshold: float = 0.35
    boundary_event_threshold: float = 0.50
    minimum_duration_seconds: float = 0.20
    maximum_duration_seconds: float = 8.00
    confirmation_samples: int = 2
    anchor_score_threshold: float = 0.50
    anchor_nms_iou_threshold: float = 0.70
    anchor_peak_radius_frames: int = 1
    anchor_end_event_threshold: float = 0.0
    minimum_anchor_gap_seconds: float = 0.0
    anchor_start_event_threshold: float = 0.0
    anchor_start_refinement_seconds: float = 0.0
    anchor_end_refinement_seconds: float = 0.0
    anchor_pair_start_events: bool = False
    # The event-pairing decoder is opt-in so checkpoints written before the
    # causal peak/recovery state was introduced retain their old behavior.
    causal_event_pairing: bool = False
    start_event_threshold: float = 0.86
    split_start_event_threshold: float = 0.0
    end_event_threshold: float = 0.20
    event_confirmation_samples: int = 5
    event_recovery_threshold: float = 0.60
    event_recovery_samples: int = 3
    strong_end_event_threshold: float = 0.50
    weak_end_presence_threshold: float = 1.0
    preserve_first_strong_end_candidate: bool = False
    preserve_strong_end_minimum_duration_seconds: float = 0.0
    track_confirmation_score_threshold: float = 0.0
    track_confirmation_max_pending_segments: int = 0
    minimum_start_gap_seconds: float = 0.30
    start_confirmation_presence_threshold: float = 0.0
    start_confirmation_samples: int = 0
    start_confirmation_window_samples: int = 0
    end_refinement_frames: int = 1
    end_refinement_event_threshold: float = 0.50

    def __post_init__(self) -> None:
        _validate_probability("score_threshold", self.score_threshold)
        _validate_probability("presence_on_threshold", self.presence_on_threshold)
        _validate_probability("presence_off_threshold", self.presence_off_threshold)
        _validate_probability("boundary_event_threshold", self.boundary_event_threshold)
        _validate_probability("anchor_score_threshold", self.anchor_score_threshold)
        _validate_probability(
            "anchor_nms_iou_threshold", self.anchor_nms_iou_threshold
        )
        _validate_probability(
            "anchor_end_event_threshold", self.anchor_end_event_threshold
        )
        if self.presence_off_threshold >= self.presence_on_threshold:
            raise ValueError(
                "presence_off_threshold must be less than presence_on_threshold"
            )
        _validate_non_negative(
            "minimum_duration_seconds", self.minimum_duration_seconds
        )
        _validate_non_negative(
            "minimum_anchor_gap_seconds", self.minimum_anchor_gap_seconds
        )
        _validate_probability(
            "anchor_start_event_threshold", self.anchor_start_event_threshold
        )
        _validate_probability("start_event_threshold", self.start_event_threshold)
        _validate_probability(
            "split_start_event_threshold", self.split_start_event_threshold
        )
        _validate_probability("end_event_threshold", self.end_event_threshold)
        _validate_probability(
            "event_recovery_threshold", self.event_recovery_threshold
        )
        _validate_probability(
            "strong_end_event_threshold", self.strong_end_event_threshold
        )
        _validate_probability(
            "weak_end_presence_threshold", self.weak_end_presence_threshold
        )
        if not isinstance(self.preserve_first_strong_end_candidate, bool):
            raise ValueError(
                "preserve_first_strong_end_candidate must be boolean"
            )
        _validate_non_negative(
            "preserve_strong_end_minimum_duration_seconds",
            self.preserve_strong_end_minimum_duration_seconds,
        )
        _validate_probability(
            "track_confirmation_score_threshold",
            self.track_confirmation_score_threshold,
        )
        _validate_probability(
            "end_refinement_event_threshold", self.end_refinement_event_threshold
        )
        _validate_non_negative(
            "anchor_start_refinement_seconds",
            self.anchor_start_refinement_seconds,
        )
        _validate_non_negative(
            "anchor_end_refinement_seconds", self.anchor_end_refinement_seconds
        )
        _validate_non_negative(
            "minimum_start_gap_seconds", self.minimum_start_gap_seconds
        )
        _validate_probability(
            "start_confirmation_presence_threshold",
            self.start_confirmation_presence_threshold,
        )
        if (
            isinstance(self.maximum_duration_seconds, bool)
            or not isinstance(self.maximum_duration_seconds, Real)
            or not math.isfinite(self.maximum_duration_seconds)
            or self.maximum_duration_seconds <= self.minimum_duration_seconds
        ):
            raise ValueError(
                "maximum_duration_seconds must be finite and greater than minimum_duration_seconds"
            )
        if (
            isinstance(self.confirmation_samples, bool)
            or not isinstance(self.confirmation_samples, Integral)
            or self.confirmation_samples <= 0
        ):
            raise ValueError("confirmation_samples must be a positive integer")
        if (
            isinstance(self.event_confirmation_samples, bool)
            or not isinstance(self.event_confirmation_samples, Integral)
            or self.event_confirmation_samples <= 0
        ):
            raise ValueError("event_confirmation_samples must be a positive integer")
        if (
            isinstance(self.event_recovery_samples, bool)
            or not isinstance(self.event_recovery_samples, Integral)
            or self.event_recovery_samples <= 0
        ):
            raise ValueError("event_recovery_samples must be a positive integer")
        for name, value in (
            ("start_confirmation_samples", self.start_confirmation_samples),
            (
                "start_confirmation_window_samples",
                self.start_confirmation_window_samples,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.track_confirmation_max_pending_segments, bool)
            or not isinstance(
                self.track_confirmation_max_pending_segments, Integral
            )
            or self.track_confirmation_max_pending_segments < 0
        ):
            raise ValueError(
                "track_confirmation_max_pending_segments must be a non-negative integer"
            )
        if (self.track_confirmation_score_threshold == 0.0) != (
            self.track_confirmation_max_pending_segments == 0
        ):
            raise ValueError(
                "track confirmation threshold and pending limit must both be disabled or positive"
            )
        if (self.start_confirmation_samples == 0) != (
            self.start_confirmation_window_samples == 0
        ):
            raise ValueError(
                "start confirmation samples and window must both be zero or positive"
            )
        if (
            self.start_confirmation_samples
            > self.start_confirmation_window_samples + 1
        ):
            raise ValueError(
                "start_confirmation_samples cannot exceed the peak plus confirmation window"
            )
        if (
            isinstance(self.anchor_peak_radius_frames, bool)
            or not isinstance(self.anchor_peak_radius_frames, Integral)
            or self.anchor_peak_radius_frames < 0
        ):
            raise ValueError("anchor_peak_radius_frames must be a non-negative integer")
        if (
            isinstance(self.end_refinement_frames, bool)
            or not isinstance(self.end_refinement_frames, Integral)
            or self.end_refinement_frames < 0
        ):
            raise ValueError("end_refinement_frames must be a non-negative integer")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class StreamingSegmentDecoder:
    """Convert causal per-sample probabilities into finalized subtitle segments."""

    def __init__(
        self, config: StreamingDecoderConfig = StreamingDecoderConfig()
    ) -> None:
        if not isinstance(config, StreamingDecoderConfig):
            raise TypeError("config must be a StreamingDecoderConfig")
        self.config = config
        self._closed = False
        self._last_timestamp: float | None = None
        self._last_duration = 0.0
        self._last_end_evidence = 0.0
        self._end_event_above_threshold = False

        self._active_start: float | None = None
        self._active_presence_sum = 0.0
        self._active_presence_count = 0
        self._active_start_evidence = 0.0

        self._on_start: float | None = None
        self._on_presence_sum = 0.0
        self._on_count = 0
        self._on_start_evidence = 0.0

        self._off_start: float | None = None
        self._off_presence_sum = 0.0
        self._off_count = 0
        self._off_end_evidence = 0.0

    def push(
        self,
        timestamp_seconds: float,
        duration_seconds: float,
        presence_probability: float,
        start_event_probability: float,
        end_event_probability: float,
    ) -> tuple[SegmentPrediction, ...]:
        """Consume one presentation-ordered sample and return newly finalized segments."""
        if self._closed:
            raise RuntimeError("cannot push samples after the decoder is closed")
        timestamp = self._validate_sample(
            timestamp_seconds,
            duration_seconds,
            presence_probability,
            start_event_probability,
            end_event_probability,
        )
        duration = float(duration_seconds)
        presence = float(presence_probability)
        start_event = float(start_event_probability)
        end_event = float(end_event_probability)
        self._last_timestamp = timestamp
        self._last_duration = duration
        self._last_end_evidence = end_event
        end_event_triggered = (
            end_event >= self.config.boundary_event_threshold
            and not self._end_event_above_threshold
        )
        self._end_event_above_threshold = (
            end_event >= self.config.boundary_event_threshold
        )

        emitted: list[SegmentPrediction] = []
        if self._active_start is not None:
            maximum_end = self._active_start + self.config.maximum_duration_seconds
            if timestamp >= maximum_end:
                prediction = self._finish_active(
                    maximum_end,
                    end_event if timestamp == maximum_end else 0.0,
                    include_pending_absence=True,
                )
                if prediction is not None:
                    emitted.append(prediction)
            elif end_event_triggered:
                prediction = self._finish_active(
                    timestamp,
                    end_event,
                    include_pending_absence=False,
                )
                if prediction is not None:
                    emitted.append(prediction)
            else:
                prediction = self._consume_active(timestamp, presence, end_event)
                if prediction is not None:
                    emitted.append(prediction)
                    return tuple(emitted)
                return tuple(emitted)

        # Processing the closing sample again permits an end and a new start at
        # the same timestamp without retaining the previous segment.
        self._consume_inactive(timestamp, presence, start_event)
        return tuple(emitted)

    def close(self) -> tuple[SegmentPrediction, ...]:
        """Flush a valid active tail and permanently close the decoder."""
        if self._closed:
            return ()
        emitted: tuple[SegmentPrediction, ...] = ()
        if self._active_start is not None and self._last_timestamp is not None:
            tail_end = min(
                self._last_timestamp + self._last_duration,
                self._active_start + self.config.maximum_duration_seconds,
            )
            prediction = self._finish_active(
                tail_end,
                self._last_end_evidence,
                include_pending_absence=True,
            )
            if prediction is not None:
                emitted = (prediction,)
        self._clear_state()
        self._closed = True
        return emitted

    def _validate_sample(
        self,
        timestamp_seconds: float,
        duration_seconds: float,
        presence_probability: float,
        start_event_probability: float,
        end_event_probability: float,
    ) -> float:
        _validate_non_negative("timestamp_seconds", timestamp_seconds)
        timestamp = float(timestamp_seconds)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("sample timestamps must be strictly increasing")
        _validate_non_negative("duration_seconds", duration_seconds)
        _validate_probability("presence_probability", presence_probability)
        _validate_probability("start_event_probability", start_event_probability)
        _validate_probability("end_event_probability", end_event_probability)
        return timestamp

    def _consume_inactive(
        self,
        timestamp: float,
        presence: float,
        start_event: float,
    ) -> None:
        if (
            start_event >= self.config.boundary_event_threshold
            and presence >= self.config.presence_on_threshold
        ):
            self._begin_active(timestamp, presence, 1, start_event)
            return
        if presence < self.config.presence_on_threshold:
            self._reset_pending_on()
            return
        if self._on_count == 0:
            self._on_start = timestamp
        self._on_presence_sum += presence
        self._on_count += 1
        self._on_start_evidence = max(self._on_start_evidence, start_event)
        if self._on_count >= self.config.confirmation_samples:
            if self._on_start is None:
                raise RuntimeError("pending presence start is missing")
            self._begin_active(
                self._on_start,
                self._on_presence_sum,
                self._on_count,
                self._on_start_evidence,
            )

    def _consume_active(
        self,
        timestamp: float,
        presence: float,
        end_event: float,
    ) -> SegmentPrediction | None:
        if presence <= self.config.presence_off_threshold:
            if self._off_count == 0:
                self._off_start = timestamp
            self._off_presence_sum += presence
            self._off_count += 1
            self._off_end_evidence = max(self._off_end_evidence, end_event)
            if self._off_count >= self.config.confirmation_samples:
                if self._off_start is None:
                    raise RuntimeError("pending absence start is missing")
                return self._finish_active(
                    self._off_start,
                    self._off_end_evidence,
                    include_pending_absence=False,
                )
            return None

        self._commit_pending_absence()
        self._active_presence_sum += presence
        self._active_presence_count += 1
        return None

    def _begin_active(
        self,
        start_seconds: float,
        presence_sum: float,
        presence_count: int,
        start_evidence: float,
    ) -> None:
        self._active_start = start_seconds
        self._active_presence_sum = presence_sum
        self._active_presence_count = presence_count
        self._active_start_evidence = start_evidence
        self._reset_pending_on()
        self._reset_pending_off()

    def _finish_active(
        self,
        end_seconds: float,
        end_evidence: float,
        *,
        include_pending_absence: bool,
    ) -> SegmentPrediction | None:
        if self._active_start is None:
            return None
        if include_pending_absence:
            self._commit_pending_absence()
        start_seconds = self._active_start
        duration = end_seconds - start_seconds
        prediction: SegmentPrediction | None = None
        if (
            self.config.minimum_duration_seconds
            <= duration
            <= self.config.maximum_duration_seconds
            and self._active_presence_count > 0
        ):
            mean_presence = self._active_presence_sum / self._active_presence_count
            confidence = (
                mean_presence + self._active_start_evidence + end_evidence
            ) / 3.0
            confidence = min(1.0, max(0.0, confidence))
            if confidence >= self.config.score_threshold:
                prediction = SegmentPrediction(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    confidence=confidence,
                )
        self._active_start = None
        self._active_presence_sum = 0.0
        self._active_presence_count = 0
        self._active_start_evidence = 0.0
        self._reset_pending_off()
        return prediction

    def _commit_pending_absence(self) -> None:
        self._active_presence_sum += self._off_presence_sum
        self._active_presence_count += self._off_count
        self._reset_pending_off()

    def _reset_pending_on(self) -> None:
        self._on_start = None
        self._on_presence_sum = 0.0
        self._on_count = 0
        self._on_start_evidence = 0.0

    def _reset_pending_off(self) -> None:
        self._off_start = None
        self._off_presence_sum = 0.0
        self._off_count = 0
        self._off_end_evidence = 0.0

    def _clear_state(self) -> None:
        self._last_timestamp = None
        self._last_duration = 0.0
        self._last_end_evidence = 0.0
        self._end_event_above_threshold = False
        self._active_start = None
        self._active_presence_sum = 0.0
        self._active_presence_count = 0
        self._active_start_evidence = 0.0
        self._reset_pending_on()
        self._reset_pending_off()


@dataclass(frozen=True)
class _CausalEventPeak:
    index: int
    timestamp: float
    probability: float
    presence: float
    previous_timestamp: float | None = None


class _CausalEventPeakTracker:
    """Hold a thresholded event run until its causal peak is known."""

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        self._candidate: _CausalEventPeak | None = None
        self._last_timestamp: float | None = None

    def push(
        self,
        index: int,
        timestamp: float,
        probability: float,
        presence: float,
    ) -> _CausalEventPeak | None:
        if probability >= self.threshold:
            candidate = _CausalEventPeak(
                index=index,
                timestamp=timestamp,
                probability=probability,
                presence=presence,
                previous_timestamp=self._last_timestamp,
            )
            if self._candidate is None or probability > self._candidate.probability:
                self._candidate = candidate
            self._last_timestamp = timestamp
            return None
        peak = self._candidate
        self._candidate = None
        self._last_timestamp = timestamp
        return peak

    def close(self) -> _CausalEventPeak | None:
        peak = self._candidate
        self._candidate = None
        self._last_timestamp = None
        return peak


class StreamingEventPairDecoder:
    """Decode causal start/end event peaks with delayed confirmation.

    A peak is emitted only after its following sample arrives, so the selected
    boundary remains causal while avoiding repeated threshold crossings. End
    events are held briefly: a sustained presence recovery cancels a transient
    end, while a subsequent start can still use the held end as a split point.
    """

    def __init__(self, config: StreamingDecoderConfig) -> None:
        if not isinstance(config, StreamingDecoderConfig):
            raise TypeError("config must be a StreamingDecoderConfig")
        self.config = config
        self._closed = False
        self._sample_index = 0
        self._last_timestamp: float | None = None
        self._last_duration = 0.0
        history_size = max(
            8,
            config.end_refinement_frames
            + config.event_confirmation_samples
            + config.event_recovery_samples
            + 4,
            config.start_confirmation_window_samples + 4,
        )
        self._recent_timestamps: deque[tuple[int, float]] = deque(
            maxlen=history_size
        )
        self._start_peaks = _CausalEventPeakTracker(config.start_event_threshold)
        self._end_peaks = _CausalEventPeakTracker(config.end_event_threshold)

        self._active_start: _CausalEventPeak | None = None
        self._active_presence_sum = 0.0
        self._active_presence_count = 0
        self._active_start_evidence = 0.0

        self._pending_start: _CausalEventPeak | None = None
        self._pending_start_age = 0
        self._pending_start_presence_hits = 0
        self._pending_start_presence_sum = 0.0
        self._pending_start_presence_count = 0

        self._pending_end: _CausalEventPeak | None = None
        self._last_end_candidate: _CausalEventPeak | None = None
        self._first_strong_end_candidate: _CausalEventPeak | None = None
        self._pending_age = 0
        self._recovery_count = 0
        self._last_strong_end_timestamp = -math.inf
        self._track_confirmed = config.track_confirmation_score_threshold == 0.0
        self._pending_track_predictions: deque[SegmentPrediction] = deque()

    def push(
        self,
        timestamp_seconds: float,
        duration_seconds: float,
        presence_probability: float,
        start_event_probability: float,
        end_event_probability: float,
    ) -> tuple[SegmentPrediction, ...]:
        if self._closed:
            raise RuntimeError("cannot push samples after the decoder is closed")
        timestamp = self._validate_sample(
            timestamp_seconds,
            duration_seconds,
            presence_probability,
            start_event_probability,
            end_event_probability,
        )
        duration = float(duration_seconds)
        presence = float(presence_probability)
        index = self._sample_index
        self._sample_index += 1
        self._last_timestamp = timestamp
        self._last_duration = duration
        self._recent_timestamps.append((index, timestamp))

        pending_before = self._pending_end is not None
        end_peak = self._end_peaks.push(
            index, timestamp, float(end_event_probability), presence
        )
        start_peak = self._start_peaks.push(
            index, timestamp, float(start_event_probability), presence
        )
        emitted: list[SegmentPrediction] = []

        # End evidence is processed first so a close/start transition at the
        # same observed sample can split the old segment cleanly.
        self._consume_end_peak(end_peak)
        self._consume_start_peak(start_peak, emitted)

        activated_from_pending = False
        if self._active_start is None and self._pending_start is not None:
            activated_from_pending = self._advance_pending_start(presence)

        if self._active_start is not None:
            if pending_before and self._pending_end is not None:
                self._pending_age += 1
                if presence >= self.config.event_recovery_threshold:
                    self._recovery_count += 1
                else:
                    self._recovery_count = 0
                if self._recovery_count >= self.config.event_recovery_samples:
                    self._last_end_candidate = self._pending_end
                    self._pending_end = None
                    self._pending_age = 0
                    self._recovery_count = 0
                elif (
                    self._pending_age >= self.config.event_confirmation_samples
                ):
                    prediction = self._finish_active(self._pending_end)
                    if prediction is not None:
                        self._emit_prediction(prediction, emitted)
            elif self._pending_end is None and not activated_from_pending:
                self._active_presence_sum += presence
                self._active_presence_count += 1

            if self._active_start is not None:
                maximum_end = (
                    self._active_start.timestamp
                    + self.config.maximum_duration_seconds
                )
                if timestamp >= maximum_end:
                    prediction = self._finish_active_at(maximum_end, 0.0)
                    if prediction is not None:
                        self._emit_prediction(prediction, emitted)
        return tuple(emitted)

    def close(self) -> tuple[SegmentPrediction, ...]:
        if self._closed:
            return ()
        emitted: list[SegmentPrediction] = []
        # Flush any event run that was still above threshold at end of input.
        self._consume_end_peak(self._end_peaks.close())
        self._consume_start_peak(self._start_peaks.close(), emitted)
        if self._active_start is not None and self._last_timestamp is not None:
            if self._pending_end is not None:
                prediction = self._finish_active(self._pending_end)
            elif self._last_end_candidate is not None:
                prediction = self._finish_active(self._last_end_candidate)
            else:
                tail_end = min(
                    self._last_timestamp + self._last_duration,
                    self._active_start.timestamp
                    + self.config.maximum_duration_seconds,
                )
                prediction = self._finish_active_at(tail_end, 0.0)
            if prediction is not None:
                self._emit_prediction(prediction, emitted)
        self._clear_state()
        self._closed = True
        return tuple(emitted)

    def _validate_sample(
        self,
        timestamp_seconds: float,
        duration_seconds: float,
        presence_probability: float,
        start_event_probability: float,
        end_event_probability: float,
    ) -> float:
        _validate_non_negative("timestamp_seconds", timestamp_seconds)
        timestamp = float(timestamp_seconds)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("sample timestamps must be strictly increasing")
        _validate_non_negative("duration_seconds", duration_seconds)
        _validate_probability("presence_probability", presence_probability)
        _validate_probability("start_event_probability", start_event_probability)
        _validate_probability("end_event_probability", end_event_probability)
        return timestamp

    def _consume_end_peak(self, peak: _CausalEventPeak | None) -> None:
        if peak is None:
            return
        if peak.probability >= self.config.strong_end_event_threshold:
            self._last_strong_end_timestamp = peak.timestamp
        if self._active_start is None or peak.timestamp <= self._active_start.timestamp:
            return
        if (
            peak.probability < self.config.strong_end_event_threshold
            and peak.presence > self.config.weak_end_presence_threshold
        ):
            return
        if (
            self.config.preserve_first_strong_end_candidate
            and peak.probability >= self.config.strong_end_event_threshold
            and self._first_strong_end_candidate is None
            and peak.timestamp - self._active_start.timestamp
            >= self.config.preserve_strong_end_minimum_duration_seconds
        ):
            self._first_strong_end_candidate = peak
        self._pending_end = peak
        self._pending_age = 0
        self._recovery_count = 0

    def _consume_start_peak(
        self,
        peak: _CausalEventPeak | None,
        emitted: list[SegmentPrediction],
    ) -> None:
        if peak is None:
            return
        if (
            peak.timestamp - self._last_strong_end_timestamp
            < self.config.minimum_start_gap_seconds
        ):
            return
        if self._active_start is None:
            if peak.presence >= self.config.presence_on_threshold:
                self._begin_active(peak)
            elif self.config.start_confirmation_window_samples > 0:
                self._queue_pending_start(peak)
            return
        split_start_threshold = (
            self.config.split_start_event_threshold
            if self.config.split_start_event_threshold > 0.0
            else self.config.start_event_threshold
        )
        if peak.probability < split_start_threshold:
            return
        if peak.presence < self.config.presence_on_threshold:
            return
        if peak.timestamp <= self._active_start.timestamp:
            return
        candidate = self._pending_end or self._last_end_candidate
        if candidate is None:
            previous_timestamp = self._timestamp_for_index(peak.index - 1)
            if previous_timestamp is None:
                return
            candidate = _CausalEventPeak(
                index=peak.index - 1,
                timestamp=previous_timestamp,
                probability=0.0,
                presence=0.0,
                previous_timestamp=self._timestamp_for_index(peak.index - 2),
            )
        prediction = self._finish_active(candidate)
        if prediction is not None:
            self._emit_prediction(prediction, emitted)
        self._begin_active(peak)

    def _emit_prediction(
        self,
        prediction: SegmentPrediction,
        emitted: list[SegmentPrediction],
    ) -> None:
        if self._track_confirmed:
            emitted.append(prediction)
            return
        if (
            prediction.confidence
            >= self.config.track_confirmation_score_threshold
        ):
            self._track_confirmed = True
            emitted.extend(self._pending_track_predictions)
            self._pending_track_predictions.clear()
            emitted.append(prediction)
            return
        limit = self.config.track_confirmation_max_pending_segments
        if len(self._pending_track_predictions) >= limit:
            self._pending_track_predictions.popleft()
        self._pending_track_predictions.append(prediction)

    def _queue_pending_start(self, peak: _CausalEventPeak) -> None:
        self._pending_start = peak
        self._pending_start_age = 0
        self._pending_start_presence_hits = int(
            peak.presence >= self.config.start_confirmation_presence_threshold
        )
        self._pending_start_presence_sum = peak.presence
        self._pending_start_presence_count = 1
        if (
            self._pending_start_presence_hits
            >= self.config.start_confirmation_samples
        ):
            self._activate_pending_start()

    def _advance_pending_start(self, presence: float) -> bool:
        if self._pending_start is None:
            return False
        self._pending_start_age += 1
        self._pending_start_presence_sum += presence
        self._pending_start_presence_count += 1
        if presence >= self.config.start_confirmation_presence_threshold:
            self._pending_start_presence_hits += 1
        if (
            self._pending_start_presence_hits
            >= self.config.start_confirmation_samples
        ):
            self._activate_pending_start()
            return True
        if (
            self._pending_start_age
            >= self.config.start_confirmation_window_samples
        ):
            self._clear_pending_start()
        return False

    def _activate_pending_start(self) -> None:
        peak = self._pending_start
        if peak is None:
            return
        presence_sum = self._pending_start_presence_sum
        presence_count = self._pending_start_presence_count
        self._clear_pending_start()
        self._begin_active(peak)
        self._active_presence_sum = presence_sum
        self._active_presence_count = presence_count

    def _clear_pending_start(self) -> None:
        self._pending_start = None
        self._pending_start_age = 0
        self._pending_start_presence_hits = 0
        self._pending_start_presence_sum = 0.0
        self._pending_start_presence_count = 0

    def _begin_active(self, peak: _CausalEventPeak) -> None:
        self._clear_pending_start()
        self._active_start = peak
        self._active_presence_sum = peak.presence
        self._active_presence_count = 1
        self._active_start_evidence = peak.probability
        self._pending_end = None
        self._last_end_candidate = None
        self._first_strong_end_candidate = None
        self._pending_age = 0
        self._recovery_count = 0

    def _finish_active(self, candidate: _CausalEventPeak) -> SegmentPrediction | None:
        if (
            self.config.preserve_first_strong_end_candidate
            and self._first_strong_end_candidate is not None
            and self._first_strong_end_candidate.timestamp <= candidate.timestamp
        ):
            candidate = self._first_strong_end_candidate
        end_timestamp = candidate.timestamp
        if (
            candidate.probability < self.config.end_refinement_event_threshold
            and self.config.end_refinement_frames > 0
        ):
            refined = (
                candidate.previous_timestamp
                if self.config.end_refinement_frames == 1
                else self._timestamp_for_index(
                    candidate.index - self.config.end_refinement_frames
                )
            )
            if refined is not None:
                end_timestamp = refined
        return self._finish_active_at(end_timestamp, candidate.probability)

    def _finish_active_at(
        self,
        end_timestamp: float,
        end_evidence: float,
    ) -> SegmentPrediction | None:
        if self._active_start is None:
            return None
        start_timestamp = self._active_start.timestamp
        duration = end_timestamp - start_timestamp
        prediction: SegmentPrediction | None = None
        if (
            self.config.minimum_duration_seconds
            <= duration
            <= self.config.maximum_duration_seconds
            and self._active_presence_count > 0
        ):
            mean_presence = self._active_presence_sum / self._active_presence_count
            confidence = (
                mean_presence + self._active_start_evidence + end_evidence
            ) / 3.0
            confidence = min(1.0, max(0.0, confidence))
            if confidence >= self.config.score_threshold:
                prediction = SegmentPrediction(
                    start_seconds=start_timestamp,
                    end_seconds=end_timestamp,
                    confidence=confidence,
                )
        self._active_start = None
        self._active_presence_sum = 0.0
        self._active_presence_count = 0
        self._active_start_evidence = 0.0
        self._pending_end = None
        self._last_end_candidate = None
        self._first_strong_end_candidate = None
        self._pending_age = 0
        self._recovery_count = 0
        return prediction

    def _timestamp_for_index(self, index: int) -> float | None:
        for sample_index, timestamp in reversed(self._recent_timestamps):
            if sample_index == index:
                return timestamp
            if sample_index < index:
                break
        return None

    def _clear_state(self) -> None:
        self._last_timestamp = None
        self._last_duration = 0.0
        self._recent_timestamps.clear()
        self._clear_pending_start()
        self._active_start = None
        self._active_presence_sum = 0.0
        self._active_presence_count = 0
        self._active_start_evidence = 0.0
        self._pending_end = None
        self._last_end_candidate = None
        self._first_strong_end_candidate = None
        self._pending_age = 0
        self._recovery_count = 0
        self._last_strong_end_timestamp = -math.inf
        self._pending_track_predictions.clear()


class StreamingAnchorDecoder:
    """Decode causal end-anchor proposals with a fixed local peak buffer."""

    def __init__(self, config: StreamingDecoderConfig) -> None:
        if not isinstance(config, StreamingDecoderConfig):
            raise TypeError("config must be a StreamingDecoderConfig")
        self.config = config
        self._closed = False
        self._pending: tuple[float, float, float, float, float, float] | None = None
        self._pending_age = 0
        self._last_timestamp: float | None = None
        self._last_duration = 0.0
        self._last_prediction: SegmentPrediction | None = None
        self._last_anchor_timestamp: float | None = None
        self._start_event_history: deque[tuple[float, float]] = deque()
        self._end_event_history: deque[tuple[float, float]] = deque()

    def push(
        self,
        timestamp_seconds: float,
        duration_seconds: float,
        anchor_probability: float,
        start_offset_seconds: float,
        end_offset_seconds: float,
        end_event_probability: float = 0.0,
        start_event_probability: float = 0.0,
    ) -> tuple[SegmentPrediction, ...]:
        if self._closed:
            raise RuntimeError("cannot push samples after the decoder is closed")
        _validate_non_negative("timestamp_seconds", timestamp_seconds)
        _validate_non_negative("duration_seconds", duration_seconds)
        _validate_probability("anchor_probability", anchor_probability)
        _validate_probability("end_event_probability", end_event_probability)
        _validate_probability("start_event_probability", start_event_probability)
        for name, value in (
            ("start_offset_seconds", start_offset_seconds),
            ("end_offset_seconds", end_offset_seconds),
        ):
            if not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        timestamp = float(timestamp_seconds)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("sample timestamps must be strictly increasing")
        self._last_timestamp = timestamp
        self._last_duration = float(duration_seconds)
        self._start_event_history.append(
            (timestamp, float(start_event_probability))
        )
        self._end_event_history.append((timestamp, float(end_event_probability)))
        history_start = timestamp - max(
            self.config.maximum_duration_seconds,
            self.config.anchor_start_refinement_seconds,
            self.config.anchor_end_refinement_seconds,
        ) - 1.0
        while self._start_event_history and self._start_event_history[0][0] < history_start:
            self._start_event_history.popleft()
        while self._end_event_history and self._end_event_history[0][0] < history_start:
            self._end_event_history.popleft()

        emitted: list[SegmentPrediction] = []
        score = float(anchor_probability)
        end_event = float(end_event_probability)
        qualifies = (
            end_event >= self.config.anchor_end_event_threshold
        )
        candidate = (
            timestamp,
            score,
            float(start_offset_seconds),
            float(end_offset_seconds),
            float(duration_seconds),
            score * end_event if self.config.anchor_end_event_threshold > 0.0 else score,
        )
        if self._pending is not None:
            if (
                qualifies
                and score >= self.config.anchor_score_threshold
                and candidate[5] > self._pending[5]
            ):
                self._pending = candidate
                self._pending_age = 0
            else:
                self._pending_age += 1
                if self._pending_age > self.config.anchor_peak_radius_frames:
                    prediction = self._finish_pending()
                    if prediction is not None:
                        emitted.append(prediction)
                    if qualifies and score >= self.config.anchor_score_threshold:
                        self._pending = candidate
                        self._pending_age = 0
        elif qualifies and score >= self.config.anchor_score_threshold:
            self._pending = candidate
            self._pending_age = 0

        if self._pending is not None and self._pending_age > self.config.anchor_peak_radius_frames:
            prediction = self._finish_pending()
            if prediction is not None:
                emitted.append(prediction)
        return tuple(emitted)

    def close(self) -> tuple[SegmentPrediction, ...]:
        if self._closed:
            return ()
        emitted: tuple[SegmentPrediction, ...] = ()
        prediction = self._finish_pending()
        if prediction is not None:
            emitted = (prediction,)
        self._pending = None
        self._closed = True
        return emitted

    def _finish_pending(self) -> SegmentPrediction | None:
        if self._pending is None:
            return None
        timestamp, score, start_offset, end_offset, _, _ = self._pending
        self._pending = None
        self._pending_age = 0
        if (
            self._last_anchor_timestamp is not None
            and timestamp - self._last_anchor_timestamp
            < max(
                self.config.minimum_duration_seconds,
                self.config.minimum_anchor_gap_seconds,
            )
        ):
            return None
        start = timestamp + start_offset
        end = timestamp + end_offset
        if self.config.anchor_start_refinement_seconds > 0.0:
            start = self._refine_boundary(
                start,
                self.config.anchor_start_refinement_seconds,
                self.config.anchor_start_event_threshold,
                self._start_event_history,
            )
        if self.config.anchor_pair_start_events:
            start = self._pair_start_event(
                timestamp,
                start,
                self.config.anchor_start_event_threshold,
                self._start_event_history,
                self._last_anchor_timestamp,
                self.config.minimum_anchor_gap_seconds,
            )
        if self.config.anchor_end_refinement_seconds > 0.0:
            end = self._refine_boundary(
                end,
                self.config.anchor_end_refinement_seconds,
                self.config.anchor_end_event_threshold,
                self._end_event_history,
            )
        duration = end - start
        if (
            start < 0.0
            or duration < self.config.minimum_duration_seconds
            or duration > self.config.maximum_duration_seconds
        ):
            return None
        prediction = SegmentPrediction(start, end, score)
        previous = self._last_prediction
        if previous is not None:
            intersection = max(
                0.0,
                min(previous.end_seconds, prediction.end_seconds)
                - max(previous.start_seconds, prediction.start_seconds),
            )
            union = max(previous.end_seconds, prediction.end_seconds) - min(
                previous.start_seconds, prediction.start_seconds
            )
            iou = intersection / union if union > 0.0 else 0.0
            if iou >= self.config.anchor_nms_iou_threshold:
                return None
        self._last_prediction = prediction
        self._last_anchor_timestamp = timestamp
        return prediction

    @staticmethod
    def _refine_boundary(
        estimate: float,
        window: float,
        event_threshold: float,
        history: deque[tuple[float, float]],
    ) -> float:
        candidates = [
            (-probability, abs(timestamp - estimate), timestamp)
            for timestamp, probability in history
            if abs(timestamp - estimate) <= window
            and probability >= event_threshold
        ]
        if not candidates:
            return estimate
        return min(candidates)[2]

    @staticmethod
    def _pair_start_event(
        anchor_timestamp: float,
        estimate: float,
        event_threshold: float,
        history: deque[tuple[float, float]],
        previous_anchor_timestamp: float | None,
        minimum_gap_seconds: float,
    ) -> float:
        lower_bound = (
            -math.inf
            if previous_anchor_timestamp is None
            else previous_anchor_timestamp + max(0.2, minimum_gap_seconds)
        )
        candidates = [
            (probability, timestamp)
            for timestamp, probability in history
            if lower_bound < timestamp < anchor_timestamp
            and anchor_timestamp - timestamp <= 8.0
            and probability >= event_threshold
        ]
        if not candidates:
            return estimate
        return max(candidates)[1]


def decode_stream_anchor_predictions(
    predictions: np.ndarray,
    timestamps: np.ndarray,
    durations: np.ndarray,
    config: StreamingDecoderConfig = StreamingDecoderConfig(),
) -> list[SegmentPrediction]:
    """Decode `[anchor, start offset, end offset]` stream predictions."""
    predictions = np.asarray(predictions)
    timestamps = np.asarray(timestamps)
    durations = np.asarray(durations)
    if predictions.ndim != 2 or predictions.shape[1] != 6:
        raise ValueError("anchor predictions must have shape [samples,6]")
    if timestamps.shape != (len(predictions),) or durations.shape != (len(predictions),):
        raise ValueError("timestamps and durations must match anchor predictions")
    decoder = StreamingAnchorDecoder(config)
    segments: list[SegmentPrediction] = []
    for row, timestamp, duration in zip(
        predictions, timestamps, durations, strict=True
    ):
        segments.extend(
            decoder.push(
                float(timestamp),
                float(duration),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[2]),
                float(row[1]),
            )
        )
    segments.extend(decoder.close())
    return segments


def decode_stream_predictions(
    predictions: np.ndarray,
    timestamps: np.ndarray,
    durations: np.ndarray,
    config: StreamingDecoderConfig = StreamingDecoderConfig(),
) -> list[SegmentPrediction]:
    """Decode `[presence, start event, end event]` probabilities in stream order."""
    predictions = np.asarray(predictions)
    timestamps = np.asarray(timestamps)
    durations = np.asarray(durations)
    if predictions.ndim != 2 or predictions.shape[1] != 3:
        raise ValueError("predictions must have shape [samples,3]")
    sample_count = len(predictions)
    if timestamps.shape != (sample_count,):
        raise ValueError("timestamps must have shape [samples]")
    if durations.shape != (sample_count,):
        raise ValueError("durations must have shape [samples]")

    decoder: StreamingSegmentDecoder | StreamingEventPairDecoder
    decoder = (
        StreamingEventPairDecoder(config)
        if config.causal_event_pairing
        else StreamingSegmentDecoder(config)
    )
    segments: list[SegmentPrediction] = []
    for prediction, timestamp, duration in zip(
        predictions, timestamps, durations, strict=True
    ):
        segments.extend(
            decoder.push(
                timestamp_seconds=timestamp,
                duration_seconds=duration,
                presence_probability=prediction[0],
                start_event_probability=prediction[1],
                end_event_probability=prediction[2],
            )
        )
    segments.extend(decoder.close())
    return segments
