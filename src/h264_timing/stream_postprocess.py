from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from .postprocess import SegmentPrediction


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

    def __post_init__(self) -> None:
        _validate_probability("score_threshold", self.score_threshold)
        _validate_probability("presence_on_threshold", self.presence_on_threshold)
        _validate_probability("presence_off_threshold", self.presence_off_threshold)
        _validate_probability("boundary_event_threshold", self.boundary_event_threshold)
        if self.presence_off_threshold >= self.presence_on_threshold:
            raise ValueError(
                "presence_off_threshold must be less than presence_on_threshold"
            )
        _validate_non_negative(
            "minimum_duration_seconds", self.minimum_duration_seconds
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

    decoder = StreamingSegmentDecoder(config)
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
