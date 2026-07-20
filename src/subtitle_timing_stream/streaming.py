from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import FEATURE_VERSION, STREAM_CHECKPOINT_FORMAT, STREAM_CHECKPOINT_VERSION
from subtitle_timing_core.postprocess import SegmentPrediction
from .stream_model import StreamingH264SubtitleModel, StreamingModelConfig
from .stream_postprocess import (
    StreamingAnchorDecoder,
    StreamingDecoderConfig,
    StreamingEventPairDecoder,
    StreamingSegmentDecoder,
)


@dataclass(frozen=True)
class StreamSample:
    """One chronological feature sample consumed by streaming inference."""

    timestamp_seconds: float
    duration_seconds: float
    features: Sequence[float] | np.ndarray
    tokens: Sequence[int] | np.ndarray | None = None


class StreamingSegmentDetector:
    """Stateful causal detector that owns model and segment-decoder state."""

    def __init__(
        self,
        model: StreamingH264SubtitleModel,
        feature_mean: Sequence[float] | np.ndarray | torch.Tensor,
        feature_std: Sequence[float] | np.ndarray | torch.Tensor,
        decoder_config: StreamingDecoderConfig = StreamingDecoderConfig(),
        *,
        device: str | torch.device = "cpu",
        feature_names: Sequence[str] | None = None,
    ) -> None:
        resolved_device = _select_device(device)
        self._model = model.to(resolved_device).eval()
        self._model.prepare_step_inference()
        self._device = resolved_device
        self._model_dtype = _model_dtype(self._model)
        self._feature_mean = _feature_statistic(
            feature_mean,
            name="feature mean",
            feature_count=self._model.config.feature_count,
            device=resolved_device,
            dtype=self._model_dtype,
        )
        self._feature_std = _feature_statistic(
            feature_std,
            name="feature standard deviation",
            feature_count=self._model.config.feature_count,
            device=resolved_device,
            dtype=self._model_dtype,
        )
        if torch.any(self._feature_std <= 0.0).item():
            raise ValueError("feature standard deviation must be positive")
        self._feature_scale = self._feature_std.reciprocal()
        self._feature_offset = -self._feature_mean * self._feature_scale
        self._feature_names = _validate_feature_names(
            feature_names,
            feature_count=self._model.config.feature_count,
        )
        self._decoder_config = decoder_config
        if model.config.use_segment_head:
            self._decoder = StreamingAnchorDecoder(decoder_config)
        elif decoder_config.causal_event_pairing:
            self._decoder = StreamingEventPairDecoder(decoder_config)
        else:
            self._decoder = StreamingSegmentDecoder(decoder_config)
        self._model_state: Any | None = None
        self._last_timestamp_seconds: float | None = None
        self._closed = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "auto",
    ) -> StreamingSegmentDetector:
        """Load a strict causal-streaming checkpoint into a fresh detector."""

        path = Path(checkpoint_path).expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"invalid streaming checkpoint: {path}")
        if (
            checkpoint.get("format") != STREAM_CHECKPOINT_FORMAT
            or checkpoint.get("version") != STREAM_CHECKPOINT_VERSION
            or checkpoint.get("feature_version") != FEATURE_VERSION
        ):
            raise ValueError(f"unsupported streaming checkpoint: {path}")
        if checkpoint.get("model_output_contract") not in {
            "causal_streaming_presence_events",
            "causal_streaming_presence_events_anchor",
        }:
            raise ValueError(
                f"checkpoint does not contain causal streaming output: {path}"
            )

        required = (
            "model_config",
            "model",
            "feature_names",
            "feature_mean",
            "feature_std",
            "streaming_decoder_config",
        )
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(
                "streaming checkpoint is missing required fields: " + ", ".join(missing)
            )
        try:
            model_config_values = dict(checkpoint["model_config"])
            # Version-one checkpoints predate the optional anchor heads.
            if not any(
                key.startswith("segment_anchor_head")
                for key in checkpoint["model"]
            ):
                model_config_values["use_segment_head"] = False
            model_config = StreamingModelConfig(**model_config_values)
            decoder_config = StreamingDecoderConfig(
                **checkpoint["streaming_decoder_config"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid streaming checkpoint configuration: {path}"
            ) from error

        model = StreamingH264SubtitleModel(model_config)
        model.load_state_dict(checkpoint["model"], strict=True)
        return cls(
            model,
            checkpoint["feature_mean"],
            checkpoint["feature_std"],
            decoder_config,
            device=device,
            feature_names=checkpoint["feature_names"],
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def retained_state_elements(self) -> int:
        """Return tensor elements retained solely to continue model inference."""

        return _tensor_elements(self._model_state)

    def push(self, sample: StreamSample) -> tuple[SegmentPrediction, ...]:
        """Process one sample and return segments finalized by that sample."""

        self._ensure_open()
        if not isinstance(sample, StreamSample):
            raise TypeError("stream input must contain StreamSample values")
        timestamp, duration = _validate_timing(sample)
        if (
            self._last_timestamp_seconds is not None
            and timestamp <= self._last_timestamp_seconds
        ):
            raise ValueError("stream sample timestamps must be strictly increasing")
        feature_row = _sample_features(
            sample.features, self._model.config.feature_count
        )
        token_row = _sample_tokens(
            sample.tokens,
            token_count=self._model.config.token_count,
            required=self._model.config.use_byte_branch,
        )
        features = torch.as_tensor(
            feature_row,
            device=self._device,
            dtype=self._model_dtype,
        ).reshape(1, 1, -1)
        features = torch.addcmul(
            self._feature_offset, features, self._feature_scale
        )
        tokens = torch.as_tensor(
            token_row,
            device=self._device,
            dtype=torch.long,
        ).reshape(1, 1, -1)

        with torch.inference_mode():
            output, next_state = self._model.forward_step(
                features,
                tokens,
                self._model_state,
            )
            values = _single_frame_output_values(
                output,
                use_segment_head=self._model.config.use_segment_head,
            )

        if self._model.config.use_segment_head:
            finalized = self._decoder.push(
                timestamp,
                duration,
                values[3],
                values[4],
                values[5],
                values[2],
                values[1],
            )
        else:
            finalized = self._decoder.push(
                timestamp,
                duration,
                values[0],
                values[1],
                values[2],
            )
        self._model_state = next_state
        self._last_timestamp_seconds = timestamp
        return finalized

    def push_many(
        self, samples: Iterable[StreamSample]
    ) -> tuple[SegmentPrediction, ...]:
        """Process one chronological chunk in a single model invocation."""

        self._ensure_open()
        chunk = tuple(samples)
        if not chunk:
            return ()
        if len(chunk) == 1:
            return self.push(chunk[0])

        timestamps: list[float] = []
        durations: list[float] = []
        feature_rows: list[np.ndarray] = []
        token_rows: list[np.ndarray] = []
        previous_timestamp = self._last_timestamp_seconds
        for sample in chunk:
            if not isinstance(sample, StreamSample):
                raise TypeError("stream input must contain StreamSample values")
            timestamp, duration = _validate_timing(sample)
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError("stream sample timestamps must be strictly increasing")
            previous_timestamp = timestamp
            timestamps.append(timestamp)
            durations.append(duration)
            feature_rows.append(
                _sample_features(sample.features, self._model.config.feature_count)
            )
            token_rows.append(
                _sample_tokens(
                    sample.tokens,
                    token_count=self._model.config.token_count,
                    required=self._model.config.use_byte_branch,
                )
            )

        features = torch.as_tensor(
            np.stack(feature_rows),
            device=self._device,
            dtype=self._model_dtype,
        ).unsqueeze(0)
        features = torch.addcmul(
            self._feature_offset, features, self._feature_scale
        )
        tokens = torch.as_tensor(
            np.stack(token_rows),
            device=self._device,
            dtype=torch.long,
        ).unsqueeze(0)

        with torch.inference_mode():
            output, next_state = self._model.forward_stream(
                features,
                tokens,
                self._model_state,
            )
            expected_presence_shape = (1, len(chunk))
            expected_boundary_shape = (1, len(chunk), 2)
            if output.presence_logits.shape != expected_presence_shape:
                raise RuntimeError(
                    "streaming model presence output must have shape "
                    f"{expected_presence_shape}"
                )
            if output.boundary_event_logits.shape != expected_boundary_shape:
                raise RuntimeError(
                    "streaming model boundary output must have shape "
                    f"{expected_boundary_shape}"
                )
            if self._model.config.use_segment_head:
                expected_anchor_shape = (1, len(chunk))
                if (
                    output.segment_anchor_logits.shape != expected_anchor_shape
                    or output.segment_start_offsets_seconds.shape != expected_anchor_shape
                    or output.segment_end_offsets_seconds.shape != expected_anchor_shape
                ):
                    raise RuntimeError(
                        "streaming model segment-anchor outputs must have shape "
                        f"{expected_anchor_shape}"
                    )
            if not torch.isfinite(output.presence_logits).all().item() or not (
                torch.isfinite(output.boundary_event_logits).all().item()
            ):
                raise RuntimeError("streaming model output must be finite")
            if self._model.config.use_segment_head and not (
                torch.isfinite(output.segment_anchor_logits).all().item()
                and torch.isfinite(output.segment_start_offsets_seconds).all().item()
                and torch.isfinite(output.segment_end_offsets_seconds).all().item()
            ):
                raise RuntimeError("streaming model segment-anchor output must be finite")
            presence_probabilities = (
                torch.sigmoid(output.presence_logits[0]).detach().cpu().tolist()
            )
            boundary_probabilities = (
                torch.sigmoid(output.boundary_event_logits[0]).detach().cpu().tolist()
            )
            if self._model.config.use_segment_head:
                anchor_probabilities = (
                    torch.sigmoid(output.segment_anchor_logits[0])
                    .detach()
                    .cpu()
                    .tolist()
                )
                start_offsets = (
                    output.segment_start_offsets_seconds[0].detach().cpu().tolist()
                )
                end_offsets = (
                    output.segment_end_offsets_seconds[0].detach().cpu().tolist()
                )

        finalized: list[SegmentPrediction] = []
        if self._model.config.use_segment_head:
            for timestamp, duration, anchor, start_offset, end_offset, boundaries in zip(
                timestamps,
                durations,
                anchor_probabilities,
                start_offsets,
                end_offsets,
                boundary_probabilities,
                strict=True,
            ):
                finalized.extend(
                    self._decoder.push(
                        timestamp,
                        duration,
                        float(anchor),
                        float(start_offset),
                        float(end_offset),
                        float(boundaries[1]),
                        float(boundaries[0]),
                    )
                )
        else:
            for timestamp, duration, presence, boundaries in zip(
                timestamps,
                durations,
                presence_probabilities,
                boundary_probabilities,
                strict=True,
            ):
                finalized.extend(
                    self._decoder.push(
                        timestamp,
                        duration,
                        float(presence),
                        float(boundaries[0]),
                        float(boundaries[1]),
                    )
                )
        self._model_state = next_state
        self._last_timestamp_seconds = timestamps[-1]
        return tuple(finalized)

    def close(self) -> tuple[SegmentPrediction, ...]:
        """Flush the open segment once and release causal continuation state."""

        if self._closed:
            return ()
        self._closed = True
        try:
            return self._decoder.close()
        finally:
            self._model_state = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("streaming detector is closed")


def _select_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # This small stateful model is launch-bound on MPS at its deployment chunk
    # sizes. CPU is substantially faster; MPS remains available explicitly.
    return torch.device("cpu")


def _model_dtype(model: StreamingH264SubtitleModel) -> torch.dtype:
    parameter = next(model.parameters(), None)
    if parameter is None or not parameter.dtype.is_floating_point:
        raise ValueError("streaming model must contain floating-point parameters")
    return parameter.dtype


def _feature_statistic(
    values: Sequence[float] | np.ndarray | torch.Tensor,
    *,
    name: str,
    feature_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    try:
        statistic = torch.as_tensor(values, device=device, dtype=dtype).detach().clone()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector") from error
    if statistic.shape != (feature_count,):
        raise ValueError(f"{name} must have shape ({feature_count},)")
    if not torch.isfinite(statistic).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return statistic.reshape(1, 1, feature_count)


def _validate_feature_names(
    values: Sequence[str] | None, *, feature_count: int
) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raise ValueError("feature names must be a sequence of names")
    names = tuple(values)
    if len(names) != feature_count:
        raise ValueError(f"feature names must contain {feature_count} entries")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    return names


def _validate_timing(sample: StreamSample) -> tuple[float, float]:
    timestamp = float(sample.timestamp_seconds)
    duration = float(sample.duration_seconds)
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("stream sample timestamp must be finite and non-negative")
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("stream sample duration must be finite and non-negative")
    return timestamp, duration


def _sample_features(
    values: Sequence[float] | np.ndarray, feature_count: int
) -> np.ndarray:
    try:
        features = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("stream sample features must be numeric") from error
    if features.shape != (feature_count,):
        raise ValueError(f"stream sample features must have shape ({feature_count},)")
    if not np.isfinite(features).all():
        raise ValueError("stream sample features must contain only finite values")
    return features


def _sample_tokens(
    values: Sequence[int] | np.ndarray | None,
    *,
    token_count: int,
    required: bool,
) -> np.ndarray:
    if values is None:
        if required:
            raise ValueError("stream sample tokens are required by this model")
        return np.zeros((token_count,), dtype=np.int64)
    tokens = np.asarray(values)
    if tokens.shape != (token_count,):
        raise ValueError(f"stream sample tokens must have shape ({token_count},)")
    if tokens.dtype.kind not in "iu":
        raise ValueError("stream sample tokens must contain integers")
    if np.any(tokens < 0) or np.any(tokens > 255):
        raise ValueError("stream sample tokens must be in [0,255]")
    return tokens.astype(np.int64, copy=False)


def _tensor_elements(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, dict):
        return sum(_tensor_elements(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_elements(item) for item in value)
    return 0


def _single_frame_output_values(
    output: Any, *, use_segment_head: bool
) -> list[float]:
    if output.presence_logits.shape != (1, 1):
        raise RuntimeError("streaming model presence output must have shape (1, 1)")
    if output.boundary_event_logits.shape != (1, 1, 2):
        raise RuntimeError("streaming model boundary output must have shape (1, 1, 2)")
    tensors = [
        output.presence_logits.reshape(-1),
        output.boundary_event_logits.reshape(-1),
    ]
    probability_count = 3
    if use_segment_head:
        expected_shape = (1, 1)
        if (
            output.segment_anchor_logits.shape != expected_shape
            or output.segment_start_offsets_seconds.shape != expected_shape
            or output.segment_end_offsets_seconds.shape != expected_shape
        ):
            raise RuntimeError(
                "streaming model segment-anchor outputs must have shape (1, 1)"
            )
        tensors.extend(
            (
                output.segment_anchor_logits.reshape(-1),
                output.segment_start_offsets_seconds.reshape(-1),
                output.segment_end_offsets_seconds.reshape(-1),
            )
        )
        probability_count = 4
    values = torch.cat(tensors)
    if not torch.isfinite(values).all().item():
        raise RuntimeError("streaming model output must be finite")
    values[:probability_count].sigmoid_()
    return values.detach().cpu().tolist()
