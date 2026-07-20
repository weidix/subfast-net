from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from subtitle_timing_core.dataset import FeatureCache, read_manifest
from subtitle_timing_stream.stream_train import StreamingTrainSettings, train_streaming

from . import FEATURE_FORMAT, INPUT_DOMAIN


class FullFrameTrainSettings(BaseModel):
    manifest: Path
    output_dir: Path
    epochs: int = Field(default=24, gt=0)
    batch_size: int = Field(default=64, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    window_frames: int = Field(default=512, gt=15)
    stride_frames: int = Field(default=256, gt=0)
    boundary_event_sigma_seconds: float = Field(default=0.05, gt=0.0)
    target_recall: float = Field(default=1.0, gt=0.0, le=1.0)
    minimum_duration_seconds: float = Field(default=0.20, ge=0.0)
    maximum_duration_seconds: float = Field(default=8.0, gt=0.0)
    width: int = Field(default=64, ge=16)
    temporal_layers: int = Field(default=6, ge=1, le=9)
    recurrent_layers: int = Field(default=1, ge=1, le=3)
    dropout: float = Field(default=0.10, ge=0.0, lt=1.0)
    use_segment_head: bool = False
    segment_boundary_weight: float = Field(default=2.0, ge=0.0)
    segment_loss_weight: float = Field(default=0.25, ge=0.0)
    negative_weight: float = Field(default=1.0, gt=0.0)
    boundary_event_loss_weight: float = Field(default=1.5, ge=0.0)
    clean_negative_weight: float = Field(default=4.0, ge=1.0)
    short_segment_weight: float = Field(default=2.0, ge=1.0)
    max_train_windows: int | None = Field(default=None, gt=0)
    max_val_windows: int | None = Field(default=None, gt=0)
    inference_chunk_frames: int = Field(default=128, gt=0)
    initial_checkpoint: Path | None = None
    calibration_profile: Literal["fast", "full"] = "fast"
    seed: int = 2026
    device: str = "auto"
    validation_mode: Literal["held_out", "diagnostic_temporal"] = "held_out"
    temporal_guard_seconds: float = Field(default=10.0, ge=0.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.stride_frames > self.window_frames:
            raise ValueError("stride_frames must not exceed window_frames")
        if self.maximum_duration_seconds <= self.minimum_duration_seconds:
            raise ValueError("maximum duration must exceed minimum duration")
        return self


def _validate_manifest(manifest: Path) -> None:
    records = read_manifest(manifest)
    if not records:
        raise ValueError("full-frame manifest contains no records")
    expected_settings: dict[str, object] | None = None
    for record in records:
        cache = FeatureCache(record.feature_dir)
        try:
            if (
                cache.meta.get("format") != FEATURE_FORMAT
                or cache.meta.get("input_domain") != INPUT_DOMAIN
                or cache.meta.get("spatial_contract", {}).get("requested_roi")
                != "full_frame"
            ):
                raise ValueError(
                    f"manifest contains a non-full-frame cache: {record.feature_dir}"
                )
            current_settings = dict(cache.meta["full_frame_feature_settings"])
            if expected_settings is None:
                expected_settings = current_settings
            elif current_settings != expected_settings:
                raise ValueError(
                    "all full-frame caches must use the same feature settings"
                )
        finally:
            cache.release()


def train_full_frame(settings: FullFrameTrainSettings) -> dict[str, object]:
    manifest = settings.manifest.expanduser().resolve()
    _validate_manifest(manifest)
    summary = train_streaming(
        StreamingTrainSettings(
            manifest=manifest,
            output_dir=settings.output_dir,
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            window_frames=settings.window_frames,
            stride_frames=settings.stride_frames,
            boundary_event_sigma_seconds=settings.boundary_event_sigma_seconds,
            target_recall=settings.target_recall,
            minimum_duration_seconds=settings.minimum_duration_seconds,
            maximum_duration_seconds=settings.maximum_duration_seconds,
            width=settings.width,
            temporal_layers=settings.temporal_layers,
            recurrent_layers=settings.recurrent_layers,
            dropout=settings.dropout,
            use_byte_branch=False,
            use_segment_head=settings.use_segment_head,
            segment_boundary_weight=settings.segment_boundary_weight,
            segment_loss_weight=settings.segment_loss_weight,
            negative_weight=settings.negative_weight,
            boundary_event_loss_weight=settings.boundary_event_loss_weight,
            clean_negative_weight=settings.clean_negative_weight,
            short_segment_weight=settings.short_segment_weight,
            max_train_windows=settings.max_train_windows,
            max_val_windows=settings.max_val_windows,
            inference_chunk_frames=settings.inference_chunk_frames,
            initial_checkpoint=settings.initial_checkpoint,
            calibration_profile=(
                "fast" if settings.calibration_profile == "fast" else "standard"
            ),
            seed=settings.seed,
            device=settings.device,
            validation_mode=settings.validation_mode,
            temporal_guard_seconds=settings.temporal_guard_seconds,
            input_domain=INPUT_DOMAIN,
        )
    )
    summary["full_frame_feature_input"] = True
    summary["fixed_subtitle_roi"] = False
    summary_path = settings.output_dir.expanduser().resolve() / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
