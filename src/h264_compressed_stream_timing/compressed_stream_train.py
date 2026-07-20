from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import Field
from torch.optim import AdamW
from torch.utils.data import DataLoader

from . import (
    COMPRESSED_STREAM_CHECKPOINT_FORMAT,
    COMPRESSED_STREAM_CHECKPOINT_VERSION,
)
from subtitle_timing_core.dataset import (
    FeatureCache,
    compute_feature_stats,
    ensure_source_disjoint,
    read_manifest,
)
from subtitle_timing_core.formats import H264_TIMING_FEATURE_VERSION as FEATURE_VERSION
from subtitle_timing_core.metrics import match_intervals
from subtitle_timing_stream import (
    STREAM_CHECKPOINT_FORMAT,
    STREAM_CHECKPOINT_VERSION,
)
from subtitle_timing_stream.stream_dataset import (
    StreamingLoadedRecord,
    StreamingTimingWindowDataset,
    load_streaming_records,
)
from subtitle_timing_stream.stream_model import (
    StreamingH264SubtitleModel,
    StreamingModelConfig,
)
from subtitle_timing_stream.stream_postprocess import StreamingDecoderConfig
from subtitle_timing_stream.stream_predict import predict_stream_cache
from subtitle_timing_stream.stream_train import (
    StreamingTrainSettings,
    _boundary_event_positive_weights,
    _calibrate_decoder,
    _checkpoint_selection_key,
    _decode_records,
    _frame_tolerance,
    _predict_records,
    _prepare_output_dir,
    _presence_positive_weight,
    _seed_everything,
    _segment_anchor_positive_weight,
    _select_device,
    _validation_metrics,
    _window_loss,
)


class CompressedStreamingTrainSettings(StreamingTrainSettings):
    """Settings for the deployment-strict compressed-domain stream family."""

    batch_size: int = Field(default=64, gt=0)
    width: int = Field(default=128, ge=16)
    use_byte_branch: bool = True
    use_segment_head: bool = False
    boundary_event_loss_weight: float = Field(default=1.5, ge=0.0)
    clean_negative_weight: float = Field(default=4.0, ge=1.0)
    short_segment_weight: float = Field(default=2.0, ge=1.0)
    visual_teacher_checkpoint: Path | None = None
    visual_teacher_manifest: Path | None = None
    visual_distillation_weight: float = Field(default=0.5, ge=0.0)
    calibration_interval_epochs: int = Field(default=5, gt=0)


class _CompressedFeatureCacheView:
    """Expose only compressed arrays while retaining cache timing metadata."""

    def __init__(self, cache: FeatureCache) -> None:
        self._cache = cache
        self.features = cache.compressed_features
        self.feature_names = cache.compressed_feature_names

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cache, name)


class _TeacherSupervisedWindowDataset(StreamingTimingWindowDataset):
    def __init__(
        self,
        *args: Any,
        visual_teacher_predictions: list[np.ndarray],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if len(visual_teacher_predictions) != len(self.records):
            raise ValueError("visual teacher predictions must match training records")
        self.visual_teacher_predictions = visual_teacher_predictions

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        batch = super().__getitem__(index)
        record_index, core_start = self.items[index]
        record = self.records[record_index]
        core_stop = min(core_start + self.stride_frames, len(record.cache.timestamps))
        source_start = max(0, core_start - self.history_frames)
        source_stop = core_stop
        source_length = source_stop - source_start
        left_padding = self.history_frames - (core_start - source_start)
        destination_stop = left_padding + source_length
        teacher = np.zeros((self.window_frames, 3), dtype=np.float32)
        teacher[left_padding:destination_stop] = self.visual_teacher_predictions[
            record_index
        ][source_start:source_stop]
        batch["visual_teacher_probabilities"] = torch.from_numpy(teacher)
        return batch


def _compressed_records(
    records: list[StreamingLoadedRecord],
) -> list[StreamingLoadedRecord]:
    return [
        StreamingLoadedRecord(
            record=item.record,
            cache=_CompressedFeatureCacheView(item.cache),  # type: ignore[arg-type]
            intervals=item.intervals,
            presence_targets=item.presence_targets,
            boundary_event_targets=item.boundary_event_targets,
            segment_anchor_targets=item.segment_anchor_targets,
        )
        for item in records
    ]


def _visual_teacher_predictions(
    checkpoint_path: Path,
    records: list[StreamingLoadedRecord],
    *,
    chunk_frames: int,
    device: torch.device,
) -> list[np.ndarray]:
    path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != STREAM_CHECKPOINT_FORMAT
        or checkpoint.get("version") != STREAM_CHECKPOINT_VERSION
        or checkpoint.get("feature_version") != FEATURE_VERSION
        or not checkpoint.get("visual_feature_settings")
    ):
        raise ValueError(f"unsupported visual teacher checkpoint: {path}")
    model_config_values = dict(checkpoint["model_config"])
    if not any(
        name.startswith("segment_anchor_head") for name in checkpoint["model"]
    ):
        model_config_values["use_segment_head"] = False
    teacher_model = StreamingH264SubtitleModel(
        StreamingModelConfig(**model_config_values)
    ).to(device)
    teacher_model.load_state_dict(checkpoint["model"], strict=True)
    predictions: list[np.ndarray] = []
    for item in records:
        if item.cache.feature_names != list(checkpoint["feature_names"]):
            raise ValueError(
                "visual teacher feature schema differs from the training manifest"
            )
        predictions.append(
            predict_stream_cache(
                teacher_model,
                item.cache,
                feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
                feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
                chunk_frames=chunk_frames,
                device=device,
                include_segments=False,
            )[:, :3]
        )
    return predictions


def _maximum_boundary_drift_frames(
    predictions: list[tuple[StreamingLoadedRecord, np.ndarray]],
    decoder_config: StreamingDecoderConfig,
) -> float:
    maximum = 0.0
    for item, segments in _decode_records(predictions, decoder_config):
        predicted = [
            segment.to_interval()
            for segment in segments
            if segment.confidence >= decoder_config.score_threshold
        ]
        matches = match_intervals(predicted, item.intervals)
        frame_seconds = _frame_tolerance(item)
        if frame_seconds <= 0.0:
            raise ValueError("validation cache has no positive frame duration")
        for predicted_index, target_index, _ in matches:
            prediction = predicted[predicted_index]
            target = item.intervals[target_index]
            maximum = max(
                maximum,
                abs(prediction.start_seconds - target.start_seconds) / frame_seconds,
                abs(prediction.end_seconds - target.end_seconds) / frame_seconds,
            )
    return float(maximum)


def _quality_gate(metrics: dict[str, float]) -> dict[str, float | bool]:
    recall = float(metrics.get("interval_recall_iou50", 0.0))
    interval_f1 = float(metrics.get("interval_f1_iou50", 0.0))
    frame_f1 = float(metrics.get("segment_f1_1frame", 0.0))
    drift = float(metrics.get("maximum_boundary_drift_frames", float("inf")))
    passed = (
        recall >= 1.0 - 1e-12
        and interval_f1 >= 1.0 - 1e-12
        and frame_f1 >= 1.0 - 1e-12
        and drift <= 1.0 + 1e-6
    )
    return {
        "passed": passed,
        "required_validation_recall": 1.0,
        "validation_recall": recall,
        "required_validation_interval_f1": 1.0,
        "validation_interval_f1": interval_f1,
        "required_validation_1frame_f1": 1.0,
        "validation_1frame_f1": frame_f1,
        "maximum_allowed_boundary_drift_frames": 1.0,
        "maximum_boundary_drift_frames": drift,
    }


def _checkpoint(
    model: StreamingH264SubtitleModel,
    *,
    model_config: StreamingModelConfig,
    decoder_config: StreamingDecoderConfig,
    settings: CompressedStreamingTrainSettings,
    train_records: list[StreamingLoadedRecord],
    val_records: list[StreamingLoadedRecord],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    presence_positive_weight: float,
    boundary_event_positive_weights: np.ndarray,
    segment_anchor_positive_weight: float | None,
    calibration_metrics: dict[str, float],
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    first_cache = train_records[0].cache
    return {
        "format": COMPRESSED_STREAM_CHECKPOINT_FORMAT,
        "version": COMPRESSED_STREAM_CHECKPOINT_VERSION,
        "feature_version": FEATURE_VERSION,
        "input_domain": "h264_compressed_only",
        "pixel_decode_required": False,
        "model": model.state_dict(),
        "model_config": model_config.to_dict(),
        "model_output_contract": (
            "causal_compressed_streaming_presence_events_anchor"
            if model_config.use_segment_head
            else "causal_compressed_streaming_presence_events"
        ),
        "compressed_feature_names": list(first_cache.feature_names),
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "feature_settings": dict(first_cache.meta["feature_settings"]),
        "spatial_contract": dict(first_cache.meta["spatial_contract"]),
        "payload_tail_ratio": float(
            first_cache.meta["feature_settings"]["payload_tail_ratio"]
        ),
        "streaming_decoder_config": decoder_config.to_dict(),
        "inference_chunk_frames": settings.inference_chunk_frames,
        "window_frames": settings.window_frames,
        "stride_frames": settings.stride_frames,
        "boundary_event_sigma_seconds": settings.boundary_event_sigma_seconds,
        "presence_positive_weight": presence_positive_weight,
        "boundary_event_positive_weights": torch.from_numpy(
            boundary_event_positive_weights
        ),
        "segment_anchor_positive_weight": segment_anchor_positive_weight,
        "validation_mode": settings.validation_mode,
        "training_provenance": {
            "manifest": str(settings.manifest.expanduser().resolve()),
            "settings": settings.model_dump(mode="json"),
            "train_video_ids": [item.record.video_id for item in train_records],
            "val_video_ids": [item.record.video_id for item in val_records],
            "train_source_groups": sorted(
                {item.record.source_group for item in train_records}
            ),
            "val_source_groups": sorted(
                {item.record.source_group for item in val_records}
            ),
            "causal_window_context_frames": (
                settings.window_frames - settings.stride_frames
            ),
            "calibration_split": "train",
            "visual_sidecars_available_during_training": bool(
                first_cache.visual_feature_settings
            ),
            "visual_features_used_as_model_input": False,
            "visual_teacher_distillation_weight": (
                settings.visual_distillation_weight
                if settings.visual_teacher_checkpoint is not None
                else 0.0
            ),
            "visual_teacher_checkpoint": (
                str(settings.visual_teacher_checkpoint.expanduser().resolve())
                if settings.visual_teacher_checkpoint is not None
                else None
            ),
            "visual_teacher_manifest": (
                str(settings.visual_teacher_manifest.expanduser().resolve())
                if settings.visual_teacher_manifest is not None
                else None
            ),
        },
        "calibration_metrics": calibration_metrics,
        "epoch": epoch,
        "metrics": metrics,
    }


def train_compressed_streaming(
    settings: CompressedStreamingTrainSettings,
) -> dict[str, Any]:
    """Train a causal model whose inference input is H.264 compressed data only."""

    output_dir, metrics_path = _prepare_output_dir(settings.output_dir)
    _seed_everything(settings.seed)
    device = _select_device(settings.device)
    raw_train_records = load_streaming_records(
        read_manifest(settings.manifest, split="train"),
        boundary_event_sigma_seconds=settings.boundary_event_sigma_seconds,
    )
    raw_val_records = load_streaming_records(
        read_manifest(settings.manifest, split="val"),
        boundary_event_sigma_seconds=settings.boundary_event_sigma_seconds,
    )
    visual_teacher_predictions: list[np.ndarray] | None = None
    if (
        settings.visual_teacher_checkpoint is not None
        and settings.visual_distillation_weight > 0.0
    ):
        teacher_records = raw_train_records
        if settings.visual_teacher_manifest is not None:
            teacher_records = load_streaming_records(
                read_manifest(settings.visual_teacher_manifest, split="train"),
                boundary_event_sigma_seconds=settings.boundary_event_sigma_seconds,
            )
            student_ids = [item.record.video_id for item in raw_train_records]
            teacher_ids = [item.record.video_id for item in teacher_records]
            if teacher_ids != student_ids:
                raise ValueError(
                    "visual teacher manifest train records must match the student manifest"
                )
            for student, teacher in zip(
                raw_train_records, teacher_records, strict=True
            ):
                if not np.array_equal(
                    student.cache.timestamps, teacher.cache.timestamps
                ):
                    raise ValueError(
                        "visual teacher and student cache timelines differ for "
                        f"{student.record.video_id}"
                    )
        try:
            visual_teacher_predictions = _visual_teacher_predictions(
                settings.visual_teacher_checkpoint,
                teacher_records,
                chunk_frames=settings.inference_chunk_frames,
                device=device,
            )
        finally:
            if teacher_records is not raw_train_records:
                for item in teacher_records:
                    item.cache.release()
    train_records = _compressed_records(raw_train_records)
    val_records = _compressed_records(raw_val_records)
    if not train_records or not val_records:
        raise ValueError("manifest must contain at least one train and one val source video")
    ensure_source_disjoint(
        train_records,
        val_records,
        allow_same_source_temporal=settings.validation_mode == "diagnostic_temporal",
        temporal_guard_seconds=settings.temporal_guard_seconds,
    )

    presence_positive_weight = _presence_positive_weight(train_records)
    boundary_event_positive_weights = _boundary_event_positive_weights(train_records)
    segment_anchor_positive_weight = (
        _segment_anchor_positive_weight(train_records)
        if settings.use_segment_head
        else None
    )
    presence_weight_tensor = torch.tensor(presence_positive_weight, device=device)
    boundary_weight_tensor = torch.from_numpy(boundary_event_positive_weights).to(
        device
    )
    segment_anchor_weight_tensor = (
        torch.tensor(segment_anchor_positive_weight, device=device)
        if segment_anchor_positive_weight is not None
        else None
    )
    feature_mean, feature_std = compute_feature_stats(train_records)  # type: ignore[arg-type]
    train_dataset_type = (
        _TeacherSupervisedWindowDataset
        if visual_teacher_predictions is not None
        else StreamingTimingWindowDataset
    )
    train_dataset_arguments: dict[str, Any] = {}
    if visual_teacher_predictions is not None:
        train_dataset_arguments["visual_teacher_predictions"] = (
            visual_teacher_predictions
        )
    train_dataset = train_dataset_type(
        train_records,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window_frames=settings.window_frames,
        stride_frames=settings.stride_frames,
        max_windows=settings.max_train_windows,
        clean_negative_weight=settings.clean_negative_weight,
        short_segment_weight=settings.short_segment_weight,
        **train_dataset_arguments,
    )
    val_dataset = StreamingTimingWindowDataset(
        val_records,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window_frames=settings.window_frames,
        stride_frames=settings.stride_frames,
        max_windows=settings.max_val_windows,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=0,
    )

    first_cache = train_records[0].cache
    model_config = StreamingModelConfig(
        feature_count=first_cache.features.shape[1],
        token_count=first_cache.tokens.shape[1],
        width=settings.width,
        temporal_layers=settings.temporal_layers,
        recurrent_layers=settings.recurrent_layers,
        dropout=settings.dropout,
        use_byte_branch=settings.use_byte_branch,
        use_segment_head=settings.use_segment_head,
    )
    model = StreamingH264SubtitleModel(model_config).to(device)
    initial_epoch = 0
    teacher_parameter_count = 0
    active_decoder_config: StreamingDecoderConfig | None = None
    active_calibration_metrics: dict[str, float] = {}
    if (
        settings.visual_teacher_checkpoint is not None
        and settings.initial_checkpoint is None
    ):
        teacher_path = settings.visual_teacher_checkpoint.expanduser().resolve()
        teacher_checkpoint = torch.load(
            teacher_path, map_location="cpu", weights_only=False
        )
        if (
            not isinstance(teacher_checkpoint, dict)
            or teacher_checkpoint.get("format") != STREAM_CHECKPOINT_FORMAT
            or teacher_checkpoint.get("version") != STREAM_CHECKPOINT_VERSION
            or teacher_checkpoint.get("feature_version") != FEATURE_VERSION
            or "visual_feature_settings" not in teacher_checkpoint
        ):
            raise ValueError(f"unsupported visual teacher checkpoint: {teacher_path}")
        student_state = model.state_dict()
        compatible_teacher_state = {
            name: value
            for name, value in teacher_checkpoint.get("model", {}).items()
            if name in student_state and student_state[name].shape == value.shape
        }
        if not compatible_teacher_state:
            raise ValueError("visual teacher has no architecture-compatible parameters")
        model.load_state_dict(compatible_teacher_state, strict=False)
        teacher_parameter_count = sum(
            int(value.numel()) for value in compatible_teacher_state.values()
        )
    if settings.initial_checkpoint is not None:
        checkpoint_path = settings.initial_checkpoint.expanduser().resolve()
        initial_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            not isinstance(initial_checkpoint, dict)
            or initial_checkpoint.get("format")
            != COMPRESSED_STREAM_CHECKPOINT_FORMAT
            or initial_checkpoint.get("version")
            != COMPRESSED_STREAM_CHECKPOINT_VERSION
            or initial_checkpoint.get("feature_version") != FEATURE_VERSION
        ):
            raise ValueError(
                f"unsupported initial compressed streaming checkpoint: {checkpoint_path}"
            )
        if dict(initial_checkpoint.get("model_config", {})) != model_config.to_dict():
            raise ValueError(
                "initial compressed checkpoint model config differs from training settings"
            )
        if initial_checkpoint.get("compressed_feature_names") != first_cache.feature_names:
            raise ValueError(
                "initial compressed checkpoint feature schema differs from the manifest"
            )
        model.load_state_dict(initial_checkpoint["model"], strict=True)
        initial_epoch = int(initial_checkpoint.get("epoch", 0))
        if initial_epoch < 0:
            raise ValueError("initial checkpoint epoch must be non-negative")
        active_decoder_config = StreamingDecoderConfig(
            **dict(initial_checkpoint["streaming_decoder_config"])
        )
        active_calibration_metrics = dict(
            initial_checkpoint.get("calibration_metrics", {})
        )

    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    best_selection: tuple | None = None
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_decoder_config: StreamingDecoderConfig | None = None
    last_metrics: dict[str, float] = {}
    for local_epoch in range(1, settings.epochs + 1):
        epoch = initial_epoch + local_epoch
        train_losses = _window_loss(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            presence_positive_weight=presence_weight_tensor,
            boundary_event_positive_weights=boundary_weight_tensor,
            segment_anchor_positive_weight=segment_anchor_weight_tensor,
            segment_boundary_weight=settings.segment_boundary_weight,
            segment_loss_weight=settings.segment_loss_weight,
            negative_weight=settings.negative_weight,
            boundary_event_loss_weight=settings.boundary_event_loss_weight,
            visual_distillation_weight=(
                settings.visual_distillation_weight
                if visual_teacher_predictions is not None
                else 0.0
            ),
        )
        val_losses = _window_loss(
            model,
            val_loader,
            device=device,
            optimizer=None,
            presence_positive_weight=presence_weight_tensor,
            boundary_event_positive_weights=boundary_weight_tensor,
            segment_anchor_positive_weight=segment_anchor_weight_tensor,
            segment_boundary_weight=settings.segment_boundary_weight,
            segment_loss_weight=settings.segment_loss_weight,
            negative_weight=settings.negative_weight,
            boundary_event_loss_weight=settings.boundary_event_loss_weight,
            visual_distillation_weight=0.0,
        )
        if (
            active_decoder_config is None
            or local_epoch % settings.calibration_interval_epochs == 0
        ):
            calibration_predictions = _predict_records(
                model,
                train_records,
                feature_mean=feature_mean,
                feature_std=feature_std,
                chunk_frames=settings.inference_chunk_frames,
                device=device,
            )
            active_decoder_config, active_calibration_metrics = _calibrate_decoder(
                calibration_predictions,
                settings=settings,
            )
        decoder_config = active_decoder_config
        calibration_metrics = active_calibration_metrics
        validation_predictions = _predict_records(
            model,
            val_records,
            feature_mean=feature_mean,
            feature_std=feature_std,
            chunk_frames=settings.inference_chunk_frames,
            device=device,
        )
        validation_metrics = _validation_metrics(
            validation_predictions,
            decoder_config,
            target_recall=settings.target_recall,
        )
        validation_metrics["maximum_boundary_drift_frames"] = (
            _maximum_boundary_drift_frames(validation_predictions, decoder_config)
        )
        last_metrics = {
            **{f"train_{key}": value for key, value in train_losses.items()},
            **{f"val_{key}": value for key, value in val_losses.items()},
            "calibration_interval_recall_iou50": calibration_metrics[
                "interval_recall_iou50"
            ],
            "calibration_interval_precision_iou50": calibration_metrics[
                "interval_precision_iou50"
            ],
            "calibration_recall_target_met": calibration_metrics[
                "recall_target_met"
            ],
            **validation_metrics,
        }
        metric_record = {"epoch": epoch, **last_metrics}
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(metric_record, ensure_ascii=False, allow_nan=False) + "\n"
            )
        checkpoint = _checkpoint(
            model,
            model_config=model_config,
            decoder_config=decoder_config,
            settings=settings,
            train_records=train_records,
            val_records=val_records,
            feature_mean=feature_mean,
            feature_std=feature_std,
            presence_positive_weight=presence_positive_weight,
            boundary_event_positive_weights=boundary_event_positive_weights,
            segment_anchor_positive_weight=segment_anchor_positive_weight,
            calibration_metrics=calibration_metrics,
            epoch=epoch,
            metrics=last_metrics,
        )
        torch.save(checkpoint, output_dir / "last.pt")
        selection = _checkpoint_selection_key(
            validation_metrics,
            validation_loss=val_losses["loss"],
            target_recall=settings.target_recall,
        )
        if best_selection is None or selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            best_metrics = dict(last_metrics)
            best_decoder_config = decoder_config
            torch.save(checkpoint, output_dir / "best.pt")
        print(json.dumps(metric_record, ensure_ascii=False), flush=True)
        if bool(_quality_gate(validation_metrics)["passed"]):
            break

    quality_gate = _quality_gate(best_metrics)
    summary: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "quality_gate": quality_gate,
        "best_streaming_decoder_config": (
            best_decoder_config.to_dict() if best_decoder_config is not None else None
        ),
        "input_domain": "h264_compressed_only",
        "compressed_numeric_feature_count": model_config.feature_count,
        "compressed_byte_token_count": (
            model_config.token_count if model_config.use_byte_branch else 0
        ),
        "visual_feature_input_count": 0,
        "pixel_decode_required_for_inference": False,
        "visual_teacher_checkpoint": (
            str(settings.visual_teacher_checkpoint.expanduser().resolve())
            if settings.visual_teacher_checkpoint is not None
            else None
        ),
        "visual_teacher_manifest": (
            str(settings.visual_teacher_manifest.expanduser().resolve())
            if settings.visual_teacher_manifest is not None
            else None
        ),
        "visual_teacher_parameters_loaded": teacher_parameter_count,
        "visual_teacher_supervision_used": visual_teacher_predictions is not None,
        "visual_distillation_weight": (
            settings.visual_distillation_weight
            if visual_teacher_predictions is not None
            else 0.0
        ),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_sources": len(train_records),
        "val_sources": len(val_records),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "device": str(device),
        "validation_contract": (
            "held_out_by_declared_source_group_and_container_fingerprint"
            if settings.validation_mode == "held_out"
            else "diagnostic_same_source_temporal_not_held_out"
        ),
        "initial_checkpoint_epoch": initial_epoch,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
