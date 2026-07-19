from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import FEATURE_VERSION, STREAM_CHECKPOINT_FORMAT, STREAM_CHECKPOINT_VERSION
from .dataset import compute_feature_stats, ensure_source_disjoint, read_manifest
from .metrics import IntervalMetricSample, aggregate_interval_metrics
from .postprocess import SegmentPrediction
from .stream_dataset import (
    StreamingLoadedRecord,
    StreamingTimingWindowDataset,
    load_streaming_records,
    use_visual_only_features,
)
from .stream_loss import streaming_detection_loss
from .stream_model import StreamingH264SubtitleModel, StreamingModelConfig
from .stream_postprocess import (
    StreamingDecoderConfig,
    decode_stream_anchor_predictions,
    decode_stream_predictions,
)
from .stream_predict import predict_stream_cache


class StreamingTrainSettings(BaseModel):
    manifest: Path
    output_dir: Path
    epochs: int = Field(default=20, gt=0)
    batch_size: int = Field(default=8, gt=0)
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
    use_byte_branch: bool = False
    use_segment_head: bool = True
    segment_boundary_weight: float = Field(default=2.0, ge=0.0)
    segment_loss_weight: float = Field(default=0.25, ge=0.0)
    negative_weight: float = Field(default=1.0, gt=0.0)
    boundary_event_loss_weight: float = Field(default=1.0, ge=0.0)
    clean_negative_weight: float = Field(default=1.0, ge=1.0)
    short_segment_weight: float = Field(default=1.0, ge=1.0)
    max_train_windows: int | None = Field(default=None, gt=0)
    max_val_windows: int | None = Field(default=None, gt=0)
    seed: int = 2026
    device: str = "auto"
    validation_mode: Literal["held_out", "diagnostic_temporal"] = "held_out"
    temporal_guard_seconds: float = Field(default=10.0, ge=0.0)
    inference_chunk_frames: int = Field(default=128, gt=0)
    initial_checkpoint: Path | None = None
    input_domain: Literal["combined", "visual_only"] = "combined"

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.stride_frames > self.window_frames:
            raise ValueError("stride_frames must not exceed window_frames")
        if self.maximum_duration_seconds <= self.minimum_duration_seconds:
            raise ValueError("maximum duration must be greater than minimum duration")
        if self.input_domain == "visual_only" and self.use_byte_branch:
            raise ValueError("visual-only streaming training cannot use byte tokens")
        return self


StreamingPredictions = list[tuple[StreamingLoadedRecord, np.ndarray]]
DecodedRecords = list[tuple[StreamingLoadedRecord, list[SegmentPrediction]]]


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # Some PyTorch builds report MPS availability on hosts whose runtime
        # OS cannot actually create an MPS tensor. Probe once before selecting
        # it so auto mode remains usable on those hosts.
        try:
            torch.empty(1, device="mps")
        except RuntimeError:
            pass
        else:
            return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _presence_positive_weight(records: list[StreamingLoadedRecord]) -> float:
    frame_count = sum(len(item.presence_targets) for item in records)
    positive_mass = sum(
        float(item.presence_targets.sum(dtype=np.float64)) for item in records
    )
    if frame_count <= 0 or positive_mass <= 0.0 or positive_mass >= frame_count:
        raise ValueError("presence targets need positive and negative examples")
    weight = (frame_count - positive_mass) / positive_mass
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("could not derive a finite presence positive weight")
    # Sparse end anchors otherwise dominate the shared representation during
    # the first epochs and turn every frame into a positive proposal.
    return float(min(weight, 50.0))


def _boundary_event_positive_weights(
    records: list[StreamingLoadedRecord],
) -> np.ndarray:
    frame_count = sum(len(item.boundary_event_targets) for item in records)
    positive_mass = sum(
        (item.boundary_event_targets.sum(axis=0, dtype=np.float64) for item in records),
        start=np.zeros((2,), dtype=np.float64),
    )
    if frame_count <= 0 or np.any(positive_mass <= 0.0):
        raise ValueError("both boundary event channels need positive examples")
    weights = (frame_count - positive_mass) / positive_mass
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("could not derive finite boundary event weights")
    return weights.astype(np.float32)


def _segment_anchor_positive_weight(
    records: list[StreamingLoadedRecord],
) -> float:
    frame_count = sum(len(item.segment_anchor_targets) for item in records)
    positive_count = sum(
        float(item.segment_anchor_targets[:, 0].sum(dtype=np.float64))
        for item in records
    )
    if frame_count <= 0 or positive_count <= 0.0 or positive_count >= frame_count:
        raise ValueError("segment anchor targets need positive and negative examples")
    weight = (frame_count - positive_count) / positive_count
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("could not derive a finite segment anchor positive weight")
    # The focal factor already emphasizes rare anchors.  Capping the class
    # weight prevents a sparse end target from making clean frames look like
    # candidate anchors during early training.
    return float(min(weight, 50.0))


def _window_loss(
    model: StreamingH264SubtitleModel,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: AdamW | None,
    presence_positive_weight: torch.Tensor,
    boundary_event_positive_weights: torch.Tensor,
    segment_anchor_positive_weight: torch.Tensor | None = None,
    segment_boundary_weight: float = 2.0,
    segment_loss_weight: float = 1.0,
    negative_weight: float = 1.0,
    boundary_event_loss_weight: float = 1.0,
    visual_distillation_weight: float = 0.0,
) -> dict[str, float]:
    if visual_distillation_weight < 0.0:
        raise ValueError("visual_distillation_weight must be non-negative")
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "presence_loss": 0.0,
        "start_event_loss": 0.0,
        "end_event_loss": 0.0,
        "segment_anchor_loss": 0.0,
        "segment_start_loss": 0.0,
        "segment_end_loss": 0.0,
        "segment_anchor_iou_loss": 0.0,
    }
    if visual_distillation_weight > 0.0:
        totals["visual_distillation_loss"] = 0.0
    total_frames = 0.0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in tqdm(
            loader, desc="stream-train" if training else "stream-val", leave=False
        ):
            features = batch["features"].to(device)
            tokens = batch["tokens"]
            if model.config.use_byte_branch:
                tokens = tokens.to(device)
            presence_targets = batch["presence_targets"].to(device)
            boundary_targets = batch["boundary_event_targets"].to(device)
            segment_anchor_targets = batch["segment_anchor_targets"].to(device)
            mask = batch["mask"].to(device)
            loss_weights = batch["loss_weights"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(features, tokens)
            loss, components = streaming_detection_loss(
                output,
                presence_targets,
                boundary_targets,
                mask,
                presence_positive_weight,
                boundary_event_positive_weights,
                segment_anchor_targets=(
                    segment_anchor_targets if model.config.use_segment_head else None
                ),
                segment_anchor_positive_weight=(
                    segment_anchor_positive_weight
                    if model.config.use_segment_head
                    else None
                ),
                segment_boundary_weight=segment_boundary_weight,
                segment_loss_weight=segment_loss_weight,
                negative_weight=negative_weight,
                boundary_event_loss_weight=boundary_event_loss_weight,
                sample_weight=loss_weights,
            )
            if visual_distillation_weight > 0.0:
                if "visual_teacher_probabilities" not in batch:
                    raise ValueError(
                        "visual teacher probabilities are required for distillation"
                    )
                teacher = batch["visual_teacher_probabilities"].to(device)
                if teacher.shape != (*presence_targets.shape, 3):
                    raise ValueError(
                        "visual teacher probabilities must have shape [batch,time,3]"
                    )
                student_logits = torch.cat(
                    (output.presence_logits.unsqueeze(-1), output.boundary_event_logits),
                    dim=-1,
                )
                distillation_per_frame = torch.nn.functional.binary_cross_entropy_with_logits(
                    student_logits,
                    teacher,
                    reduction="none",
                ).mean(dim=-1)
                distillation_denominator = mask.sum().clamp_min(1.0)
                distillation_loss = (
                    distillation_per_frame * mask
                ).sum() / distillation_denominator
                loss = loss + visual_distillation_weight * distillation_loss
                components["visual_distillation_loss"] = float(
                    distillation_loss.detach()
                )
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            valid_frames = float(mask.sum().detach())
            if valid_frames <= 0.0:
                continue
            totals["loss"] += float(loss.detach()) * valid_frames
            for key, value in components.items():
                totals[key] += value * valid_frames
            total_frames += valid_frames
    if total_frames <= 0.0:
        raise ValueError("streaming timing dataset contains no valid frames")
    return {key: value / total_frames for key, value in totals.items()}


def _predict_records(
    model: StreamingH264SubtitleModel,
    records: list[StreamingLoadedRecord],
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    chunk_frames: int,
    device: torch.device,
) -> StreamingPredictions:
    predictions: StreamingPredictions = []
    for item in records:
        predictions.append(
            (
                item,
                predict_stream_cache(
                    model,
                    item.cache,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    chunk_frames=chunk_frames,
                    device=device,
                    include_segments=model.config.use_segment_head,
                ),
            )
        )
    return predictions


def _frame_tolerance(item: StreamingLoadedRecord) -> float:
    timestamps = np.asarray(item.cache.timestamps, dtype=np.float64)
    positive_steps = np.diff(timestamps)
    positive_steps = positive_steps[positive_steps > 0.0]
    return float(np.median(positive_steps)) if positive_steps.size else 1.0 / 30.0


def _decode_records(
    predictions: StreamingPredictions,
    config: StreamingDecoderConfig,
) -> DecodedRecords:
    return [
        (
            item,
            (
                decode_stream_anchor_predictions(
                    probabilities,
                    np.asarray(item.cache.timestamps, dtype=np.float64),
                    np.asarray(item.cache.durations, dtype=np.float64),
                    config,
                )
                if probabilities.shape[1] == 6
                else decode_stream_predictions(
                    probabilities,
                    np.asarray(item.cache.timestamps, dtype=np.float64),
                    np.asarray(item.cache.durations, dtype=np.float64),
                    config,
                )
            ),
        )
        for item, probabilities in predictions
    ]


def _metric_samples(
    decoded: DecodedRecords,
    *,
    score_threshold: float,
) -> list[IntervalMetricSample]:
    samples: list[IntervalMetricSample] = []
    for item, segments in decoded:
        cache_start, cache_end = item.cache.coverage_range_seconds
        samples.append(
            IntervalMetricSample(
                predicted=[
                    segment.to_interval()
                    for segment in segments
                    if segment.confidence >= score_threshold
                ],
                target=item.intervals,
                video_duration_seconds=cache_end - cache_start,
                frame_tolerance_seconds=_frame_tolerance(item),
            )
        )
    return samples


def _decoded_metrics(
    decoded: DecodedRecords,
    config: StreamingDecoderConfig,
    *,
    target_recall: float,
    include_roles: bool,
) -> dict[str, float]:
    metrics = aggregate_interval_metrics(
        _metric_samples(decoded, score_threshold=config.score_threshold)
    )
    if include_roles:
        role_prefixes = {
            "subtitle_signal": "signal",
            "clean_control": "clean",
            "source_timing": "source_timing",
        }
        for role, prefix in role_prefixes.items():
            role_decoded = [
                item
                for item in decoded
                if item[0].record.signal_validation_role == role
            ]
            if role_decoded:
                role_metrics = aggregate_interval_metrics(
                    _metric_samples(
                        role_decoded,
                        score_threshold=config.score_threshold,
                    )
                )
                metrics.update(
                    {f"{prefix}_{key}": value for key, value in role_metrics.items()}
                )
    metrics.update(
        {
            "score_threshold": config.score_threshold,
            "presence_on_threshold": config.presence_on_threshold,
            "presence_off_threshold": config.presence_off_threshold,
            "boundary_event_threshold": config.boundary_event_threshold,
            "confirmation_samples": float(config.confirmation_samples),
            "anchor_score_threshold": config.anchor_score_threshold,
            "anchor_nms_iou_threshold": config.anchor_nms_iou_threshold,
            "anchor_peak_radius_frames": float(config.anchor_peak_radius_frames),
            "anchor_end_event_threshold": config.anchor_end_event_threshold,
            "minimum_anchor_gap_seconds": config.minimum_anchor_gap_seconds,
            "anchor_start_event_threshold": config.anchor_start_event_threshold,
            "anchor_start_refinement_seconds": config.anchor_start_refinement_seconds,
            "anchor_end_refinement_seconds": config.anchor_end_refinement_seconds,
            "anchor_pair_start_events": float(config.anchor_pair_start_events),
            "causal_event_pairing": float(config.causal_event_pairing),
            "start_event_threshold": config.start_event_threshold,
            "end_event_threshold": config.end_event_threshold,
            "event_confirmation_samples": float(config.event_confirmation_samples),
            "event_recovery_threshold": config.event_recovery_threshold,
            "event_recovery_samples": float(config.event_recovery_samples),
            "strong_end_event_threshold": config.strong_end_event_threshold,
            "minimum_start_gap_seconds": config.minimum_start_gap_seconds,
            "end_refinement_frames": float(config.end_refinement_frames),
            "end_refinement_event_threshold": config.end_refinement_event_threshold,
            "recall_target": target_recall,
            "recall_target_met": float(
                metrics["interval_recall_iou50"] + 1e-12 >= target_recall
                and metrics["segment_recall_1frame"] + 1e-12 >= target_recall
            ),
        }
    )
    return metrics


def _calibration_key(metrics: dict[str, float], *, target_recall: float) -> tuple:
    recall = metrics["interval_recall_iou50"]
    strict_recall = metrics["segment_recall_1frame"]
    target_met = (
        recall + 1e-12 >= target_recall
        and strict_recall + 1e-12 >= target_recall
    )
    if target_met:
        return (
            1,
            metrics["segment_f1_1frame"],
            metrics["interval_f1_iou50"],
            metrics["segment_precision_1frame"],
            metrics["matched_interval_mean_iou"],
            -metrics["start_error_p95_seconds"],
            -metrics["end_error_p95_seconds"],
            -metrics["false_intervals_per_minute"],
            -metrics["anchor_score_threshold"],
        )
    return (
        0,
        strict_recall,
        recall,
        metrics["segment_f1_1frame"],
        metrics["segment_recall_100ms"],
        metrics["matched_interval_mean_iou"],
        metrics["interval_precision_iou50"],
        -metrics["false_intervals_per_minute"],
        -metrics["anchor_score_threshold"],
    )


def _calibrate_decoder(
    predictions: StreamingPredictions,
    *,
    settings: StreamingTrainSettings,
) -> tuple[StreamingDecoderConfig, dict[str, float]]:
    best: tuple[tuple, StreamingDecoderConfig, dict[str, float]] | None = None
    if predictions and predictions[0][1].shape[1] == 6:
        # Keep the grid deliberately small because decoding is stateful, but
        # include the causal start-event pairing that is useful when the
        # regressed start offset is noisy on a cold stream.
        for anchor_threshold in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85):
            for end_event_threshold, minimum_anchor_gap in (
                (0.0, 0.0),
                (0.50, 0.50),
                (0.70, 0.50),
            ):
                for nms_threshold in (0.70, 0.90, 0.98):
                    for peak_radius in (0, 1, 2):
                        pair_variants = (
                            (False, 0.0, 0.0),
                            (True, 0.30, 1.0),
                            (True, 0.50, 1.5),
                            (True, 0.70, 2.0),
                        )
                        for pair_starts, start_threshold, refinement in pair_variants:
                            config = StreamingDecoderConfig(
                                score_threshold=anchor_threshold,
                                anchor_score_threshold=anchor_threshold,
                                anchor_nms_iou_threshold=nms_threshold,
                                anchor_peak_radius_frames=peak_radius,
                                anchor_end_event_threshold=end_event_threshold,
                                minimum_anchor_gap_seconds=minimum_anchor_gap,
                                anchor_start_event_threshold=start_threshold,
                                anchor_start_refinement_seconds=refinement,
                                anchor_pair_start_events=pair_starts,
                                minimum_duration_seconds=settings.minimum_duration_seconds,
                                maximum_duration_seconds=settings.maximum_duration_seconds,
                            )
                            metrics = _decoded_metrics(
                                _decode_records(predictions, config),
                                config,
                                target_recall=settings.target_recall,
                                include_roles=False,
                            )
                            key = _calibration_key(
                                metrics,
                                target_recall=settings.target_recall,
                            )
                            if best is None or key > best[0]:
                                best = (key, config, metrics)
        if best is None:
            raise ValueError("streaming anchor calibration found no candidates")
        return best[1], best[2]

    observed_durations = [
        interval.end_seconds - interval.start_seconds
        for item, _ in predictions
        for interval in item.intervals
    ]
    paired_minimum_duration = max(
        settings.minimum_duration_seconds,
        min(0.50, min(observed_durations, default=0.50)),
    )
    for start_threshold in (0.80, 0.84, 0.86, 0.88):
        for end_threshold in (0.15, 0.20):
            for event_confirmation, recovery_samples in (
                (4, 2),
                (4, 3),
                (5, 3),
            ):
                for score_threshold in (0.0, 0.25):
                    config = StreamingDecoderConfig(
                        score_threshold=score_threshold,
                        presence_on_threshold=0.50,
                        presence_off_threshold=0.35,
                        boundary_event_threshold=0.50,
                        minimum_duration_seconds=paired_minimum_duration,
                        maximum_duration_seconds=settings.maximum_duration_seconds,
                        causal_event_pairing=True,
                        start_event_threshold=start_threshold,
                        end_event_threshold=end_threshold,
                        event_confirmation_samples=event_confirmation,
                        event_recovery_threshold=0.60,
                        event_recovery_samples=recovery_samples,
                        strong_end_event_threshold=0.50,
                        minimum_start_gap_seconds=0.30,
                        end_refinement_frames=1,
                        end_refinement_event_threshold=0.50,
                    )
                    metrics = _decoded_metrics(
                        _decode_records(predictions, config),
                        config,
                        target_recall=settings.target_recall,
                        include_roles=False,
                    )
                    key = _calibration_key(
                        metrics,
                        target_recall=settings.target_recall,
                    )
                    if best is None or key > best[0]:
                        best = (key, config, metrics)

    presence_thresholds = (
        (0.25, 0.10),
        (0.35, 0.15),
        (0.35, 0.20),
        (0.50, 0.20),
        (0.50, 0.35),
        (0.65, 0.35),
        (0.65, 0.50),
    )
    for presence_on, presence_off in presence_thresholds:
        for event_threshold in (0.25, 0.35, 0.50, 0.65, 0.80):
            for confirmation_samples in (1, 2):
                for score_threshold in (0.0, 0.10, 0.25):
                    config = StreamingDecoderConfig(
                        score_threshold=score_threshold,
                        presence_on_threshold=presence_on,
                        presence_off_threshold=presence_off,
                        boundary_event_threshold=event_threshold,
                        confirmation_samples=confirmation_samples,
                        minimum_duration_seconds=settings.minimum_duration_seconds,
                        maximum_duration_seconds=settings.maximum_duration_seconds,
                    )
                    decoded = _decode_records(predictions, config)
                    metrics = _decoded_metrics(
                        decoded,
                        config,
                        target_recall=settings.target_recall,
                        include_roles=False,
                    )
                    key = _calibration_key(
                        metrics,
                        target_recall=settings.target_recall,
                    )
                    if best is None or key > best[0]:
                        best = (key, config, metrics)
    if best is None:
        raise ValueError("streaming decoder calibration found no candidates")
    return best[1], best[2]


def _validation_metrics(
    predictions: StreamingPredictions,
    config: StreamingDecoderConfig,
    *,
    target_recall: float,
) -> dict[str, float]:
    return _decoded_metrics(
        _decode_records(predictions, config),
        config,
        target_recall=target_recall,
        include_roles=True,
    )


def _checkpoint_selection_key(
    metrics: dict[str, float],
    *,
    validation_loss: float,
    target_recall: float,
) -> tuple:
    target_met = (
        metrics["interval_recall_iou50"] + 1e-12 >= target_recall
        and metrics["segment_recall_1frame"] + 1e-12 >= target_recall
    )
    return (
        float(target_met),
        metrics["segment_recall_1frame"],
        metrics["interval_recall_iou50"],
        metrics["segment_f1_1frame"],
        metrics["interval_f1_iou50"],
        metrics["segment_precision_1frame"],
        metrics["matched_interval_mean_iou"],
        -metrics["start_error_p95_seconds"],
        -metrics["end_error_p95_seconds"],
        -metrics["false_intervals_per_minute"],
        -validation_loss,
    )


def _checkpoint(
    model: StreamingH264SubtitleModel,
    *,
    model_config: StreamingModelConfig,
    decoder_config: StreamingDecoderConfig,
    settings: StreamingTrainSettings,
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
) -> dict:
    first_cache = train_records[0].cache
    training_provenance = {
        "manifest": str(settings.manifest.expanduser().resolve()),
        "settings": settings.model_dump(mode="json"),
        "train_video_ids": [item.record.video_id for item in train_records],
        "val_video_ids": [item.record.video_id for item in val_records],
        "train_source_groups": sorted(
            {item.record.source_group for item in train_records}
        ),
        "val_source_groups": sorted({item.record.source_group for item in val_records}),
        "causal_window_context_frames": settings.window_frames - settings.stride_frames,
        "calibration_split": "train",
    }
    return {
        "format": STREAM_CHECKPOINT_FORMAT,
        "version": STREAM_CHECKPOINT_VERSION,
        "feature_version": FEATURE_VERSION,
        "model": model.state_dict(),
        "model_config": model_config.to_dict(),
        "model_output_contract": (
            "causal_streaming_presence_events_anchor"
            if model_config.use_segment_head
            else "causal_streaming_presence_events"
        ),
        "feature_names": first_cache.feature_names,
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "feature_settings": dict(first_cache.meta["feature_settings"]),
        "visual_feature_settings": dict(first_cache.visual_feature_settings or {}),
        "input_domain": settings.input_domain,
        "pixel_decode_required": settings.input_domain == "visual_only",
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
        "training_provenance": training_provenance,
        "calibration_metrics": calibration_metrics,
        "epoch": epoch,
        "metrics": metrics,
    }


def _prepare_output_dir(path: Path) -> tuple[Path, Path]:
    output_dir = path.expanduser().resolve()
    collision_names = ("metrics.jsonl", "last.pt", "best.pt", "summary.json")
    collisions = [name for name in collision_names if (output_dir / name).exists()]
    if collisions:
        raise FileExistsError(
            "training output already contains a run and resume is not implemented: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, output_dir / "metrics.jsonl"


def train_streaming(settings: StreamingTrainSettings) -> dict:
    """Train and calibrate the independent causal streaming timing model."""

    output_dir, metrics_path = _prepare_output_dir(settings.output_dir)
    _seed_everything(settings.seed)
    device = _select_device(settings.device)
    train_records = load_streaming_records(
        read_manifest(settings.manifest, split="train"),
        boundary_event_sigma_seconds=settings.boundary_event_sigma_seconds,
    )
    val_records = load_streaming_records(
        read_manifest(settings.manifest, split="val"),
        boundary_event_sigma_seconds=settings.boundary_event_sigma_seconds,
    )
    if not train_records or not val_records:
        raise ValueError(
            "manifest must contain at least one train and one val source video"
        )
    ensure_source_disjoint(
        train_records,
        val_records,
        allow_same_source_temporal=settings.validation_mode == "diagnostic_temporal",
        temporal_guard_seconds=settings.temporal_guard_seconds,
    )
    if settings.input_domain == "visual_only":
        train_records = use_visual_only_features(train_records)
        val_records = use_visual_only_features(val_records)

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
    feature_mean, feature_std = compute_feature_stats(train_records)
    train_dataset = StreamingTimingWindowDataset(
        train_records,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window_frames=settings.window_frames,
        stride_frames=settings.stride_frames,
        max_windows=settings.max_train_windows,
        clean_negative_weight=settings.clean_negative_weight,
        short_segment_weight=settings.short_segment_weight,
    )
    val_dataset = StreamingTimingWindowDataset(
        val_records,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window_frames=settings.window_frames,
        stride_frames=settings.stride_frames,
        max_windows=settings.max_val_windows,
        clean_negative_weight=1.0,
        short_segment_weight=1.0,
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
    if settings.initial_checkpoint is not None:
        checkpoint_path = settings.initial_checkpoint.expanduser().resolve()
        initial_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            not isinstance(initial_checkpoint, dict)
            or initial_checkpoint.get("format") != STREAM_CHECKPOINT_FORMAT
            or initial_checkpoint.get("version") != STREAM_CHECKPOINT_VERSION
            or initial_checkpoint.get("feature_version") != FEATURE_VERSION
        ):
            raise ValueError(f"unsupported initial streaming checkpoint: {checkpoint_path}")
        initial_model_config = dict(initial_checkpoint.get("model_config", {}))
        # Checkpoints written before the optional anchor head did not record
        # this flag.  A missing value is equivalent to the disabled head.
        if "use_segment_head" not in initial_model_config:
            initial_model_config["use_segment_head"] = False
        if initial_model_config != model_config.to_dict():
            raise ValueError("initial checkpoint model config differs from training settings")
        if initial_checkpoint.get("feature_names") != first_cache.feature_names:
            raise ValueError("initial checkpoint feature schema differs from the manifest")
        model.load_state_dict(initial_checkpoint["model"], strict=True)
        initial_epoch = int(initial_checkpoint.get("epoch", 0))
        if initial_epoch < 0:
            raise ValueError("initial checkpoint epoch must be non-negative")
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
        )
        calibration_predictions = _predict_records(
            model,
            train_records,
            feature_mean=feature_mean,
            feature_std=feature_std,
            chunk_frames=settings.inference_chunk_frames,
            device=device,
        )
        decoder_config, calibration_metrics = _calibrate_decoder(
            calibration_predictions,
            settings=settings,
        )
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
        last_metrics = {
            **{f"train_{key}": value for key, value in train_losses.items()},
            **{f"val_{key}": value for key, value in val_losses.items()},
            "calibration_interval_recall_iou50": calibration_metrics[
                "interval_recall_iou50"
            ],
            "calibration_interval_precision_iou50": calibration_metrics[
                "interval_precision_iou50"
            ],
            "calibration_recall_target_met": calibration_metrics["recall_target_met"],
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
        print(json.dumps(metric_record, ensure_ascii=False))

    summary = {
        "best_epoch": best_epoch,
        "recall_target": settings.target_recall,
        "best_recall_target_met": bool(best_metrics.get("recall_target_met", 0.0)),
        "best_interval_recall_iou50": best_metrics.get("interval_recall_iou50", 0.0),
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "best_streaming_decoder_config": (
            best_decoder_config.to_dict() if best_decoder_config is not None else None
        ),
        "presence_positive_weight": presence_positive_weight,
        "boundary_event_positive_weights": boundary_event_positive_weights.tolist(),
        "segment_anchor_positive_weight": segment_anchor_positive_weight,
        "use_byte_branch": settings.use_byte_branch,
        "use_segment_head": settings.use_segment_head,
        "input_domain": settings.input_domain,
        "numeric_feature_count": model_config.feature_count,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_sources": len(train_records),
        "val_sources": len(val_records),
        "train_source_groups": len(
            {item.record.source_group for item in train_records}
        ),
        "val_source_groups": len({item.record.source_group for item in val_records}),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "device": str(device),
        "model_output_contract": (
            "causal_streaming_presence_events_anchor"
            if settings.use_segment_head
            else "causal_streaming_presence_events"
        ),
        "inference_chunk_frames": settings.inference_chunk_frames,
        "visual_pixel_decode": first_cache.visual_feature_settings is not None,
        "validation_contract": (
            "held_out_by_declared_source_group_and_container_fingerprint"
            if settings.validation_mode == "held_out"
            else "diagnostic_same_source_temporal_not_held_out"
        ),
        "streaming_decoder_calibration_split": "train",
        "initial_checkpoint_epoch": initial_epoch,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
