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

from . import CHECKPOINT_FORMAT, CHECKPOINT_VERSION, FEATURE_VERSION
from .dataset import (
    LoadedRecord,
    TimingWindowDataset,
    compute_feature_stats,
    ensure_source_disjoint,
    load_records,
    read_manifest,
)
from .loss import segment_detection_loss
from .metrics import IntervalMetricSample, aggregate_interval_metrics
from .model import H264SubtitleSegmentModel, ModelConfig
from .postprocess import SegmentPrediction, SegmentSelectionConfig, select_segments
from .predict import predict_cache


class TrainSettings(BaseModel):
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
    temporal_layers: int = Field(default=7, ge=1, le=9)
    recurrent_layers: int = Field(default=1, ge=1, le=3)
    dropout: float = Field(default=0.10, ge=0.0, lt=1.0)
    use_byte_branch: bool = False
    num_workers: int = Field(default=0, ge=0, le=0)
    max_train_windows: int | None = Field(default=None, gt=0)
    max_val_windows: int | None = Field(default=None, gt=0)
    seed: int = 2026
    device: str = "auto"
    validation_mode: Literal["held_out", "diagnostic_temporal"] = "held_out"
    temporal_guard_seconds: float = Field(default=10.0, ge=0.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.stride_frames > self.window_frames:
            raise ValueError("stride_frames must not exceed window_frames")
        if self.maximum_duration_seconds <= self.minimum_duration_seconds:
            raise ValueError("maximum duration must be greater than minimum duration")
        return self


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _proposal_positive_weight(records: list[LoadedRecord]) -> float:
    frame_count = sum(len(item.segment_targets) for item in records)
    positive_mass = sum(
        float(item.segment_targets[:, 0].sum(dtype=np.float64)) for item in records
    )
    if frame_count <= 0 or positive_mass <= 0.0 or positive_mass >= frame_count:
        raise ValueError("segment proposal targets need positive and negative examples")
    weight = (frame_count - positive_mass) / positive_mass
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError("could not derive a finite proposal positive weight")
    return float(weight)


def _boundary_event_positive_weights(records: list[LoadedRecord]) -> np.ndarray:
    frame_count = sum(len(item.boundary_event_targets) for item in records)
    positive_mass = sum(
        (
            item.boundary_event_targets.sum(axis=0, dtype=np.float64)
            for item in records
        ),
        start=np.zeros((2,), dtype=np.float64),
    )
    if frame_count <= 0 or np.any(positive_mass <= 0.0):
        raise ValueError("both boundary event channels need positive examples")
    weights = (frame_count - positive_mass) / positive_mass
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("could not derive finite boundary event weights")
    return weights.astype(np.float32)


def _window_loss(
    model: H264SubtitleSegmentModel,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: AdamW | None,
    proposal_positive_weight: torch.Tensor,
    boundary_event_positive_weights: torch.Tensor,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "proposal_loss": 0.0,
        "start_boundary_loss": 0.0,
        "end_boundary_loss": 0.0,
        "temporal_iou_loss": 0.0,
        "start_event_loss": 0.0,
        "end_event_loss": 0.0,
    }
    total_frames = 0.0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in tqdm(loader, desc="train" if training else "val", leave=False):
            features = batch["features"].to(device)
            tokens = batch["tokens"]
            if model.config.use_byte_branch:
                tokens = tokens.to(device)
            segment_targets = batch["segment_targets"].to(device)
            boundary_event_targets = batch["boundary_event_targets"].to(device)
            regression_mask = batch["regression_mask"].to(device)
            mask = batch["mask"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(features, tokens)
            loss, components = segment_detection_loss(
                output,
                segment_targets,
                boundary_event_targets,
                mask,
                regression_mask,
                proposal_positive_weight=proposal_positive_weight,
                boundary_event_positive_weights=boundary_event_positive_weights,
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
        raise ValueError("timing dataset contains no valid frames")
    return {key: value / total_frames for key, value in totals.items()}


def _predict_records(
    model: H264SubtitleSegmentModel,
    records: list[LoadedRecord],
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    settings: TrainSettings,
    device: torch.device,
) -> list[tuple[LoadedRecord, np.ndarray]]:
    predictions: list[tuple[LoadedRecord, np.ndarray]] = []
    for item in records:
        proposals = predict_cache(
            model,
            item.cache,
            feature_mean=feature_mean,
            feature_std=feature_std,
            window_frames=settings.window_frames,
            hop_frames=settings.stride_frames,
            batch_size=settings.batch_size,
            device=device,
        )
        predictions.append((item, proposals))
    return predictions


def _frame_tolerance(item: LoadedRecord) -> float:
    timestamps = np.asarray(item.cache.timestamps)
    positive_steps = np.diff(timestamps)
    positive_steps = positive_steps[positive_steps > 0.0]
    return float(np.median(positive_steps)) if positive_steps.size else 1.0 / 30.0


def _decoded_samples(
    predictions: list[tuple[LoadedRecord, np.ndarray]],
    config: SegmentSelectionConfig,
) -> list[IntervalMetricSample]:
    samples: list[IntervalMetricSample] = []
    for item, proposals in predictions:
        predicted = [
            segment.to_interval()
            for segment in select_segments(
                proposals,
                np.asarray(item.cache.timestamps),
                config=config,
            )
        ]
        cache_start, cache_end = item.cache.coverage_range_seconds
        samples.append(
            IntervalMetricSample(
                predicted=predicted,
                target=item.intervals,
                video_duration_seconds=cache_end - cache_start,
                frame_tolerance_seconds=_frame_tolerance(item),
            )
        )
    return samples


def _held_out_metrics(
    predictions: list[tuple[LoadedRecord, np.ndarray]],
    config: SegmentSelectionConfig,
    *,
    target_recall: float,
) -> dict[str, float]:
    metrics = aggregate_interval_metrics(_decoded_samples(predictions, config))
    role_prefixes = {
        "subtitle_signal": "signal",
        "clean_control": "clean",
        "source_timing": "source_timing",
    }
    for role, prefix in role_prefixes.items():
        role_predictions = [
            item
            for item in predictions
            if item[0].record.signal_validation_role == role
        ]
        if role_predictions:
            role_metrics = aggregate_interval_metrics(
                _decoded_samples(role_predictions, config)
            )
            metrics.update(
                {f"{prefix}_{key}": value for key, value in role_metrics.items()}
            )
    metrics.update(
        {
            "score_threshold": config.score_threshold,
            "nms_iou_threshold": config.nms_iou_threshold,
            "boundary_event_threshold": config.boundary_event_threshold,
            "start_boundary_refinement_seconds": (
                config.start_boundary_refinement_seconds
            ),
            "end_boundary_refinement_seconds": config.end_boundary_refinement_seconds,
            "end_event_relative_threshold": config.end_event_relative_threshold,
            "boundary_events_required": float(config.require_boundary_events),
            "recall_target": target_recall,
            "recall_target_met": float(
                metrics["interval_recall_iou50"] + 1e-12 >= target_recall
            ),
        }
    )
    return metrics


def _score_thresholds() -> list[float]:
    return sorted(
        {
            0.0,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        }
    )


def _calibration_key(metrics: dict[str, float], *, target_recall: float) -> tuple:
    recall = metrics["interval_recall_iou50"]
    target_met = recall + 1e-12 >= target_recall
    if target_met:
        return (
            1,
            metrics["segment_recall_1frame"],
            metrics["segment_recall_100ms"],
            metrics["matched_interval_mean_iou"],
            -metrics["start_error_p95_seconds"],
            -metrics["end_error_p95_seconds"],
            metrics["interval_precision_iou50"],
            -metrics["false_intervals_per_minute"],
            metrics["score_threshold"],
        )
    return (
        0,
        recall,
        metrics["segment_recall_1frame"],
        metrics["segment_recall_100ms"],
        metrics["matched_interval_mean_iou"],
        metrics["interval_precision_iou50"],
        -metrics["false_intervals_per_minute"],
        metrics["score_threshold"],
    )


def _boundary_calibration_key(
    metrics: dict[str, float], *, target_recall: float
) -> tuple:
    recall = metrics["interval_recall_iou50"]
    return (
        float(recall + 1e-12 >= target_recall),
        recall,
        metrics["segment_recall_1frame"],
        metrics["segment_recall_100ms"],
        -metrics["boundary_event_threshold"],
        metrics["matched_interval_mean_iou"],
        -metrics["start_error_p95_seconds"],
        -metrics["end_error_p95_seconds"],
        metrics["interval_precision_iou50"],
        -metrics["false_intervals_per_minute"],
    )


def _metrics_for_selected_segments(
    selected_by_record: list[tuple[LoadedRecord, list[SegmentPrediction]]],
    *,
    config: SegmentSelectionConfig,
    target_recall: float,
) -> dict[str, float]:
    samples: list[IntervalMetricSample] = []
    for item, selected in selected_by_record:
        cache_start, cache_end = item.cache.coverage_range_seconds
        samples.append(
            IntervalMetricSample(
                predicted=[
                    segment.to_interval()
                    for segment in selected
                    if segment.confidence >= config.score_threshold
                ],
                target=item.intervals,
                video_duration_seconds=cache_end - cache_start,
                frame_tolerance_seconds=_frame_tolerance(item),
            )
        )
    metrics = aggregate_interval_metrics(samples)
    metrics.update(
        {
            "score_threshold": config.score_threshold,
            "nms_iou_threshold": config.nms_iou_threshold,
            "boundary_event_threshold": config.boundary_event_threshold,
            "start_boundary_refinement_seconds": (
                config.start_boundary_refinement_seconds
            ),
            "end_boundary_refinement_seconds": config.end_boundary_refinement_seconds,
            "end_event_relative_threshold": config.end_event_relative_threshold,
            "boundary_events_required": float(config.require_boundary_events),
            "recall_target": target_recall,
            "recall_target_met": float(
                metrics["interval_recall_iou50"] + 1e-12 >= target_recall
            ),
        }
    )
    return metrics


def _calibrate_segment_selection(
    predictions: list[tuple[LoadedRecord, np.ndarray]],
    *,
    settings: TrainSettings,
) -> tuple[SegmentSelectionConfig, dict[str, float]]:
    direct_best: tuple[tuple, SegmentSelectionConfig, dict[str, float]] | None = None
    for nms_iou_threshold in (0.30, 0.50, 0.70, 0.85, 0.95):
        base_config = SegmentSelectionConfig(
            score_threshold=0.0,
            nms_iou_threshold=nms_iou_threshold,
            minimum_duration_seconds=settings.minimum_duration_seconds,
            maximum_duration_seconds=settings.maximum_duration_seconds,
            start_boundary_refinement_seconds=0.0,
            end_boundary_refinement_seconds=0.0,
            require_boundary_events=False,
        )
        selected_by_record = [
            (
                item,
                select_segments(
                    proposals,
                    np.asarray(item.cache.timestamps),
                    config=base_config,
                ),
            )
            for item, proposals in predictions
        ]
        for score_threshold in _score_thresholds():
            config = SegmentSelectionConfig(
                score_threshold=score_threshold,
                nms_iou_threshold=nms_iou_threshold,
                minimum_duration_seconds=settings.minimum_duration_seconds,
                maximum_duration_seconds=settings.maximum_duration_seconds,
                start_boundary_refinement_seconds=0.0,
                end_boundary_refinement_seconds=0.0,
                require_boundary_events=False,
            )
            metrics = _metrics_for_selected_segments(
                selected_by_record,
                config=config,
                target_recall=settings.target_recall,
            )
            key = _calibration_key(metrics, target_recall=settings.target_recall)
            if direct_best is None or key > direct_best[0]:
                direct_best = (key, config, metrics)
    if direct_best is None:
        raise ValueError("segment selection calibration found no candidates")

    direct_config = direct_best[1]
    direct_metrics = direct_best[2]
    best: tuple[tuple, SegmentSelectionConfig, dict[str, float]] = (
        _boundary_calibration_key(
            direct_metrics,
            target_recall=settings.target_recall,
        ),
        direct_config,
        direct_metrics,
    )
    for end_refinement_seconds in (0.40, 0.60, 0.80, 1.00, 1.20, 1.50):
        for end_relative_threshold in (0.70, 0.80, 0.90):
            for require_boundary_events in (False, True):
                config = SegmentSelectionConfig(
                    score_threshold=direct_config.score_threshold,
                    nms_iou_threshold=direct_config.nms_iou_threshold,
                    minimum_duration_seconds=direct_config.minimum_duration_seconds,
                    maximum_duration_seconds=direct_config.maximum_duration_seconds,
                    peak_radius_frames=direct_config.peak_radius_frames,
                    boundary_event_threshold=0.20,
                    start_boundary_refinement_seconds=0.60,
                    end_boundary_refinement_seconds=end_refinement_seconds,
                    end_event_relative_threshold=end_relative_threshold,
                    require_boundary_events=require_boundary_events,
                )
                selected_by_record = [
                    (
                        item,
                        select_segments(
                            proposals,
                            np.asarray(item.cache.timestamps),
                            config=config,
                        ),
                    )
                    for item, proposals in predictions
                ]
                metrics = _metrics_for_selected_segments(
                    selected_by_record,
                    config=config,
                    target_recall=settings.target_recall,
                )
                key = _boundary_calibration_key(
                    metrics,
                    target_recall=settings.target_recall,
                )
                if key > best[0]:
                    best = (key, config, metrics)
    return best[1], best[2]


def _checkpoint_selection_key(
    metrics: dict[str, float], *, validation_loss: float, target_recall: float
) -> tuple:
    return (
        float(metrics["interval_recall_iou50"] + 1e-12 >= target_recall),
        metrics["interval_recall_iou50"],
        metrics["segment_recall_1frame"],
        metrics["segment_recall_100ms"],
        metrics["matched_interval_mean_iou"],
        -metrics["start_error_p95_seconds"],
        -metrics["end_error_p95_seconds"],
        metrics["interval_precision_iou50"],
        -metrics["false_intervals_per_minute"],
        -validation_loss,
    )


def train(settings: TrainSettings) -> dict:
    _seed_everything(settings.seed)
    device = select_device(settings.device)
    train_records = load_records(
        read_manifest(settings.manifest, split="train"),
        boundary_event_sigma_seconds=settings.boundary_event_sigma_seconds,
    )
    val_records = load_records(
        read_manifest(settings.manifest, split="val"),
        boundary_event_sigma_seconds=settings.boundary_event_sigma_seconds,
    )
    if not train_records or not val_records:
        raise ValueError("manifest must contain at least one train and one val source video")
    if any(
        item.cache.visual_feature_settings is None
        for item in [*train_records, *val_records]
    ):
        raise ValueError(
            "direct segment training requires visual feature sidecars for every cache"
        )
    ensure_source_disjoint(
        train_records,
        val_records,
        allow_same_source_temporal=settings.validation_mode == "diagnostic_temporal",
        temporal_guard_seconds=settings.temporal_guard_seconds,
    )
    proposal_positive_weight = _proposal_positive_weight(train_records)
    positive_weight_tensor = torch.tensor(proposal_positive_weight, device=device)
    boundary_event_positive_weights = _boundary_event_positive_weights(train_records)
    boundary_event_weight_tensor = torch.from_numpy(
        boundary_event_positive_weights
    ).to(device)
    feature_mean, feature_std = compute_feature_stats(train_records)
    train_dataset = TimingWindowDataset(
        train_records,
        feature_mean=feature_mean,
        feature_std=feature_std,
        window_frames=settings.window_frames,
        stride_frames=settings.stride_frames,
        max_windows=settings.max_train_windows,
    )
    val_dataset = TimingWindowDataset(
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
        num_workers=settings.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
    )
    first_cache = train_records[0].cache
    train_durations = [
        interval.end_seconds - interval.start_seconds
        for item in train_records
        for interval in item.intervals
    ]
    if not train_durations:
        raise ValueError("training split contains no subtitle segments")
    model_config = ModelConfig(
        feature_count=first_cache.features.shape[1],
        token_count=first_cache.tokens.shape[1],
        width=settings.width,
        temporal_layers=settings.temporal_layers,
        recurrent_layers=settings.recurrent_layers,
        dropout=settings.dropout,
        use_byte_branch=settings.use_byte_branch,
        initial_boundary_distance_seconds=float(np.median(train_durations)) / 2.0,
    )
    model = H264SubtitleSegmentModel(model_config).to(device)
    optimizer = AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    output_dir = settings.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() or (output_dir / "last.pt").exists():
        raise FileExistsError(
            f"training output already contains a run and resume is not implemented: {output_dir}"
        )
    best_selection: tuple | None = None
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    last_metrics: dict[str, float] = {}
    for epoch in range(1, settings.epochs + 1):
        train_losses = _window_loss(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            proposal_positive_weight=positive_weight_tensor,
            boundary_event_positive_weights=boundary_event_weight_tensor,
        )
        val_losses = _window_loss(
            model,
            val_loader,
            device=device,
            optimizer=None,
            proposal_positive_weight=positive_weight_tensor,
            boundary_event_positive_weights=boundary_event_weight_tensor,
        )
        calibration_predictions = _predict_records(
            model,
            train_records,
            feature_mean=feature_mean,
            feature_std=feature_std,
            settings=settings,
            device=device,
        )
        selection_config, calibration_metrics = _calibrate_segment_selection(
            calibration_predictions,
            settings=settings,
        )
        validation_predictions = _predict_records(
            model,
            val_records,
            feature_mean=feature_mean,
            feature_std=feature_std,
            settings=settings,
            device=device,
        )
        held_out = _held_out_metrics(
            validation_predictions,
            selection_config,
            target_recall=settings.target_recall,
        )
        last_metrics = {
            **{f"train_{key}": value for key, value in train_losses.items()},
            **{f"val_{key}": value for key, value in val_losses.items()},
            "calibration_interval_recall_iou50": calibration_metrics[
                "interval_recall_iou50"
            ],
            "calibration_recall_target_met": calibration_metrics["recall_target_met"],
            **held_out,
        }
        record = {"epoch": epoch, **last_metrics}
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        checkpoint = _checkpoint(
            model,
            model_config=model_config,
            settings=settings,
            feature_names=first_cache.feature_names,
            feature_mean=feature_mean,
            feature_std=feature_std,
            feature_settings=dict(first_cache.meta["feature_settings"]),
            visual_feature_settings=dict(first_cache.visual_feature_settings or {}),
            payload_tail_ratio=float(
                first_cache.meta["feature_settings"]["payload_tail_ratio"]
            ),
            spatial_contract=dict(first_cache.meta["spatial_contract"]),
            proposal_positive_weight=proposal_positive_weight,
            boundary_event_positive_weights=boundary_event_positive_weights,
            segment_selection_config=selection_config,
            calibration_metrics=calibration_metrics,
            epoch=epoch,
            metrics=last_metrics,
        )
        torch.save(checkpoint, output_dir / "last.pt")
        selection = _checkpoint_selection_key(
            held_out,
            validation_loss=val_losses["loss"],
            target_recall=settings.target_recall,
        )
        if best_selection is None or selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            best_metrics = dict(last_metrics)
            torch.save(checkpoint, output_dir / "best.pt")
        print(json.dumps(record, ensure_ascii=False))
    summary = {
        "best_epoch": best_epoch,
        "recall_target": settings.target_recall,
        "best_recall_target_met": bool(best_metrics.get("recall_target_met", 0.0)),
        "best_interval_recall_iou50": best_metrics.get("interval_recall_iou50", 0.0),
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "proposal_positive_weight": proposal_positive_weight,
        "boundary_event_positive_weights": boundary_event_positive_weights.tolist(),
        "use_byte_branch": settings.use_byte_branch,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_sources": len(train_records),
        "val_sources": len(val_records),
        "train_source_groups": len({item.record.source_group for item in train_records}),
        "val_source_groups": len({item.record.source_group for item in val_records}),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "device": str(device),
        "model_output_contract": "direct_scored_start_end_segments",
        "visual_pixel_decode": True,
        "validation_contract": (
            "held_out_by_declared_source_group_and_container_fingerprint"
            if settings.validation_mode == "held_out"
            else "diagnostic_same_source_temporal_not_held_out"
        ),
        "segment_selection_calibration_split": "train",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _checkpoint(
    model: H264SubtitleSegmentModel,
    *,
    model_config: ModelConfig,
    settings: TrainSettings,
    feature_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_settings: dict,
    visual_feature_settings: dict,
    payload_tail_ratio: float,
    spatial_contract: dict,
    proposal_positive_weight: float,
    boundary_event_positive_weights: np.ndarray,
    segment_selection_config: SegmentSelectionConfig,
    calibration_metrics: dict[str, float],
    epoch: int,
    metrics: dict[str, float],
) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "feature_version": FEATURE_VERSION,
        "model": model.state_dict(),
        "model_config": model_config.to_dict(),
        "model_output_contract": "direct_scored_start_end_segments",
        "feature_names": feature_names,
        "feature_mean": torch.from_numpy(feature_mean),
        "feature_std": torch.from_numpy(feature_std),
        "feature_settings": feature_settings,
        "visual_feature_settings": visual_feature_settings,
        "window_frames": settings.window_frames,
        "hop_frames": settings.stride_frames,
        "validation_mode": settings.validation_mode,
        "payload_tail_ratio": payload_tail_ratio,
        "spatial_contract": spatial_contract,
        "proposal_positive_weight": proposal_positive_weight,
        "boundary_event_positive_weights": torch.from_numpy(
            boundary_event_positive_weights
        ),
        "boundary_event_sigma_seconds": settings.boundary_event_sigma_seconds,
        "segment_selection_config": segment_selection_config.to_dict(),
        "segment_selection_calibration_split": "train",
        "calibration_target_recall": settings.target_recall,
        "calibration_recall_target_met": bool(
            calibration_metrics["recall_target_met"]
        ),
        "calibration_metrics": calibration_metrics,
        "epoch": epoch,
        "metrics": metrics,
    }
