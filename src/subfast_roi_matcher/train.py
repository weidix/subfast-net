from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from subfast_roi_data.data import RoiPairDataset, collate_pair_batch
from subfast_roi_data.pairs import (
    RoiPairSelection,
    ScheduledPairBatch,
    build_pair_epoch_schedule,
    build_pair_pools,
    select_pairs,
)
from subfast_roi_data.training import format_dataset_summary, model_parameter_count, parse_roi_size, seed_everything, synchronize_device
from subfast_shared.runtime import choose_device

from .config import RoiPairTrainSettings
from .metrics import pair_score_metrics
from .model import RoiPairMatcher, trace_pair_matcher_for_inference


def make_dataset(settings: RoiPairTrainSettings, *, train: bool) -> RoiPairDataset:
    return RoiPairDataset(
        settings.train_roots if train else [settings.val_root],
        resize_roi=settings.resize_roi,
        segment_aware_limit=not train,
        load_subtitle_masks=train,
    )


def assert_disjoint_datasets(
    train_dataset: RoiPairDataset,
    val_dataset: RoiPairDataset,
) -> None:
    train_roots = {sample.root.resolve() for sample in train_dataset.samples}
    val_roots = {sample.root.resolve() for sample in val_dataset.samples}
    if train_roots & val_roots:
        raise ValueError("validation root overlaps a training root")

    def video_frame(sample: Any) -> tuple[str, int] | None:
        if sample.video_id is None or sample.frame_index is None:
            return None
        return str(sample.video_id), int(sample.frame_index)

    def source_image(sample: Any) -> str | None:
        value = sample.annotation.get("source_image")
        return str(value) if value else None

    train_video_frames = {identity for sample in train_dataset.samples if (identity := video_frame(sample)) is not None}
    val_video_frames = {identity for sample in val_dataset.samples if (identity := video_frame(sample)) is not None}
    if train_video_frames & val_video_frames:
        raise ValueError("validation contains source video frames that also occur in training")
    train_sources = {identity for sample in train_dataset.samples if (identity := source_image(sample)) is not None}
    val_sources = {identity for sample in val_dataset.samples if (identity := source_image(sample)) is not None}
    if train_sources & val_sources:
        raise ValueError("validation contains source images that also occur in training")


def pair_selection_for_dataset(
    dataset: RoiPairDataset,
    settings: RoiPairTrainSettings,
) -> RoiPairSelection:
    samples = dataset.samples
    return select_pairs(
        presence=torch.tensor([1.0 if sample.has_subtitle else 0.0 for sample in samples]),
        segment_ids=[sample.segment_id for sample in samples],
        roots=[str(sample.root) for sample in samples],
        video_ids=[sample.video_id for sample in samples],
        ocr_texts=[sample.ocr_text for sample in samples],
        adjacent_segment_ids=[
            dataset.adjacent_segment_ids_by_sample_id.get((str(sample.root), sample.sample_id), frozenset())
            for sample in samples
        ],
        ocr_negative_enabled=settings.ocr_negative_enabled,
        ocr_negative_max_similarity=settings.ocr_negative_max_similarity,
        ocr_negative_ratio=settings.ocr_negative_ratio,
    )


def require_pair_classes(selection: RoiPairSelection, *, name: str) -> None:
    if selection.local_positive_pairs <= 0:
        raise ValueError(f"{name} has no positive same-subtitle pairs")
    if selection.negative_pairs <= 0:
        raise ValueError(f"{name} has no negative different-subtitle pairs")


def load_scheduled_pair_batch(
    dataset: RoiPairDataset,
    scheduled: ScheduledPairBatch,
) -> Any:
    return collate_pair_batch([dataset[index] for index in scheduled.sample_indices])


class ScheduledPairBatchDataset(Dataset):
    def __init__(
        self,
        dataset: RoiPairDataset,
        batches: tuple[ScheduledPairBatch, ...],
    ) -> None:
        self.dataset = dataset
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> tuple[ScheduledPairBatch, Any]:
        scheduled = self.batches[index]
        return scheduled, load_scheduled_pair_batch(self.dataset, scheduled)


def identity_batch(item: Any) -> Any:
    return item


def pair_index_tensors(
    scheduled: ScheduledPairBatch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = torch.tensor([pair.i for pair in scheduled.pairs], dtype=torch.long, device=device)
    right = torch.tensor([pair.j for pair in scheduled.pairs], dtype=torch.long, device=device)
    targets = torch.tensor([pair.same for pair in scheduled.pairs], dtype=torch.float32, device=device)
    return left, right, targets


def apply_photometric_jitter(images: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0.0:
        return images
    shape = (images.shape[0], 1, 1, 1)
    gain = 1.0 + (torch.rand(shape, device=images.device) * 2.0 - 1.0) * strength
    bias = (torch.rand(shape, device=images.device) * 2.0 - 1.0) * strength
    return images * gain + bias


def hard_tail_gap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
    tail_ratio: float = 0.10,
) -> torch.Tensor:
    positives = logits[targets > 0.5]
    negatives = logits[targets <= 0.5]
    if positives.numel() == 0 or negatives.numel() == 0:
        return logits.sum() * 0.0
    positive_count = max(1, math.ceil(positives.numel() * tail_ratio))
    negative_count = max(1, math.ceil(negatives.numel() * tail_ratio))
    positive_tail = positives.topk(positive_count, largest=False).values.mean()
    negative_tail = negatives.topk(negative_count, largest=True).values.mean()
    return F.softplus(negative_tail - positive_tail + margin)


def pair_mask_loss(
    mask_logits: torch.Tensor,
    subtitle_masks: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
) -> torch.Tensor:
    target = torch.maximum(
        subtitle_masks.index_select(0, left_indices),
        subtitle_masks.index_select(0, right_indices),
    )
    target = F.interpolate(target, size=mask_logits.shape[-2:], mode="area").clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(mask_logits, target)


def cache_validation_images(
    dataset: RoiPairDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> torch.Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pair_batch,
    )
    batches = [batch.images.to(dtype=torch.float16) for batch in tqdm(loader, desc="cache validation ROI", leave=False)]
    return torch.cat(batches, dim=0)


@torch.inference_mode()
def validate_pair_model(
    model: RoiPairMatcher,
    images: torch.Tensor,
    selection: RoiPairSelection,
    device: torch.device,
    settings: RoiPairTrainSettings,
) -> tuple[dict[str, float], list[float]]:
    model.eval()
    scores: list[float] = []
    loss_sum = 0.0
    pair_count = len(selection.pairs)
    for start in range(0, pair_count, settings.validation_batch_size):
        pairs = selection.pairs[start : start + settings.validation_batch_size]
        left_indices = torch.tensor([pair.i for pair in pairs], dtype=torch.long)
        right_indices = torch.tensor([pair.j for pair in pairs], dtype=torch.long)
        left = images.index_select(0, left_indices).to(device=device, dtype=torch.float32)
        right = images.index_select(0, right_indices).to(device=device, dtype=torch.float32)
        targets = torch.tensor([pair.same for pair in pairs], dtype=torch.float32, device=device)
        logits, _ = model(left, right)
        loss_sum += float(F.binary_cross_entropy_with_logits(logits, targets, reduction="sum").detach().cpu())
        scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
    metrics = pair_score_metrics(scores, selection, threshold=settings.threshold)
    metrics["validation_pair_bce"] = loss_sum / max(1, pair_count)
    for source in ("local", "ocr"):
        scoped = [
            score < settings.threshold
            for score, pair in zip(scores, selection.pairs, strict=True)
            if not pair.same and pair.source == source
        ]
        metrics[f"pair_{source}_negative_accuracy"] = sum(scoped) / len(scoped) if scoped else 0.0
    return metrics, scores


def checkpoint_rank(metrics: dict[str, float]) -> tuple[int, int, int, float]:
    gap = float(metrics["pair_gap"])
    false_positive = int(metrics["pair_false_positive_count"])
    false_negative = int(metrics["pair_false_negative_count"])
    return (
        false_positive + false_negative,
        false_positive,
        0 if gap > 0.0 else 1,
        -gap,
    )


def checkpoint_payload(
    settings: RoiPairTrainSettings,
    model: RoiPairMatcher,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    metrics: dict[str, float],
    best_epoch: int,
    best_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_type": "roi_pair_matcher",
        "architecture_version": model.architecture_version,
        "pooling_version": model.pooling_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "settings": settings.model_dump(mode="json"),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }


def inference_payload(
    settings: RoiPairTrainSettings,
    model: RoiPairMatcher,
    *,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_type": "roi_pair_matcher",
        "architecture_version": model.architecture_version,
        "pooling_version": model.pooling_version,
        "model": model.state_dict(),
        "resize_roi": list(settings.resize_roi),
        "threshold": settings.threshold,
        "ocr_negative_enabled": settings.ocr_negative_enabled,
        "ocr_negative_max_similarity": settings.ocr_negative_max_similarity,
        "ocr_negative_ratio": settings.ocr_negative_ratio,
        "epoch": epoch,
        "metrics": metrics,
    }


def save_pair_scores(
    path: Path,
    dataset: RoiPairDataset,
    selection: RoiPairSelection,
    scores: list[float],
    threshold: float,
) -> None:
    lines: list[str] = []
    for pair, score in zip(selection.pairs, scores, strict=True):
        left = dataset.samples[pair.i]
        right = dataset.samples[pair.j]
        lines.append(
            json.dumps(
                {
                    "left_sample_id": left.sample_id,
                    "right_sample_id": right.sample_id,
                    "left_text": left.ocr_text,
                    "right_text": right.ocr_text,
                    "same": pair.same,
                    "source": pair.source,
                    "score": score,
                    "prediction": score >= threshold,
                    "correct": (score >= threshold) == pair.same,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    settings: RoiPairTrainSettings,
    *,
    completed_epoch: int,
    best_epoch: int,
    best_metrics: dict[str, float],
    model: RoiPairMatcher,
) -> None:
    best_path = settings.output_dir / "best.pt"
    inference_path = settings.output_dir / "best_inference.pt"
    summary = {
        "record_type": "roi_pair_training_summary",
        "completed_epoch": completed_epoch,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "best_inference_checkpoint": str(inference_path),
        "metrics_log": str(settings.output_dir / "metrics.jsonl"),
        "pair_scores": str(settings.output_dir / "best_pair_scores.jsonl"),
        "architecture_version": model.architecture_version,
        "model_parameters": model_parameter_count(model),
        "best_checkpoint_bytes": best_path.stat().st_size if best_path.exists() else 0,
        "best_inference_checkpoint_bytes": inference_path.stat().st_size if inference_path.exists() else 0,
        "inference_benchmark": {
            "scope": "conv_bn_fused_traced_optimized_forward_only",
            "device": str(next(model.parameters()).device),
            "torch_version": str(torch.__version__),
            "dtype": "float32",
            "batch_size": 1,
            "input_height": settings.resize_roi[1],
            "input_width": settings.resize_roi[0],
            "warmup": 40,
            "iterations": 500,
            "target_median_ms": 0.8,
        },
        "validation": best_metrics,
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")


def load_pair_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[RoiPairMatcher, dict[str, Any]]:
    if path.is_dir():
        path = path / "best.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "roi_pair_matcher":
        raise RuntimeError(f"invalid ROI pair matcher checkpoint: {path}")
    model = RoiPairMatcher().to(device)
    architecture_version = int(checkpoint.get("architecture_version", 1))
    if architecture_version != model.architecture_version:
        raise RuntimeError(
            f"unsupported ROI pair matcher architecture_version={architecture_version}; "
            f"runtime=v{model.architecture_version}; retrain instead of resuming RGB pair features"
        )
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def load_pair_inference_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[torch.jit.ScriptModule, dict[str, Any]]:
    """Load the portable weights and prepare the optimized runtime on the target device."""
    model, checkpoint = load_pair_checkpoint(path, device)
    raw_settings = dict(checkpoint.get("settings") or {})
    resize_roi = tuple(
        checkpoint.get("resize_roi")
        or raw_settings.get("resize_roi")
        or RoiPairTrainSettings().resize_roi
    )
    width, height = int(resize_roi[0]), int(resize_roi[1])
    left = torch.zeros((1, 3, height, width), dtype=torch.float32, device=device)
    right = torch.zeros_like(left)
    return trace_pair_matcher_for_inference(model, left, right), checkpoint


@torch.inference_mode()
def measure_pair_latency(
    model: RoiPairMatcher,
    left: torch.Tensor,
    right: torch.Tensor,
    device: torch.device,
    *,
    warmup: int = 40,
    iterations: int = 500,
) -> tuple[float, float]:
    model.eval()
    left = left[:1].to(device=device, dtype=torch.float32)
    right = right[:1].to(device=device, dtype=torch.float32)
    inference_model = trace_pair_matcher_for_inference(model, left, right)
    for _ in range(warmup):
        inference_model(left, right)
    synchronize_device(device)
    timings: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        inference_model(left, right)
        synchronize_device(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return timings[len(timings) // 2], timings[min(len(timings) - 1, int(len(timings) * 0.90))]


def run_training(settings: RoiPairTrainSettings) -> dict[str, float]:
    seed_everything(settings.seed)
    device = choose_device(settings.device)
    if settings.minimum_learning_rate > settings.learning_rate:
        raise ValueError("minimum learning rate cannot exceed learning rate")
    if settings.output_dir.exists() and (settings.output_dir / "metrics.jsonl").exists() and settings.resume is None:
        raise FileExistsError(f"output already contains a training run: {settings.output_dir}")
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_dataset(settings, train=True)
    val_dataset = make_dataset(settings, train=False)
    assert_disjoint_datasets(train_dataset, val_dataset)
    print(f"device={device}", flush=True)
    print(format_dataset_summary("train", train_dataset), flush=True)
    print(format_dataset_summary("val", val_dataset), flush=True)

    train_pair_pools = build_pair_pools(
        train_dataset.samples,
        ocr_negative_enabled=settings.ocr_negative_enabled,
        ocr_negative_max_similarity=settings.ocr_negative_max_similarity,
    )
    if not train_pair_pools.positive_pairs:
        raise ValueError("training dataset has no positive same-subtitle pairs")
    usable_ocr_negatives = (
        len(train_pair_pools.ocr_negative_pairs)
        if settings.ocr_negative_enabled and settings.ocr_negative_ratio > 0.0
        else 0
    )
    if len(train_pair_pools.local_negative_pairs) + usable_ocr_negatives <= 0:
        raise ValueError("training dataset has no negative different-subtitle pairs")
    val_selection = pair_selection_for_dataset(val_dataset, settings)
    require_pair_classes(val_selection, name="validation dataset")
    val_images = cache_validation_images(
        val_dataset,
        batch_size=settings.validation_batch_size,
        num_workers=settings.num_workers,
    )
    print(
        f"train_pair_pool positive={len(train_pair_pools.positive_pairs)} "
        f"local_negative={len(train_pair_pools.local_negative_pairs)} "
        f"ocr_negative={len(train_pair_pools.ocr_negative_pairs)}",
        flush=True,
    )
    print(
        f"validation_pairs total={len(val_selection.pairs)} positive={val_selection.local_positive_pairs} "
        f"local_negative={val_selection.local_negative_pairs} ocr_negative={val_selection.ocr_negative_pairs}",
        flush=True,
    )

    model = RoiPairMatcher().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    start_epoch = 1
    step = 0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    resume_requires_rerank = False
    if settings.resume is not None:
        loaded_model, checkpoint = load_pair_checkpoint(settings.resume, device)
        model.load_state_dict(loaded_model.state_dict())
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        step = int(checkpoint.get("step", 0))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
        best_metrics = dict(checkpoint.get("best_metrics") or checkpoint.get("metrics") or {})
        resume_requires_rerank = int(checkpoint.get("pooling_version", 1)) != model.pooling_version
        if resume_requires_rerank:
            best_metrics = {}
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"resumed={settings.resume} start_epoch={start_epoch}", flush=True)
    best_rank = checkpoint_rank(best_metrics) if best_metrics else None
    metrics_path = settings.output_dir / "metrics.jsonl"
    local_best_path = settings.output_dir / "best.pt"
    if settings.resume is not None and (resume_requires_rerank or not local_best_path.exists()):
        if local_best_path.exists():
            candidate_model, candidate_checkpoint = load_pair_checkpoint(local_best_path, device)
        else:
            candidate_model, candidate_checkpoint = model, checkpoint
        current_validation, current_scores = validate_pair_model(
            candidate_model,
            val_images,
            val_selection,
            device,
            settings,
        )
        current_metrics = {
            **dict(candidate_checkpoint.get("metrics") or {}),
            **current_validation,
        }
        best_epoch = int(candidate_checkpoint.get("epoch", start_epoch - 1))
        best_metrics = current_metrics
        best_rank = checkpoint_rank(best_metrics)
        if local_best_path.exists():
            refreshed_checkpoint = dict(candidate_checkpoint)
            refreshed_checkpoint.update(
                {
                    "architecture_version": candidate_model.architecture_version,
                    "pooling_version": candidate_model.pooling_version,
                    "metrics": best_metrics,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                }
            )
            torch.save(refreshed_checkpoint, local_best_path)
        else:
            torch.save(
                checkpoint_payload(
                    settings,
                    candidate_model,
                    optimizer,
                    epoch=best_epoch,
                    step=step,
                    metrics=best_metrics,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                ),
                local_best_path,
            )
        torch.save(
            inference_payload(settings, candidate_model, epoch=best_epoch, metrics=best_metrics),
            settings.output_dir / "best_inference.pt",
        )
        save_pair_scores(
            settings.output_dir / "best_pair_scores.jsonl",
            val_dataset,
            val_selection,
            current_scores,
            settings.threshold,
        )
        if resume_requires_rerank:
            print("reranked existing best checkpoint with width-peak pooling", flush=True)

    for epoch in range(start_epoch, settings.epochs + 1):
        if settings.epochs == 1:
            progress = 0.0
        else:
            progress = (epoch - 1) / (settings.epochs - 1)
        learning_rate = settings.minimum_learning_rate + 0.5 * (
            settings.learning_rate - settings.minimum_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        schedule = build_pair_epoch_schedule(
            train_dataset.samples,
            batch_size=settings.batch_size,
            negative_ratio=settings.negative_ratio,
            ocr_negative_enabled=settings.ocr_negative_enabled,
            ocr_negative_max_similarity=settings.ocr_negative_max_similarity,
            ocr_negative_ratio=settings.ocr_negative_ratio,
            seed=settings.seed,
            epoch=epoch,
            pair_pools=train_pair_pools,
        )
        model.train()
        epoch_start = time.perf_counter()
        total_loss_sum = 0.0
        pair_loss_sum = 0.0
        mask_loss_sum = 0.0
        gap_loss_sum = 0.0
        correct = 0
        trained_pairs = 0
        pair_batch_loader = DataLoader(
            ScheduledPairBatchDataset(train_dataset, schedule.batches),
            batch_size=None,
            shuffle=False,
            num_workers=settings.num_workers,
            collate_fn=identity_batch,
        )
        progress_bar = tqdm(pair_batch_loader, desc=f"roi pair epoch {epoch}/{settings.epochs}", leave=False)
        for batch_index, (scheduled, batch) in enumerate(progress_bar, start=1):
            images = apply_photometric_jitter(batch.images.to(device), settings.photometric_jitter)
            if batch.subtitle_masks is None:
                raise RuntimeError("subtitle masks are required for pair matcher training")
            subtitle_masks = batch.subtitle_masks.to(device)
            left_indices, right_indices, targets = pair_index_tensors(scheduled, device)
            optimizer.zero_grad(set_to_none=True)
            logits, mask_logits = model(
                images.index_select(0, left_indices),
                images.index_select(0, right_indices),
            )
            pair_loss = F.binary_cross_entropy_with_logits(logits, targets)
            mask_loss = pair_mask_loss(mask_logits, subtitle_masks, left_indices, right_indices)
            gap_loss = hard_tail_gap_loss(logits, targets, margin=settings.tail_gap_margin)
            total_loss = (
                pair_loss
                + settings.mask_loss_weight * mask_loss
                + settings.tail_gap_loss_weight * gap_loss
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            count = len(scheduled.pairs)
            trained_pairs += count
            step += 1
            total_loss_sum += float(total_loss.detach().cpu()) * count
            pair_loss_sum += float(pair_loss.detach().cpu()) * count
            mask_loss_sum += float(mask_loss.detach().cpu()) * count
            gap_loss_sum += float(gap_loss.detach().cpu()) * count
            correct += int(((logits >= 0.0) == (targets > 0.5)).sum().detach().cpu())
            if batch_index % settings.log_interval == 0:
                progress_bar.set_postfix(
                    loss=f"{total_loss_sum / trained_pairs:.4f}",
                    acc=f"{correct / trained_pairs:.4f}",
                )

        validation_metrics, scores = validate_pair_model(model, val_images, val_selection, device, settings)
        epoch_seconds = time.perf_counter() - epoch_start
        metrics = {
            "record_type": "roi_pair_validation",
            "epoch": float(epoch),
            "step": float(step),
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
            "train_pair_loss": pair_loss_sum / max(1, trained_pairs),
            "train_mask_loss": mask_loss_sum / max(1, trained_pairs),
            "train_tail_gap_loss": gap_loss_sum / max(1, trained_pairs),
            "train_total_loss": total_loss_sum / max(1, trained_pairs),
            "train_pair_accuracy": correct / max(1, trained_pairs),
            "train_pairs": float(trained_pairs),
            "train_positive_pairs": float(schedule.positive_pair_count),
            "train_negative_pairs": float(schedule.negative_pair_count),
            "train_positive_pair_repeat_rate": schedule.positive_pair_repeat_rate,
            "train_negative_pair_repeat_rate": schedule.negative_pair_repeat_rate,
            "model_parameters": float(model_parameter_count(model)),
            **validation_metrics,
        }
        append_metrics(metrics_path, metrics)

        rank = checkpoint_rank(metrics)
        is_best = best_rank is None or rank < best_rank
        if is_best:
            best_rank = rank
            best_epoch = epoch
            best_metrics = metrics
            torch.save(
                checkpoint_payload(
                    settings,
                    model,
                    optimizer,
                    epoch=epoch,
                    step=step,
                    metrics=metrics,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                ),
                settings.output_dir / "best.pt",
            )
            torch.save(
                inference_payload(settings, model, epoch=epoch, metrics=metrics),
                settings.output_dir / "best_inference.pt",
            )
            save_pair_scores(
                settings.output_dir / "best_pair_scores.jsonl",
                val_dataset,
                val_selection,
                scores,
                settings.threshold,
            )

        torch.save(
            checkpoint_payload(
                settings,
                model,
                optimizer,
                epoch=epoch,
                step=step,
                metrics=metrics,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
            ),
            settings.output_dir / "last.pt",
        )
        write_summary(
            settings,
            completed_epoch=epoch,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            model=model,
        )
        print(
            f"epoch={epoch}/{settings.epochs} train_acc={metrics['train_pair_accuracy']:.4f} "
            f"val_acc={metrics['pair_accuracy']:.4f} fp={int(metrics['pair_false_positive_count'])} "
            f"fn={int(metrics['pair_false_negative_count'])} gap={metrics['pair_gap']:.6f} "
            f"auc={metrics['pair_roc_auc']:.6f} seconds={epoch_seconds:.2f} best_epoch={best_epoch}",
            flush=True,
        )

    best_model, _ = load_pair_checkpoint(settings.output_dir / "best.pt", device)
    current_validation, best_scores = validate_pair_model(best_model, val_images, val_selection, device, settings)
    best_metrics = {**best_metrics, **current_validation}
    save_pair_scores(
        settings.output_dir / "best_pair_scores.jsonl",
        val_dataset,
        val_selection,
        best_scores,
        settings.threshold,
    )
    first_pair = val_selection.pairs[0]
    latency_median, latency_p90 = measure_pair_latency(
        best_model,
        val_images[first_pair.i : first_pair.i + 1],
        val_images[first_pair.j : first_pair.j + 1],
        device,
    )
    best_metrics = dict(best_metrics)
    best_metrics["pair_forward_median_ms"] = latency_median
    best_metrics["pair_forward_p90_ms"] = latency_p90
    best_metrics["pair_forward_target_ms"] = 0.8
    best_metrics["pair_forward_target_met"] = float(latency_median <= 0.8)
    write_summary(
        settings,
        completed_epoch=settings.epochs,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        model=best_model,
    )
    return best_metrics


def parse_args(argv: list[str] | None = None) -> RoiPairTrainSettings:
    defaults = RoiPairTrainSettings()
    parser = argparse.ArgumentParser(description="Train the direct ROI same-subtitle pair matcher.")
    parser.add_argument("--train-root", dest="train_roots", type=Path, action="append")
    parser.add_argument("--val-root", type=Path, default=defaults.val_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resize-roi", type=parse_roi_size, default=defaults.resize_roi)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--validation-batch-size", type=int, default=defaults.validation_batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--min-lr", type=float, default=defaults.minimum_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--negative-ratio", type=float, default=defaults.negative_ratio)
    parser.add_argument("--ocr-negative-enabled", action=argparse.BooleanOptionalAction, default=defaults.ocr_negative_enabled)
    parser.add_argument("--ocr-negative-max-similarity", type=float, default=defaults.ocr_negative_max_similarity)
    parser.add_argument("--ocr-negative-ratio", type=float, default=defaults.ocr_negative_ratio)
    parser.add_argument("--mask-loss-weight", type=float, default=defaults.mask_loss_weight)
    parser.add_argument("--tail-gap-loss-weight", type=float, default=defaults.tail_gap_loss_weight)
    parser.add_argument("--tail-gap-margin", type=float, default=defaults.tail_gap_margin)
    parser.add_argument("--photometric-jitter", type=float, default=defaults.photometric_jitter)
    parser.add_argument("--threshold", type=float, default=defaults.threshold)
    parser.add_argument("--log-interval", type=int, default=defaults.log_interval)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    args = parser.parse_args(argv)
    return RoiPairTrainSettings(
        train_roots=args.train_roots or defaults.train_roots,
        val_root=args.val_root,
        output_dir=args.output_dir,
        resume=args.resume,
        resize_roi=args.resize_roi,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        minimum_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        negative_ratio=args.negative_ratio,
        ocr_negative_enabled=args.ocr_negative_enabled,
        ocr_negative_max_similarity=args.ocr_negative_max_similarity,
        ocr_negative_ratio=args.ocr_negative_ratio,
        mask_loss_weight=args.mask_loss_weight,
        tail_gap_loss_weight=args.tail_gap_loss_weight,
        tail_gap_margin=args.tail_gap_margin,
        photometric_jitter=args.photometric_jitter,
        threshold=args.threshold,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
    )


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a direct ROI pair matcher checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resize-roi", type=parse_roi_size)
    parser.add_argument("--batch-size", type=int, default=RoiPairTrainSettings().validation_batch_size)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default=RoiPairTrainSettings().device)
    return parser.parse_args(argv)


def run_validation(args: argparse.Namespace) -> dict[str, float]:
    device = choose_device(args.device)
    model, checkpoint = load_pair_checkpoint(args.checkpoint, device)
    raw_settings = dict(checkpoint.get("settings") or {})
    resize_roi = args.resize_roi or tuple(checkpoint.get("resize_roi") or raw_settings.get("resize_roi") or (256, 64))
    threshold = float(
        args.threshold
        if args.threshold is not None
        else checkpoint.get("threshold", raw_settings.get("threshold", 0.5))
    )
    settings = RoiPairTrainSettings(
        train_roots=[args.root],
        val_root=args.root,
        resize_roi=resize_roi,
        validation_batch_size=args.batch_size,
        ocr_negative_enabled=bool(
            checkpoint.get(
                "ocr_negative_enabled",
                raw_settings.get("ocr_negative_enabled", RoiPairTrainSettings().ocr_negative_enabled),
            )
        ),
        ocr_negative_max_similarity=float(
            checkpoint.get(
                "ocr_negative_max_similarity",
                raw_settings.get(
                    "ocr_negative_max_similarity",
                    RoiPairTrainSettings().ocr_negative_max_similarity,
                ),
            )
        ),
        ocr_negative_ratio=float(
            checkpoint.get(
                "ocr_negative_ratio",
                raw_settings.get("ocr_negative_ratio", RoiPairTrainSettings().ocr_negative_ratio),
            )
        ),
        threshold=threshold,
        device=args.device,
    )
    dataset = make_dataset(settings, train=False)
    selection = pair_selection_for_dataset(dataset, settings)
    require_pair_classes(selection, name="validation dataset")
    images = cache_validation_images(dataset, batch_size=args.batch_size, num_workers=0)
    metrics, _ = validate_pair_model(model, images, selection, device, settings)
    first_pair = selection.pairs[0]
    median_ms, p90_ms = measure_pair_latency(
        model,
        images[first_pair.i : first_pair.i + 1],
        images[first_pair.j : first_pair.j + 1],
        device,
    )
    metrics.update(
        {
            "record_type": "roi_pair_checkpoint_validation",
            "checkpoint_epoch": float(checkpoint.get("epoch", 0)),
            "model_parameters": float(model_parameter_count(model)),
            "pair_forward_median_ms": median_ms,
            "pair_forward_p90_ms": p90_ms,
            "pair_forward_target_ms": 0.8,
            "pair_forward_target_met": float(median_ms <= 0.8),
        }
    )
    print(format_dataset_summary("validate", dataset), flush=True)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))
    return metrics


def main(argv: list[str] | None = None) -> None:
    metrics = run_training(parse_args(argv))
    print(json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True))


def main_validate(argv: list[str] | None = None) -> None:
    run_validation(parse_validate_args(argv))
