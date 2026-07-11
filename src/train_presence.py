from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .roi_presence_config import RoiPresenceTrainSettings
from .roi_presence_dataset import RoiPresenceDataset, collate_presence_batch
from .roi_presence_loss import (
    composite_valid_mask,
    counterfactual_presence_loss,
    erase_subtitle_regions,
    positive_with_region_mask,
    presence_importance_weights,
    subtitle_region_loss,
    transplant_subtitle_regions,
)
from .roi_presence_metrics import checkpoint_rank, checkpoint_score
from .roi_presence_model import RoiPresenceModel
from .roi_presence_sampler import PresenceBalancedBatchSampler
from .roi_presence_validation import load_previous_scores, validate_presence
from .train import choose_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_roi_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI resize dimensions must be positive")
    return width, height


def make_dataset(settings: RoiPresenceTrainSettings, *, train: bool) -> RoiPresenceDataset:
    return RoiPresenceDataset(
        settings.train_roots if train else [settings.val_root],
        resize_roi=settings.resize_roi,
        resize_mode=settings.resize_mode,
        max_samples=settings.max_train_samples if train else settings.max_val_samples,
        negative_ratio=None if train else settings.val_negative_ratio,
        load_subtitle_masks=True,
    )


def make_training_loader(
    dataset: RoiPresenceDataset,
    settings: RoiPresenceTrainSettings,
) -> tuple[DataLoader, PresenceBalancedBatchSampler | None]:
    if settings.train_negative_ratio is None:
        return (
            DataLoader(
                dataset,
                batch_size=settings.batch_size,
                shuffle=True,
                num_workers=settings.num_workers,
                collate_fn=collate_presence_batch,
            ),
            None,
        )
    sampler = PresenceBalancedBatchSampler(
        dataset.samples,
        batch_size=settings.batch_size,
        negative_ratio=settings.train_negative_ratio,
        seed=settings.seed,
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=settings.num_workers,
            collate_fn=collate_presence_batch,
        ),
        sampler,
    )


def format_dataset_summary(name: str, dataset: RoiPresenceDataset) -> str:
    summary = dataset.summary
    roots = ", ".join(f"{root}={count}" for root, count in sorted(summary.roots.items()))
    return (
        f"{name}: samples={summary.total} positive={summary.positive} empty={summary.empty} "
        f"positive_ratio={summary.positive_ratio:.3f} empty_ratio={summary.empty_ratio:.3f} "
        f"text_distractor_negatives={dataset.text_distractor_negatives} "
        f"positive_without_region={dataset.positive_without_region} "
        f"positive_without_donor={dataset.positive_without_donor} "
        f"roi_size={summary.roi_size[0]}x{summary.roi_size[1]} roots=[{roots}]"
    )


def resolve_resume_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    last = path / "last.pt"
    if last.is_file():
        return last
    best = path / "best.pt"
    if best.is_file():
        return best
    raise FileNotFoundError(f"resume checkpoint not found: {path}")


def checkpoint_payload(
    settings: RoiPresenceTrainSettings,
    model: RoiPresenceModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    best_epoch: int,
    best_metrics: dict[str, float],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_type": "roi_presence",
        "architecture_version": model.architecture_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "settings": settings.model_dump(mode="json"),
        "metrics": metrics,
        "epoch": epoch,
        "step": step,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "score_contract": {
            "kind": "subtitle_region_evidence_score",
            "positive_prior": (
                settings.score_positive_prior
                if settings.score_positive_prior is not None
                else metrics.get("train_data_positive_prior")
            ),
            "decision_threshold": settings.decision_threshold,
            "subtitle_specificity_evaluable": bool(metrics.get("subtitle_specificity_evaluable", 0.0)),
            "preprocessing": {
                "resize_roi": settings.resize_roi,
                "resize_mode": settings.resize_mode,
                "normalized_padding_value": 0.0,
                "valid_mask": "explicit_or_exact-zero-derived",
            },
        },
    }


def inference_payload(
    settings: RoiPresenceTrainSettings,
    model: RoiPresenceModel,
    *,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_type": "roi_presence",
        "architecture_version": model.architecture_version,
        "model": model.state_dict(),
        "resize_roi": list(settings.resize_roi) if settings.resize_roi is not None else None,
        "resize_mode": settings.resize_mode,
        "width": settings.width,
        "evidence_kernel_size": settings.evidence_kernel_size,
        "evidence_temperature": settings.evidence_temperature,
        "decision_threshold": settings.decision_threshold,
        "epoch": epoch,
        "metrics": metrics,
        "score_contract": {
            "kind": "subtitle_region_evidence_score",
            "positive_prior": (
                settings.score_positive_prior
                if settings.score_positive_prior is not None
                else metrics.get("train_data_positive_prior")
            ),
            "decision_threshold": settings.decision_threshold,
            "subtitle_specificity_evaluable": bool(metrics.get("subtitle_specificity_evaluable", 0.0)),
            "preprocessing": {
                "resize_roi": settings.resize_roi,
                "resize_mode": settings.resize_mode,
                "normalized_padding_value": 0.0,
                "valid_mask": "explicit_or_exact-zero-derived",
            },
        },
    }


def run_training(settings: RoiPresenceTrainSettings) -> dict[str, float]:
    seed_everything(settings.seed)
    device = choose_device(settings.device)
    if settings.output_dir.exists() and (settings.output_dir / "metrics.jsonl").exists() and settings.resume is None:
        raise FileExistsError(f"output already contains a training run: {settings.output_dir}")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = make_dataset(settings, train=True)
    val_dataset = make_dataset(settings, train=False)
    if train_dataset.summary.roi_size != val_dataset.summary.roi_size:
        raise ValueError(
            f"train/val ROI size mismatch: train={train_dataset.summary.roi_size} "
            f"val={val_dataset.summary.roi_size}; pass --resize-roi WIDTHxHEIGHT for explicit resize"
        )
    if not len(train_dataset):
        raise RuntimeError("no ROI training samples found")
    if not len(val_dataset):
        raise RuntimeError("no ROI validation samples found")
    train_loader, sampler = make_training_loader(train_dataset, settings)
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        collate_fn=collate_presence_batch,
    )
    model = RoiPresenceModel(
        width=settings.width,
        evidence_kernel_size=settings.evidence_kernel_size,
        evidence_temperature=settings.evidence_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    start_epoch = 1
    global_step = 0
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    last_metrics: dict[str, float] = {}
    if settings.resume is not None:
        checkpoint_path = resolve_resume_checkpoint(settings.resume)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "roi_presence":
            raise RuntimeError(f"invalid ROI Presence checkpoint: {checkpoint_path}")
        checkpoint_version = int(checkpoint.get("architecture_version", 1))
        if checkpoint_version != model.architecture_version:
            raise RuntimeError(
                f"ROI Presence architecture mismatch: checkpoint=v{checkpoint_version} "
                f"trainer=v{model.architecture_version}; V1 BatchNorm/top-k checkpoints cannot resume V2 training"
            )
        checkpoint_settings = checkpoint.get("settings") or {}
        for name in ("resize_roi", "resize_mode", "evidence_kernel_size", "evidence_temperature", "width"):
            current = settings.model_dump(mode="json").get(name)
            previous = checkpoint_settings.get(name)
            if previous != current:
                raise RuntimeError(
                    f"resume setting mismatch for {name}: checkpoint={previous!r} current={current!r}"
                )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("step", 0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_metrics = dict(checkpoint.get("best_metrics") or checkpoint.get("metrics") or {})
        last_metrics = dict(checkpoint.get("metrics") or {})
        local_best_checkpoint = settings.output_dir / "best.pt"
        if not local_best_checkpoint.is_file():
            print(
                "warning: resume target has no local copy of the historical best checkpoint; "
                "best selection will restart from the first continued epoch",
                flush=True,
            )
            best_epoch = 0
            best_metrics = None
        print(f"resume={checkpoint_path} start_epoch={start_epoch} step={global_step}", flush=True)
    end_epoch = start_epoch + settings.epochs - 1
    print(f"device={device}", flush=True)
    overlap = any(root.resolve() == settings.val_root.resolve() for root in settings.train_roots)
    if overlap:
        print("warning: validation root overlaps a training root; this run is not held-out quality evidence", flush=True)
    print(format_dataset_summary("train", train_dataset), flush=True)
    print(format_dataset_summary("val", val_dataset), flush=True)
    if train_dataset.positive_without_region or val_dataset.positive_without_region:
        raise RuntimeError(
            "V2 dense supervision requires a region label for every positive sample: "
            f"train_missing={train_dataset.positive_without_region} "
            f"val_missing={val_dataset.positive_without_region}"
        )
    if train_dataset.positive_without_donor or val_dataset.positive_without_donor:
        raise RuntimeError(
            "V2 counterfactual supervision requires at least one empty ROI donor: "
            f"train_missing={train_dataset.positive_without_donor} "
            f"val_missing={val_dataset.positive_without_donor}"
        )
    specificity_evaluable = (
        train_dataset.text_distractor_negatives > 0 and val_dataset.text_distractor_negatives > 0
    )
    if settings.require_text_distractor_negatives and not specificity_evaluable:
        raise RuntimeError(
            "subtitle specificity requires negative samples that retain non-subtitle text boxes; "
            "set reviewed has_subtitle=false while keeping their label boxes"
        )
    if not specificity_evaluable:
        print(
            "warning: no reviewed non-subtitle-text negatives in both train and validation; "
            "this run can verify text-region presence but cannot verify subtitle-vs-watermark/UI specificity",
            flush=True,
        )
    sampled_positive_prior = (
        sampler.positive_slots / settings.batch_size
        if sampler is not None
        else train_dataset.summary.positive_ratio
    )
    target_positive_prior = settings.score_positive_prior or train_dataset.summary.positive_ratio
    print(
        f"score_prior: sampled_positive={sampled_positive_prior:.6f} "
        f"target_positive={target_positive_prior:.6f}",
        flush=True,
    )
    metrics_path = settings.output_dir / "metrics.jsonl"
    previous_samples_path = settings.output_dir / ".previous_presence_samples.jsonl"
    metrics_mode = "a" if settings.resume is not None else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, end_epoch + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            epoch_start = time.perf_counter()
            train_loss = 0.0
            train_presence_loss = 0.0
            train_region_loss = 0.0
            train_region_bce = 0.0
            train_region_dice = 0.0
            train_region_projection = 0.0
            train_counterfactual_loss = 0.0
            positive_samples = 0
            negative_samples = 0
            batches = 0
            progress = tqdm(train_loader, desc=f"roi presence epoch {epoch}/{end_epoch}", leave=False)
            for batch_index, batch in enumerate(progress, start=1):
                batch_start = time.perf_counter()
                images = batch.images.to(device)
                presence = batch.presence.to(device)
                valid_masks = batch.valid_masks.to(device)
                donor_available = batch.donor_available.to(device)
                if batch.subtitle_masks is None:
                    raise RuntimeError("ROI Presence V2 training requires subtitle masks")
                subtitle_masks = batch.subtitle_masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, region_logits = model.forward_with_presence_map(images, valid_masks)
                classification_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    presence,
                    weight=presence_importance_weights(
                        presence,
                        sampled_positive_prior=sampled_positive_prior,
                        target_positive_prior=target_positive_prior,
                    ),
                )
                region = subtitle_region_loss(
                    region_logits,
                    subtitle_masks,
                    presence,
                    valid_masks,
                    dice_weight=settings.region_dice_weight,
                    projection_weight=settings.region_projection_weight,
                    text_distractor_weight=settings.text_distractor_weight,
                )
                valid_counterfactual = (
                    positive_with_region_mask(subtitle_masks, presence) & donor_available
                )
                if settings.counterfactual_loss_weight > 0.0 and bool(valid_counterfactual.any()):
                    selected_images = images[valid_counterfactual]
                    selected_masks = subtitle_masks[valid_counterfactual]
                    selected_valid_masks = valid_masks[valid_counterfactual]
                    donors = batch.donor_images[valid_counterfactual.cpu()].to(device)
                    donor_valid_masks = batch.donor_valid_masks[valid_counterfactual.cpu()].to(device)
                    seam_donors = batch.seam_donor_images[valid_counterfactual.cpu()].to(device)
                    seam_donor_valid_masks = batch.seam_donor_valid_masks[
                        valid_counterfactual.cpu()
                    ].to(device)
                    variant_images = [
                        erase_subtitle_regions(
                            selected_images,
                            selected_masks,
                            donor_images=donors,
                        ),
                        transplant_subtitle_regions(selected_images, selected_masks, donors),
                        erase_subtitle_regions(
                            donors,
                            selected_masks,
                            donor_images=seam_donors,
                        ),
                    ]
                    variant_valid_masks = [
                        composite_valid_mask(selected_masks, donor_valid_masks, selected_valid_masks),
                        composite_valid_mask(selected_masks, selected_valid_masks, donor_valid_masks),
                        composite_valid_mask(selected_masks, seam_donor_valid_masks, donor_valid_masks),
                    ]
                    variant_logits = model(
                        torch.cat(variant_images),
                        torch.cat(variant_valid_masks),
                    )
                    variant_size = selected_images.shape[0]
                    erased_logits = variant_logits[:variant_size]
                    transplanted_logits = variant_logits[variant_size : 2 * variant_size]
                    seam_control_logits = variant_logits[2 * variant_size :]
                    counterfactual = counterfactual_presence_loss(
                        logits[valid_counterfactual],
                        erased_logits,
                        transplanted_logits,
                        seam_control_logits,
                        margin=settings.counterfactual_margin,
                    )
                    counterfactual_loss = counterfactual.total
                else:
                    counterfactual_loss = logits.sum() * 0.0
                loss = (
                    classification_loss
                    + settings.region_loss_weight * region.total
                    + settings.counterfactual_loss_weight * counterfactual_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                presence_loss_value = float(classification_loss.detach().cpu())
                region_loss_value = float(region.total.detach().cpu())
                counterfactual_loss_value = float(counterfactual_loss.detach().cpu())
                batch_positive = int((presence > 0.5).sum())
                batch_negative = len(batch.sample_ids) - batch_positive
                train_loss += loss_value
                train_presence_loss += presence_loss_value
                train_region_loss += region_loss_value
                train_region_bce += float(region.bce.detach().cpu())
                train_region_dice += float(region.dice.detach().cpu())
                train_region_projection += float(region.projection.detach().cpu())
                train_counterfactual_loss += counterfactual_loss_value
                positive_samples += batch_positive
                negative_samples += batch_negative
                batches += 1
                global_step += 1
                batch_time = max(time.perf_counter() - batch_start, 1e-9)
                step_metrics = {
                    "record_type": "presence_train_step",
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "epoch_batch": float(batch_index),
                    "epoch_batches": float(len(train_loader)),
                    "total_loss": loss_value,
                    "presence_loss": presence_loss_value,
                    "region_loss": region_loss_value,
                    "region_bce": float(region.bce.detach().cpu()),
                    "region_dice": float(region.dice.detach().cpu()),
                    "region_projection": float(region.projection.detach().cpu()),
                    "counterfactual_loss": counterfactual_loss_value,
                    "positive_samples": float(batch_positive),
                    "negative_samples": float(batch_negative),
                    "negative_ratio": batch_negative / max(1, len(batch.sample_ids)),
                    "samples_per_second": len(batch.sample_ids) / batch_time,
                    "batch_time": batch_time,
                }
                progress.set_postfix(loss=f"{loss_value:.4f}")
                if global_step == 1 or batch_index == len(train_loader) or global_step % settings.log_interval == 0:
                    metrics_file.write(json.dumps(step_metrics, sort_keys=True) + "\n")
                    metrics_file.flush()
            previous_scores = load_previous_scores(previous_samples_path)
            last_metrics = validate_presence(
                model,
                val_loader,
                device,
                decision_threshold=settings.decision_threshold,
                region_loss_weight=settings.region_loss_weight,
                region_dice_weight=settings.region_dice_weight,
                region_projection_weight=settings.region_projection_weight,
                text_distractor_weight=settings.text_distractor_weight,
                counterfactual_loss_weight=settings.counterfactual_loss_weight,
                counterfactual_margin=settings.counterfactual_margin,
                diagnostics_path=previous_samples_path,
                previous_scores=previous_scores,
            )
            last_metrics.update(
                {
                    "record_type": "presence_validation",
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "train_loss": train_loss / max(1, batches),
                    "train_presence_loss": train_presence_loss / max(1, batches),
                    "train_region_loss": train_region_loss / max(1, batches),
                    "train_region_bce": train_region_bce / max(1, batches),
                    "train_region_dice": train_region_dice / max(1, batches),
                    "train_region_projection": train_region_projection / max(1, batches),
                    "train_counterfactual_loss": train_counterfactual_loss / max(1, batches),
                    "train_positive_samples": float(positive_samples),
                    "train_negative_samples": float(negative_samples),
                    "train_negative_ratio": negative_samples / max(1, positive_samples + negative_samples),
                    "train_samples": float(len(train_dataset)),
                    "val_samples": float(len(val_dataset)),
                    "train_data_positive_prior": train_dataset.summary.positive_ratio,
                    "score_target_positive_prior": target_positive_prior,
                    "train_text_distractor_negatives": float(train_dataset.text_distractor_negatives),
                    "val_text_distractor_negatives": float(val_dataset.text_distractor_negatives),
                    "validation_overlaps_training": overlap,
                    "epoch_seconds": time.perf_counter() - epoch_start,
                }
            )
            last_metrics["checkpoint_score"] = checkpoint_score(last_metrics)
            checkpoint_saved = best_metrics is None or checkpoint_rank(last_metrics) > checkpoint_rank(best_metrics)
            if checkpoint_saved:
                best_epoch = epoch
                best_metrics = dict(last_metrics)
            last_metrics["best_epoch"] = float(best_epoch)
            last_metrics["best_checkpoint_score"] = checkpoint_score(best_metrics or last_metrics)
            payload = checkpoint_payload(
                settings,
                model,
                optimizer,
                epoch=epoch,
                step=global_step,
                best_epoch=best_epoch,
                best_metrics=best_metrics or {},
                metrics=last_metrics,
            )
            if checkpoint_saved:
                torch.save(payload, settings.output_dir / "best.pt")
                torch.save(
                    inference_payload(settings, model, epoch=epoch, metrics=last_metrics),
                    settings.output_dir / "best_inference.pt",
                )
                shutil.copyfile(
                    previous_samples_path,
                    settings.output_dir / "best_presence_scores.jsonl",
                )
            torch.save(payload, settings.output_dir / "last.pt")
            metrics_file.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            metrics_file.flush()
            print(
                f"epoch={epoch}/{end_epoch} train_loss={last_metrics['train_loss']:.4f} "
                f"val_acc={last_metrics['presence_accuracy']:.4f} fp={int(last_metrics['presence_fp'])} "
                f"fn={int(last_metrics['presence_fn'])} gap={last_metrics['presence_gap']:.6f} "
                f"auc={last_metrics['presence_roc_auc']:.6f} "
                f"seconds={last_metrics['epoch_seconds']:.2f} best_epoch={best_epoch}",
                flush=True,
            )
    best = best_metrics or last_metrics
    summary = {
        "record_type": "presence_training_summary",
        "completed_epoch": end_epoch,
        "epochs_run": settings.epochs,
        "best_epoch": best_epoch,
        "best_step": int(best.get("step", 0)),
        "best_checkpoint": str(settings.output_dir / "best.pt"),
        "best_inference_checkpoint": str(settings.output_dir / "best_inference.pt"),
        "metrics_log": str(metrics_path),
        "presence_scores": str(settings.output_dir / "best_presence_scores.jsonl"),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_checkpoint_bytes": (settings.output_dir / "best.pt").stat().st_size,
        "best_inference_checkpoint_bytes": (settings.output_dir / "best_inference.pt").stat().st_size,
        "checkpoint_score": checkpoint_score(best),
        "validation": {
            key: float(best.get(key, 0.0))
            for key in (
                "presence_f1",
                "presence_accuracy",
                "presence_precision",
                "presence_recall",
                "presence_tp",
                "presence_fp",
                "presence_fn",
                "presence_tn",
                "presence_positive_score",
                "presence_negative_score",
                "presence_roc_auc",
                "presence_positive_p05",
                "presence_positive_p10",
                "presence_positive_p50",
                "presence_negative_p90",
                "presence_negative_p95",
                "presence_negative_p99",
                "presence_min_positive_score",
                "presence_max_negative_score",
                "presence_gap",
                "presence_robust_gap",
                "presence_positive_lower_tail_mean_1pct",
                "presence_negative_upper_tail_mean_1pct",
                "presence_tail_gap_1pct",
                "presence_brier",
                "presence_nll",
                "presence_ece",
                "presence_score_drift_max",
                "presence_score_drift_upper_tail_mean_1pct",
                "presence_threshold_flip_count",
                "short_presence_score_drift_upper_tail_mean_1pct",
                "short_presence_threshold_flip_count",
                "presence_batch_context_max_abs_logit_delta",
                "presence_best_f1_threshold",
                "presence_best_f1",
                "presence_zero_error_threshold_exists",
                "normal_presence_f1",
                "short_presence_f1",
                "segment_presence_f1",
                "segment_presence_recall",
                "region_iou",
                "region_dice",
                "region_pointing_accuracy",
                "region_contrast",
                "negative_region_max_score_p95",
                "negative_region_activation_area",
                "counterfactual_erased_flip_rate",
                "counterfactual_score_drop",
                "counterfactual_score_drop_lower_tail_1pct",
                "counterfactual_transplanted_recall",
                "counterfactual_seam_control_fpr",
                "text_distractor_count",
                "text_distractor_fpr",
                "subtitle_specificity_evaluable",
                "val_loss",
            )
        },
        "data": {
            "train_samples": float(best.get("train_samples", 0.0)),
            "val_samples": float(best.get("val_samples", 0.0)),
            "train_text_distractor_negatives": int(best.get("train_text_distractor_negatives", 0.0)),
            "val_text_distractor_negatives": int(best.get("val_text_distractor_negatives", 0.0)),
            "validation_overlaps_training": bool(best.get("validation_overlaps_training", False)),
        },
        "score_contract": {
            "kind": "subtitle_region_evidence_score",
            "target_positive_prior": target_positive_prior,
            "decision_threshold": settings.decision_threshold,
            "subtitle_specificity_evaluable": bool(best.get("subtitle_specificity_evaluable", 0.0)),
            "preprocessing": {
                "resize_roi": settings.resize_roi,
                "resize_mode": settings.resize_mode,
                "normalized_padding_value": 0.0,
                "valid_mask": "explicit_or_exact-zero-derived",
            },
        },
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    previous_samples_path.unlink(missing_ok=True)
    return last_metrics


def _resolve_negative_ratio(
    parser: argparse.ArgumentParser,
    *,
    positive_ratio: float | None,
    negative_ratio: float | None,
    default: float | None,
    positive_name: str,
    negative_name: str,
) -> float | None:
    if positive_ratio is not None and negative_ratio is not None:
        parser.error(f"use only one of {positive_name} or {negative_name}")
    value = 1.0 - positive_ratio if positive_ratio is not None else negative_ratio
    if value is None:
        return default
    if not 0.0 <= value <= 1.0:
        parser.error(f"{positive_name if positive_ratio is not None else negative_name} must be in [0, 1]")
    return value


def parse_args(argv: list[str] | None = None) -> RoiPresenceTrainSettings:
    defaults = RoiPresenceTrainSettings()
    parser = argparse.ArgumentParser(description="Train the ROI subtitle presence model.")
    parser.add_argument("--train-root", type=Path, action="append", dest="train_roots")
    parser.add_argument("--val-root", type=Path, default=defaults.val_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resize-roi", type=parse_roi_size)
    parser.add_argument("--resize-mode", choices=("letterbox", "stretch"), default=defaults.resize_mode)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--train-positive-ratio", type=float)
    parser.add_argument("--train-negative-ratio", type=float)
    parser.add_argument("--val-positive-ratio", type=float)
    parser.add_argument("--val-negative-ratio", type=float)
    parser.add_argument("--score-positive-prior", type=float, default=defaults.score_positive_prior)
    parser.add_argument("--region-loss-weight", type=float, default=defaults.region_loss_weight)
    parser.add_argument("--region-dice-weight", type=float, default=defaults.region_dice_weight)
    parser.add_argument(
        "--region-projection-weight",
        type=float,
        default=defaults.region_projection_weight,
    )
    parser.add_argument("--text-distractor-weight", type=float, default=defaults.text_distractor_weight)
    parser.add_argument(
        "--counterfactual-loss-weight",
        type=float,
        default=defaults.counterfactual_loss_weight,
    )
    parser.add_argument("--counterfactual-margin", type=float, default=defaults.counterfactual_margin)
    parser.add_argument("--evidence-kernel-size", type=int, default=defaults.evidence_kernel_size)
    parser.add_argument("--evidence-temperature", type=float, default=defaults.evidence_temperature)
    parser.add_argument("--decision-threshold", type=float, default=defaults.decision_threshold)
    parser.add_argument("--require-text-distractor-negatives", action="store_true")
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--log-interval", type=int, default=defaults.log_interval)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    args = parser.parse_args(argv)
    train_negative_ratio = _resolve_negative_ratio(
        parser,
        positive_ratio=args.train_positive_ratio,
        negative_ratio=args.train_negative_ratio,
        default=defaults.train_negative_ratio,
        positive_name="--train-positive-ratio",
        negative_name="--train-negative-ratio",
    )
    val_negative_ratio = _resolve_negative_ratio(
        parser,
        positive_ratio=args.val_positive_ratio,
        negative_ratio=args.val_negative_ratio,
        default=defaults.val_negative_ratio,
        positive_name="--val-positive-ratio",
        negative_name="--val-negative-ratio",
    )
    for name, value in (
        ("--batch-size", args.batch_size),
        ("--epochs", args.epochs),
        ("--width", args.width),
        ("--log-interval", args.log_interval),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be non-negative")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    for name, value in (
        ("--region-loss-weight", args.region_loss_weight),
        ("--region-dice-weight", args.region_dice_weight),
        ("--region-projection-weight", args.region_projection_weight),
        ("--text-distractor-weight", args.text_distractor_weight),
        ("--counterfactual-loss-weight", args.counterfactual_loss_weight),
        ("--counterfactual-margin", args.counterfactual_margin),
    ):
        if value < 0.0:
            parser.error(f"{name} must be non-negative")
    if args.evidence_kernel_size <= 1 or args.evidence_kernel_size % 2 == 0:
        parser.error("--evidence-kernel-size must be an odd integer greater than 1")
    if args.evidence_temperature <= 0.0:
        parser.error("--evidence-temperature must be positive")
    if args.score_positive_prior is not None and not 0.0 < args.score_positive_prior < 1.0:
        parser.error("--score-positive-prior must be in (0, 1)")
    if not 0.0 < args.decision_threshold < 1.0:
        parser.error("--decision-threshold must be in (0, 1)")
    for name, value in (
        ("--max-train-samples", args.max_train_samples),
        ("--max-val-samples", args.max_val_samples),
    ):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    return RoiPresenceTrainSettings(
        train_roots=args.train_roots or defaults.train_roots,
        val_root=args.val_root,
        output_dir=args.output_dir,
        resume=args.resume,
        resize_roi=args.resize_roi,
        resize_mode=args.resize_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        train_negative_ratio=train_negative_ratio,
        val_negative_ratio=val_negative_ratio,
        score_positive_prior=args.score_positive_prior,
        region_loss_weight=args.region_loss_weight,
        region_dice_weight=args.region_dice_weight,
        region_projection_weight=args.region_projection_weight,
        text_distractor_weight=args.text_distractor_weight,
        counterfactual_loss_weight=args.counterfactual_loss_weight,
        counterfactual_margin=args.counterfactual_margin,
        evidence_kernel_size=args.evidence_kernel_size,
        evidence_temperature=args.evidence_temperature,
        decision_threshold=args.decision_threshold,
        require_text_distractor_negatives=args.require_text_distractor_negatives,
        width=args.width,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    metrics = run_training(parse_args(argv))
    print(json.dumps(metrics, indent=2, sort_keys=True))
