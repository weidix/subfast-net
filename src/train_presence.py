from __future__ import annotations

import argparse
import json
import random
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
from .roi_presence_loss import presence_loss_weights, short_positive_mask_loss
from .roi_presence_metrics import checkpoint_rank, checkpoint_score
from .roi_presence_model import RoiPresenceModel
from .roi_presence_sampler import PresenceBalancedBatchSampler
from .roi_presence_validation import validate_presence
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
        max_samples=settings.max_train_samples if train else settings.max_val_samples,
        negative_ratio=None if train else settings.val_negative_ratio,
        load_subtitle_masks=train and settings.short_positive_mask_loss_weight > 0.0,
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
        f"roi_size={summary.roi_size[0]}x{summary.roi_size[1]} roots=[{roots}]"
    )


def resolve_resume_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    epoch_checkpoint = path / "model.pt"
    if epoch_checkpoint.is_file():
        return epoch_checkpoint
    candidates = sorted((path / "epoch_outputs").glob("epoch_*/model.pt"))
    if candidates:
        return candidates[-1]
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
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "settings": settings.model_dump(mode="json"),
        "metrics": metrics,
        "epoch": epoch,
        "step": step,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }


def run_training(settings: RoiPresenceTrainSettings) -> dict[str, float]:
    seed_everything(settings.seed)
    device = choose_device(settings.device)
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
        presence_topk_ratio=settings.presence_topk_ratio,
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
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("step", 0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_metrics = dict(checkpoint.get("best_metrics") or checkpoint.get("metrics") or {})
        last_metrics = dict(checkpoint.get("metrics") or {})
        print(f"resume={checkpoint_path} start_epoch={start_epoch} step={global_step}", flush=True)
    end_epoch = start_epoch + settings.epochs - 1
    print(f"roi_presence device={device} output_dir={settings.output_dir}", flush=True)
    print(
        f"config: batch_size={settings.batch_size} epochs={settings.epochs} "
        f"lr={settings.learning_rate:g} weight_decay={settings.weight_decay:g} "
        f"width={settings.width} presence_topk_ratio={settings.presence_topk_ratio:g} "
        f"short_positive_loss_weight={settings.short_positive_loss_weight:g} "
        f"short_positive_mask_loss_weight={settings.short_positive_mask_loss_weight:g} "
        f"train_negative_ratio={settings.train_negative_ratio} "
        f"val_negative_ratio={settings.val_negative_ratio} resize_roi={settings.resize_roi}",
        flush=True,
    )
    overlap = any(root.resolve() == settings.val_root.resolve() for root in settings.train_roots)
    if overlap:
        print("warning: validation root overlaps a training root; this run is not held-out quality evidence", flush=True)
    print(format_dataset_summary("train", train_dataset), flush=True)
    print(format_dataset_summary("val", val_dataset), flush=True)
    metrics_path = settings.output_dir / "metrics.jsonl"
    metrics_mode = "a" if settings.resume is not None else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, end_epoch + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            epoch_start = time.perf_counter()
            train_loss = 0.0
            train_presence_loss = 0.0
            train_mask_loss = 0.0
            positive_samples = 0
            negative_samples = 0
            batches = 0
            progress = tqdm(train_loader, desc=f"presence epoch {epoch}/{end_epoch}", unit="batch")
            for batch_index, batch in enumerate(progress, start=1):
                batch_start = time.perf_counter()
                images = batch.images.to(device)
                presence = batch.presence.to(device)
                optimizer.zero_grad(set_to_none=True)
                if settings.short_positive_mask_loss_weight > 0.0:
                    if batch.subtitle_masks is None:
                        raise RuntimeError("subtitle masks are required when short positive mask loss is enabled")
                    logits, textness_map = model.forward_with_presence_map(images)
                    mask_loss = short_positive_mask_loss(
                        textness_map,
                        batch.subtitle_masks,
                        presence,
                        batch.ocr_texts,
                        settings.short_positive_mask_loss_weight,
                    )
                else:
                    logits = model(images)
                    mask_loss = logits.sum() * 0.0
                classification_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    presence,
                    weight=presence_loss_weights(
                        presence,
                        batch.ocr_texts,
                        settings.short_positive_loss_weight,
                    ),
                )
                loss = classification_loss + mask_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                presence_loss_value = float(classification_loss.detach().cpu())
                mask_loss_value = float(mask_loss.detach().cpu())
                batch_positive = int((presence > 0.5).sum())
                batch_negative = len(batch.sample_ids) - batch_positive
                train_loss += loss_value
                train_presence_loss += presence_loss_value
                train_mask_loss += mask_loss_value
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
                    "mask_loss": mask_loss_value,
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
            last_metrics = validate_presence(model, val_loader, device)
            last_metrics.update(
                {
                    "record_type": "presence_validation",
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "train_loss": train_loss / max(1, batches),
                    "train_presence_loss": train_presence_loss / max(1, batches),
                    "train_mask_loss": train_mask_loss / max(1, batches),
                    "train_positive_samples": float(positive_samples),
                    "train_negative_samples": float(negative_samples),
                    "train_negative_ratio": negative_samples / max(1, positive_samples + negative_samples),
                    "train_samples": float(len(train_dataset)),
                    "val_samples": float(len(val_dataset)),
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
            epoch_dir = settings.output_dir / "epoch_outputs" / f"epoch_{epoch:04}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
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
            torch.save(payload, epoch_dir / "model.pt")
            (epoch_dir / "metrics.json").write_text(
                json.dumps(last_metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if checkpoint_saved:
                torch.save(payload, settings.output_dir / "best.pt")
                torch.save(payload, settings.output_dir / "best_presence.pt")
            metrics_file.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            metrics_file.flush()
            print(
                f"presence epoch {epoch}/{end_epoch}\n"
                f"  loss: train={last_metrics['train_loss']:.4f} val={last_metrics['val_loss']:.4f}\n"
                f"  presence: f1={last_metrics['presence_f1']:.4f} "
                f"accuracy={last_metrics['presence_accuracy']:.4f} "
                f"tp={last_metrics['presence_tp']:.0f} fp={last_metrics['presence_fp']:.0f} "
                f"fn={last_metrics['presence_fn']:.0f} tn={last_metrics['presence_tn']:.0f}\n"
                f"  separation: positive_min={last_metrics['presence_min_positive_score']:.6f} "
                f"negative_max={last_metrics['presence_max_negative_score']:.6f} "
                f"gap={last_metrics['presence_gap']:+.6f} "
                f"auc={last_metrics['presence_roc_auc']:.6f} "
                f"best_threshold={last_metrics['presence_best_f1_threshold']:.6f} "
                f"zero_error={str(bool(last_metrics['presence_zero_error_threshold_exists'])).lower()}\n"
                f"  best: epoch={best_epoch} score={last_metrics['best_checkpoint_score']:.4f} "
                f"gap={float((best_metrics or last_metrics).get('presence_gap', 0.0)):+.6f} "
                f"saved={str(checkpoint_saved).lower()}",
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
        "best_epoch_checkpoint": str(
            settings.output_dir / "epoch_outputs" / f"epoch_{best_epoch:04}" / "model.pt"
        ),
        "metrics_log": str(metrics_path),
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
                "presence_best_f1_threshold",
                "presence_best_f1",
                "presence_zero_error_threshold_exists",
                "normal_presence_f1",
                "short_presence_f1",
                "val_loss",
            )
        },
        "data": {
            "train_samples": float(best.get("train_samples", 0.0)),
            "val_samples": float(best.get("val_samples", 0.0)),
            "validation_overlaps_training": bool(best.get("validation_overlaps_training", False)),
        },
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
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
    parser.add_argument("--short-positive-loss-weight", type=float, default=defaults.short_positive_loss_weight)
    parser.add_argument(
        "--short-positive-mask-loss-weight",
        type=float,
        default=defaults.short_positive_mask_loss_weight,
    )
    parser.add_argument("--presence-topk-ratio", type=float, default=defaults.presence_topk_ratio)
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
    if args.short_positive_loss_weight <= 0.0:
        parser.error("--short-positive-loss-weight must be positive")
    if args.short_positive_mask_loss_weight < 0.0:
        parser.error("--short-positive-mask-loss-weight must be non-negative")
    if not 0.0 < args.presence_topk_ratio <= 1.0:
        parser.error("--presence-topk-ratio must be in (0, 1]")
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
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        train_negative_ratio=train_negative_ratio,
        val_negative_ratio=val_negative_ratio,
        short_positive_loss_weight=args.short_positive_loss_weight,
        short_positive_mask_loss_weight=args.short_positive_mask_loss_weight,
        presence_topk_ratio=args.presence_topk_ratio,
        width=args.width,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    metrics = run_training(parse_args(argv))
    print(json.dumps(metrics, indent=2, sort_keys=True))
