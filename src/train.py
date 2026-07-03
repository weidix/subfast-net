from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainSettings
from .dataset import SubtitleDataset, collate_batch
from .loss import detection_loss
from .metrics import ImageMetrics, evaluate_image
from .model import SubtitleDetector
from .postprocess import logits_to_boxes


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(settings: TrainSettings, train: bool) -> SubtitleDataset:
    roots = settings.train_roots if train else [settings.val_root]
    return SubtitleDataset(
        roots=roots,
        image_size=settings.image_size,
        stride=settings.stride,
        max_samples=settings.max_train_samples if train else settings.max_val_samples,
        empty_ratio=settings.train_empty_sample_ratio if train else settings.val_empty_sample_ratio,
        pooling_size=settings.pooling_size,
        kernel_scale=settings.kernel_scale,
        min_kernel_width=settings.min_kernel_width,
        min_kernel_height=settings.min_kernel_height,
    )


def format_dataset_summary(name: str, dataset: SubtitleDataset) -> str:
    summary = dataset.summary
    roots = ", ".join(f"{root}={count}" for root, count in sorted(summary.roots.items()))
    return (
        f"{name}: samples={summary.total} labeled={summary.labeled} empty={summary.empty} "
        f"labeled_ratio={summary.labeled_ratio:.3f} empty_ratio={summary.empty_ratio:.3f} roots=[{roots}]"
    )


def format_epoch_summary(epoch: int, total_epochs: int, metrics: dict[str, float]) -> str:
    return "\n".join(
        [
            f"epoch {epoch}/{total_epochs}",
            f"  loss: train={metrics['train_loss']:.4f} val={metrics['val_loss']:.4f}",
            (
                f"  train_parts: region_bce={metrics['train_region_bce']:.4f} "
                f"kernel_bce={metrics['train_kernel_bce']:.4f} "
                f"region_dice={metrics['train_region_dice']:.4f} "
                f"kernel_dice={metrics['train_kernel_dice']:.4f}"
            ),
            (
                f"  validation: precision={metrics['precision']:.4f} "
                f"recall={metrics['recall']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"tp={metrics['true_positive']:.0f} "
                f"fp={metrics['false_positive']:.0f} "
                f"fn={metrics['false_negative']:.0f}"
            ),
        ]
    )


def format_train_step(epoch: int, total_epochs: int, batch_index: int, total_batches: int, samples_seen: int, samples_total: int, metrics: dict[str, float]) -> str:
    return (
        f"epoch={epoch}/{total_epochs} batch={batch_index}/{total_batches} "
        f"samples={samples_seen}/{samples_total} step={metrics['step']:.0f} "
        f"total_loss={metrics['total_loss']:.5f} "
        f"region_bce={metrics['region_bce']:.5f} kernel_bce={metrics['kernel_bce']:.5f} "
        f"region_dice={metrics['region_dice']:.5f} kernel_dice={metrics['kernel_dice']:.5f} "
        f"positive_region_ratio={metrics['positive_region_ratio']:.5f} "
        f"positive_kernel_ratio={metrics['positive_kernel_ratio']:.5f} "
        f"samples/s={metrics['samples_per_second']:.2f}"
    )


def batch_positive_ratio(target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask > 0.5
    if not torch.any(valid):
        return 0.0
    return float(((target > 0.5) & valid).sum().detach().cpu()) / float(valid.sum().detach().cpu())


def box_to_json(box) -> dict[str, float]:
    return {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2}


@torch.no_grad()
def validate(
    model: SubtitleDetector,
    loader: DataLoader,
    device: torch.device,
    settings: TrainSettings,
    *,
    collect_outputs: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    aggregate = ImageMetrics(0, 0, 0)
    total_loss = 0.0
    batches = 0
    outputs: list[dict[str, Any]] = []
    output_limit = settings.max_epoch_output_samples
    for batch in loader:
        images = batch.images.to(device)
        regions = batch.regions.to(device)
        kernels = batch.kernels.to(device)
        masks = batch.training_masks.to(device)
        logits = model(images)
        loss = detection_loss(logits, regions, kernels, masks)
        total_loss += float(loss.total.detach().cpu())
        batches += 1
        cpu_logits = logits.detach().cpu().numpy()
        for index, targets in enumerate(batch.boxes):
            detections = logits_to_boxes(
                cpu_logits[index, 0],
                cpu_logits[index, 1],
                region_threshold=settings.region_threshold,
                kernel_threshold=settings.kernel_threshold,
                max_width_ratio=settings.max_detection_width_ratio,
            )
            result = evaluate_image([det.box for det in detections], targets, settings.iou_threshold)
            aggregate = ImageMetrics(
                aggregate.true_positive + result.true_positive,
                aggregate.false_positive + result.false_positive,
                aggregate.false_negative + result.false_negative,
            )
            if collect_outputs and (output_limit is None or len(outputs) < output_limit):
                outputs.append(
                    {
                        "sample_id": batch.sample_ids[index],
                        "target_boxes": [box_to_json(box) for box in targets],
                        "detections": [
                            {"box": box_to_json(det.box), "score": det.score}
                            for det in detections
                        ],
                        "true_positive": result.true_positive,
                        "false_positive": result.false_positive,
                        "false_negative": result.false_negative,
                    }
                )
    metrics = {
        "val_loss": total_loss / max(1, batches),
        "precision": aggregate.precision,
        "recall": aggregate.recall,
        "f1": aggregate.f1,
        "true_positive": float(aggregate.true_positive),
        "false_positive": float(aggregate.false_positive),
        "false_negative": float(aggregate.false_negative),
    }
    return metrics, outputs


def epoch_output_dir(settings: TrainSettings, epoch: int) -> Path:
    return settings.output_dir / "epoch_outputs" / f"epoch_{epoch:04}"


def write_epoch_outputs(settings: TrainSettings, epoch: int, step: int, metrics: dict[str, float], outputs: list[dict[str, Any]]) -> Path:
    output_dir = epoch_output_dir(settings, epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "outputs.json"
    payload = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "region_threshold": settings.region_threshold,
        "kernel_threshold": settings.kernel_threshold,
        "iou_threshold": settings.iou_threshold,
        "samples": outputs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def resolve_resume_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    epoch_checkpoint = path / "model.pt"
    if epoch_checkpoint.is_file():
        return epoch_checkpoint
    epoch_output_root = path / "epoch_outputs"
    if epoch_output_root.is_dir():
        candidates = sorted(epoch_output_root.glob("epoch_*/model.pt"))
        if candidates:
            return candidates[-1]
    best_checkpoint = path / "best.pt"
    if best_checkpoint.is_file():
        return best_checkpoint
    raise FileNotFoundError(f"resume checkpoint not found: {path}")


def checkpoint_payload(
    settings: TrainSettings,
    epoch: int,
    step: int,
    best_f1: float,
    best_epoch: int,
    model: SubtitleDetector,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "settings": settings.model_dump(mode="json"),
        "metrics": metrics,
        "epoch": epoch,
        "step": step,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
    }


def load_resume_checkpoint(
    resume: Path,
    model: SubtitleDetector,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[Path, int, int, float, int, dict[str, float]]:
    checkpoint_path = resolve_resume_checkpoint(resume)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"invalid resume checkpoint: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    metrics = dict(checkpoint.get("metrics") or {})
    completed_epoch = int(checkpoint.get("epoch", metrics.get("epoch", 0)))
    step = int(checkpoint.get("step", metrics.get("step", 0)))
    best_f1 = float(checkpoint.get("best_f1", metrics.get("f1", -1.0)))
    best_epoch = int(checkpoint.get("best_epoch", metrics.get("best_epoch", completed_epoch)))
    return checkpoint_path, completed_epoch + 1, step, best_f1, best_epoch, metrics


def save_epoch_checkpoint(
    settings: TrainSettings,
    epoch: int,
    step: int,
    best_f1: float,
    best_epoch: int,
    model: SubtitleDetector,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
) -> Path:
    output_dir = epoch_output_dir(settings, epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model.pt"
    torch.save(
        checkpoint_payload(settings, epoch, step, best_f1, best_epoch, model, optimizer, metrics),
        path,
    )
    return path


def run_training(settings: TrainSettings) -> dict[str, float]:
    seed_everything(settings.seed)
    device = choose_device(settings.device)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = make_dataset(settings, train=True)
    val_dataset = make_dataset(settings, train=False)
    if len(train_dataset) == 0:
        raise RuntimeError("no training samples found")
    if len(val_dataset) == 0:
        raise RuntimeError("no validation samples found")
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=settings.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers, collate_fn=collate_batch)
    model = SubtitleDetector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    best_f1 = -1.0
    best_epoch = 0
    last_metrics: dict[str, float] = {}
    global_step = 0
    start_epoch = 1
    resume_checkpoint_path: Path | None = None
    if settings.resume is not None:
        (
            resume_checkpoint_path,
            start_epoch,
            global_step,
            best_f1,
            best_epoch,
            last_metrics,
        ) = load_resume_checkpoint(settings.resume, model, optimizer, device)
    metrics_path = settings.output_dir / "metrics.jsonl"
    print(f"device={device} output_dir={settings.output_dir}", flush=True)
    if resume_checkpoint_path is not None:
        print(
            f"resume={resume_checkpoint_path} start_epoch={start_epoch} step={global_step} "
            f"best_epoch={best_epoch} best_f1={best_f1:.4f}",
            flush=True,
        )
    print(
        f"config: image_size={settings.image_size} batch_size={settings.batch_size} epochs={settings.epochs} "
        f"lr={settings.learning_rate:g} weight_decay={settings.weight_decay:g} log_interval={settings.log_interval} "
        f"max_train_samples={settings.max_train_samples} max_val_samples={settings.max_val_samples} "
        f"train_empty_sample_ratio={settings.train_empty_sample_ratio} val_empty_sample_ratio={settings.val_empty_sample_ratio} "
        f"pooling_size={settings.pooling_size} "
        f"region_threshold={settings.region_threshold:g} kernel_threshold={settings.kernel_threshold:g}",
        flush=True,
    )
    print(format_dataset_summary("train", train_dataset), flush=True)
    print(format_dataset_summary("val", val_dataset), flush=True)
    metrics_mode = "a" if settings.resume is not None else "w"
    with metrics_path.open(metrics_mode) as metrics_file:
        for epoch in range(start_epoch, settings.epochs + 1):
            model.train()
            train_loss = 0.0
            train_region_bce = 0.0
            train_kernel_bce = 0.0
            train_region_dice = 0.0
            train_kernel_dice = 0.0
            epoch_start = time.perf_counter()
            batches = 0
            progress = tqdm(train_loader, desc=f"epoch {epoch}/{settings.epochs}", leave=False)
            total_batches = len(train_loader)
            for batch_index, batch in enumerate(progress, start=1):
                batch_start = time.perf_counter()
                images = batch.images.to(device)
                regions = batch.regions.to(device)
                kernels = batch.kernels.to(device)
                masks = batch.training_masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = detection_loss(model(images), regions, kernels, masks)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                value = float(loss.total.detach().cpu())
                train_loss += value
                train_region_bce += float(loss.region_bce.detach().cpu())
                train_kernel_bce += float(loss.kernel_bce.detach().cpu())
                train_region_dice += float(loss.region_dice.detach().cpu())
                train_kernel_dice += float(loss.kernel_dice.detach().cpu())
                batches += 1
                global_step += 1
                samples_seen = min(batch_index * settings.batch_size, len(train_dataset))
                batch_time = max(time.perf_counter() - batch_start, 1e-9)
                step_metrics = {
                    "record_type": "train_step",
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "epoch_batch": float(batch_index),
                    "epoch_batches": float(total_batches),
                    "epoch_samples_processed": float(samples_seen),
                    "epoch_samples_total": float(len(train_dataset)),
                    "total_loss": value,
                    "region_bce": float(loss.region_bce.detach().cpu()),
                    "kernel_bce": float(loss.kernel_bce.detach().cpu()),
                    "region_dice": float(loss.region_dice.detach().cpu()),
                    "kernel_dice": float(loss.kernel_dice.detach().cpu()),
                    "positive_region_ratio": batch_positive_ratio(regions, masks),
                    "positive_kernel_ratio": batch_positive_ratio(kernels, masks),
                    "samples_per_second": float(len(batch.sample_ids)) / batch_time,
                    "batch_time": batch_time,
                }
                progress.set_postfix(loss=f"{value:.4f}")
                should_log = global_step == 1 or batch_index == total_batches or global_step % max(1, settings.log_interval) == 0
                if should_log:
                    metrics_file.write(json.dumps(step_metrics, sort_keys=True) + "\n")
                    metrics_file.flush()
            last_metrics, epoch_outputs = validate(model, val_loader, device, settings, collect_outputs=settings.save_epoch_outputs)
            last_metrics.update(
                {
                    "record_type": "validation",
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "train_loss": train_loss / max(1, batches),
                    "train_region_bce": train_region_bce / max(1, batches),
                    "train_kernel_bce": train_kernel_bce / max(1, batches),
                    "train_region_dice": train_region_dice / max(1, batches),
                    "train_kernel_dice": train_kernel_dice / max(1, batches),
                    "train_samples": float(len(train_dataset)),
                    "val_samples": float(len(val_dataset)),
                    "epoch_seconds": time.perf_counter() - epoch_start,
                }
            )
            checkpoint_saved = last_metrics["f1"] >= best_f1
            if checkpoint_saved:
                best_f1 = last_metrics["f1"]
                best_epoch = epoch
            last_metrics["best_epoch"] = float(best_epoch)
            epoch_output_path: Path | None = None
            if settings.save_epoch_outputs:
                epoch_output_path = write_epoch_outputs(settings, epoch, global_step, last_metrics, epoch_outputs)
                last_metrics["epoch_output_samples"] = float(len(epoch_outputs))
            epoch_checkpoint_path = save_epoch_checkpoint(settings, epoch, global_step, best_f1, best_epoch, model, optimizer, last_metrics)
            metrics_file.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            metrics_file.flush()
            if checkpoint_saved:
                torch.save(
                    checkpoint_payload(settings, epoch, global_step, best_f1, best_epoch, model, optimizer, last_metrics),
                    settings.output_dir / "best.pt",
                )
            epoch_message = format_epoch_summary(epoch, settings.epochs, last_metrics)
            if epoch_output_path is not None:
                epoch_message += f"\n  output: epoch_output={epoch_output_path} samples={len(epoch_outputs)}"
            epoch_message += f"\n  checkpoint: epoch={epoch_checkpoint_path}"
            if checkpoint_saved:
                epoch_message += f" best=true step={global_step}"
            print(epoch_message, flush=True)
    (settings.output_dir / "summary.json").write_text(json.dumps(last_metrics, indent=2, sort_keys=True))
    return last_metrics


def parse_args(argv: list[str] | None = None) -> TrainSettings:
    parser = argparse.ArgumentParser(description="Train a local PyTorch subtitle-region detector.")
    parser.add_argument("--train-root", type=Path, action="append", dest="train_roots")
    parser.add_argument("--val-root", type=Path, default=Path("data/validation_samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pytorch_run"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-samples", type=int, help="Maximum training sample count.")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--positive-ratio", type=float, help="Target labeled/positive training sample ratio in [0, 1].")
    parser.add_argument("--negative-ratio", type=float, help="Target empty/negative training sample ratio in [0, 1].")
    parser.add_argument("--val-positive-ratio", type=float, help="Target labeled/positive validation sample ratio in [0, 1].")
    parser.add_argument("--val-negative-ratio", type=float, help="Target empty/negative validation sample ratio in [0, 1].")
    parser.add_argument("--pooling-size", type=int, default=TrainSettings().pooling_size)
    parser.add_argument("--region-threshold", type=float, default=TrainSettings().region_threshold)
    parser.add_argument("--kernel-threshold", type=float, default=TrainSettings().kernel_threshold)
    parser.add_argument("--iou-threshold", type=float, default=TrainSettings().iou_threshold)
    parser.add_argument("--max-detection-width-ratio", type=float, default=TrainSettings().max_detection_width_ratio)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-epoch-outputs", action="store_true")
    parser.add_argument("--max-epoch-output-samples", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    def resolve_empty_ratio(
        *,
        positive_ratio: float | None,
        negative_ratio: float | None,
        default: float | None,
        positive_name: str,
        negative_name: str,
    ) -> float | None:
        ratio_args = [value is not None for value in (positive_ratio, negative_ratio)]
        if sum(ratio_args) > 1:
            parser.error(f"use only one of {positive_name} or {negative_name}")
        if positive_ratio is not None:
            if not 0.0 <= positive_ratio <= 1.0:
                parser.error(f"{positive_name} must be in [0, 1]")
            return 1.0 - positive_ratio
        if negative_ratio is not None:
            if not 0.0 <= negative_ratio <= 1.0:
                parser.error(f"{negative_name} must be in [0, 1]")
            return negative_ratio
        return default

    train_empty_ratio = resolve_empty_ratio(
        positive_ratio=args.positive_ratio,
        negative_ratio=args.negative_ratio,
        default=TrainSettings().train_empty_sample_ratio,
        positive_name="--positive-ratio",
        negative_name="--negative-ratio",
    )
    val_empty_ratio = resolve_empty_ratio(
        positive_ratio=args.val_positive_ratio,
        negative_ratio=args.val_negative_ratio,
        default=TrainSettings().val_empty_sample_ratio,
        positive_name="--val-positive-ratio",
        negative_name="--val-negative-ratio",
    )
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else args.max_samples
    return TrainSettings(
        train_roots=args.train_roots if args.train_roots is not None else TrainSettings().train_roots,
        val_root=args.val_root,
        output_dir=args.output_dir,
        resume=args.resume,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_train_samples=max_train_samples,
        max_val_samples=args.max_val_samples,
        log_interval=args.log_interval,
        save_epoch_outputs=not args.no_epoch_outputs,
        max_epoch_output_samples=args.max_epoch_output_samples,
        train_empty_sample_ratio=train_empty_ratio,
        val_empty_sample_ratio=val_empty_ratio,
        pooling_size=args.pooling_size,
        region_threshold=args.region_threshold,
        kernel_threshold=args.kernel_threshold,
        iou_threshold=args.iou_threshold,
        max_detection_width_ratio=args.max_detection_width_ratio,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    metrics = run_training(parse_args(argv))
    print(json.dumps(metrics, indent=2, sort_keys=True))
