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

from .roi_config import RoiTrainSettings
from .roi_dataset import RoiPresenceEmbeddingDataset, collate_roi_batch
from .roi_loss import roi_presence_embedding_loss
from .roi_metrics import embedding_metrics, presence_metrics
from .roi_model import RoiPresenceEmbeddingModel
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


def make_dataset(settings: RoiTrainSettings, train: bool) -> RoiPresenceEmbeddingDataset:
    roots = settings.train_roots if train else [settings.val_root]
    return RoiPresenceEmbeddingDataset(
        roots,
        resize_roi=settings.resize_roi,
        max_samples=settings.max_train_samples if train else settings.max_val_samples,
        empty_ratio=settings.negative_ratio if train else settings.val_negative_ratio,
        segment_aware_limit=not train,
    )


def format_dataset_summary(name: str, dataset: RoiPresenceEmbeddingDataset) -> str:
    summary = dataset.summary
    roots = ", ".join(f"{root}={count}" for root, count in sorted(summary.roots.items()))
    return (
        f"{name}: samples={summary.total} positive={summary.positive} empty={summary.empty} "
        f"positive_ratio={summary.positive_ratio:.3f} empty_ratio={summary.empty_ratio:.3f} "
        f"positive_segments={summary.positive_segments} repeated_positive_segments={summary.repeated_positive_segments} "
        f"same_segment_pairs={summary.same_segment_pairs} "
        f"roi_size={summary.roi_size[0]}x{summary.roi_size[1]} roots=[{roots}]"
    )


def format_epoch_summary(epoch: int, total_epochs: int, metrics: dict[str, float]) -> str:
    return (
        f"epoch {epoch}/{total_epochs} "
        f"train_loss={metrics['train_loss']:.4f} "
        f"presence_loss={metrics['train_presence_loss']:.4f} "
        f"embedding_loss={metrics['train_embedding_loss']:.4f} "
        f"val_loss={metrics['val_loss']:.4f} "
        f"presence_f1={metrics['presence_f1']:.4f} "
        f"presence_accuracy={metrics['presence_accuracy']:.4f} "
        f"embedding_acc={metrics['embedding_pair_accuracy']:.4f} "
        f"same_sim={metrics['embedding_same_similarity']:.4f} "
        f"diff_sim={metrics['embedding_diff_similarity']:.4f}"
    )


def epoch_output_dir(settings: RoiTrainSettings, epoch: int) -> Path:
    return settings.output_dir / "epoch_outputs" / f"epoch_{epoch:04}"


def checkpoint_payload(
    settings: RoiTrainSettings,
    epoch: int,
    step: int,
    best_presence_f1: float,
    best_epoch: int,
    model: RoiPresenceEmbeddingModel,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_type": "roi_presence_embedding",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "settings": settings.model_dump(mode="json"),
        "metrics": metrics,
        "epoch": epoch,
        "step": step,
        "best_presence_f1": best_presence_f1,
        "best_epoch": best_epoch,
    }


def make_model(settings: RoiTrainSettings) -> RoiPresenceEmbeddingModel:
    return RoiPresenceEmbeddingModel(
        width=settings.width,
        embedding_dim=settings.embedding_dim,
        embedding_head_type=settings.embedding_head_type,
        embedding_sequence_channels=settings.embedding_sequence_channels,
    )


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


def load_resume_checkpoint(
    resume: Path,
    model: RoiPresenceEmbeddingModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    current_embedding_head_type: str,
) -> tuple[Path, int, int, float, int, dict[str, float]]:
    checkpoint_path = resolve_resume_checkpoint(resume)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "roi_presence_embedding":
        raise RuntimeError(f"invalid ROI Presence+Embedding checkpoint: {checkpoint_path}")
    raw_settings = dict(checkpoint.get("settings") or {})
    checkpoint_embedding_head_type = str(raw_settings.get("embedding_head_type", "gap"))
    partial_model_load = checkpoint_embedding_head_type != current_embedding_head_type
    model.load_state_dict(checkpoint["model"], strict=not partial_model_load)
    if "optimizer" in checkpoint and not partial_model_load:
        optimizer.load_state_dict(checkpoint["optimizer"])
    metrics = dict(checkpoint.get("metrics") or {})
    completed_epoch = int(checkpoint.get("epoch", metrics.get("epoch", 0)))
    step = int(checkpoint.get("step", metrics.get("step", 0)))
    best_presence_f1 = float(checkpoint.get("best_presence_f1", metrics.get("presence_f1", -1.0)))
    best_epoch = int(checkpoint.get("best_epoch", metrics.get("best_epoch", completed_epoch)))
    return checkpoint_path, completed_epoch + 1, step, best_presence_f1, best_epoch, metrics


def load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[RoiPresenceEmbeddingModel, dict[str, Any]]:
    checkpoint_path = resolve_resume_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "roi_presence_embedding":
        raise RuntimeError(f"invalid ROI Presence+Embedding checkpoint: {checkpoint_path}")
    raw_settings = dict(checkpoint.get("settings") or {})
    model = RoiPresenceEmbeddingModel(
        width=int(raw_settings.get("width", RoiTrainSettings().width)),
        embedding_dim=int(raw_settings.get("embedding_dim", RoiTrainSettings().embedding_dim)),
        embedding_head_type=str(raw_settings.get("embedding_head_type", RoiTrainSettings().embedding_head_type)),
        embedding_sequence_channels=int(raw_settings.get("embedding_sequence_channels", RoiTrainSettings().embedding_sequence_channels)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.no_grad()
def measure_roi_forward_time(
    model: RoiPresenceEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
    *,
    warmup_batches: int = 5,
    measure_batches: int = 25,
) -> float:
    model.eval()
    batches = iter(loader)
    for _ in range(warmup_batches):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        _ = model(batch.images.to(device))
    synchronize_device(device)
    total_images = 0
    start = time.perf_counter()
    for _ in range(measure_batches):
        try:
            batch = next(batches)
        except StopIteration:
            break
        images = batch.images.to(device)
        _ = model(images)
        total_images += int(images.shape[0])
    synchronize_device(device)
    elapsed = time.perf_counter() - start
    return (elapsed * 1000.0 / total_images) if total_images else 0.0


def save_epoch_checkpoint(
    settings: RoiTrainSettings,
    epoch: int,
    step: int,
    best_presence_f1: float,
    best_epoch: int,
    model: RoiPresenceEmbeddingModel,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
) -> Path:
    output_dir = epoch_output_dir(settings, epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model.pt"
    torch.save(
        checkpoint_payload(settings, epoch, step, best_presence_f1, best_epoch, model, optimizer, metrics),
        path,
    )
    return path


@torch.no_grad()
def validate(
    model: RoiPresenceEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
    settings: RoiTrainSettings,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_presence_loss = 0.0
    total_embedding_loss = 0.0
    batches = 0
    logits_all: list[torch.Tensor] = []
    presence_all: list[torch.Tensor] = []
    embedding_all: list[torch.Tensor] = []
    segment_ids: list[str] = []
    roots: list[str] = []
    video_ids: list[str | None] = []
    frame_indices: list[int | None] = []
    ocr_texts: list[str] = []
    for batch in loader:
        images = batch.images.to(device)
        presence = batch.presence.to(device)
        presence_logit, embedding = model(images)
        loss = roi_presence_embedding_loss(
            presence_logit,
            embedding,
            presence,
            batch.segment_ids,
            roots=batch.roots,
            video_ids=batch.video_ids,
            frame_indices=batch.frame_indices,
            ocr_texts=batch.ocr_texts,
            embedding_loss_weight=settings.embedding_loss_weight,
            embedding_loss_alpha=settings.embedding_loss_alpha,
            embedding_pair_frame_window=settings.embedding_pair_frame_window,
            embedding_ocr_negative_enabled=settings.embedding_ocr_negative_enabled,
            embedding_ocr_negative_max_similarity=settings.embedding_ocr_negative_max_similarity,
            embedding_positive_consistency_beta=settings.embedding_positive_consistency_beta,
            embedding_positive_consistency_margin=settings.embedding_positive_consistency_margin,
            embedding_temperature=settings.embedding_temperature,
        )
        total_loss += float(loss.total.detach().cpu())
        total_presence_loss += float(loss.presence_loss.detach().cpu())
        total_embedding_loss += float(loss.embedding_loss.detach().cpu())
        batches += 1
        logits_all.append(presence_logit.detach().cpu())
        presence_all.append(presence.detach().cpu())
        embedding_all.append(embedding.detach().cpu())
        segment_ids.extend(batch.segment_ids)
        roots.extend(batch.roots)
        video_ids.extend(batch.video_ids)
        frame_indices.extend(batch.frame_indices)
        ocr_texts.extend(batch.ocr_texts)
    logits = torch.cat(logits_all)
    presence = torch.cat(presence_all)
    embedding = torch.cat(embedding_all)
    metrics = {
        "val_loss": total_loss / max(1, batches),
        "val_presence_loss": total_presence_loss / max(1, batches),
        "val_embedding_loss": total_embedding_loss / max(1, batches),
    }
    metrics.update(presence_metrics(logits, presence))
    metrics.update(
        embedding_metrics(
            embedding,
            presence,
            segment_ids,
            roots=roots,
            video_ids=video_ids,
            frame_indices=frame_indices,
            ocr_texts=ocr_texts,
            frame_window=settings.embedding_pair_frame_window,
            ocr_negative_enabled=settings.embedding_ocr_negative_enabled,
            ocr_negative_max_similarity=settings.embedding_ocr_negative_max_similarity,
            threshold=settings.embedding_similarity_threshold,
        )
    )
    return metrics


def run_training(settings: RoiTrainSettings) -> dict[str, float]:
    seed_everything(settings.seed)
    device = choose_device(settings.device)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = make_dataset(settings, train=True)
    val_dataset = make_dataset(settings, train=False)
    if train_dataset.summary.roi_size != val_dataset.summary.roi_size:
        raise ValueError(
            f"train/val ROI size mismatch: train={train_dataset.summary.roi_size} val={val_dataset.summary.roi_size}; "
            "pass --resize-roi WIDTHxHEIGHT for explicit resize"
        )
    if len(train_dataset) == 0:
        raise RuntimeError("no ROI training samples found")
    if len(val_dataset) == 0:
        raise RuntimeError("no ROI validation samples found")
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        collate_fn=collate_roi_batch,
    )
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers, collate_fn=collate_roi_batch)
    model = make_model(settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    best_presence_f1 = -1.0
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
            best_presence_f1,
            best_epoch,
            last_metrics,
        ) = load_resume_checkpoint(
            settings.resume,
            model,
            optimizer,
            device,
            current_embedding_head_type=settings.embedding_head_type,
        )
    metrics_path = settings.output_dir / "metrics.jsonl"
    print(f"roi_presence_embedding device={device} output_dir={settings.output_dir}", flush=True)
    if resume_checkpoint_path is not None:
        print(
            f"resume={resume_checkpoint_path} start_epoch={start_epoch} step={global_step} "
            f"best_epoch={best_epoch} best_presence_f1={best_presence_f1:.4f}",
            flush=True,
        )
    print(
        f"config: batch_size={settings.batch_size} epochs={settings.epochs} lr={settings.learning_rate:g} "
        f"weight_decay={settings.weight_decay:g} embedding_dim={settings.embedding_dim} "
        f"embedding_head={settings.embedding_head_type} sequence_channels={settings.embedding_sequence_channels} "
        f"embedding_loss_weight={settings.embedding_loss_weight:g} resize_roi={settings.resize_roi} "
        f"embedding_loss_alpha={settings.embedding_loss_alpha:g} "
        f"embedding_pair_frame_window={settings.embedding_pair_frame_window} "
        f"embedding_ocr_negative_enabled={settings.embedding_ocr_negative_enabled} "
        f"embedding_ocr_negative_max_similarity={settings.embedding_ocr_negative_max_similarity:g} "
        f"embedding_positive_consistency_beta={settings.embedding_positive_consistency_beta:g} "
        f"embedding_positive_consistency_margin={settings.embedding_positive_consistency_margin:g} "
        f"max_train_samples={settings.max_train_samples} max_val_samples={settings.max_val_samples} "
        f"negative_ratio={settings.negative_ratio} val_negative_ratio={settings.val_negative_ratio}",
        flush=True,
    )
    print(format_dataset_summary("train", train_dataset), flush=True)
    print(format_dataset_summary("val", val_dataset), flush=True)
    metrics_mode = "a" if settings.resume is not None else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, settings.epochs + 1):
            model.train()
            train_loss = 0.0
            train_presence_loss = 0.0
            train_embedding_loss = 0.0
            train_embedding_pairs = 0
            train_embedding_local_positive_pairs = 0
            train_embedding_local_negative_pairs = 0
            train_embedding_ocr_negative_pairs = 0
            train_embedding_skipped_pairs = 0
            batches = 0
            epoch_start = time.perf_counter()
            progress = tqdm(train_loader, desc=f"roi epoch {epoch}/{settings.epochs}", leave=False)
            total_batches = len(train_loader)
            for batch_index, batch in enumerate(progress, start=1):
                batch_start = time.perf_counter()
                images = batch.images.to(device)
                presence = batch.presence.to(device)
                optimizer.zero_grad(set_to_none=True)
                presence_logit, embedding = model(images)
                loss = roi_presence_embedding_loss(
                    presence_logit,
                    embedding,
                    presence,
                    batch.segment_ids,
                    roots=batch.roots,
                    video_ids=batch.video_ids,
                    frame_indices=batch.frame_indices,
                    ocr_texts=batch.ocr_texts,
                    embedding_loss_weight=settings.embedding_loss_weight,
                    embedding_loss_alpha=settings.embedding_loss_alpha,
                    embedding_pair_frame_window=settings.embedding_pair_frame_window,
                    embedding_ocr_negative_enabled=settings.embedding_ocr_negative_enabled,
                    embedding_ocr_negative_max_similarity=settings.embedding_ocr_negative_max_similarity,
                    embedding_positive_consistency_beta=settings.embedding_positive_consistency_beta,
                    embedding_positive_consistency_margin=settings.embedding_positive_consistency_margin,
                    embedding_temperature=settings.embedding_temperature,
                )
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += float(loss.total.detach().cpu())
                train_presence_loss += float(loss.presence_loss.detach().cpu())
                train_embedding_loss += float(loss.embedding_loss.detach().cpu())
                train_embedding_pairs += loss.embedding_pairs
                train_embedding_local_positive_pairs += loss.embedding_local_positive_pairs
                train_embedding_local_negative_pairs += loss.embedding_local_negative_pairs
                train_embedding_ocr_negative_pairs += loss.embedding_ocr_negative_pairs
                train_embedding_skipped_pairs += loss.embedding_skipped_pairs
                batches += 1
                global_step += 1
                batch_time = max(time.perf_counter() - batch_start, 1e-9)
                step_metrics = {
                    "record_type": "roi_train_step",
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "epoch_batch": float(batch_index),
                    "epoch_batches": float(total_batches),
                    "total_loss": float(loss.total.detach().cpu()),
                    "presence_loss": float(loss.presence_loss.detach().cpu()),
                    "embedding_loss": float(loss.embedding_loss.detach().cpu()),
                    "embedding_bce_loss": float(loss.embedding_bce_loss.detach().cpu()),
                    "positive_consistency_loss": float(loss.positive_consistency_loss.detach().cpu()),
                    "embedding_pairs": float(loss.embedding_pairs),
                    "embedding_local_positive_pairs": float(loss.embedding_local_positive_pairs),
                    "embedding_local_negative_pairs": float(loss.embedding_local_negative_pairs),
                    "embedding_ocr_negative_pairs": float(loss.embedding_ocr_negative_pairs),
                    "embedding_skipped_pairs": float(loss.embedding_skipped_pairs),
                    "positive_samples": float((presence > 0.5).sum().detach().cpu()),
                    "samples_per_second": float(len(batch.sample_ids)) / batch_time,
                    "batch_time": batch_time,
                }
                progress.set_postfix(loss=f"{step_metrics['total_loss']:.4f}")
                should_log = global_step == 1 or batch_index == total_batches or global_step % max(1, settings.log_interval) == 0
                if should_log:
                    metrics_file.write(json.dumps(step_metrics, sort_keys=True) + "\n")
                    metrics_file.flush()
            last_metrics = validate(model, val_loader, device, settings)
            last_metrics.update(
                {
                    "record_type": "roi_validation",
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "train_loss": train_loss / max(1, batches),
                    "train_presence_loss": train_presence_loss / max(1, batches),
                    "train_embedding_loss": train_embedding_loss / max(1, batches),
                    "train_embedding_pairs": float(train_embedding_pairs),
                    "train_embedding_local_positive_pairs": float(train_embedding_local_positive_pairs),
                    "train_embedding_local_negative_pairs": float(train_embedding_local_negative_pairs),
                    "train_embedding_ocr_negative_pairs": float(train_embedding_ocr_negative_pairs),
                    "train_embedding_skipped_pairs": float(train_embedding_skipped_pairs),
                    "train_samples": float(len(train_dataset)),
                    "val_samples": float(len(val_dataset)),
                    "val_positive_segments": float(val_dataset.summary.positive_segments),
                    "val_repeated_positive_segments": float(val_dataset.summary.repeated_positive_segments),
                    "val_same_segment_pairs": float(val_dataset.summary.same_segment_pairs),
                    "epoch_seconds": time.perf_counter() - epoch_start,
                }
            )
            checkpoint_saved = last_metrics["presence_f1"] >= best_presence_f1
            if checkpoint_saved:
                best_presence_f1 = last_metrics["presence_f1"]
                best_epoch = epoch
            last_metrics["best_epoch"] = float(best_epoch)
            epoch_checkpoint_path = save_epoch_checkpoint(
                settings,
                epoch,
                global_step,
                best_presence_f1,
                best_epoch,
                model,
                optimizer,
                last_metrics,
            )
            metrics_file.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            metrics_file.flush()
            if checkpoint_saved:
                torch.save(
                    checkpoint_payload(settings, epoch, global_step, best_presence_f1, best_epoch, model, optimizer, last_metrics),
                    settings.output_dir / "best.pt",
                )
            epoch_message = format_epoch_summary(epoch, settings.epochs, last_metrics)
            epoch_message += f" epoch_checkpoint={epoch_checkpoint_path}"
            if checkpoint_saved:
                epoch_message += f" checkpoint=best step={global_step}"
            print(epoch_message, flush=True)
    (settings.output_dir / "summary.json").write_text(json.dumps(last_metrics, indent=2, sort_keys=True), encoding="utf-8")
    return last_metrics


def parse_args(argv: list[str] | None = None) -> RoiTrainSettings:
    parser = argparse.ArgumentParser(description="Train ROI Presence + Embedding subtitle model.")
    parser.add_argument("--train-root", type=Path, action="append", dest="train_roots")
    parser.add_argument("--val-root", type=Path, default=RoiTrainSettings().val_root)
    parser.add_argument("--output-dir", type=Path, default=RoiTrainSettings().output_dir)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resize-roi", type=parse_roi_size)
    parser.add_argument("--batch-size", type=int, default=RoiTrainSettings().batch_size)
    parser.add_argument("--epochs", type=int, default=RoiTrainSettings().epochs)
    parser.add_argument("--lr", type=float, default=RoiTrainSettings().learning_rate)
    parser.add_argument("--max-samples", type=int, help="Maximum ROI training sample count.")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--positive-ratio", type=float, help="Target subtitle-present ROI training sample ratio in [0, 1].")
    parser.add_argument("--negative-ratio", type=float, help="Target no-subtitle ROI training sample ratio in [0, 1].")
    parser.add_argument("--val-positive-ratio", type=float, help="Target subtitle-present ROI validation sample ratio in [0, 1].")
    parser.add_argument("--val-negative-ratio", type=float, help="Target no-subtitle ROI validation sample ratio in [0, 1].")
    parser.add_argument("--embedding-loss-weight", type=float, default=RoiTrainSettings().embedding_loss_weight)
    parser.add_argument("--embedding-loss-alpha", type=float, default=RoiTrainSettings().embedding_loss_alpha)
    parser.add_argument("--embedding-pair-frame-window", type=int, default=RoiTrainSettings().embedding_pair_frame_window)
    parser.add_argument(
        "--embedding-ocr-negative-enabled",
        action=argparse.BooleanOptionalAction,
        default=RoiTrainSettings().embedding_ocr_negative_enabled,
    )
    parser.add_argument("--embedding-ocr-negative-max-similarity", type=float, default=RoiTrainSettings().embedding_ocr_negative_max_similarity)
    parser.add_argument("--embedding-positive-consistency-beta", type=float, default=RoiTrainSettings().embedding_positive_consistency_beta)
    parser.add_argument("--embedding-positive-consistency-margin", type=float, default=RoiTrainSettings().embedding_positive_consistency_margin)
    parser.add_argument("--embedding-temperature", type=float, default=RoiTrainSettings().embedding_temperature)
    parser.add_argument("--embedding-similarity-threshold", type=float, default=RoiTrainSettings().embedding_similarity_threshold)
    parser.add_argument("--embedding-head", choices=("gap", "hybrid_lite"), default=RoiTrainSettings().embedding_head_type)
    parser.add_argument("--embedding-sequence-channels", type=int, default=RoiTrainSettings().embedding_sequence_channels)
    parser.add_argument("--width", type=int, default=RoiTrainSettings().width)
    parser.add_argument("--embedding-dim", type=int, default=RoiTrainSettings().embedding_dim)
    parser.add_argument("--log-interval", type=int, default=RoiTrainSettings().log_interval)
    parser.add_argument("--device", default=RoiTrainSettings().device)
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

    empty_ratio = resolve_empty_ratio(
        positive_ratio=args.positive_ratio,
        negative_ratio=args.negative_ratio,
        default=RoiTrainSettings().negative_ratio,
        positive_name="--positive-ratio",
        negative_name="--negative-ratio",
    )
    val_empty_ratio = resolve_empty_ratio(
        positive_ratio=args.val_positive_ratio,
        negative_ratio=args.val_negative_ratio,
        default=RoiTrainSettings().val_negative_ratio,
        positive_name="--val-positive-ratio",
        negative_name="--val-negative-ratio",
    )
    if args.embedding_pair_frame_window < 0:
        parser.error("--embedding-pair-frame-window must be non-negative")
    if not 0.0 <= args.embedding_ocr_negative_max_similarity <= 1.0:
        parser.error("--embedding-ocr-negative-max-similarity must be in [0, 1]")
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else args.max_samples
    return RoiTrainSettings(
        train_roots=args.train_roots if args.train_roots is not None else RoiTrainSettings().train_roots,
        val_root=args.val_root,
        output_dir=args.output_dir,
        resume=args.resume,
        resize_roi=args.resize_roi,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_train_samples=max_train_samples,
        max_val_samples=args.max_val_samples,
        negative_ratio=empty_ratio,
        val_negative_ratio=val_empty_ratio,
        embedding_loss_weight=args.embedding_loss_weight,
        embedding_loss_alpha=args.embedding_loss_alpha,
        embedding_pair_frame_window=args.embedding_pair_frame_window,
        embedding_ocr_negative_enabled=args.embedding_ocr_negative_enabled,
        embedding_ocr_negative_max_similarity=args.embedding_ocr_negative_max_similarity,
        embedding_positive_consistency_beta=args.embedding_positive_consistency_beta,
        embedding_positive_consistency_margin=args.embedding_positive_consistency_margin,
        embedding_temperature=args.embedding_temperature,
        embedding_similarity_threshold=args.embedding_similarity_threshold,
        embedding_head_type=args.embedding_head,
        embedding_sequence_channels=args.embedding_sequence_channels,
        width=args.width,
        embedding_dim=args.embedding_dim,
        log_interval=args.log_interval,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    metrics = run_training(parse_args(argv))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def parse_validate_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a ROI Presence + Embedding checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--root", type=Path, required=True, help="ROI dataset root to validate.")
    parser.add_argument("--resize-roi", type=parse_roi_size)
    parser.add_argument("--batch-size", type=int, default=RoiTrainSettings().batch_size)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--embedding-loss-weight", type=float, default=RoiTrainSettings().embedding_loss_weight)
    parser.add_argument("--embedding-loss-alpha", type=float)
    parser.add_argument("--embedding-pair-frame-window", type=int)
    parser.add_argument("--embedding-ocr-negative-enabled", action=argparse.BooleanOptionalAction)
    parser.add_argument("--embedding-ocr-negative-max-similarity", type=float)
    parser.add_argument("--embedding-positive-consistency-beta", type=float)
    parser.add_argument("--embedding-positive-consistency-margin", type=float)
    parser.add_argument("--embedding-temperature", type=float, default=RoiTrainSettings().embedding_temperature)
    parser.add_argument("--embedding-similarity-threshold", type=float, default=RoiTrainSettings().embedding_similarity_threshold)
    parser.add_argument("--device", default=RoiTrainSettings().device)
    args = parser.parse_args(argv)
    if args.embedding_pair_frame_window is not None and args.embedding_pair_frame_window < 0:
        parser.error("--embedding-pair-frame-window must be non-negative")
    if args.embedding_ocr_negative_max_similarity is not None and not 0.0 <= args.embedding_ocr_negative_max_similarity <= 1.0:
        parser.error("--embedding-ocr-negative-max-similarity must be in [0, 1]")
    return args


def run_validation(args: argparse.Namespace) -> dict[str, float]:
    device = choose_device(args.device)
    model, checkpoint = load_model_checkpoint(args.checkpoint, device)
    raw_settings = dict(checkpoint.get("settings") or {})
    checkpoint_resize = tuple(raw_settings["resize_roi"]) if raw_settings.get("resize_roi") is not None else None
    settings = RoiTrainSettings(
        train_roots=[args.root],
        val_root=args.root,
        resize_roi=args.resize_roi or checkpoint_resize,
        batch_size=args.batch_size,
        max_val_samples=args.max_samples,
        embedding_loss_weight=args.embedding_loss_weight,
        embedding_loss_alpha=float(raw_settings.get("embedding_loss_alpha", RoiTrainSettings().embedding_loss_alpha) if args.embedding_loss_alpha is None else args.embedding_loss_alpha),
        embedding_pair_frame_window=int(
            raw_settings.get("embedding_pair_frame_window", RoiTrainSettings().embedding_pair_frame_window)
            if args.embedding_pair_frame_window is None
            else args.embedding_pair_frame_window
        ),
        embedding_ocr_negative_enabled=bool(
            raw_settings.get("embedding_ocr_negative_enabled", RoiTrainSettings().embedding_ocr_negative_enabled)
            if args.embedding_ocr_negative_enabled is None
            else args.embedding_ocr_negative_enabled
        ),
        embedding_ocr_negative_max_similarity=float(
            raw_settings.get("embedding_ocr_negative_max_similarity", RoiTrainSettings().embedding_ocr_negative_max_similarity)
            if args.embedding_ocr_negative_max_similarity is None
            else args.embedding_ocr_negative_max_similarity
        ),
        embedding_positive_consistency_beta=float(
            raw_settings.get("embedding_positive_consistency_beta", RoiTrainSettings().embedding_positive_consistency_beta)
            if args.embedding_positive_consistency_beta is None
            else args.embedding_positive_consistency_beta
        ),
        embedding_positive_consistency_margin=float(
            raw_settings.get("embedding_positive_consistency_margin", RoiTrainSettings().embedding_positive_consistency_margin)
            if args.embedding_positive_consistency_margin is None
            else args.embedding_positive_consistency_margin
        ),
        embedding_temperature=args.embedding_temperature,
        embedding_similarity_threshold=args.embedding_similarity_threshold,
        embedding_head_type=str(raw_settings.get("embedding_head_type", RoiTrainSettings().embedding_head_type)),
        embedding_sequence_channels=int(raw_settings.get("embedding_sequence_channels", RoiTrainSettings().embedding_sequence_channels)),
        width=int(raw_settings.get("width", RoiTrainSettings().width)),
        embedding_dim=int(raw_settings.get("embedding_dim", RoiTrainSettings().embedding_dim)),
        device=args.device,
    )
    dataset = RoiPresenceEmbeddingDataset(
        [args.root],
        resize_roi=settings.resize_roi,
        max_samples=args.max_samples,
        empty_ratio=None,
        segment_aware_limit=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_roi_batch)
    metrics = validate(model, loader, device, settings)
    forward_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_roi_batch)
    metrics.update(
        {
            "record_type": "roi_checkpoint_validation",
            "checkpoint_epoch": float(checkpoint.get("epoch", 0)),
            "model_parameters": float(model_parameter_count(model)),
            "roi_forward_ms_per_img": measure_roi_forward_time(model, forward_loader, device),
            "samples": float(len(dataset)),
            "positive_samples": float(dataset.summary.positive),
            "empty_samples": float(dataset.summary.empty),
            "positive_segments": float(dataset.summary.positive_segments),
            "repeated_positive_segments": float(dataset.summary.repeated_positive_segments),
            "same_segment_pairs": float(dataset.summary.same_segment_pairs),
        }
    )
    print(format_dataset_summary("validate", dataset), flush=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main_validate(argv: list[str] | None = None) -> None:
    run_validation(parse_validate_args(argv))
