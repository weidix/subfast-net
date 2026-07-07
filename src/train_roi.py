from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
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

from .roi_config import RoiTrainSettings
from .roi_dataset import RoiPresenceEmbeddingDataset, collate_roi_batch
from .roi_loss import EmbeddingPairMemory, roi_presence_embedding_loss
from .roi_metrics import embedding_metrics, presence_metrics, similarity_percentile
from .roi_model import RoiPresenceEmbeddingModel
from .roi_pairs import normalize_ocr_text, select_embedding_pairs
from .roi_sampler import RoiBalancedBatchSampler
from .train import choose_device

_SHORT_SUBTITLE_MAX_CHARS = 2


@dataclass(frozen=True)
class RoiTrainingPhase:
    name: str
    start_epoch: int
    end_epoch: int
    learning_rate: float


def training_phases(settings: RoiTrainSettings) -> list[RoiTrainingPhase]:
    counts = (settings.presence_epochs, settings.embedding_epochs, settings.joint_epochs)
    phases: list[RoiTrainingPhase] = []
    start_epoch = 1
    for name, count, learning_rate in zip(
        ("presence", "embedding", "joint"),
        counts,
        (settings.learning_rate, settings.learning_rate, settings.joint_learning_rate),
        strict=True,
    ):
        if count <= 0:
            continue
        end_epoch = start_epoch + count - 1
        phases.append(RoiTrainingPhase(name, start_epoch, end_epoch, learning_rate))
        start_epoch = end_epoch + 1
    if not phases:
        raise ValueError("at least one ROI training phase must have a positive epoch count")
    return phases


def phase_for_epoch(phases: list[RoiTrainingPhase], epoch: int) -> RoiTrainingPhase:
    for phase in phases:
        if phase.start_epoch <= epoch <= phase.end_epoch:
            return phase
    raise ValueError(f"epoch {epoch} is outside the configured ROI training phases")


def configure_training_phase(model: RoiPresenceEmbeddingModel, phase_name: str) -> None:
    if phase_name not in {"presence", "embedding", "joint"}:
        raise ValueError(f"unsupported ROI training phase: {phase_name}")
    embedding_modules = [model.embedding_head, model.hybrid_embedding_head, model.local_contrast_embedding_head]
    for parameter in model.parameters():
        parameter.requires_grad_(phase_name == "joint")
    if phase_name == "presence":
        for module in (model.backbone, model.presence_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif phase_name == "embedding":
        for module in embedding_modules:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    model.backbone.train(phase_name in {"presence", "joint"})
    model.presence_head.train(phase_name in {"presence", "joint"})
    for module in embedding_modules:
        if module is not None:
            module.train(phase_name in {"embedding", "joint"})


def checkpoint_score(metrics: dict[str, float], phase_name: str = "joint") -> float:
    presence_score = sum(_metric(metrics, name) for name in ("global_presence_f1", "normal_presence_f1", "short_presence_f1")) / 3.0
    embedding_score = sum(
        _metric(metrics, name)
        for name in ("global_embedding_acc", "normal_embedding_acc", "style_hard_negative_embedding_acc")
    ) / 3.0
    if phase_name == "presence":
        return presence_score
    if phase_name == "embedding":
        return embedding_score
    return 0.5 * presence_score + 0.5 * embedding_score


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
        empty_ratio=None if train else settings.val_negative_ratio,
        segment_aware_limit=not train,
        load_subtitle_masks=train and settings.short_positive_mask_loss_weight > 0.0,
    )


def validation_overlaps_training(settings: RoiTrainSettings) -> bool:
    validation_root = settings.val_root.resolve()
    return any(root.resolve() == validation_root for root in settings.train_roots)


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
    phase = str(metrics.get("training_phase", ""))
    title = f"roi epoch {epoch}/{total_epochs}"
    if phase:
        title += f" phase={phase}"
    return "\n".join(
        [
            title,
            (
                f"  loss: train={metrics['train_loss']:.4f} "
                f"presence={metrics['train_presence_loss']:.4f} "
                f"embedding={metrics['train_embedding_loss']:.4f} "
                f"val={metrics['val_loss']:.4f}"
            ),
            (
                f"  presence: f1={metrics['presence_f1']:.4f} "
                f"accuracy={metrics['presence_accuracy']:.4f} "
                f"tp={metrics['presence_tp']:.0f} "
                f"fp={metrics['presence_fp']:.0f} "
                f"fn={metrics['presence_fn']:.0f} "
                f"tn={metrics['presence_tn']:.0f}"
            ),
            (
                f"  embedding: acc={metrics['embedding_pair_accuracy']:.4f} "
                f"fp={metrics['embedding_false_positive_pairs']:.0f} "
                f"fn={metrics['embedding_false_negative_pairs']:.0f} "
                f"same={metrics['embedding_same_similarity']:.4f} "
                f"diff={metrics['embedding_diff_similarity']:.4f} "
                f"hard_margin={metrics['hard_margin']:.4f} "
                f"gap={metrics.get('embedding_gap', 0.0):.4f} "
                f"best_threshold={metrics.get('embedding_best_f1_threshold', 0.0):.4f}"
            ),
            (
                f"  similarity: same_p10={metrics['same_sim_p10']:.4f} "
                f"same_p50={metrics['same_sim_p50']:.4f} "
                f"hard_neg_p50={metrics['hard_negative_sim_p50']:.4f} "
                f"hard_neg_p90={metrics['hard_negative_sim_p90']:.4f} "
                f"hard_neg_p95={metrics['hard_negative_sim_p95']:.4f}"
            ),
            (
                f"  score: current={metrics['checkpoint_score']:.4f} "
                f"best={metrics['best_checkpoint_score']:.4f} "
                f"best_epoch={metrics['best_epoch']:.0f}"
            ),
        ]
    )


def _metric(metrics: dict[str, float], name: str) -> float:
    fallback_names = {
        "global_presence_f1": ("presence_f1",),
        "normal_presence_f1": ("global_presence_f1", "presence_f1"),
        "short_presence_f1": ("global_presence_f1", "presence_f1"),
        "global_embedding_acc": ("embedding_pair_accuracy",),
        "normal_embedding_acc": ("global_embedding_acc", "embedding_pair_accuracy"),
        "style_hard_negative_embedding_acc": ("global_embedding_acc", "embedding_pair_accuracy"),
        "hard_negative_sim": ("embedding_diff_similarity",),
        "hard_negative_sim_p90": ("hard_negative_sim", "embedding_negative_p90", "embedding_diff_similarity"),
        "same_sim_p10": ("embedding_positive_p10", "embedding_same_similarity"),
        "same_sim_p50": ("embedding_positive_p50", "embedding_same_similarity"),
    }
    if name == "hard_margin":
        return _metric(metrics, "same_sim_p10") - _metric(metrics, "hard_negative_sim_p90")
    if name in metrics:
        return float(metrics[name])
    for fallback in fallback_names.get(name, ()):
        if fallback in metrics:
            return float(metrics[fallback])
    return 0.0


def should_save_roi_best_checkpoint(
    current: dict[str, float],
    best: dict[str, float] | None,
    phase_name: str = "joint",
) -> bool:
    if not best:
        return True
    return checkpoint_score(current, phase_name) > checkpoint_score(best, phase_name)


def epoch_output_dir(settings: RoiTrainSettings, epoch: int) -> Path:
    return settings.output_dir / "epoch_outputs" / f"epoch_{epoch:04}"


def phase_best_checkpoint_path(settings: RoiTrainSettings, phase_name: str) -> Path:
    return settings.output_dir / f"best_{phase_name}.pt"


def checkpoint_payload(
    settings: RoiTrainSettings,
    epoch: int,
    step: int,
    best_presence_f1: float,
    best_epoch: int,
    best_metrics: dict[str, float],
    model: RoiPresenceEmbeddingModel,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    training_phase: str,
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
        "best_metrics": best_metrics,
        "training_phase": training_phase,
    }


def make_model(settings: RoiTrainSettings) -> RoiPresenceEmbeddingModel:
    return RoiPresenceEmbeddingModel(
        width=settings.width,
        embedding_dim=settings.embedding_dim,
        embedding_head_type=settings.embedding_head_type,
        embedding_sequence_channels=settings.embedding_sequence_channels,
        presence_topk_ratio=settings.presence_topk_ratio,
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
    device: torch.device,
    *,
    current_embedding_head_type: str,
) -> tuple[Path, int, int, float, int, dict[str, float], dict[str, float], dict[str, Any] | None, str | None]:
    checkpoint_path = resolve_resume_checkpoint(resume)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "roi_presence_embedding":
        raise RuntimeError(f"invalid ROI Presence+Embedding checkpoint: {checkpoint_path}")
    raw_settings = dict(checkpoint.get("settings") or {})
    checkpoint_embedding_head_type = str(raw_settings.get("embedding_head_type", "gap"))
    partial_model_load = checkpoint_embedding_head_type != current_embedding_head_type
    model.load_state_dict(checkpoint["model"], strict=not partial_model_load)
    metrics = dict(checkpoint.get("metrics") or {})
    completed_epoch = int(checkpoint.get("epoch", metrics.get("epoch", 0)))
    step = int(checkpoint.get("step", metrics.get("step", 0)))
    best_presence_f1 = float(checkpoint.get("best_presence_f1", metrics.get("presence_f1", -1.0)))
    best_epoch = int(checkpoint.get("best_epoch", metrics.get("best_epoch", completed_epoch)))
    best_metrics = dict(checkpoint.get("best_metrics") or metrics)
    optimizer_state = checkpoint.get("optimizer") if not partial_model_load else None
    training_phase = checkpoint.get("training_phase")
    return checkpoint_path, completed_epoch + 1, step, best_presence_f1, best_epoch, metrics, best_metrics, optimizer_state, training_phase


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
        presence_topk_ratio=float(raw_settings.get("presence_topk_ratio", RoiTrainSettings().presence_topk_ratio)),
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
    best_metrics: dict[str, float],
    model: RoiPresenceEmbeddingModel,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, float],
    training_phase: str,
) -> Path:
    output_dir = epoch_output_dir(settings, epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "model.pt"
    torch.save(
        checkpoint_payload(settings, epoch, step, best_presence_f1, best_epoch, best_metrics, model, optimizer, metrics, training_phase),
        path,
    )
    return path


def save_epoch_metrics(settings: RoiTrainSettings, epoch: int, metrics: dict[str, float]) -> Path:
    output_dir = epoch_output_dir(settings, epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path


def training_summary(
    settings: RoiTrainSettings,
    *,
    completed_epoch: int,
    total_epochs: int,
    best_metrics: dict[str, float],
) -> dict[str, Any]:
    best_epoch = int(best_metrics.get("epoch", best_metrics.get("best_epoch", completed_epoch)))
    best_phase = str(best_metrics.get("training_phase", phase_for_epoch(training_phases(settings), best_epoch).name))
    return {
        "record_type": "roi_training_summary",
        "completed_epoch": completed_epoch,
        "total_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_step": int(best_metrics.get("step", 0)),
        "best_training_phase": best_phase,
        "best_checkpoint": str(settings.output_dir / "best.pt"),
        "best_phase_checkpoint": str(phase_best_checkpoint_path(settings, best_phase)),
        "best_epoch_checkpoint": str(epoch_output_dir(settings, best_epoch) / "model.pt"),
        "best_epoch_metrics": str(epoch_output_dir(settings, best_epoch) / "metrics.json"),
        "metrics_log": str(settings.output_dir / "metrics.jsonl"),
        "epoch_outputs": str(settings.output_dir / "epoch_outputs"),
        "checkpoint_score": float(best_metrics.get("checkpoint_score", 0.0)),
        "validation": {
            "presence_f1": float(best_metrics.get("presence_f1", 0.0)),
            "presence_accuracy": float(best_metrics.get("presence_accuracy", 0.0)),
            "presence_precision": float(best_metrics.get("presence_precision", 0.0)),
            "presence_recall": float(best_metrics.get("presence_recall", 0.0)),
            "presence_tp": float(best_metrics.get("presence_tp", 0.0)),
            "presence_fp": float(best_metrics.get("presence_fp", 0.0)),
            "presence_fn": float(best_metrics.get("presence_fn", 0.0)),
            "presence_tn": float(best_metrics.get("presence_tn", 0.0)),
            "normal_presence_f1": float(best_metrics.get("normal_presence_f1", 0.0)),
            "short_presence_f1": float(best_metrics.get("short_presence_f1", 0.0)),
            "embedding_pair_accuracy": float(best_metrics.get("embedding_pair_accuracy", 0.0)),
            "embedding_roc_auc": float(best_metrics.get("embedding_roc_auc", 0.0)),
            "embedding_same_similarity": float(best_metrics.get("embedding_same_similarity", 0.0)),
            "embedding_diff_similarity": float(best_metrics.get("embedding_diff_similarity", 0.0)),
            "embedding_min_positive_similarity": float(best_metrics.get("embedding_min_positive_similarity", 0.0)),
            "embedding_max_negative_similarity": float(best_metrics.get("embedding_max_negative_similarity", 0.0)),
            "embedding_gap": float(best_metrics.get("embedding_gap", 0.0)),
            "embedding_best_f1_threshold": float(best_metrics.get("embedding_best_f1_threshold", 0.0)),
            "embedding_best_f1": float(best_metrics.get("embedding_best_f1", 0.0)),
            "embedding_zero_error_threshold_exists": float(best_metrics.get("embedding_zero_error_threshold_exists", 0.0)),
            "hard_margin": float(best_metrics.get("hard_margin", 0.0)),
            "val_loss": float(best_metrics.get("val_loss", 0.0)),
            "val_presence_loss": float(best_metrics.get("val_presence_loss", 0.0)),
            "val_embedding_loss": float(best_metrics.get("val_embedding_loss", 0.0)),
        },
        "data": {
            "train_samples": float(best_metrics.get("train_samples", 0.0)),
            "val_samples": float(best_metrics.get("val_samples", 0.0)),
            "val_positive_segments": float(best_metrics.get("val_positive_segments", 0.0)),
            "val_repeated_positive_segments": float(best_metrics.get("val_repeated_positive_segments", 0.0)),
            "val_same_segment_pairs": float(best_metrics.get("val_same_segment_pairs", 0.0)),
            "validation_overlaps_training": bool(best_metrics.get("validation_overlaps_training", False)),
        },
    }


def _presence_metrics_for_mask(logits: torch.Tensor, presence: torch.Tensor, mask: torch.Tensor, prefix: str) -> dict[str, float]:
    if not bool(mask.any()):
        return {
            f"{prefix}_presence_f1": 0.0,
            f"{prefix}_presence_accuracy": 0.0,
        }
    scoped = presence_metrics(logits[mask], presence[mask])
    return {
        f"{prefix}_presence_f1": scoped["presence_f1"],
        f"{prefix}_presence_accuracy": scoped["presence_accuracy"],
    }


def _pair_accuracy(scores: list[float], targets: list[bool], threshold: float) -> float:
    if not scores:
        return 0.0
    correct = sum(int((score >= threshold) == target) for score, target in zip(scores, targets, strict=True))
    return correct / len(scores)


def scoped_validation_metrics(
    logits: torch.Tensor,
    presence: torch.Tensor,
    embedding: torch.Tensor,
    segment_ids: list[str],
    *,
    roots: list[str],
    video_ids: list[str | None],
    frame_indices: list[int | None],
    ocr_texts: list[str],
    frame_window: int,
    ocr_negative_enabled: bool,
    ocr_negative_max_similarity: float,
    threshold: float,
) -> dict[str, float]:
    positive_mask = presence > 0.5
    short_positive = torch.tensor(
        [
            bool(is_positive) and 0 < len(normalize_ocr_text(text)) <= _SHORT_SUBTITLE_MAX_CHARS
            for is_positive, text in zip(positive_mask.tolist(), ocr_texts, strict=True)
        ],
        dtype=torch.bool,
    )
    normal_positive = positive_mask & ~short_positive
    negative_mask = ~positive_mask
    metrics = {
        "global_presence_f1": _metric(presence_metrics(logits, presence), "presence_f1"),
    }
    metrics.update(_presence_metrics_for_mask(logits, presence, negative_mask | normal_positive, "normal"))
    metrics.update(_presence_metrics_for_mask(logits, presence, negative_mask | short_positive, "short"))

    selection = select_embedding_pairs(
        presence=presence,
        segment_ids=segment_ids,
        roots=roots,
        video_ids=video_ids,
        frame_indices=frame_indices,
        ocr_texts=ocr_texts,
        frame_window=frame_window,
        ocr_negative_enabled=ocr_negative_enabled,
        ocr_negative_max_similarity=ocr_negative_max_similarity,
    )
    normal_scores: list[float] = []
    normal_targets: list[bool] = []
    same_scores: list[float] = []
    hard_negative_scores: list[float] = []
    hard_negative_targets: list[bool] = []
    for pair in selection.pairs:
        score = float((embedding[pair.i] * embedding[pair.j]).sum().detach().cpu())
        if pair.same:
            same_scores.append(score)
        if not bool(short_positive[pair.i]) and not bool(short_positive[pair.j]):
            normal_scores.append(score)
            normal_targets.append(pair.same)
        if pair.source == "local" and not pair.same:
            hard_negative_scores.append(score)
            hard_negative_targets.append(False)
    hard_negative_sim_p90 = similarity_percentile(hard_negative_scores, 0.90)
    same_sim_p10 = similarity_percentile(same_scores, 0.10)

    metrics.update(
        {
            "normal_embedding_acc": _pair_accuracy(normal_scores, normal_targets, threshold),
            "style_hard_negative_embedding_acc": _pair_accuracy(hard_negative_scores, hard_negative_targets, threshold),
            "hard_negative_sim": sum(hard_negative_scores) / len(hard_negative_scores) if hard_negative_scores else 0.0,
            "hard_negative_sim_p50": similarity_percentile(hard_negative_scores, 0.50),
            "hard_negative_sim_p90": hard_negative_sim_p90,
            "hard_negative_sim_p95": similarity_percentile(hard_negative_scores, 0.95),
            "same_sim_p50": similarity_percentile(same_scores, 0.50),
            "same_sim_p10": same_sim_p10,
            "hard_margin": same_sim_p10 - hard_negative_sim_p90 if same_scores and hard_negative_scores else 0.0,
            "style_hard_negative_pairs": float(len(hard_negative_scores)),
        }
    )
    return metrics


def embedding_error_pair_records(
    embedding: torch.Tensor,
    presence: torch.Tensor,
    segment_ids: list[str],
    *,
    sample_ids: list[str],
    image_paths: list[str],
    roots: list[str],
    video_ids: list[str | None],
    frame_indices: list[int | None],
    ocr_texts: list[str],
    frame_window: int,
    ocr_negative_enabled: bool,
    ocr_negative_max_similarity: float,
    threshold: float,
) -> list[dict[str, Any]]:
    selection = select_embedding_pairs(
        presence=presence,
        segment_ids=segment_ids,
        roots=roots,
        video_ids=video_ids,
        frame_indices=frame_indices,
        ocr_texts=ocr_texts,
        frame_window=frame_window,
        ocr_negative_enabled=ocr_negative_enabled,
        ocr_negative_max_similarity=ocr_negative_max_similarity,
    )
    records: list[dict[str, Any]] = []
    for pair in selection.pairs:
        score = float((embedding[pair.i] * embedding[pair.j]).sum().detach().cpu())
        prediction = score >= threshold
        if prediction == pair.same:
            continue
        pair_type = "fn" if pair.same else "fp"
        records.append(
            {
                "pair_type": pair_type,
                "score": score,
                "threshold": threshold,
                "severity": (threshold - score) if pair.same else (score - threshold),
                "source": pair.source,
                "root": roots[pair.i],
                "video_a": video_ids[pair.i],
                "video_b": video_ids[pair.j],
                "frame_a": frame_indices[pair.i],
                "frame_b": frame_indices[pair.j],
                "sample_a": sample_ids[pair.i],
                "sample_b": sample_ids[pair.j],
                "segment_a": segment_ids[pair.i],
                "segment_b": segment_ids[pair.j],
                "ocr_a": ocr_texts[pair.i],
                "ocr_b": ocr_texts[pair.j],
                "image_path_a": image_paths[pair.i],
                "image_path_b": image_paths[pair.j],
            }
        )
    records.sort(key=lambda record: (str(record["pair_type"]), -float(record["severity"])))
    return records


def write_embedding_error_pair_audit(records: list[dict[str, Any]], path: Path, *, html_limit: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    rows = []
    for record in records[:html_limit]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['pair_type']))}</td>"
            f"<td>{float(record['score']):.4f}</td>"
            f"<td>{html.escape(str(record['segment_a']))}<br>{html.escape(str(record['segment_b']))}</td>"
            f"<td>{html.escape(str(record['sample_a']))}<br>{html.escape(str(record['sample_b']))}</td>"
            f"<td>{html.escape(str(record['ocr_a']))}<br>{html.escape(str(record['ocr_b']))}</td>"
            f"<td><img src=\"{html.escape(str(record['image_path_a']))}\"></td>"
            f"<td><img src=\"{html.escape(str(record['image_path_b']))}\"></td>"
            "</tr>"
        )
    html_path = path.with_suffix(".html")
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><meta charset=\"utf-8\">",
                "<style>body{font-family:sans-serif}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px;vertical-align:top}img{max-width:360px;max-height:120px}</style>",
                "<table>",
                "<tr><th>type</th><th>score</th><th>segments</th><th>samples</th><th>ocr</th><th>image A</th><th>image B</th></tr>",
                *rows,
                "</table>",
            ]
        ),
        encoding="utf-8",
    )


@torch.no_grad()
def validate(
    model: RoiPresenceEmbeddingModel,
    loader: DataLoader,
    device: torch.device,
    settings: RoiTrainSettings,
    error_pair_audit_path: Path | None = None,
) -> dict[str, float]:
    model.eval()
    batch_losses: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    presence_all: list[torch.Tensor] = []
    embedding_all: list[torch.Tensor] = []
    segment_ids: list[str] = []
    sample_ids: list[str] = []
    image_paths: list[str] = []
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
            embedding_negative_ratio=settings.embedding_negative_ratio,
            embedding_supcon_weight=settings.embedding_supcon_weight,
            embedding_tail_gamma_positive=settings.embedding_tail_gamma_positive,
            embedding_tail_gamma_negative=settings.embedding_tail_gamma_negative,
            embedding_tail_hard_negative_weight=settings.embedding_tail_hard_negative_weight,
        )
        batch_losses.append(torch.stack((loss.total, loss.presence_loss, loss.embedding_loss)))
        logits_all.append(presence_logit)
        presence_all.append(presence)
        embedding_all.append(embedding)
        segment_ids.extend(batch.segment_ids)
        sample_ids.extend(batch.sample_ids)
        image_paths.extend(batch.image_paths)
        roots.extend(batch.roots)
        video_ids.extend(batch.video_ids)
        frame_indices.extend(batch.frame_indices)
        ocr_texts.extend(batch.ocr_texts)
    logits = torch.cat(logits_all).cpu()
    presence = torch.cat(presence_all).cpu()
    embedding = torch.cat(embedding_all).cpu()
    batch_loss_values = torch.stack(batch_losses).cpu().tolist()
    batches = len(batch_loss_values)
    metrics = {
        "val_loss": sum(values[0] for values in batch_loss_values) / max(1, batches),
        "val_presence_loss": sum(values[1] for values in batch_loss_values) / max(1, batches),
        "val_embedding_loss": sum(values[2] for values in batch_loss_values) / max(1, batches),
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
    metrics["global_presence_f1"] = metrics["presence_f1"]
    metrics["global_embedding_acc"] = metrics["embedding_pair_accuracy"]
    metrics.update(
        scoped_validation_metrics(
            logits,
            presence,
            embedding,
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
    if error_pair_audit_path is not None:
        write_embedding_error_pair_audit(
            embedding_error_pair_records(
                embedding,
                presence,
                segment_ids,
                sample_ids=sample_ids,
                image_paths=image_paths,
                roots=roots,
                video_ids=video_ids,
                frame_indices=frame_indices,
                ocr_texts=ocr_texts,
                frame_window=settings.embedding_pair_frame_window,
                ocr_negative_enabled=settings.embedding_ocr_negative_enabled,
                ocr_negative_max_similarity=settings.embedding_ocr_negative_max_similarity,
                threshold=settings.embedding_similarity_threshold,
            ),
            error_pair_audit_path,
        )
    return metrics


def short_positive_presence_weights(presence: torch.Tensor, ocr_texts: list[str], weight: float) -> torch.Tensor | None:
    if weight == 1.0:
        return None
    weights = torch.ones_like(presence)
    short_positive = [
        bool(is_positive) and 0 < len(normalize_ocr_text(text)) <= _SHORT_SUBTITLE_MAX_CHARS
        for is_positive, text in zip((presence > 0.5).detach().cpu().tolist(), ocr_texts, strict=True)
    ]
    if any(short_positive):
        weights[torch.tensor(short_positive, dtype=torch.bool, device=presence.device)] = weight
    return weights


def short_positive_mask_loss(
    textness_map: torch.Tensor,
    subtitle_masks: torch.Tensor,
    presence: torch.Tensor,
    ocr_texts: list[str],
    weight: float,
) -> torch.Tensor:
    if weight <= 0.0:
        return textness_map.sum() * 0.0
    short_positive = torch.tensor(
        [
            bool(is_positive) and 0 < len(normalize_ocr_text(text)) <= _SHORT_SUBTITLE_MAX_CHARS
            for is_positive, text in zip((presence > 0.5).detach().cpu().tolist(), ocr_texts, strict=True)
        ],
        dtype=torch.bool,
        device=presence.device,
    )
    if not bool(short_positive.any()):
        return textness_map.sum() * 0.0
    target = F.interpolate(
        subtitle_masks.to(device=textness_map.device, dtype=textness_map.dtype),
        size=textness_map.shape[-2:],
        mode="area",
    ).clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(textness_map[short_positive], target[short_positive]) * weight


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
    train_batch_sampler: RoiBalancedBatchSampler | None = None
    if settings.negative_ratio is not None:
        train_batch_sampler = RoiBalancedBatchSampler(
            train_dataset.samples,
            batch_size=settings.batch_size,
            negative_ratio=settings.negative_ratio,
            frame_window=settings.embedding_pair_frame_window,
            embedding_samples_per_segment=settings.embedding_samples_per_segment,
            seed=settings.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=settings.num_workers,
            collate_fn=collate_roi_batch,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=settings.num_workers,
            collate_fn=collate_roi_batch,
        )
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=settings.num_workers, collate_fn=collate_roi_batch)
    model = make_model(settings).to(device)
    phases = training_phases(settings)
    total_epochs = phases[-1].end_epoch
    optimizer: torch.optim.Optimizer | None = None
    active_phase: RoiTrainingPhase | None = None
    best_presence_f1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    last_metrics: dict[str, float] = {}
    global_step = 0
    start_epoch = 1
    resume_checkpoint_path: Path | None = None
    resume_optimizer_state: dict[str, Any] | None = None
    resume_training_phase: str | None = None
    if settings.resume is not None:
        (
            resume_checkpoint_path,
            start_epoch,
            global_step,
            best_presence_f1,
            best_epoch,
            last_metrics,
            best_metrics,
            resume_optimizer_state,
            resume_training_phase,
        ) = load_resume_checkpoint(
            settings.resume,
            model,
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
        f"config: batch_size={settings.batch_size} epochs={total_epochs} lr={settings.learning_rate:g} "
        f"joint_lr={settings.joint_learning_rate:g} "
        f"phase_epochs={','.join(f'{phase.name}:{phase.end_epoch - phase.start_epoch + 1}' for phase in phases)} "
        f"weight_decay={settings.weight_decay:g} embedding_dim={settings.embedding_dim} "
        f"embedding_head={settings.embedding_head_type} sequence_channels={settings.embedding_sequence_channels} "
        f"presence_topk_ratio={settings.presence_topk_ratio:g} "
        f"short_positive_loss_weight={settings.short_positive_loss_weight:g} "
        f"short_positive_mask_loss_weight={settings.short_positive_mask_loss_weight:g} "
        f"embedding_loss_weight={settings.embedding_loss_weight:g} resize_roi={settings.resize_roi} "
        f"embedding_loss_alpha={settings.embedding_loss_alpha:g} "
        f"embedding_pair_frame_window={settings.embedding_pair_frame_window} "
        f"embedding_ocr_negative_enabled={settings.embedding_ocr_negative_enabled} "
        f"embedding_ocr_negative_max_similarity={settings.embedding_ocr_negative_max_similarity:g} "
        f"embedding_positive_consistency_beta={settings.embedding_positive_consistency_beta:g} "
        f"embedding_positive_consistency_margin={settings.embedding_positive_consistency_margin:g} "
        f"embedding_samples_per_segment={settings.embedding_samples_per_segment} "
        f"embedding_supcon_weight={settings.embedding_supcon_weight:g} "
        f"embedding_tail_gamma_positive={settings.embedding_tail_gamma_positive:g} "
        f"embedding_tail_gamma_negative={settings.embedding_tail_gamma_negative:g} "
        f"embedding_tail_hard_negative_weight={settings.embedding_tail_hard_negative_weight:g} "
        f"max_train_samples={settings.max_train_samples} max_val_samples={settings.max_val_samples} "
        f"presence_negative_ratio={settings.negative_ratio} "
        f"embedding_negative_ratio={settings.embedding_negative_ratio} "
        f"val_negative_ratio={settings.val_negative_ratio}",
        flush=True,
    )
    has_validation_overlap = validation_overlaps_training(settings)
    if has_validation_overlap:
        print("warning: validation root overlaps a training root; this run is not held-out quality evidence", flush=True)
    print(format_dataset_summary("train", train_dataset), flush=True)
    print(format_dataset_summary("val", val_dataset), flush=True)
    if settings.negative_ratio is not None and (train_dataset.summary.positive == 0 or train_dataset.summary.empty == 0):
        realized_ratio = 1.0 if train_dataset.summary.positive == 0 else 0.0
        if settings.negative_ratio != realized_ratio:
            print(
                f"warning: presence_negative_ratio={settings.negative_ratio} cannot be achieved because "
                f"the training dataset contains only one presence class; realized_ratio={realized_ratio}",
                flush=True,
            )
    metrics_mode = "a" if settings.resume is not None else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch, total_epochs + 1):
            phase = phase_for_epoch(phases, epoch)
            if active_phase is None or active_phase.name != phase.name:
                previous_phase = active_phase
                handoff_phase_name = previous_phase.name if previous_phase is not None else resume_training_phase
                if handoff_phase_name is not None and handoff_phase_name != phase.name:
                    handoff_path = phase_best_checkpoint_path(settings, handoff_phase_name)
                    if handoff_path.is_file():
                        handoff = torch.load(handoff_path, map_location=device)
                        model.load_state_dict(handoff["model"])
                configure_training_phase(model, phase.name)
                trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                optimizer = torch.optim.AdamW(trainable_parameters, lr=phase.learning_rate, weight_decay=settings.weight_decay)
                if active_phase is None and resume_optimizer_state is not None and resume_training_phase == phase.name:
                    try:
                        optimizer.load_state_dict(resume_optimizer_state)
                    except ValueError:
                        print("warning: resume optimizer state does not match the current training phase; using a fresh optimizer", flush=True)
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = phase.learning_rate
                if resume_training_phase != phase.name:
                    best_metrics = None
                    best_epoch = 0
                active_phase = phase
                print(
                    f"training_phase={phase.name} epochs={phase.start_epoch}-{phase.end_epoch} lr={phase.learning_rate:g} "
                    f"trainable_parameters={sum(parameter.numel() for parameter in trainable_parameters)}",
                    flush=True,
                )
            assert optimizer is not None
            if train_batch_sampler is not None:
                train_batch_sampler.set_epoch(epoch)
            embedding_pair_memory = EmbeddingPairMemory(settings.embedding_pair_frame_window)
            model.train()
            configure_training_phase(model, phase.name)
            train_loss = 0.0
            train_presence_loss = 0.0
            train_short_mask_loss = 0.0
            train_embedding_loss = 0.0
            train_embedding_memory_loss = 0.0
            train_embedding_memory_pairs = 0
            train_embedding_pairs = 0
            train_embedding_local_positive_pairs = 0
            train_embedding_local_negative_pairs = 0
            train_embedding_ocr_negative_pairs = 0
            train_embedding_skipped_pairs = 0
            train_presence_positive_samples = 0
            train_presence_negative_samples = 0
            train_embedding_candidate_positive_pairs = 0
            train_embedding_candidate_negative_pairs = 0
            train_embedding_positive_pairs = 0
            train_embedding_negative_pairs = 0
            train_embedding_batches_without_positive_pairs = 0
            batches = 0
            epoch_start = time.perf_counter()
            progress = tqdm(train_loader, desc=f"roi epoch {epoch}/{total_epochs} {phase.name}", leave=False)
            total_batches = len(train_loader)
            for batch_index, batch in enumerate(progress, start=1):
                batch_start = time.perf_counter()
                images = batch.images.to(device)
                presence = batch.presence.to(device)
                optimizer.zero_grad(set_to_none=True)
                if phase.name == "embedding":
                    embedding = model.forward_embedding(images)
                    presence_logit = presence.new_zeros(presence.shape)
                    mask_loss = embedding.sum() * 0.0
                elif settings.short_positive_mask_loss_weight > 0.0:
                    presence_logit, embedding, textness_map = model.forward_with_presence_map(images)
                    if batch.subtitle_masks is None:
                        raise RuntimeError("subtitle masks are required when short positive mask loss is enabled")
                    mask_loss = short_positive_mask_loss(
                        textness_map,
                        batch.subtitle_masks,
                        presence,
                        batch.ocr_texts,
                        settings.short_positive_mask_loss_weight,
                    )
                else:
                    presence_logit, embedding = model(images)
                    mask_loss = presence_logit.sum() * 0.0
                loss = roi_presence_embedding_loss(
                    presence_logit,
                    embedding,
                    presence,
                    batch.segment_ids,
                    presence_loss_weights=(
                        short_positive_presence_weights(
                            presence,
                            batch.ocr_texts,
                            settings.short_positive_loss_weight,
                        )
                        if phase.name != "embedding"
                        else None
                    ),
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
                    embedding_negative_ratio=settings.embedding_negative_ratio,
                    embedding_supcon_weight=settings.embedding_supcon_weight,
                    embedding_tail_gamma_positive=settings.embedding_tail_gamma_positive,
                    embedding_tail_gamma_negative=settings.embedding_tail_gamma_negative,
                    embedding_tail_hard_negative_weight=settings.embedding_tail_hard_negative_weight,
                    presence_loss_enabled=phase.name != "embedding",
                )
                memory_loss, memory_pairs = embedding_pair_memory.loss_and_update(
                    embedding,
                    presence,
                    batch.segment_ids,
                    batch.roots,
                    batch.video_ids,
                    batch.frame_indices,
                    batch.ocr_texts,
                )
                if phase.name == "presence":
                    total_loss = loss.presence_loss + mask_loss
                elif phase.name == "embedding":
                    total_loss = settings.embedding_loss_weight * (loss.embedding_loss + memory_loss)
                else:
                    total_loss = loss.total + mask_loss + settings.embedding_loss_weight * memory_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
                optimizer.step()
                (
                    total_loss_value,
                    presence_loss_value,
                    mask_loss_value,
                    embedding_loss_value,
                    embedding_margin_loss_value,
                    memory_loss_value,
                    positive_consistency_loss_value,
                    supervised_contrastive_loss_value,
                ) = torch.stack(
                    (
                        total_loss,
                        loss.presence_loss,
                        mask_loss,
                        loss.embedding_loss,
                        loss.embedding_margin_loss,
                        memory_loss,
                        loss.positive_consistency_loss,
                        loss.supervised_contrastive_loss,
                    )
                ).detach().cpu().tolist()
                train_loss += total_loss_value
                train_presence_loss += presence_loss_value
                train_short_mask_loss += mask_loss_value
                train_embedding_loss += embedding_loss_value
                train_embedding_memory_loss += memory_loss_value
                train_embedding_memory_pairs += memory_pairs
                train_embedding_pairs += loss.embedding_pairs
                train_embedding_local_positive_pairs += loss.embedding_local_positive_pairs
                train_embedding_local_negative_pairs += loss.embedding_local_negative_pairs
                train_embedding_ocr_negative_pairs += loss.embedding_ocr_negative_pairs
                train_embedding_skipped_pairs += loss.embedding_skipped_pairs
                positive_samples = int((batch.presence > 0.5).sum())
                negative_samples = len(batch.sample_ids) - positive_samples
                train_presence_positive_samples += positive_samples
                train_presence_negative_samples += negative_samples
                train_embedding_candidate_positive_pairs += loss.embedding_candidate_positive_pairs
                train_embedding_candidate_negative_pairs += loss.embedding_candidate_negative_pairs
                train_embedding_positive_pairs += loss.embedding_selected_positive_pairs
                train_embedding_negative_pairs += loss.embedding_selected_negative_pairs
                train_embedding_batches_without_positive_pairs += int(loss.embedding_selected_positive_pairs == 0)
                batches += 1
                global_step += 1
                batch_time = max(time.perf_counter() - batch_start, 1e-9)
                step_metrics = {
                    "record_type": "roi_train_step",
                    "training_phase": phase.name,
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "epoch_batch": float(batch_index),
                    "epoch_batches": float(total_batches),
                    "total_loss": total_loss_value,
                    "presence_loss": presence_loss_value,
                    "short_mask_loss": mask_loss_value,
                    "embedding_loss": embedding_loss_value,
                    "embedding_margin_loss": embedding_margin_loss_value,
                    "embedding_memory_loss": memory_loss_value,
                    "embedding_memory_pairs": float(memory_pairs),
                    "positive_consistency_loss": positive_consistency_loss_value,
                    "supervised_contrastive_loss": supervised_contrastive_loss_value,
                    "embedding_pairs": float(loss.embedding_pairs),
                    "embedding_local_positive_pairs": float(loss.embedding_local_positive_pairs),
                    "embedding_local_negative_pairs": float(loss.embedding_local_negative_pairs),
                    "embedding_ocr_negative_pairs": float(loss.embedding_ocr_negative_pairs),
                    "embedding_skipped_pairs": float(loss.embedding_skipped_pairs),
                    "presence_positive_samples": float(positive_samples),
                    "presence_negative_samples": float(negative_samples),
                    "presence_negative_ratio": negative_samples / max(1, len(batch.sample_ids)),
                    "embedding_candidate_positive_pairs": float(loss.embedding_candidate_positive_pairs),
                    "embedding_candidate_negative_pairs": float(loss.embedding_candidate_negative_pairs),
                    "embedding_positive_pairs": float(loss.embedding_selected_positive_pairs),
                    "embedding_negative_pairs": float(loss.embedding_selected_negative_pairs),
                    "embedding_negative_ratio": loss.embedding_selected_negative_pairs
                    / max(1, loss.embedding_selected_positive_pairs + loss.embedding_selected_negative_pairs),
                    "samples_per_second": float(len(batch.sample_ids)) / batch_time,
                    "batch_time": batch_time,
                }
                progress.set_postfix(loss=f"{step_metrics['total_loss']:.4f}")
                should_log = global_step == 1 or batch_index == total_batches or global_step % max(1, settings.log_interval) == 0
                if should_log:
                    metrics_file.write(json.dumps(step_metrics, sort_keys=True) + "\n")
                    metrics_file.flush()
            last_metrics = validate(
                model,
                val_loader,
                device,
                settings,
                error_pair_audit_path=settings.output_dir / "error_pairs" / f"epoch_{epoch:04d}.jsonl",
            )
            last_metrics.update(
                {
                    "record_type": "roi_validation",
                    "training_phase": phase.name,
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "train_loss": train_loss / max(1, batches),
                    "train_presence_loss": train_presence_loss / max(1, batches),
                    "train_short_mask_loss": train_short_mask_loss / max(1, batches),
                    "train_embedding_loss": train_embedding_loss / max(1, batches),
                    "train_embedding_memory_loss": train_embedding_memory_loss / max(1, batches),
                    "train_embedding_memory_pairs": float(train_embedding_memory_pairs),
                    "train_embedding_pairs": float(train_embedding_pairs),
                    "train_embedding_local_positive_pairs": float(train_embedding_local_positive_pairs),
                    "train_embedding_local_negative_pairs": float(train_embedding_local_negative_pairs),
                    "train_embedding_ocr_negative_pairs": float(train_embedding_ocr_negative_pairs),
                    "train_embedding_skipped_pairs": float(train_embedding_skipped_pairs),
                    "train_presence_positive_samples": float(train_presence_positive_samples),
                    "train_presence_negative_samples": float(train_presence_negative_samples),
                    "train_presence_negative_ratio": train_presence_negative_samples
                    / max(1, train_presence_positive_samples + train_presence_negative_samples),
                    "train_embedding_candidate_positive_pairs": float(train_embedding_candidate_positive_pairs),
                    "train_embedding_candidate_negative_pairs": float(train_embedding_candidate_negative_pairs),
                    "train_embedding_positive_pairs": float(train_embedding_positive_pairs),
                    "train_embedding_negative_pairs": float(train_embedding_negative_pairs),
                    "train_embedding_negative_ratio": train_embedding_negative_pairs
                    / max(1, train_embedding_positive_pairs + train_embedding_negative_pairs),
                    "train_embedding_batches_without_positive_pairs": float(train_embedding_batches_without_positive_pairs),
                    "validation_overlaps_training": has_validation_overlap,
                    "train_samples": float(len(train_dataset)),
                    "val_samples": float(len(val_dataset)),
                    "val_positive_segments": float(val_dataset.summary.positive_segments),
                    "val_repeated_positive_segments": float(val_dataset.summary.repeated_positive_segments),
                    "val_same_segment_pairs": float(val_dataset.summary.same_segment_pairs),
                    "epoch_seconds": time.perf_counter() - epoch_start,
                }
            )
            last_metrics["checkpoint_score"] = checkpoint_score(last_metrics, phase.name)
            checkpoint_saved = should_save_roi_best_checkpoint(last_metrics, best_metrics, phase.name)
            if checkpoint_saved:
                best_presence_f1 = max(best_presence_f1, last_metrics["presence_f1"])
                best_epoch = epoch
                best_metrics = dict(last_metrics)
            last_metrics["best_epoch"] = float(best_epoch)
            last_metrics["best_checkpoint_score"] = checkpoint_score(best_metrics or last_metrics, phase.name)
            epoch_checkpoint_path = save_epoch_checkpoint(
                settings,
                epoch,
                global_step,
                best_presence_f1,
                best_epoch,
                best_metrics or {},
                model,
                optimizer,
                last_metrics,
                phase.name,
            )
            epoch_metrics_path = save_epoch_metrics(settings, epoch, last_metrics)
            metrics_file.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            metrics_file.flush()
            if checkpoint_saved:
                payload = checkpoint_payload(
                    settings,
                    epoch,
                    global_step,
                    best_presence_f1,
                    best_epoch,
                    best_metrics or {},
                    model,
                    optimizer,
                    last_metrics,
                    phase.name,
                )
                torch.save(payload, phase_best_checkpoint_path(settings, phase.name))
                torch.save(payload, settings.output_dir / "best.pt")
            epoch_message = format_epoch_summary(epoch, total_epochs, last_metrics)
            epoch_message += f"\n  output: checkpoint={epoch_checkpoint_path} metrics={epoch_metrics_path}"
            epoch_message += f"\n  best: checkpoint={settings.output_dir / 'best.pt'} saved={str(checkpoint_saved).lower()} step={global_step}"
            print(epoch_message, flush=True)
    summary = training_summary(
        settings,
        completed_epoch=total_epochs,
        total_epochs=total_epochs,
        best_metrics=best_metrics or last_metrics,
    )
    (settings.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return last_metrics


def parse_args(argv: list[str] | None = None) -> RoiTrainSettings:
    parser = argparse.ArgumentParser(description="Train ROI Presence + Embedding subtitle model.")
    parser.add_argument("--train-root", type=Path, action="append", dest="train_roots")
    parser.add_argument("--val-root", type=Path, default=RoiTrainSettings().val_root)
    parser.add_argument("--output-dir", type=Path, default=RoiTrainSettings().output_dir)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resize-roi", type=parse_roi_size)
    parser.add_argument("--batch-size", type=int, default=RoiTrainSettings().batch_size)
    parser.add_argument("--presence-epochs", type=int, default=RoiTrainSettings().presence_epochs)
    parser.add_argument("--embedding-epochs", type=int, default=RoiTrainSettings().embedding_epochs)
    parser.add_argument("--joint-epochs", type=int, default=RoiTrainSettings().joint_epochs)
    parser.add_argument("--lr", type=float, default=RoiTrainSettings().learning_rate)
    parser.add_argument("--joint-lr", type=float, default=RoiTrainSettings().joint_learning_rate)
    parser.add_argument("--max-samples", type=int, help="Maximum ROI training sample count.")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--positive-ratio", type=float, help="Target subtitle-present fraction in each training batch.")
    parser.add_argument("--negative-ratio", type=float, help="Target no-subtitle fraction in each training batch.")
    parser.add_argument("--val-positive-ratio", type=float, help="Target subtitle-present ROI validation sample ratio in [0, 1].")
    parser.add_argument("--val-negative-ratio", type=float, help="Target no-subtitle ROI validation sample ratio in [0, 1].")
    parser.add_argument("--short-positive-loss-weight", type=float, default=RoiTrainSettings().short_positive_loss_weight)
    parser.add_argument("--short-positive-mask-loss-weight", type=float, default=RoiTrainSettings().short_positive_mask_loss_weight)
    parser.add_argument("--embedding-loss-weight", type=float, default=RoiTrainSettings().embedding_loss_weight)
    parser.add_argument("--embedding-loss-alpha", type=float, default=RoiTrainSettings().embedding_loss_alpha)
    parser.add_argument(
        "--embedding-negative-ratio",
        type=float,
        default=RoiTrainSettings().embedding_negative_ratio,
        help="Target different-segment fraction among selected in-batch embedding pairs.",
    )
    parser.add_argument("--embedding-samples-per-segment", type=int, default=RoiTrainSettings().embedding_samples_per_segment)
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
    parser.add_argument("--embedding-supcon-weight", type=float, default=RoiTrainSettings().embedding_supcon_weight)
    parser.add_argument("--embedding-tail-gamma-positive", type=float, default=RoiTrainSettings().embedding_tail_gamma_positive)
    parser.add_argument("--embedding-tail-gamma-negative", type=float, default=RoiTrainSettings().embedding_tail_gamma_negative)
    parser.add_argument("--embedding-tail-hard-negative-weight", type=float, default=RoiTrainSettings().embedding_tail_hard_negative_weight)
    parser.add_argument("--embedding-similarity-threshold", type=float, default=RoiTrainSettings().embedding_similarity_threshold)
    parser.add_argument("--presence-topk-ratio", type=float, default=RoiTrainSettings().presence_topk_ratio)
    parser.add_argument("--embedding-head", choices=("gap", "hybrid_lite", "local_contrast"), default=RoiTrainSettings().embedding_head_type)
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
    phase_epoch_values = (args.presence_epochs, args.embedding_epochs, args.joint_epochs)
    if any(value < 0 for value in phase_epoch_values):
        parser.error("phase epoch counts must be non-negative")
    if not any(value > 0 for value in phase_epoch_values):
        parser.error("at least one phase epoch count must be positive")
    if args.joint_lr <= 0.0:
        parser.error("--joint-lr must be positive")
    if not 0.0 <= args.embedding_negative_ratio <= 1.0:
        parser.error("--embedding-negative-ratio must be in [0, 1]")
    if args.embedding_samples_per_segment < 1:
        parser.error("--embedding-samples-per-segment must be positive")
    if args.embedding_tail_gamma_positive <= 0.0:
        parser.error("--embedding-tail-gamma-positive must be positive")
    if args.embedding_tail_gamma_negative <= 0.0:
        parser.error("--embedding-tail-gamma-negative must be positive")
    if args.embedding_tail_hard_negative_weight <= 0.0:
        parser.error("--embedding-tail-hard-negative-weight must be positive")
    if args.embedding_supcon_weight < 0.0:
        parser.error("--embedding-supcon-weight must be non-negative")
    if args.short_positive_loss_weight <= 0.0:
        parser.error("--short-positive-loss-weight must be positive")
    if args.short_positive_mask_loss_weight < 0.0:
        parser.error("--short-positive-mask-loss-weight must be non-negative")
    if args.presence_topk_ratio <= 0.0 or args.presence_topk_ratio > 1.0:
        parser.error("--presence-topk-ratio must be in (0, 1]")
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
        presence_epochs=args.presence_epochs,
        embedding_epochs=args.embedding_epochs,
        joint_epochs=args.joint_epochs,
        learning_rate=args.lr,
        joint_learning_rate=args.joint_lr,
        max_train_samples=max_train_samples,
        max_val_samples=args.max_val_samples,
        negative_ratio=empty_ratio,
        val_negative_ratio=val_empty_ratio,
        short_positive_loss_weight=args.short_positive_loss_weight,
        short_positive_mask_loss_weight=args.short_positive_mask_loss_weight,
        embedding_loss_weight=args.embedding_loss_weight,
        embedding_loss_alpha=args.embedding_loss_alpha,
        embedding_negative_ratio=args.embedding_negative_ratio,
        embedding_samples_per_segment=args.embedding_samples_per_segment,
        embedding_pair_frame_window=args.embedding_pair_frame_window,
        embedding_ocr_negative_enabled=args.embedding_ocr_negative_enabled,
        embedding_ocr_negative_max_similarity=args.embedding_ocr_negative_max_similarity,
        embedding_positive_consistency_beta=args.embedding_positive_consistency_beta,
        embedding_positive_consistency_margin=args.embedding_positive_consistency_margin,
        embedding_temperature=args.embedding_temperature,
        embedding_supcon_weight=args.embedding_supcon_weight,
        embedding_tail_gamma_positive=args.embedding_tail_gamma_positive,
        embedding_tail_gamma_negative=args.embedding_tail_gamma_negative,
        embedding_tail_hard_negative_weight=args.embedding_tail_hard_negative_weight,
        embedding_similarity_threshold=args.embedding_similarity_threshold,
        presence_topk_ratio=args.presence_topk_ratio,
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
    parser.add_argument("--error-pair-audit", type=Path)
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
        embedding_negative_ratio=float(raw_settings.get("embedding_negative_ratio", RoiTrainSettings().embedding_negative_ratio)),
        embedding_samples_per_segment=int(raw_settings.get("embedding_samples_per_segment", RoiTrainSettings().embedding_samples_per_segment)),
        embedding_supcon_weight=float(raw_settings.get("embedding_supcon_weight", RoiTrainSettings().embedding_supcon_weight)),
        embedding_tail_gamma_positive=float(raw_settings.get("embedding_tail_gamma_positive", RoiTrainSettings().embedding_tail_gamma_positive)),
        embedding_tail_gamma_negative=float(raw_settings.get("embedding_tail_gamma_negative", RoiTrainSettings().embedding_tail_gamma_negative)),
        embedding_tail_hard_negative_weight=float(
            raw_settings.get("embedding_tail_hard_negative_weight", RoiTrainSettings().embedding_tail_hard_negative_weight)
        ),
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
    metrics = validate(model, loader, device, settings, error_pair_audit_path=args.error_pair_audit)
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
