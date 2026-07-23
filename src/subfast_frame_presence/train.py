from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from subfast_shared.runtime import choose_device

from .config import FramePresenceTrainSettings
from .dataset import (
    RESIZE_ALIGNMENT,
    RESIZE_ALIGNMENT_MODE,
    RESIZE_INTERPOLATION,
    FramePresenceDataset,
    FramePresenceMacroBatch,
    aligned_resize_size,
    collate_frame_presence_batch,
    is_roi_root,
)
from .loss import FramePresenceLoss, FramePresenceLossInput, frame_presence_macro_loss
from .metrics import acceptance, checkpoint_rank, presence_metrics
from .model import FramePresenceModel, fuse_frame_presence_for_inference
from .sampler import MixedMacroBatchSampler


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOSS_NAMES = ("loss", "presence_bce", "presence_margin", "region_bce", "region_dice")
_BENCHMARK_WARMUP = 20
_BENCHMARK_ITERATIONS = 100
_BENCHMARK_ROUNDS = 7
_V3_MPS_MEDIAN_MS = 0.84
_MPS_TARGET_MEDIAN_MS = _V3_MPS_MEDIAN_MS / 2.0


def parse_frame_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected integer WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("frame size dimensions must be positive")
    return width, height


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def format_dataset_summary(name: str, dataset: FramePresenceDataset) -> str:
    summary = dataset.summary
    roots = ", ".join(f"{root}={count}" for root, count in sorted(summary.roots.items()))
    sample_types = ", ".join(f"{kind}={count}" for kind, count in sorted(summary.sample_types.items()))
    return (
        f"{name}: samples={summary.total} positive={summary.positive} empty={summary.empty} "
        f"manual_drop_image_exclusions={summary.dropped} positive_ratio={summary.positive_ratio:.3f} "
        f"empty_ratio={summary.empty_ratio:.3f} types=[{sample_types}] roots=[{roots}]"
    )


def make_dataset(settings: FramePresenceTrainSettings, *, train: bool) -> FramePresenceDataset:
    return FramePresenceDataset(
        settings.train_roots if train else settings.val_roots,
        resize_scale=settings.resize_scale,
        min_subtitle_short_edge=settings.min_subtitle_short_edge,
        protect_small_subtitles=train,
        random_crop_views=settings.random_crop_views if train else 0,
        random_crop_scale=(settings.random_crop_min_scale, settings.random_crop_max_scale),
        seed=settings.seed,
        max_samples=settings.max_train_samples if train else settings.max_val_samples,
    )


def _input_scope(settings: FramePresenceTrainSettings) -> str:
    return "mixed_full_frame_roi" if any(is_roi_root(root) for root in settings.train_roots) else "full_frame"


def _domain_metrics(metrics: dict[str, float], sample_type: str) -> dict[str, float]:
    prefix = f"{sample_type}_"
    return {
        name.removeprefix(prefix): value
        for name, value in metrics.items()
        if name.startswith(prefix)
    }


def _validation_acceptance(metrics: dict[str, float], *, complete_validation: bool) -> dict[str, bool]:
    return acceptance(metrics, complete_validation=complete_validation)


def _validation_rank(metrics: dict[str, float]) -> tuple[float, ...]:
    return checkpoint_rank(metrics)


def _json_write(path: Path, payload: object) -> None:
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _jsonl_write(path: Path, rows: list[dict[str, object]]) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    partial.replace(path)


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_diff() -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=_PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def _write_reproducibility_files(
    output_dir: Path,
    settings: FramePresenceTrainSettings,
    train_dataset: FramePresenceDataset,
    val_dataset: FramePresenceDataset,
) -> None:
    code_files = sorted((_PROJECT_ROOT / "src" / "subfast_frame_presence").glob("*.py"))
    tracked_files = code_files + [
        _PROJECT_ROOT / "pyproject.toml",
        _PROJECT_ROOT / "uv.lock",
    ]
    source_patch = _git_diff()
    snapshot = {
        "git_revision": _git_revision(),
        "source_patch": "source.patch" if source_patch else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "files": {
            str(path.relative_to(_PROJECT_ROOT)): _sha256(path)
            for path in tracked_files
            if path.is_file()
        },
    }
    run_config = {
        "model_name": settings.model_name,
        "architecture_version": settings.architecture_version,
        "settings": settings.model_dump(mode="json"),
        "input_contract": {
            "input": "one RGB full frame, ROI, or arbitrary image region per model item",
            "source_scope": _input_scope(settings),
            "shape": [3, "aligned_height", "aligned_width"],
            "pixel_values": "RGB float32 scaled to [0,1]",
            "resize": {
                "mode": "source_scale_then_stretch_align",
                "resize_scale": settings.resize_scale,
                "align_to": RESIZE_ALIGNMENT,
                "alignment_mode": RESIZE_ALIGNMENT_MODE,
                "interpolation": RESIZE_INTERPOLATION,
                "reference_source_size": list(settings.reference_source_size),
                "reference_output_size": list(
                    aligned_resize_size(settings.reference_source_size, settings.resize_scale)
                ),
            },
            "training_augmentation": {
                "random_crop_views": settings.random_crop_views,
                "random_crop_scale": [settings.random_crop_min_scale, settings.random_crop_max_scale],
                "positive_crop_rule": "fully retain every refined subtitle box",
            },
            "small_subtitle_protection": {
                "min_subtitle_short_edge": settings.min_subtitle_short_edge,
                "maximum_resize_scale": 1.0,
                "positive_rule": "minimum scale satisfying every refined box when possible",
                "negative_rule": "sample actual positive resize-scale distribution",
                "validation_uses_protection": False,
                "warnings": "small_subtitle_warnings.jsonl",
            },
            "prohibited": ["ROI mask", "padding"],
        },
        "training_contract": {
            "batching": "mixed logical macro batch split into exact-HxW execution micro batches",
            "optimizer_steps_per_macro_batch": 1,
            "loss_reduction": "global across the complete logical macro batch",
            "normalization": settings.normalization,
            "gradient_clip_norm": settings.gradient_clip_norm,
            "initialization": "fresh random Kaiming initialization",
        },
        "evaluation_contract": {
            "decision_threshold": 0.5,
            "max_epochs": 10,
            "acceptance": {
                "recall": 1.0,
                "f1": 1.0,
                "false_positive": 0,
                "false_negative": 0,
                "gap": 0.8,
            },
        },
        "datasets": {
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "train_manifest": "data_manifest_train.jsonl",
            "validation_manifest": "data_manifest_validation.jsonl",
        },
        "initialization": {"type": "fresh_random", "seed": settings.seed},
    }
    _json_write(output_dir / "source_snapshot.json", snapshot)
    if source_patch:
        (output_dir / "source.patch").write_text(source_patch, encoding="utf-8")
    _json_write(output_dir / "run_config.json", run_config)
    _jsonl_write(output_dir / "data_manifest_train.jsonl", train_dataset.manifest())
    _jsonl_write(output_dir / "data_manifest_validation.jsonl", val_dataset.manifest())


def _loss_kwargs(settings: FramePresenceTrainSettings) -> dict[str, float]:
    return {
        "region_loss_weight": settings.region_loss_weight,
        "region_dice_weight": settings.region_dice_weight,
        "margin_loss_weight": settings.margin_loss_weight,
        "positive_logit_margin": settings.positive_logit_margin,
        "negative_logit_margin": settings.negative_logit_margin,
    }


def _loss_values(loss: FramePresenceLoss) -> dict[str, float]:
    return {
        "loss": float(loss.total.detach().cpu()),
        "presence_bce": float(loss.presence_bce.detach().cpu()),
        "presence_margin": float(loss.presence_margin.detach().cpu()),
        "region_bce": float(loss.region_bce.detach().cpu()),
        "region_dice": float(loss.region_dice.detach().cpu()),
    }


def _make_train_loader(
    dataset: FramePresenceDataset,
    settings: FramePresenceTrainSettings,
    epoch: int,
) -> DataLoader:
    dataset.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_sampler=MixedMacroBatchSampler(
            dataset,
            batch_size=settings.batch_size,
            seed=settings.seed,
            epoch=epoch,
        ),
        num_workers=settings.num_workers,
        pin_memory=False,
        collate_fn=collate_frame_presence_batch,
    )


def _make_val_loader(dataset: FramePresenceDataset, settings: FramePresenceTrainSettings) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=False,
        collate_fn=collate_frame_presence_batch,
    )


def train_epoch(
    model: FramePresenceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    settings: FramePresenceTrainSettings,
    epoch: int,
    global_step: int,
    metrics_path: Path,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = {name: 0.0 for name in _LOSS_NAMES}
    sample_count = 0
    epoch_start = time.perf_counter()
    progress = tqdm(loader, desc=f"frame presence epoch {epoch}/{settings.epochs}", leave=False)
    for batch_index, batch in enumerate(progress, start=1):
        batch_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_inputs: list[FramePresenceLossInput] = []
        for micro in batch.micro_batches:
            images = micro.images.to(device)
            presence_logits, region_logits = model.forward_with_presence_map(images)
            loss_inputs.append(
                FramePresenceLossInput(
                    presence_logits=presence_logits,
                    region_logits=region_logits,
                    presence=micro.presence.to(device),
                    subtitle_masks=micro.subtitle_masks.to(device),
                    supervision_masks=micro.supervision_masks.to(device),
                )
            )
        loss = frame_presence_macro_loss(
            loss_inputs,
            **_loss_kwargs(settings),
        )
        loss.total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=settings.gradient_clip_norm,
        )
        optimizer.step()
        batch_size = batch.size
        values = _loss_values(loss)
        for name, value in values.items():
            totals[name] += value * batch_size
        sample_count += batch_size
        global_step += 1
        progress.set_postfix(loss=f"{values['loss']:.4f}")
        if global_step == 1 or batch_index == len(loader) or global_step % settings.log_interval == 0:
            _append_jsonl(
                metrics_path,
                {
                    "record_type": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "epoch_batch": batch_index,
                    "epoch_batches": len(loader),
                    "samples": sample_count,
                    "samples_per_second": batch_size / max(time.perf_counter() - batch_start, 1e-9),
                    "execution_micro_batches": len(batch.micro_batches),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    **values,
                },
            )
    metrics = {f"train_{name}": value / max(1, sample_count) for name, value in totals.items()}
    metrics["epoch_seconds"] = time.perf_counter() - epoch_start
    return metrics, global_step


@torch.inference_mode()
def validate(
    model: FramePresenceModel,
    loader: DataLoader,
    *,
    device: torch.device,
    settings: FramePresenceTrainSettings,
    scores_path: Path,
) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in _LOSS_NAMES}
    sample_count = 0
    logits_all: list[torch.Tensor] = []
    presence_all: list[torch.Tensor] = []
    sample_types_all: list[str] = []
    rows: list[dict[str, object]] = []
    for batch in loader:
        loss_inputs: list[FramePresenceLossInput] = []
        batch_size = batch.size
        for micro in batch.micro_batches:
            images = micro.images.to(device)
            presence = micro.presence.to(device)
            presence_logits, region_logits = model.forward_with_presence_map(images)
            loss_inputs.append(
                FramePresenceLossInput(
                    presence_logits=presence_logits,
                    region_logits=region_logits,
                    presence=presence,
                    subtitle_masks=micro.subtitle_masks.to(device),
                    supervision_masks=micro.supervision_masks.to(device),
                )
            )
            scores = torch.sigmoid(presence_logits).detach().cpu()
            logits_cpu = presence_logits.detach().cpu()
            targets = presence.detach().cpu()
            for sample_id, root, image_path, sample_type, score, logit, target in zip(
                micro.sample_ids,
                micro.roots,
                micro.image_paths,
                micro.sample_types,
                scores.tolist(),
                logits_cpu.tolist(),
                targets.tolist(),
                strict=True,
            ):
                rows.append(
                    {
                        "sample_key": f"{root}::{sample_id}",
                        "sample_id": sample_id,
                        "root": root,
                        "image_path": image_path,
                        "sample_type": sample_type,
                        "target": int(target > 0.5),
                        "presence_logit": logit,
                        "presence_score": score,
                        "prediction": int(score >= 0.5),
                        "decision_threshold": 0.5,
                    }
                )
            logits_all.append(logits_cpu)
            presence_all.append(targets)
            sample_types_all.extend(micro.sample_types)
        loss = frame_presence_macro_loss(loss_inputs, **_loss_kwargs(settings))
        values = _loss_values(loss)
        for name, value in values.items():
            totals[name] += value * batch_size
        sample_count += batch_size
    _jsonl_write(scores_path, rows)
    metrics = {
        f"val_{name}": value / max(1, sample_count)
        for name, value in totals.items()
    }
    all_logits = torch.cat(logits_all)
    all_presence = torch.cat(presence_all)
    metrics.update(presence_metrics(all_logits, all_presence))
    for sample_type in sorted(set(sample_types_all)):
        indices = torch.tensor([kind == sample_type for kind in sample_types_all], dtype=torch.bool)
        type_metrics = presence_metrics(all_logits[indices], all_presence[indices])
        metrics.update({f"{sample_type}_{name}": value for name, value in type_metrics.items()})
    return metrics


def resolve_resume_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("last.pt", "best.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"resume checkpoint not found: {path}")


def _checkpoint_payload(
    settings: FramePresenceTrainSettings,
    model: FramePresenceModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    best_epoch: int,
    best_metrics: dict[str, float],
    metrics: dict[str, float],
    warning_count: int,
    unique_warning_samples: int,
) -> dict[str, Any]:
    return {
        "model_name": settings.model_name,
        "model_type": "frame_presence",
        "architecture_version": model.architecture_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "settings": settings.model_dump(mode="json"),
        "model_settings": {
            "width": settings.width,
            "normalization": settings.normalization,
        },
        "preprocessing": {
            "input": f"complete_rgb_{_input_scope(settings)}",
            "input_value_range": [0.0, 1.0],
            "source_value_range": [0.0, 255.0],
            "value_scale": 1.0 / 255.0,
            "resize": {
                "mode": "source_scale_then_stretch_align",
                "resize_scale": settings.resize_scale,
                "align_to": RESIZE_ALIGNMENT,
                "alignment_mode": RESIZE_ALIGNMENT_MODE,
                "interpolation": RESIZE_INTERPOLATION,
                "reference_source_size": list(settings.reference_source_size),
                "reference_output_size": list(
                    aligned_resize_size(settings.reference_source_size, settings.resize_scale)
                ),
            },
            "padding": "none",
            "training_small_subtitle_protection": {
                "min_subtitle_short_edge": settings.min_subtitle_short_edge,
                "max_resize_scale": 1.0,
                "negative_scale_sampling": "positive_actual_resize_scale_distribution",
            },
            "validation_small_subtitle_protection": "disabled",
        },
        "training_contract": {
            "scheme": "A" if settings.normalization == "none" else "B",
            "normalization": settings.normalization,
            "macro_batch_size": settings.batch_size,
            "micro_batch_grouping": "exact_height_width",
            "padding": "none",
            "optimizer_step": "once_per_macro_batch",
            "loss_reduction": "global_per_macro_batch",
            "gradient_clip_norm": settings.gradient_clip_norm,
            "initialization": "fresh_random",
        },
        "small_subtitle_warnings": {
            "file": "small_subtitle_warnings.jsonl",
            "count": warning_count,
            "unique_samples": unique_warning_samples,
        },
        "score_contract": {
            "decision_threshold": 0.5,
            "transform": "sigmoid",
            "postprocessing": "none",
        },
        "epoch": epoch,
        "step": step,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "metrics": metrics,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }


def _inference_payload(
    checkpoint: dict[str, Any],
    model: FramePresenceModel,
) -> dict[str, Any]:
    optimized = fuse_frame_presence_for_inference(model).to("cpu")
    payload = {
        key: checkpoint[key]
        for key in (
            "model_type",
            "model_name",
            "architecture_version",
            "settings",
            "model_settings",
            "preprocessing",
            "score_contract",
            "training_contract",
            "small_subtitle_warnings",
            "epoch",
            "metrics",
        )
    }
    payload.update(
        {
            "model": optimized.state_dict(),
            "inference_fused": False,
            "operator_fusion": "none (V5 contains no BatchNorm)",
        }
    )
    return payload


def load_frame_presence_inference_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[FramePresenceModel, dict[str, Any]]:
    if path.is_dir():
        path = path / "best_inference.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "frame_presence":
        raise RuntimeError(f"invalid frame presence checkpoint: {path}")
    if int(checkpoint.get("architecture_version", -1)) != FramePresenceModel.architecture_version:
        raise RuntimeError(
            f"frame presence architecture mismatch: checkpoint=v{checkpoint.get('architecture_version')} "
            f"runtime=v{FramePresenceModel.architecture_version}"
        )
    model_settings = dict(checkpoint.get("model_settings") or {})
    width = int(model_settings.get("width", FramePresenceTrainSettings().width))
    normalization = str(model_settings.get("normalization", "none"))
    model = FramePresenceModel(width=width, normalization=normalization).eval()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def measure_frame_presence_latency(
    model: FramePresenceModel,
    images: torch.Tensor,
    device: torch.device,
    *,
    warmup: int = _BENCHMARK_WARMUP,
    iterations: int = _BENCHMARK_ITERATIONS,
    rounds: int = _BENCHMARK_ROUNDS,
) -> tuple[float, float]:
    """Measure steady-state batch-one eager forwards with one sync per timed window."""
    image = images[:1].to(device=device, dtype=torch.float32)
    for _ in range(warmup):
        model(image)
    _synchronize_device(device)
    timings: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            model(image)
        _synchronize_device(device)
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        timings.append(elapsed_ms / iterations)
    timings.sort()
    return timings[len(timings) // 2], timings[min(len(timings) - 1, int(len(timings) * 0.90))]


def _load_resume(
    path: Path,
    settings: FramePresenceTrainSettings,
    model: FramePresenceModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, int, dict[str, float], tuple[float, ...] | None]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "frame_presence":
        raise RuntimeError(f"invalid frame presence checkpoint: {path}")
    if int(checkpoint.get("architecture_version", -1)) != model.architecture_version:
        raise RuntimeError("resume checkpoint architecture does not match")
    previous = dict(checkpoint.get("settings") or {})
    current = settings.model_dump(mode="json")
    for name in (
        "train_roots",
        "val_roots",
        "resize_scale",
        "min_subtitle_short_edge",
        "resize_alignment",
        "resize_alignment_mode",
        "resize_interpolation",
        "random_crop_views",
        "random_crop_min_scale",
        "random_crop_max_scale",
        "width",
        "normalization",
        "seed",
    ):
        if previous.get(name) != current.get(name):
            raise RuntimeError(
                f"resume setting mismatch for {name}: checkpoint={previous.get(name)!r} current={current.get(name)!r}"
            )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    rng_state = checkpoint.get("rng_state")
    if isinstance(rng_state, dict):
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
    completed_epoch = int(checkpoint.get("epoch", 0))
    if completed_epoch >= settings.epochs:
        raise RuntimeError(
            f"resume checkpoint already completed epoch {completed_epoch}; --epochs must remain above it and no greater than 10"
        )
    metrics = {key: float(value) for key, value in dict(checkpoint.get("best_metrics") or {}).items()}
    rank = _validation_rank(metrics) if metrics else None
    return completed_epoch + 1, int(checkpoint.get("step", 0)), int(checkpoint.get("best_epoch", 0)), metrics, rank


def run_training(settings: FramePresenceTrainSettings) -> dict[str, object]:
    seed_everything(settings.seed)
    device = choose_device(settings.device)
    output_dir = settings.output_dir.expanduser().resolve()
    metrics_path = output_dir / "metrics.jsonl"
    warnings_path = output_dir / "small_subtitle_warnings.jsonl"
    warning_summary_path = output_dir / "small_subtitle_summary.json"
    if metrics_path.exists() and settings.resume is None:
        raise FileExistsError(f"output already contains a training run: {output_dir}")
    train_roots = {root.expanduser().resolve() for root in settings.train_roots}
    val_roots = {root.expanduser().resolve() for root in settings.val_roots}
    if train_roots & val_roots:
        raise ValueError("validation roots must not overlap training roots")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = make_dataset(settings, train=True)
    val_dataset = make_dataset(settings, train=False)
    if not len(train_dataset):
        raise RuntimeError("no training samples found")
    if not len(val_dataset):
        raise RuntimeError("no validation samples found")
    if train_dataset.summary.positive == 0 or train_dataset.summary.empty == 0:
        raise RuntimeError("training data must include both subtitle-presence classes")
    if val_dataset.summary.positive == 0 or val_dataset.summary.empty == 0:
        raise RuntimeError("validation data must include both subtitle-presence classes")
    model = FramePresenceModel(width=settings.width, normalization=settings.normalization).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    start_epoch = 1
    global_step = 0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_rank: tuple[float, ...] | None = None
    if settings.resume is not None:
        checkpoint_path = resolve_resume_checkpoint(settings.resume)
        start_epoch, global_step, best_epoch, best_metrics, best_rank = _load_resume(
            checkpoint_path,
            settings,
            model,
            optimizer,
            device,
        )
        print(f"resume={checkpoint_path} start_epoch={start_epoch} step={global_step}", flush=True)
    train_dataset.set_epoch(start_epoch)
    warning_rows: list[dict[str, object]] = []
    if settings.resume is not None and warnings_path.is_file():
        warning_rows = [json.loads(line) for line in warnings_path.read_text(encoding="utf-8").splitlines() if line]
    warning_rows.extend(warning.to_dict() for warning in train_dataset.small_subtitle_warnings)
    _jsonl_write(warnings_path, warning_rows)
    _write_reproducibility_files(output_dir, settings, train_dataset, val_dataset)
    print(f"{settings.model_name} device={device} output_dir={output_dir}", flush=True)
    print(
        f"config: resize_scale={settings.resize_scale:g} align_to={RESIZE_ALIGNMENT} "
        f"alignment={RESIZE_ALIGNMENT_MODE} interpolation={RESIZE_INTERPOLATION} "
        f"min_subtitle_short_edge={settings.min_subtitle_short_edge:g}px "
        f"batch_size={settings.batch_size} epochs={settings.epochs} lr={settings.learning_rate:g} "
        f"weight_decay={settings.weight_decay:g} width={settings.width} "
        f"normalization={settings.normalization}",
        flush=True,
    )
    print(format_dataset_summary("train", train_dataset), flush=True)
    print(format_dataset_summary("val", val_dataset), flush=True)
    val_loader = _make_val_loader(val_dataset, settings)
    last_metrics: dict[str, float] = {}
    completed_epoch = start_epoch - 1
    with metrics_path.open("a" if settings.resume is not None else "w", encoding="utf-8"):
        pass
    for epoch in range(start_epoch, settings.epochs + 1):
        train_loader = _make_train_loader(train_dataset, settings, epoch)
        if epoch != start_epoch:
            warning_rows.extend(warning.to_dict() for warning in train_dataset.small_subtitle_warnings)
            _jsonl_write(warnings_path, warning_rows)
        train_metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            settings=settings,
            epoch=epoch,
            global_step=global_step,
            metrics_path=metrics_path,
        )
        epoch_dir = output_dir / "epoch_outputs" / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        validation_metrics = validate(
            model,
            val_loader,
            device=device,
            settings=settings,
            scores_path=epoch_dir / "validation_scores.jsonl",
        )
        rank = _validation_rank(validation_metrics)
        is_best = best_rank is None or rank > best_rank
        if is_best:
            best_rank = rank
            best_epoch = epoch
            best_metrics = dict(validation_metrics)
        epoch_metrics = {
            "record_type": "epoch",
            "epoch": float(epoch),
            "step": float(global_step),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **validation_metrics,
            "best_epoch": float(best_epoch),
        }
        epoch_metrics["acceptance_pass"] = float(
            all(_validation_acceptance(
                validation_metrics,
                complete_validation=settings.max_val_samples is None,
            ).values())
        )
        payload = _checkpoint_payload(
            settings,
            model,
            optimizer,
            epoch=epoch,
            step=global_step,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            metrics=epoch_metrics,
            warning_count=len(warning_rows),
            unique_warning_samples=len(
                {(str(row["image_path"]), str(row["sample_id"])) for row in warning_rows}
            ),
        )
        torch.save(payload, epoch_dir / "checkpoint.pt")
        torch.save(payload, output_dir / "last.pt")
        if is_best:
            torch.save(payload, output_dir / "best.pt")
            torch.save(_inference_payload(payload, model), output_dir / "best_inference.pt")
        _json_write(epoch_dir / "metrics.json", epoch_metrics)
        _append_jsonl(metrics_path, epoch_metrics)
        last_metrics = validation_metrics
        completed_epoch = epoch
        print(
            f"epoch={epoch}/{settings.epochs} loss={validation_metrics['val_loss']:.4f} "
            f"accuracy={validation_metrics['presence_accuracy']:.6f} "
            f"f1={validation_metrics['presence_f1']:.6f} "
            f"recall={validation_metrics['presence_recall']:.6f} "
            f"fp={int(validation_metrics['presence_fp'])} fn={int(validation_metrics['presence_fn'])} "
            f"gap={validation_metrics['presence_gap']:.6f} "
            f"accepted={bool(epoch_metrics['acceptance_pass'])}",
            flush=True,
        )
        print(
            f"  full_frame_f1={validation_metrics['full_frame_presence_f1']:.6f} "
            f"full_frame_fp={int(validation_metrics['full_frame_presence_fp'])} "
            f"full_frame_fn={int(validation_metrics['full_frame_presence_fn'])} "
            f"roi_f1={validation_metrics['roi_presence_f1']:.6f} "
            f"roi_fp={int(validation_metrics['roi_presence_fp'])} "
            f"roi_fn={int(validation_metrics['roi_presence_fn'])}",
            flush=True,
        )
        if settings.early_stop and bool(epoch_metrics["acceptance_pass"]):
            break
    if not best_metrics:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.to("cpu")
    del optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    benchmark = _benchmark_checkpoint_in_fresh_process(
        output_dir / "best_inference.pt",
        device=device,
        image_size=aligned_resize_size(settings.reference_source_size, settings.resize_scale),
    )
    _json_write(output_dir / "benchmark.json", benchmark)
    _append_jsonl(metrics_path, benchmark)
    for checkpoint_name in ("best.pt", "best_inference.pt", "last.pt"):
        checkpoint_path = output_dir / checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["inference_benchmark"] = benchmark
        torch.save(checkpoint, checkpoint_path)
    print(
        f"inference median_ms={float(benchmark['median_ms']):.6f} "
        f"p90_round_ms={float(benchmark['p90_round_ms']):.6f}",
        flush=True,
    )
    final_acceptance = _validation_acceptance(
        best_metrics,
        complete_validation=settings.max_val_samples is None,
    )
    warning_summary: dict[str, object] = {
        "model_name": settings.model_name,
        "architecture_version": settings.architecture_version,
        "warning_events": len(warning_rows),
        "unique_warning_samples": len(
            {(str(row["image_path"]), str(row["sample_id"])) for row in warning_rows}
        ),
        "unsatisfied_events_at_scale_1": sum(not bool(row["protection_satisfied"]) for row in warning_rows),
        "warnings_file": str(warnings_path),
    }
    _json_write(warning_summary_path, warning_summary)
    for checkpoint_name in ("best.pt", "best_inference.pt", "last.pt"):
        checkpoint_path = output_dir / checkpoint_name
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["small_subtitle_warnings"] = warning_summary
        torch.save(checkpoint, checkpoint_path)
    summary: dict[str, object] = {
        "record_type": "frame_presence_training_summary",
        "model_name": settings.model_name,
        "model_type": "frame_presence",
        "architecture_version": model.architecture_version,
        "device": str(device),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_scope": _input_scope(settings),
        "full_frame_input": True,
        "roi_input": any(is_roi_root(root) for root in settings.train_roots),
        "input_preprocessing": {
            "resize_scale": settings.resize_scale,
            "align_to": RESIZE_ALIGNMENT,
            "alignment_mode": RESIZE_ALIGNMENT_MODE,
            "interpolation": RESIZE_INTERPOLATION,
            "input_value_range": [0.0, 1.0],
            "padding": "none",
        },
        "training_scheme": "A" if settings.normalization == "none" else "B",
        "normalization": settings.normalization,
        "completed_epoch": completed_epoch,
        "epoch_limit": settings.epochs,
        "best_epoch": best_epoch,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "validation": best_metrics,
        "last_validation": last_metrics,
        "initialization": {"type": "fresh_random", "seed": settings.seed},
        "small_subtitle_warnings": warning_summary,
        "inference_benchmark": benchmark,
        "acceptance": final_acceptance,
        "accepted": all(final_acceptance.values()),
        "best_checkpoint": str(output_dir / "best.pt"),
        "best_inference_checkpoint": str(output_dir / "best_inference.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "metrics": str(metrics_path),
        "benchmark": str(output_dir / "benchmark.json"),
        "source_snapshot": str(output_dir / "source_snapshot.json"),
        "run_config": str(output_dir / "run_config.json"),
        "small_subtitle_warning_summary": str(warning_summary_path),
    }
    _json_write(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> FramePresenceTrainSettings:
    defaults = FramePresenceTrainSettings()
    parser = argparse.ArgumentParser(description="Train subtitle presence on full frames, ROIs, and random crops.")
    parser.add_argument("--train-root", type=Path, action="append")
    parser.add_argument("--val-root", dest="val_roots", metavar="VAL_ROOT", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=defaults.early_stop,
    )
    parser.add_argument("--resize-scale", type=float, default=defaults.resize_scale)
    parser.add_argument(
        "--min-subtitle-short-edge",
        type=float,
        default=defaults.min_subtitle_short_edge,
    )
    parser.add_argument(
        "--reference-source-size",
        type=parse_frame_size,
        default=defaults.reference_source_size,
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--random-crop-views", type=int, default=defaults.random_crop_views)
    parser.add_argument("--random-crop-min-scale", type=float, default=defaults.random_crop_min_scale)
    parser.add_argument("--random-crop-max-scale", type=float, default=defaults.random_crop_max_scale)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument(
        "--normalization",
        choices=("none", "group_norm"),
        default=defaults.normalization,
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=defaults.gradient_clip_norm)
    parser.add_argument("--log-interval", type=int, default=defaults.log_interval)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 10:
        parser.error("--epochs must be between 1 and 10")
    for name, value in (
        ("--batch-size", args.batch_size),
        ("--width", args.width),
        ("--log-interval", args.log_interval),
    ):
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be non-negative")
    if not 0.0 < args.resize_scale <= 1.0:
        parser.error("--resize-scale must satisfy 0 < scale <= 1")
    if args.min_subtitle_short_edge <= 0.0:
        parser.error("--min-subtitle-short-edge must be positive")
    if args.gradient_clip_norm <= 0.0:
        parser.error("--gradient-clip-norm must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 0 <= args.random_crop_views <= 8:
        parser.error("--random-crop-views must be between 0 and 8")
    if not 0.0 < args.random_crop_min_scale <= args.random_crop_max_scale <= 1.0:
        parser.error("random crop scales must satisfy 0 < min <= max <= 1")
    for name, value in (("--max-train-samples", args.max_train_samples), ("--max-val-samples", args.max_val_samples)):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    return FramePresenceTrainSettings(
        train_roots=args.train_root or defaults.train_roots,
        val_roots=args.val_roots or defaults.val_roots,
        output_dir=args.output_dir,
        resume=args.resume,
        early_stop=args.early_stop,
        resize_scale=args.resize_scale,
        min_subtitle_short_edge=args.min_subtitle_short_edge,
        reference_source_size=args.reference_source_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        random_crop_views=args.random_crop_views,
        random_crop_min_scale=args.random_crop_min_scale,
        random_crop_max_scale=args.random_crop_max_scale,
        width=args.width,
        normalization=args.normalization,
        gradient_clip_norm=args.gradient_clip_norm,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
    )


def parse_benchmark_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = FramePresenceTrainSettings()
    parser = argparse.ArgumentParser(description="Benchmark full-frame Presence batch-1 latency.")
    parser.add_argument("checkpoint", type=Path, nargs="?", default=defaults.output_dir / "best_inference.pt")
    parser.add_argument("--image-size", type=parse_frame_size)
    parser.add_argument("--warmup", type=int, default=_BENCHMARK_WARMUP)
    parser.add_argument("--iterations", type=int, default=_BENCHMARK_ITERATIONS)
    parser.add_argument("--rounds", type=int, default=_BENCHMARK_ROUNDS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0 or args.rounds <= 0:
        parser.error("--iterations and --rounds must be positive")
    return args


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = choose_device(args.device)
    model, checkpoint = load_frame_presence_inference_checkpoint(args.checkpoint, device)
    preprocessing = dict(checkpoint.get("preprocessing") or {})
    resize = preprocessing.get("resize")
    if args.image_size is not None:
        image_size = tuple(args.image_size)
    elif isinstance(resize, dict):
        image_size = tuple(resize.get("reference_output_size") or ())
    else:
        image_size = tuple(resize or ()) if isinstance(resize, (list, tuple)) else ()
    if len(image_size) != 2:
        defaults = FramePresenceTrainSettings()
        image_size = aligned_resize_size(defaults.reference_source_size, defaults.resize_scale)
    width, height = int(image_size[0]), int(image_size[1])
    images = torch.rand(
        (1, 3, height, width),
        generator=torch.Generator().manual_seed(2026),
    )
    median_ms, p90_ms = measure_frame_presence_latency(
        model,
        images,
        device,
        warmup=args.warmup,
        iterations=args.iterations,
        rounds=args.rounds,
    )
    metrics: dict[str, object] = {
        "record_type": "frame_presence_inference_benchmark",
        "model_name": checkpoint.get("model_name", model.model_name),
        "checkpoint": str(args.checkpoint),
        "architecture_version": model.architecture_version,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "scope": "v5_no_normalization_eager_forward_only",
        "device": str(device),
        "torch_version": str(torch.__version__),
        "dtype": "float32",
        "batch_size": 1,
        "input_height": height,
        "input_width": width,
        "warmup": args.warmup,
        "iterations_per_round": args.iterations,
        "rounds": args.rounds,
        "median_ms": median_ms,
        "p90_round_ms": p90_ms,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return metrics


def _benchmark_checkpoint_in_fresh_process(
    checkpoint: Path,
    *,
    device: torch.device,
    image_size: tuple[int, int],
) -> dict[str, object]:
    command = [
        sys.executable,
        "-c",
        "from subfast_frame_presence.train import main_benchmark; main_benchmark()",
        str(checkpoint),
        "--device",
        str(device),
        "--image-size",
        f"{image_size[0]}x{image_size[1]}",
        "--warmup",
        str(_BENCHMARK_WARMUP),
        "--iterations",
        str(_BENCHMARK_ITERATIONS),
        "--rounds",
        str(_BENCHMARK_ROUNDS),
    ]
    result = subprocess.run(
        command,
        cwd=_PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"fresh-process inference benchmark failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("fresh-process inference benchmark returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("fresh-process inference benchmark returned a non-object result")
    return payload


def main(argv: list[str] | None = None) -> None:
    summary = run_training(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["accepted"]:
        print("frame presence acceptance requirements were not met", file=sys.stderr)
        raise SystemExit(1)


def main_benchmark(argv: list[str] | None = None) -> None:
    run_benchmark(parse_benchmark_args(argv))


if __name__ == "__main__":
    main()
