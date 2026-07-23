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
from .dataset import FramePresenceDataset, collate_frame_presence_batch, is_roi_root
from .loss import FramePresenceLoss, frame_presence_loss
from .metrics import acceptance, checkpoint_rank, presence_metrics
from .model import FramePresenceModel, fuse_frame_presence_for_inference


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
        f"dropped={summary.dropped} positive_ratio={summary.positive_ratio:.3f} "
        f"empty_ratio={summary.empty_ratio:.3f} types=[{sample_types}] roots=[{roots}]"
    )


def make_dataset(settings: FramePresenceTrainSettings, *, train: bool) -> FramePresenceDataset:
    return FramePresenceDataset(
        settings.train_roots if train else settings.val_roots,
        image_size=settings.image_size,
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
    checks = acceptance(metrics, complete_validation=complete_validation)
    for sample_type in ("full_frame", "roi"):
        domain = _domain_metrics(metrics, sample_type)
        if not domain:
            checks[f"{sample_type}_present"] = False
            continue
        for name, passed in acceptance(domain, complete_validation=complete_validation).items():
            if name != "validation_complete":
                checks[f"{sample_type}_{name}"] = passed
    return checks


def _validation_rank(metrics: dict[str, float]) -> tuple[float, ...]:
    domain_f1 = [
        _domain_metrics(metrics, sample_type).get("presence_f1", 0.0)
        for sample_type in ("full_frame", "roi")
    ]
    domain_recall = [
        _domain_metrics(metrics, sample_type).get("presence_recall", 0.0)
        for sample_type in ("full_frame", "roi")
    ]
    return (round(min(domain_f1), 8), round(min(domain_recall), 8), *checkpoint_rank(metrics))


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
    initialization: dict[str, object] | None,
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
        "settings": settings.model_dump(mode="json"),
        "input_contract": {
            "input": "one RGB full frame, ROI, or arbitrary image region per model item",
            "source_scope": _input_scope(settings),
            "shape": [3, settings.image_size[1], settings.image_size[0]],
            "pixel_values": "raw RGB values represented as float32",
            "inference_preprocessing": "stretch resize only",
            "training_augmentation": {
                "random_crop_views": settings.random_crop_views,
                "random_crop_scale": [settings.random_crop_min_scale, settings.random_crop_max_scale],
                "positive_crop_rule": "fully retain at least one subtitle box",
            },
            "prohibited": ["ROI mask", "padding", "normalization"],
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
        "initialization": initialization,
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
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=False,
        generator=torch.Generator().manual_seed(settings.seed + epoch),
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
        images = batch.images.to(device)
        subtitle_masks = batch.subtitle_masks.to(device)
        supervision_masks = batch.supervision_masks.to(device)
        presence = batch.presence.to(device)
        optimizer.zero_grad(set_to_none=True)
        presence_logits, region_logits = model.forward_with_presence_map(images)
        loss = frame_presence_loss(
            presence_logits,
            region_logits,
            presence,
            subtitle_masks,
            supervision_masks,
            **_loss_kwargs(settings),
        )
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        batch_size = images.shape[0]
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
        images = batch.images.to(device)
        subtitle_masks = batch.subtitle_masks.to(device)
        supervision_masks = batch.supervision_masks.to(device)
        presence = batch.presence.to(device)
        presence_logits, region_logits = model.forward_with_presence_map(images)
        loss = frame_presence_loss(
            presence_logits,
            region_logits,
            presence,
            subtitle_masks,
            supervision_masks,
            **_loss_kwargs(settings),
        )
        batch_size = images.shape[0]
        values = _loss_values(loss)
        for name, value in values.items():
            totals[name] += value * batch_size
        scores = torch.sigmoid(presence_logits).detach().cpu()
        logits_cpu = presence_logits.detach().cpu()
        targets = presence.detach().cpu()
        for sample_id, root, image_path, sample_type, score, logit, target in zip(
            batch.sample_ids,
            batch.roots,
            batch.image_paths,
            batch.sample_types,
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
                }
            )
        logits_all.append(logits_cpu)
        presence_all.append(targets)
        sample_types_all.extend(batch.sample_types)
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


def resolve_initial_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    for name in ("best.pt", "last.pt"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"initial checkpoint not found: {path}")


def _load_initial_weights(
    path: Path,
    settings: FramePresenceTrainSettings,
    model: FramePresenceModel,
) -> dict[str, object]:
    checkpoint_path = resolve_initial_checkpoint(path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "frame_presence":
        raise RuntimeError(f"invalid frame presence initialization checkpoint: {checkpoint_path}")
    if bool(checkpoint.get("inference_fused")):
        raise RuntimeError("initialization requires a training checkpoint, not a fused inference checkpoint")
    if int(checkpoint.get("architecture_version", -1)) != model.architecture_version:
        raise RuntimeError("initialization checkpoint architecture does not match")
    model_settings = dict(checkpoint.get("model_settings") or {})
    if int(model_settings.get("width", -1)) != settings.width:
        raise RuntimeError("initialization checkpoint width does not match")
    preprocessing = dict(checkpoint.get("preprocessing") or {})
    if tuple(preprocessing.get("resize") or ()) != settings.image_size:
        raise RuntimeError("initialization checkpoint image size does not match")
    model.load_state_dict(checkpoint["model"])
    return {
        "checkpoint": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", 0)),
        "architecture_version": int(checkpoint["architecture_version"]),
        "optimizer": "reset",
    }


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
) -> dict[str, Any]:
    return {
        "model_type": "frame_presence",
        "architecture_version": model.architecture_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "settings": settings.model_dump(mode="json"),
        "model_settings": {
            "width": settings.width,
        },
        "preprocessing": {
            "input": f"complete_rgb_{_input_scope(settings)}",
            "resize": list(settings.image_size),
            "resize_mode": "stretch",
            "other_preprocessing": "none",
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
            "architecture_version",
            "settings",
            "model_settings",
            "preprocessing",
            "score_contract",
            "epoch",
            "metrics",
        )
    }
    payload.update(
        {
            "model": optimized.state_dict(),
            "inference_fused": True,
            "operator_fusion": "Conv2d-BatchNorm2d parameter folding",
        }
    )
    if checkpoint.get("initialization") is not None:
        payload["initialization"] = checkpoint["initialization"]
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
    model = FramePresenceModel(width=width).eval()
    if bool(checkpoint.get("inference_fused")):
        model = fuse_frame_presence_for_inference(model)
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
        "image_size",
        "random_crop_views",
        "random_crop_min_scale",
        "random_crop_max_scale",
        "width",
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
    model = FramePresenceModel(width=settings.width).to(device)
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
    initialization: dict[str, object] | None = None
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
    elif settings.init_checkpoint is not None:
        initialization = _load_initial_weights(settings.init_checkpoint, settings, model)
        print(
            f"init_checkpoint={initialization['checkpoint']} source_epoch={initialization['source_epoch']} "
            "optimizer=reset",
            flush=True,
        )
    _write_reproducibility_files(output_dir, settings, train_dataset, val_dataset, initialization)
    print(f"frame_presence device={device} output_dir={output_dir}", flush=True)
    print(
        f"config: image_size={settings.image_size[0]}x{settings.image_size[1]} "
        f"batch_size={settings.batch_size} epochs={settings.epochs} lr={settings.learning_rate:g} "
        f"weight_decay={settings.weight_decay:g} width={settings.width}",
        flush=True,
    )
    print(format_dataset_summary("train", train_dataset), flush=True)
    print(format_dataset_summary("val", val_dataset), flush=True)
    val_loader = _make_val_loader(val_dataset, settings)
    last_metrics: dict[str, float] = {}
    initial_metrics: dict[str, float] | None = None
    completed_epoch = start_epoch - 1
    with metrics_path.open("a" if settings.resume is not None else "w", encoding="utf-8"):
        pass
    if initialization is not None:
        initial_dir = output_dir / "epoch_outputs" / "epoch_0000"
        initial_dir.mkdir(parents=True, exist_ok=True)
        initial_metrics = validate(
            model,
            val_loader,
            device=device,
            settings=settings,
            scores_path=initial_dir / "validation_scores.jsonl",
        )
        initial_record: dict[str, object] = {
            "record_type": "initial_validation",
            "epoch": 0,
            **initial_metrics,
        }
        _json_write(initial_dir / "metrics.json", initial_record)
        _append_jsonl(metrics_path, initial_record)
        print(
            f"initial accuracy={initial_metrics['presence_accuracy']:.6f} "
            f"f1={initial_metrics['presence_f1']:.6f} "
            f"recall={initial_metrics['presence_recall']:.6f} "
            f"fp={int(initial_metrics['presence_fp'])} fn={int(initial_metrics['presence_fn'])}",
            flush=True,
        )
    for epoch in range(start_epoch, settings.epochs + 1):
        train_metrics, global_step = train_epoch(
            model,
            _make_train_loader(train_dataset, settings, epoch),
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
        )
        if initialization is not None:
            payload["initialization"] = initialization
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
        image_size=settings.image_size,
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
        f"p90_round_ms={float(benchmark['p90_round_ms']):.6f} "
        f"speedup_vs_v3={benchmark['speedup_vs_v3']} target_met={benchmark['target_met']}",
        flush=True,
    )
    final_acceptance = _validation_acceptance(
        best_metrics,
        complete_validation=settings.max_val_samples is None,
    )
    if device.type == "mps":
        final_acceptance["inference_speed"] = bool(benchmark["target_met"])
    summary: dict[str, object] = {
        "record_type": "frame_presence_training_summary",
        "model_type": "frame_presence",
        "architecture_version": model.architecture_version,
        "device": str(device),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_scope": _input_scope(settings),
        "full_frame_input": True,
        "roi_input": any(is_roi_root(root) for root in settings.train_roots),
        "input_preprocessing": "stretch resize; training also uses labeled random crops",
        "completed_epoch": completed_epoch,
        "epoch_limit": settings.epochs,
        "best_epoch": best_epoch,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "validation": best_metrics,
        "initial_validation": initial_metrics,
        "last_validation": last_metrics,
        "initialization": initialization,
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
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=defaults.early_stop,
    )
    parser.add_argument("--image-size", type=parse_frame_size, default=defaults.image_size)
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
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 0 <= args.random_crop_views <= 8:
        parser.error("--random-crop-views must be between 0 and 8")
    if not 0.0 < args.random_crop_min_scale <= args.random_crop_max_scale <= 1.0:
        parser.error("random crop scales must satisfy 0 < min <= max <= 1")
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    for name, value in (("--max-train-samples", args.max_train_samples), ("--max-val-samples", args.max_val_samples)):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    return FramePresenceTrainSettings(
        train_roots=args.train_root or defaults.train_roots,
        val_roots=args.val_roots or defaults.val_roots,
        output_dir=args.output_dir,
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
        early_stop=args.early_stop,
        image_size=args.image_size,
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
    image_size = tuple(args.image_size or preprocessing.get("resize") or FramePresenceTrainSettings().image_size)
    width, height = int(image_size[0]), int(image_size[1])
    images = torch.rand(
        (1, 3, height, width),
        generator=torch.Generator().manual_seed(2026),
    ) * 255.0
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
        "checkpoint": str(args.checkpoint),
        "architecture_version": model.architecture_version,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "scope": "conv_bn_folded_eager_forward_only",
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
        "v3_mps_median_ms": _V3_MPS_MEDIAN_MS if device.type == "mps" else None,
        "speedup_vs_v3": _V3_MPS_MEDIAN_MS / median_ms if device.type == "mps" else None,
        "target_median_ms": _MPS_TARGET_MEDIAN_MS if device.type == "mps" else None,
        "target_met": median_ms <= _MPS_TARGET_MEDIAN_MS if device.type == "mps" else None,
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
