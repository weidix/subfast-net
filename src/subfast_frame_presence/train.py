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
from .dataset import FramePresenceDataset, collate_frame_presence_batch
from .loss import FramePresenceLoss, frame_presence_loss
from .metrics import acceptance, checkpoint_rank, presence_metrics
from .model import FramePresenceModel


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOSS_NAMES = ("loss", "presence_bce", "presence_margin", "region_bce", "region_dice")


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
    return (
        f"{name}: samples={summary.total} positive={summary.positive} empty={summary.empty} "
        f"dropped={summary.dropped} positive_ratio={summary.positive_ratio:.3f} "
        f"empty_ratio={summary.empty_ratio:.3f} roots=[{roots}]"
    )


def make_dataset(settings: FramePresenceTrainSettings, *, train: bool) -> FramePresenceDataset:
    return FramePresenceDataset(
        settings.train_roots if train else [settings.val_root],
        image_size=settings.image_size,
        max_samples=settings.max_train_samples if train else settings.max_val_samples,
    )


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
    snapshot = {
        "git_revision": _git_revision(),
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
            "input": "one complete RGB frame per model item",
            "shape": [3, settings.image_size[1], settings.image_size[0]],
            "pixel_values": "raw RGB values represented as float32",
            "preprocessing": "stretch resize only",
            "prohibited": ["ROI crop", "ROI mask", "padding", "normalization", "augmentation"],
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
    }
    _json_write(output_dir / "source_snapshot.json", snapshot)
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
        for sample_id, root, image_path, score, logit, target in zip(
            batch.sample_ids,
            batch.roots,
            batch.image_paths,
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
                    "target": int(target > 0.5),
                    "presence_logit": logit,
                    "presence_score": score,
                    "prediction": int(score >= 0.5),
                }
            )
        logits_all.append(logits_cpu)
        presence_all.append(targets)
        sample_count += batch_size
    _jsonl_write(scores_path, rows)
    metrics = {
        f"val_{name}": value / max(1, sample_count)
        for name, value in totals.items()
    }
    metrics.update(presence_metrics(torch.cat(logits_all), torch.cat(presence_all)))
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
) -> dict[str, Any]:
    return {
        "model_type": "frame_presence",
        "architecture_version": model.architecture_version,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "settings": settings.model_dump(mode="json"),
        "model_settings": {
            "width": settings.width,
            "evidence_kernel_size": settings.evidence_kernel_size,
        },
        "preprocessing": {
            "input": "complete_rgb_frame",
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


def _inference_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        key: checkpoint[key]
        for key in (
            "model_type",
            "architecture_version",
            "model",
            "settings",
            "model_settings",
            "preprocessing",
            "score_contract",
            "epoch",
            "metrics",
        )
    }


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
    for name in ("train_roots", "val_root", "image_size", "width", "evidence_kernel_size", "seed"):
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
    rank = checkpoint_rank(metrics) if metrics else None
    return completed_epoch + 1, int(checkpoint.get("step", 0)), int(checkpoint.get("best_epoch", 0)), metrics, rank


def run_training(settings: FramePresenceTrainSettings) -> dict[str, object]:
    seed_everything(settings.seed)
    device = choose_device(settings.device)
    output_dir = settings.output_dir.expanduser().resolve()
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and settings.resume is None:
        raise FileExistsError(f"output already contains a training run: {output_dir}")
    if any(root.expanduser().resolve() == settings.val_root.expanduser().resolve() for root in settings.train_roots):
        raise ValueError("validation root must not overlap a training root")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = make_dataset(settings, train=True)
    val_dataset = make_dataset(settings, train=False)
    if not len(train_dataset):
        raise RuntimeError("no full-frame training samples found")
    if not len(val_dataset):
        raise RuntimeError("no full-frame validation samples found")
    if train_dataset.summary.positive == 0 or train_dataset.summary.empty == 0:
        raise RuntimeError("training data must include both subtitle-presence classes")
    if val_dataset.summary.positive == 0 or val_dataset.summary.empty == 0:
        raise RuntimeError("validation data must include both subtitle-presence classes")
    _write_reproducibility_files(output_dir, settings, train_dataset, val_dataset)
    model = FramePresenceModel(
        width=settings.width,
        evidence_kernel_size=settings.evidence_kernel_size,
    ).to(device)
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
    print(f"frame_presence device={device} output_dir={output_dir}", flush=True)
    print(
        f"config: image_size={settings.image_size[0]}x{settings.image_size[1]} "
        f"batch_size={settings.batch_size} epochs={settings.epochs} lr={settings.learning_rate:g} "
        f"weight_decay={settings.weight_decay:g} width={settings.width} "
        f"evidence_kernel_size={settings.evidence_kernel_size}",
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
        rank = checkpoint_rank(validation_metrics)
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
            all(acceptance(validation_metrics, complete_validation=settings.max_val_samples is None).values())
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
        torch.save(payload, epoch_dir / "checkpoint.pt")
        torch.save(payload, output_dir / "last.pt")
        if is_best:
            torch.save(payload, output_dir / "best.pt")
            torch.save(_inference_payload(payload), output_dir / "best_inference.pt")
        _json_write(epoch_dir / "metrics.json", epoch_metrics)
        _append_jsonl(metrics_path, epoch_metrics)
        last_metrics = validation_metrics
        completed_epoch = epoch
        print(
            f"epoch={epoch}/{settings.epochs} loss={validation_metrics['val_loss']:.4f} "
            f"f1={validation_metrics['presence_f1']:.6f} "
            f"recall={validation_metrics['presence_recall']:.6f} "
            f"fp={int(validation_metrics['presence_fp'])} fn={int(validation_metrics['presence_fn'])} "
            f"gap={validation_metrics['presence_gap']:.6f} "
            f"accepted={bool(epoch_metrics['acceptance_pass'])}",
            flush=True,
        )
        if bool(epoch_metrics["acceptance_pass"]):
            break
    if not best_metrics:
        raise RuntimeError("training did not produce a validation checkpoint")
    final_acceptance = acceptance(
        best_metrics,
        complete_validation=settings.max_val_samples is None,
    )
    summary: dict[str, object] = {
        "record_type": "frame_presence_training_summary",
        "model_type": "frame_presence",
        "architecture_version": model.architecture_version,
        "device": str(device),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "full_frame_input": True,
        "input_preprocessing": "stretch resize only",
        "completed_epoch": completed_epoch,
        "epoch_limit": settings.epochs,
        "best_epoch": best_epoch,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "validation": best_metrics,
        "last_validation": last_metrics,
        "acceptance": final_acceptance,
        "accepted": all(final_acceptance.values()),
        "best_checkpoint": str(output_dir / "best.pt"),
        "best_inference_checkpoint": str(output_dir / "best_inference.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "metrics": str(metrics_path),
        "source_snapshot": str(output_dir / "source_snapshot.json"),
        "run_config": str(output_dir / "run_config.json"),
    }
    _json_write(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> FramePresenceTrainSettings:
    defaults = FramePresenceTrainSettings()
    parser = argparse.ArgumentParser(description="Train full-frame subtitle presence from scratch.")
    parser.add_argument("--train-root", type=Path, action="append")
    parser.add_argument("--val-root", type=Path, default=defaults.val_root)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-size", type=parse_frame_size, default=defaults.image_size)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--evidence-kernel-size", type=int, default=defaults.evidence_kernel_size)
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
    if args.evidence_kernel_size <= 1 or args.evidence_kernel_size % 2 == 0:
        parser.error("--evidence-kernel-size must be an odd integer greater than 1")
    for name, value in (("--max-train-samples", args.max_train_samples), ("--max-val-samples", args.max_val_samples)):
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive")
    return FramePresenceTrainSettings(
        train_roots=args.train_root or defaults.train_roots,
        val_root=args.val_root,
        output_dir=args.output_dir,
        resume=args.resume,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        width=args.width,
        evidence_kernel_size=args.evidence_kernel_size,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    summary = run_training(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["accepted"]:
        print("frame presence acceptance requirements were not met", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
