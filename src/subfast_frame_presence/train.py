from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from subfast_shared.runtime import choose_device

from .benchmark import (
    benchmark_coreml_model,
    benchmark_model,
    benchmark_preprocess,
    export_coreml_model,
    validate_coreml_model,
)
from .config import FramePresenceTrainSettings
from .data import FramePresenceCacheDataset
from .loss import FramePresenceLoss, frame_presence_loss
from .metrics import checkpoint_rank, presence_metrics, region_metrics
from .model import FramePresenceModel


def _json_write(path: Path, payload: object) -> None:
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def _epoch_learning_rate(settings: FramePresenceTrainSettings, epoch: int) -> float:
    if settings.epochs <= 1:
        return settings.min_learning_rate
    progress = (epoch - 1) / (settings.epochs - 1)
    return settings.min_learning_rate + 0.5 * (
        settings.learning_rate - settings.min_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _loss_kwargs(settings: FramePresenceTrainSettings) -> dict[str, float]:
    return {
        "presence_margin_weight": settings.presence_margin_weight,
        "presence_hard_fraction": settings.presence_hard_fraction,
        "positive_margin": settings.presence_positive_margin,
        "negative_margin": settings.presence_negative_margin,
        "region_loss_weight": settings.region_loss_weight,
        "region_bce_weight": settings.region_bce_weight,
        "region_positive_margin": settings.region_positive_margin,
        "region_negative_margin": settings.region_negative_margin,
        "dice_weight": settings.region_dice_weight,
        "projection_weight": settings.region_projection_weight,
        "boundary_weight": settings.region_boundary_weight,
        "boundary_margin": settings.region_boundary_margin,
        "boundary_hard_fraction": settings.region_boundary_hard_fraction,
        "area_weight": settings.region_area_weight,
        "soft_area_limit": settings.region_soft_area_limit,
        "area_hard_fraction": settings.region_area_hard_fraction,
        "area_activation_threshold": settings.heatmap_threshold,
        "area_temperature": settings.region_area_temperature,
        "edge_weight": settings.region_edge_weight,
    }


def _augment_batch(
    images: torch.Tensor,
    focus: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = images.shape[0]
    luma = images[:, :1]
    gain = torch.empty((batch_size, 1, 1, 1), device=images.device).uniform_(0.85, 1.15)
    bias = torch.empty((batch_size, 1, 1, 1), device=images.device).uniform_(-0.10, 0.10)
    luma = (luma * gain + bias).clamp(-1.0, 1.0)
    focus = (focus * gain + bias).clamp(-1.0, 1.0)
    noise_scale = torch.empty((batch_size, 1, 1, 1), device=images.device).uniform_(0.0, 0.025)
    luma = (luma + torch.randn_like(luma) * noise_scale).clamp(-1.0, 1.0)
    focus = (focus + torch.randn_like(focus) * noise_scale).clamp(-1.0, 1.0)
    flipped = torch.rand((batch_size, 1, 1, 1), device=images.device) < 0.5
    luma = torch.where(flipped, torch.flip(luma, dims=(-1,)), luma)
    focus = torch.where(flipped, torch.flip(focus, dims=(-1,)), focus)
    targets = torch.where(flipped, torch.flip(targets, dims=(-1,)), targets)
    return torch.cat((luma, images[:, 1:]), dim=1), focus, targets


def _loss_values(loss: FramePresenceLoss) -> dict[str, float]:
    return {
        "loss": float(loss.total.detach().cpu()),
        "presence_bce": float(loss.presence_bce.detach().cpu()),
        "presence_margin": float(loss.presence_margin.detach().cpu()),
        "region_bce": float(loss.region_bce.detach().cpu()),
        "region_dice": float(loss.region_dice.detach().cpu()),
        "region_projection": float(loss.region_projection.detach().cpu()),
        "region_boundary": float(loss.region_boundary.detach().cpu()),
        "region_area": float(loss.region_area.detach().cpu()),
        "region_edge": float(loss.region_edge.detach().cpu()),
    }


def _counterfactual_batch(
    images: torch.Tensor,
    focus: torch.Tensor,
    focus_mode: torch.Tensor,
    targets: torch.Tensor,
    presence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    image_parts: list[torch.Tensor] = []
    focus_parts: list[torch.Tensor] = []
    mode_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    presence_parts: list[torch.Tensor] = []
    for mode in (0, 1, 2):
        selected_mode = (focus_mode - mode).abs() < 0.5
        positive_indices = torch.nonzero(
            (presence > 0.5) & selected_mode,
            as_tuple=False,
        ).flatten()
        negative_indices = torch.nonzero(
            (presence <= 0.5) & selected_mode,
            as_tuple=False,
        ).flatten()
        if not positive_indices.numel() or not negative_indices.numel():
            continue
        shuffled_negative = negative_indices[
            torch.randperm(negative_indices.numel(), device=images.device)
        ]
        donor_indices = shuffled_negative[
            torch.arange(positive_indices.numel(), device=images.device)
            % shuffled_negative.numel()
        ]
        full_mask = F.interpolate(
            targets[positive_indices],
            size=images.shape[-2:],
            mode="nearest",
        )
        full_alpha = F.avg_pool2d(
            F.max_pool2d(full_mask, kernel_size=5, stride=1, padding=2),
            kernel_size=3,
            stride=1,
            padding=1,
        ).clamp(0.0, 1.0)
        if mode == 1:
            focus_target = targets[positive_indices, :, 50:72, 7:57]
        elif mode == 2:
            focus_target = targets[positive_indices, :, 63:72, 5:59]
        else:
            focus_target = targets[positive_indices, :, 61:72, 10:54]
        focus_mask = F.interpolate(focus_target, size=focus.shape[-2:], mode="nearest")
        focus_alpha = F.avg_pool2d(
            F.max_pool2d(focus_mask, kernel_size=5, stride=1, padding=2),
            kernel_size=3,
            stride=1,
            padding=1,
        ).clamp(0.0, 1.0)
        positive_luma = images[positive_indices, :1]
        donor_luma = images[donor_indices, :1]
        erased_luma = positive_luma * (1.0 - full_alpha) + donor_luma * full_alpha
        transplanted_luma = positive_luma * full_alpha + donor_luma * (1.0 - full_alpha)
        coordinates = images[positive_indices, 1:]
        image_parts.extend(
            (
                torch.cat((erased_luma, coordinates), dim=1),
                torch.cat((transplanted_luma, coordinates), dim=1),
            )
        )
        positive_focus = focus[positive_indices]
        donor_focus = focus[donor_indices]
        focus_parts.extend(
            (
                positive_focus * (1.0 - focus_alpha) + donor_focus * focus_alpha,
                positive_focus * focus_alpha + donor_focus * (1.0 - focus_alpha),
            )
        )
        positive_targets = targets[positive_indices]
        mode_parts.extend((focus_mode[positive_indices], focus_mode[positive_indices]))
        target_parts.extend((torch.zeros_like(positive_targets), positive_targets))
        presence_parts.extend(
            (
                torch.zeros_like(presence[positive_indices]),
                torch.ones_like(presence[positive_indices]),
            )
        )
    if not image_parts:
        return None
    return tuple(
        torch.cat(parts, dim=0)
        for parts in (image_parts, focus_parts, mode_parts, target_parts, presence_parts)
    )  # type: ignore[return-value]


def _weighted_loss(
    original: FramePresenceLoss,
    counterfactual: FramePresenceLoss,
    weight: float,
) -> FramePresenceLoss:
    return FramePresenceLoss(
        **{
            field: getattr(original, field) + weight * getattr(counterfactual, field)
            for field in FramePresenceLoss.__dataclass_fields__
        }
    )


def train_epoch(
    model: FramePresenceModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    settings: FramePresenceTrainSettings,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    sample_count = 0
    for batch in loader:
        images = batch["image"].to(device)
        focus = batch["focus"].to(device)
        focus_mode = batch["focus_mode"].to(device)
        targets = batch["target"].to(device)
        presence = batch["presence"].to(device)
        if settings.augment:
            images, focus, targets = _augment_batch(images, focus, targets)
        counterfactual = (
            _counterfactual_batch(images, focus, focus_mode, targets, presence)
            if settings.counterfactual_weight > 0.0
            else None
        )
        optimizer.zero_grad(set_to_none=True)
        if counterfactual is None:
            model_images = images
            model_focus = focus
            model_focus_mode = focus_mode
        else:
            model_images = torch.cat((images, counterfactual[0]), dim=0)
            model_focus = torch.cat((focus, counterfactual[1]), dim=0)
            model_focus_mode = torch.cat((focus_mode, counterfactual[2]), dim=0)
        (
            all_presence_logits,
            all_region_logits,
            all_compact_region_logits,
        ) = model.forward_with_components(
            model_images,
            model_focus,
            model_focus_mode,
        )
        batch_size = images.shape[0]
        loss = frame_presence_loss(
            all_presence_logits[:batch_size],
            all_region_logits[:batch_size],
            all_compact_region_logits[:batch_size],
            presence,
            targets,
            **_loss_kwargs(settings),
        )
        if counterfactual is not None:
            counterfactual_loss = frame_presence_loss(
                all_presence_logits[batch_size:],
                all_region_logits[batch_size:],
                all_compact_region_logits[batch_size:],
                counterfactual[4],
                counterfactual[3],
                **_loss_kwargs(settings),
            )
            loss = _weighted_loss(loss, counterfactual_loss, settings.counterfactual_weight)
        loss.total.backward()
        optimizer.step()
        sample_count += batch_size
        for name, value in _loss_values(loss).items():
            totals[name] = totals.get(name, 0.0) + value * batch_size
    return {
        f"train_{name}": value / max(1, sample_count)
        for name, value in totals.items()
    }


@torch.inference_mode()
def validate(
    model: FramePresenceModel,
    loader: DataLoader,
    *,
    device: torch.device,
    settings: FramePresenceTrainSettings,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    model.eval()
    loss_sum = 0.0
    count = 0
    logits_all: list[torch.Tensor] = []
    region_logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    presence_all: list[torch.Tensor] = []
    indices_all: list[torch.Tensor] = []
    for batch in loader:
        images = batch["image"].to(device)
        focus = batch["focus"].to(device)
        focus_mode = batch["focus_mode"].to(device)
        targets = batch["target"].to(device)
        presence = batch["presence"].to(device)
        presence_logits, region_logits, compact_region_logits = model.forward_with_components(
            images,
            focus,
            focus_mode,
        )
        loss = frame_presence_loss(
            presence_logits,
            region_logits,
            compact_region_logits,
            presence,
            targets,
            **_loss_kwargs(settings),
        )
        batch_size = images.shape[0]
        loss_sum += float(loss.total.detach().cpu()) * batch_size
        count += batch_size
        logits_all.append(presence_logits.cpu())
        region_logits_all.append(region_logits.cpu())
        targets_all.append(targets.cpu())
        presence_all.append(presence.cpu())
        indices_all.append(batch["index"].cpu())
    logits = torch.cat(logits_all)
    region_logits = torch.cat(region_logits_all)
    targets = torch.cat(targets_all)
    presence = torch.cat(presence_all)
    indices = torch.cat(indices_all)
    metrics = {"val_loss": loss_sum / max(1, count)}
    metrics.update(
        presence_metrics(
            logits,
            presence,
            threshold=settings.decision_threshold,
        )
    )
    localization, region_records = region_metrics(
        region_logits,
        targets,
        presence,
        threshold=settings.heatmap_threshold,
    )
    metrics.update(localization)
    scores = torch.sigmoid(logits)
    diagnostics: list[dict[str, float]] = []
    for output_index, dataset_index in enumerate(indices.tolist()):
        record = {
            "index": float(dataset_index),
            "target": float(presence[output_index]),
            "score": float(scores[output_index]),
        }
        record.update(region_records[output_index])
        diagnostics.append(record)
    return metrics, diagnostics


@torch.inference_mode()
def calibrate_presence_gap(
    model: FramePresenceModel,
    loader: DataLoader,
    *,
    device: torch.device,
    target_score: float,
) -> dict[str, float | bool]:
    """Affine-calibrate separable validation logits to a symmetric score gap."""
    model.eval()
    logits_all: list[torch.Tensor] = []
    presence_all: list[torch.Tensor] = []
    for batch in loader:
        presence_logits, _ = model(
            batch["image"].to(device),
            batch["focus"].to(device),
            batch["focus_mode"].to(device),
        )
        logits_all.append(presence_logits.cpu())
        presence_all.append(batch["presence"].cpu())
    logits = torch.cat(logits_all)
    presence = torch.cat(presence_all) > 0.5
    if not bool(presence.any()) or not bool((~presence).any()):
        return {"applied": False, "ordered": False}
    minimum_positive = float(logits[presence].min())
    maximum_negative = float(logits[~presence].max())
    if minimum_positive <= maximum_negative:
        return {
            "applied": False,
            "ordered": False,
            "minimum_positive_logit": minimum_positive,
            "maximum_negative_logit": maximum_negative,
        }
    midpoint = 0.5 * (minimum_positive + maximum_negative)
    half_gap = 0.5 * (minimum_positive - maximum_negative)
    target_logit = math.log(target_score / (1.0 - target_score))
    factor = target_logit / half_gap
    old_scale = float(model.presence_output_scale.detach().cpu())
    old_bias = float(model.presence_output_bias.detach().cpu())
    model.presence_output_scale.fill_(old_scale * factor)
    model.presence_output_bias.fill_(factor * (old_bias - midpoint))
    return {
        "applied": True,
        "ordered": True,
        "target_score": target_score,
        "factor": factor,
        "minimum_positive_logit_before": minimum_positive,
        "maximum_negative_logit_before": maximum_negative,
    }


def _resolve_resume(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "last.pt"
    if not path.exists():
        raise ValueError(f"resume checkpoint does not exist: {path}")
    return path


def _checkpoint_payload(
    model: FramePresenceModel,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    settings: FramePresenceTrainSettings,
    train_data: FramePresenceCacheDataset,
    val_data: FramePresenceCacheDataset,
    metrics: dict[str, float],
    rank: tuple[float, ...],
) -> dict[str, Any]:
    contract = train_data.contract
    return {
        "model_type": "frame_presence",
        "kind": "subfast_frame_presence_training_checkpoint",
        "architecture_version": model.architecture_version,
        "epoch": epoch,
        "model": {
            "width": model.width,
            "kernel_size": model.kernel_size,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "preprocessing": {
            "input_kind": "full_frame_luma_xy_plus_automatic_rgb_focus",
            "input_width": contract.input_width,
            "input_height": contract.input_height,
            "focus_width": contract.focus_width,
            "focus_height": contract.focus_height,
            "normalization": "luma_uint8_to_minus_one_plus_one_xy_minus_one_plus_one",
            "resize_mode": "full_frame_stretch",
            "focus_resize_mode": "aspect_preserving_letterbox",
            "heatmap_stride_x": contract.heatmap_stride_x,
            "heatmap_stride_y": contract.heatmap_stride_y,
            "heatmap_width": contract.heatmap_width,
            "heatmap_height": contract.heatmap_height,
        },
        "outputs": {
            "presence": "sigmoid(coherent focus evidence + full-frame context)",
            "region": "sigmoid(interleaved subtitle-enclosing contour samples in full-frame coordinates)",
            "decision_threshold": settings.decision_threshold,
            "heatmap_threshold": settings.heatmap_threshold,
            "compact_activation_threshold": model.compact_activation_threshold,
            "expanded_activation_threshold": model.expanded_activation_threshold,
        },
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "settings": settings.model_dump(mode="json"),
        "train_cache": train_data.summary,
        "val_cache": val_data.summary,
        "validation": metrics,
        "checkpoint_rank": rank,
    }


def _inference_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: checkpoint[key]
        for key in (
            "model_type",
            "kind",
            "architecture_version",
            "epoch",
            "model",
            "preprocessing",
            "outputs",
            "model_state",
            "validation",
        )
    }
    if "deployment" in checkpoint:
        payload["deployment"] = checkpoint["deployment"]
    return payload


def _write_diagnostics(
    path: Path,
    diagnostics: list[dict[str, float]],
    dataset: FramePresenceCacheDataset,
) -> None:
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", encoding="utf-8") as file:
        for values in diagnostics:
            index = int(values["index"])
            record = dict(dataset.records[index])
            record.update(values)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    partial.replace(path)


def _acceptance(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "presence_recall": metrics["presence_recall"] == 1.0,
        "presence_f1": metrics["presence_f1"] == 1.0,
        "presence_gap": metrics["presence_gap"] >= 0.8,
        "region_contains_all": metrics["region_bbox_containment_rate"] == 1.0,
        "region_heatmap_area": metrics["region_area_limit_pass_rate"] == 1.0
        and metrics["region_heatmap_area_ratio_max"] <= 1.0,
        "region_heatmap_extra_area": metrics["region_extra_area_limit_pass_rate"] == 1.0
        and metrics["region_overflow_ratio_max"] <= 1.0,
        "region_not_full_frame": metrics["region_bbox_full_frame_ratio_max"] <= 0.25,
    }


def run_training(settings: FramePresenceTrainSettings) -> dict[str, object]:
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    device = choose_device(settings.device)
    train_data = FramePresenceCacheDataset(settings.train_cache)
    val_data = FramePresenceCacheDataset(settings.val_cache)
    if train_data.contract != val_data.contract:
        raise ValueError("training and validation caches use different preprocessing contracts")
    train_roots = train_data.summary.get("source_roots", {})
    if not isinstance(train_roots, dict) or len(train_roots) < 6:
        raise ValueError("training cache must cover all six configured sample roots")
    model = FramePresenceModel(
        width=settings.width,
        kernel_size=settings.kernel_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    completed_epoch = 0
    best_rank: tuple[float, ...] | None = None
    resumed_metrics: dict[str, float] = {}
    if settings.resume is not None:
        checkpoint = torch.load(_resolve_resume(settings.resume), map_location="cpu", weights_only=False)
        if int(checkpoint.get("architecture_version", -1)) != model.architecture_version:
            raise ValueError("resume checkpoint architecture version does not match")
        model_geometry = checkpoint.get("model", {})
        if (
            model_geometry.get("width") != settings.width
            or model_geometry.get("kernel_size") != settings.kernel_size
        ):
            raise ValueError("resume checkpoint model geometry does not match")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        completed_epoch = int(checkpoint["epoch"])
        stored_rank = checkpoint.get("checkpoint_rank")
        if stored_rank is not None:
            best_rank = tuple(float(value) for value in stored_rank)
        resumed_metrics = {
            key: float(value)
            for key, value in checkpoint.get("validation", {}).items()
        }

    output_dir = settings.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    train_loader = DataLoader(
        train_data,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        persistent_workers=settings.num_workers > 0,
        generator=torch.Generator().manual_seed(settings.seed),
    )
    val_loader = DataLoader(
        val_data,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        persistent_workers=settings.num_workers > 0,
    )
    best_epoch = completed_epoch
    best_metrics = resumed_metrics
    last_metrics = resumed_metrics
    for epoch in range(completed_epoch + 1, settings.epochs + 1):
        learning_rate = _epoch_learning_rate(settings, epoch)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            settings=settings,
        )
        validation, diagnostics = validate(
            model,
            val_loader,
            device=device,
            settings=settings,
        )
        rank = checkpoint_rank(validation)
        row = {
            "record_type": "epoch",
            "epoch": epoch,
            "learning_rate": learning_rate,
            **train_metrics,
            **validation,
        }
        _append_jsonl(metrics_path, row)
        checkpoint = _checkpoint_payload(
            model,
            optimizer,
            epoch=epoch,
            settings=settings,
            train_data=train_data,
            val_data=val_data,
            metrics=validation,
            rank=rank,
        )
        torch.save(checkpoint, output_dir / "last.pt")
        _write_diagnostics(output_dir / "last_scores.jsonl", diagnostics, val_data)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            best_metrics = validation
            torch.save(checkpoint, output_dir / "best.pt")
            torch.save(_inference_payload(checkpoint), output_dir / "best_inference.pt")
            _write_diagnostics(output_dir / "best_scores.jsonl", diagnostics, val_data)
        last_metrics = validation
        acceptance = _acceptance(validation)
        print(
            f"epoch={epoch} loss={validation['val_loss']:.4f} "
            f"f1={validation['presence_f1']:.6f} recall={validation['presence_recall']:.6f} "
            f"fp={int(validation['presence_fp'])} fn={int(validation['presence_fn'])} "
            f"gap={validation['presence_gap']:.6f} "
            f"contain={validation['region_bbox_containment_rate']:.6f} "
            f"overflow={validation['region_overflow_ratio']:.4f} "
            f"accepted={all(acceptance.values())}",
            flush=True,
        )

    best_path = output_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("training did not produce a best checkpoint")
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    calibration = calibrate_presence_gap(
        model,
        val_loader,
        device=device,
        target_score=settings.presence_calibration_target_score,
    )
    best_metrics, best_diagnostics = validate(
        model,
        val_loader,
        device=device,
        settings=settings,
    )
    best_checkpoint["model_state"] = model.state_dict()
    best_checkpoint["validation"] = best_metrics
    best_checkpoint["checkpoint_rank"] = checkpoint_rank(best_metrics)
    best_checkpoint["presence_calibration"] = calibration
    torch.save(best_checkpoint, best_path)
    torch.save(_inference_payload(best_checkpoint), output_dir / "best_inference.pt")
    _write_diagnostics(output_dir / "best_scores.jsonl", best_diagnostics, val_data)
    _append_jsonl(
        metrics_path,
        {
            "record_type": "presence_calibration",
            "epoch": best_epoch,
            **calibration,
            **best_metrics,
        },
    )
    benchmark_rows = benchmark_model(
        model,
        device=device,
        input_width=train_data.contract.input_width,
        input_height=train_data.contract.input_height,
        focus_width=train_data.contract.focus_width,
        focus_height=train_data.contract.focus_height,
        batch_sizes=[1, 2, 4, 8, 16, 32],
        warmup=settings.benchmark_warmup,
        iterations=settings.benchmark_iterations,
    )
    preprocess_benchmark = benchmark_preprocess(
        np.zeros((1080, 1920), dtype=np.uint8),
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        input_width=train_data.contract.input_width,
        input_height=train_data.contract.input_height,
        focus_width=train_data.contract.focus_width,
        focus_height=train_data.contract.focus_height,
        warmup=settings.benchmark_warmup,
        iterations=settings.benchmark_iterations,
    )
    coreml_path = export_coreml_model(
        model,
        output_dir / "best.mlpackage",
        input_width=train_data.contract.input_width,
        input_height=train_data.contract.input_height,
        focus_width=train_data.contract.focus_width,
        focus_height=train_data.contract.focus_height,
        maximum_batch_size=32,
    )
    coreml_benchmark_rows = benchmark_coreml_model(
        coreml_path,
        input_width=train_data.contract.input_width,
        input_height=train_data.contract.input_height,
        focus_width=train_data.contract.focus_width,
        focus_height=train_data.contract.focus_height,
        batch_sizes=[1, 2, 4, 8, 16, 32],
        warmup=settings.benchmark_warmup,
        iterations=settings.benchmark_iterations,
    )
    coreml_validation = validate_coreml_model(
        coreml_path,
        settings.val_cache,
        batch_size=32,
        decision_threshold=settings.decision_threshold,
        heatmap_threshold=settings.heatmap_threshold,
    )
    single = coreml_benchmark_rows[0]
    acceptance = _acceptance(coreml_validation)
    acceptance.update(
        {
            "single_frame_inference": single["median_ms"] < 0.3,
            "preprocessing": preprocess_benchmark["median_ms"] < 0.1,
            "batch_scaling": coreml_benchmark_rows[-1]["median_ms_per_frame"]
            < single["median_ms_per_frame"],
        }
    )
    summary: dict[str, object] = {
        "record_type": "frame_presence_training_summary",
        "best_epoch": best_epoch,
        "completed_epoch": settings.epochs,
        "device": str(device),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "full_frame_input": True,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "train_source_roots": train_data.summary.get("source_roots"),
        "validation_source_roots": val_data.summary.get("source_roots"),
        "validation": best_metrics,
        "coreml_validation": coreml_validation,
        "last_validation": last_metrics,
        "preprocessing_benchmark": preprocess_benchmark,
        "inference_benchmark": coreml_benchmark_rows,
        "pytorch_inference_benchmark": benchmark_rows,
        "inference_backend": "Core ML flexible batch 1...32, all compute units",
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "best_checkpoint": str(best_path),
        "best_inference_checkpoint": str(output_dir / "best_inference.pt"),
        "best_coreml_model": str(coreml_path),
        "metrics": str(metrics_path),
        "validation_scores": str(output_dir / "best_scores.jsonl"),
    }
    _json_write(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = FramePresenceTrainSettings()
    parser = argparse.ArgumentParser(description="Train subfast_frame_presence from scratch.")
    parser.add_argument("--train-cache", type=Path, default=defaults.train_cache)
    parser.add_argument("--val-cache", type=Path, default=defaults.val_cache)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--min-lr", type=float, default=defaults.min_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--kernel-size", type=int, default=defaults.kernel_size)
    parser.add_argument("--presence-margin-weight", type=float, default=defaults.presence_margin_weight)
    parser.add_argument("--presence-hard-fraction", type=float, default=defaults.presence_hard_fraction)
    parser.add_argument("--positive-margin", type=float, default=defaults.presence_positive_margin)
    parser.add_argument("--negative-margin", type=float, default=defaults.presence_negative_margin)
    parser.add_argument("--region-loss-weight", type=float, default=defaults.region_loss_weight)
    parser.add_argument("--region-bce-weight", type=float, default=defaults.region_bce_weight)
    parser.add_argument("--region-positive-margin", type=float, default=defaults.region_positive_margin)
    parser.add_argument("--region-negative-margin", type=float, default=defaults.region_negative_margin)
    parser.add_argument("--region-dice-weight", type=float, default=defaults.region_dice_weight)
    parser.add_argument(
        "--region-projection-weight",
        type=float,
        default=defaults.region_projection_weight,
    )
    parser.add_argument("--region-boundary-weight", type=float, default=defaults.region_boundary_weight)
    parser.add_argument("--region-boundary-margin", type=float, default=defaults.region_boundary_margin)
    parser.add_argument(
        "--region-boundary-hard-fraction",
        type=float,
        default=defaults.region_boundary_hard_fraction,
    )
    parser.add_argument("--region-area-weight", type=float, default=defaults.region_area_weight)
    parser.add_argument("--region-soft-area-limit", type=float, default=defaults.region_soft_area_limit)
    parser.add_argument(
        "--region-area-hard-fraction",
        type=float,
        default=defaults.region_area_hard_fraction,
    )
    parser.add_argument(
        "--region-area-temperature",
        type=float,
        default=defaults.region_area_temperature,
    )
    parser.add_argument("--region-edge-weight", type=float, default=defaults.region_edge_weight)
    parser.add_argument("--decision-threshold", type=float, default=defaults.decision_threshold)
    parser.add_argument("--heatmap-threshold", type=float, default=defaults.heatmap_threshold)
    parser.add_argument(
        "--presence-calibration-target-score",
        type=float,
        default=defaults.presence_calibration_target_score,
    )
    parser.add_argument("--counterfactual-weight", type=float, default=defaults.counterfactual_weight)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--benchmark-warmup", type=int, default=defaults.benchmark_warmup)
    parser.add_argument("--benchmark-iterations", type=int, default=defaults.benchmark_iterations)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = FramePresenceTrainSettings(
        train_cache=args.train_cache,
        val_cache=args.val_cache,
        output_dir=args.output_dir,
        resume=args.resume,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        width=args.width,
        kernel_size=args.kernel_size,
        presence_margin_weight=args.presence_margin_weight,
        presence_hard_fraction=args.presence_hard_fraction,
        presence_positive_margin=args.positive_margin,
        presence_negative_margin=args.negative_margin,
        region_loss_weight=args.region_loss_weight,
        region_bce_weight=args.region_bce_weight,
        region_positive_margin=args.region_positive_margin,
        region_negative_margin=args.region_negative_margin,
        region_dice_weight=args.region_dice_weight,
        region_projection_weight=args.region_projection_weight,
        region_boundary_weight=args.region_boundary_weight,
        region_boundary_margin=args.region_boundary_margin,
        region_boundary_hard_fraction=args.region_boundary_hard_fraction,
        region_area_weight=args.region_area_weight,
        region_soft_area_limit=args.region_soft_area_limit,
        region_area_hard_fraction=args.region_area_hard_fraction,
        region_area_temperature=args.region_area_temperature,
        region_edge_weight=args.region_edge_weight,
        decision_threshold=args.decision_threshold,
        heatmap_threshold=args.heatmap_threshold,
        presence_calibration_target_score=args.presence_calibration_target_score,
        counterfactual_weight=args.counterfactual_weight,
        augment=not args.no_augment,
        seed=args.seed,
        device=args.device,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_iterations=args.benchmark_iterations,
    )
    summary = run_training(settings)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
