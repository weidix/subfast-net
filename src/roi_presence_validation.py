from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .roi_presence_loss import (
    composite_valid_mask,
    counterfactual_presence_loss,
    erase_subtitle_regions,
    positive_with_region_mask,
    resize_candidate_masks,
    short_positive_mask,
    subtitle_region_loss,
    subtitle_region_targets,
    transplant_subtitle_regions,
)
from .roi_presence_metrics import (
    presence_metrics,
    region_localization_metrics,
    scoped_presence_metrics,
    segment_presence_metrics,
    text_distractor_metrics,
)
from .roi_presence_model import RoiPresenceModel


def load_previous_scores(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        scores[str(record["sample_key"])] = float(record["score"])
    return scores


def _tail_mean(values: list[float], fraction: float, *, largest: bool) -> float:
    if not values:
        return 0.0
    count = max(1, math.ceil(len(values) * fraction))
    ordered = sorted(values, reverse=largest)
    return sum(ordered[:count]) / count


@torch.no_grad()
def validate_presence(
    model: RoiPresenceModel,
    loader: DataLoader,
    device: torch.device,
    *,
    decision_threshold: float,
    region_loss_weight: float,
    region_dice_weight: float,
    region_projection_weight: float,
    text_distractor_weight: float,
    counterfactual_loss_weight: float,
    counterfactual_margin: float,
    diagnostics_path: Path | None = None,
    previous_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    model.eval()
    presence_loss_sum = 0.0
    counterfactual_loss_sum = 0.0
    sample_count = 0
    counterfactual_count = 0
    logits_all: list[torch.Tensor] = []
    presence_all: list[torch.Tensor] = []
    region_logits_all: list[torch.Tensor] = []
    region_targets_all: list[torch.Tensor] = []
    candidate_masks_all: list[torch.Tensor] = []
    valid_masks_all: list[torch.Tensor] = []
    erased_logits_by_index: dict[int, float] = {}
    transplanted_logits_by_index: dict[int, float] = {}
    seam_control_logits_by_index: dict[int, float] = {}
    sample_ids: list[str] = []
    segment_ids: list[str] = []
    roots: list[str] = []
    image_paths: list[str] = []
    ocr_texts: list[str] = []
    batch_context_delta = 0.0

    for batch_index, batch in enumerate(loader):
        images = batch.images.to(device)
        presence = batch.presence.to(device)
        valid_masks = batch.valid_masks.to(device)
        donor_available = batch.donor_available.to(device)
        if batch.subtitle_masks is None:
            raise RuntimeError("ROI Presence validation requires subtitle masks")
        subtitle_masks = batch.subtitle_masks.to(device)
        logits, region_logits = model.forward_with_presence_map(images, valid_masks)
        batch_size = images.shape[0]
        presence_loss_sum += float(
            F.binary_cross_entropy_with_logits(logits, presence, reduction="sum").detach().cpu()
        )
        valid_counterfactual = positive_with_region_mask(subtitle_masks, presence) & donor_available
        valid_count = int(valid_counterfactual.sum())
        if valid_count:
            selected_images = images[valid_counterfactual]
            selected_masks = subtitle_masks[valid_counterfactual]
            selected_valid_masks = valid_masks[valid_counterfactual]
            donors = batch.donor_images[valid_counterfactual.cpu()].to(device)
            donor_valid_masks = batch.donor_valid_masks[valid_counterfactual.cpu()].to(device)
            seam_donors = batch.seam_donor_images[valid_counterfactual.cpu()].to(device)
            seam_donor_valid_masks = batch.seam_donor_valid_masks[
                valid_counterfactual.cpu()
            ].to(device)
            erased_images = erase_subtitle_regions(
                selected_images,
                selected_masks,
                donor_images=donors,
            )
            transplanted_images = transplant_subtitle_regions(selected_images, selected_masks, donors)
            seam_control_images = erase_subtitle_regions(
                donors,
                selected_masks,
                donor_images=seam_donors,
            )
            variant_images = [erased_images, transplanted_images, seam_control_images]
            variant_valid_masks = [
                composite_valid_mask(selected_masks, donor_valid_masks, selected_valid_masks),
                composite_valid_mask(selected_masks, selected_valid_masks, donor_valid_masks),
                composite_valid_mask(selected_masks, seam_donor_valid_masks, donor_valid_masks),
            ]
            variant_logits = model(
                torch.cat(variant_images),
                torch.cat(variant_valid_masks),
            )
            erased_logits = variant_logits[:valid_count]
            transplanted_logits = variant_logits[valid_count : 2 * valid_count]
            seam_control_logits = variant_logits[2 * valid_count :]
            counterfactual = counterfactual_presence_loss(
                logits[valid_counterfactual],
                erased_logits,
                transplanted_logits,
                seam_control_logits,
                margin=counterfactual_margin,
            )
            counterfactual_loss_sum += float(counterfactual.total.detach().cpu()) * valid_count
            global_indices = (valid_counterfactual.nonzero(as_tuple=False).flatten() + sample_count).tolist()
            for offset, global_index in enumerate(global_indices):
                erased_logits_by_index[int(global_index)] = float(erased_logits[offset].detach().cpu())
                transplanted_logits_by_index[int(global_index)] = float(
                    transplanted_logits[offset].detach().cpu()
                )
                seam_control_logits_by_index[int(global_index)] = float(
                    seam_control_logits[offset].detach().cpu()
                )
            counterfactual_count += valid_count

        if batch_index == 0:
            individual_logits = torch.cat(
                [
                    model(
                        images[index : index + 1],
                        valid_masks[index : index + 1],
                    )
                    for index in range(min(batch_size, 8))
                ]
            )
            batch_context_delta = float(
                (individual_logits - logits[: individual_logits.shape[0]]).abs().max().detach().cpu()
            )

        targets = subtitle_region_targets(
            subtitle_masks,
            presence,
            region_logits.shape[-2:],
            valid_masks,
        )
        valid_region = (
            F.interpolate(valid_masks, size=region_logits.shape[-2:], mode="area") > 0.5
        ).to(region_logits.dtype)
        candidates = resize_candidate_masks(subtitle_masks, region_logits.shape[-2:]) * valid_region
        logits_all.append(logits.cpu())
        presence_all.append(presence.cpu())
        region_logits_all.append(region_logits.cpu())
        region_targets_all.append(targets.cpu())
        candidate_masks_all.append(candidates.cpu())
        valid_masks_all.append(valid_region.cpu())
        sample_ids.extend(batch.sample_ids)
        segment_ids.extend(batch.segment_ids)
        roots.extend(batch.roots)
        image_paths.extend(batch.image_paths)
        ocr_texts.extend(batch.ocr_texts)
        sample_count += batch_size

    logits = torch.cat(logits_all)
    presence = torch.cat(presence_all)
    region_logits = torch.cat(region_logits_all)
    region_targets = torch.cat(region_targets_all)
    candidate_masks = torch.cat(candidate_masks_all)
    valid_masks = torch.cat(valid_masks_all)
    validation_region = subtitle_region_loss(
        region_logits,
        candidate_masks,
        presence,
        valid_masks,
        dice_weight=region_dice_weight,
        projection_weight=region_projection_weight,
        text_distractor_weight=text_distractor_weight,
    )
    metrics = {
        "val_presence_loss": presence_loss_sum / max(1, sample_count),
        "val_region_loss": float(validation_region.total),
        "val_counterfactual_loss": counterfactual_loss_sum / max(1, counterfactual_count),
        "presence_batch_context_max_abs_logit_delta": batch_context_delta,
    }
    metrics["val_loss"] = (
        metrics["val_presence_loss"]
        + region_loss_weight * metrics["val_region_loss"]
        + counterfactual_loss_weight * metrics["val_counterfactual_loss"]
    )
    metrics.update(presence_metrics(logits, presence, threshold=decision_threshold))
    metrics.update(
        scoped_presence_metrics(
            logits,
            presence,
            ocr_texts,
            threshold=decision_threshold,
        )
    )
    metrics.update(
        segment_presence_metrics(
            logits,
            presence,
            segment_ids,
            roots,
            threshold=decision_threshold,
        )
    )
    region_metrics, region_records = region_localization_metrics(
        region_logits,
        region_targets,
        valid_masks,
    )
    metrics.update(region_metrics)
    region_probability = torch.sigmoid(region_logits)
    valid_region_bool = valid_masks > 0.5
    region_max_scores: list[float] = []
    region_activation_areas: list[float] = []
    for index in range(region_probability.shape[0]):
        valid_probability = region_probability[index][valid_region_bool[index]]
        region_max_scores.append(float(valid_probability.max()) if valid_probability.numel() else 0.0)
        region_activation_areas.append(
            float((valid_probability >= 0.5).to(torch.float32).mean())
            if valid_probability.numel()
            else 0.0
        )
        region_records[index]["region_max_score"] = region_max_scores[-1]
        region_records[index]["region_activation_area"] = region_activation_areas[-1]
    negative_indices = (presence <= 0.5).nonzero(as_tuple=False).flatten().tolist()
    negative_max_scores = [region_max_scores[index] for index in negative_indices]
    negative_activation_areas = [region_activation_areas[index] for index in negative_indices]
    metrics["negative_region_max_score_p95"] = (
        sorted(negative_max_scores)[round((len(negative_max_scores) - 1) * 0.95)]
        if negative_max_scores
        else 0.0
    )
    metrics["negative_region_activation_area"] = (
        sum(negative_activation_areas) / len(negative_activation_areas)
        if negative_activation_areas
        else 0.0
    )
    distractor_metrics, distractor_mask = text_distractor_metrics(
        logits,
        presence,
        candidate_masks,
        threshold=decision_threshold,
    )
    metrics.update(distractor_metrics)

    scores = torch.sigmoid(logits)
    erased_drops: list[float] = []
    erased_scores: list[float] = []
    transplanted_scores: list[float] = []
    transplanted_deltas: list[float] = []
    seam_control_scores: list[float] = []
    for index, erased_logit in erased_logits_by_index.items():
        erased_score = float(torch.sigmoid(torch.tensor(erased_logit)))
        erased_scores.append(erased_score)
        erased_drops.append(float(scores[index]) - erased_score)
        if index in transplanted_logits_by_index:
            transplanted_score = float(torch.sigmoid(torch.tensor(transplanted_logits_by_index[index])))
            transplanted_scores.append(transplanted_score)
            transplanted_deltas.append(abs(float(scores[index]) - transplanted_score))
        if index in seam_control_logits_by_index:
            seam_control_scores.append(
                float(torch.sigmoid(torch.tensor(seam_control_logits_by_index[index])))
            )
    metrics.update(
        {
            "counterfactual_evaluable_count": float(len(erased_scores)),
            "counterfactual_erased_score": sum(erased_scores) / len(erased_scores) if erased_scores else 0.0,
            "counterfactual_erased_flip_rate": (
                sum(score < decision_threshold for score in erased_scores) / len(erased_scores)
                if erased_scores
                else 0.0
            ),
            "counterfactual_score_drop": sum(erased_drops) / len(erased_drops) if erased_drops else 0.0,
            "counterfactual_score_drop_lower_tail_1pct": _tail_mean(
                erased_drops,
                0.01,
                largest=False,
            ),
            "counterfactual_transplanted_recall": (
                sum(score >= decision_threshold for score in transplanted_scores) / len(transplanted_scores)
                if transplanted_scores
                else 0.0
            ),
            "counterfactual_transplanted_abs_score_delta": (
                sum(transplanted_deltas) / len(transplanted_deltas) if transplanted_deltas else 0.0
            ),
            "counterfactual_seam_control_fpr": (
                sum(score >= decision_threshold for score in seam_control_scores)
                / len(seam_control_scores)
                if seam_control_scores
                else 0.0
            ),
        }
    )

    previous_scores = previous_scores or {}
    short_mask = short_positive_mask(presence, ocr_texts).cpu()
    drift_values: list[float] = []
    short_drift_values: list[float] = []
    threshold_flips = 0
    short_threshold_flips = 0
    records: list[dict[str, object]] = []
    for index, sample_id in enumerate(sample_ids):
        sample_key = f"{roots[index]}::{sample_id}"
        score = float(scores[index])
        previous_score = previous_scores.get(sample_key)
        drift = abs(score - previous_score) if previous_score is not None else None
        if drift is not None:
            drift_values.append(drift)
            flipped = int((score >= decision_threshold) != (previous_score >= decision_threshold))
            threshold_flips += flipped
            if bool(short_mask[index]):
                short_drift_values.append(drift)
                short_threshold_flips += flipped
        erased_logit = erased_logits_by_index.get(index)
        transplanted_logit = transplanted_logits_by_index.get(index)
        seam_control_logit = seam_control_logits_by_index.get(index)
        record: dict[str, object] = {
            "sample_key": sample_key,
            "sample_id": sample_id,
            "segment_id": segment_ids[index],
            "root": roots[index],
            "image_path": image_paths[index],
            "ocr_text": ocr_texts[index],
            "target": int(presence[index] > 0.5),
            "sample_kind": (
                "subtitle"
                if presence[index] > 0.5
                else "text_distractor"
                if distractor_mask[index]
                else "empty"
            ),
            "short_positive": bool(short_mask[index]),
            "logit": float(logits[index]),
            "score": score,
            "predicted": int(score >= decision_threshold),
            "previous_score": previous_score,
            "score_drift": drift,
            **region_records[index],
        }
        if erased_logit is not None:
            erased_score = float(torch.sigmoid(torch.tensor(erased_logit)))
            record["erased_score"] = erased_score
            record["erased_score_drop"] = score - erased_score
        if transplanted_logit is not None:
            record["transplanted_score"] = float(torch.sigmoid(torch.tensor(transplanted_logit)))
        if seam_control_logit is not None:
            record["seam_control_score"] = float(torch.sigmoid(torch.tensor(seam_control_logit)))
        records.append(record)

    metrics.update(
        {
            "presence_score_drift_max": max(drift_values) if drift_values else 0.0,
            "presence_score_drift_upper_tail_mean_1pct": _tail_mean(
                drift_values,
                0.01,
                largest=True,
            ),
            "presence_threshold_flip_count": float(threshold_flips),
            "short_presence_score_drift_upper_tail_mean_1pct": _tail_mean(
                short_drift_values,
                0.01,
                largest=True,
            ),
            "short_presence_threshold_flip_count": float(short_threshold_flips),
        }
    )
    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )
    return metrics
