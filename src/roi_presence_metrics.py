from __future__ import annotations

import torch

from .roi_presence_loss import short_positive_mask


def presence_metrics(logits: torch.Tensor, presence: torch.Tensor) -> dict[str, float]:
    predicted = torch.sigmoid(logits) >= 0.5
    target = presence > 0.5
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    tn = int((~predicted & ~target).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "presence_accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "presence_tp": float(tp),
        "presence_fp": float(fp),
        "presence_fn": float(fn),
        "presence_tn": float(tn),
    }


def scoped_presence_metrics(
    logits: torch.Tensor,
    presence: torch.Tensor,
    ocr_texts: list[str],
) -> dict[str, float]:
    short_positive = short_positive_mask(presence, ocr_texts).cpu()
    positive = presence > 0.5
    negative = ~positive

    def metric_for(mask: torch.Tensor, prefix: str) -> dict[str, float]:
        if not bool(mask.any()):
            return {f"{prefix}_presence_f1": 0.0, f"{prefix}_presence_accuracy": 0.0}
        metrics = presence_metrics(logits[mask], presence[mask])
        return {
            f"{prefix}_presence_f1": metrics["presence_f1"],
            f"{prefix}_presence_accuracy": metrics["presence_accuracy"],
        }

    metrics = {"global_presence_f1": presence_metrics(logits, presence)["presence_f1"]}
    metrics.update(metric_for(negative | (positive & ~short_positive), "normal"))
    metrics.update(metric_for(negative | short_positive, "short"))
    return metrics


def checkpoint_score(metrics: dict[str, float]) -> float:
    return sum(
        float(metrics[name])
        for name in ("global_presence_f1", "normal_presence_f1", "short_presence_f1")
    ) / 3.0
