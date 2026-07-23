from __future__ import annotations

import torch


def presence_metrics(
    logits: torch.Tensor,
    presence: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    scores = torch.sigmoid(logits.detach().cpu())
    targets = presence.detach().cpu() > 0.5
    predicted = scores >= threshold
    true_positive = int((predicted & targets).sum())
    false_positive = int((predicted & ~targets).sum())
    false_negative = int((~predicted & targets).sum())
    true_negative = int((~predicted & ~targets).sum())
    accuracy = (true_positive + true_negative) / targets.numel() if targets.numel() else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive_scores = scores[targets]
    negative_scores = scores[~targets]
    min_positive = float(positive_scores.min()) if positive_scores.numel() else 0.0
    max_negative = float(negative_scores.max()) if negative_scores.numel() else 0.0
    return {
        "presence_precision": precision,
        "presence_accuracy": accuracy,
        "presence_recall": recall,
        "presence_f1": f1,
        "presence_tp": float(true_positive),
        "presence_fp": float(false_positive),
        "presence_fn": float(false_negative),
        "presence_tn": float(true_negative),
        "presence_min_positive_score": min_positive,
        "presence_max_negative_score": max_negative,
        "presence_gap": min_positive - max_negative,
        "presence_decision_threshold": threshold,
    }


def acceptance(metrics: dict[str, float], *, complete_validation: bool) -> dict[str, bool]:
    return {
        "validation_complete": complete_validation,
        "recall": metrics["presence_recall"] == 1.0,
        "f1": metrics["presence_f1"] == 1.0,
        "no_false_positive": metrics["presence_fp"] == 0.0,
        "no_false_negative": metrics["presence_fn"] == 0.0,
        "gap": metrics["presence_gap"] >= 0.8,
    }


def checkpoint_rank(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        float(metrics["presence_f1"] == 1.0 and metrics["presence_recall"] == 1.0),
        float(metrics["presence_gap"] >= 0.8),
        round(metrics["presence_f1"], 8),
        round(metrics["presence_recall"], 8),
        round(metrics["presence_gap"], 8),
        -metrics["presence_fp"],
        -metrics["presence_fn"],
    )


__all__ = ["acceptance", "checkpoint_rank", "presence_metrics"]
