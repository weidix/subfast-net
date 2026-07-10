from __future__ import annotations

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .roi_presence_metrics import presence_metrics, scoped_presence_metrics
from .roi_presence_model import RoiPresenceModel


@torch.no_grad()
def validate_presence(
    model: RoiPresenceModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    logits_all: list[torch.Tensor] = []
    presence_all: list[torch.Tensor] = []
    ocr_texts: list[str] = []
    for batch in loader:
        logits = model(batch.images.to(device))
        presence = batch.presence.to(device)
        losses.append(float(F.binary_cross_entropy_with_logits(logits, presence).detach().cpu()))
        logits_all.append(logits.cpu())
        presence_all.append(presence.cpu())
        ocr_texts.extend(batch.ocr_texts)
    logits = torch.cat(logits_all)
    presence = torch.cat(presence_all)
    metrics = {
        "val_loss": sum(losses) / max(1, len(losses)),
        "val_presence_loss": sum(losses) / max(1, len(losses)),
    }
    metrics.update(presence_metrics(logits, presence))
    metrics.update(scoped_presence_metrics(logits, presence, ocr_texts))
    return metrics
