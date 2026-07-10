from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RoiPresenceTrainSettings(BaseModel):
    train_roots: list[Path] = Field(default_factory=lambda: [Path("data/roi_samples6")])
    val_root: Path = Path("data/roi_samples6")
    output_dir: Path = Path("outputs/roi_presence_run")
    resume: Path | None = None
    resize_roi: tuple[int, int] | None = None
    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    num_workers: int = Field(default=0, ge=0)
    max_train_samples: int | None = Field(default=None, gt=0)
    max_val_samples: int | None = Field(default=None, gt=0)
    train_negative_ratio: float | None = Field(default=0.35, ge=0.0, le=1.0)
    val_negative_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    short_positive_loss_weight: float = Field(default=1.0, gt=0.0)
    short_positive_mask_loss_weight: float = Field(default=0.0, ge=0.0)
    presence_topk_ratio: float = Field(default=0.05, gt=0.0, le=1.0)
    width: int = Field(default=32, gt=0)
    log_interval: int = Field(default=10, gt=0)
    seed: int = 2026
    device: str = "auto"
