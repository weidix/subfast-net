from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RoiPresenceTrainSettings(BaseModel):
    train_roots: list[Path] = Field(default_factory=lambda: [Path("data/roi_samples6")])
    val_root: Path = Path("data/roi_validation_samples")
    output_dir: Path = Path("outputs/roi_presence_run")
    resume: Path | None = None
    resize_roi: tuple[int, int] | None = None
    resize_mode: str = "letterbox"
    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=1, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    num_workers: int = Field(default=0, ge=0)
    max_train_samples: int | None = Field(default=None, gt=0)
    max_val_samples: int | None = Field(default=None, gt=0)
    train_negative_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    val_negative_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    score_positive_prior: float | None = Field(default=None, gt=0.0, lt=1.0)
    region_loss_weight: float = Field(default=1.0, ge=0.0)
    region_dice_weight: float = Field(default=1.0, ge=0.0)
    region_projection_weight: float = Field(default=0.25, ge=0.0)
    text_distractor_weight: float = Field(default=4.0, ge=0.0)
    counterfactual_loss_weight: float = Field(default=0.5, ge=0.0)
    counterfactual_margin: float = Field(default=2.0, ge=0.0)
    evidence_kernel_size: int = Field(default=5, gt=1)
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    require_text_distractor_negatives: bool = False
    width: int = Field(default=16, gt=0)
    log_interval: int = Field(default=10, gt=0)
    seed: int = 2026
    device: str = "auto"
