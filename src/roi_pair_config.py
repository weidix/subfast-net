from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RoiPairTrainSettings(BaseModel):
    train_roots: list[Path] = Field(default_factory=lambda: [Path("data/roi_samples6")])
    val_root: Path = Path("data/roi_validation_samples")
    output_dir: Path = Path("outputs/roi_pair_matcher")
    resume: Path | None = None
    resize_roi: tuple[int, int] = (256, 32)
    batch_size: int = Field(default=128, ge=2)
    validation_batch_size: int = Field(default=256, gt=0)
    epochs: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    minimum_learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    num_workers: int = Field(default=0, ge=0)
    negative_ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    ocr_negative_enabled: bool = True
    ocr_negative_max_similarity: float = Field(default=0.2, ge=0.0, le=1.0)
    ocr_negative_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    mask_loss_weight: float = Field(default=0.2, ge=0.0)
    tail_gap_loss_weight: float = Field(default=0.25, ge=0.0)
    tail_gap_margin: float = Field(default=1.0, ge=0.0)
    photometric_jitter: float = Field(default=0.05, ge=0.0, le=0.25)
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    log_interval: int = Field(default=50, gt=0)
    seed: int = 2026
    device: str = "auto"
