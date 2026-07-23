from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FramePresenceTrainSettings(BaseModel):
    model_name: Literal["Frame Presence V5"] = "Frame Presence V5"
    architecture_version: Literal[5] = 5
    train_roots: list[Path] = Field(
        default_factory=lambda: [
            Path("data/generated_samples1"),
            Path("data/generated_samples2"),
            Path("data/generated_samples3"),
            Path("data/generated_samples4"),
            Path("data/generated_samples5"),
            Path("data/generated_samples6"),
            Path("data/roi_samples1"),
            Path("data/roi_samples2"),
            Path("data/roi_samples3"),
            Path("data/roi_samples4"),
            Path("data/roi_samples5"),
            Path("data/roi_samples6"),
        ]
    )
    val_roots: list[Path] = Field(
        default_factory=lambda: [
            Path("data/validation_samples"),
            Path("data/roi_validation_samples"),
        ]
    )
    output_dir: Path = Path("outputs/frame_presence_v5")
    resume: Path | None = None
    early_stop: bool = True
    resize_scale: float = Field(default=0.25, gt=0.0, le=1.0)
    resize_alignment: Literal[16] = 16
    resize_alignment_mode: Literal["nearest_multiple_half_up"] = "nearest_multiple_half_up"
    resize_interpolation: Literal["bilinear"] = "bilinear"
    min_subtitle_short_edge: float = Field(default=8.0, gt=0.0)
    reference_source_size: tuple[int, int] = (1920, 1080)
    batch_size: int = Field(default=24, gt=0)
    epochs: int = Field(default=10, gt=0, le=10)
    learning_rate: float = Field(default=1.5e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    num_workers: int = Field(default=0, ge=0)
    max_train_samples: int | None = Field(default=None, gt=0)
    max_val_samples: int | None = Field(default=None, gt=0)
    random_crop_views: int = Field(default=1, ge=0, le=8)
    random_crop_min_scale: float = Field(default=0.3, gt=0.0, le=1.0)
    random_crop_max_scale: float = Field(default=0.9, gt=0.0, le=1.0)
    width: int = Field(default=24, gt=0)
    normalization: Literal["none", "group_norm"] = "none"
    gradient_clip_norm: float = Field(default=5.0, gt=0.0)
    region_loss_weight: float = Field(default=1.0, ge=0.0)
    region_dice_weight: float = Field(default=0.5, ge=0.0)
    margin_loss_weight: float = Field(default=0.5, ge=0.0)
    positive_logit_margin: float = Field(default=4.0, gt=0.0)
    negative_logit_margin: float = Field(default=-4.0, lt=0.0)
    log_interval: int = Field(default=50, gt=0)
    seed: int = 2026
    device: str = "auto"

    @model_validator(mode="after")
    def validate_geometry(self) -> "FramePresenceTrainSettings":
        width, height = self.reference_source_size
        if width <= 0 or height <= 0:
            raise ValueError("reference_source_size dimensions must be positive")
        if self.random_crop_min_scale > self.random_crop_max_scale:
            raise ValueError("random_crop_min_scale must not exceed random_crop_max_scale")
        return self
