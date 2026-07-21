from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class FramePresencePreprocessSettings(BaseModel):
    input_width: int = Field(default=256, ge=32)
    input_height: int = Field(default=144, ge=32)
    focus_width: int = Field(default=256, ge=32)
    focus_height: int = Field(default=32, ge=16)
    heatmap_stride_x: int = Field(default=4, gt=0)
    heatmap_stride_y: int = Field(default=2, gt=0)


class FramePresenceTrainSettings(BaseModel):
    train_cache: Path = Path("data/frame_presence_train_cache")
    val_cache: Path = Path("data/frame_presence_validation_cache")
    output_dir: Path = Path("outputs/frame_presence")
    resume: Path | None = None
    batch_size: int = Field(default=256, gt=0)
    epochs: int = Field(default=20, gt=0)
    learning_rate: float = Field(default=2e-3, gt=0.0)
    min_learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    num_workers: int = Field(default=0, ge=0)
    width: int = Field(default=16, gt=0)
    kernel_size: int = Field(default=5, gt=1)
    presence_margin_weight: float = Field(default=1.0, ge=0.0)
    presence_hard_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    presence_positive_margin: float = 3.0
    presence_negative_margin: float = -3.0
    region_loss_weight: float = Field(default=1.0, ge=0.0)
    region_bce_weight: float = Field(default=1.0, ge=0.0)
    region_positive_margin: float = 2.0
    region_negative_margin: float = -1.0
    region_dice_weight: float = Field(default=1.0, ge=0.0)
    region_projection_weight: float = Field(default=0.5, ge=0.0)
    region_boundary_weight: float = Field(default=1.0, ge=0.0)
    region_boundary_margin: float = 2.0
    region_boundary_hard_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    region_area_weight: float = Field(default=0.0, ge=0.0)
    region_soft_area_limit: float = Field(default=2.0, gt=0.0)
    region_area_hard_fraction: float = Field(default=0.05, gt=0.0, le=1.0)
    region_area_temperature: float = Field(default=10.0, gt=0.0)
    region_edge_weight: float = Field(default=1.0, ge=0.0)
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    heatmap_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    presence_calibration_target_score: float = Field(default=0.95, gt=0.5, lt=1.0)
    counterfactual_weight: float = Field(default=1.0, ge=0.0)
    augment: bool = True
    seed: int = 2026
    device: str = "auto"
    benchmark_warmup: int = Field(default=50, ge=1)
    benchmark_iterations: int = Field(default=500, ge=1)
