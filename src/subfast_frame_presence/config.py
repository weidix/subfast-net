from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class FramePresenceTrainSettings(BaseModel):
    train_roots: list[Path] = Field(
        default_factory=lambda: [
            Path("data/generated_samples1"),
            Path("data/generated_samples2"),
            Path("data/generated_samples3"),
            Path("data/generated_samples4"),
            Path("data/generated_samples5"),
            Path("data/generated_samples6"),
        ]
    )
    val_root: Path = Path("data/validation_samples")
    output_dir: Path = Path("outputs/frame_presence_v3")
    resume: Path | None = None
    image_size: tuple[int, int] = (512, 288)
    batch_size: int = Field(default=24, gt=0)
    epochs: int = Field(default=10, gt=0, le=10)
    learning_rate: float = Field(default=1.5e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    num_workers: int = Field(default=0, ge=0)
    max_train_samples: int | None = Field(default=None, gt=0)
    max_val_samples: int | None = Field(default=None, gt=0)
    width: int = Field(default=24, gt=0)
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
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size dimensions must be positive")
        if width % 16 or height % 16:
            raise ValueError("image_size dimensions must be divisible by the encoder stride (16)")
        return self
