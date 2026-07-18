from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class TrainSettings(BaseModel):
    train_roots: list[Path] = Field(
        default_factory=lambda: [
            Path("data/generated_samples1"),
            Path("data/generated_samples2"),
            Path("data/generated_samples3"),
            Path("data/generated_samples4"),
        ]
    )
    val_root: Path = Path("data/validation_samples")
    output_dir: Path = Path("outputs/pytorch_run")
    resume: Path | None = None
    image_size: int = 256
    stride: int = 32
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    log_interval: int = 10
    save_epoch_outputs: bool = True
    max_epoch_output_samples: int | None = 32
    train_empty_sample_ratio: float | None = 0.35
    val_empty_sample_ratio: float | None = None
    pooling_size: int = 9
    kernel_scale: float = 0.1
    min_kernel_width: float = 3.0
    min_kernel_height: float = 3.0
    region_threshold: float = 0.5
    kernel_threshold: float = 0.5
    max_detection_width_ratio: float = 1.0
    iou_threshold: float = 0.5
    seed: int = 2026
    device: str = "auto"
