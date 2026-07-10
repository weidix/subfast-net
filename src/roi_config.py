from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class RoiTrainSettings(BaseModel):
    train_roots: list[Path] = Field(default_factory=lambda: [Path("data/roi_samples6")])
    val_root: Path = Path("data/roi_samples6")
    output_dir: Path = Path("outputs/roi_presence_embedding_run")
    resume: Path | None = None
    resize_roi: tuple[int, int] | None = None
    presence_batch_size: int = Field(default=16, gt=0)
    embedding_batch_size: int = Field(default=16, gt=0)
    joint_presence_batch_size: int | None = Field(default=None, gt=0)
    joint_embedding_batch_size: int | None = Field(default=None, gt=0)
    presence_epochs: int = Field(default=1, ge=0)
    embedding_epochs: int = Field(default=1, ge=0)
    joint_epochs: int = Field(default=1, ge=0)
    learning_rate: float = 3e-4
    joint_learning_rate: float = Field(default=3e-5, gt=0.0)
    weight_decay: float = 1e-4
    num_workers: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    presence_negative_ratio: float | None = Field(default=0.35, ge=0.0, le=1.0)
    val_negative_ratio: float | None = None
    short_positive_loss_weight: float = 1.0
    short_positive_mask_loss_weight: float = 0.0
    embedding_loss_weight: float = 1.0
    embedding_loss_alpha: float = 1.0
    embedding_negative_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    joint_presence_negative_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    joint_embedding_batch_negative_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    embedding_ocr_negative_enabled: bool = True
    embedding_ocr_negative_max_similarity: float = 0.2
    embedding_ocr_negative_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    embedding_positive_consistency_beta: float = 0.0
    embedding_positive_consistency_margin: float = 0.75
    embedding_temperature: float = 0.1
    embedding_supcon_weight: float = Field(default=0.5, ge=0.0)
    embedding_tail_gamma_positive: float = Field(default=20.0, gt=0.0)
    embedding_tail_gamma_negative: float = Field(default=40.0, gt=0.0)
    embedding_tail_hard_negative_weight: float = Field(default=2.0, gt=0.0)
    embedding_similarity_threshold: float = 0.5
    embedding_attention_mask_loss_weight: float = Field(default=1.0, ge=0.0)
    presence_topk_ratio: float = 0.05
    width: int = 32
    embedding_dim: int = Field(default=256, gt=0)
    embedding_width_tokens: int = Field(default=32, gt=0)
    embedding_aggregation: Literal["masked_global", "width_tokens"] = "width_tokens"
    log_interval: int = 10
    seed: int = 2026
    device: str = "auto"
