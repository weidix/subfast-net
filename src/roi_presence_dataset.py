from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .roi_dataset import (
    RoiDatasetSummary,
    RoiSample,
    discover_roi_samples,
    limit_roi_samples,
    subtitle_mask_from_labels,
)


@dataclass(frozen=True)
class RoiPresenceBatch:
    images: torch.Tensor
    subtitle_masks: torch.Tensor | None
    presence: torch.Tensor
    sample_ids: list[str]
    ocr_texts: list[str]


class RoiPresenceDataset(Dataset):
    def __init__(
        self,
        roots: list[Path],
        *,
        resize_roi: tuple[int, int] | None = None,
        max_samples: int | None = None,
        negative_ratio: float | None = None,
        load_subtitle_masks: bool = True,
    ) -> None:
        self.resize_roi = resize_roi
        self.load_subtitle_masks = load_subtitle_masks
        self.expected_roi_size: tuple[int, int] | None = None
        samples: list[RoiSample] = []
        for root in roots:
            root_samples, roi_size = discover_roi_samples(root)
            if self.expected_roi_size is None:
                self.expected_roi_size = roi_size
            elif resize_roi is None and roi_size != self.expected_roi_size:
                raise ValueError(
                    f"ROI size mismatch: {root} has {roi_size}, expected {self.expected_roi_size}; "
                    "pass --resize-roi WIDTHxHEIGHT for explicit resize"
                )
            samples.extend(root_samples)
        if self.expected_roi_size is None:
            raise ValueError("no ROI roots provided")
        self.output_roi_size = resize_roi or self.expected_roi_size
        self.samples = limit_roi_samples(samples, max_samples, negative_ratio)
        self.summary = self._summarize()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb = image.convert("RGB")
            actual_size = rgb.size
            expected = tuple(sample.annotation.get("roi_size") or self.expected_roi_size)
            if actual_size != expected:
                raise ValueError(
                    f"ROI image size mismatch for {sample.image_path}: actual={actual_size} expected={expected}"
                )
            if self.resize_roi is not None and actual_size != self.resize_roi:
                rgb = rgb.resize(self.resize_roi, Image.Resampling.BILINEAR)
            elif self.resize_roi is None and actual_size != self.output_roi_size:
                raise ValueError(
                    f"ROI input size mismatch for {sample.image_path}: {actual_size} != {self.output_roi_size}"
                )
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": image_tensor,
            "subtitle_mask": (
                subtitle_mask_from_labels(
                    sample.label_path,
                    output_size=self.output_roi_size,
                    label_size=tuple(expected),
                )
                if self.load_subtitle_masks
                else None
            ),
            "presence": torch.tensor(float(sample.has_subtitle), dtype=torch.float32),
            "sample_id": sample.sample_id,
            "ocr_text": sample.ocr_text,
        }

    def _summarize(self) -> RoiDatasetSummary:
        roots: dict[str, int] = {}
        positive = 0
        for sample in self.samples:
            roots[str(sample.root)] = roots.get(str(sample.root), 0) + 1
            positive += int(sample.has_subtitle)
        total = len(self.samples)
        return RoiDatasetSummary(
            total=total,
            positive=positive,
            empty=total - positive,
            roots=roots,
            roi_size=self.output_roi_size,
        )


def collate_presence_batch(items: list[dict]) -> RoiPresenceBatch:
    max_h = max(item["image"].shape[1] for item in items)
    max_w = max(item["image"].shape[2] for item in items)

    def pad(tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor, (0, max_w - tensor.shape[2], 0, max_h - tensor.shape[1]), value=0.0)

    subtitle_masks = (
        torch.stack([pad(item["subtitle_mask"]) for item in items])
        if items[0]["subtitle_mask"] is not None
        else None
    )
    return RoiPresenceBatch(
        images=torch.stack([pad(item["image"]) for item in items]),
        subtitle_masks=subtitle_masks,
        presence=torch.stack([item["presence"] for item in items]),
        sample_ids=[item["sample_id"] for item in items],
        ocr_texts=[item["ocr_text"] for item in items],
    )
