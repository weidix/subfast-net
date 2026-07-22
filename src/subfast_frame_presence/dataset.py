from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from subfast_detector.dataset import apply_label_masks, load_label_masks, read_boxes
from subfast_shared.geometry import Box


@dataclass(frozen=True)
class FramePresenceSample:
    image_path: Path
    label_path: Path
    sample_id: str
    root: Path
    source_size: tuple[int, int]
    boxes: tuple[Box, ...]
    ignored_boxes: tuple[Box, ...]

    @property
    def has_subtitle(self) -> bool:
        return bool(self.boxes)


@dataclass(frozen=True)
class FramePresenceSummary:
    total: int
    positive: int
    empty: int
    dropped: int
    roots: dict[str, int]

    @property
    def positive_ratio(self) -> float:
        return self.positive / self.total if self.total else 0.0

    @property
    def empty_ratio(self) -> float:
        return self.empty / self.total if self.total else 0.0


@dataclass(frozen=True)
class FramePresenceBatch:
    images: torch.Tensor
    subtitle_masks: torch.Tensor
    supervision_masks: torch.Tensor
    presence: torch.Tensor
    sample_ids: list[str]
    roots: list[str]
    image_paths: list[str]


def _rasterize_boxes(
    boxes: tuple[Box, ...],
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> torch.Tensor:
    output_width, output_height = output_size
    source_width, source_height = source_size
    mask = torch.zeros((1, output_height, output_width), dtype=torch.float32)
    for box in boxes:
        left = max(0, min(output_width, int(np.floor(box.x1 * output_width / source_width))))
        top = max(0, min(output_height, int(np.floor(box.y1 * output_height / source_height))))
        right = max(left + 1, min(output_width, int(np.ceil(box.x2 * output_width / source_width))))
        bottom = max(top + 1, min(output_height, int(np.ceil(box.y2 * output_height / source_height))))
        if right > left and bottom > top:
            mask[:, top:bottom, left:right] = 1.0
    return mask


def discover_frame_presence_samples(roots: list[Path]) -> tuple[list[FramePresenceSample], int]:
    samples: list[FramePresenceSample] = []
    dropped = 0
    for root in roots:
        image_dir = root / "images"
        label_dir = root / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"frame root must contain images/ and labels/: {root}")
        masks = load_label_masks(root)
        for image_path in sorted(image_dir.glob("*.jpg")):
            sample_id = image_path.stem
            label_path = label_dir / f"{sample_id}.txt"
            with Image.open(image_path) as image:
                source_size = image.size
            boxes = read_boxes(label_path, *source_size)
            boxes, ignored_boxes, drop = apply_label_masks(
                sample_id,
                boxes,
                masks,
                *source_size,
            )
            if drop:
                dropped += 1
                continue
            samples.append(
                FramePresenceSample(
                    image_path=image_path,
                    label_path=label_path,
                    sample_id=sample_id,
                    root=root,
                    source_size=source_size,
                    boxes=tuple(boxes),
                    ignored_boxes=tuple(ignored_boxes),
                )
            )
    return samples, dropped


def _limit_samples(
    samples: list[FramePresenceSample],
    max_samples: int | None,
) -> list[FramePresenceSample]:
    return samples if max_samples is None else samples[:max_samples]


class FramePresenceDataset(Dataset):
    """One complete RGB frame per sample; image resizing is the only input transform."""

    def __init__(
        self,
        roots: list[Path],
        *,
        image_size: tuple[int, int],
        max_samples: int | None = None,
    ) -> None:
        self.image_size = image_size
        all_samples, self.dropped = discover_frame_presence_samples(roots)
        self.samples = _limit_samples(all_samples, max_samples)
        self.summary = self._summarize()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            if image.mode != "RGB":
                raise ValueError(f"frame image must already be RGB: {sample.image_path}")
            if image.size != self.image_size:
                image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)
        subtitle_mask = _rasterize_boxes(
            sample.boxes,
            source_size=sample.source_size,
            output_size=self.image_size,
        )
        ignored_mask = _rasterize_boxes(
            sample.ignored_boxes,
            source_size=sample.source_size,
            output_size=self.image_size,
        )
        return {
            "image": torch.from_numpy(array).permute(2, 0, 1),
            "subtitle_mask": subtitle_mask,
            "supervision_mask": 1.0 - ignored_mask,
            "presence": torch.tensor(float(sample.has_subtitle), dtype=torch.float32),
            "sample_id": sample.sample_id,
            "root": str(sample.root),
            "image_path": str(sample.image_path),
        }

    def manifest(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for sample in self.samples:
            stat = sample.image_path.stat()
            records.append(
                {
                    "root": str(sample.root),
                    "sample_id": sample.sample_id,
                    "image_path": str(sample.image_path),
                    "label_path": str(sample.label_path),
                    "source_size": list(sample.source_size),
                    "image_bytes": stat.st_size,
                    "image_mtime_ns": stat.st_mtime_ns,
                    "has_subtitle": sample.has_subtitle,
                    "boxes": [[box.x1, box.y1, box.x2, box.y2] for box in sample.boxes],
                    "ignored_boxes": [
                        [box.x1, box.y1, box.x2, box.y2] for box in sample.ignored_boxes
                    ],
                }
            )
        return records

    def _summarize(self) -> FramePresenceSummary:
        roots: dict[str, int] = {}
        positive = 0
        for sample in self.samples:
            roots[str(sample.root)] = roots.get(str(sample.root), 0) + 1
            positive += int(sample.has_subtitle)
        total = len(self.samples)
        return FramePresenceSummary(
            total=total,
            positive=positive,
            empty=total - positive,
            dropped=self.dropped,
            roots=roots,
        )


def collate_frame_presence_batch(items: list[dict[str, object]]) -> FramePresenceBatch:
    return FramePresenceBatch(
        images=torch.stack([item["image"] for item in items]),  # type: ignore[arg-type]
        subtitle_masks=torch.stack([item["subtitle_mask"] for item in items]),  # type: ignore[arg-type]
        supervision_masks=torch.stack([item["supervision_mask"] for item in items]),  # type: ignore[arg-type]
        presence=torch.stack([item["presence"] for item in items]),  # type: ignore[arg-type]
        sample_ids=[str(item["sample_id"]) for item in items],
        roots=[str(item["root"]) for item in items],
        image_paths=[str(item["image_path"]) for item in items],
    )


__all__ = [
    "FramePresenceBatch",
    "FramePresenceDataset",
    "FramePresenceSample",
    "FramePresenceSummary",
    "collate_frame_presence_batch",
    "discover_frame_presence_samples",
]
