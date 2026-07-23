from __future__ import annotations

import json
import random
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
    sample_types: dict[str, int]

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
    sample_types: list[str]


@dataclass(frozen=True)
class _DatasetItem:
    sample: FramePresenceSample
    sample_type: str
    crop_view: int = 0


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


def is_roi_root(root: Path) -> bool:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    roi_size = summary.get("roi_size") if isinstance(summary, dict) else None
    return isinstance(roi_size, list) and len(roi_size) == 2


def _crop_boxes(boxes: tuple[Box, ...], crop: tuple[int, int, int, int]) -> tuple[Box, ...]:
    left, top, right, bottom = crop
    cropped: list[Box] = []
    for box in boxes:
        x1 = max(box.x1, left)
        y1 = max(box.y1, top)
        x2 = min(box.x2, right)
        y2 = min(box.y2, bottom)
        if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
            cropped.append(Box(x1 - left, y1 - top, x2 - left, y2 - top))
    return tuple(cropped)


def _random_crop_box(
    sample: FramePresenceSample,
    *,
    rng: random.Random,
    min_scale: float,
    max_scale: float,
) -> tuple[int, int, int, int]:
    source_width, source_height = sample.source_size
    scale = rng.uniform(min_scale, max_scale)
    crop_width = max(1, min(source_width, round(source_width * scale)))
    crop_height = max(1, min(source_height, round(source_height * scale)))

    if sample.boxes:
        target = rng.choice(sample.boxes)
        target_width = int(np.ceil(target.x2)) - int(np.floor(target.x1))
        target_height = int(np.ceil(target.y2)) - int(np.floor(target.y1))
        crop_width = min(source_width, max(crop_width, target_width))
        crop_height = min(source_height, max(crop_height, target_height))
        min_left = max(0, int(np.ceil(target.x2)) - crop_width)
        max_left = min(source_width - crop_width, int(np.floor(target.x1)))
        min_top = max(0, int(np.ceil(target.y2)) - crop_height)
        max_top = min(source_height - crop_height, int(np.floor(target.y1)))
        left = rng.randint(min_left, max(min_left, max_left))
        top = rng.randint(min_top, max(min_top, max_top))
    else:
        left = rng.randint(0, source_width - crop_width)
        top = rng.randint(0, source_height - crop_height)
    return left, top, left + crop_width, top + crop_height


class FramePresenceDataset(Dataset):
    """Mixed full-frame, ROI, and reproducible random-crop presence samples."""

    def __init__(
        self,
        roots: list[Path],
        *,
        image_size: tuple[int, int],
        random_crop_views: int = 0,
        random_crop_scale: tuple[float, float] = (0.3, 0.9),
        seed: int = 0,
        max_samples: int | None = None,
    ) -> None:
        self.image_size = image_size
        self.random_crop_scale = random_crop_scale
        self.seed = seed
        self.epoch = 0
        full_roots = [root for root in roots if not is_roi_root(root)]
        roi_roots = [root for root in roots if is_roi_root(root)]
        full_samples, full_dropped = discover_frame_presence_samples(full_roots)
        roi_samples, roi_dropped = discover_frame_presence_samples(roi_roots)
        self.dropped = full_dropped + roi_dropped
        items: list[_DatasetItem] = []
        for index in range(max(len(full_samples), len(roi_samples))):
            if index < len(full_samples):
                items.append(_DatasetItem(full_samples[index], "full_frame"))
            if index < len(roi_samples):
                items.append(_DatasetItem(roi_samples[index], "roi"))
            if index < len(full_samples):
                items.extend(
                    _DatasetItem(full_samples[index], "random_crop", crop_view)
                    for crop_view in range(random_crop_views)
                )
        self.items = items if max_samples is None else items[:max_samples]
        self.samples = [item.sample for item in self.items]
        self.summary = self._summarize()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        sample = item.sample
        crop = (0, 0, sample.source_size[0], sample.source_size[1])
        boxes = sample.boxes
        ignored_boxes = sample.ignored_boxes
        source_size = sample.source_size
        sample_id = sample.sample_id
        if item.sample_type == "random_crop":
            rng = random.Random(f"{self.seed}:{self.epoch}:{index}:{item.crop_view}")
            crop = _random_crop_box(
                sample,
                rng=rng,
                min_scale=self.random_crop_scale[0],
                max_scale=self.random_crop_scale[1],
            )
            boxes = _crop_boxes(sample.boxes, crop)
            ignored_boxes = _crop_boxes(sample.ignored_boxes, crop)
            source_size = (crop[2] - crop[0], crop[3] - crop[1])
            sample_id = f"{sample.sample_id}#crop{item.crop_view}"
        with Image.open(sample.image_path) as image:
            if image.mode != "RGB":
                raise ValueError(f"frame image must already be RGB: {sample.image_path}")
            if item.sample_type == "random_crop":
                image = image.crop(crop)
            if image.size != self.image_size:
                image = image.resize(self.image_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32)
        subtitle_mask = _rasterize_boxes(
            boxes,
            source_size=source_size,
            output_size=self.image_size,
        )
        ignored_mask = _rasterize_boxes(
            ignored_boxes,
            source_size=source_size,
            output_size=self.image_size,
        )
        return {
            "image": torch.from_numpy(array).permute(2, 0, 1),
            "subtitle_mask": subtitle_mask,
            "supervision_mask": 1.0 - ignored_mask,
            "presence": torch.tensor(float(bool(boxes)), dtype=torch.float32),
            "sample_id": sample_id,
            "root": str(sample.root),
            "image_path": str(sample.image_path),
            "sample_type": item.sample_type,
        }

    def manifest(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for item in self.items:
            sample = item.sample
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
                    "sample_type": item.sample_type,
                    "crop_view": item.crop_view if item.sample_type == "random_crop" else None,
                    "boxes": [[box.x1, box.y1, box.x2, box.y2] for box in sample.boxes],
                    "ignored_boxes": [
                        [box.x1, box.y1, box.x2, box.y2] for box in sample.ignored_boxes
                    ],
                }
            )
        return records

    def _summarize(self) -> FramePresenceSummary:
        roots: dict[str, int] = {}
        sample_types: dict[str, int] = {}
        positive = 0
        for item in self.items:
            sample = item.sample
            roots[str(sample.root)] = roots.get(str(sample.root), 0) + 1
            sample_types[item.sample_type] = sample_types.get(item.sample_type, 0) + 1
            positive += int(sample.has_subtitle)
        total = len(self.samples)
        return FramePresenceSummary(
            total=total,
            positive=positive,
            empty=total - positive,
            dropped=self.dropped,
            roots=roots,
            sample_types=sample_types,
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
        sample_types=[str(item["sample_type"]) for item in items],
    )


__all__ = [
    "FramePresenceBatch",
    "FramePresenceDataset",
    "FramePresenceSample",
    "FramePresenceSummary",
    "collate_frame_presence_batch",
    "discover_frame_presence_samples",
    "is_roi_root",
]
