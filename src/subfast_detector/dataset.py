from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from zlib import crc32

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from subfast_shared.geometry import Box, LetterboxShape, letterbox_shape, yolo_to_box
from subfast_shared.vision import IMAGENET_MEAN, IMAGENET_STD

from .targets import build_targets


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    sample_id: str
    root: Path


@dataclass(frozen=True)
class Batch:
    images: torch.Tensor
    regions: torch.Tensor
    kernels: torch.Tensor
    training_masks: torch.Tensor
    boxes: list[list[Box]]
    shapes: list[LetterboxShape]
    sample_ids: list[str]


@dataclass(frozen=True)
class DatasetSummary:
    total: int
    labeled: int
    empty: int
    roots: dict[str, int]

    @property
    def labeled_ratio(self) -> float:
        return self.labeled / self.total if self.total else 0.0

    @property
    def empty_ratio(self) -> float:
        return self.empty / self.total if self.total else 0.0


def discover_samples(roots: list[Path]) -> list[Sample]:
    samples: list[Sample] = []
    for root in roots:
        image_dir = root / "images"
        label_dir = root / "labels"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob("*.jpg")):
            sample_id = image_path.stem
            samples.append(Sample(image_path=image_path, label_path=label_dir / f"{sample_id}.txt", sample_id=sample_id, root=root))
    return samples


def load_label_masks(root: Path) -> dict[str, dict[str, dict]]:
    path = root / "label_masks.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("items", {})


def read_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            values = tuple(float(v) for v in parts)
        except ValueError:
            continue
        box = yolo_to_box(values, width, height)
        if box.width > 0.5 and box.height > 0.5:
            boxes.append(box)
    return boxes


def apply_label_masks(sample_id: str, boxes: list[Box], masks: dict[str, dict[str, dict]], width: int, height: int) -> tuple[list[Box], list[Box], bool]:
    records = masks.get(sample_id, {})
    kept = list(boxes)
    ignore_regions: list[Box] = []
    drop_image = False
    for key, record in records.items():
        if record.get("drop_image"):
            drop_image = True
        if record.get("ignore_region"):
            ignore_regions.append(Box(*record["ignore_region"]).clip(width, height))
        if record.get("add_bbox"):
            kept.append(Box(*record["add_bbox"]).clip(width, height))
        index: int | None = None
        try:
            index = int(key)
        except ValueError:
            index = None
        if index is not None and 0 <= index < len(kept):
            if record.get("masked") or record.get("deleted") or record.get("unreliable") or record.get("exclude_from_loss"):
                kept[index] = Box(0, 0, 0, 0)
            if record.get("bbox"):
                kept[index] = Box(*record["bbox"]).clip(width, height)
    return [box for box in kept if box.width > 0.5 and box.height > 0.5], ignore_regions, drop_image


def sample_has_nonempty_label(sample: Sample, masks: dict[str, dict[str, dict]] | None = None) -> bool:
    if sample.label_path.exists() and bool(sample.label_path.read_text().strip()):
        return True
    records = (masks or {}).get(sample.sample_id, {})
    return any(record.get("add_bbox") for record in records.values())


def _spread_samples(samples: list[Sample]) -> list[Sample]:
    return sorted(samples, key=lambda sample: crc32(f"{sample.root.name}/{sample.sample_id}".encode("utf-8")))


def _push_balanced_by_root(grouped: dict[Path, list[Sample]], count: int) -> list[Sample]:
    roots = sorted(grouped, key=str)
    cursors = {root: 0 for root in roots}
    selected: list[Sample] = []
    while len(selected) < count:
        pushed = False
        for root in roots:
            if len(selected) >= count:
                break
            cursor = cursors[root]
            root_samples = grouped[root]
            if cursor < len(root_samples):
                selected.append(root_samples[cursor])
                cursors[root] = cursor + 1
                pushed = True
        if not pushed:
            break
    return selected


def _interleave_labeled_and_empty(labeled: list[Sample], empty: list[Sample]) -> list[Sample]:
    if not labeled:
        return empty
    if not empty:
        return labeled
    selected: list[Sample] = []
    empty_iter = iter(empty)
    gap = max(1, len(labeled) // max(1, len(empty)))
    for index, sample in enumerate(labeled):
        selected.append(sample)
        if (index + 1) % gap == 0:
            try:
                selected.append(next(empty_iter))
            except StopIteration:
                pass
    selected.extend(empty_iter)
    return selected


def limit_samples(
    samples: list[Sample],
    max_samples: int | None,
    empty_ratio: float | None,
    masks_by_root: dict[Path, dict[str, dict[str, dict]]] | None = None,
) -> list[Sample]:
    if max_samples is None or len(samples) <= max_samples:
        return samples
    if empty_ratio is None:
        return samples[:max_samples]
    empty_target = min(max_samples, int(round(max_samples * empty_ratio)))
    labeled_target = max_samples - empty_target
    labeled_by_root: dict[Path, list[Sample]] = {}
    empty_by_root: dict[Path, list[Sample]] = {}
    for sample in samples:
        masks = (masks_by_root or {}).get(sample.root, {})
        target = labeled_by_root if sample_has_nonempty_label(sample, masks) else empty_by_root
        target.setdefault(sample.root, []).append(sample)
    for grouped in (labeled_by_root, empty_by_root):
        for root, root_samples in grouped.items():
            grouped[root] = _spread_samples(root_samples)
    labeled = _push_balanced_by_root(labeled_by_root, labeled_target)
    empty = _push_balanced_by_root(empty_by_root, empty_target)
    if len(labeled) + len(empty) < max_samples:
        remaining = max_samples - len(labeled) - len(empty)
        labeled.extend(_push_balanced_by_root(labeled_by_root, len(labeled) + remaining)[len(labeled):])
    if len(labeled) + len(empty) < max_samples:
        remaining = max_samples - len(labeled) - len(empty)
        empty.extend(_push_balanced_by_root(empty_by_root, len(empty) + remaining)[len(empty):])
    return _interleave_labeled_and_empty(labeled, empty)[:max_samples]


class SubtitleDataset(Dataset):
    def __init__(
        self,
        roots: list[Path],
        image_size: int,
        stride: int = 32,
        max_samples: int | None = None,
        empty_ratio: float | None = None,
        pooling_size: int = 9,
        kernel_scale: float = 0.1,
        min_kernel_width: float = 3.0,
        min_kernel_height: float = 3.0,
    ) -> None:
        self.image_size = image_size
        self.stride = stride
        self.pooling_size = pooling_size
        self.kernel_scale = kernel_scale
        self.min_kernel_width = min_kernel_width
        self.min_kernel_height = min_kernel_height
        masks_by_root = {root: load_label_masks(root) for root in roots}
        samples = discover_samples(roots)
        self.samples: list[Sample] = []
        self._masks_by_root = masks_by_root
        for sample in samples:
            with Image.open(sample.image_path) as img:
                width, height = img.size
            boxes = read_boxes(sample.label_path, width, height)
            _, _, drop = apply_label_masks(sample.sample_id, boxes, masks_by_root.get(sample.root, {}), width, height)
            if not drop:
                self.samples.append(sample)
        self.samples = limit_samples(self.samples, max_samples, empty_ratio, masks_by_root)
        self.summary = summarize_samples(self.samples, masks_by_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            shape = letterbox_shape(width, height, self.image_size, self.stride)
            resized = rgb.resize((shape.resized_width, shape.resized_height), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = F.pad(image, (0, shape.padded_width - shape.resized_width, 0, shape.padded_height - shape.resized_height), value=0.0)

        boxes = read_boxes(sample.label_path, width, height)
        boxes, ignore_regions, _ = apply_label_masks(sample.sample_id, boxes, self._masks_by_root.get(sample.root, {}), width, height)
        scaled_boxes = [box.transform(shape.scale_x, shape.scale_y).clip(shape.padded_width, shape.padded_height) for box in boxes]
        scaled_ignore = [box.transform(shape.scale_x, shape.scale_y).clip(shape.padded_width, shape.padded_height) for box in ignore_regions]
        targets = build_targets(
            shape.padded_width,
            shape.padded_height,
            scaled_boxes,
            scaled_ignore,
            pooling_size=self.pooling_size,
            kernel_scale=self.kernel_scale,
            min_kernel_width=self.min_kernel_width,
            min_kernel_height=self.min_kernel_height,
        )
        return {
            "image": image,
            "region": torch.from_numpy(targets.region).unsqueeze(0),
            "kernel": torch.from_numpy(targets.kernel).unsqueeze(0),
            "training_mask": torch.from_numpy(targets.training_mask).unsqueeze(0),
            "boxes": scaled_boxes,
            "shape": shape,
            "sample_id": sample.sample_id,
        }


def collate_batch(items: list[dict]) -> Batch:
    max_h = max(item["image"].shape[1] for item in items)
    max_w = max(item["image"].shape[2] for item in items)

    def pad_tensor(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        return F.pad(tensor, (0, max_w - tensor.shape[2], 0, max_h - tensor.shape[1]), value=value)

    return Batch(
        images=torch.stack([pad_tensor(item["image"], 0.0) for item in items]),
        regions=torch.stack([pad_tensor(item["region"], 0.0) for item in items]),
        kernels=torch.stack([pad_tensor(item["kernel"], 0.0) for item in items]),
        training_masks=torch.stack([pad_tensor(item["training_mask"], 0.0) for item in items]),
        boxes=[item["boxes"] for item in items],
        shapes=[item["shape"] for item in items],
        sample_ids=[item["sample_id"] for item in items],
    )


def summarize_samples(samples: list[Sample], masks_by_root: dict[Path, dict[str, dict[str, dict]]] | None = None) -> DatasetSummary:
    labeled = 0
    roots: dict[str, int] = {}
    for sample in samples:
        masks = (masks_by_root or {}).get(sample.root, {})
        if sample_has_nonempty_label(sample, masks):
            labeled += 1
        roots[str(sample.root)] = roots.get(str(sample.root), 0) + 1
    total = len(samples)
    return DatasetSummary(total=total, labeled=labeled, empty=total - labeled, roots=roots)
