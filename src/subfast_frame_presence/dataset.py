from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from subfast_detector.dataset import apply_label_masks, load_label_masks, read_boxes
from subfast_shared.geometry import Box


RESIZE_ALIGNMENT = 16
RESIZE_ALIGNMENT_MODE = "nearest_multiple_half_up"
RESIZE_INTERPOLATION = "bilinear"
SMALL_SUBTITLE_WARNING_EDGE = 8.0


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
class SmallSubtitleWarning:
    epoch: int
    sample_id: str
    image_path: str
    sample_type: str
    original_size: tuple[int, int]
    standard_output_size: tuple[int, int]
    standard_min_short_edge: float
    protection_scale: float
    protected_output_size: tuple[int, int]
    protection_satisfied: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FramePresenceMicroBatch:
    images: torch.Tensor
    subtitle_masks: torch.Tensor
    supervision_masks: torch.Tensor
    presence: torch.Tensor
    sample_ids: list[str]
    roots: list[str]
    image_paths: list[str]
    sample_types: list[str]
    resize_scales: list[float]
    output_size: tuple[int, int]


@dataclass(frozen=True)
class FramePresenceMacroBatch:
    micro_batches: tuple[FramePresenceMicroBatch, ...]

    @property
    def size(self) -> int:
        return sum(micro.images.shape[0] for micro in self.micro_batches)


@dataclass(frozen=True)
class _DatasetItem:
    sample: FramePresenceSample
    sample_type: str
    crop_view: int = 0


@dataclass(frozen=True)
class _PreparedItem:
    crop: tuple[int, int, int, int]
    source_size: tuple[int, int]
    boxes: tuple[Box, ...]
    ignored_boxes: tuple[Box, ...]
    sample_id: str
    resize_scale: float
    standard_output_size: tuple[int, int]
    output_size: tuple[int, int]
    standard_min_short_edge: float | None


def align_dimension(value: float, alignment: int = RESIZE_ALIGNMENT) -> int:
    if value <= 0.0:
        raise ValueError("scaled image dimension must be positive")
    return max(alignment, int(math.floor(value / alignment + 0.5)) * alignment)


def aligned_resize_size(
    source_size: tuple[int, int],
    resize_scale: float,
    alignment: int = RESIZE_ALIGNMENT,
) -> tuple[int, int]:
    if not 0.0 < resize_scale <= 1.0:
        raise ValueError("resize_scale must satisfy 0 < scale <= 1")
    width, height = source_size
    return align_dimension(width * resize_scale, alignment), align_dimension(height * resize_scale, alignment)


def _minimum_mapped_short_edge(
    boxes: tuple[Box, ...],
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> float | None:
    if not boxes:
        return None
    source_width, source_height = source_size
    output_width, output_height = output_size
    return min(
        min(box.width * output_width / source_width, box.height * output_height / source_height)
        for box in boxes
    )


def subtitle_protection_geometry(
    boxes: tuple[Box, ...],
    *,
    source_size: tuple[int, int],
    resize_scale: float,
    min_short_edge: float,
) -> tuple[float, tuple[int, int], tuple[int, int], float | None, bool]:
    standard_size = aligned_resize_size(source_size, resize_scale)
    standard_short_edge = _minimum_mapped_short_edge(
        boxes,
        source_size=source_size,
        output_size=standard_size,
    )
    if standard_short_edge is None or standard_short_edge >= min_short_edge:
        return resize_scale, standard_size, standard_size, standard_short_edge, True

    source_width, source_height = source_size
    required_width = max(min_short_edge * source_width / box.width for box in boxes)
    required_height = max(min_short_edge * source_height / box.height for box in boxes)
    aligned_width = math.ceil(required_width / RESIZE_ALIGNMENT) * RESIZE_ALIGNMENT
    aligned_height = math.ceil(required_height / RESIZE_ALIGNMENT) * RESIZE_ALIGNMENT
    protected_scale = min(
        1.0,
        max(
            resize_scale,
            (aligned_width - RESIZE_ALIGNMENT / 2) / source_width + 1e-12,
            (aligned_height - RESIZE_ALIGNMENT / 2) / source_height + 1e-12,
        ),
    )
    protected_size = aligned_resize_size(source_size, protected_scale)
    protected_short_edge = _minimum_mapped_short_edge(
        boxes,
        source_size=source_size,
        output_size=protected_size,
    )
    return (
        protected_scale,
        standard_size,
        protected_size,
        standard_short_edge,
        bool(protected_short_edge is not None and protected_short_edge + 1e-6 >= min_short_edge),
    )


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
            boxes, ignored_boxes, drop_image = apply_label_masks(
                sample_id,
                boxes,
                masks,
                *source_size,
            )
            if drop_image:
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
        if x2 > x1 and y2 > y1:
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
        target = Box(
            min(box.x1 for box in sample.boxes),
            min(box.y1 for box in sample.boxes),
            max(box.x2 for box in sample.boxes),
            max(box.y2 for box in sample.boxes),
        )
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
    """V5 variable-shape frame presence samples without padding."""

    def __init__(
        self,
        roots: list[Path],
        *,
        resize_scale: float = 0.25,
        min_subtitle_short_edge: float = 8.0,
        protect_small_subtitles: bool = False,
        random_crop_views: int = 0,
        random_crop_scale: tuple[float, float] = (0.3, 0.9),
        seed: int = 0,
        max_samples: int | None = None,
    ) -> None:
        self.resize_scale = resize_scale
        self.min_subtitle_short_edge = min_subtitle_short_edge
        self.protect_small_subtitles = protect_small_subtitles
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
        self._prepared: list[_PreparedItem] = []
        self.small_subtitle_warnings: list[SmallSubtitleWarning] = []
        self.positive_resize_scales: tuple[float, ...] = ()
        self.set_epoch(0, emit_warnings=False)

    def set_epoch(self, epoch: int, *, emit_warnings: bool = True) -> None:
        if self._prepared and self.epoch == epoch:
            return
        self.epoch = epoch
        bases: list[tuple[_DatasetItem, tuple[int, int, int, int], tuple[Box, ...], tuple[Box, ...], tuple[int, int], str]] = []
        positive_scales: list[float] = []
        positive_geometries: dict[int, tuple[float, tuple[int, int], tuple[int, int], float | None, bool]] = {}
        for index, item in enumerate(self.items):
            sample = item.sample
            crop = (0, 0, sample.source_size[0], sample.source_size[1])
            boxes = sample.boxes
            ignored_boxes = sample.ignored_boxes
            source_size = sample.source_size
            sample_id = sample.sample_id
            if item.sample_type == "random_crop":
                rng = random.Random(f"{self.seed}:{epoch}:{index}:{item.crop_view}")
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
            bases.append((item, crop, boxes, ignored_boxes, source_size, sample_id))
            if boxes:
                geometry = subtitle_protection_geometry(
                    boxes,
                    source_size=source_size,
                    resize_scale=self.resize_scale,
                    min_short_edge=self.min_subtitle_short_edge,
                )
                positive_geometries[index] = geometry
                positive_scales.append(geometry[0] if self.protect_small_subtitles else self.resize_scale)
        self.positive_resize_scales = tuple(positive_scales)

        prepared: list[_PreparedItem] = []
        warnings: list[SmallSubtitleWarning] = []
        for index, (item, crop, boxes, ignored_boxes, source_size, sample_id) in enumerate(bases):
            if boxes:
                protected_scale, standard_size, protected_size, standard_short_edge, satisfied = positive_geometries[index]
                actual_scale = protected_scale if self.protect_small_subtitles else self.resize_scale
                output_size = protected_size if self.protect_small_subtitles else standard_size
                if self.protect_small_subtitles and standard_short_edge is not None and standard_short_edge < SMALL_SUBTITLE_WARNING_EDGE:
                    warning = SmallSubtitleWarning(
                        epoch=epoch,
                        sample_id=sample_id,
                        image_path=str(item.sample.image_path),
                        sample_type=item.sample_type,
                        original_size=source_size,
                        standard_output_size=standard_size,
                        standard_min_short_edge=standard_short_edge,
                        protection_scale=actual_scale,
                        protected_output_size=output_size,
                        protection_satisfied=satisfied,
                    )
                    warnings.append(warning)
                    if emit_warnings:
                        print(
                            "WARNING small subtitle: "
                            f"sample_id={warning.sample_id} image_path={warning.image_path} "
                            f"original_size={source_size[0]}x{source_size[1]} "
                            f"standard_output_size={standard_size[0]}x{standard_size[1]} "
                            f"min_short_edge={standard_short_edge:.4f}px "
                            f"protection_scale={actual_scale:.8f} "
                            f"protected_output_size={output_size[0]}x{output_size[1]} "
                            f"satisfied={satisfied}",
                            flush=True,
                        )
            else:
                standard_size = aligned_resize_size(source_size, self.resize_scale)
                standard_short_edge = None
                if self.protect_small_subtitles and positive_scales:
                    rng = random.Random(f"{self.seed}:negative-scale:{epoch}:{index}")
                    actual_scale = rng.choice(positive_scales)
                else:
                    actual_scale = self.resize_scale
                output_size = aligned_resize_size(source_size, actual_scale)
            prepared.append(
                _PreparedItem(
                    crop=crop,
                    source_size=source_size,
                    boxes=boxes,
                    ignored_boxes=ignored_boxes,
                    sample_id=sample_id,
                    resize_scale=actual_scale,
                    standard_output_size=standard_size,
                    output_size=output_size,
                    standard_min_short_edge=standard_short_edge,
                )
            )
        self._prepared = prepared
        self.small_subtitle_warnings = warnings

    def __len__(self) -> int:
        return len(self.samples)

    def sample_type_for_index(self, index: int) -> str:
        return self.items[index].sample_type

    def presence_for_index(self, index: int) -> bool:
        return bool(self._prepared[index].boxes)

    def output_size_for_index(self, index: int) -> tuple[int, int]:
        return self._prepared[index].output_size

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        sample = item.sample
        prepared = self._prepared[index]
        with Image.open(sample.image_path) as image:
            if image.mode != "RGB":
                raise ValueError(f"frame image must already be RGB: {sample.image_path}")
            if item.sample_type == "random_crop":
                image = image.crop(prepared.crop)
            if image.size != prepared.output_size:
                image = image.resize(prepared.output_size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        subtitle_mask = _rasterize_boxes(
            prepared.boxes,
            source_size=prepared.source_size,
            output_size=prepared.output_size,
        )
        ignored_mask = _rasterize_boxes(
            prepared.ignored_boxes,
            source_size=prepared.source_size,
            output_size=prepared.output_size,
        )
        return {
            "image": torch.from_numpy(array).permute(2, 0, 1),
            "subtitle_mask": subtitle_mask,
            "supervision_mask": 1.0 - ignored_mask,
            "presence": torch.tensor(float(bool(prepared.boxes)), dtype=torch.float32),
            "sample_id": prepared.sample_id,
            "root": str(sample.root),
            "image_path": str(sample.image_path),
            "sample_type": item.sample_type,
            "resize_scale": prepared.resize_scale,
            "output_size": prepared.output_size,
        }

    def manifest(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for item, prepared in zip(self.items, self._prepared, strict=True):
            sample = item.sample
            stat = sample.image_path.stat()
            records.append(
                {
                    "root": str(sample.root),
                    "sample_id": prepared.sample_id,
                    "image_path": str(sample.image_path),
                    "label_path": str(sample.label_path),
                    "source_size": list(prepared.source_size),
                    "standard_output_size": list(prepared.standard_output_size),
                    "output_size": list(prepared.output_size),
                    "resize_scale": prepared.resize_scale,
                    "standard_min_subtitle_short_edge": prepared.standard_min_short_edge,
                    "image_bytes": stat.st_size,
                    "image_mtime_ns": stat.st_mtime_ns,
                    "has_subtitle": bool(prepared.boxes),
                    "sample_type": item.sample_type,
                    "crop_view": item.crop_view if item.sample_type == "random_crop" else None,
                    "boxes": [[box.x1, box.y1, box.x2, box.y2] for box in prepared.boxes],
                    "ignored_boxes": [
                        [box.x1, box.y1, box.x2, box.y2] for box in prepared.ignored_boxes
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


def collate_frame_presence_batch(items: list[dict[str, object]]) -> FramePresenceMacroBatch:
    grouped: OrderedDict[tuple[int, int], list[dict[str, object]]] = OrderedDict()
    for item in items:
        output_size = tuple(item["output_size"])  # type: ignore[arg-type]
        grouped.setdefault(output_size, []).append(item)
    micro_batches: list[FramePresenceMicroBatch] = []
    for output_size, group in grouped.items():
        micro_batches.append(
            FramePresenceMicroBatch(
                images=torch.stack([item["image"] for item in group]),  # type: ignore[arg-type]
                subtitle_masks=torch.stack([item["subtitle_mask"] for item in group]),  # type: ignore[arg-type]
                supervision_masks=torch.stack([item["supervision_mask"] for item in group]),  # type: ignore[arg-type]
                presence=torch.stack([item["presence"] for item in group]),  # type: ignore[arg-type]
                sample_ids=[str(item["sample_id"]) for item in group],
                roots=[str(item["root"]) for item in group],
                image_paths=[str(item["image_path"]) for item in group],
                sample_types=[str(item["sample_type"]) for item in group],
                resize_scales=[float(item["resize_scale"]) for item in group],
                output_size=output_size,
            )
        )
    return FramePresenceMacroBatch(tuple(micro_batches))


__all__ = [
    "FramePresenceDataset",
    "FramePresenceMacroBatch",
    "FramePresenceMicroBatch",
    "FramePresenceSample",
    "FramePresenceSummary",
    "RESIZE_ALIGNMENT",
    "RESIZE_ALIGNMENT_MODE",
    "RESIZE_INTERPOLATION",
    "SmallSubtitleWarning",
    "align_dimension",
    "aligned_resize_size",
    "collate_frame_presence_batch",
    "discover_frame_presence_samples",
    "is_roi_root",
    "subtitle_protection_geometry",
]
