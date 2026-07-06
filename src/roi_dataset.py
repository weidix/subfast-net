from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from zlib import crc32

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .geometry import Box, yolo_to_box
from .roi_pairs import parse_frame_index, parse_video_frame_from_sample_id, parse_video_id

REVIEW_FILENAME = "segment_review.json"


@dataclass(frozen=True)
class RoiSample:
    image_path: Path
    label_path: Path
    sample_id: str
    root: Path
    has_subtitle: bool
    segment_id: str
    video_id: str | None
    frame_index: int | None
    ocr_text: str
    annotation: dict


@dataclass(frozen=True)
class RoiBatch:
    images: torch.Tensor
    subtitle_masks: torch.Tensor | None
    presence: torch.Tensor
    segment_ids: list[str]
    sample_ids: list[str]
    roots: list[str]
    video_ids: list[str | None]
    frame_indices: list[int | None]
    ocr_texts: list[str]


@dataclass(frozen=True)
class RoiDatasetSummary:
    total: int
    positive: int
    empty: int
    roots: dict[str, int]
    roi_size: tuple[int, int]
    positive_segments: int = 0
    repeated_positive_segments: int = 0
    same_segment_pairs: int = 0

    @property
    def positive_ratio(self) -> float:
        return self.positive / self.total if self.total else 0.0

    @property
    def empty_ratio(self) -> float:
        return self.empty / self.total if self.total else 0.0


def load_summary(root: Path) -> dict:
    path = root / "summary.json"
    if not path.exists():
        raise ValueError(f"missing ROI summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "roi_size" not in data or len(data["roi_size"]) != 2:
        raise ValueError(f"ROI summary missing roi_size: {path}")
    return data


def load_review(root: Path) -> dict[str, dict]:
    path = root / REVIEW_FILENAME
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 2:
        raise ValueError(f"unsupported ROI review version in {path}")
    return dict(data.get("items", {}))


def annotation_sample_id(item: dict, index: int) -> str:
    image = Path(str(item.get("image", ""))).stem
    return image or f"sample_{index:06d}"


def read_annotations(root: Path) -> list[dict]:
    path = root / "annotations.jsonl"
    if not path.exists():
        raise ValueError(f"missing ROI annotations: {path}")
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def annotation_video_id(item: dict, sample_id: str) -> str | None:
    source_annotation = item.get("source_annotation") if isinstance(item.get("source_annotation"), dict) else {}
    video_id = (
        parse_video_id(item.get("video_id"))
        or parse_video_id(item.get("sequence_id"))
        or parse_video_id(item.get("source_video"))
        or parse_video_id(source_annotation.get("video_id"))
        or parse_video_id(source_annotation.get("sequence_id"))
        or parse_video_id(source_annotation.get("source_video"))
    )
    if video_id is not None:
        return video_id
    parsed_video, _ = parse_video_frame_from_sample_id(sample_id)
    return parsed_video


def annotation_frame_index(item: dict, sample_id: str) -> int | None:
    source_annotation = item.get("source_annotation") if isinstance(item.get("source_annotation"), dict) else {}
    frame_index = parse_frame_index(item.get("frame_index"))
    if frame_index is not None:
        return frame_index
    frame_index = parse_frame_index(source_annotation.get("frame_index"))
    if frame_index is not None:
        return frame_index
    _, parsed_frame = parse_video_frame_from_sample_id(sample_id)
    return parsed_frame


def annotation_ocr_text(item: dict) -> str:
    text = item.get("ocr_text_normalized") or item.get("ocr_text") or ""
    return str(text)


def read_roi_label_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            values = tuple(float(part) for part in parts)
        except ValueError:
            continue
        box = yolo_to_box(values, width, height)
        if box.width > 0.5 and box.height > 0.5:
            boxes.append(box)
    return boxes


def subtitle_mask_from_labels(
    label_path: Path,
    *,
    output_size: tuple[int, int],
    label_size: tuple[int, int],
) -> torch.Tensor:
    output_w, output_h = output_size
    mask = torch.zeros((1, output_h, output_w), dtype=torch.float32)
    roi_w, roi_h = label_size
    if roi_w <= 0 or roi_h <= 0:
        return mask
    boxes = read_roi_label_boxes(label_path, roi_w, roi_h)
    if not boxes:
        return mask
    scale_x = output_w / roi_w
    scale_y = output_h / roi_h
    for box in boxes:
        x1 = max(0.0, min(box.x1, float(roi_w)))
        y1 = max(0.0, min(box.y1, float(roi_h)))
        x2 = max(0.0, min(box.x2, float(roi_w)))
        y2 = max(0.0, min(box.y2, float(roi_h)))
        if x2 <= x1 or y2 <= y1:
            continue
        left = max(0, min(output_w, int(np.floor(x1 * scale_x))))
        top = max(0, min(output_h, int(np.floor(y1 * scale_y))))
        right = max(left + 1, min(output_w, int(np.ceil(x2 * scale_x))))
        bottom = max(top + 1, min(output_h, int(np.ceil(y2 * scale_y))))
        mask[:, top:bottom, left:right] = 1.0
    return mask


def discover_roi_samples(root: Path) -> tuple[list[RoiSample], tuple[int, int]]:
    summary = load_summary(root)
    roi_size = (int(summary["roi_size"][0]), int(summary["roi_size"][1]))
    labels_dir = root / "labels"
    if not labels_dir.is_dir():
        raise ValueError(f"missing ROI labels dir: {labels_dir}")
    review = load_review(root)
    samples: list[RoiSample] = []
    for index, item in enumerate(read_annotations(root)):
        sample_id = annotation_sample_id(item, index)
        reviewed = review.get(sample_id, {})
        has_subtitle = bool(reviewed.get("has_subtitle", item.get("has_subtitle", False)))
        default_segment = str(item.get("segment_marker") or sample_id)
        segment_id = str(reviewed.get("segment_id") or default_segment)
        image_path = root / str(item.get("image"))
        label_path = labels_dir / f"{sample_id}.txt"
        if not label_path.exists():
            raise ValueError(f"missing ROI label file: {label_path}")
        samples.append(
            RoiSample(
                image_path=image_path,
                label_path=label_path,
                sample_id=sample_id,
                root=root,
                has_subtitle=has_subtitle,
                segment_id=segment_id,
                video_id=annotation_video_id(item, sample_id),
                frame_index=annotation_frame_index(item, sample_id),
                ocr_text=annotation_ocr_text(item),
                annotation=item,
            )
        )
    return samples, roi_size


def _spread_samples(samples: list[RoiSample]) -> list[RoiSample]:
    return sorted(samples, key=lambda sample: crc32(f"{sample.root.name}/{sample.sample_id}".encode("utf-8")))


def _spread_segment_groups(groups: dict[str, list[RoiSample]]) -> list[list[RoiSample]]:
    return [
        _spread_samples(group)
        for _, group in sorted(
            groups.items(),
            key=lambda item: crc32(f"{item[1][0].root.name}/{item[0]}".encode("utf-8")),
        )
    ]


def _take_segment_groups(groups: list[list[RoiSample]], target: int) -> list[RoiSample]:
    selected: list[RoiSample] = []
    for group in groups:
        remaining = target - len(selected)
        if remaining <= 0:
            break
        if len(group) <= remaining:
            selected.extend(group)
        elif remaining >= 2:
            selected.extend(group[:remaining])
    return selected


def limit_roi_validation_samples(
    samples: list[RoiSample],
    max_samples: int | None,
    empty_ratio: float | None,
) -> list[RoiSample]:
    if max_samples is None or len(samples) <= max_samples:
        return samples
    positives = [sample for sample in samples if sample.has_subtitle]
    empties = _spread_samples([sample for sample in samples if not sample.has_subtitle])
    if empty_ratio is None:
        empty_ratio = len(empties) / len(samples) if samples else 0.0
    empty_target = min(max_samples, int(round(max_samples * empty_ratio)))
    positive_target = max_samples - empty_target
    groups_by_segment: dict[str, list[RoiSample]] = {}
    for sample in positives:
        groups_by_segment.setdefault(sample.segment_id, []).append(sample)
    paired_groups = _spread_segment_groups({key: group for key, group in groups_by_segment.items() if len(group) >= 2})
    single_groups = _spread_segment_groups({key: group for key, group in groups_by_segment.items() if len(group) < 2})
    selected_positives = _take_segment_groups(paired_groups, positive_target)
    if len(selected_positives) < positive_target:
        selected_positives.extend(_take_segment_groups(single_groups, positive_target - len(selected_positives)))
    selected = selected_positives[:positive_target] + empties[:empty_target]
    if len(selected) < max_samples:
        chosen = {id(sample) for sample in selected}
        remainder = _spread_samples([sample for sample in samples if id(sample) not in chosen])
        selected.extend(remainder[: max_samples - len(selected)])
    return _spread_samples(selected[:max_samples])


def limit_roi_samples(
    samples: list[RoiSample],
    max_samples: int | None,
    empty_ratio: float | None,
) -> list[RoiSample]:
    if max_samples is None or len(samples) <= max_samples:
        return samples
    if empty_ratio is None:
        return samples[:max_samples]
    empty_target = min(max_samples, int(round(max_samples * empty_ratio)))
    positive_target = max_samples - empty_target
    positives = _spread_samples([sample for sample in samples if sample.has_subtitle])
    empties = _spread_samples([sample for sample in samples if not sample.has_subtitle])
    selected = positives[:positive_target] + empties[:empty_target]
    if len(selected) < max_samples:
        selected.extend(positives[positive_target : positive_target + max_samples - len(selected)])
    if len(selected) < max_samples:
        selected.extend(empties[empty_target : empty_target + max_samples - len(selected)])
    return _spread_samples(selected[:max_samples])


class RoiPresenceEmbeddingDataset(Dataset):
    def __init__(
        self,
        roots: list[Path],
        *,
        resize_roi: tuple[int, int] | None = None,
        max_samples: int | None = None,
        empty_ratio: float | None = None,
        segment_aware_limit: bool = False,
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
        if segment_aware_limit:
            self.samples = limit_roi_validation_samples(samples, max_samples, empty_ratio)
        else:
            self.samples = limit_roi_samples(samples, max_samples, empty_ratio)
        self.summary = self._summarize()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        with Image.open(sample.image_path) as img:
            rgb = img.convert("RGB")
            actual_size = rgb.size
            expected = tuple(sample.annotation.get("roi_size") or self.expected_roi_size)
            if actual_size != expected:
                raise ValueError(f"ROI image size mismatch for {sample.image_path}: actual={actual_size} expected={expected}")
            if self.resize_roi is not None and actual_size != self.resize_roi:
                rgb = rgb.resize(self.resize_roi, Image.Resampling.BILINEAR)
            elif self.resize_roi is None and actual_size != self.output_roi_size:
                raise ValueError(f"ROI input size mismatch for {sample.image_path}: {actual_size} != {self.output_roi_size}")
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": image,
            "subtitle_mask": (
                subtitle_mask_from_labels(
                    sample.label_path,
                    output_size=self.output_roi_size,
                    label_size=tuple(expected),
                )
                if self.load_subtitle_masks
                else None
            ),
            "presence": torch.tensor(1.0 if sample.has_subtitle else 0.0, dtype=torch.float32),
            "segment_id": sample.segment_id,
            "sample_id": sample.sample_id,
            "root": str(sample.root),
            "video_id": sample.video_id,
            "frame_index": sample.frame_index,
            "ocr_text": sample.ocr_text,
        }

    def _summarize(self) -> RoiDatasetSummary:
        roots: dict[str, int] = {}
        positive = 0
        for sample in self.samples:
            roots[str(sample.root)] = roots.get(str(sample.root), 0) + 1
            positive += int(sample.has_subtitle)
        total = len(self.samples)
        positive_segments = Counter(sample.segment_id for sample in self.samples if sample.has_subtitle)
        return RoiDatasetSummary(
            total=total,
            positive=positive,
            empty=total - positive,
            roots=roots,
            roi_size=self.output_roi_size,
            positive_segments=len(positive_segments),
            repeated_positive_segments=sum(1 for count in positive_segments.values() if count >= 2),
            same_segment_pairs=sum(count * (count - 1) // 2 for count in positive_segments.values()),
        )

def collate_roi_batch(items: list[dict]) -> RoiBatch:
    max_h = max(item["image"].shape[1] for item in items)
    max_w = max(item["image"].shape[2] for item in items)

    def pad_image(tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor, (0, max_w - tensor.shape[2], 0, max_h - tensor.shape[1]), value=0.0)

    def pad_mask(tensor: torch.Tensor) -> torch.Tensor:
        return F.pad(tensor, (0, max_w - tensor.shape[2], 0, max_h - tensor.shape[1]), value=0.0)

    subtitle_masks = (
        torch.stack([pad_mask(item["subtitle_mask"]) for item in items])
        if items[0]["subtitle_mask"] is not None
        else None
    )

    return RoiBatch(
        images=torch.stack([pad_image(item["image"]) for item in items]),
        subtitle_masks=subtitle_masks,
        presence=torch.stack([item["presence"] for item in items]),
        segment_ids=[item["segment_id"] for item in items],
        sample_ids=[item["sample_id"] for item in items],
        roots=[item["root"] for item in items],
        video_ids=[item["video_id"] for item in items],
        frame_indices=[item["frame_index"] for item in items],
        ocr_texts=[item["ocr_text"] for item in items],
    )
