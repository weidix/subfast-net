from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zlib import crc32

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from subfast_roi_data.data import (
    RoiDatasetSummary,
    RoiSample,
    discover_roi_samples,
    limit_roi_samples,
    subtitle_mask_from_labels,
)
from subfast_shared.vision import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class RoiPresenceBatch:
    images: torch.Tensor
    subtitle_masks: torch.Tensor | None
    valid_masks: torch.Tensor
    donor_images: torch.Tensor
    donor_valid_masks: torch.Tensor
    seam_donor_images: torch.Tensor
    seam_donor_valid_masks: torch.Tensor
    donor_available: torch.Tensor
    presence: torch.Tensor
    sample_ids: list[str]
    segment_ids: list[str]
    roots: list[str]
    image_paths: list[str]
    ocr_texts: list[str]


class RoiPresenceDataset(Dataset):
    def __init__(
        self,
        roots: list[Path],
        *,
        resize_roi: tuple[int, int] | None = None,
        resize_mode: str = "letterbox",
        max_samples: int | None = None,
        negative_ratio: float | None = None,
        load_subtitle_masks: bool = True,
    ) -> None:
        self.resize_roi = resize_roi
        if resize_mode not in {"letterbox", "stretch"}:
            raise ValueError("resize_mode must be 'letterbox' or 'stretch'")
        self.resize_mode = resize_mode
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
        negative_by_root: dict[Path, list[RoiSample]] = {}
        all_negatives: list[RoiSample] = []
        for sample in samples:
            if sample.has_subtitle:
                continue
            negative_by_root.setdefault(sample.root, []).append(sample)
            all_negatives.append(sample)
        self.donor_samples: list[tuple[RoiSample | None, RoiSample | None]] = []
        for sample in self.samples:
            if not sample.has_subtitle:
                self.donor_samples.append((None, None))
                continue
            donor_pool = negative_by_root.get(sample.root) or all_negatives
            if not donor_pool:
                self.donor_samples.append((None, None))
                continue
            start = crc32(f"{sample.root}/{sample.sample_id}".encode("utf-8")) % len(donor_pool)
            second = (start + max(1, len(donor_pool) // 2)) % len(donor_pool)
            self.donor_samples.append((donor_pool[start], donor_pool[second]))
        self.positive_without_donor = sum(
            int(sample.has_subtitle and donor is None)
            for sample, (donor, _) in zip(self.samples, self.donor_samples, strict=True)
        )
        candidate_text = [self._sample_has_candidate_text(sample) for sample in self.samples]
        self.text_distractor_negatives = sum(
            int(not sample.has_subtitle and has_candidate)
            for sample, has_candidate in zip(self.samples, candidate_text, strict=True)
        )
        self.positive_without_region = sum(
            int(sample.has_subtitle and not has_candidate)
            for sample, has_candidate in zip(self.samples, candidate_text, strict=True)
        )
        self.summary = self._summarize()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image_tensor, subtitle_mask, valid_mask = self._load_sample_tensors(
            sample,
            load_subtitle_mask=self.load_subtitle_masks,
        )
        donor_sample, seam_donor_sample = self.donor_samples[index]
        if donor_sample is not None and seam_donor_sample is not None:
            donor_image, _, donor_valid_mask = self._load_sample_tensors(
                donor_sample,
                load_subtitle_mask=False,
            )
            seam_donor_image, _, seam_donor_valid_mask = self._load_sample_tensors(
                seam_donor_sample,
                load_subtitle_mask=False,
            )
            donor_available = True
        else:
            donor_image = image_tensor
            donor_valid_mask = valid_mask
            seam_donor_image = image_tensor
            seam_donor_valid_mask = valid_mask
            donor_available = False
        return {
            "image": image_tensor,
            "subtitle_mask": subtitle_mask,
            "valid_mask": valid_mask,
            "donor_image": donor_image,
            "donor_valid_mask": donor_valid_mask,
            "seam_donor_image": seam_donor_image,
            "seam_donor_valid_mask": seam_donor_valid_mask,
            "donor_available": donor_available,
            "presence": torch.tensor(float(sample.has_subtitle), dtype=torch.float32),
            "sample_id": sample.sample_id,
            "segment_id": sample.segment_id,
            "root": str(sample.root),
            "image_path": str(sample.image_path),
            "ocr_text": sample.ocr_text,
        }

    def _load_sample_tensors(
        self,
        sample: RoiSample,
        *,
        load_subtitle_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        with Image.open(sample.image_path) as image:
            rgb = image.convert("RGB")
            actual_size = rgb.size
            expected = tuple(sample.annotation.get("roi_size") or self.expected_roi_size)
            if actual_size != expected:
                raise ValueError(
                    f"ROI image size mismatch for {sample.image_path}: actual={actual_size} expected={expected}"
                )
            if self.resize_roi is None and actual_size != self.output_roi_size:
                raise ValueError(
                    f"ROI input size mismatch for {sample.image_path}: {actual_size} != {self.output_roi_size}"
                )
            array = np.asarray(rgb, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        valid_mask = torch.ones((1, actual_size[1], actual_size[0]), dtype=torch.float32)
        subtitle_mask = (
            subtitle_mask_from_labels(
                sample.label_path,
                output_size=actual_size,
                label_size=tuple(expected),
            )
            if load_subtitle_mask
            else None
        )
        if self.resize_roi is not None and actual_size != self.resize_roi:
            image_tensor, subtitle_mask, valid_mask = self._resize_tensors(
                image_tensor,
                subtitle_mask,
                valid_mask,
            )
        return image_tensor, subtitle_mask, valid_mask

    def _resize_tensors(
        self,
        image: torch.Tensor,
        subtitle_mask: torch.Tensor | None,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.resize_roi is None:
            return image, subtitle_mask, valid_mask
        target_width, target_height = self.resize_roi
        source_height, source_width = image.shape[-2:]
        if self.resize_mode == "stretch":
            resized_image = F.interpolate(
                image.unsqueeze(0),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            resized_mask = (
                F.interpolate(
                    subtitle_mask.unsqueeze(0),
                    size=(target_height, target_width),
                    mode="nearest",
                ).squeeze(0)
                if subtitle_mask is not None
                else None
            )
            return (
                resized_image,
                resized_mask,
                valid_mask.new_ones((1, target_height, target_width)),
            )

        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, round(source_width * scale)))
        resized_height = max(1, min(target_height, round(source_height * scale)))
        resized_image = F.interpolate(
            image.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        resized_mask = (
            F.interpolate(
                subtitle_mask.unsqueeze(0),
                size=(resized_height, resized_width),
                mode="nearest",
            ).squeeze(0)
            if subtitle_mask is not None
            else None
        )
        resized_valid_mask = F.interpolate(
            valid_mask.unsqueeze(0),
            size=(resized_height, resized_width),
            mode="nearest",
        ).squeeze(0)
        pad_left = (target_width - resized_width) // 2
        pad_right = target_width - resized_width - pad_left
        pad_top = (target_height - resized_height) // 2
        pad_bottom = target_height - resized_height - pad_top
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        resized_image = F.pad(resized_image, padding, value=0.0)
        if resized_mask is not None:
            resized_mask = F.pad(resized_mask, padding, value=0.0)
        resized_valid_mask = F.pad(resized_valid_mask, padding, value=0.0)
        return resized_image, resized_mask, resized_valid_mask

    @staticmethod
    def _sample_has_candidate_text(sample: RoiSample) -> bool:
        return bool(sample.label_path.read_text(encoding="utf-8").strip())

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
        valid_masks=torch.stack([pad(item["valid_mask"]) for item in items]),
        donor_images=torch.stack([pad(item["donor_image"]) for item in items]),
        donor_valid_masks=torch.stack([pad(item["donor_valid_mask"]) for item in items]),
        seam_donor_images=torch.stack([pad(item["seam_donor_image"]) for item in items]),
        seam_donor_valid_masks=torch.stack(
            [pad(item["seam_donor_valid_mask"]) for item in items]
        ),
        donor_available=torch.tensor([item["donor_available"] for item in items], dtype=torch.bool),
        presence=torch.stack([item["presence"] for item in items]),
        sample_ids=[item["sample_id"] for item in items],
        segment_ids=[item["segment_id"] for item in items],
        roots=[item["root"] for item in items],
        image_paths=[item["image_path"] for item in items],
        ocr_texts=[item["ocr_text"] for item in items],
    )
