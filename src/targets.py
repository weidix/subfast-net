from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Box


@dataclass(frozen=True)
class TargetMaps:
    region: np.ndarray
    kernel: np.ndarray
    training_mask: np.ndarray


def _fill(mask: np.ndarray, box: Box, value: float) -> None:
    h, w = mask.shape
    x1 = int(np.floor(max(0.0, min(float(w), box.x1))))
    y1 = int(np.floor(max(0.0, min(float(h), box.y1))))
    x2 = int(np.ceil(max(0.0, min(float(w), box.x2))))
    y2 = int(np.ceil(max(0.0, min(float(h), box.y2))))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = value


def _fill_instance(mask: np.ndarray, box: Box, instance_id: int) -> None:
    h, w = mask.shape
    x1 = int(np.floor(max(0.0, min(float(w), box.x1))))
    y1 = int(np.floor(max(0.0, min(float(h), box.y1))))
    x2 = int(np.ceil(max(0.0, min(float(w), box.x2))))
    y2 = int(np.ceil(max(0.0, min(float(h), box.y2))))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = instance_id


def _shrink(box: Box, scale: float, min_width: float, min_height: float) -> Box:
    width = box.width if box.width <= min_width else min(max(box.width * scale, min_width), box.width)
    height = box.height if box.height <= min_height else min(max(box.height * scale, min_height), box.height)
    cx = (box.x1 + box.x2) * 0.5
    cy = (box.y1 + box.y2) * 0.5
    return Box(cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5)


def _min_pool_instances(instances: np.ndarray, pooling_size: int) -> np.ndarray:
    if pooling_size <= 1:
        return instances.copy()
    if pooling_size % 2 == 0:
        raise ValueError("pooling_size must be odd")
    radius = pooling_size // 2
    pooled = np.zeros_like(instances)
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        mask = instances == instance_id
        padded = np.pad(mask, radius, mode="constant", constant_values=True)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (pooling_size, pooling_size))
        keep = np.all(windows, axis=(-1, -2)) & mask
        pooled[keep] = instance_id
    return pooled


def build_targets(
    width: int,
    height: int,
    boxes: list[Box],
    ignore_regions: list[Box] | None = None,
    pooling_size: int = 9,
    kernel_scale: float = 0.1,
    min_kernel_width: float = 3.0,
    min_kernel_height: float = 3.0,
) -> TargetMaps:
    instances = np.zeros((height, width), dtype=np.int32)
    kernel = np.zeros((height, width), dtype=np.float32)
    training_mask = np.ones((height, width), dtype=np.float32)
    for ignore in ignore_regions or []:
        _fill(training_mask, ignore, 0.0)
    clipped_boxes: list[Box] = []
    for instance_id, box in enumerate(boxes, start=1):
        clipped = box.clip(width, height)
        if clipped.width <= 0.5 or clipped.height <= 0.5:
            continue
        clipped_boxes.append(clipped)
        _fill_instance(instances, clipped, instance_id)
    region = (instances > 0).astype(np.float32)
    pooled = _min_pool_instances(instances, pooling_size)
    kernel[pooled > 0] = 1.0
    owner = np.zeros((height, width), dtype=np.int32)
    for instance_id, clipped in enumerate(clipped_boxes, start=1):
        shrink = _shrink(clipped, kernel_scale, min_kernel_width, min_kernel_height)
        h, w = kernel.shape
        x1 = int(np.floor(max(0.0, min(float(w), shrink.x1))))
        y1 = int(np.floor(max(0.0, min(float(h), shrink.y1))))
        x2 = int(np.ceil(max(0.0, min(float(w), shrink.x2))))
        y2 = int(np.ceil(max(0.0, min(float(h), shrink.y2))))
        for y in range(y1, y2):
            for x in range(x1, x2):
                if owner[y, x] == 0 or owner[y, x] == instance_id:
                    owner[y, x] = instance_id
                    kernel[y, x] = 1.0
                else:
                    kernel[y, x] = 0.0
    return TargetMaps(region=region, kernel=kernel, training_mask=training_mask)
