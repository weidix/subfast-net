from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from subfast_shared.geometry import Box


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float


@dataclass
class _ComponentStats:
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    confidence_sum: float
    kernel_weight_sum: float
    kernel_x_sum: float
    count: int

    @classmethod
    def empty(cls, width: int, height: int) -> "_ComponentStats":
        return cls(
            min_x=width,
            min_y=height,
            max_x=0,
            max_y=0,
            confidence_sum=0.0,
            kernel_weight_sum=0.0,
            kernel_x_sum=0.0,
            count=0,
        )

    def add(self, x: int, y: int, region_prob: float, kernel_prob: float) -> None:
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)
        self.confidence_sum += region_prob * 0.8 + kernel_prob * 0.2
        self.kernel_weight_sum += kernel_prob
        self.kernel_x_sum += kernel_prob * (x + 0.5)
        self.count += 1

    @property
    def confidence(self) -> float:
        return self.confidence_sum / max(1, self.count)

    @property
    def fallback_box(self) -> Box:
        return Box(float(self.min_x), float(self.min_y), float(self.max_x + 1), float(self.max_y + 1))

    @property
    def kernel_center_x(self) -> float | None:
        if self.kernel_weight_sum <= 0.0:
            return None
        return self.kernel_x_sum / self.kernel_weight_sum


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _neighbors(x: int, y: int, width: int, height: int):
    if x > 0:
        yield x - 1, y
    if x + 1 < width:
        yield x + 1, y
    if y > 0:
        yield x, y - 1
    if y + 1 < height:
        yield x, y + 1


def logits_to_boxes(
    region_logits: np.ndarray,
    kernel_logits: np.ndarray,
    region_threshold: float = 0.5,
    kernel_threshold: float = 0.5,
    min_size: float = 3.0,
    max_width_ratio: float = 1.0,
) -> list[Detection]:
    region_prob = _sigmoid(region_logits.astype(np.float32))
    kernel_prob = _sigmoid(kernel_logits.astype(np.float32))
    region = region_prob >= region_threshold
    kernel = (kernel_prob >= kernel_threshold) & region
    height, width = region.shape
    owner = np.full((height, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int, int]] = deque()
    seed_id = 0
    for y in range(height):
        for x in range(width):
            if not kernel[y, x] or owner[y, x] != -1:
                continue
            owner[y, x] = seed_id
            queue.append((x, y, seed_id))
            while queue:
                cx, cy, sid = queue.popleft()
                for nx, ny in _neighbors(cx, cy, width, height):
                    if kernel[ny, nx] and owner[ny, nx] == -1:
                        owner[ny, nx] = sid
                        queue.append((nx, ny, sid))
            seed_id += 1

    queue.clear()
    for y in range(height):
        for x in range(width):
            sid = owner[y, x]
            if sid >= 0:
                queue.append((x, y, int(sid)))
    while queue:
        cx, cy, sid = queue.popleft()
        for nx, ny in _neighbors(cx, cy, width, height):
            if region[ny, nx] and owner[ny, nx] == -1:
                owner[ny, nx] = sid
                queue.append((nx, ny, sid))

    detections: list[Detection] = []
    for sid in range(seed_id):
        ys, xs = np.where(owner == sid)
        if len(xs) == 0:
            continue
        stats = _ComponentStats.empty(width, height)
        for y, x in zip(ys, xs):
            stats.add(int(x), int(y), float(region_prob[y, x]), float(kernel_prob[y, x]))
        box = _refined_component_box(owner, sid, region_prob, stats)
        box = _clamp_component_width(box, stats, width, max_width_ratio)
        if box.width >= min_size and box.height >= min_size:
            detections.append(Detection(box=box, score=stats.confidence))
    return _suppress_non_bottom_candidates_when_bottom_subtitle_exists(detections, height)


def _refined_component_box(owner: np.ndarray, sid: int, region_prob: np.ndarray, fallback: _ComponentStats) -> Box:
    owned = owner == sid
    if not np.any(owned):
        return fallback.fallback_box
    threshold = max(float(region_prob[owned].max()) * 0.95, 0.5)
    refined = owned & (region_prob >= threshold)
    ys, xs = np.where(refined)
    if len(xs) == 0:
        return fallback.fallback_box
    return Box(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _clamp_component_width(box: Box, stats: _ComponentStats, output_width: int, max_width_ratio: float) -> Box:
    if max_width_ratio <= 0.0 or max_width_ratio >= 1.0 or output_width <= 0:
        return box
    max_width = float(output_width) * max_width_ratio
    if box.width <= max_width:
        return box
    center = stats.kernel_center_x
    if center is None:
        center = (box.x1 + box.x2) * 0.5
    center = min(max(center, 0.0), float(output_width))
    x1 = center - max_width * 0.5
    x2 = center + max_width * 0.5
    if x1 < 0.0:
        x2 -= x1
        x1 = 0.0
    if x2 > output_width:
        overflow = x2 - float(output_width)
        x1 = max(0.0, x1 - overflow)
        x2 = float(output_width)
    return Box(x1, box.y1, x2, box.y2)


def _suppress_non_bottom_candidates_when_bottom_subtitle_exists(detections: list[Detection], height: int) -> list[Detection]:
    if len(detections) <= 1 or height <= 0:
        return detections
    bottom_start = float(height) * 0.75
    if not any(_center_y(detection.box) >= bottom_start for detection in detections):
        return detections
    return [detection for detection in detections if _center_y(detection.box) >= bottom_start]


def _center_y(box: Box) -> float:
    return (box.y1 + box.y2) * 0.5
