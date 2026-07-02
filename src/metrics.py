from __future__ import annotations

from dataclasses import dataclass

from .geometry import Box, box_iou


@dataclass(frozen=True)
class ImageMetrics:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denom if denom else 0.0


def evaluate_image(predictions: list[Box], targets: list[Box], iou_threshold: float = 0.5) -> ImageMetrics:
    predictions = merge_adjacent_subtitle_boxes(predictions)
    targets = merge_adjacent_subtitle_boxes(targets)
    matched_targets: set[int] = set()
    true_positive = 0
    false_positive = 0
    for prediction in predictions:
        best_iou = 0.0
        best_index = -1
        for index, target in enumerate(targets):
            if index in matched_targets:
                continue
            iou = box_iou(prediction, target)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            true_positive += 1
            matched_targets.add(best_index)
        else:
            false_positive += 1
    return ImageMetrics(true_positive, false_positive, len(targets) - len(matched_targets))


def merge_adjacent_subtitle_boxes(boxes: list[Box], max_vertical_gap_ratio: float = 0.75, min_horizontal_overlap: float = 0.45) -> list[Box]:
    if len(boxes) <= 1:
        return boxes
    remaining = sorted(boxes, key=lambda box: (box.y1, box.x1))
    changed = True
    while changed:
        changed = False
        merged: list[Box] = []
        used = [False] * len(remaining)
        for i, box in enumerate(remaining):
            if used[i]:
                continue
            current = box
            used[i] = True
            for j in range(i + 1, len(remaining)):
                other = remaining[j]
                if used[j]:
                    continue
                if _should_merge(current, other, max_vertical_gap_ratio, min_horizontal_overlap):
                    current = _union(current, other)
                    used[j] = True
                    changed = True
            merged.append(current)
        remaining = sorted(merged, key=lambda box: (box.y1, box.x1))
    return remaining


def _should_merge(a: Box, b: Box, max_vertical_gap_ratio: float, min_horizontal_overlap: float) -> bool:
    vertical_overlap = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    min_height = max(min(a.height, b.height), 1.0)
    max_height = max(a.height, b.height, 1.0)
    horizontal_gap = _axis_gap(a.x1, a.x2, b.x1, b.x2)
    if vertical_overlap / min_height >= 0.5:
        return horizontal_gap <= max(max_height * 1.5, 48.0)

    vertical_gap = _axis_gap(a.y1, a.y2, b.y1, b.y2)
    horizontal_overlap = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    min_width = max(min(a.width, b.width), 1.0)
    return horizontal_overlap / min_width >= min_horizontal_overlap and vertical_gap <= max_height * max_vertical_gap_ratio


def _axis_gap(a1: float, a2: float, b1: float, b2: float) -> float:
    if a2 < b1:
        return b1 - a2
    if b2 < a1:
        return a1 - b2
    return 0.0


def _union(a: Box, b: Box) -> Box:
    return Box(min(a.x1, b.x1), min(a.y1, b.y1), max(a.x2, b.x2), max(a.y2, b.y2))
