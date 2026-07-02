from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        ).ordered()

    def ordered(self) -> "Box":
        return Box(min(self.x1, self.x2), min(self.y1, self.y2), max(self.x1, self.x2), max(self.y1, self.y2))

    def transform(self, sx: float, sy: float, dx: float = 0.0, dy: float = 0.0) -> "Box":
        return Box(self.x1 * sx + dx, self.y1 * sy + dy, self.x2 * sx + dx, self.y2 * sy + dy).ordered()


@dataclass(frozen=True)
class LetterboxShape:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    padded_width: int
    padded_height: int
    scale_x: float
    scale_y: float


def align_up(value: int, stride: int) -> int:
    return int(ceil(value / stride) * stride)


def letterbox_shape(width: int, height: int, size: int, stride: int = 32) -> LetterboxShape:
    scale = size / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return LetterboxShape(
        original_width=width,
        original_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        padded_width=align_up(resized_width, stride),
        padded_height=align_up(resized_height, stride),
        scale_x=resized_width / width,
        scale_y=resized_height / height,
    )


def yolo_to_box(values: Iterable[float], width: int, height: int) -> Box:
    _, x_center, y_center, box_width, box_height = values
    cx = float(x_center) * width
    cy = float(y_center) * height
    bw = float(box_width) * width
    bh = float(box_height) * height
    return Box(cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5).clip(width, height)


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0.0 else 0.0

