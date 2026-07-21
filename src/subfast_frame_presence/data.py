from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CACHE_VERSION = 3
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)
SUMMARY_FILENAME = "summary.json"
SAMPLES_FILENAME = "samples.jsonl"
IMAGES_FILENAME = "images.npy"
FOCUS_FILENAME = "focus.npy"
FOCUS_MODE_FILENAME = "focus_mode.npy"
TARGETS_FILENAME = "targets.npy"
PRESENCE_FILENAME = "presence.npy"
BOUNDS_FILENAME = "bounds.npy"


@dataclass(frozen=True)
class CacheContract:
    input_width: int
    input_height: int
    focus_width: int
    focus_height: int
    heatmap_stride_x: int
    heatmap_stride_y: int
    heatmap_width: int
    heatmap_height: int


def heatmap_size(
    input_width: int,
    input_height: int,
    stride_x: int,
    stride_y: int,
) -> tuple[int, int]:
    return math.ceil(input_width / stride_x), math.ceil(input_height / stride_y)


def rasterize_boxes(
    boxes: list[tuple[float, float, float, float]],
    *,
    source_width: int,
    source_height: int,
    heatmap_width: int,
    heatmap_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source-pixel boxes into the model heatmap coordinate space."""
    target = np.zeros((heatmap_height, heatmap_width), dtype=np.uint8)
    if not boxes:
        return target, np.full((4,), -1, dtype=np.int16)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("invalid source dimensions")
    for x1, y1, x2, y2 in boxes:
        left = max(0, min(heatmap_width, math.floor(x1 / source_width * heatmap_width)))
        top = max(0, min(heatmap_height, math.floor(y1 / source_height * heatmap_height)))
        right = max(
            left + 1,
            min(heatmap_width, math.ceil(x2 / source_width * heatmap_width)),
        )
        bottom = max(
            top + 1,
            min(heatmap_height, math.ceil(y2 / source_height * heatmap_height)),
        )
        target[top:bottom, left:right] = 1
    rows, columns = np.nonzero(target)
    if not len(rows):
        raise ValueError("positive source boxes produced an empty heatmap target")
    bounds = np.asarray(
        [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1],
        dtype=np.int16,
    )
    return target, bounds


def load_cache_contract(root: Path) -> tuple[dict[str, object], CacheContract]:
    summary_path = root / SUMMARY_FILENAME
    if not summary_path.exists():
        raise ValueError(f"missing frame-presence cache summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("version", -1)) != CACHE_VERSION:
        raise ValueError(f"unsupported frame-presence cache version: {summary.get('version')}")
    preprocessing = summary.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError(f"cache summary has no preprocessing contract: {summary_path}")
    contract = CacheContract(
        input_width=int(preprocessing["input_width"]),
        input_height=int(preprocessing["input_height"]),
        focus_width=int(preprocessing["focus_width"]),
        focus_height=int(preprocessing["focus_height"]),
        heatmap_stride_x=int(preprocessing["heatmap_stride_x"]),
        heatmap_stride_y=int(preprocessing["heatmap_stride_y"]),
        heatmap_width=int(preprocessing["heatmap_width"]),
        heatmap_height=int(preprocessing["heatmap_height"]),
    )
    expected = heatmap_size(
        contract.input_width,
        contract.input_height,
        contract.heatmap_stride_x,
        contract.heatmap_stride_y,
    )
    if expected != (contract.heatmap_width, contract.heatmap_height):
        raise ValueError(f"cache heatmap geometry is inconsistent: {summary_path}")
    return summary, contract


class FramePresenceCacheDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.summary, self.contract = load_cache_contract(self.root)
        self.images = np.load(self.root / IMAGES_FILENAME, mmap_mode="r")
        self.focus = np.load(self.root / FOCUS_FILENAME, mmap_mode="r")
        self.focus_mode = np.load(self.root / FOCUS_MODE_FILENAME, mmap_mode="r")
        self.targets = np.load(self.root / TARGETS_FILENAME, mmap_mode="r")
        self.presence = np.load(self.root / PRESENCE_FILENAME, mmap_mode="r")
        self.bounds = np.load(self.root / BOUNDS_FILENAME, mmap_mode="r")
        count = int(self.summary["samples"])
        expected_shapes = {
            "images": (count, self.contract.input_height, self.contract.input_width),
            "focus": (count, self.contract.focus_height, self.contract.focus_width, 3),
            "focus_mode": (count,),
            "targets": (count, self.contract.heatmap_height, self.contract.heatmap_width),
            "presence": (count,),
            "bounds": (count, 4),
        }
        for name, expected_shape in expected_shapes.items():
            actual = getattr(self, name).shape
            if actual != expected_shape:
                raise ValueError(f"cache {name} shape is {actual}, expected {expected_shape}")
        samples_path = self.root / SAMPLES_FILENAME
        self.records = [
            json.loads(line)
            for line in samples_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(self.records) != count:
            raise ValueError(f"cache sample metadata count is {len(self.records)}, expected {count}")
        y_position = torch.linspace(
            -1.0,
            1.0,
            self.contract.input_height,
            dtype=torch.float32,
        ).view(1, self.contract.input_height, 1).expand(
            1,
            self.contract.input_height,
            self.contract.input_width,
        )
        x_position = torch.linspace(
            -1.0,
            1.0,
            self.contract.input_width,
            dtype=torch.float32,
        ).view(1, 1, self.contract.input_width).expand(
            1,
            self.contract.input_height,
            self.contract.input_width,
        )
        self.position_channels = torch.cat((x_position, y_position), dim=0)

    def __len__(self) -> int:
        return int(self.presence.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        image = torch.from_numpy(np.array(self.images[index], copy=True)).to(torch.float32)
        image = image.unsqueeze(0).mul_(1.0 / 127.5).sub_(1.0)
        image = torch.cat((image, self.position_channels), dim=0)
        focus = torch.from_numpy(np.array(self.focus[index], copy=True)).permute(2, 0, 1)
        padding = (focus == torch.tensor((124, 116, 104)).view(3, 1, 1)).all(dim=0)
        focus = focus.to(torch.float32).mul_(1.0 / 255.0)
        focus = (focus - IMAGENET_MEAN) / IMAGENET_STD
        focus[:, padding] = 0.0
        target = torch.from_numpy(np.array(self.targets[index], copy=True)).to(torch.float32)
        return {
            "image": image,
            "focus": focus,
            "focus_mode": torch.tensor(float(self.focus_mode[index]), dtype=torch.float32),
            "target": target.unsqueeze(0),
            "presence": torch.tensor(float(self.presence[index]), dtype=torch.float32),
            "bounds": torch.from_numpy(np.array(self.bounds[index], copy=True)).to(torch.int64),
            "index": torch.tensor(index, dtype=torch.int64),
        }
