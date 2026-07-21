from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from subfast_detector.dataset import apply_label_masks, load_label_masks, read_boxes
from subfast_shared.geometry import Box

from .config import FramePresencePreprocessSettings
from .data import (
    BOUNDS_FILENAME,
    CACHE_VERSION,
    FOCUS_FILENAME,
    FOCUS_MODE_FILENAME,
    IMAGES_FILENAME,
    PRESENCE_FILENAME,
    SAMPLES_FILENAME,
    SUMMARY_FILENAME,
    TARGETS_FILENAME,
    heatmap_size,
    rasterize_boxes,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
NORMAL_FOCUS_BOX = (0.16, 0.86, 0.84, 1.0)
WIDE_FOCUS_BOX = (0.12, 0.70, 0.88, 1.0)
LEGACY_FOCUS_BOX = (0.08, 0.88, 0.92, 1.0)


@dataclass(frozen=True)
class SourceSample:
    root: Path
    image_path: Path
    sample_id: str
    width: int
    height: int
    boxes: tuple[Box, ...]


def _annotation_dimensions(root: Path) -> dict[str, tuple[int, int]]:
    path = root / "annotations.jsonl"
    if not path.exists():
        return {}
    dimensions: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            image_name = Path(str(record.get("image", ""))).name
            if image_name and record.get("image_width") and record.get("image_height"):
                dimensions[image_name] = (
                    int(record["image_width"]),
                    int(record["image_height"]),
                )
    return dimensions


def discover_source_samples(root: Path) -> tuple[list[SourceSample], int]:
    root = root.expanduser().resolve()
    image_dir = root / "images"
    label_dir = root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(f"source root must contain images/ and labels/: {root}")
    masks = load_label_masks(root)
    dimensions = _annotation_dimensions(root)
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    samples: list[SourceSample] = []
    dropped = 0
    for image_path in image_paths:
        size = dimensions.get(image_path.name)
        if size is None:
            with Image.open(image_path) as image:
                size = image.size
        width, height = size
        sample_id = image_path.stem
        boxes = read_boxes(label_dir / f"{sample_id}.txt", width, height)
        boxes, _, drop_image = apply_label_masks(
            sample_id,
            boxes,
            masks,
            width,
            height,
        )
        if drop_image:
            dropped += 1
            continue
        samples.append(
            SourceSample(
                root=root,
                image_path=image_path,
                sample_id=sample_id,
                width=width,
                height=height,
                boxes=tuple(boxes),
            )
        )
    return samples, dropped


def _focus_image(
    source: Image.Image,
    *,
    source_width: int,
    source_height: int,
    settings: FramePresencePreprocessSettings,
) -> tuple[np.ndarray, int, tuple[float, float, float, float]]:
    if source_width / source_height > 2.0:
        focus_mode = 1
        box = WIDE_FOCUS_BOX
    elif source_width <= 1280:
        focus_mode = 2
        box = LEGACY_FOCUS_BOX
    else:
        focus_mode = 0
        box = NORMAL_FOCUS_BOX
    left = round(box[0] * source_width)
    top = round(box[1] * source_height)
    right = round(box[2] * source_width)
    bottom = round(box[3] * source_height)
    crop = source.convert("RGB").crop((left, top, right, bottom))
    scale = min(settings.focus_width / crop.width, settings.focus_height / crop.height)
    resized_width = max(1, min(settings.focus_width, round(crop.width * scale)))
    resized_height = max(1, min(settings.focus_height, round(crop.height * scale)))
    resized = crop.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (settings.focus_width, settings.focus_height), (124, 116, 104))
    canvas.paste(
        resized,
        (
            (settings.focus_width - resized_width) // 2,
            (settings.focus_height - resized_height) // 2,
        ),
    )
    return np.asarray(canvas, dtype=np.uint8), focus_mode, box


def _convert_sample(
    sample: SourceSample,
    settings: FramePresencePreprocessSettings,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, dict[str, object]]:
    with Image.open(sample.image_path) as source:
        resized = source.convert("L").resize(
            (settings.input_width, settings.input_height),
            Image.Resampling.BILINEAR,
        )
        image = np.asarray(resized, dtype=np.uint8)
        focus, focus_mode, focus_box = _focus_image(
            source,
            source_width=sample.width,
            source_height=sample.height,
            settings=settings,
        )
    heatmap_width, heatmap_height = heatmap_size(
        settings.input_width,
        settings.input_height,
        settings.heatmap_stride_x,
        settings.heatmap_stride_y,
    )
    box_values = [(box.x1, box.y1, box.x2, box.y2) for box in sample.boxes]
    target, bounds = rasterize_boxes(
        box_values,
        source_width=sample.width,
        source_height=sample.height,
        heatmap_width=heatmap_width,
        heatmap_height=heatmap_height,
    )
    if sample.boxes:
        union = [
            min(box.x1 for box in sample.boxes) / sample.width,
            min(box.y1 for box in sample.boxes) / sample.height,
            max(box.x2 for box in sample.boxes) / sample.width,
            max(box.y2 for box in sample.boxes) / sample.height,
        ]
    else:
        union = None
    record: dict[str, object] = {
        "key": f"{sample.root.name}/{sample.sample_id}",
        "root": str(sample.root),
        "sample_id": sample.sample_id,
        "image": str(sample.image_path),
        "source_size": [sample.width, sample.height],
        "focus_mode": focus_mode,
        "focus_box_normalized": focus_box,
        "presence": bool(sample.boxes),
        "source_union_box_normalized": union,
        "heatmap_bounds": bounds.tolist() if sample.boxes else None,
        "heatmap_target_area": int(target.sum()),
    }
    return image, focus, focus_mode, target, bounds, record


def prepare_cache(
    source_roots: list[Path],
    output: Path,
    *,
    settings: FramePresencePreprocessSettings,
    workers: int,
    overwrite: bool,
) -> dict[str, object]:
    if (settings.heatmap_stride_y, settings.heatmap_stride_x) != (2, 4):
        raise ValueError("the current frame-presence architecture requires heatmap stride 4x2")
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError(f"output cache is not empty; pass --overwrite to replace it: {output}")
    output.mkdir(parents=True, exist_ok=True)
    known_files = (
        IMAGES_FILENAME,
        FOCUS_FILENAME,
        FOCUS_MODE_FILENAME,
        TARGETS_FILENAME,
        PRESENCE_FILENAME,
        BOUNDS_FILENAME,
        SAMPLES_FILENAME,
        SUMMARY_FILENAME,
    )
    if overwrite:
        for name in known_files:
            (output / name).unlink(missing_ok=True)
            (output / f"{name}.partial").unlink(missing_ok=True)

    samples: list[SourceSample] = []
    root_summary: dict[str, dict[str, int]] = {}
    for source_root in source_roots:
        root_samples, dropped = discover_source_samples(source_root)
        positive = sum(bool(sample.boxes) for sample in root_samples)
        root_summary[str(source_root.expanduser().resolve())] = {
            "samples": len(root_samples),
            "positive": positive,
            "empty": len(root_samples) - positive,
            "dropped": dropped,
        }
        samples.extend(root_samples)
    if not samples:
        raise ValueError("no eligible source samples found")

    heatmap_width, heatmap_height = heatmap_size(
        settings.input_width,
        settings.input_height,
        settings.heatmap_stride_x,
        settings.heatmap_stride_y,
    )
    count = len(samples)
    partial_paths = {
        name: output / f"{name}.partial"
        for name in (
            IMAGES_FILENAME,
            FOCUS_FILENAME,
            FOCUS_MODE_FILENAME,
            TARGETS_FILENAME,
            PRESENCE_FILENAME,
            BOUNDS_FILENAME,
        )
    }
    images = np.lib.format.open_memmap(
        partial_paths[IMAGES_FILENAME],
        mode="w+",
        dtype=np.uint8,
        shape=(count, settings.input_height, settings.input_width),
    )
    focus = np.lib.format.open_memmap(
        partial_paths[FOCUS_FILENAME],
        mode="w+",
        dtype=np.uint8,
        shape=(count, settings.focus_height, settings.focus_width, 3),
    )
    focus_mode = np.lib.format.open_memmap(
        partial_paths[FOCUS_MODE_FILENAME],
        mode="w+",
        dtype=np.uint8,
        shape=(count,),
    )
    targets = np.lib.format.open_memmap(
        partial_paths[TARGETS_FILENAME],
        mode="w+",
        dtype=np.uint8,
        shape=(count, heatmap_height, heatmap_width),
    )
    presence = np.lib.format.open_memmap(
        partial_paths[PRESENCE_FILENAME],
        mode="w+",
        dtype=np.uint8,
        shape=(count,),
    )
    bounds = np.lib.format.open_memmap(
        partial_paths[BOUNDS_FILENAME],
        mode="w+",
        dtype=np.int16,
        shape=(count, 4),
    )
    records_path = output / f"{SAMPLES_FILENAME}.partial"
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor, records_path.open(
            "w", encoding="utf-8"
        ) as records_file:
            converted = executor.map(
                lambda sample: _convert_sample(sample, settings),
                samples,
            )
            for index, converted_sample in enumerate(
                tqdm(converted, total=count, desc="prepare frame-presence cache")
            ):
                image, sample_focus, sample_focus_mode, target, sample_bounds, record = (
                    converted_sample
                )
                images[index] = image
                focus[index] = sample_focus
                focus_mode[index] = sample_focus_mode
                targets[index] = target
                presence[index] = int(bool(record["presence"]))
                bounds[index] = sample_bounds
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        for array in (images, focus, focus_mode, targets, presence, bounds):
            array.flush()
        del images, focus, focus_mode, targets, presence, bounds
        for name, partial_path in partial_paths.items():
            partial_path.replace(output / name)
        records_path.replace(output / SAMPLES_FILENAME)
    except BaseException:
        for partial_path in (*partial_paths.values(), records_path):
            partial_path.unlink(missing_ok=True)
        raise

    positive_count = sum(bool(sample.boxes) for sample in samples)
    summary: dict[str, object] = {
        "version": CACHE_VERSION,
        "kind": "subfast_frame_presence_cache",
        "samples": count,
        "positive": positive_count,
        "empty": count - positive_count,
        "source_roots": root_summary,
        "preprocessing": {
            "input_kind": "full_frame_luma_cached_coordinates_added_by_dataset",
            "normalization": "uint8_to_minus_one_plus_one",
            "input_width": settings.input_width,
            "input_height": settings.input_height,
            "focus_width": settings.focus_width,
            "focus_height": settings.focus_height,
            "focus_boxes_normalized": {
                "normal": NORMAL_FOCUS_BOX,
                "wide": WIDE_FOCUS_BOX,
                "legacy": LEGACY_FOCUS_BOX,
            },
            "heatmap_stride_x": settings.heatmap_stride_x,
            "heatmap_stride_y": settings.heatmap_stride_y,
            "heatmap_width": heatmap_width,
            "heatmap_height": heatmap_height,
            "resize_mode": "stretch",
            "interpolation": "bilinear",
        },
        "files": {
            IMAGES_FILENAME: os.path.getsize(output / IMAGES_FILENAME),
            FOCUS_FILENAME: os.path.getsize(output / FOCUS_FILENAME),
            FOCUS_MODE_FILENAME: os.path.getsize(output / FOCUS_MODE_FILENAME),
            TARGETS_FILENAME: os.path.getsize(output / TARGETS_FILENAME),
            PRESENCE_FILENAME: os.path.getsize(output / PRESENCE_FILENAME),
            BOUNDS_FILENAME: os.path.getsize(output / BOUNDS_FILENAME),
        },
    }
    summary_partial = output / f"{SUMMARY_FILENAME}.partial"
    summary_partial.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_partial.replace(output / SUMMARY_FILENAME)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = FramePresencePreprocessSettings()
    parser = argparse.ArgumentParser(
        description="Build a memory-mapped full-frame cache for subfast_frame_presence."
    )
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-width", type=int, default=defaults.input_width)
    parser.add_argument("--input-height", type=int, default=defaults.input_height)
    parser.add_argument("--focus-width", type=int, default=defaults.focus_width)
    parser.add_argument("--focus-height", type=int, default=defaults.focus_height)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    settings = FramePresencePreprocessSettings(
        input_width=args.input_width,
        input_height=args.input_height,
        focus_width=args.focus_width,
        focus_height=args.focus_height,
    )
    summary = prepare_cache(
        args.source_root,
        args.output,
        settings=settings,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
