from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from review_generated_labels import LabelBox, parse_label_file


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
PADDLEOCR_RECOGNITION_MODEL = "PP-OCRv6_medium_rec"


@dataclass(frozen=True)
class SourceSample:
    stem: str
    image_path: Path
    label_path: Path
    boxes: list[LabelBox]
    width: int
    height: int
    annotation: dict[str, Any]


@dataclass(frozen=True)
class PixelBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-size subtitle ROI samples from an existing training sample directory."
    )
    parser.add_argument("samples_dir", type=Path, help="Input directory containing images/ and labels/.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output ROI dataset directory.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep empty-subtitle images by cropping the common ROI anchor.",
    )
    parser.add_argument(
        "--copy-labels",
        action="store_true",
        help="Also write a labels/ directory with one full-image ROI label for positive samples.",
    )
    parser.add_argument(
        "--include-dropped-images",
        action="store_true",
        help="Include images marked with __image__.drop_image in label_masks.json. By default they are skipped.",
    )
    return parser.parse_args(argv)


def load_annotations(samples_dir: Path) -> dict[str, dict[str, Any]]:
    path = samples_dir / "annotations.jsonl"
    if not path.exists():
        return {}
    items: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_name = Path(str(item.get("image", ""))).name
            if image_name:
                items[Path(image_name).stem] = item
    return items


def load_label_masks(samples_dir: Path) -> dict[str, dict[str, dict[str, object]]]:
    path = samples_dir / "label_masks.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("items", {}))


def label_is_masked(stem: str, label_index: int, masks: dict[str, dict[str, dict[str, object]]]) -> bool:
    marker = masks.get(stem, {}).get(str(label_index), {})
    return bool(
        marker.get("masked")
        or marker.get("deleted")
        or marker.get("unreliable")
        or marker.get("exclude_from_loss")
    )


def image_is_dropped(stem: str, masks: dict[str, dict[str, dict[str, object]]]) -> bool:
    marker = masks.get(stem, {}).get("__image__", {})
    return bool(marker.get("drop_image"))


def discover_samples(samples_dir: Path, skip_dropped_images: bool = True) -> list[SourceSample]:
    images_dir = samples_dir / "images"
    labels_dir = samples_dir / "labels"
    if not images_dir.is_dir():
        raise ValueError(f"missing images dir: {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"missing labels dir: {labels_dir}")

    annotations = load_annotations(samples_dir)
    masks = load_label_masks(samples_dir)
    samples: list[SourceSample] = []
    for image_path in sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ):
        with Image.open(image_path) as image:
            width, height = image.size
        stem = image_path.stem
        if skip_dropped_images and image_is_dropped(stem, masks):
            continue
        label_path = labels_dir / f"{stem}.txt"
        boxes = [
            label
            for label in parse_label_file(label_path)
            if not label_is_masked(stem, label.index, masks)
        ]
        samples.append(
            SourceSample(
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                boxes=boxes,
                width=width,
                height=height,
                annotation=annotations.get(stem, {}),
            )
        )
    return samples


def label_to_pixel_box(label: LabelBox, image_width: int, image_height: int) -> PixelBox:
    rect = label.to_rect(image_width, image_height)
    x1 = int(max(0, min(image_width, round(float(rect["x"])))))
    y1 = int(max(0, min(image_height, round(float(rect["y"])))))
    x2 = int(max(x1, min(image_width, round(float(rect["x"]) + float(rect["width"])))))
    y2 = int(max(y1, min(image_height, round(float(rect["y"]) + float(rect["height"])))))
    return PixelBox(x1, y1, x2, y2)


def union_pixel_box(sample: SourceSample) -> PixelBox | None:
    boxes = [label_to_pixel_box(label, sample.width, sample.height) for label in sample.boxes]
    boxes = [box for box in boxes if box.width > 0 and box.height > 0]
    if not boxes:
        return None
    return PixelBox(
        x1=min(box.x1 for box in boxes),
        y1=min(box.y1 for box in boxes),
        x2=max(box.x2 for box in boxes),
        y2=max(box.y2 for box in boxes),
    )


def common_roi_size(samples: list[SourceSample]) -> tuple[int, int]:
    boxes = [box for sample in samples if (box := union_pixel_box(sample)) is not None]
    if not boxes:
        raise ValueError("no subtitle labels found")
    x1 = min(box.x1 for box in boxes)
    y1 = min(box.y1 for box in boxes)
    x2 = max(box.x2 for box in boxes)
    y2 = max(box.y2 for box in boxes)
    return x2 - x1, y2 - y1


def common_roi_anchor(samples: list[SourceSample]) -> tuple[int, int]:
    boxes = [box for sample in samples if (box := union_pixel_box(sample)) is not None]
    if not boxes:
        raise ValueError("no subtitle labels found")
    x1 = min(box.x1 for box in boxes)
    y1 = min(box.y1 for box in boxes)
    x2 = max(box.x2 for box in boxes)
    y2 = max(box.y2 for box in boxes)
    return (x1 + x2) // 2, (y1 + y2) // 2


def crop_box_for_sample(
    sample: SourceSample,
    roi_width: int,
    roi_height: int,
    anchor: tuple[int, int],
) -> PixelBox | None:
    center_x, center_y = anchor

    x1 = max(0, min(sample.width - roi_width, center_x - roi_width // 2))
    y1 = max(0, min(sample.height - roi_height, center_y - roi_height // 2))
    x2 = min(sample.width, x1 + roi_width)
    y2 = min(sample.height, y1 + roi_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return PixelBox(x1, y1, x2, y2)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def create_text_recognizer(model_name: str) -> Any:
    try:
        from paddleocr import TextRecognition  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install paddleocr and paddlepaddle.") from exc
    return TextRecognition(model_name=model_name)


def extract_recognized_text(result: Any) -> str:
    if result is None:
        return ""
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, dict):
        value = result.get("rec_texts")
        if isinstance(value, list):
            return "".join(str(part) for part in value if str(part))
        for key in ("rec_text", "text", "label"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        for value in result.values():
            text = extract_recognized_text(value)
            if text:
                return text
    if isinstance(result, list):
        parts = [extract_recognized_text(item) for item in result]
        return "".join(part for part in parts if part)
    if isinstance(result, tuple):
        parts = [extract_recognized_text(item) for item in result]
        return "".join(part for part in parts if part)
    if isinstance(result, str):
        return result
    return ""


def recognize_image_text(recognizer: Any, image_path: Path) -> str:
    return extract_recognized_text(recognizer.predict(str(image_path)))


def boxes_are_on_same_text_line(left: PixelBox, right: PixelBox) -> bool:
    vertical_overlap = min(left.y2, right.y2) - max(left.y1, right.y1)
    min_height = max(1, min(left.height, right.height))
    if vertical_overlap > 0 and vertical_overlap / min_height >= 0.45:
        return True
    left_center_y = (left.y1 + left.y2) / 2
    right_center_y = (right.y1 + right.y2) / 2
    return abs(left_center_y - right_center_y) <= max(4.0, min_height * 0.35)


def reading_order_boxes(boxes: list[PixelBox]) -> list[PixelBox]:
    rows: list[list[PixelBox]] = []
    for box in sorted(boxes, key=lambda item: ((item.y1 + item.y2) / 2, item.x1)):
        for row in rows:
            row_bounds = PixelBox(
                min(item.x1 for item in row),
                min(item.y1 for item in row),
                max(item.x2 for item in row),
                max(item.y2 for item in row),
            )
            if boxes_are_on_same_text_line(row_bounds, box):
                row.append(box)
                break
        else:
            rows.append([box])

    ordered: list[PixelBox] = []
    for row in sorted(rows, key=lambda items: min(item.y1 for item in items)):
        ordered.extend(sorted(row, key=lambda item: item.x1))
    return ordered


def recognize_sample_text(recognizer: Any, sample: SourceSample, boxes: list[PixelBox]) -> str:
    ordered_boxes = reading_order_boxes(boxes)
    parts: list[str] = []
    with Image.open(sample.image_path) as source, tempfile.TemporaryDirectory() as tmp_dir:
        source = source.convert("RGB")
        tmp_path = Path(tmp_dir)
        for index, box in enumerate(ordered_boxes):
            crop_path = tmp_path / f"{sample.stem}_{index}.jpg"
            source.crop((box.x1, box.y1, box.x2, box.y2)).save(crop_path, quality=95)
            text = recognize_image_text(recognizer, crop_path)
            if text:
                parts.append(text)
    return "".join(parts)


def write_full_roi_label(path: Path) -> None:
    path.write_text("0 0.500000 0.500000 1.000000 1.000000\n", encoding="utf-8")


def save_fixed_roi_image(
    *,
    source_path: Path,
    output_path: Path,
    crop_box: PixelBox,
    roi_width: int,
    roi_height: int,
) -> None:
    with Image.open(source_path) as image:
        crop = image.convert("RGB").crop((crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2))
    canvas = Image.new("RGB", (roi_width, roi_height), (0, 0, 0))
    canvas.paste(crop, (0, 0))
    canvas.save(output_path, quality=95)


def prepare_roi_samples(args: argparse.Namespace) -> int:
    samples_dir = args.samples_dir.resolve()
    output = args.output.resolve()
    samples = discover_samples(samples_dir, skip_dropped_images=not args.include_dropped_images)
    roi_width, roi_height = common_roi_size(samples)
    roi_anchor = common_roi_anchor(samples)

    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)

    try:
        recognizer = create_text_recognizer(PADDLEOCR_RECOGNITION_MODEL)
    except Exception as exc:
        recognizer = None
        print(f"warning: OCR unavailable, using source label boxes for subtitle presence: {exc}", file=sys.stderr)
    annotations_path = output / "annotations.jsonl"
    kept = 0
    positive = 0
    empty = 0
    annotation_rows: list[dict[str, Any]] = []

    for sample in samples:
        has_label_box = bool(sample.boxes)
        if not has_label_box and not args.keep_empty:
            continue
        crop_box = crop_box_for_sample(sample, roi_width, roi_height, roi_anchor)
        if crop_box is None:
            continue

        subtitle_boxes = [
            label_to_pixel_box(label, sample.width, sample.height)
            for label in sample.boxes
        ]
        subtitle_boxes = [box for box in subtitle_boxes if box.width > 0 and box.height > 0]
        subtitle_boxes = reading_order_boxes(subtitle_boxes)

        output_image = images_dir / f"{sample.stem}.jpg"
        save_fixed_roi_image(
            source_path=sample.image_path,
            output_path=output_image,
            crop_box=crop_box,
            roi_width=roi_width,
            roi_height=roi_height,
        )

        ocr_text = ""
        if recognizer is not None:
            try:
                ocr_text = recognize_sample_text(recognizer, sample, subtitle_boxes)
            except Exception as exc:
                print(f"warning: OCR failed for {sample.stem}, continuing without OCR text: {exc}", file=sys.stderr)
        if recognizer is not None:
            has_subtitle = bool(normalize_text(ocr_text))
            presence_method = "source_label_box_ocr"
        else:
            has_subtitle = has_label_box
            presence_method = "source_label_box_ocr_unavailable"

        if args.copy_labels:
            label_path = labels_dir / f"{sample.stem}.txt"
            if has_subtitle:
                write_full_roi_label(label_path)
            else:
                label_path.write_text("", encoding="utf-8")

        annotation_rows.append(
            {
                "image": str(output_image.relative_to(output).as_posix()),
                "source_image": str(sample.image_path),
                "source_sample_id": sample.stem,
                "source_annotation": sample.annotation,
                "image_width": roi_width,
                "image_height": roi_height,
                "roi_size": [roi_width, roi_height],
                "source_roi": [crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2],
                "source_subtitle_boxes": [[box.x1, box.y1, box.x2, box.y2] for box in subtitle_boxes],
                "has_subtitle": has_subtitle,
                "subtitle_presence_method": presence_method,
                "ocr_text": ocr_text,
                "ocr_text_normalized": normalize_text(ocr_text),
            }
        )
        kept += 1
        positive += int(has_subtitle)
        empty += int(not has_subtitle)

    with annotations_path.open("w", encoding="utf-8") as annotations:
        for row in annotation_rows:
            annotations.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "version": 1,
        "source_samples_dir": str(samples_dir),
        "roi_size": [roi_width, roi_height],
        "samples": kept,
        "positive": positive,
        "empty": empty,
        "kept_empty": bool(args.keep_empty),
        "subtitle_presence_method": "source_label_box_ocr" if recognizer is not None else "source_label_box_ocr_unavailable",
        "paddleocr_model": PADDLEOCR_RECOGNITION_MODEL,
        "ocr_available": recognizer is not None,
        "manual_segment_labeling_required": True,
        "annotations": str(annotations_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {kept} ROI samples to {output}")
    print(f"roi_size={roi_width}x{roi_height} positive={positive} empty={empty}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return prepare_roi_samples(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
