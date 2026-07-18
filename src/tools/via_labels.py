from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


CLASS_NAME = "subtitle"
CLASS_ID = 0
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_annotation_dimensions(path: Path | None) -> dict[str, tuple[int, int]]:
    if path is None or not path.exists():
        return {}

    dimensions: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image = Path(item["image"]).name
            size = (int(item["image_width"]), int(item["image_height"]))
            # Keep both keys so annotations generated with a relative path or
            # a different image suffix still resolve for a label stem.
            dimensions[image] = size
            dimensions[Path(image).stem] = size
    return dimensions


def find_image(images_dir: Path, stem: str) -> Path:
    """Return the existing image for a label stem, preferring common formats."""

    for suffix in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    # Preserve the historical filename in VIA output when dimensions come
    # from annotations or explicit defaults and the image is not available.
    return images_dir / f"{stem}.jpg"


def image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index < len(data):
            while index < len(data) and data[index] == 0xFF:
                index += 1
            marker = data[index]
            index += 1
            if marker in (0xD8, 0xD9):
                continue
            segment_len = struct.unpack(">H", data[index : index + 2])[0]
            if 0xC0 <= marker <= 0xC3:
                height, width = struct.unpack(">HH", data[index + 3 : index + 7])
                return int(width), int(height)
            index += segment_len
    raise ValueError(f"Unsupported or invalid image file: {path}")


def resolve_dimensions(
    image_path: Path,
    annotation_dimensions: dict[str, tuple[int, int]],
    default_width: int | None,
    default_height: int | None,
) -> tuple[int, int]:
    if image_path.name in annotation_dimensions:
        return annotation_dimensions[image_path.name]
    if image_path.stem in annotation_dimensions:
        return annotation_dimensions[image_path.stem]
    if image_path.exists():
        return image_size(image_path)
    if default_width is not None and default_height is not None:
        return default_width, default_height
    raise ValueError(
        f"Missing dimensions for {image_path.name}; provide images, annotations.jsonl, or --image-width/--image-height."
    )


def yolo_to_rect(
    line: str, image_width: int, image_height: int
) -> dict[str, int | str]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label line: {line}")

    _, center_x, center_y, width, height = parts
    box_width = float(width) * image_width
    box_height = float(height) * image_height
    x = float(center_x) * image_width - box_width / 2
    y = float(center_y) * image_height - box_height / 2

    return {
        "name": "rect",
        "x": round(x),
        "y": round(y),
        "width": round(box_width),
        "height": round(box_height),
    }


def rect_to_yolo(rect: dict[str, Any], image_width: int, image_height: int) -> str:
    x = float(rect["x"])
    y = float(rect["y"])
    width = float(rect["width"])
    height = float(rect["height"])
    center_x = (x + width / 2) / image_width
    center_y = (y + height / 2) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return (
        f"{CLASS_ID} {center_x:.6f} {center_y:.6f} "
        f"{norm_width:.6f} {norm_height:.6f}"
    )


def labels_to_via(
    labels_dir: Path,
    images_dir: Path,
    annotations_path: Path | None = None,
    default_width: int | None = None,
    default_height: int | None = None,
) -> dict[str, Any]:
    annotation_dimensions = load_annotation_dimensions(annotations_path)
    via: dict[str, Any] = {}

    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = find_image(images_dir, label_path.stem)
        width, height = resolve_dimensions(
            image_path, annotation_dimensions, default_width, default_height
        )
        regions = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            regions.append(
                {
                    "shape_attributes": yolo_to_rect(line, width, height),
                    "region_attributes": {"class": CLASS_NAME},
                }
            )

        size = image_path.stat().st_size if image_path.exists() else 0
        filename = image_path.name
        via[f"{filename}{size}"] = {
            "filename": filename,
            "size": size,
            "regions": regions,
            "file_attributes": {"width": width, "height": height},
        }

    return via


def via_to_labels(via: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in via.values():
        filename = item["filename"]
        file_attributes = item.get("file_attributes", {})
        width = int(file_attributes.get("width", 0))
        height = int(file_attributes.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"Missing file_attributes width/height for {filename}")

        lines = []
        for region in item.get("regions", []):
            rect = region.get("shape_attributes", {})
            if rect.get("name") != "rect":
                continue
            lines.append(rect_to_yolo(rect, width, height))

        label_path = output_dir / f"{Path(filename).stem}.txt"
        content = "\n".join(lines)
        label_path.write_text((content + "\n") if content else "", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO subtitle labels to/from VGG Image Annotator JSON."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    to_via = subparsers.add_parser("labels-to-via")
    to_via.add_argument("--labels-dir", type=Path, required=True)
    to_via.add_argument("--images-dir", type=Path, required=True)
    to_via.add_argument("--annotations", type=Path)
    to_via.add_argument("--image-width", type=int)
    to_via.add_argument("--image-height", type=int)
    to_via.add_argument("-o", "--output", type=Path, required=True)

    to_labels = subparsers.add_parser("via-to-labels")
    to_labels.add_argument("via_json", type=Path)
    to_labels.add_argument("--labels-dir", type=Path, required=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "labels-to-via":
        via = labels_to_via(
            labels_dir=args.labels_dir,
            images_dir=args.images_dir,
            annotations_path=args.annotations,
            default_width=args.image_width,
            default_height=args.image_height,
        )
        args.output.write_text(
            json.dumps(via, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif args.command == "via-to-labels":
        via = json.loads(args.via_json.read_text(encoding="utf-8"))
        via_to_labels(via, args.labels_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
