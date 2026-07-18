from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 0


@dataclass(frozen=True)
class SampleBox:
    x: int
    y: int
    width: int
    height: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize subtitle detector training samples from subtitles and background images."
    )
    parser.add_argument("subtitle_file", type=Path, help="Input .srt or plain text subtitle file.")
    parser.add_argument("image_dir", type=Path, help="Directory containing background images.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--count", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation.")
    parser.add_argument("--font", type=Path, help="Optional TrueType/OpenType font file.")
    parser.add_argument("--font-size-min", type=int, default=28, help="Minimum subtitle font size.")
    parser.add_argument("--font-size-max", type=int, default=48, help="Maximum subtitle font size.")
    parser.add_argument("--margin", type=int, default=24, help="Minimum distance from image edges.")
    parser.add_argument(
        "--placement-region",
        help="Optional subtitle placement region as x1,y1,x2,y2. Default uses lower 45%% of the image.",
    )
    parser.add_argument("--boxed-images", action="store_true", help="Also write preview images with boxes.")
    return parser.parse_args(argv)


def parse_subtitle_file(path: Path, text: str | None = None) -> list[str]:
    if text is None:
        text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".srt" or "-->" in text:
        return parse_srt_text(text)
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_srt_text(text: str) -> list[str]:
    subtitles: list[str] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        kept = [
            line
            for line in lines
            if not line.isdigit() and "-->" not in line
        ]
        if kept:
            subtitles.append("\n".join(kept))
    return subtitles


def collect_image_paths(image_dir: Path, paths: Iterable[Path] | None = None) -> list[Path]:
    require_existing_file = paths is None
    if paths is None:
        paths = image_dir.rglob("*")
    return sorted(
        path
        for path in paths
        if path.suffix.lower() in IMAGE_SUFFIXES
        and (not require_existing_file or path.is_file())
    )


def yolo_label_from_box(box: SampleBox, image_width: int, image_height: int) -> str:
    center_x = (box.x + box.width / 2) / image_width
    center_y = (box.y + box.height / 2) / image_height
    width = box.width / image_width
    height = box.height / image_height
    return f"{CLASS_ID} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def sample_stem(index: int) -> str:
    return f"synthetic_{index:06d}"


def parse_region(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--placement-region must use x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--placement-region requires x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def load_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install pillow.") from exc
    return Image, ImageDraw, ImageFont


def default_font_candidates() -> list[Path]:
    return [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]


def load_font(ImageFont: Any, font_path: Path | None, size: int) -> Any:
    candidates = [font_path] if font_path else default_font_candidates()
    for candidate in candidates:
        if candidate and candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def measure_text(draw: Any, text: str, font: Any, stroke_width: int) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0), text, font=font, stroke_width=stroke_width, spacing=6
    )
    return int(right - left), int(bottom - top)


def wrap_subtitle(draw: Any, text: str, font: Any, max_width: int, stroke_width: int) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        units = line.split(" ") if " " in line else list(line)
        current = ""
        separator = " " if " " in line else ""
        for unit in units:
            candidate = unit if not current else f"{current}{separator}{unit}"
            width, _ = measure_text(draw, candidate, font, stroke_width)
            if width <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = unit
        if current:
            lines.append(current)
    return "\n".join(lines)


def choose_box(
    rng: random.Random,
    image_width: int,
    image_height: int,
    text_width: int,
    text_height: int,
    margin: int,
    placement_region: tuple[int, int, int, int] | None,
) -> SampleBox:
    if placement_region is None:
        x1, y1, x2, y2 = margin, int(image_height * 0.55), image_width - margin, image_height - margin
    else:
        x1, y1, x2, y2 = placement_region
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))

    box_width = min(text_width + margin, x2 - x1)
    box_height = min(text_height + margin, y2 - y1)
    min_x = x1
    max_x = max(x1, x2 - box_width)
    min_y = y1
    max_y = max(y1, y2 - box_height)
    return SampleBox(
        x=rng.randint(min_x, max_x),
        y=rng.randint(min_y, max_y),
        width=box_width,
        height=box_height,
    )


def draw_subtitle_sample(
    *,
    image: Any,
    subtitle: str,
    rng: random.Random,
    ImageDraw: Any,
    ImageFont: Any,
    font_path: Path | None,
    font_size_min: int,
    font_size_max: int,
    margin: int,
    placement_region: tuple[int, int, int, int] | None,
) -> tuple[Any, SampleBox, str]:
    width, height = image.size
    font_size = rng.randint(font_size_min, font_size_max)
    stroke_width = max(2, font_size // 12)
    font = load_font(ImageFont, font_path, font_size)
    draw = ImageDraw.Draw(image)
    if placement_region is None:
        max_text_width = max(1, width - margin * 4)
    else:
        region_width = max(1, placement_region[2] - placement_region[0])
        max_text_width = max(1, region_width - margin)
    wrapped = wrap_subtitle(draw, subtitle, font, max_text_width, stroke_width)
    text_width, text_height = measure_text(draw, wrapped, font, stroke_width)
    box = choose_box(
        rng,
        width,
        height,
        text_width,
        text_height,
        margin,
        placement_region,
    )
    text_x = box.x + max(0, (box.width - text_width) // 2)
    text_y = box.y + max(0, (box.height - text_height) // 2)
    draw.multiline_text(
        (text_x, text_y),
        wrapped,
        font=font,
        fill=(255, 255, 255),
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0),
        spacing=6,
        align="center",
    )
    return image, box, wrapped


def synthesize_samples(args: argparse.Namespace) -> int:
    if args.count <= 0:
        raise ValueError("--count must be greater than 0")
    if args.font_size_min <= 0 or args.font_size_max < args.font_size_min:
        raise ValueError("--font-size-max must be >= --font-size-min and both must be positive")
    if args.margin < 0:
        raise ValueError("--margin must be greater than or equal to 0")

    Image, ImageDraw, ImageFont = load_pillow()
    subtitles = parse_subtitle_file(args.subtitle_file)
    if not subtitles:
        raise ValueError(f"subtitle file has no usable subtitle text: {args.subtitle_file}")

    image_paths = collect_image_paths(args.image_dir)
    if not image_paths:
        raise ValueError(f"image directory has no supported images: {args.image_dir}")

    rng = random.Random(args.seed)
    output = args.output
    image_dir = output / "images"
    label_dir = output / "labels"
    boxed_dir = output / "boxed_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    if args.boxed_images:
        boxed_dir.mkdir(parents=True, exist_ok=True)

    placement_region = parse_region(args.placement_region)
    annotations_path = output / "annotations.jsonl"
    with annotations_path.open("w", encoding="utf-8") as annotations:
        for index in range(1, args.count + 1):
            source_image = rng.choice(image_paths)
            subtitle = rng.choice(subtitles)
            with Image.open(source_image) as source:
                image = source.convert("RGB")
            sample_image, box, wrapped = draw_subtitle_sample(
                image=image,
                subtitle=subtitle,
                rng=rng,
                ImageDraw=ImageDraw,
                ImageFont=ImageFont,
                font_path=args.font,
                font_size_min=args.font_size_min,
                font_size_max=args.font_size_max,
                margin=args.margin,
                placement_region=placement_region,
            )

            stem = sample_stem(index)
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            sample_image.save(image_path, quality=92)
            yolo_label = yolo_label_from_box(box, sample_image.width, sample_image.height)
            label_path.write_text(yolo_label + "\n", encoding="utf-8")

            boxed_path = None
            if args.boxed_images:
                boxed = sample_image.copy()
                boxed_draw = ImageDraw.Draw(boxed)
                boxed_draw.rectangle(
                    [box.x, box.y, box.x + box.width, box.y + box.height],
                    outline=(0, 255, 0),
                    width=2,
                )
                boxed_path = boxed_dir / f"{stem}.jpg"
                boxed.save(boxed_path, quality=92)

            annotations.write(
                json.dumps(
                    {
                        "image": str(image_path.as_posix()),
                        "source_image": str(source_image),
                        "subtitle": subtitle,
                        "rendered_subtitle": wrapped,
                        "image_width": sample_image.width,
                        "image_height": sample_image.height,
                        "bbox": {
                            "x": box.x,
                            "y": box.y,
                            "width": box.width,
                            "height": box.height,
                        },
                        "yolo_label": yolo_label,
                        "boxed_image": str(boxed_path.as_posix()) if boxed_path else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"generated {args.count} synthetic samples in {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return synthesize_samples(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
