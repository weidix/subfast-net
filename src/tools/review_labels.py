from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
MASKS_FILENAME = "label_masks.json"
IMAGE_MASK_KEY = "__image__"
ADD_BOX_PREFIX = "__add_"
DEFAULT_MERGE_GAP_RATIO = 0.65
DEFAULT_MERGE_HEIGHT_TOLERANCE = 0.22
DEFAULT_MERGE_CENTER_TOLERANCE = 0.45


@dataclass(frozen=True)
class LabelBox:
    index: int
    class_id: str
    cx: float
    cy: float
    width: float
    height: float
    raw: str

    def to_rect(self, image_width: int, image_height: int) -> dict[str, float | int | str]:
        width = self.width * image_width
        height = self.height * image_height
        x = self.cx * image_width - width / 2
        y = self.cy * image_height - height / 2
        return {
            "index": self.index,
            "class_id": self.class_id,
            "x": round(x, 2),
            "y": round(y, 2),
            "width": round(width, 2),
            "height": round(height, 2),
            "center_y": round(y + height / 2, 2),
            "area": round(width * height, 2),
            "aspect": round(width / height, 4) if height else 0,
            "raw": self.raw,
        }


def image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        import struct

        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if data.startswith(b"\xff\xd8"):
        import struct

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
    raise ValueError(f"unsupported or invalid image file: {path}")


def load_annotation_dimensions(path: Path) -> dict[str, tuple[int, int]]:
    if not path.exists():
        return {}
    dimensions: dict[str, tuple[int, int]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            dimensions[Path(item["image"]).name] = (
                int(item["image_width"]),
                int(item["image_height"]),
            )
    return dimensions


def load_masks(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("items", {}))


def save_masks(path: Path, items: dict[str, dict[str, dict[str, object]]]) -> None:
    payload = {
        "version": 1,
        "description": "Manual subtitle label suppression markers. Original label txt files are not modified.",
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_label_file(path: Path) -> list[LabelBox]:
    if not path.exists():
        return []
    boxes = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"invalid YOLO label line in {path}: {raw}")
        class_id, cx, cy, width, height = parts
        boxes.append(
            LabelBox(
                index=index,
                class_id=class_id,
                cx=float(cx),
                cy=float(cy),
                width=float(width),
                height=float(height),
                raw=raw,
            )
        )
    return boxes


def write_label_box(path: Path, label_index: int, rect: dict[str, float], image_width: int, image_height: int) -> None:
    if not path.exists():
        raise ValueError(f"missing label file: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if label_index < 0 or label_index >= len(lines):
        raise ValueError(f"label index out of range in {path}: {label_index}")
    parts = lines[label_index].strip().split()
    if len(parts) != 5:
        raise ValueError(f"invalid YOLO label line in {path}: {lines[label_index].strip()}")

    x = max(0.0, min(float(rect["x"]), float(image_width - 1)))
    y = max(0.0, min(float(rect["y"]), float(image_height - 1)))
    width = max(1.0, min(float(rect["width"]), float(image_width) - x))
    height = max(1.0, min(float(rect["height"]), float(image_height) - y))
    cx = (x + width / 2) / image_width
    cy = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height

    lines[label_index] = (
        f"{parts[0]} {cx:.6f} {cy:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_label_box(path: Path, rect: dict[str, float], image_width: int, image_height: int, class_id: str = "0") -> LabelBox:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    index = len(lines)
    x = max(0.0, min(float(rect["x"]), float(image_width - 1)))
    y = max(0.0, min(float(rect["y"]), float(image_height - 1)))
    width = max(1.0, min(float(rect["width"]), float(image_width) - x))
    height = max(1.0, min(float(rect["height"]), float(image_height) - y))
    cx = (x + width / 2) / image_width
    cy = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    raw = f"{class_id} {cx:.6f} {cy:.6f} {normalized_width:.6f} {normalized_height:.6f}"
    lines.append(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return LabelBox(index, class_id, cx, cy, normalized_width, normalized_height, raw)


def union_rect(boxes: list[dict[str, object]]) -> dict[str, float]:
    x1 = min(float(box["x"]) for box in boxes)
    y1 = min(float(box["y"]) for box in boxes)
    x2 = max(float(box["x"]) + float(box["width"]) for box in boxes)
    y2 = max(float(box["y"]) + float(box["height"]) for box in boxes)
    return {
        "x": round(x1, 2),
        "y": round(y1, 2),
        "width": round(x2 - x1, 2),
        "height": round(y2 - y1, 2),
    }


def merge_label_boxes(
    path: Path,
    label_indices: list[int],
    rect: dict[str, float],
    image_width: int,
    image_height: int,
) -> int:
    if len(label_indices) < 2:
        raise ValueError("merge requires at least two label indexes")
    keep_index = min(label_indices)
    write_label_box(path, keep_index, rect, image_width, image_height)
    return keep_index


def merge_candidate_groups(
    boxes: list[dict[str, object]],
    gap_ratio: float = DEFAULT_MERGE_GAP_RATIO,
    height_tolerance: float = DEFAULT_MERGE_HEIGHT_TOLERANCE,
    center_tolerance: float = DEFAULT_MERGE_CENTER_TOLERANCE,
) -> list[dict[str, object]]:
    active_boxes = [box for box in boxes if not bool(box.get("masked", False))]
    if len(active_boxes) < 2:
        return []

    parent = {int(box["index"]): int(box["index"]) for box in active_boxes}
    by_index = {int(box["index"]): box for box in active_boxes}

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            next_index = parent[index]
            parent[index] = root
            index = next_index
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    def close_enough(left: dict[str, object], right: dict[str, object]) -> bool:
        left_height = float(left["height"])
        right_height = float(right["height"])
        avg_height = (left_height + right_height) / 2
        if avg_height <= 0:
            return False
        height_delta = abs(left_height - right_height) / avg_height
        left_center = float(left["y"]) + left_height / 2
        right_center = float(right["y"]) + right_height / 2
        center_delta = abs(left_center - right_center) / avg_height
        left_x1 = float(left["x"])
        left_x2 = left_x1 + float(left["width"])
        right_x1 = float(right["x"])
        right_x2 = right_x1 + float(right["width"])
        horizontal_gap = max(0.0, max(left_x1, right_x1) - min(left_x2, right_x2))
        return (
            height_delta <= height_tolerance
            and center_delta <= center_tolerance
            and horizontal_gap <= avg_height * gap_ratio
        )

    for left_pos, left in enumerate(active_boxes):
        for right in active_boxes[left_pos + 1 :]:
            if close_enough(left, right):
                union(int(left["index"]), int(right["index"]))

    grouped: dict[int, list[dict[str, object]]] = {}
    for index, box in by_index.items():
        grouped.setdefault(find(index), []).append(box)

    candidates = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda box: (float(box["x"]), int(box["index"])))
        rect = union_rect(ordered)
        candidates.append(
            {
                "indices": [int(box["index"]) for box in ordered],
                "rect": {
                    **rect,
                    "area": round(rect["width"] * rect["height"], 2),
                    "aspect": round(rect["width"] / rect["height"], 4) if rect["height"] else 0,
                },
            }
        )
    return sorted(candidates, key=lambda item: item["indices"][0])


def filter_matches(box: dict[str, object], query: dict[str, str]) -> bool:
    area_lt = parse_float(query.get("area_lt"))
    area_gt = parse_float(query.get("area_gt"))
    width_lt = parse_float(query.get("width_lt"))
    width_gt = parse_float(query.get("width_gt"))
    height_lt = parse_float(query.get("height_lt"))
    height_gt = parse_float(query.get("height_gt"))
    y_lt = parse_float(query.get("y_lt"))
    y_gt = parse_float(query.get("y_gt"))
    checks = []
    if area_lt is not None:
        checks.append(float(box["area"]) < area_lt)
    if area_gt is not None:
        checks.append(float(box["area"]) > area_gt)
    if width_lt is not None:
        checks.append(float(box["width"]) < width_lt)
    if width_gt is not None:
        checks.append(float(box["width"]) > width_gt)
    if height_lt is not None:
        checks.append(float(box["height"]) < height_lt)
    if height_gt is not None:
        checks.append(float(box["height"]) > height_gt)
    if y_lt is not None:
        checks.append(float(box["center_y"]) < y_lt)
    if y_gt is not None:
        checks.append(float(box["center_y"]) > y_gt)
    return all(checks) if checks else True


def has_box_filter(query: dict[str, str]) -> bool:
    return any(
        query.get(name) not in (None, "")
        for name in ("area_lt", "area_gt", "width_lt", "width_gt", "height_lt", "height_gt", "y_lt", "y_gt")
    )


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_float_or_default(value: str | None, default: float) -> float:
    parsed = parse_float(value)
    return default if parsed is None else parsed


def merge_params_from_query(query: dict[str, str]) -> dict[str, float]:
    return {
        "gap_ratio": parse_float_or_default(query.get("merge_gap_ratio"), DEFAULT_MERGE_GAP_RATIO),
        "height_tolerance": parse_float_or_default(
            query.get("merge_height_tolerance"), DEFAULT_MERGE_HEIGHT_TOLERANCE
        ),
        "center_tolerance": parse_float_or_default(
            query.get("merge_center_tolerance"), DEFAULT_MERGE_CENTER_TOLERANCE
        ),
    }


class ReviewApp:
    def __init__(self, samples_dir: Path) -> None:
        self.samples_dir = samples_dir.resolve()
        self.images_dir = self.samples_dir / "images"
        self.labels_dir = self.samples_dir / "labels"
        self.annotations_path = self.samples_dir / "annotations.jsonl"
        self.masks_path = self.samples_dir / MASKS_FILENAME
        if not self.images_dir.is_dir():
            raise ValueError(f"missing images dir: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise ValueError(f"missing labels dir: {self.labels_dir}")
        self.dimensions = load_annotation_dimensions(self.annotations_path)
        self.masks = load_masks(self.masks_path)

    def image_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def state(self, query: dict[str, str]) -> dict[str, object]:
        include_empty = query.get("include_empty") == "1"
        only_empty = query.get("only_empty") == "1"
        only_dropped = query.get("only_dropped") == "1"
        show_masked = query.get("show_masked", "1") == "1"
        samples = []
        total_boxes = 0
        masked_boxes = 0
        dropped_images = 0

        for image_path in self.image_paths():
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            width, height = self.dimensions.get(image_path.name) or image_size(image_path)
            image_marker = self.masks.get(image_path.stem, {}).get(IMAGE_MASK_KEY, {})
            drop_image = bool(image_marker.get("drop_image", False))
            if drop_image:
                dropped_images += 1
            if only_dropped and not drop_image:
                continue
            label_boxes = parse_label_file(label_path)
            empty_label = len(label_boxes) == 0
            boxes = []
            for label_box in label_boxes:
                rect = label_box.to_rect(width, height)
                marker = self.masks.get(image_path.stem, {}).get(str(label_box.index), {})
                rect["masked"] = bool(marker.get("masked", False))
                rect["reason"] = marker.get("reason", "")
                rect["matched"] = filter_matches(rect, query)
                total_boxes += 1
                if rect["masked"]:
                    masked_boxes += 1
                if show_masked or not rect["masked"]:
                    boxes.append(rect)

            matched_boxes = [box for box in boxes if box["matched"]]
            if only_empty and not empty_label:
                continue
            if not include_empty and not boxes:
                continue
            if not only_empty and not matched_boxes and has_box_filter(query):
                continue
            samples.append(
                {
                    "stem": image_path.stem,
                    "image": image_path.name,
                    "width": width,
                    "height": height,
                    "boxes": boxes,
                    "matched_count": len(matched_boxes),
                    "empty_label": empty_label,
                    "drop_image": drop_image,
                    "drop_reason": image_marker.get("reason", ""),
                }
            )

        return {
            "samples_dir": str(self.samples_dir),
            "mask_file": str(self.masks_path),
            "samples": samples,
            "stats": {
                "images": len(samples),
                "total_boxes": total_boxes,
                "masked_boxes": masked_boxes,
                "dropped_images": dropped_images,
            },
        }

    def merge_candidates(self, query: dict[str, str]) -> dict[str, object]:
        merge_params = merge_params_from_query(query)
        items = []
        for image_path in self.image_paths():
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            width, height = self.dimensions.get(image_path.name) or image_size(image_path)
            boxes = []
            for label_box in parse_label_file(label_path):
                rect = label_box.to_rect(width, height)
                marker = self.masks.get(image_path.stem, {}).get(str(label_box.index), {})
                rect["masked"] = bool(marker.get("masked", False))
                rect["reason"] = marker.get("reason", "")
                boxes.append(rect)
            for candidate_index, candidate in enumerate(merge_candidate_groups(boxes, **merge_params)):
                source_indices = set(candidate["indices"])
                items.append(
                    {
                        "id": f"{image_path.stem}:{','.join(str(index) for index in candidate['indices'])}",
                        "candidate_index": candidate_index,
                        "stem": image_path.stem,
                        "image": image_path.name,
                        "width": width,
                        "height": height,
                        "indices": candidate["indices"],
                        "rect": candidate["rect"],
                        "boxes": [box for box in boxes if int(box["index"]) in source_indices],
                    }
                )
        return {
            "samples_dir": str(self.samples_dir),
            "mask_file": str(self.masks_path),
            "items": items,
            "stats": {"merge_candidates": len(items)},
            "merge_params": merge_params,
        }

    def set_mask(self, stem: str, label_index: int, masked: bool, reason: str) -> None:
        stem_items = self.masks.setdefault(stem, {})
        key = str(label_index)
        if masked:
            stem_items[key] = {
                "masked": True,
                "reason": reason or "manual",
                "updated_at": int(time.time()),
            }
        else:
            stem_items.pop(key, None)
            if not stem_items:
                self.masks.pop(stem, None)
        save_masks(self.masks_path, self.masks)

    def set_image_drop(self, stem: str, drop_image: bool, reason: str) -> None:
        self.image_info(stem)
        stem_items = self.masks.setdefault(stem, {})
        if drop_image:
            stem_items[IMAGE_MASK_KEY] = {
                "drop_image": True,
                "reason": reason or "manual",
                "updated_at": int(time.time()),
            }
        else:
            stem_items.pop(IMAGE_MASK_KEY, None)
            if not stem_items:
                self.masks.pop(stem, None)
        save_masks(self.masks_path, self.masks)

    def set_box(self, stem: str, label_index: int, rect: dict[str, float]) -> dict[str, float | int | str]:
        image_path, width, height = self.image_info(stem)
        label_path = self.labels_dir / f"{stem}.txt"
        write_label_box(label_path, label_index, rect, width, height)
        for label_box in parse_label_file(label_path):
            if label_box.index == label_index:
                return label_box.to_rect(width, height)
        raise ValueError(f"updated label index not found: {label_index}")

    def add_box(self, stem: str, rect: dict[str, float]) -> dict[str, float | int | str]:
        image_path, width, height = self.image_info(stem)
        label_path = self.labels_dir / f"{stem}.txt"
        label_box = append_label_box(label_path, rect, width, height)
        return label_box.to_rect(width, height)

    def merge_boxes(self, stem: str, label_indices: list[int]) -> dict[str, object]:
        image_path, width, height = self.image_info(stem)
        label_path = self.labels_dir / f"{stem}.txt"
        boxes = [label_box.to_rect(width, height) for label_box in parse_label_file(label_path)]
        selected = [box for box in boxes if int(box["index"]) in set(label_indices)]
        if len(selected) != len(set(label_indices)):
            raise ValueError(f"merge label indexes not found in {label_path}: {label_indices}")
        rect = union_rect(selected)
        keep_index = merge_label_boxes(label_path, label_indices, rect, width, height)
        for label_index in label_indices:
            if label_index != keep_index:
                self.set_mask(stem, label_index, True, "merge")
        for label_box in parse_label_file(label_path):
            if label_box.index == keep_index:
                merged_box = label_box.to_rect(width, height)
                merged_box["masked"] = False
                merged_box["reason"] = ""
                return {"keep_index": keep_index, "box": merged_box, "masked_indices": sorted(set(label_indices) - {keep_index})}
        raise ValueError(f"updated merge label index not found: {keep_index}")

    def image_info(self, stem: str) -> tuple[Path, int, int]:
        image_path = next(
            (
                path
                for path in self.images_dir.iterdir()
                if path.is_file()
                and path.stem == stem
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            None,
        )
        if image_path is None:
            raise ValueError(f"missing image for stem: {stem}")
        width, height = self.dimensions.get(image_path.name) or image_size(image_path)
        return image_path, width, height


def make_handler(app: ReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            print(format % args, file=sys.stderr)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.write_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/merge-review":
                self.write_bytes(MERGE_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                self.write_json(app.state(query))
                return
            if parsed.path == "/api/merge-candidates":
                query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
                self.write_json(app.merge_candidates(query))
                return
            if parsed.path.startswith("/images/"):
                filename = unquote(parsed.path.removeprefix("/images/"))
                image_path = (app.images_dir / filename).resolve()
                if app.images_dir not in image_path.parents or not image_path.exists():
                    self.send_error(404)
                    return
                content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                self.write_bytes(image_path.read_bytes(), content_type)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in ("/api/mask", "/api/image-drop", "/api/box", "/api/add-box", "/api/merge"):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                if parsed.path == "/api/mask":
                    app.set_mask(
                        stem=str(payload["stem"]),
                        label_index=int(payload["index"]),
                        masked=bool(payload["masked"]),
                        reason=str(payload.get("reason", "manual")),
                    )
                    self.write_json({"ok": True})
                elif parsed.path == "/api/image-drop":
                    app.set_image_drop(
                        stem=str(payload["stem"]),
                        drop_image=bool(payload["drop_image"]),
                        reason=str(payload.get("reason", "manual")),
                    )
                    self.write_json({"ok": True})
                elif parsed.path == "/api/box":
                    rect = app.set_box(
                        stem=str(payload["stem"]),
                        label_index=int(payload["index"]),
                        rect={
                            "x": float(payload["x"]),
                            "y": float(payload["y"]),
                            "width": float(payload["width"]),
                            "height": float(payload["height"]),
                        },
                    )
                    self.write_json({"ok": True, "box": rect})
                elif parsed.path == "/api/add-box":
                    rect = app.add_box(
                        stem=str(payload["stem"]),
                        rect={
                            "x": float(payload["x"]),
                            "y": float(payload["y"]),
                            "width": float(payload["width"]),
                            "height": float(payload["height"]),
                        },
                    )
                    self.write_json({"ok": True, "box": rect})
                else:
                    result = app.merge_boxes(
                        stem=str(payload["stem"]),
                        label_indices=[int(index) for index in payload["indices"]],
                    )
                    self.write_json({"ok": True, **result})
            except Exception as exc:
                self.send_error(400, str(exc))

        def write_json(self, payload: object) -> None:
            self.write_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def write_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>generated label review</title>
  <style>
    :root { font-family: Arial, "Microsoft YaHei", sans-serif; color: #1f2328; background: #f6f8fa; }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body { height: 100vh; margin: 0; padding: 16px; overflow: hidden; }
    main { display: grid; grid-template-columns: minmax(520px, 1fr) minmax(360px, 430px) minmax(280px, 340px); grid-template-rows: auto minmax(0, 1fr); gap: 14px; height: calc(100vh - 32px); max-width: 1840px; margin: 0 auto; overflow: hidden; }
    .panel, .stage { min-width: 0; min-height: 0; background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .panel { display: flex; flex-direction: column; overflow: auto; }
    .info-panel { gap: 10px; }
    .stage { display: grid; grid-template-rows: minmax(0, 1fr) auto; gap: 10px; background: #111; overflow: hidden; }
    .stage.hide-full-image .crop-preview { max-height: 132px; }
    .canvas-wrap { min-width: 0; min-height: 0; display: grid; place-items: center; overflow: hidden; }
    canvas { max-width: 100%; max-height: 100%; background: #000; }
    .crop-preview { width: 100%; min-height: 112px; max-height: 132px; display: flex; gap: 10px; align-items: stretch; overflow-x: auto; overflow-y: hidden; padding: 8px; border: 1px solid #30363d; border-radius: 6px; background: #0d1117; }
    .crop-preview.empty { align-items: center; justify-content: center; color: #8b949e; font-size: 13px; }
    .crop-thumb { flex: 0 0 auto; height: 100px; min-width: 96px; max-width: 240px; display: grid; grid-template-rows: minmax(0, 1fr) 18px; gap: 4px; padding: 4px; border: 1px solid #30363d; border-radius: 6px; background: #161b22; color: #c9d1d9; cursor: pointer; }
    .crop-thumb.active { border-color: #ff2bd6; box-shadow: 0 0 0 2px rgba(255, 43, 214, .45) inset; }
    .crop-thumb canvas { width: 100%; height: 78px; object-fit: contain; }
    .crop-label { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; line-height: 18px; text-align: center; }
    .topbar { grid-column: 1 / -1; display: flex; align-items: center; justify-content: space-between; gap: 12px; min-width: 0; }
    h1 { margin: 0; font-size: 20px; }
    label { display: grid; gap: 4px; font-size: 13px; }
    input { width: 100%; min-width: 0; height: 30px; padding: 4px 8px; border: 1px solid #d0d7de; border-radius: 6px; }
    input[type="checkbox"] { width: 18px; height: 18px; padding: 0; flex: 0 0 auto; }
    button { height: 32px; border: 1px solid #8c959f; border-radius: 6px; background: #fff; cursor: pointer; }
    .filters { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .filter-row { display: grid; grid-template-columns: 18px minmax(0, 1fr); gap: 3px 6px; align-items: center; padding: 5px; border: 1px solid #d0d7de; border-radius: 6px; background: #f6f8fa; }
    .filter-row input[type="checkbox"] { justify-self: center; }
    .filter-row input[type="number"] { grid-column: 1 / -1; height: 26px; }
    .filter-title { min-width: 0; font-size: 12px; line-height: 16px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .filter-help { display: none; }
    .jump-row { display: grid; grid-template-columns: 1fr 72px; gap: 8px; align-items: end; margin-top: 10px; }
    .jump-status { margin-top: 6px; min-height: 16px; font-size: 12px; color: #57606a; }
    .jump-status.error { color: #cf222e; font-weight: 600; }
    .checks { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 10px; margin: 10px 0; }
    .checks label { display: inline-flex; grid-auto-flow: column; align-items: center; gap: 6px; min-width: 0; font-size: 13px; }
    .action-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 8px 0 4px; }
    .action-grid button { width: 100%; min-width: 0; }
    .action-grid .primary-action { background: #0969da; border-color: #0969da; color: #fff; }
    .action-grid .warn-action { background: #fff8c5; border-color: #d4a72c; }
    .action-grid .active-action { background: #ddf4ff; border-color: #0969da; color: #0969da; font-weight: 700; }
    .meta, .help, .box-list { font-size: 13px; line-height: 1.45; }
    .help kbd { display: inline-block; min-width: 48px; padding: 2px 5px; border: 1px solid #d0d7de; border-radius: 4px; background: #f6f8fa; text-align: center; }
    .box-list { display: grid; flex: 1 1 auto; gap: 6px; min-height: 120px; max-height: none; overflow: auto; }
    .box { padding: 6px; border: 1px solid #d0d7de; border-radius: 6px; }
    .box.active { border: 3px solid #8250df; background: #fbefff; box-shadow: 0 0 0 2px #d8b9ff inset; }
    .box.masked { opacity: .58; text-decoration: line-through; }
    .box.matched { border-left: 5px solid #bf8700; }
    .link-button { display: inline-grid; place-items: center; height: 32px; padding: 0 10px; border: 1px solid #8c959f; border-radius: 6px; background: #fff; color: #1f2328; text-decoration: none; font-size: 13px; }
    .row { display: flex; gap: 8px; align-items: center; justify-content: space-between; }
    .path { word-break: break-all; color: #57606a; }
    .save-status { margin-top: 8px; font-size: 12px; color: #57606a; }
    .save-status.error { color: #cf222e; font-weight: 600; }
    .edit-status { margin-top: 6px; font-size: 12px; color: #57606a; }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; grid-template-rows: auto minmax(220px, 42vh) minmax(0, 1fr) minmax(0, .8fr); }
      .filters { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <div class="topbar">
    <h1>generated_samples label review</h1>
    <a class="link-button" href="/merge-review" target="_blank">打开融合审批</a>
  </div>
  <section class="stage" id="stage">
    <div class="canvas-wrap"><canvas id="canvas"></canvas></div>
    <div class="crop-preview empty" id="cropPreview">无有效选框</div>
  </section>
  <aside class="panel">
    <div class="filters">
      <div class="filter-row">
        <input id="areaEnabled" type="checkbox" aria-label="启用面积小于过滤">
        <div><div class="filter-title">面积小于</div><div class="filter-help">找出面积小于阈值的框，单位：像素²</div></div>
        <input id="areaLt" type="number" min="0" placeholder="2000">
      </div>
      <div class="filter-row">
        <input id="areaGtEnabled" type="checkbox" aria-label="启用面积大于过滤">
        <div><div class="filter-title">面积大于</div><div class="filter-help">找出面积大于阈值的框，单位：像素²</div></div>
        <input id="areaGt" type="number" min="0" placeholder="120000">
      </div>
      <div class="filter-row">
        <input id="widthLtEnabled" type="checkbox" aria-label="启用宽度小于过滤">
        <div><div class="filter-title">宽度小于</div><div class="filter-help">找出宽度小于阈值的框，单位：像素</div></div>
        <input id="widthLt" type="number" min="0" placeholder="80">
      </div>
      <div class="filter-row">
        <input id="widthGtEnabled" type="checkbox" aria-label="启用宽度大于过滤">
        <div><div class="filter-title">宽度大于</div><div class="filter-help">找出宽度大于阈值的框，单位：像素</div></div>
        <input id="widthGt" type="number" min="0" placeholder="900">
      </div>
      <div class="filter-row">
        <input id="heightLtEnabled" type="checkbox" aria-label="启用高度小于过滤">
        <div><div class="filter-title">高度小于</div><div class="filter-help">找出高度小于阈值的框，单位：像素</div></div>
        <input id="heightLt" type="number" min="0" placeholder="20">
      </div>
      <div class="filter-row">
        <input id="heightGtEnabled" type="checkbox" aria-label="启用高度大于过滤">
        <div><div class="filter-title">高度大于</div><div class="filter-help">找出高度大于阈值的框，单位：像素</div></div>
        <input id="heightGt" type="number" min="0" placeholder="120">
      </div>
      <div class="filter-row">
        <input id="yLtEnabled" type="checkbox" aria-label="启用位置高于过滤">
        <div><div class="filter-title">位置高于</div><div class="filter-help">区域中心 Y 小于阈值，单位：像素</div></div>
        <input id="yLt" type="number" min="0" placeholder="320">
      </div>
      <div class="filter-row">
        <input id="yGtEnabled" type="checkbox" aria-label="启用位置低于过滤">
        <div><div class="filter-title">位置低于</div><div class="filter-help">区域中心 Y 大于阈值，单位：像素</div></div>
        <input id="yGt" type="number" min="0" placeholder="720">
      </div>
    </div>
    <div class="jump-row">
      <label>跳转图像
        <input id="jumpImage" type="text" placeholder="序号或文件名片段">
      </label>
      <button id="jumpButton">跳转</button>
    </div>
    <div class="jump-status" id="jumpStatus"></div>
    <div class="checks">
      <label><input id="includeEmpty" type="checkbox"> 显示无区域图像</label>
      <label><input id="onlyEmpty" type="checkbox"> 仅显示无区域图像</label>
      <label><input id="onlyDropped" type="checkbox"> 仅显示已屏蔽图像</label>
      <label><input id="showMasked" type="checkbox" checked> 显示已屏蔽框</label>
      <label><input id="showFullImage" type="checkbox" checked> 显示大图</label>
    </div>
    <div class="action-grid">
      <button class="primary-action" id="reload">应用过滤</button>
      <button id="addBoxMode">新增选框</button>
      <button id="maskMatched">屏蔽匹配框</button>
      <button class="warn-action" id="toggleImageDrop">屏蔽图像</button>
    </div>
    <hr>
    <div class="box-list" id="boxList"></div>
  </aside>
  <aside class="panel info-panel">
    <div class="save-status" id="saveStatus">保存状态：空闲</div>
    <div class="meta" id="meta">加载中...</div>
    <div class="path" id="paths"></div>
    <hr>
    <div class="help">
      <div><kbd>N / →</kbd> 下一张；<kbd>P / ←</kbd> 上一张</div>
      <div><kbd>J / ↓</kbd> 下一个框；<kbd>K / ↑</kbd> 上一个框</div>
      <div><kbd>Del/M</kbd> 屏蔽/取消当前框；<kbd>A</kbd> 屏蔽当前匹配框</div>
      <div><kbd>X</kbd> 屏蔽/取消当前图像；<kbd>B</kbd> 新增选框</div>
      <div><kbd>Ctrl+A</kbd> 全选当前图选框；<kbd>Esc</kbd> 取消多选</div>
      <div><kbd>拖拽</kbd> 移动选框；拖拽边/角调整大小</div>
      <div><kbd>Ctrl+方向</kbd> 微移；<kbd>Ctrl+Shift+方向</kbd> 缩放</div>
      <div><kbd>G</kbd> 聚焦跳转；<kbd>R</kbd> 重新载入</div>
    </div>
    <div class="edit-status" id="editStatus">编辑：单击选中，拖动框或边角调整</div>
  </aside>
</main>
<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const stage = document.getElementById("stage");
const ids = [
  "areaEnabled", "areaLt", "areaGtEnabled", "areaGt",
  "widthLtEnabled", "widthLt", "widthGtEnabled", "widthGt",
  "heightLtEnabled", "heightLt", "heightGtEnabled", "heightGt",
  "yLtEnabled", "yLt", "yGtEnabled", "yGt",
  "includeEmpty", "onlyEmpty", "onlyDropped", "showMasked", "showFullImage", "jumpImage"
];
const filterIds = ids.filter(id => !["jumpImage", "showFullImage"].includes(id));
const el = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
const meta = document.getElementById("meta");
const paths = document.getElementById("paths");
const boxList = document.getElementById("boxList");
const cropPreview = document.getElementById("cropPreview");
const saveStatus = document.getElementById("saveStatus");
const editStatus = document.getElementById("editStatus");
const jumpStatus = document.getElementById("jumpStatus");
let state = null, imageIndex = 0, boxIndex = 0, img = new Image();
let pendingSaves = 0;
let selectedBoxes = new Set();
let drag = null;
let addBoxMode = false;
let activeYGuide = "";
const HANDLE_SIZE = 10;
const HANDLE_HIT_SIZE = 18;
const GUIDE_HIT_SIZE = 16;
const MIN_BOX_SIZE = 4;

function qs() {
  const p = new URLSearchParams();
  if (el.areaEnabled.checked && el.areaLt.value) p.set("area_lt", el.areaLt.value);
  if (el.areaGtEnabled.checked && el.areaGt.value) p.set("area_gt", el.areaGt.value);
  if (el.widthLtEnabled.checked && el.widthLt.value) p.set("width_lt", el.widthLt.value);
  if (el.widthGtEnabled.checked && el.widthGt.value) p.set("width_gt", el.widthGt.value);
  if (el.heightLtEnabled.checked && el.heightLt.value) p.set("height_lt", el.heightLt.value);
  if (el.heightGtEnabled.checked && el.heightGt.value) p.set("height_gt", el.heightGt.value);
  if (el.yLtEnabled.checked && el.yLt.value) p.set("y_lt", el.yLt.value);
  if (el.yGtEnabled.checked && el.yGt.value) p.set("y_gt", el.yGt.value);
  if (el.includeEmpty.checked || el.onlyEmpty.checked || el.onlyDropped.checked) p.set("include_empty", "1");
  if (el.onlyEmpty.checked) p.set("only_empty", "1");
  if (el.onlyDropped.checked) p.set("only_dropped", "1");
  p.set("show_masked", el.showMasked.checked ? "1" : "0");
  return p.toString();
}

async function loadState(keepPosition = false) {
  const oldStem = current()?.stem;
  state = await fetch("/api/state?" + qs()).then(r => r.json());
  if (keepPosition && oldStem) {
    imageIndex = Math.max(0, state.samples.findIndex(s => s.stem === oldStem));
  }
  if (imageIndex < 0) imageIndex = 0;
  if (imageIndex >= state.samples.length) imageIndex = Math.max(0, state.samples.length - 1);
  boxIndex = 0;
  selectedBoxes.clear();
  if (state.samples.length) el.jumpImage.value = String(imageIndex + 1);
  await loadImage();
}

function current() { return state?.samples?.[imageIndex]; }
function currentBox() { return current()?.boxes?.[boxIndex]; }

async function loadImage() {
  const sample = current();
  if (!sample) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    meta.textContent = "没有匹配图像";
    boxList.innerHTML = "";
    renderCropPreview(null);
    return;
  }
  img = new Image();
  img.onload = () => {
    canvas.width = sample.width;
    canvas.height = sample.height;
    draw();
  };
  img.src = "/images/" + encodeURIComponent(sample.image);
}

function draw() {
  const sample = current();
  stage.classList.toggle("hide-full-image", !el.showFullImage.checked);
  if (!sample) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    renderCropPreview(null);
    return;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (el.showFullImage.checked) {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  } else {
    drawImagePlaceholder(sample);
  }
  if (sample.drop_image) {
    ctx.save();
    ctx.fillStyle = "rgba(207, 34, 46, .22)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(207, 34, 46, .94)";
    ctx.fillRect(16, 16, 160, 34);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 20px Arial";
    ctx.fillText("图像已屏蔽", 28, 40);
    ctx.restore();
  }
  drawYGuideFills(sample);
  sample.boxes.forEach((box, i) => {
    const active = i === boxIndex;
    const selected = selectedBoxes.has(i);
    const stroke = active ? "#ff2bd6" : (box.masked ? "#cf222e" : (box.matched ? "#d29922" : "#2da44e"));
    ctx.save();
    if (active || selected) {
      ctx.lineWidth = 9;
      ctx.strokeStyle = "#ffffff";
      ctx.setLineDash([]);
      ctx.strokeRect(box.x - 2, box.y - 2, box.width + 4, box.height + 4);
    }
    ctx.lineWidth = active ? 5 : 2;
    ctx.strokeStyle = stroke;
    ctx.setLineDash(box.masked ? [8, 5] : []);
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    if (active) drawHandles(box);
    const label = `${box.index}${active ? " selected" : ""}${selected ? " multi" : ""}${box.masked ? " masked" : ""}`;
    ctx.font = active ? "bold 20px Arial" : "18px Arial";
    const labelX = box.x + 4;
    const labelY = Math.max(24, box.y - 6);
    const labelWidth = ctx.measureText(label).width + 10;
    ctx.fillStyle = active ? "rgba(255, 43, 214, .92)" : stroke;
    ctx.fillRect(labelX - 3, labelY - 19, labelWidth, 24);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, labelX + 2, labelY);
    ctx.restore();
  });
  drawYGuideLines(sample);
  renderPanel();
  renderCropPreview(sample);
}

function drawImagePlaceholder(sample) {
  ctx.save();
  ctx.fillStyle = "#101820";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(255, 255, 255, .12)";
  ctx.lineWidth = 1;
  const step = Math.max(40, Math.round(Math.min(sample.width, sample.height) / 12));
  for (let x = 0; x <= sample.width; x += step) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, sample.height);
    ctx.stroke();
  }
  for (let y = 0; y <= sample.height; y += step) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(sample.width, y);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(255, 255, 255, .75)";
  ctx.font = "bold 20px Arial";
  ctx.fillText("大图已隐藏", 18, 34);
  ctx.font = "16px Arial";
  ctx.fillText(`${sample.width}x${sample.height}`, 18, 58);
  ctx.restore();
}

function drawYGuideFills(sample) {
  const guides = yGuides(sample);
  for (const guide of guides) {
    ctx.save();
    ctx.fillStyle = guide.fill;
    if (guide.id === "yLt") ctx.fillRect(0, 0, sample.width, guide.y);
    else ctx.fillRect(0, guide.y, sample.width, sample.height - guide.y);
    ctx.restore();
  }
}

function drawYGuideLines(sample) {
  const guides = yGuides(sample);
  for (const guide of guides) {
    ctx.save();

    ctx.strokeStyle = guide.color;
    ctx.lineWidth = guide.id === activeYGuide ? 5 : 3;
    ctx.setLineDash(guide.id === activeYGuide ? [] : [12, 8]);
    ctx.beginPath();
    ctx.moveTo(0, guide.y);
    ctx.lineTo(sample.width, guide.y);
    ctx.stroke();
    ctx.setLineDash([]);

    const label = `${guide.title} ${Math.round(guide.y)}px`;
    ctx.font = "bold 18px Arial";
    const labelWidth = ctx.measureText(label).width + 16;
    const labelY = Math.max(28, Math.min(sample.height - 8, guide.y - 8));
    ctx.fillStyle = guide.color;
    ctx.fillRect(10, labelY - 22, labelWidth, 26);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, 18, labelY - 3);
    ctx.restore();
  }
}

function yGuides(sample) {
  const guides = [];
  if (el.yLtEnabled.checked && el.yLt.value !== "") {
    guides.push({
      id: "yLt",
      y: clampY(Number(el.yLt.value), sample),
      title: "位置高于",
      color: "#0969da",
      fill: "rgba(9, 105, 218, .10)",
    });
  }
  if (el.yGtEnabled.checked && el.yGt.value !== "") {
    guides.push({
      id: "yGt",
      y: clampY(Number(el.yGt.value), sample),
      title: "位置低于",
      color: "#bf8700",
      fill: "rgba(191, 135, 0, .12)",
    });
  }
  return guides;
}

function clampY(value, sample) {
  const number = Number.isFinite(value) ? value : 0;
  return Math.max(0, Math.min(sample.height, number));
}

function drawHandles(box) {
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#ff2bd6";
  ctx.lineWidth = 2;
  const handleSize = handleVisualSize();
  for (const handle of boxHandles(box)) {
    ctx.fillRect(handle.x - handleSize / 2, handle.y - handleSize / 2, handleSize, handleSize);
    ctx.strokeRect(handle.x - handleSize / 2, handle.y - handleSize / 2, handleSize, handleSize);
  }
}

function boxHandles(box) {
  const midX = box.x + box.width / 2;
  const midY = box.y + box.height / 2;
  return [
    {name: "nw", x: box.x, y: box.y},
    {name: "n", x: midX, y: box.y},
    {name: "ne", x: box.x + box.width, y: box.y},
    {name: "e", x: box.x + box.width, y: midY},
    {name: "se", x: box.x + box.width, y: box.y + box.height},
    {name: "s", x: midX, y: box.y + box.height},
    {name: "sw", x: box.x, y: box.y + box.height},
    {name: "w", x: box.x, y: midY},
  ];
}

function handleVisualSize() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return HANDLE_SIZE;
  const scale = Math.max(canvas.width / rect.width, canvas.height / rect.height);
  return Math.max(HANDLE_SIZE, 10 * scale);
}

function renderPanel() {
  const sample = current();
  meta.textContent = `${imageIndex + 1}/${state.samples.length} ${sample.image} | ${sample.width}x${sample.height} | boxes=${sample.boxes.length} matched=${sample.matched_count} | ${sample.drop_image ? "图像已屏蔽" : "图像有效"} | total=${state.stats.total_boxes} masked=${state.stats.masked_boxes} dropped=${state.stats.dropped_images}`;
  paths.textContent = `samples=${state.samples_dir} mask=${state.mask_file}`;
  const imageDropButton = document.getElementById("toggleImageDrop");
  const addBoxButton = document.getElementById("addBoxMode");
  imageDropButton.textContent = sample.drop_image ? "取消屏蔽图像" : "屏蔽图像";
  imageDropButton.classList.toggle("active-action", sample.drop_image);
  addBoxButton.textContent = addBoxMode ? "退出新增" : "新增选框";
  addBoxButton.classList.toggle("active-action", addBoxMode);
  boxList.innerHTML = sample.boxes.map((box, i) => `
    <div class="box ${i === boxIndex ? "active" : ""} ${selectedBoxes.has(i) ? "active" : ""} ${box.masked ? "masked" : ""} ${box.matched ? "matched" : ""}" data-i="${i}">
      #${box.index} ${selectedBoxes.has(i) ? "[已选]" : ""} ${box.masked ? "[屏蔽]" : "[有效]"} ${box.matched ? "[匹配]" : ""}
      <br>x=${box.x} y=${box.y} cy=${box.center_y} w=${box.width} h=${box.height} area=${box.area} ratio=${box.aspect}
    </div>`).join("");
  boxList.querySelectorAll(".box").forEach(node => node.onclick = (event) => {
    boxIndex = Number(node.dataset.i);
    if (event.ctrlKey || event.metaKey) toggleSelectedBox(boxIndex);
    draw();
  });
  editStatus.textContent = selectedBoxes.size
    ? `编辑：已选择 ${selectedBoxes.size} 个选框，Del/M 会批量屏蔽或取消`
    : addBoxMode
    ? "编辑：新增选框模式，拖拽空白区域创建"
    : "编辑：单击选中，拖动框或边角调整";
}

function renderCropPreview(sample) {
  const visibleBoxes = sample?.boxes?.map((box, i) => ({box, i})).filter(item => !item.box.masked) || [];
  cropPreview.classList.toggle("empty", !visibleBoxes.length);
  cropPreview.innerHTML = "";
  if (!visibleBoxes.length || !img.complete || !img.naturalWidth) {
    cropPreview.textContent = "无有效选框";
    return;
  }

  for (const {box, i} of visibleBoxes) {
    const source = clampSourceRect(box, sample);
    if (!source.width || !source.height) continue;
    const thumb = document.createElement("button");
    thumb.type = "button";
    thumb.className = `crop-thumb ${i === boxIndex ? "active" : ""}`;
    thumb.dataset.i = String(i);
    thumb.title = `#${box.index} x=${box.x} y=${box.y} w=${box.width} h=${box.height}`;

    const cropCanvas = document.createElement("canvas");
    const scale = Math.min(2, 210 / source.width, 78 / source.height);
    cropCanvas.width = Math.max(24, Math.round(source.width * scale));
    cropCanvas.height = Math.max(24, Math.round(source.height * scale));
    const cropCtx = cropCanvas.getContext("2d");
    cropCtx.imageSmoothingEnabled = true;
    cropCtx.drawImage(
      img,
      source.x,
      source.y,
      source.width,
      source.height,
      0,
      0,
      cropCanvas.width,
      cropCanvas.height
    );

    const label = document.createElement("div");
    label.className = "crop-label";
    label.textContent = `#${box.index} ${Math.round(box.width)}x${Math.round(box.height)}`;
    thumb.append(cropCanvas, label);
    thumb.onclick = () => {
      boxIndex = Number(thumb.dataset.i);
      draw();
    };
    cropPreview.appendChild(thumb);
  }

  if (!cropPreview.children.length) {
    cropPreview.classList.add("empty");
    cropPreview.textContent = "无有效选框";
  }
}

function clampSourceRect(box, sample) {
  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const right = Math.min(sample.width, Math.ceil(box.x + box.width));
  const bottom = Math.min(sample.height, Math.ceil(box.y + box.height));
  return {
    x,
    y,
    width: Math.max(0, right - x),
    height: Math.max(0, bottom - y),
  };
}

function updateSaveStatus(error = "") {
  saveStatus.classList.toggle("error", Boolean(error));
  if (error) {
    saveStatus.textContent = `保存状态：失败 - ${error}`;
  } else if (pendingSaves > 0) {
    saveStatus.textContent = `保存状态：后台保存中 (${pendingSaves})`;
  } else {
    saveStatus.textContent = "保存状态：已保存";
  }
}

function queueMaskSave(stem, box, previousMasked, nextMasked, saveVersion, reason = "manual") {
  pendingSaves += 1;
  updateSaveStatus();
  fetch("/api/mask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({stem, index: box.index, masked: nextMasked, reason})
  }).then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  }).catch(error => {
    if (box.saveVersion === saveVersion) {
      box.masked = previousMasked;
      state.stats.masked_boxes += previousMasked ? 1 : -1;
      draw();
    }
    updateSaveStatus(error.message || "unknown error");
  }).finally(() => {
    pendingSaves = Math.max(0, pendingSaves - 1);
    if (!saveStatus.classList.contains("error")) updateSaveStatus();
  });
}

function queueBoxSave(sample, box, previousRect, saveVersion) {
  pendingSaves += 1;
  updateSaveStatus();
  fetch("/api/box", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      stem: sample.stem,
      index: box.index,
      x: box.x,
      y: box.y,
      width: box.width,
      height: box.height,
    })
  }).then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }).then(payload => {
    if (box.editVersion === saveVersion && payload.box) {
      Object.assign(box, payload.box);
      box.masked = Boolean(box.masked);
      box.matched = filterBoxMatch(box);
      draw();
    }
  }).catch(error => {
    if (box.editVersion === saveVersion) {
      Object.assign(box, previousRect);
      draw();
    }
    updateSaveStatus(error.message || "unknown error");
  }).finally(() => {
    pendingSaves = Math.max(0, pendingSaves - 1);
    if (!saveStatus.classList.contains("error")) updateSaveStatus();
  });
}

function queueAddBoxSave(sample, box, saveVersion) {
  pendingSaves += 1;
  updateSaveStatus();
  fetch("/api/add-box", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      stem: sample.stem,
      x: box.x,
      y: box.y,
      width: box.width,
      height: box.height,
    })
  }).then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }).then(payload => {
    if (box.editVersion === saveVersion && payload.box) {
      Object.assign(box, payload.box);
      box.masked = false;
      box.reason = "";
      box.matched = filterBoxMatch(box);
      draw();
    }
  }).catch(error => {
    const pos = sample.boxes.indexOf(box);
    if (pos >= 0) sample.boxes.splice(pos, 1);
    boxIndex = Math.max(0, Math.min(boxIndex, sample.boxes.length - 1));
    draw();
    updateSaveStatus(error.message || "unknown error");
  }).finally(() => {
    pendingSaves = Math.max(0, pendingSaves - 1);
    if (!saveStatus.classList.contains("error")) updateSaveStatus();
  });
}

function filterBoxMatch(box) {
  const checks = [];
  if (el.areaEnabled.checked && el.areaLt.value) checks.push(box.area < Number(el.areaLt.value));
  if (el.areaGtEnabled.checked && el.areaGt.value) checks.push(box.area > Number(el.areaGt.value));
  if (el.widthLtEnabled.checked && el.widthLt.value) checks.push(box.width < Number(el.widthLt.value));
  if (el.widthGtEnabled.checked && el.widthGt.value) checks.push(box.width > Number(el.widthGt.value));
  if (el.heightLtEnabled.checked && el.heightLt.value) checks.push(box.height < Number(el.heightLt.value));
  if (el.heightGtEnabled.checked && el.heightGt.value) checks.push(box.height > Number(el.heightGt.value));
  if (el.yLtEnabled.checked && el.yLt.value) checks.push(box.center_y < Number(el.yLt.value));
  if (el.yGtEnabled.checked && el.yGt.value) checks.push(box.center_y > Number(el.yGt.value));
  return checks.length ? checks.every(Boolean) : true;
}

function normalizeBoxMetrics(box) {
  box.x = round2(box.x);
  box.y = round2(box.y);
  box.width = round2(box.width);
  box.height = round2(box.height);
  box.center_y = round2(box.y + box.height / 2);
  box.area = round2(box.width * box.height);
  box.aspect = box.height ? Math.round((box.width / box.height) * 10000) / 10000 : 0;
  box.matched = filterBoxMatch(box);
}

function round2(value) {
  return Math.round(value * 100) / 100;
}

function commitBoxEdit(sample, box, previousRect) {
  normalizeBoxMetrics(box);
  box.editVersion = (box.editVersion || 0) + 1;
  draw();
  queueBoxSave(sample, box, previousRect, box.editVersion);
}

function applyMaskOptimistic(sample, box, nextMasked, reason = "manual") {
  const previousMasked = Boolean(box.masked);
  if (previousMasked === nextMasked) return;
  box.saveVersion = (box.saveVersion || 0) + 1;
  box.masked = nextMasked;
  state.stats.masked_boxes += nextMasked ? 1 : -1;
  draw();
  queueMaskSave(sample.stem, box, previousMasked, nextMasked, box.saveVersion, reason);
}

function toggleCurrentImageDrop() {
  const sample = current();
  if (!sample) return;
  const next = !sample.drop_image;
  sample.drop_image = next;
  state.stats.dropped_images += next ? 1 : -1;
  draw();
  pendingSaves += 1;
  updateSaveStatus();
  fetch("/api/image-drop", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({stem: sample.stem, drop_image: next, reason: "manual"})
  }).then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  }).catch(error => {
    sample.drop_image = !next;
    state.stats.dropped_images += next ? -1 : 1;
    draw();
    updateSaveStatus(error.message || "unknown error");
  }).finally(() => {
    pendingSaves = Math.max(0, pendingSaves - 1);
    if (!saveStatus.classList.contains("error")) updateSaveStatus();
  });
}

function toggleCurrentMask() {
  const sample = current();
  if (!sample) return;
  const indexes = selectedBoxes.size ? [...selectedBoxes] : [boxIndex];
  const boxes = indexes.map(i => sample.boxes[i]).filter(Boolean);
  if (!boxes.length) return;
  const nextMasked = boxes.some(box => !box.masked);
  for (const box of boxes) applyMaskOptimistic(sample, box, nextMasked);
}

function maskMatchedCurrent() {
  const sample = current();
  if (!sample) return;
  for (const box of sample.boxes.filter(b => b.matched && !b.masked)) {
    applyMaskOptimistic(sample, box, true, "filter");
  }
}

function moveImage(delta) {
  if (!state?.samples?.length) return;
  imageIndex = Math.max(0, Math.min(state.samples.length - 1, imageIndex + delta));
  boxIndex = 0;
  selectedBoxes.clear();
  el.jumpImage.value = String(imageIndex + 1);
  updateJumpStatus("");
  loadImage();
}

function jumpToImage() {
  if (!state?.samples?.length) return;
  const query = el.jumpImage.value.trim();
  if (!query) {
    updateJumpStatus("请输入序号或文件名片段", true);
    return;
  }

  let nextIndex = -1;
  if (/^\d+$/.test(query)) {
    const number = Number(query);
    if (number >= 1 && number <= state.samples.length) nextIndex = number - 1;
  } else {
    const needle = query.toLowerCase();
    nextIndex = state.samples.findIndex(sample =>
      sample.image.toLowerCase().includes(needle) || sample.stem.toLowerCase().includes(needle)
    );
  }

  if (nextIndex < 0) {
    updateJumpStatus(`未找到：${query}`, true);
    return;
  }

  imageIndex = nextIndex;
  boxIndex = 0;
  selectedBoxes.clear();
  el.jumpImage.value = String(imageIndex + 1);
  updateJumpStatus(`已跳转到 ${imageIndex + 1}/${state.samples.length}`);
  loadImage();
}

function updateJumpStatus(message, error = false) {
  jumpStatus.classList.toggle("error", error);
  jumpStatus.textContent = message;
}

function moveBox(delta) {
  const boxes = current()?.boxes || [];
  if (!boxes.length) return;
  boxIndex = (boxIndex + delta + boxes.length) % boxes.length;
  draw();
}

function toggleSelectedBox(index) {
  if (selectedBoxes.has(index)) selectedBoxes.delete(index);
  else selectedBoxes.add(index);
}

function selectAllBoxes() {
  const boxes = current()?.boxes || [];
  selectedBoxes = new Set(boxes.map((_, i) => i));
  draw();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function handleHitRadius() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return HANDLE_HIT_SIZE;
  const scale = Math.max(canvas.width / rect.width, canvas.height / rect.height);
  return Math.max(HANDLE_HIT_SIZE, 14 * scale);
}

function guideHitRadius() {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return GUIDE_HIT_SIZE;
  const scale = Math.max(canvas.width / rect.width, canvas.height / rect.height);
  return Math.max(GUIDE_HIT_SIZE, 12 * scale);
}

function hitYGuide(point) {
  const sample = current();
  if (!sample) return "";
  const radius = guideHitRadius();
  const guides = yGuides(sample);
  const active = guides.find(guide => guide.id === activeYGuide);
  if (active && Math.abs(point.y - active.y) <= radius) return active.id;
  for (const guide of guides) {
    if (Math.abs(point.y - guide.y) <= radius) return guide.id;
  }
  return "";
}

function hitHandle(box, point) {
  const radius = handleHitRadius();
  for (const handle of boxHandles(box)) {
    if (Math.abs(point.x - handle.x) <= radius && Math.abs(point.y - handle.y) <= radius) {
      return handle.name;
    }
  }
  return "";
}

function pointInBox(box, point) {
  return point.x >= box.x && point.x <= box.x + box.width && point.y >= box.y && point.y <= box.y + box.height;
}

function hitBox(point) {
  const boxes = current()?.boxes || [];
  for (let i = boxes.length - 1; i >= 0; i -= 1) {
    if (pointInBox(boxes[i], point)) return i;
  }
  return -1;
}

function hitEditableTarget(point) {
  const boxes = current()?.boxes || [];
  const activeBox = boxes[boxIndex];
  if (activeBox) {
    const activeHandle = hitHandle(activeBox, point);
    if (activeHandle) return {index: boxIndex, mode: activeHandle};
  }

  for (const selectedIndex of selectedBoxes) {
    const box = boxes[selectedIndex];
    if (!box || selectedIndex === boxIndex) continue;
    const handle = hitHandle(box, point);
    if (handle) return {index: selectedIndex, mode: handle};
  }

  for (let i = boxes.length - 1; i >= 0; i -= 1) {
    const handle = hitHandle(boxes[i], point);
    if (handle) return {index: i, mode: handle};
  }

  const hit = hitBox(point);
  return hit >= 0 ? {index: hit, mode: "move"} : null;
}

function clampEditedBox(box, sample) {
  let x1 = box.x;
  let y1 = box.y;
  let x2 = box.x + box.width;
  let y2 = box.y + box.height;
  x1 = Math.max(0, Math.min(sample.width - MIN_BOX_SIZE, x1));
  y1 = Math.max(0, Math.min(sample.height - MIN_BOX_SIZE, y1));
  x2 = Math.max(x1 + MIN_BOX_SIZE, Math.min(sample.width, x2));
  y2 = Math.max(y1 + MIN_BOX_SIZE, Math.min(sample.height, y2));
  box.x = x1;
  box.y = y1;
  box.width = x2 - x1;
  box.height = y2 - y1;
  normalizeBoxMetrics(box);
}

function rectFromPoints(startPoint, point, sample) {
  const x1 = Math.max(0, Math.min(sample.width, Math.min(startPoint.x, point.x)));
  const y1 = Math.max(0, Math.min(sample.height, Math.min(startPoint.y, point.y)));
  const x2 = Math.max(0, Math.min(sample.width, Math.max(startPoint.x, point.x)));
  const y2 = Math.max(0, Math.min(sample.height, Math.max(startPoint.y, point.y)));
  return {x: x1, y: y1, width: x2 - x1, height: y2 - y1};
}

function applyDrag(box, sample, mode, startBox, dx, dy) {
  if (mode === "move") {
    box.x = startBox.x + dx;
    box.y = startBox.y + dy;
  } else {
    let x1 = startBox.x;
    let y1 = startBox.y;
    let x2 = startBox.x + startBox.width;
    let y2 = startBox.y + startBox.height;
    if (mode.includes("w")) x1 += dx;
    if (mode.includes("e")) x2 += dx;
    if (mode.includes("n")) y1 += dy;
    if (mode.includes("s")) y2 += dy;
    box.x = Math.min(x1, x2 - MIN_BOX_SIZE);
    box.y = Math.min(y1, y2 - MIN_BOX_SIZE);
    box.width = Math.max(MIN_BOX_SIZE, x2 - x1);
    box.height = Math.max(MIN_BOX_SIZE, y2 - y1);
  }
  clampEditedBox(box, sample);
}

function nudgeBox(dx, dy, resize = false) {
  const sample = current();
  const box = currentBox();
  if (!sample || !box) return;
  const previousRect = {...box};
  if (resize) {
    box.width += dx;
    box.height += dy;
  } else {
    box.x += dx;
    box.y += dy;
  }
  clampEditedBox(box, sample);
  commitBoxEdit(sample, box, previousRect);
}

function cursorForMode(mode) {
  if (mode === "yGuide") return "ns-resize";
  if (mode === "new") return "crosshair";
  if (mode === "move") return "move";
  if (mode === "n" || mode === "s") return "ns-resize";
  if (mode === "e" || mode === "w") return "ew-resize";
  if (mode === "nw" || mode === "se") return "nwse-resize";
  if (mode === "ne" || mode === "sw") return "nesw-resize";
  return "default";
}

canvas.addEventListener("pointerdown", event => {
  const sample = current();
  if (!sample) return;
  const point = canvasPoint(event);
  const yGuide = hitYGuide(point);
  if (yGuide) {
    activeYGuide = yGuide;
    drag = {
      mode: "yGuide",
      guide: yGuide,
      startPoint: point,
      previousValue: el[yGuide].value,
    };
    canvas.setPointerCapture(event.pointerId);
    draw();
    return;
  }
  if (addBoxMode) {
    const box = {
      index: "new",
      class_id: "0",
      x: point.x,
      y: point.y,
      width: MIN_BOX_SIZE,
      height: MIN_BOX_SIZE,
      center_y: round2(point.y + MIN_BOX_SIZE / 2),
      area: MIN_BOX_SIZE * MIN_BOX_SIZE,
      aspect: 1,
      raw: "",
      masked: false,
      reason: "",
      matched: true,
    };
    sample.boxes.push(box);
    boxIndex = sample.boxes.length - 1;
    selectedBoxes.clear();
    drag = {
      mode: "new",
      startPoint: point,
      startBox: {...box},
      previousRect: {...box},
    };
    canvas.setPointerCapture(event.pointerId);
    draw();
    return;
  }
  const target = hitEditableTarget(point);
  if (!target) return;
  boxIndex = target.index;
  if (event.ctrlKey || event.metaKey) toggleSelectedBox(target.index);
  else if (!selectedBoxes.has(target.index)) selectedBoxes.clear();
  const box = currentBox();
  drag = {
    mode: target.mode,
    startPoint: point,
    startBox: {...box},
    previousRect: {...box},
  };
  canvas.setPointerCapture(event.pointerId);
  draw();
});

canvas.addEventListener("pointermove", event => {
  if (!drag) return;
  const sample = current();
  const box = currentBox();
  if (!sample) return;
  const point = canvasPoint(event);
  if (drag.mode === "yGuide") {
    setYGuideValue(drag.guide, point.y, sample);
  } else if (!box) {
    return;
  } else if (drag.mode === "new") {
    Object.assign(box, rectFromPoints(drag.startPoint, point, sample));
    clampEditedBox(box, sample);
  } else {
    applyDrag(box, sample, drag.mode, drag.startBox, point.x - drag.startPoint.x, point.y - drag.startPoint.y);
  }
  draw();
});

canvas.addEventListener("mousemove", event => {
  if (drag) {
    canvas.style.cursor = cursorForMode(drag.mode);
    return;
  }
  if (addBoxMode) {
    canvas.style.cursor = "crosshair";
    return;
  }
  const guide = hitYGuide(canvasPoint(event));
  if (guide) {
    canvas.style.cursor = "ns-resize";
    return;
  }
  const target = hitEditableTarget(canvasPoint(event));
  canvas.style.cursor = target ? cursorForMode(target.mode) : "default";
});

canvas.addEventListener("mouseleave", () => {
  if (!drag) canvas.style.cursor = "default";
});

canvas.addEventListener("pointerup", event => {
  if (!drag) return;
  const sample = current();
  const box = currentBox();
  const previousRect = drag.previousRect;
  const mode = drag.mode;
  drag = null;
  canvas.releasePointerCapture(event.pointerId);
  if (mode === "yGuide") {
    loadState(true);
    return;
  }
  if (sample && box) {
    if (mode === "new") {
      clampEditedBox(box, sample);
      box.editVersion = (box.editVersion || 0) + 1;
      queueAddBoxSave(sample, box, box.editVersion);
      addBoxMode = false;
      draw();
    } else {
      commitBoxEdit(sample, box, previousRect);
    }
  }
});

canvas.addEventListener("pointercancel", () => {
  const box = currentBox();
  if (drag?.mode === "yGuide") {
    el[drag.guide].value = drag.previousValue;
    updateLocalMatches();
  } else if (drag && box) {
    Object.assign(box, drag.previousRect);
  }
  if (drag?.mode === "new") {
    const sample = current();
    if (sample && box) {
      const pos = sample.boxes.indexOf(box);
      if (pos >= 0) sample.boxes.splice(pos, 1);
    }
  }
  drag = null;
  draw();
});

function setYGuideValue(guide, y, sample) {
  const value = String(Math.round(clampY(y, sample)));
  el[guide].value = value;
  el[`${guide}Enabled`].checked = true;
  activeYGuide = guide;
  updateLocalMatches();
}

function updateLocalMatches() {
  const sample = current();
  if (!sample) return;
  for (const box of sample.boxes) box.matched = filterBoxMatch(box);
  sample.matched_count = sample.boxes.filter(box => box.matched).length;
}

document.getElementById("reload").onclick = () => loadState();
document.getElementById("maskMatched").onclick = maskMatchedCurrent;
document.getElementById("toggleImageDrop").onclick = toggleCurrentImageDrop;
document.getElementById("addBoxMode").onclick = () => { addBoxMode = !addBoxMode; draw(); };
document.getElementById("jumpButton").onclick = jumpToImage;
el.jumpImage.addEventListener("keydown", event => {
  if (event.key === "Enter") {
    event.preventDefault();
    jumpToImage();
  }
});
function ensureYGuideValue(inputId) {
  const sample = current();
  if (!sample || el[inputId].value !== "" || !el[`${inputId}Enabled`].checked) return;
  const box = currentBox();
  const y = box ? box.center_y : sample.height / 2;
  el[inputId].value = String(Math.round(clampY(y, sample)));
}

function prepareFilterChange(id) {
  if (id === "yLtEnabled") {
    activeYGuide = "yLt";
    ensureYGuideValue("yLt");
  } else if (id === "yGtEnabled") {
    activeYGuide = "yGt";
    ensureYGuideValue("yGt");
  } else if (id === "yLt" || id === "yGt") {
    activeYGuide = id;
  }
}

filterIds.forEach(id => el[id].addEventListener("change", () => {
  prepareFilterChange(id);
  loadState();
}));
[
  ["areaEnabled", "areaLt"],
  ["areaGtEnabled", "areaGt"],
  ["widthLtEnabled", "widthLt"],
  ["widthGtEnabled", "widthGt"],
  ["heightLtEnabled", "heightLt"],
  ["heightGtEnabled", "heightGt"],
  ["yLtEnabled", "yLt"],
  ["yGtEnabled", "yGt"],
].forEach(([checkboxId, inputId]) => {
  el[inputId].addEventListener("input", () => {
    if (el[inputId].value) el[checkboxId].checked = true;
    if (inputId === "yLt" || inputId === "yGt") {
      activeYGuide = inputId;
      updateLocalMatches();
      draw();
    }
  });
});
["yLt", "yGt"].forEach(inputId => {
  el[inputId].addEventListener("focus", () => {
    activeYGuide = inputId;
    draw();
  });
  el[`${inputId}Enabled`].addEventListener("change", () => {
    activeYGuide = inputId;
    ensureYGuideValue(inputId);
    draw();
  });
});
el.onlyEmpty.addEventListener("change", () => {
  if (el.onlyEmpty.checked) el.includeEmpty.checked = true;
});
el.onlyDropped.addEventListener("change", () => {
  if (el.onlyDropped.checked) el.includeEmpty.checked = true;
});
el.showFullImage.addEventListener("change", draw);
document.addEventListener("keydown", async (event) => {
  if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    if (event.key !== "Escape") return;
    document.activeElement.blur();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a") {
    event.preventDefault();
    selectAllBoxes();
  } else if ((event.ctrlKey || event.metaKey) && ["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(event.key)) {
    event.preventDefault();
    const step = event.altKey ? 10 : 1;
    const dx = event.key === "ArrowRight" ? step : (event.key === "ArrowLeft" ? -step : 0);
    const dy = event.key === "ArrowDown" ? step : (event.key === "ArrowUp" ? -step : 0);
    nudgeBox(dx, dy, event.shiftKey);
  } else if (event.key === "Escape") {
    addBoxMode = false;
    selectedBoxes.clear();
    draw();
  } else if (event.key === "n" || event.key === "ArrowRight") moveImage(1);
  else if (event.key === "p" || event.key === "ArrowLeft") moveImage(-1);
  else if (event.key === "j" || event.key === "ArrowDown") moveBox(1);
  else if (event.key === "k" || event.key === "ArrowUp") moveBox(-1);
  else if (event.key === "Delete" || event.key === "m") toggleCurrentMask();
  else if (event.key === "a") maskMatchedCurrent();
  else if (event.key === "x") toggleCurrentImageDrop();
  else if (event.key === "b") {
    addBoxMode = !addBoxMode;
    draw();
  }
  else if (event.key === "g") {
    el.jumpImage.focus();
    el.jumpImage.select();
  }
  else if (event.key === "r") await loadState(true);
});
loadState();
</script>
</body>
</html>
"""


MERGE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>merge candidate review</title>
  <style>
    :root { font-family: Arial, "Microsoft YaHei", sans-serif; color: #1f2328; background: #f6f8fa; }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body { height: 100vh; margin: 0; padding: 14px; overflow: hidden; }
    main { display: grid; grid-template-columns: minmax(340px, 440px) minmax(520px, 1fr) minmax(280px, 360px); grid-template-rows: auto minmax(0, 1fr); gap: 12px; height: calc(100vh - 28px); max-width: 1900px; margin: 0 auto; }
    header { grid-column: 1 / -1; display: flex; align-items: center; justify-content: space-between; gap: 12px; min-width: 0; }
    h1 { margin: 0; font-size: 20px; }
    .header-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .panel, .stage { min-width: 0; min-height: 0; background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .panel { display: flex; flex-direction: column; gap: 10px; overflow: auto; }
    .stage { display: grid; grid-template-rows: minmax(0, 1fr) auto; gap: 10px; background: #111; overflow: hidden; }
    .canvas-wrap { min-width: 0; min-height: 0; display: grid; place-items: center; overflow: hidden; }
    canvas { max-width: 100%; max-height: 100%; background: #000; }
    label { display: grid; gap: 4px; font-size: 13px; }
    input { width: 100%; min-width: 0; height: 30px; padding: 4px 8px; border: 1px solid #d0d7de; border-radius: 6px; }
    button, .link-button { display: inline-grid; place-items: center; height: 32px; padding: 0 10px; border: 1px solid #8c959f; border-radius: 6px; background: #fff; color: #1f2328; text-decoration: none; cursor: pointer; font-size: 13px; }
    button.primary { color: #fff; background: #1f883d; border-color: #1f883d; }
    .filters { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .candidate-list { display: grid; gap: 7px; overflow: auto; min-height: 0; }
    .candidate { display: grid; gap: 4px; padding: 8px; border: 1px solid #d0d7de; border-radius: 6px; background: #fff; cursor: pointer; }
    .candidate.active { border: 3px solid #0969da; background: #ddf4ff; }
    .candidate-title { font-weight: 700; font-size: 13px; }
    .muted, .status, .help, .detail { color: #57606a; font-size: 13px; line-height: 1.45; }
    .status.error { color: #cf222e; font-weight: 700; }
    .status.ok { color: #1f883d; font-weight: 700; }
    .preview-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; min-height: 150px; }
    .preview-panel { min-width: 0; display: grid; grid-template-rows: 20px minmax(0, 1fr); gap: 6px; padding: 8px; border: 1px solid #30363d; border-radius: 6px; background: #0d1117; color: #c9d1d9; }
    .preview-title { font-size: 13px; font-weight: 700; color: #f0f6fc; }
    .thumb-row { min-height: 112px; display: flex; gap: 8px; overflow-x: auto; overflow-y: hidden; }
    .thumb { flex: 0 0 auto; display: grid; grid-template-rows: minmax(0, 1fr) 18px; gap: 4px; min-width: 110px; max-width: 260px; height: 100px; padding: 4px; border: 1px solid #30363d; border-radius: 6px; background: #161b22; }
    .thumb canvas { width: 100%; height: 76px; object-fit: contain; }
    .thumb-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: center; font-size: 12px; }
    .actions { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    kbd { display: inline-grid; place-items: center; min-width: 42px; padding: 2px 5px; border: 1px solid #d0d7de; border-radius: 4px; background: #f6f8fa; color: #1f2328; }
    @media (max-width: 1100px) {
      body { overflow: auto; height: auto; }
      main { grid-template-columns: 1fr; grid-template-rows: auto minmax(260px, 42vh) minmax(220px, 1fr) auto; height: auto; min-height: calc(100vh - 28px); }
      .preview-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <h1>区域融合审批</h1>
    <div class="header-actions">
      <a class="link-button" href="/">返回普通审核</a>
      <button id="reload">重新载入</button>
    </div>
  </header>
  <aside class="panel">
    <div class="filters">
      <label>距离
        <input id="mergeGapRatio" type="number" min="0" step="0.05" value="0.65">
      </label>
      <label>高度差
        <input id="mergeHeightTolerance" type="number" min="0" step="0.05" value="0.22">
      </label>
      <label>中线差
        <input id="mergeCenterTolerance" type="number" min="0" step="0.05" value="0.45">
      </label>
    </div>
    <div class="status" id="status">加载中...</div>
    <div class="candidate-list" id="candidateList"></div>
  </aside>
  <section class="stage">
    <div class="canvas-wrap"><canvas id="canvas"></canvas></div>
    <div class="preview-grid">
      <div class="preview-panel">
        <div class="preview-title">融合前</div>
        <div class="thumb-row" id="beforeThumbs"></div>
      </div>
      <div class="preview-panel">
        <div class="preview-title">融合后</div>
        <div class="thumb-row" id="afterThumbs"></div>
      </div>
    </div>
  </section>
  <aside class="panel">
    <div class="actions">
      <button class="primary" id="approve">审批融合</button>
      <button id="prev">上一条</button>
      <button id="next">下一条</button>
    </div>
    <div class="detail" id="detail">没有候选</div>
    <hr>
    <div class="help">
      <div><kbd>Enter/A</kbd> 审批当前融合</div>
      <div><kbd>J/→</kbd> 下一条；<kbd>K/←</kbd> 上一条</div>
      <div><kbd>R</kbd> 重新载入候选</div>
    </div>
  </aside>
</main>
<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const candidateList = document.getElementById("candidateList");
const beforeThumbs = document.getElementById("beforeThumbs");
const afterThumbs = document.getElementById("afterThumbs");
const detail = document.getElementById("detail");
const status = document.getElementById("status");
const controls = {
  mergeGapRatio: document.getElementById("mergeGapRatio"),
  mergeHeightTolerance: document.getElementById("mergeHeightTolerance"),
  mergeCenterTolerance: document.getElementById("mergeCenterTolerance"),
};
let items = [];
let activeIndex = 0;
let approvedIds = new Set();
let approvedCount = 0;
let img = new Image();
let busy = false;

function qs() {
  const p = new URLSearchParams();
  if (controls.mergeGapRatio.value) p.set("merge_gap_ratio", controls.mergeGapRatio.value);
  if (controls.mergeHeightTolerance.value) p.set("merge_height_tolerance", controls.mergeHeightTolerance.value);
  if (controls.mergeCenterTolerance.value) p.set("merge_center_tolerance", controls.mergeCenterTolerance.value);
  return p.toString();
}

async function loadCandidates(keepId = "") {
  setStatus("加载中...");
  const payload = await fetch("/api/merge-candidates?" + qs()).then(response => response.json());
  items = payload.items || [];
  approvedIds = new Set([...approvedIds].filter(id => items.some(item => item.id === id)));
  const keepIndex = keepId ? items.findIndex(item => item.id === keepId) : activeIndex;
  activeIndex = Math.max(0, Math.min(items.length - 1, keepIndex >= 0 ? keepIndex : activeIndex));
  setStatus(`待审批 ${items.length} 条 | 已审批 ${approvedCount} 条 | samples=${payload.samples_dir}`);
  renderList();
  await loadImage();
}

async function loadImage() {
  const item = current();
  if (!item) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    detail.textContent = "没有候选";
    beforeThumbs.textContent = "没有候选区域";
    afterThumbs.textContent = "没有候选区域";
    renderList();
    return;
  }
  img = new Image();
  img.onload = () => {
    canvas.width = item.width;
    canvas.height = item.height;
    draw();
  };
  img.src = "/images/" + encodeURIComponent(item.image);
}

function current() {
  return items[activeIndex];
}

function draw() {
  const item = current();
  if (!item) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  for (const box of item.boxes) drawBox(box, "#2da44e", `#${box.index}`);
  drawBox(item.rect, "#0969da", "merge", true);
  detail.textContent = `${activeIndex + 1}/${items.length} ${item.image} | ${item.width}x${item.height} | 区域 ${item.indices.map(index => "#" + index).join(" + ")} | 合并框 x=${item.rect.x} y=${item.rect.y} w=${item.rect.width} h=${item.rect.height}`;
  renderThumbs(item);
  renderList();
}

function drawBox(box, color, label, dashed = false) {
  ctx.save();
  ctx.lineWidth = dashed ? 5 : 3;
  ctx.strokeStyle = color;
  ctx.setLineDash(dashed ? [14, 8] : []);
  ctx.strokeRect(box.x, box.y, box.width, box.height);
  ctx.font = "bold 18px Arial";
  const labelWidth = ctx.measureText(label).width + 12;
  ctx.fillStyle = color;
  ctx.fillRect(box.x + 4, Math.max(4, box.y + 4), labelWidth, 24);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, box.x + 10, Math.max(23, box.y + 23));
  ctx.restore();
}

function renderThumbs(item) {
  beforeThumbs.innerHTML = "";
  afterThumbs.innerHTML = "";
  for (const box of item.boxes) {
    beforeThumbs.appendChild(makeThumb(box, item, `#${box.index}`));
  }
  afterThumbs.appendChild(makeThumb(item.rect, item, "融合后"));
}

function makeThumb(box, item, text) {
    const source = clampSourceRect(box, item);
    if (!source.width || !source.height) {
      const empty = document.createElement("div");
      empty.className = "thumb";
      empty.textContent = "无有效裁剪";
      return empty;
    }
    const thumb = document.createElement("div");
    thumb.className = "thumb";
    const cropCanvas = document.createElement("canvas");
    const scale = Math.min(2, 230 / source.width, 76 / source.height);
    cropCanvas.width = Math.max(24, Math.round(source.width * scale));
    cropCanvas.height = Math.max(24, Math.round(source.height * scale));
    cropCanvas.getContext("2d").drawImage(img, source.x, source.y, source.width, source.height, 0, 0, cropCanvas.width, cropCanvas.height);
    const label = document.createElement("div");
    label.className = "thumb-label";
    label.textContent = text;
    thumb.append(cropCanvas, label);
    return thumb;
}

function clampSourceRect(box, item) {
  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const right = Math.min(item.width, Math.ceil(box.x + box.width));
  const bottom = Math.min(item.height, Math.ceil(box.y + box.height));
  return {x, y, width: Math.max(0, right - x), height: Math.max(0, bottom - y)};
}

function renderList() {
  if (!items.length) {
    candidateList.textContent = "没有待审批融合候选";
    return;
  }
  candidateList.innerHTML = items.map((item, i) => `
    <div class="candidate ${i === activeIndex ? "active" : ""}" data-i="${i}">
      <div class="candidate-title">${i + 1}. ${item.image}</div>
      <div>区域 ${item.indices.map(index => "#" + index).join(" + ")}</div>
      <div class="muted">x=${item.rect.x} y=${item.rect.y} w=${item.rect.width} h=${item.rect.height}${approvedIds.has(item.id) ? " | 已审批" : ""}</div>
    </div>`).join("");
  candidateList.querySelectorAll(".candidate").forEach(node => {
    node.onclick = () => selectCandidate(Number(node.dataset.i));
  });
  candidateList.querySelector(".candidate.active")?.scrollIntoView({block: "nearest"});
}

async function selectCandidate(index) {
  if (!items.length) return;
  activeIndex = Math.max(0, Math.min(items.length - 1, index));
  await loadImage();
}

async function approveCurrent() {
  const item = current();
  if (!item || busy) return;
  busy = true;
  setStatus(`审批中：${item.image} ${item.indices.map(index => "#" + index).join(" + ")}`);
  try {
    const response = await fetch("/api/merge", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({stem: item.stem, indices: item.indices})
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    approvedIds.add(item.id);
    approvedCount += 1;
    const nextIndex = Math.min(activeIndex, Math.max(0, items.length - 2));
    items.splice(activeIndex, 1);
    activeIndex = nextIndex;
    setStatus(`已审批并写回 | 已审批 ${approvedCount} 条`, "ok");
    renderList();
    await loadImage();
  } catch (error) {
    setStatus(`审批失败：${error.message || "unknown error"}`, "error");
  } finally {
    busy = false;
  }
}

async function move(delta) {
  await selectCandidate(activeIndex + delta);
}

function setStatus(message, kind = "") {
  status.classList.toggle("error", kind === "error");
  status.classList.toggle("ok", kind === "ok");
  status.textContent = message;
}

document.getElementById("approve").onclick = approveCurrent;
document.getElementById("prev").onclick = () => move(-1);
document.getElementById("next").onclick = () => move(1);
document.getElementById("reload").onclick = () => loadCandidates(current()?.id || "");
Object.values(controls).forEach(input => input.addEventListener("change", () => loadCandidates()));
document.addEventListener("keydown", event => {
  if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    if (event.key !== "Escape") return;
    document.activeElement.blur();
    return;
  }
  const key = event.key.toLowerCase();
  if (key === "enter" || key === "a") {
    event.preventDefault();
    approveCurrent();
  } else if (key === "j" || event.key === "ArrowRight") {
    event.preventDefault();
    move(1);
  } else if (key === "k" || event.key === "ArrowLeft") {
    event.preventDefault();
    move(-1);
  } else if (key === "r") {
    event.preventDefault();
    loadCandidates(current()?.id || "");
  }
});
loadCandidates();
</script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review generated_samples YOLO labels and mark boxes as suppressed without deleting label lines."
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("data/generated_samples"),
        help="Directory containing images/, labels/, and optionally annotations.jsonl.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser automatically.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        app = ReviewApp(args.samples_dir)
        server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/"
    print(f"serving {app.samples_dir}")
    print(f"mask markers: {app.masks_path}")
    print(f"open: {url}")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
