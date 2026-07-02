from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import mimetypes
import re
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


REVIEW_FILENAME = "segment_review.json"
SIMILAR_TEXT_RATIO = 0.86
SIMILAR_TEXT_MAX_EDIT_DISTANCE = 3
SHORT_TEXT_MAX_EDIT_DISTANCE = 1


class SegmentReviewApp:
    def __init__(self, samples_dir: Path, *, verbose_requests: bool = False) -> None:
        self.samples_dir = samples_dir.resolve()
        self.images_dir = self.samples_dir / "images"
        self.annotations_path = self.samples_dir / "annotations.jsonl"
        self.review_path = self.samples_dir / REVIEW_FILENAME
        self.verbose_requests = verbose_requests
        if not self.images_dir.is_dir():
            raise ValueError(f"missing images dir: {self.images_dir}")
        if not self.annotations_path.exists():
            raise ValueError(f"missing annotations file: {self.annotations_path}")
        self.items = self.load_items()
        self.review = self.load_review()

    def load_items(self) -> list[dict[str, object]]:
        items = []
        with self.annotations_path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item["index"] = index
                item["id"] = item_id(item, index)
                item["stem"] = Path(str(item["image"])).stem
                items.append(item)
        return items

    def load_review(self) -> dict[str, dict[str, object]]:
        if not self.review_path.exists():
            return {}
        data = json.loads(self.review_path.read_text(encoding="utf-8"))
        if "items" not in data:
            return {}
        loaded = dict(data["items"])
        if data.get("version") != 2:
            return self.migrate_legacy_review(loaded)
        return loaded

    def migrate_legacy_review(self, legacy: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
        review = {}
        for item in self.items:
            marker = str(item.get("segment_marker") or item["id"])
            old = legacy.get(marker, {})
            segment_id = str(old.get("replacement_marker") or marker)
            review[str(item["id"])] = {
                "has_subtitle": bool(item.get("has_subtitle")),
                "segment_id": segment_id,
                "note": str(old.get("note", "")),
                "updated_at": int(time.time()),
            }
        return review

    def save_review(self) -> None:
        payload = {
            "version": 2,
            "description": "Manual ROI subtitle presence and segment identity labeling. Same segment_id means same subtitle segment.",
            "items": self.review,
        }
        self.review_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def reviewed_item(self, item: dict[str, object]) -> dict[str, object]:
        review = self.review.get(str(item["id"]), {})
        initial_segment_id = str(item.get("segment_marker") or item["id"])
        has_subtitle = bool(review.get("has_subtitle", item.get("has_subtitle", False)))
        segment_id = str(review.get("segment_id") or initial_segment_id)
        return {
            **item,
            "initial_segment_id": initial_segment_id,
            "review_has_subtitle": has_subtitle,
            "review_segment_id": segment_id,
            "review_note": str(review.get("note", "")),
            "reviewed": str(item["id"]) in self.review,
            "segment_changed": segment_id != initial_segment_id,
        }

    def state(
        self,
        *,
        similarity_ratio: float = SIMILAR_TEXT_RATIO,
        edit_distance_tolerance: int = SIMILAR_TEXT_MAX_EDIT_DISTANCE,
    ) -> dict[str, object]:
        items = [self.reviewed_item(item) for item in self.items]
        groups_by_id: dict[str, list[dict[str, object]]] = {}
        for item in items:
            if not bool(item["review_has_subtitle"]):
                continue
            groups_by_id.setdefault(str(item["review_segment_id"]), []).append(item)

        groups: list[dict[str, object]] = []
        empty_run = 0
        for item in items:
            if not bool(item["review_has_subtitle"]):
                last = groups[-1] if groups else None
                if not last or not last.get("no_subtitle"):
                    empty_run += 1
                    groups.append(
                        {
                            "segment_id": f"__no_subtitle_{empty_run}__",
                            "count": 0,
                            "items": [],
                            "note": "",
                            "no_subtitle": True,
                            "empty_run": empty_run,
                        }
                    )
                groups[-1]["items"].append(item)  # type: ignore[index,union-attr]
                groups[-1]["count"] = len(groups[-1]["items"])  # type: ignore[arg-type,index]
                continue

            segment_id = str(item["review_segment_id"])
            last = groups[-1] if groups else None
            if not last or last.get("no_subtitle") or str(last["segment_id"]) != segment_id:
                groups.append(
                    {
                        "segment_id": segment_id,
                        "count": 0,
                        "items": [],
                        "note": "",
                    }
                )
            groups[-1]["items"].append(item)  # type: ignore[index,union-attr]
            groups[-1]["count"] = len(groups[-1]["items"])  # type: ignore[arg-type,index]
            groups[-1]["note"] = first_note(groups[-1]["items"])  # type: ignore[arg-type,index]

        return {
            "samples_dir": str(self.samples_dir),
            "review_file": str(self.review_path),
            "items": items,
            "groups": groups,
            "similar_candidates": self.similar_candidates(
                items,
                similarity_ratio=similarity_ratio,
                edit_distance_tolerance=edit_distance_tolerance,
            ),
            "candidate_thresholds": {
                "similarity_ratio": similarity_ratio,
                "edit_distance_tolerance": edit_distance_tolerance,
            },
            "stats": {
                "samples": len(items),
                "subtitle_samples": sum(1 for item in items if bool(item["review_has_subtitle"])),
                "empty_samples": sum(1 for item in items if not bool(item["review_has_subtitle"])),
                "segments": len(groups_by_id),
                "edited_samples": len(self.review),
            },
        }

    def set_item(self, sample_id: str, has_subtitle: bool, segment_id: str, note: str) -> None:
        item = self.find_item(sample_id)
        default_segment_id = str(item.get("segment_marker") or sample_id)
        normalized_segment_id = segment_id.strip() or default_segment_id
        self.set_review_record(sample_id, has_subtitle, normalized_segment_id, note)
        self.save_review()

    def set_review_record(self, sample_id: str, has_subtitle: bool, segment_id: str, note: str) -> None:
        self.review[sample_id] = {
            "has_subtitle": has_subtitle,
            "segment_id": segment_id,
            "note": note.strip(),
            "updated_at": int(time.time()),
        }

    def set_group_segment(self, old_segment_id: str, new_segment_id: str) -> None:
        normalized = new_segment_id.strip()
        if not normalized:
            raise ValueError("segment_id cannot be empty")
        for item in self.items:
            reviewed = self.reviewed_item(item)
            if bool(reviewed["review_has_subtitle"]) and reviewed["review_segment_id"] == old_segment_id:
                self.set_review_record(str(item["id"]), True, normalized, str(reviewed.get("review_note", "")))
        self.save_review()

    def split_segment_at(self, sample_id: str) -> str:
        selected = self.reviewed_item(self.find_item(sample_id))
        if not bool(selected["review_has_subtitle"]):
            raise ValueError("cannot split at a no-subtitle sample")
        old_segment_id = str(selected["review_segment_id"])
        selected_index = int(selected["index"])
        new_segment_id = self.unique_segment_id(sample_id, old_segment_id)
        changed = 0
        for item in self.items:
            reviewed = self.reviewed_item(item)
            if (
                bool(reviewed["review_has_subtitle"])
                and str(reviewed["review_segment_id"]) == old_segment_id
                and int(reviewed["index"]) >= selected_index
            ):
                self.set_review_record(str(item["id"]), True, new_segment_id, str(reviewed.get("review_note", "")))
                changed += 1
        if changed == 0:
            raise ValueError("nothing to split")
        self.save_review()
        return new_segment_id

    def merge_segment_to_previous(self, sample_id: str) -> str:
        selected = self.reviewed_item(self.find_item(sample_id))
        if not bool(selected["review_has_subtitle"]):
            raise ValueError("cannot merge a no-subtitle sample")
        current_segment_id = str(selected["review_segment_id"])
        groups = [group for group in self.state()["groups"] if not group.get("no_subtitle")]
        group_index = next(
            (
                index
                for index, group in enumerate(groups)
                if str(group["segment_id"]) == current_segment_id
            ),
            -1,
        )
        if group_index <= 0:
            raise ValueError("there is no previous subtitle segment")
        previous_segment_id = str(groups[group_index - 1]["segment_id"])
        for item in self.items:
            reviewed = self.reviewed_item(item)
            if bool(reviewed["review_has_subtitle"]) and str(reviewed["review_segment_id"]) == current_segment_id:
                self.set_review_record(str(item["id"]), True, previous_segment_id, str(reviewed.get("review_note", "")))
        self.save_review()
        return previous_segment_id

    def merge_item_segments(self, left_id: str, right_id: str) -> str:
        target_segment_id, changed = self.merge_item_segments_in_memory(left_id, right_id)
        if changed:
            self.save_review()
        return target_segment_id

    def merge_item_segments_in_memory(self, left_id: str, right_id: str) -> tuple[str, bool]:
        left = self.reviewed_item(self.find_item(left_id))
        right = self.reviewed_item(self.find_item(right_id))
        if not bool(left["review_has_subtitle"]) or not bool(right["review_has_subtitle"]):
            raise ValueError("cannot merge no-subtitle samples")
        target_segment_id = str(left["review_segment_id"])
        source_segment_id = str(right["review_segment_id"])
        if target_segment_id == source_segment_id:
            return target_segment_id, False
        changed = False
        for item in self.items:
            reviewed = self.reviewed_item(item)
            if bool(reviewed["review_has_subtitle"]) and str(reviewed["review_segment_id"]) == source_segment_id:
                self.set_review_record(
                    str(item["id"]),
                    True,
                    target_segment_id,
                    str(reviewed.get("review_note", "")),
                )
                changed = True
        return target_segment_id, changed

    def merge_item_segment_pairs(self, pairs: list[dict[str, object]]) -> int:
        changed = 0
        for pair in pairs:
            _, pair_changed = self.merge_item_segments_in_memory(
                left_id=str(pair["left_id"]),
                right_id=str(pair["right_id"]),
            )
            if pair_changed:
                changed += 1
        if changed:
            self.save_review()
        return changed

    def merge_exact_ocr_segments(self) -> int:
        changed = False
        changed_segments = 0
        previous: dict[str, object] | None = None
        for item in self.items:
            reviewed = self.reviewed_item(item)
            if not bool(reviewed["review_has_subtitle"]):
                previous = None
                continue
            text = item_ocr_normalized(reviewed)
            if not text:
                previous = reviewed
                continue
            if previous is not None:
                previous_text = item_ocr_normalized(previous)
                if previous_text == text and str(previous["review_segment_id"]) != str(reviewed["review_segment_id"]):
                    previous_segment_id = str(previous["review_segment_id"])
                    current_segment_id = str(reviewed["review_segment_id"])
                    for merge_item in self.items:
                        merge_reviewed = self.reviewed_item(merge_item)
                        if (
                            bool(merge_reviewed["review_has_subtitle"])
                            and str(merge_reviewed["review_segment_id"]) == current_segment_id
                        ):
                            note = str(merge_reviewed.get("review_note", ""))
                            if not note:
                                note = "merged by exact OCR text"
                            self.set_review_record(str(merge_item["id"]), True, previous_segment_id, note)
                    reviewed = self.reviewed_item(item)
                    changed = True
                    changed_segments += 1
            previous = reviewed
        if changed:
            self.save_review()
        return changed_segments

    def similar_candidates(
        self,
        items: list[dict[str, object]],
        *,
        similarity_ratio: float = SIMILAR_TEXT_RATIO,
        edit_distance_tolerance: int = SIMILAR_TEXT_MAX_EDIT_DISTANCE,
    ) -> list[dict[str, object]]:
        candidates = []
        previous: dict[str, object] | None = None
        for item in items:
            if not bool(item["review_has_subtitle"]):
                previous = None
                continue
            if previous is None:
                previous = item
                continue
            left_text = item_ocr_normalized(previous)
            right_text = item_ocr_normalized(item)
            same_segment = str(previous["review_segment_id"]) == str(item["review_segment_id"])
            if left_text and right_text and not same_segment:
                score = text_similarity(left_text, right_text)
                distance = edit_distance(left_text, right_text, max_distance=edit_distance_tolerance + 1)
                exact_match = left_text == right_text
                if exact_match or is_strict_ocr_near_duplicate(
                    left_text,
                    right_text,
                    score,
                    distance,
                    similarity_ratio=similarity_ratio,
                    edit_distance_tolerance=edit_distance_tolerance,
                ):
                    candidates.append(
                        {
                            "left_id": previous["id"],
                            "right_id": item["id"],
                            "left_segment_id": previous["review_segment_id"],
                            "right_segment_id": item["review_segment_id"],
                            "left_text": str(previous.get("ocr_text") or ""),
                            "right_text": str(item.get("ocr_text") or ""),
                            "left_text_normalized": left_text,
                            "right_text_normalized": right_text,
                            "match_type": "exact" if exact_match else "text-near",
                            "similarity": round(score, 4),
                            "edit_distance": distance,
                        }
                    )
            previous = item
        return candidates

    def unique_segment_id(self, base_segment_id: str, old_segment_id: str) -> str:
        existing = {
            str(self.reviewed_item(item)["review_segment_id"])
            for item in self.items
        }
        if base_segment_id != old_segment_id and base_segment_id not in existing:
            return base_segment_id
        suffix = 1
        while True:
            candidate = f"{base_segment_id}__split_{suffix}"
            if candidate not in existing:
                return candidate
            suffix += 1

    def find_item(self, sample_id: str) -> dict[str, object]:
        for item in self.items:
            if str(item["id"]) == sample_id:
                return item
        raise ValueError(f"unknown sample id: {sample_id}")


def item_id(item: dict[str, object], index: int) -> str:
    source_sample_id = str(item.get("source_sample_id", "")).strip()
    if source_sample_id:
        return source_sample_id
    stem = Path(str(item.get("image", f"sample_{index}"))).stem
    return stem or f"sample_{index}"


def first_note(items: list[dict[str, object]]) -> str:
    for item in items:
        note = str(item.get("review_note", ""))
        if note:
            return note
    return ""


def normalize_ocr_text(value: object) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", "", text)
    return text.casefold()


def item_ocr_normalized(item: dict[str, object]) -> str:
    return normalize_ocr_text(item.get("ocr_text_normalized") or item.get("omlx_text_normalized") or item.get("ocr_text") or item.get("omlx_text"))


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def is_strict_ocr_near_duplicate(
    left: str,
    right: str,
    score: float,
    distance: int,
    *,
    similarity_ratio: float = SIMILAR_TEXT_RATIO,
    edit_distance_tolerance: int = SIMILAR_TEXT_MAX_EDIT_DISTANCE,
) -> bool:
    shorter = min(len(left), len(right))
    if shorter == 0:
        return False
    if shorter < 6 and edit_distance_tolerance == SIMILAR_TEXT_MAX_EDIT_DISTANCE:
        return distance <= SHORT_TEXT_MAX_EDIT_DISTANCE and score >= min(similarity_ratio, 0.8)
    return score >= similarity_ratio and distance <= edit_distance_tolerance


def edit_distance(left: str, right: str, *, max_distance: int | None = None) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if max_distance is not None and abs(len(left) - len(right)) > max_distance:
        return max_distance
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        row_min = current[0]
        for right_index, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            current.append(value)
            row_min = min(row_min, value)
        if max_distance is not None and row_min > max_distance:
            return max_distance
        previous = current
    return previous[-1]


def parse_similarity_ratio(value: object) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return SIMILAR_TEXT_RATIO
    return max(0.0, min(1.0, ratio))


def parse_edit_distance_tolerance(value: object) -> int:
    try:
        tolerance = int(value)
    except (TypeError, ValueError):
        return SIMILAR_TEXT_MAX_EDIT_DISTANCE
    return max(0, min(30, tolerance))


def make_handler(app: SegmentReviewApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            if app.verbose_requests:
                print(format % args, file=sys.stderr)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.write_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/ocr-review":
                self.write_bytes(OCR_REVIEW_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                self.write_json(
                    app.state(
                        similarity_ratio=parse_similarity_ratio(query.get("similarity_ratio", [""])[0]),
                        edit_distance_tolerance=parse_edit_distance_tolerance(
                            query.get("edit_distance_tolerance", [""])[0]
                        ),
                    )
                )
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
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            try:
                if parsed.path == "/api/item":
                    app.set_item(
                        sample_id=str(payload["id"]),
                        has_subtitle=bool(payload["has_subtitle"]),
                        segment_id=str(payload.get("segment_id", "")),
                        note=str(payload.get("note", "")),
                    )
                    self.write_json({"ok": True})
                    return
                if parsed.path == "/api/group-segment":
                    app.set_group_segment(
                        old_segment_id=str(payload["old_segment_id"]),
                        new_segment_id=str(payload["new_segment_id"]),
                    )
                    self.write_json({"ok": True})
                    return
                if parsed.path == "/api/split-segment-at":
                    new_segment_id = app.split_segment_at(sample_id=str(payload["id"]))
                    self.write_json({"ok": True, "segment_id": new_segment_id})
                    return
                if parsed.path == "/api/merge-segment-to-previous":
                    segment_id = app.merge_segment_to_previous(sample_id=str(payload["id"]))
                    self.write_json({"ok": True, "segment_id": segment_id})
                    return
                if parsed.path == "/api/merge-similar-candidate":
                    segment_id = app.merge_item_segments(
                        left_id=str(payload["left_id"]),
                        right_id=str(payload["right_id"]),
                    )
                    self.write_json({"ok": True, "segment_id": segment_id})
                    return
                if parsed.path == "/api/merge-similar-candidates":
                    changed = app.merge_item_segment_pairs(list(payload.get("pairs", [])))
                    self.write_json({"ok": True, "changed": changed})
                    return
                if parsed.path == "/api/merge-exact-ocr":
                    changed = app.merge_exact_ocr_segments()
                    self.write_json({"ok": True, "changed": changed})
                    return
            except Exception as exc:
                self.send_error(400, str(exc))
                return
            self.send_error(404)

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
  <title>ROI 字幕人工标注</title>
  <style>
    :root {
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: #202124;
      background: #f6f7f9;
      --border: #d7dce2;
      --muted: #5f6368;
      --accent: #0b57d0;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      height: 100vh;
      background: #f6f7f9;
    }
    aside {
      min-width: 0;
      min-height: 0;
      border-right: 1px solid var(--border);
      background: #fff;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      overflow: hidden;
    }
    section {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr) auto;
      overflow: hidden;
    }
    h1 { margin: 0 0 6px; font-size: 18px; }
    .meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      word-break: break-all;
    }
    .sidebar-head {
      padding: 12px;
      border-bottom: 1px solid var(--border);
    }
    .segments {
      display: grid;
      align-content: start;
      gap: 4px;
      overflow: auto;
      padding: 8px;
    }
    .segment {
      width: 100%;
      min-height: 42px;
      text-align: left;
      border: 1px solid transparent;
      border-left: 4px solid transparent;
      background: #fff;
      padding: 7px 8px;
      cursor: pointer;
    }
    .segment.active {
      border-color: var(--accent);
      border-left-color: var(--accent);
      background: #eef4ff;
    }
    .segment.empty {
      color: var(--muted);
      background: #fafbfc;
      border-left-color: #9aa0a6;
    }
    .segment-id {
      font-size: 13px;
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      padding: 12px 14px;
      background: #fff;
      border-bottom: 1px solid var(--border);
    }
    .actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-start;
    }
    button {
      min-height: 34px;
      border: 1px solid #b7bec7;
      background: #fff;
      cursor: pointer;
      padding: 0 12px;
      font-size: 13px;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    button:disabled {
      background: #f1f3f4;
      border-color: #d7dce2;
      color: #9aa0a6;
      cursor: default;
    }
    button.danger { border-color: #d93025; color: #d93025; }
    input {
      height: 34px;
      border: 1px solid #b7bec7;
      padding: 4px 8px;
      min-width: 260px;
      font-size: 13px;
    }
    .status {
      padding: 8px 14px;
      color: var(--muted);
      font-size: 13px;
      background: #fff;
      border-bottom: 1px solid var(--border);
      min-height: 34px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .mode.active, .toggle.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .shortcuts {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      padding: 8px 14px;
      background: #fff;
      border-bottom: 1px solid #dfe3e8;
    }
    .shortcuts button {
      min-height: 30px;
      padding: 0 9px;
      font-size: 12px;
    }
    .workspace {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 12px;
      padding: 12px;
      overflow: hidden;
    }
    .stage {
      min-width: 0;
      min-height: 0;
      background: #fff;
      border: 1px solid var(--border);
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
    }
    .stage-image {
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 12px;
      background: #111316;
    }
    .stage-image img {
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
      cursor: zoom-in;
    }
    .editor {
      background: #fff;
      border-top: 1px solid var(--border);
      display: grid;
      gap: 8px;
      padding: 12px;
    }
    .editor input { min-width: 0; width: 100%; }
    .editor-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 8px;
    }
    .inspector {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 10px;
      overflow: hidden;
    }
    .row {
      display: flex;
      gap: 5px;
      align-items: center;
      flex-wrap: wrap;
    }
    .row.space { justify-content: space-between; }
    .sample-id {
      font-size: 13px;
      font-weight: 700;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .ocr-box {
      min-height: 88px;
      padding: 10px;
      border: 1px solid var(--border);
      background: #fff;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.5;
    }
    .context-strip {
      min-height: 0;
      display: grid;
      gap: 6px;
    }
    .context-strip:empty { display: none; }
    .context-row {
      min-width: 0;
      border: 1px solid #ccd4dd;
      background: #fff;
      padding: 7px;
      display: grid;
      gap: 6px;
    }
    .context-row.empty {
      border-left: 4px solid #9aa0a6;
      background: #fafbfc;
    }
    .context-head {
      min-width: 0;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .context-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
      color: #303134;
    }
    .context-thumbs {
      display: flex;
      gap: 5px;
      overflow: hidden;
    }
    .context-thumb {
      width: 64px;
      height: 34px;
      object-fit: contain;
      background: #101214;
      border: 1px solid #dfe3e8;
      cursor: zoom-in;
      flex: 0 0 auto;
    }
    .context-more {
      height: 34px;
      display: inline-flex;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      flex: 0 0 auto;
    }
    .filmstrip {
      min-height: 0;
      overflow: auto;
      display: grid;
      align-content: start;
      gap: 6px;
    }
    .card {
      min-width: 0;
      background: #fff;
      border: 1px solid var(--border);
      display: grid;
      grid-template-columns: 108px minmax(0, 1fr);
      gap: 8px;
      padding: 6px;
      cursor: pointer;
    }
    .card.active {
      border-color: var(--accent);
      outline: 2px solid var(--accent);
      outline-offset: -2px;
      background: #eef4ff;
    }
    .card.empty { opacity: 0.72; }
    .roi {
      width: 100%;
      height: 58px;
      object-fit: contain;
      background: #101214;
      display: block;
      cursor: zoom-in;
    }
    .card-main { min-width: 0; display: grid; gap: 4px; align-content: center; }
    .badge { font-size: 12px; color: var(--muted); }
    .viewer {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.86);
      z-index: 20;
      padding: 24px;
    }
    .viewer.open { display: flex; }
    .viewer img {
      max-width: 96vw;
      max-height: 88vh;
      object-fit: contain;
      background: #000;
    }
    @media (max-width: 1100px) {
      main { grid-template-columns: 240px minmax(0, 1fr); }
      .workspace { grid-template-columns: minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) 220px; }
    }
  </style>
</head>
<body>
<main id="app">
  <aside>
    <div class="sidebar-head">
      <h1>字幕分组</h1>
      <div class="meta" id="meta">加载中...</div>
    </div>
    <div class="segments" id="segments"></div>
  </aside>
  <section>
    <div class="topbar">
      <div>
        <h1 id="title">未选择</h1>
        <div class="meta" id="detail"></div>
      </div>
      <div class="actions">
        <button id="openCandidateReview" class="primary">OCR 合并</button>
        <button id="subtitleOnly" class="mode">仅显示有字幕</button>
        <input id="groupSegment" placeholder="当前组 segment_id">
        <button id="renameGroup">保存组</button>
      </div>
    </div>
    <div class="shortcuts">
      <button id="prevGroup">↑ 上组</button>
      <button id="nextGroup">↓ 下组</button>
      <button id="selectPrev">← 前图</button>
      <button id="selectNext">→ 后图</button>
      <button id="toggleCurrent">Space 有/无字幕</button>
      <button id="splitCurrent">X 从此处打断</button>
      <button id="mergeCurrentGroup">M 当前段并入上一段</button>
      <button id="saveCurrent">Enter 保存</button>
      <button id="zoomCurrent">V 放大</button>
      <button id="editSegment">E 改段ID</button>
      <button id="editNote">N 备注</button>
    </div>
    <div class="workspace">
      <div class="stage">
        <div class="stage-image"><img id="stageImage" alt=""></div>
        <div class="editor">
          <div class="row space">
            <div class="sample-id" id="sampleId"></div>
            <button id="toggleEditor" class="toggle">有字幕</button>
          </div>
          <div class="editor-grid">
            <input id="segmentInput" placeholder="segment_id">
            <input id="noteInput" placeholder="备注">
          </div>
          <div class="ocr-box" id="ocrText"></div>
        </div>
      </div>
      <div class="inspector">
        <div class="context-strip" id="contextStrip"></div>
        <div class="status" id="status">↑/↓ 换分组，←/→ 换图片。方向与界面方向一致。</div>
        <div class="filmstrip" id="grid"></div>
      </div>
    </div>
  </section>
</main>
<div class="viewer" id="viewer"><img id="viewerImg" alt=""></div>
<script>
let state = null;
let showOnlySubtitle = false;
let groupIndex = 0;
let selectedItemIndex = 0;
const segmentsEl = document.getElementById("segments");
const gridEl = document.getElementById("grid");
const contextStripEl = document.getElementById("contextStrip");
const appEl = document.getElementById("app");
const metaEl = document.getElementById("meta");
const titleEl = document.getElementById("title");
const detailEl = document.getElementById("detail");
const statusEl = document.getElementById("status");
const groupSegmentEl = document.getElementById("groupSegment");
const viewerEl = document.getElementById("viewer");
const viewerImg = document.getElementById("viewerImg");
const stageImageEl = document.getElementById("stageImage");
const sampleIdEl = document.getElementById("sampleId");
const toggleEditorEl = document.getElementById("toggleEditor");
const segmentInputEl = document.getElementById("segmentInput");
const noteInputEl = document.getElementById("noteInput");
const ocrTextEl = document.getElementById("ocrText");

async function loadState(keepSegment = "", keepItemId = "") {
  state = await fetch("/api/state").then(r => r.json());
  const groups = visibleGroups();
  if (keepSegment) {
    const found = groups.findIndex(group => group.segment_id === keepSegment);
    if (found >= 0) groupIndex = found;
  }
  if (groupIndex >= groups.length) groupIndex = Math.max(0, groups.length - 1);
  if (keepItemId) {
    const foundGroup = groups.findIndex(group => group.items.some(item => String(item.id) === String(keepItemId)));
    if (foundGroup >= 0) {
      groupIndex = foundGroup;
      selectedItemIndex = visibleItems().findIndex(item => String(item.id) === String(keepItemId));
    }
  }
  render();
}

function visibleGroups() {
  const groups = state?.groups || [];
  return showOnlySubtitle ? groups.filter(group => !group.no_subtitle) : groups;
}

function currentGroup() {
  return visibleGroups()[groupIndex];
}

function visibleItems() {
  const items = currentGroup()?.items || [];
  return showOnlySubtitle ? items.filter(item => item.review_has_subtitle) : items;
}

function render() {
  const stats = state.stats;
  const groups = visibleGroups();
  if (groupIndex >= groups.length) groupIndex = Math.max(0, groups.length - 1);
  metaEl.textContent = `${stats.samples} 张 | 有字幕 ${stats.subtitle_samples} | 无字幕 ${stats.empty_samples} | 字幕段 ${stats.segments} | 已改 ${stats.edited_samples} | OCR 候选 ${state.similar_candidates.length} | ${state.review_file}`;
  document.getElementById("subtitleOnly").classList.toggle("active", showOnlySubtitle);
  segmentsEl.innerHTML = groups.map((group, index) => `
    <button class="segment ${index === groupIndex ? "active" : ""} ${group.no_subtitle ? "empty" : ""}" data-index="${index}">
      <div class="segment-id">${escapeHtml(groupLabel(group))}</div>
      <div class="meta">${group.count} 张${group.no_subtitle ? " | 组间空字幕" : ""}${group.note ? " | " + escapeHtml(group.note) : ""}</div>
    </button>
  `).join("");
  segmentsEl.querySelectorAll("button").forEach(button => {
    button.onclick = () => {
      groupIndex = Number(button.dataset.index);
      selectedItemIndex = 0;
      render();
    };
  });

  const group = currentGroup();
  const items = visibleItems();
  if (!items.length) {
    titleEl.textContent = "没有符合条件的样本";
    detailEl.textContent = showOnlySubtitle ? "当前筛选只显示有字幕内容" : "";
    contextStripEl.innerHTML = "";
    gridEl.innerHTML = "";
    stageImageEl.removeAttribute("src");
    sampleIdEl.textContent = "";
    ocrTextEl.textContent = "";
    return;
  }
  titleEl.textContent = groupLabel(group);
  detailEl.textContent = `${items.length} 张${showOnlySubtitle ? " | 仅有字幕" : ""}${group.no_subtitle ? " | 组间空字幕，可检查它是否切断前后字幕组" : " | 检查当前组是否为同一句字幕"}`;
  groupSegmentEl.value = group.no_subtitle ? "" : group.segment_id;
  groupSegmentEl.disabled = Boolean(group.no_subtitle);
  document.getElementById("renameGroup").disabled = Boolean(group.no_subtitle);
  selectedItemIndex = Math.max(0, Math.min(items.length - 1, selectedItemIndex));
  renderContextStrip(group);
  gridEl.innerHTML = items.map(item => renderCard(item)).join("");
  wireCards();
  updateActiveSegment(true);
  updateSelectedCard(true);
}

function renderContextStrip(group) {
  if (showOnlySubtitle || !state?.groups?.length) {
    contextStripEl.innerHTML = "";
    return;
  }
  const groups = state.groups;
  const allIndex = groups.indexOf(group);
  if (allIndex < 0) {
    contextStripEl.innerHTML = "";
    return;
  }
  const rows = [
    renderNeighborGroup(groups[allIndex - 1], allIndex - 1, "上一组"),
    renderNeighborGroup(groups[allIndex + 1], allIndex + 1, "下一组"),
  ].filter(Boolean);
  contextStripEl.innerHTML = rows.join("");
  contextStripEl.querySelectorAll(".context-thumb").forEach(img => {
    img.onclick = event => {
      event.stopPropagation();
      viewerImg.src = img.dataset.full;
      viewerEl.classList.add("open");
    };
  });
  contextStripEl.querySelectorAll(".context-row").forEach(row => {
    row.onclick = () => selectGroupByAllIndex(Number(row.dataset.allIndex));
  });
}

function renderNeighborGroup(group, allIndex, label) {
  if (!group) return "";
  const thumbs = group.items.slice(0, 4).map(item => {
    const image = `/images/${encodeURIComponent(imageName(item.image))}`;
    return `<img class="context-thumb" src="${image}" data-full="${image}" alt="">`;
  }).join("");
  const more = group.items.length > 4 ? `<span class="context-more">+${group.items.length - 4}</span>` : "";
  return `
    <div class="context-row ${group.no_subtitle ? "empty" : ""}" data-all-index="${allIndex}">
      <div class="context-head">
        <span>${escapeHtml(label)}</span>
        <span>${group.count} 张${group.no_subtitle ? " 空字幕" : " 有字幕"}</span>
      </div>
      <div class="context-title">${escapeHtml(groupLabel(group))}</div>
      <div class="context-thumbs">${thumbs}${more}</div>
    </div>
  `;
}

function renderCard(item) {
  const image = `/images/${encodeURIComponent(imageName(item.image))}`;
  return `
    <div class="card ${item.review_has_subtitle ? "" : "empty"}" data-id="${escapeAttr(item.id)}" tabindex="0">
      <img class="roi" src="${image}" data-full="${image}" alt="">
      <div class="card-main">
        <div class="sample-id">${escapeHtml(item.id)}</div>
        <div class="badge">${item.review_has_subtitle ? "有字幕" : "无字幕"} | frame=${escapeHtml(frameIndex(item))}</div>
        <div class="meta">OCR=${escapeHtml(item.ocr_text || "")}</div>
      </div>
    </div>
  `;
}

function wireCards() {
  gridEl.querySelectorAll(".roi").forEach(img => {
    img.onclick = () => {
      viewerImg.src = img.dataset.full;
      viewerEl.classList.add("open");
    };
  });
  gridEl.querySelectorAll(".card").forEach(card => {
    const id = card.dataset.id;
    card.onclick = event => {
      setSelectedItem(cardIndex(card), true);
    };
    card.onfocus = () => setSelectedItem(cardIndex(card), true);
  });
}

async function toggleSubtitle(id) {
  const item = itemById(id);
  const nextHasSubtitle = !item.review_has_subtitle;
  const segment = segmentInputEl.value || item.review_segment_id;
  await saveItem(id, nextHasSubtitle, segment, noteInputEl.value);
}

async function saveCard(id) {
  const item = itemById(id);
  await saveItem(
    id,
    item.review_has_subtitle,
    segmentInputEl.value,
    noteInputEl.value,
  );
}

async function splitCard(id) {
  await splitSegmentAt(id);
}

async function saveItem(id, hasSubtitle, segmentId, note) {
  statusEl.textContent = "保存中...";
  const response = await fetch("/api/item", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id, has_subtitle: hasSubtitle, segment_id: segmentId, note}),
  });
  if (!response.ok) {
    statusEl.textContent = "保存失败";
    return;
  }
  statusEl.textContent = "已保存";
  await loadState(segmentId, id);
}

async function renameGroup() {
  const group = currentGroup();
  const nextSegment = groupSegmentEl.value.trim();
  if (!group || group.no_subtitle || !nextSegment) return;
  statusEl.textContent = "保存当前组...";
  const response = await fetch("/api/group-segment", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({old_segment_id: group.segment_id, new_segment_id: nextSegment}),
  });
  if (!response.ok) {
    statusEl.textContent = "保存失败";
    return;
  }
  statusEl.textContent = "当前组已修改";
  await loadState(nextSegment);
}

async function splitSegmentAt(id) {
  statusEl.textContent = "正在从当前位置打断...";
  const response = await fetch("/api/split-segment-at", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id}),
  });
  if (!response.ok) {
    statusEl.textContent = await response.text();
    return;
  }
  const payload = await response.json();
  statusEl.textContent = "已从当前位置打断";
  await loadState(payload.segment_id, id);
}

async function mergeCurrentSegmentToPrevious(id) {
  statusEl.textContent = "正在合并当前段...";
  const response = await fetch("/api/merge-segment-to-previous", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({id}),
  });
  if (!response.ok) {
    statusEl.textContent = await response.text();
    return;
  }
  const payload = await response.json();
  statusEl.textContent = "当前段已并入上一段";
  await loadState(payload.segment_id, id);
}

function itemById(id) {
  return state.items.find(item => String(item.id) === String(id));
}

function currentItem() {
  return visibleItems()?.[selectedItemIndex] || null;
}

function currentCard() {
  const item = currentItem();
  if (!item) return null;
  return gridEl.querySelector(`.card[data-id="${cssEscape(String(item.id))}"]`);
}

function cardIndex(card) {
  return [...gridEl.querySelectorAll(".card")].indexOf(card);
}

function setSelectedItem(index, scroll = true) {
  const items = visibleItems();
  if (!items.length) return;
  selectedItemIndex = Math.max(0, Math.min(items.length - 1, index));
  updateSelectedCard(scroll);
}

function selectItem(delta) {
  setSelectedItem(selectedItemIndex + delta);
}

function updateSelectedCard(scroll) {
  const selected = currentItem();
  gridEl.querySelectorAll(".card").forEach((card, index) => {
    card.classList.toggle("active", index === selectedItemIndex);
  });
  const card = currentCard();
  if (card && scroll) card.scrollIntoView({block: "nearest", inline: "nearest"});
  const item = selected;
  if (item) {
    const image = `/images/${encodeURIComponent(imageName(item.image))}`;
    stageImageEl.src = image;
    stageImageEl.dataset.full = image;
    sampleIdEl.textContent = `${selectedItemIndex + 1}/${visibleItems().length}  ${item.id}`;
    toggleEditorEl.textContent = item.review_has_subtitle ? "有字幕" : "无字幕";
    toggleEditorEl.classList.toggle("active", Boolean(item.review_has_subtitle));
    segmentInputEl.value = item.review_segment_id || "";
    segmentInputEl.disabled = !item.review_has_subtitle;
    noteInputEl.value = item.review_note || "";
    ocrTextEl.textContent = item.ocr_text ? `OCR: ${item.ocr_text}` : "OCR: ";
    statusEl.textContent = `当前 ${selectedItemIndex + 1}/${visibleItems().length}: ${item.review_has_subtitle ? "有字幕" : "无字幕"} | ${item.review_segment_id}`;
    syncGroupToItem(item, scroll);
  }
}

function syncGroupToItem(item, scroll) {
  updateActiveSegment(scroll);
}

function updateActiveSegment(scroll) {
  const buttons = [...segmentsEl.querySelectorAll(".segment")];
  buttons.forEach((button, index) => {
    button.classList.toggle("active", index === groupIndex);
  });
  const active = buttons[groupIndex];
  if (active && scroll) active.scrollIntoView({block: "nearest", inline: "nearest"});
}

function toggleCurrent() {
  const item = currentItem();
  if (item) toggleSubtitle(item.id);
}

function saveCurrent() {
  const item = currentItem();
  if (item) saveCard(item.id);
}

function splitCurrent() {
  const item = currentItem();
  if (item) splitCard(item.id);
}

function mergeCurrentGroupToPrevious() {
  const item = currentItem();
  if (item) mergeCurrentSegmentToPrevious(item.id);
}

function zoomCurrent() {
  const img = stageImageEl;
  if (!img?.dataset.full) return;
  viewerImg.src = img.dataset.full;
  viewerEl.classList.add("open");
}

function focusCurrentInput(role) {
  const input = role === "segment" ? segmentInputEl : noteInputEl;
  if (input && !input.disabled) {
    input.focus();
    input.select();
  }
}

function frameIndex(item) {
  return item.source_annotation?.frame_index ?? item.index;
}

function imageName(path) {
  return String(path).split("/").pop();
}

function groupLabel(group) {
  return group.no_subtitle ? `无字幕 #${group.empty_run || ""}`.trim() : group.segment_id;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function cssEscape(value) {
  if (window.CSS?.escape) return CSS.escape(value);
  return value.replace(/["\\]/g, "\\$&");
}

function isTextEntryFocus() {
  const element = document.activeElement;
  if (!element) return false;
  if (element.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
}

function suppressFocusedButtonActivation(event) {
  const element = document.activeElement;
  if (!element || element.tagName !== "BUTTON") return false;
  if (event.key !== "Enter" && event.key !== " ") return false;
  event.preventDefault();
  event.stopPropagation();
  element.blur();
  return true;
}

function moveGroup(delta) {
  const groups = visibleGroups();
  if (!groups.length) return;
  groupIndex = Math.max(0, Math.min(groups.length - 1, groupIndex + delta));
  selectedItemIndex = 0;
  render();
}

function selectGroupByAllIndex(allIndex) {
  const target = state?.groups?.[allIndex];
  if (!target) return;
  const groups = visibleGroups();
  const index = groups.indexOf(target);
  if (index < 0) return;
  groupIndex = index;
  selectedItemIndex = 0;
  render();
}

const navigationShortcuts = {
  ArrowRight: () => selectItem(1),
  ArrowLeft: () => selectItem(-1),
  ArrowDown: () => moveGroup(1),
  ArrowUp: () => moveGroup(-1),
  j: () => moveGroup(1),
  J: () => moveGroup(1),
  k: () => moveGroup(-1),
  K: () => moveGroup(-1),
};
const navigationHold = createNavigationHoldController(navigationShortcuts);

function createNavigationHoldController(shortcuts) {
  const firstRepeatDelayMs = 170;
  const repeatDelayMs = 75;
  let activeHold = null;

  function stopActiveHold() {
    if (!activeHold) return;
    if (activeHold.timerId) window.clearTimeout(activeHold.timerId);
    activeHold = null;
  }

  function scheduleNext(delayMs) {
    if (!activeHold) return;
    const token = activeHold.token;
    activeHold.timerId = window.setTimeout(runStep, delayMs);

    function runStep() {
      if (!activeHold || activeHold.token !== token) return;
      activeHold.action();
      scheduleNext(repeatDelayMs);
    }
  }

  function press(event) {
    const action = shortcuts[event.key];
    if (!action) return false;
    event.preventDefault();
    if (event.repeat) return true;
    stopActiveHold();
    activeHold = {key: event.key, action, token: Symbol(event.key), timerId: 0};
    action();
    scheduleNext(firstRepeatDelayMs);
    return true;
  }

  function releaseKey(event) {
    if (!activeHold) return;
    if (event?.key && event.key !== activeHold.key) return;
    event.preventDefault();
    stopActiveHold();
  }

  function releaseAll() {
    stopActiveHold();
  }

  return {press, releaseKey, releaseAll};
}

document.getElementById("subtitleOnly").onclick = () => { showOnlySubtitle = !showOnlySubtitle; selectedItemIndex = 0; render(); };
document.getElementById("openCandidateReview").onclick = () => { window.location.href = "/ocr-review"; };
document.getElementById("renameGroup").onclick = renameGroup;
document.getElementById("prevGroup").onclick = () => moveGroup(-1);
document.getElementById("nextGroup").onclick = () => moveGroup(1);
document.getElementById("selectPrev").onclick = () => selectItem(-1);
document.getElementById("selectNext").onclick = () => selectItem(1);
document.getElementById("toggleCurrent").onclick = toggleCurrent;
toggleEditorEl.onclick = toggleCurrent;
document.getElementById("splitCurrent").onclick = splitCurrent;
document.getElementById("mergeCurrentGroup").onclick = mergeCurrentGroupToPrevious;
document.getElementById("saveCurrent").onclick = saveCurrent;
document.getElementById("zoomCurrent").onclick = zoomCurrent;
stageImageEl.onclick = zoomCurrent;
document.getElementById("editSegment").onclick = () => focusCurrentInput("segment");
document.getElementById("editNote").onclick = () => focusCurrentInput("note");
viewerEl.onclick = () => viewerEl.classList.remove("open");
document.addEventListener("keydown", event => {
  if (viewerEl.classList.contains("open")) {
    if (event.key === "Escape" || event.key.toLowerCase() === "v") viewerEl.classList.remove("open");
    return;
  }
  if (isTextEntryFocus()) return;
  if (suppressFocusedButtonActivation(event)) return;
  if (navigationHold.press(event)) return;
  if (event.key === " ") { event.preventDefault(); toggleCurrent(); }
  if (event.key === "Enter") saveCurrent();
  if (event.key.toLowerCase() === "x") splitCurrent();
  if (event.key.toLowerCase() === "m") mergeCurrentGroupToPrevious();
  if (event.key.toLowerCase() === "v") zoomCurrent();
  if (event.key.toLowerCase() === "e") focusCurrentInput("segment");
  if (event.key.toLowerCase() === "n") focusCurrentInput("note");
  if (event.key.toLowerCase() === "g") groupSegmentEl.focus();
  if (event.key.toLowerCase() === "r") renameGroup();
  if (event.key === "Escape") viewerEl.classList.remove("open");
});
document.addEventListener("keyup", navigationHold.releaseKey);
window.addEventListener("blur", navigationHold.releaseAll);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) navigationHold.releaseAll();
});
loadState();
</script>
</body>
</html>
"""


OCR_REVIEW_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCR 合并审查</title>
  <style>
    :root {
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: #202124;
      background: #f6f7f9;
      --border: #d7dce2;
      --muted: #5f6368;
      --accent: #0b57d0;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main { height: 100vh; display: grid; grid-template-rows: auto auto minmax(0, 1fr); }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      padding: 12px 14px;
      background: #fff;
      border-bottom: 1px solid var(--border);
    }
    h1 { margin: 0 0 5px; font-size: 18px; }
    .meta { color: var(--muted); font-size: 12px; line-height: 1.45; word-break: break-all; }
    button {
      min-height: 34px;
      border: 1px solid #b7bec7;
      background: #fff;
      cursor: pointer;
      padding: 0 12px;
      font-size: 13px;
    }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    button:disabled {
      background: #f1f3f4;
      border-color: #d7dce2;
      color: #9aa0a6;
      cursor: default;
    }
    input {
      height: 34px;
      border: 1px solid #b7bec7;
      padding: 4px 8px;
      font-size: 13px;
      width: 82px;
    }
    .actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-start;
    }
    .threshold {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    .status {
      padding: 8px 14px;
      color: var(--muted);
      font-size: 13px;
      background: #fff;
      border-bottom: 1px solid var(--border);
      min-height: 34px;
    }
    .candidate-panel {
      padding: 14px;
      overflow: auto;
      background: #f6f7f9;
    }
    .candidate {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 12px;
      height: 100%;
      font-size: 13px;
    }
    .candidate-summary {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      background: #fff;
      border: 1px solid var(--border);
      padding: 10px;
    }
    .metric {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      color: var(--muted);
      font-size: 12px;
    }
    .approval-status {
      color: #5f6368;
      font-weight: 700;
    }
    .approval-status.confirmed { color: #137333; }
    .candidate-actions button.confirm-active {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .candidate-actions button.cancel-active {
      background: #d93025;
      border-color: #d93025;
      color: #fff;
    }
    .candidate-images {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      min-height: 0;
    }
    .candidate-image {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 8px;
      min-width: 0;
      background: #fff;
      border: 1px solid var(--border);
      padding: 8px;
    }
    .candidate-image img {
      width: 100%;
      height: 100%;
      min-height: 260px;
      object-fit: contain;
      background: #101214;
      cursor: zoom-in;
    }
    .ocr-text {
      min-height: 72px;
      max-height: 140px;
      overflow: auto;
      padding: 8px;
      border: 1px solid var(--border);
      background: #fff;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .candidate-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
      background: #fff;
      border: 1px solid var(--border);
      padding: 10px;
    }
    .viewer {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, 0.86);
      z-index: 20;
      padding: 24px;
    }
    .viewer.open { display: flex; }
    .viewer img { max-width: 96vw; max-height: 88vh; object-fit: contain; background: #000; }
  </style>
</head>
<body>
<main>
  <div class="topbar">
    <div>
      <h1>OCR 合并审查</h1>
      <div class="meta" id="detail">加载中...</div>
    </div>
    <div class="actions">
      <button id="candidatePrev">← 上一个候选</button>
      <button id="candidateNext">→ 下一个候选</button>
      <button id="commitPending" class="primary">写入合并</button>
      <button id="clearPending" class="danger">清空草稿</button>
      <label class="threshold">相似度下限 <input id="similarityRatio" type="number" min="0" max="1" step="0.01" value="0.86"></label>
      <label class="threshold">编辑距容差 <input id="editDistanceTolerance" type="number" min="0" max="30" step="1" value="3"></label>
      <button id="refreshCandidates">刷新候选</button>
      <button id="back">返回分组检查</button>
    </div>
  </div>
  <div class="status" id="status">只按 OCR 归一化文本、相似度和编辑距离判断候选；不做同义词或语义合并。</div>
  <div class="candidate-panel" id="candidatePanel"></div>
</main>
<div class="viewer" id="viewer"><img id="viewerImg" alt=""></div>
<script>
let state = null;
let candidateIndex = 0;
let pendingMerges = [];
let allowNavigate = false;
const detailEl = document.getElementById("detail");
const statusEl = document.getElementById("status");
const candidatePanelEl = document.getElementById("candidatePanel");
const viewerEl = document.getElementById("viewer");
const viewerImg = document.getElementById("viewerImg");
const similarityRatioEl = document.getElementById("similarityRatio");
const editDistanceToleranceEl = document.getElementById("editDistanceTolerance");
const commitPendingEl = document.getElementById("commitPending");
const clearPendingEl = document.getElementById("clearPending");

async function init() {
  statusEl.textContent = "正在合并 OCR 完全相同的连续字幕...";
  const response = await fetch("/api/merge-exact-ocr", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({}),
  });
  if (!response.ok) {
    statusEl.textContent = await response.text();
    return;
  }
  const payload = await response.json();
  statusEl.textContent = `已直接写入 ${payload.changed} 个完全相同 OCR 合并；近似候选会先进入浏览器草稿。`;
  await loadState();
}

async function loadState() {
  const params = candidateThresholdParams();
  state = await fetch(`/api/state?${params}`).then(r => r.json());
  updateDetail();
  renderCandidate();
}

function updateDetail() {
  const thresholds = state.candidate_thresholds;
  detailEl.textContent = `${state.stats.samples} 张 | 字幕段 ${state.stats.segments} | OCR 候选 ${candidateList().length} | 已确认待写入 ${pendingMerges.length} | 相似度≥${thresholds.similarity_ratio} | 编辑距≤${thresholds.edit_distance_tolerance} | ${state.review_file}`;
  commitPendingEl.disabled = pendingMerges.length === 0;
  clearPendingEl.disabled = pendingMerges.length === 0;
}

function candidateThresholdParams() {
  const similarityRatio = clampNumber(Number(similarityRatioEl.value), 0, 1, 0.86);
  const editDistanceTolerance = Math.round(clampNumber(Number(editDistanceToleranceEl.value), 0, 30, 3));
  similarityRatioEl.value = String(similarityRatio);
  editDistanceToleranceEl.value = String(editDistanceTolerance);
  return new URLSearchParams({
    similarity_ratio: String(similarityRatio),
    edit_distance_tolerance: String(editDistanceTolerance),
  });
}

function clampNumber(value, min, max, fallback) {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, value));
}

function renderCandidate() {
  const candidates = candidateList();
  if (!candidates.length) {
    candidatePanelEl.innerHTML = `<div class="meta">没有剩余 OCR 文本近似候选</div>`;
    setCandidateButtons(null);
    updateDetail();
    return;
  }
  candidateIndex = Math.max(0, Math.min(candidates.length - 1, candidateIndex));
  const candidate = candidates[candidateIndex];
  candidatePanelEl.innerHTML = renderCandidateCard(candidate, candidateIndex + 1, candidates.length);
  setCandidateButtons(candidate);
  updateDetail();
  candidatePanelEl.querySelectorAll(".candidate-image img").forEach(img => {
    img.onclick = () => {
      viewerImg.src = img.dataset.full;
      viewerEl.classList.add("open");
    };
  });
}

function candidateList() {
  return state?.similar_candidates || [];
}

function candidateKey(candidate) {
  return `${candidate.left_id}\u0000${candidate.right_id}`;
}

function pendingCandidateIndex(candidate) {
  return pendingMerges.findIndex(merge => candidateKey(merge) === candidateKey(candidate));
}

function isCandidateConfirmed(candidate) {
  return pendingCandidateIndex(candidate) >= 0;
}

function pendingSourceIndex(candidate) {
  return pendingMerges.findIndex(merge => String(merge.right_segment_id) === String(candidate.right_segment_id));
}

function candidateApprovalState(candidate) {
  const exactIndex = pendingCandidateIndex(candidate);
  if (exactIndex >= 0) return {status: "confirmed", label: "已确认待写入"};
  const sourceIndex = pendingSourceIndex(candidate);
  if (sourceIndex >= 0) return {status: "blocked", label: "同段已在其它候选确认"};
  return {status: "pending", label: "待确认"};
}

function renderCandidateCard(candidate, ordinal, total) {
  const left = itemById(candidate.left_id);
  const right = itemById(candidate.right_id);
  const leftImage = `/images/${encodeURIComponent(imageName(left?.image || ""))}`;
  const rightImage = `/images/${encodeURIComponent(imageName(right?.image || ""))}`;
  const approval = candidateApprovalState(candidate);
  const confirmed = approval.status === "confirmed";
  const blocked = approval.status === "blocked";
  return `
    <div class="candidate">
      <div class="candidate-summary">
        <div>
          <strong>候选 ${ordinal}/${total}</strong>
          <div class="meta">${escapeHtml(candidate.left_id)} ↔ ${escapeHtml(candidate.right_id)}</div>
        </div>
        <div class="metric">
          <span>${escapeHtml(candidate.match_type)}</span>
          <span>相似度 ${escapeHtml(candidate.similarity)}</span>
          <span>编辑距 ${escapeHtml(candidate.edit_distance)}</span>
          <span class="approval-status ${confirmed ? "confirmed" : ""}">状态：${escapeHtml(approval.label)}</span>
        </div>
      </div>
      <div class="candidate-images">
        <div class="candidate-image">
          <img src="${leftImage}" data-full="${leftImage}" alt="">
          <div>
            <div class="meta">${escapeHtml(candidate.left_id)} | ${escapeHtml(candidate.left_segment_id)}</div>
            <div class="ocr-text">${escapeHtml(candidate.left_text)}</div>
          </div>
        </div>
        <div class="candidate-image">
          <img src="${rightImage}" data-full="${rightImage}" alt="">
          <div>
            <div class="meta">${escapeHtml(candidate.right_id)} | ${escapeHtml(candidate.right_segment_id)}</div>
            <div class="ocr-text">${escapeHtml(candidate.right_text)}</div>
          </div>
        </div>
      </div>
      <div class="candidate-actions">
        <span class="meta">完全相同 OCR 已直接写入；←/→ 切换候选，Enter 确认并下一条，Backspace/Delete 取消当前确认。</span>
        <button data-role="candidate-prev">← 上一个</button>
        <button data-role="candidate-next">→ 下一个</button>
        <button class="${!confirmed && !blocked ? "confirm-active" : ""}" data-role="confirm-candidate" data-left-id="${escapeAttr(candidate.left_id)}" data-right-id="${escapeAttr(candidate.right_id)}" ${confirmed || blocked ? "disabled" : ""}>Enter 确认当前</button>
        <button class="${confirmed ? "cancel-active" : ""}" data-role="cancel-candidate" data-left-id="${escapeAttr(candidate.left_id)}" data-right-id="${escapeAttr(candidate.right_id)}" ${confirmed ? "" : "disabled"}>Backspace 取消当前确认</button>
      </div>
    </div>
  `;
}

function setCandidateButtons(candidate) {
  document.getElementById("candidatePrev").onclick = () => moveCandidate(-1);
  document.getElementById("candidateNext").onclick = () => moveCandidate(1);
  const inlineConfirm = candidatePanelEl.querySelector('[data-role="confirm-candidate"]');
  const inlineCancel = candidatePanelEl.querySelector('[data-role="cancel-candidate"]');
  if (inlineConfirm) {
    inlineConfirm.onclick = event => {
      const selected = candidateFromButton(event.currentTarget);
      if (selected && candidateApprovalState(selected).status === "pending") queueSimilarCandidate(selected, true);
    };
  }
  if (inlineCancel) {
    inlineCancel.onclick = event => {
      const selected = candidateFromButton(event.currentTarget);
      if (selected && candidateApprovalState(selected).status === "confirmed") cancelCandidateConfirmation(selected);
    };
  }
  const inlinePrev = candidatePanelEl.querySelector('[data-role="candidate-prev"]');
  const inlineNext = candidatePanelEl.querySelector('[data-role="candidate-next"]');
  if (inlinePrev) inlinePrev.onclick = () => moveCandidate(-1);
  if (inlineNext) inlineNext.onclick = () => moveCandidate(1);
}

function candidateFromButton(button) {
  return (state?.similar_candidates || []).find(candidate => {
    return String(candidate.left_id) === String(button.dataset.leftId) && String(candidate.right_id) === String(button.dataset.rightId);
  });
}

function queueSimilarCandidate(candidate, moveNext = false) {
  if (isCandidateConfirmed(candidate)) {
    if (moveNext) moveCandidate(1);
    return;
  }
  if (pendingSourceIndex(candidate) >= 0) {
    statusEl.textContent = "这个右侧段已经在其它候选里确认，先取消那条确认再改。";
    renderCandidate();
    return;
  }
  pendingMerges.push({
    left_id: candidate.left_id,
    right_id: candidate.right_id,
    left_segment_id: candidate.left_segment_id,
    right_segment_id: candidate.right_segment_id,
  });
  statusEl.textContent = `已加入草稿 ${pendingMerges.length} 条，尚未写入。`;
  if (moveNext) {
    const candidates = candidateList();
    candidateIndex = Math.min(candidates.length - 1, candidateIndex + 1);
  }
  renderCandidate();
}

function cancelCandidateConfirmation(candidate) {
  const index = pendingCandidateIndex(candidate);
  if (index < 0) return;
  pendingMerges.splice(index, 1);
  statusEl.textContent = `已取消确认：${candidate.left_id} ↔ ${candidate.right_id}`;
  renderCandidate();
}

async function commitPendingMerges() {
  if (!pendingMerges.length) return;
  statusEl.textContent = `正在写入 ${pendingMerges.length} 条合并...`;
  const response = await fetch("/api/merge-similar-candidates", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({pairs: pendingMerges}),
  });
  if (!response.ok) {
    statusEl.textContent = await response.text();
    return;
  }
  const payload = await response.json();
  pendingMerges = [];
  statusEl.textContent = `已写入 ${payload.changed} 条合并。`;
  // Keeping the same index selects the next refreshed candidate after committed pairs disappear.
  await loadState();
}

function clearPendingMerges() {
  if (!pendingMerges.length) return;
  pendingMerges = [];
  statusEl.textContent = "已清空浏览器草稿，未写入文件。";
  renderCandidate();
}

function moveCandidate(delta) {
  const candidates = candidateList();
  if (!candidates.length) return;
  candidateIndex = Math.max(0, Math.min(candidates.length - 1, candidateIndex + delta));
  renderCandidate();
}

function itemById(id) {
  return state.items.find(item => String(item.id) === String(id));
}

function imageName(path) {
  return String(path).split("/").pop();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function isTextEntryFocus() {
  const element = document.activeElement;
  if (!element) return false;
  if (element.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName);
}

function suppressFocusedButtonActivation(event) {
  const element = document.activeElement;
  if (!element || element.tagName !== "BUTTON") return false;
  if (event.key === "Enter") {
    element.blur();
    return false;
  }
  if (event.key !== " ") return false;
  event.preventDefault();
  event.stopPropagation();
  element.blur();
  return true;
}

const navigationShortcuts = {
  ArrowRight: () => moveCandidate(1),
  ArrowLeft: () => moveCandidate(-1),
};
const navigationHold = createNavigationHoldController(navigationShortcuts);

function createNavigationHoldController(shortcuts) {
  const firstRepeatDelayMs = 170;
  const repeatDelayMs = 75;
  let activeHold = null;

  function stopActiveHold() {
    if (!activeHold) return;
    if (activeHold.timerId) window.clearTimeout(activeHold.timerId);
    activeHold = null;
  }

  function scheduleNext(delayMs) {
    if (!activeHold) return;
    const token = activeHold.token;
    activeHold.timerId = window.setTimeout(runStep, delayMs);

    function runStep() {
      if (!activeHold || activeHold.token !== token) return;
      activeHold.action();
      scheduleNext(repeatDelayMs);
    }
  }

  function press(event) {
    const action = shortcuts[event.key];
    if (!action) return false;
    event.preventDefault();
    if (event.repeat) return true;
    stopActiveHold();
    activeHold = {key: event.key, action, token: Symbol(event.key), timerId: 0};
    action();
    scheduleNext(firstRepeatDelayMs);
    return true;
  }

  function releaseKey(event) {
    if (!activeHold) return;
    if (event?.key && event.key !== activeHold.key) return;
    event.preventDefault();
    stopActiveHold();
  }

  function releaseAll() {
    stopActiveHold();
  }

  return {press, releaseKey, releaseAll};
}

document.getElementById("back").onclick = () => {
  if (pendingMerges.length && !window.confirm("有未写入的合并草稿，返回会丢弃它们。继续返回？")) return;
  allowNavigate = true;
  window.location.href = "/";
};
document.getElementById("refreshCandidates").onclick = () => {
  candidateIndex = 0;
  loadState();
};
commitPendingEl.onclick = commitPendingMerges;
clearPendingEl.onclick = clearPendingMerges;
viewerEl.onclick = () => viewerEl.classList.remove("open");
window.addEventListener("beforeunload", event => {
  if (allowNavigate) return;
  if (!pendingMerges.length) return;
  event.preventDefault();
  event.returnValue = "";
});
document.addEventListener("keydown", event => {
  if (viewerEl.classList.contains("open")) {
    if (event.key === "Escape" || event.key.toLowerCase() === "v") viewerEl.classList.remove("open");
    return;
  }
  if (isTextEntryFocus()) return;
  if (suppressFocusedButtonActivation(event)) return;
  if (navigationHold.press(event)) return;
  if (event.key === "Enter") {
    const candidate = candidateList()[candidateIndex];
    if (candidate) queueSimilarCandidate(candidate, true);
  }
  if (event.key === "Backspace" || event.key === "Delete") {
    event.preventDefault();
    const candidate = candidateList()[candidateIndex];
    if (candidate) cancelCandidateConfirmation(candidate);
  }
  if (event.key === "Escape") document.getElementById("back").click();
});
document.addEventListener("keyup", navigationHold.releaseKey);
window.addEventListener("blur", navigationHold.releaseAll);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) navigationHold.releaseAll();
});
init();
</script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually label ROI subtitle presence and segment identity.")
    parser.add_argument("--samples-dir", type=Path, required=True, help="ROI dataset directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser automatically.")
    parser.add_argument("--verbose-requests", action="store_true", help="Print HTTP request logs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        app = SegmentReviewApp(args.samples_dir, verbose_requests=bool(args.verbose_requests))
        server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/"
    print(f"serving {app.samples_dir}")
    print(f"segment review: {app.review_path}")
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
