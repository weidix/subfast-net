from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

import torch


@dataclass(frozen=True)
class EmbeddingPair:
    i: int
    j: int
    same: bool
    source: str


@dataclass(frozen=True)
class EmbeddingPairSelection:
    pairs: list[EmbeddingPair]
    local_positive_pairs: int
    local_negative_pairs: int
    ocr_negative_pairs: int
    skipped_pairs: int

    @property
    def embedding_pairs(self) -> int:
        return len(self.pairs)

    @property
    def negative_pairs(self) -> int:
        return self.local_negative_pairs + self.ocr_negative_pairs


_FRAME_PATTERN = re.compile(r"^(?P<video>.+)_f(?P<frame>\d+)$")
_MIN_OCR_TEXT_LENGTH = 4


def parse_video_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_frame_index(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_video_frame_from_sample_id(sample_id: str) -> tuple[str | None, int | None]:
    match = _FRAME_PATTERN.match(sample_id)
    if match is None:
        return None, None
    return match.group("video"), int(match.group("frame"))


def normalize_ocr_text(text: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    chars: list[str] = []
    for char in normalized:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            continue
        chars.append(char)
    return "".join(chars)


def ocr_text_similarity(left: str | None, right: str | None) -> float | None:
    left_norm = normalize_ocr_text(left)
    right_norm = normalize_ocr_text(right)
    if len(left_norm) < _MIN_OCR_TEXT_LENGTH or len(right_norm) < _MIN_OCR_TEXT_LENGTH:
        return None
    if left_norm == right_norm:
        return 1.0
    sequence_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_chars = set(left_norm)
    right_chars = set(right_norm)
    overlap_ratio = len(left_chars & right_chars) / max(1, len(left_chars | right_chars))
    return max(sequence_ratio, overlap_ratio)


def select_embedding_pairs(
    *,
    presence: torch.Tensor,
    segment_ids: list[str],
    roots: list[str],
    video_ids: list[str | None],
    frame_indices: list[int | None],
    ocr_texts: list[str],
    frame_window: int,
    ocr_negative_enabled: bool,
    ocr_negative_max_similarity: float,
) -> EmbeddingPairSelection:
    if frame_window < 0:
        raise ValueError("embedding pair frame window must be non-negative")
    if not 0.0 <= ocr_negative_max_similarity <= 1.0:
        raise ValueError("OCR negative max similarity must be in [0, 1]")
    pairs: list[EmbeddingPair] = []
    seen: set[tuple[int, int]] = set()
    local_positive_pairs = 0
    local_negative_pairs = 0
    ocr_negative_pairs = 0
    skipped_pairs = 0
    presence_mask = (presence.detach().cpu() > 0.5).tolist()

    for i in range(len(segment_ids)):
        for j in range(i + 1, len(segment_ids)):
            if not presence_mask[i] or not presence_mask[j]:
                skipped_pairs += 1
                continue

            same_root = roots[i] == roots[j]
            same_video = video_ids[i] is not None and video_ids[i] == video_ids[j]
            has_frames = frame_indices[i] is not None and frame_indices[j] is not None
            local_pair = (
                same_root
                and same_video
                and has_frames
                and abs(int(frame_indices[i]) - int(frame_indices[j])) <= frame_window
            )
            if local_pair:
                same = segment_ids[i] == segment_ids[j]
                pairs.append(EmbeddingPair(i=i, j=j, same=same, source="local"))
                seen.add((i, j))
                if same:
                    local_positive_pairs += 1
                else:
                    local_negative_pairs += 1
                continue

            if (
                ocr_negative_enabled
                and same_root
                and segment_ids[i] != segment_ids[j]
                and (i, j) not in seen
            ):
                similarity = ocr_text_similarity(ocr_texts[i], ocr_texts[j])
                if similarity is not None and similarity <= ocr_negative_max_similarity:
                    pairs.append(EmbeddingPair(i=i, j=j, same=False, source="ocr"))
                    seen.add((i, j))
                    ocr_negative_pairs += 1
                    continue

            skipped_pairs += 1

    return EmbeddingPairSelection(
        pairs=pairs,
        local_positive_pairs=local_positive_pairs,
        local_negative_pairs=local_negative_pairs,
        ocr_negative_pairs=ocr_negative_pairs,
        skipped_pairs=skipped_pairs,
    )
