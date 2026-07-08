from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

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
    return normalized_ocr_text_similarity(left_norm, right_norm)


@lru_cache(maxsize=65_536)
def normalized_ocr_text_similarity(left_norm: str, right_norm: str) -> float | None:
    if len(left_norm) < _MIN_OCR_TEXT_LENGTH or len(right_norm) < _MIN_OCR_TEXT_LENGTH:
        return None
    if left_norm == right_norm:
        return 1.0
    sequence_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_chars = set(left_norm)
    right_chars = set(right_norm)
    overlap_ratio = len(left_chars & right_chars) / max(1, len(left_chars | right_chars))
    return max(sequence_ratio, overlap_ratio)


@lru_cache(maxsize=65_536)
def normalized_ocr_text_similarity_at_most(left_norm: str, right_norm: str, maximum: float) -> bool:
    if len(left_norm) < _MIN_OCR_TEXT_LENGTH or len(right_norm) < _MIN_OCR_TEXT_LENGTH:
        return False
    if left_norm == right_norm:
        return 1.0 <= maximum
    left_chars, left_counts = _ocr_text_profile(left_norm)
    right_chars, right_counts = _ocr_text_profile(right_norm)
    overlap_ratio = len(left_chars & right_chars) / max(1, len(left_chars | right_chars))
    if overlap_ratio > maximum:
        return False
    matches = sum(min(count, right_counts.get(char, 0)) for char, count in left_counts.items())
    quick_ratio = 2.0 * matches / (len(left_norm) + len(right_norm))
    if quick_ratio <= maximum:
        return True
    return SequenceMatcher(None, left_norm, right_norm).ratio() <= maximum


@lru_cache(maxsize=65_536)
def _ocr_text_profile(text: str) -> tuple[frozenset[str], Counter[str]]:
    return frozenset(text), Counter(text)


def select_embedding_pairs(
    *,
    presence: torch.Tensor,
    segment_ids: list[str],
    roots: list[str],
    video_ids: list[str | None],
    ocr_texts: list[str],
    adjacent_segment_ids: list[set[str] | frozenset[str] | list[str] | tuple[str, ...]],
    ocr_negative_enabled: bool,
    ocr_negative_max_similarity: float,
    ocr_negative_ratio: float,
) -> EmbeddingPairSelection:
    if not 0.0 <= ocr_negative_max_similarity <= 1.0:
        raise ValueError("OCR negative max similarity must be in [0, 1]")
    if not 0.0 <= ocr_negative_ratio <= 1.0:
        raise ValueError("OCR negative ratio must be in [0, 1]")
    if len(adjacent_segment_ids) != len(segment_ids):
        raise ValueError("adjacent segment ids must match segment ids")
    pairs: list[EmbeddingPair] = []
    seen: set[tuple[int, int]] = set()
    local_positive_pairs = 0
    local_negative_pairs = 0
    ocr_negative_pairs = 0
    skipped_pairs = 0
    ocr_candidates: list[EmbeddingPair] = []
    total_candidate_pairs = len(segment_ids) * (len(segment_ids) - 1) // 2
    positive_indices = [index for index, value in enumerate((presence.detach().cpu() > 0.5).tolist()) if value]
    normalized_ocr_texts = (
        {index: normalize_ocr_text(ocr_texts[index]) for index in positive_indices}
        if ocr_negative_enabled
        else {}
    )

    for offset, i in enumerate(positive_indices):
        for j in positive_indices[offset + 1 :]:

            same_root = roots[i] == roots[j]
            same_video = video_ids[i] is not None and video_ids[i] == video_ids[j]
            same = same_root and segment_ids[i] == segment_ids[j]
            if same:
                pairs.append(EmbeddingPair(i=i, j=j, same=True, source="local"))
                seen.add((i, j))
                local_positive_pairs += 1
                continue

            adjacent = (
                same_root
                and same_video
                and segment_ids[j] in adjacent_segment_ids[i]
                and segment_ids[i] in adjacent_segment_ids[j]
            )
            if adjacent:
                pairs.append(EmbeddingPair(i=i, j=j, same=False, source="local"))
                seen.add((i, j))
                local_negative_pairs += 1
                continue

            if (
                ocr_negative_enabled
                and same_root
                and segment_ids[i] != segment_ids[j]
                and (i, j) not in seen
            ):
                if normalized_ocr_text_similarity_at_most(
                    normalized_ocr_texts[i],
                    normalized_ocr_texts[j],
                    ocr_negative_max_similarity,
                ):
                    ocr_candidates.append(EmbeddingPair(i=i, j=j, same=False, source="ocr"))
                    continue

            skipped_pairs += 1

    ocr_limit = max_ocr_negative_pairs(
        local_negative_pairs=local_negative_pairs,
        positive_pairs=max(local_positive_pairs, len(positive_indices)),
        ocr_negative_ratio=ocr_negative_ratio,
    )
    selected_ocr = ocr_candidates[:ocr_limit]
    pairs.extend(selected_ocr)
    ocr_negative_pairs = len(selected_ocr)
    skipped_pairs += len(ocr_candidates) - len(selected_ocr)

    skipped_pairs += total_candidate_pairs - (len(positive_indices) * (len(positive_indices) - 1) // 2)
    return EmbeddingPairSelection(
        pairs=pairs,
        local_positive_pairs=local_positive_pairs,
        local_negative_pairs=local_negative_pairs,
        ocr_negative_pairs=ocr_negative_pairs,
        skipped_pairs=skipped_pairs,
    )


def max_ocr_negative_pairs(
    *,
    local_negative_pairs: int,
    positive_pairs: int,
    ocr_negative_ratio: float,
) -> int:
    if not 0.0 <= ocr_negative_ratio <= 1.0:
        raise ValueError("OCR negative ratio must be in [0, 1]")
    if ocr_negative_ratio <= 0.0:
        return 0
    if ocr_negative_ratio >= 1.0:
        return 1 << 60
    if local_negative_pairs > 0:
        return max(1, round(local_negative_pairs * ocr_negative_ratio / (1.0 - ocr_negative_ratio)))
    return max(1, round(positive_pairs * ocr_negative_ratio)) if positive_pairs > 0 else 0


def infer_adjacent_segment_ids(
    *,
    presence: torch.Tensor,
    segment_ids: list[str],
    roots: list[str],
    video_ids: list[str | None],
    frame_indices: list[int | None],
) -> list[frozenset[str]]:
    adjacent: list[set[str]] = [set() for _ in segment_ids]
    index_by_key: dict[tuple[str, str | None, str], list[int]] = defaultdict(list)
    groups: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for index, is_positive in enumerate((presence.detach().cpu() > 0.5).tolist()):
        if not is_positive:
            continue
        video_id = video_ids[index]
        frame_index = frame_indices[index]
        if video_id is None or frame_index is None:
            continue
        segment_id = segment_ids[index]
        key = (roots[index], video_id, segment_id)
        index_by_key[key].append(index)
        groups[(roots[index], video_id)].append((int(frame_index), segment_id))

    for (root, video_id), entries in groups.items():
        previous_segment: str | None = None
        for _, segment_id in sorted(entries):
            if previous_segment is None:
                previous_segment = segment_id
                continue
            if segment_id == previous_segment:
                continue
            left_key = (root, video_id, previous_segment)
            right_key = (root, video_id, segment_id)
            for index in index_by_key[left_key]:
                adjacent[index].add(segment_id)
            for index in index_by_key[right_key]:
                adjacent[index].add(previous_segment)
            previous_segment = segment_id
    return [frozenset(values) for values in adjacent]
