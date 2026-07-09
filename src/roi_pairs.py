from __future__ import annotations

import re
import random
import unicodedata
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
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
    pair_id: str = ""


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


@dataclass(frozen=True)
class ScheduledEmbeddingBatch:
    sample_indices: tuple[int, ...]
    pairs: tuple[EmbeddingPair, ...]


@dataclass(frozen=True)
class EmbeddingPairEpochSchedule:
    batches: tuple[ScheduledEmbeddingBatch, ...]
    positive_pair_count: int
    negative_pair_count: int
    unique_positive_pair_count: int
    unique_negative_pair_count: int
    positive_pair_repeat_rate: float
    negative_pair_repeat_rate: float
    unique_positive_roi_count: int
    total_positive_roi_count: int

    @property
    def pair_count(self) -> int:
        return self.positive_pair_count + self.negative_pair_count


@dataclass(frozen=True)
class EmbeddingPairPools:
    positive_pairs: tuple[EmbeddingPair, ...]
    local_negative_pairs: tuple[EmbeddingPair, ...]
    ocr_negative_pairs: Sequence[EmbeddingPair]
    total_positive_roi_count: int

    @property
    def negative_pair_count(self) -> int:
        return len(self.local_negative_pairs) + len(self.ocr_negative_pairs)


@dataclass(frozen=True)
class _OcrGroupPair:
    left_indices: tuple[int, ...]
    right_indices: tuple[int, ...]
    excluded_offsets: tuple[int, ...]
    count: int


class LazyOcrNegativePairPool(Sequence[EmbeddingPair]):
    def __init__(self, group_pairs: Sequence[_OcrGroupPair]) -> None:
        self._group_pairs = tuple(pair for pair in group_pairs if pair.count > 0)
        total = 0
        cumulative: list[int] = []
        for pair in self._group_pairs:
            total += pair.count
            cumulative.append(total)
        self._cumulative = tuple(cumulative)
        self._count = total

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        for rank in range(self._count):
            yield self._pair_at_rank(rank)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._pair_at_rank(rank) for rank in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        return self._pair_at_rank(index)

    def sample(self, count: int, rng: random.Random) -> list[EmbeddingPair]:
        if count <= 0 or self._count <= 0:
            return []
        if count <= self._count:
            return [self._pair_at_rank(rank) for rank in rng.sample(range(self._count), count)]
        selected = [self._pair_at_rank(rank) for rank in rng.sample(range(self._count), self._count)]
        while len(selected) < count:
            refill_count = min(self._count, count - len(selected))
            selected.extend(self._pair_at_rank(rank) for rank in rng.sample(range(self._count), refill_count))
        return selected

    def _pair_at_rank(self, rank: int) -> EmbeddingPair:
        group_index = bisect_right(self._cumulative, rank)
        group_start = 0 if group_index == 0 else self._cumulative[group_index - 1]
        group_rank = rank - group_start
        group = self._group_pairs[group_index]
        raw_offset = group_rank
        for excluded_offset in group.excluded_offsets:
            if excluded_offset > raw_offset:
                break
            raw_offset += 1
        right_count = len(group.right_indices)
        left = group.left_indices[raw_offset // right_count]
        right = group.right_indices[raw_offset % right_count]
        i, j = _ordered_pair(left, right)
        return EmbeddingPair(i=i, j=j, same=False, source="ocr")


_FRAME_PATTERN = re.compile(r"^(?P<video>.+)_f(?P<frame>\d+)$")
_MIN_OCR_TEXT_LENGTH = 4
_SAME_TEXT_MIN_SIMILARITY = 0.9


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


def is_same_subtitle_text(left_norm: str, right_norm: str) -> bool:
    """True when two normalized OCR texts render the same subtitle content.

    Segment markers split on time, so adjacent segments can carry identical
    text. Such pairs are visually indistinguishable and must not be used as
    negative supervision.
    """
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    similarity = normalized_ocr_text_similarity(left_norm, right_norm)
    return similarity is not None and similarity >= _SAME_TEXT_MIN_SIMILARITY


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
    normalized_ocr_texts = {index: normalize_ocr_text(ocr_texts[index]) for index in positive_indices}

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
                if is_same_subtitle_text(normalized_ocr_texts[i], normalized_ocr_texts[j]):
                    skipped_pairs += 1
                    continue
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


def build_embedding_pair_epoch_schedule(
    samples: Sequence[object],
    *,
    batch_size: int,
    negative_ratio: float,
    ocr_negative_enabled: bool,
    ocr_negative_max_similarity: float,
    ocr_negative_ratio: float,
    seed: int,
    epoch: int,
    pair_pools: EmbeddingPairPools | None = None,
) -> EmbeddingPairEpochSchedule:
    if batch_size < 2:
        raise ValueError("embedding batch size must be at least 2 for pair scheduling")
    if not 0.0 <= negative_ratio <= 1.0:
        raise ValueError("negative_ratio must be in [0, 1]")
    if not 0.0 <= ocr_negative_max_similarity <= 1.0:
        raise ValueError("OCR negative max similarity must be in [0, 1]")
    if not 0.0 <= ocr_negative_ratio <= 1.0:
        raise ValueError("OCR negative ratio must be in [0, 1]")

    pools = pair_pools or build_embedding_pair_pools(
        samples,
        ocr_negative_enabled=ocr_negative_enabled,
        ocr_negative_max_similarity=ocr_negative_max_similarity,
    )
    positive_pool = pools.positive_pairs
    local_negative_pool = pools.local_negative_pairs
    ocr_negative_pool = pools.ocr_negative_pairs
    rng = random.Random(seed + epoch)
    positive_roi_with_pairs = {index for pair in positive_pool for index in (pair.i, pair.j)}
    positive_budget = min(len(positive_pool), len(positive_roi_with_pairs))
    selected_positive = _select_positive_pairs_for_coverage(
        positive_pool,
        target_count=positive_budget,
        rng=rng,
    )
    negative_budget = _negative_budget(
        len(selected_positive),
        negative_ratio,
        len(local_negative_pool) + len(ocr_negative_pool),
    )
    selected_local_negative = _take_pairs(local_negative_pool, min(negative_budget, len(local_negative_pool)), rng)
    ocr_limit = max_ocr_negative_pairs(
        local_negative_pairs=len(selected_local_negative),
        positive_pairs=len(selected_positive),
        ocr_negative_ratio=ocr_negative_ratio,
    )
    selected_ocr_negative = _take_pairs(
        ocr_negative_pool,
        min(max(0, negative_budget - len(selected_local_negative)), ocr_limit),
        rng,
    )
    selected_negative = selected_local_negative + selected_ocr_negative
    if len(selected_negative) < negative_budget and local_negative_pool:
        selected_negative.extend(_take_pairs(local_negative_pool, negative_budget - len(selected_negative), rng))

    selected_positive = [_with_pair_id(pair, samples) for pair in selected_positive]
    selected_negative = [_with_pair_id(pair, samples) for pair in selected_negative]
    selected_pairs = selected_positive + selected_negative
    rng.shuffle(selected_pairs)
    batches = _pack_pair_batches(selected_pairs, batch_size)
    positive_pair_ids = [pair.pair_id for pair in selected_positive]
    negative_pair_ids = [pair.pair_id for pair in selected_negative]
    positive_roi_ids = {
        _sample_roi_id(samples[index])
        for pair in selected_positive
        for index in (pair.i, pair.j)
    }
    return EmbeddingPairEpochSchedule(
        batches=tuple(batches),
        positive_pair_count=len(positive_pair_ids),
        negative_pair_count=len(negative_pair_ids),
        unique_positive_pair_count=len(set(positive_pair_ids)),
        unique_negative_pair_count=len(set(negative_pair_ids)),
        positive_pair_repeat_rate=_repeat_rate(positive_pair_ids),
        negative_pair_repeat_rate=_repeat_rate(negative_pair_ids),
        unique_positive_roi_count=len(positive_roi_ids),
        total_positive_roi_count=pools.total_positive_roi_count,
    )


def build_embedding_pair_pools(
    samples: Sequence[object],
    *,
    ocr_negative_enabled: bool,
    ocr_negative_max_similarity: float,
    skip_ocr_if_local_negative_ratio_satisfied: float | None = None,
) -> EmbeddingPairPools:
    if not 0.0 <= ocr_negative_max_similarity <= 1.0:
        raise ValueError("OCR negative max similarity must be in [0, 1]")

    positive_indices: list[int] = []
    indices_by_root_segment: dict[tuple[str, str], list[int]] = defaultdict(list)
    indices_by_root_video_segment: dict[tuple[str, object, str], list[int]] = defaultdict(list)
    timeline_by_root_video: dict[tuple[str, object], list[tuple[int, str]]] = defaultdict(list)
    normalized_text_by_index: dict[int, str] = {}
    ocr_groups_by_root: dict[str, dict[tuple[str, str], list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for index, sample in enumerate(samples):
        if not bool(getattr(sample, "has_subtitle")):
            continue
        root = _sample_root(sample)
        segment_id = str(getattr(sample, "segment_id"))
        video_id = getattr(sample, "video_id")
        frame_index = getattr(sample, "frame_index")
        positive_indices.append(index)
        indices_by_root_segment[(root, segment_id)].append(index)
        normalized_text_by_index[index] = normalize_ocr_text(str(getattr(sample, "ocr_text", "")))
        if video_id is not None and frame_index is not None:
            indices_by_root_video_segment[(root, video_id, segment_id)].append(index)
            timeline_by_root_video[(root, video_id)].append((int(frame_index), segment_id))
        if ocr_negative_enabled:
            ocr_groups_by_root[root][(segment_id, normalized_text_by_index[index])].append(index)

    positive_pairs: list[EmbeddingPair] = []
    local_negative_pairs: list[EmbeddingPair] = []
    seen: set[tuple[int, int, bool, str]] = set()
    local_negative_pairs_by_segment_pair: dict[tuple[str, str, str], set[tuple[int, int]]] = defaultdict(set)

    for _, indices in sorted(indices_by_root_segment.items()):
        for offset, i in enumerate(indices):
            for j in indices[offset + 1 :]:
                _append_pool_pair(positive_pairs, seen, samples, i, j, True, "local")

    for (root, video_id), entries in sorted(
        timeline_by_root_video.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1])),
    ):
        adjacent_segment_pairs: set[tuple[str, str]] = set()
        previous_segment: str | None = None
        for _, segment_id in sorted(entries):
            if previous_segment is None:
                previous_segment = segment_id
                continue
            if segment_id == previous_segment:
                continue
            adjacent_segment_pairs.add(tuple(sorted((previous_segment, segment_id))))
            previous_segment = segment_id
        for left_segment, right_segment in sorted(adjacent_segment_pairs):
            left_indices = indices_by_root_video_segment[(root, video_id, left_segment)]
            right_indices = indices_by_root_video_segment[(root, video_id, right_segment)]
            segment_pair_key = (root, *tuple(sorted((left_segment, right_segment))))
            for i in left_indices:
                for j in right_indices:
                    if is_same_subtitle_text(normalized_text_by_index[i], normalized_text_by_index[j]):
                        continue
                    pair_key = _ordered_pair(i, j)
                    local_negative_pairs_by_segment_pair[segment_pair_key].add(pair_key)
                    _append_pool_pair(local_negative_pairs, seen, samples, pair_key[0], pair_key[1], False, "local")

    if (
        ocr_negative_enabled
        and not _local_negative_pool_satisfies_ratio(
            positive_pairs,
            local_negative_pairs,
            skip_ocr_if_local_negative_ratio_satisfied,
        )
    ):
        ocr_group_pairs: list[_OcrGroupPair] = []
        for root, text_groups in sorted(ocr_groups_by_root.items()):
            grouped = sorted(text_groups.items())
            for offset, ((left_segment, left_text), left_indices) in enumerate(grouped):
                for (right_segment, right_text), right_indices in grouped[offset + 1 :]:
                    if left_segment == right_segment:
                        continue
                    if not normalized_ocr_text_similarity_at_most(left_text, right_text, ocr_negative_max_similarity):
                        continue
                    segment_pair_key = (root, *tuple(sorted((left_segment, right_segment))))
                    excluded_offsets = _ocr_group_excluded_offsets(
                        left_indices,
                        right_indices,
                        local_negative_pairs_by_segment_pair.get(segment_pair_key, set()),
                    )
                    pair_count = len(left_indices) * len(right_indices) - len(excluded_offsets)
                    if pair_count > 0:
                        ocr_group_pairs.append(
                            _OcrGroupPair(
                                left_indices=tuple(left_indices),
                                right_indices=tuple(right_indices),
                                excluded_offsets=excluded_offsets,
                                count=pair_count,
                            )
                        )
        ocr_negative_pool: Sequence[EmbeddingPair] = LazyOcrNegativePairPool(ocr_group_pairs)
    else:
        ocr_negative_pool = ()

    positive_pairs.sort(key=lambda pair: _pair_sort_key(samples, pair))
    local_negative_pairs.sort(key=lambda pair: _pair_sort_key(samples, pair))
    return EmbeddingPairPools(
        positive_pairs=tuple(positive_pairs),
        local_negative_pairs=tuple(local_negative_pairs),
        ocr_negative_pairs=ocr_negative_pool,
        total_positive_roi_count=len(positive_indices),
    )


def _ordered_pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def _local_negative_pool_satisfies_ratio(
    positive_pairs: Sequence[EmbeddingPair],
    local_negative_pairs: Sequence[EmbeddingPair],
    negative_ratio: float | None,
) -> bool:
    if negative_ratio is None:
        return False
    if not 0.0 <= negative_ratio <= 1.0:
        raise ValueError("negative_ratio must be in [0, 1]")
    if negative_ratio >= 1.0:
        return False
    positive_roi_with_pairs = {index for pair in positive_pairs for index in (pair.i, pair.j)}
    positive_budget = min(len(positive_pairs), len(positive_roi_with_pairs))
    return len(local_negative_pairs) >= _negative_budget(
        positive_budget,
        negative_ratio,
        len(local_negative_pairs),
    )


def _ocr_group_excluded_offsets(
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    excluded_pairs: set[tuple[int, int]],
) -> tuple[int, ...]:
    if not excluded_pairs:
        return ()
    left_position = {index: offset for offset, index in enumerate(left_indices)}
    right_position = {index: offset for offset, index in enumerate(right_indices)}
    right_count = len(right_indices)
    offsets: list[int] = []
    for i, j in excluded_pairs:
        if i in left_position and j in right_position:
            offsets.append(left_position[i] * right_count + right_position[j])
        elif j in left_position and i in right_position:
            offsets.append(left_position[j] * right_count + right_position[i])
    return tuple(sorted(set(offsets)))


def _append_pool_pair(
    pool: list[EmbeddingPair],
    seen: set[tuple[int, int, bool, str]],
    samples: Sequence[object],
    i: int,
    j: int,
    same: bool,
    source: str,
) -> None:
    i, j = _ordered_pair(i, j)
    key = (i, j, same, source)
    if key in seen:
        return
    seen.add(key)
    pool.append(EmbeddingPair(i=i, j=j, same=same, source=source))


def _with_pair_id(pair: EmbeddingPair, samples: Sequence[object]) -> EmbeddingPair:
    if pair.pair_id:
        return pair
    return EmbeddingPair(
        i=pair.i,
        j=pair.j,
        same=pair.same,
        source=pair.source,
        pair_id=_pair_id(samples, pair.i, pair.j, pair.same, pair.source),
    )


def _pair_sort_key(samples: Sequence[object], pair: EmbeddingPair) -> tuple[str, str, str, str, str]:
    left = samples[pair.i]
    right = samples[pair.j]
    return (
        _sample_root(left),
        str(getattr(left, "sample_id")),
        str(getattr(right, "sample_id")),
        "same" if pair.same else "diff",
        pair.source,
    )


def _select_positive_pairs_for_coverage(
    pool: Sequence[EmbeddingPair],
    *,
    target_count: int,
    rng: random.Random,
) -> list[EmbeddingPair]:
    if target_count <= 0 or not pool:
        return []
    shuffled = list(pool)
    rng.shuffle(shuffled)
    uncovered = {index for pair in shuffled for index in (pair.i, pair.j)}
    selected: list[EmbeddingPair] = []
    deferred: list[EmbeddingPair] = []
    selected_pairs: set[EmbeddingPair] = set()

    for pair in shuffled:
        if len(selected) >= target_count:
            break
        if pair.i in uncovered and pair.j in uncovered:
            selected.append(pair)
            selected_pairs.add(pair)
            uncovered.discard(pair.i)
            uncovered.discard(pair.j)
        else:
            deferred.append(pair)

    for pair in deferred:
        if len(selected) >= target_count or not uncovered:
            break
        if pair.i not in uncovered and pair.j not in uncovered:
            continue
        selected.append(pair)
        selected_pairs.add(pair)
        uncovered.discard(pair.i)
        uncovered.discard(pair.j)

    if len(selected) < target_count:
        for pair in shuffled:
            if len(selected) >= target_count:
                break
            if pair in selected_pairs:
                continue
            selected.append(pair)
            selected_pairs.add(pair)
    if len(selected) < target_count:
        selected.extend(_take_pairs(pool, target_count - len(selected), rng))
    return selected


def _take_pairs(pool: Sequence[EmbeddingPair], count: int, rng: random.Random) -> list[EmbeddingPair]:
    if count <= 0 or not pool:
        return []
    if isinstance(pool, LazyOcrNegativePairPool):
        return pool.sample(count, rng)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    if count <= len(shuffled):
        return shuffled[:count]
    selected = list(shuffled)
    while len(selected) < count:
        refill = list(pool)
        rng.shuffle(refill)
        selected.extend(refill[: count - len(selected)])
    return selected


def _negative_budget(positive_count: int, negative_ratio: float, negative_pool_count: int) -> int:
    if positive_count <= 0 or negative_ratio <= 0.0 or negative_pool_count <= 0:
        return 0
    if negative_ratio >= 1.0:
        return negative_pool_count
    return max(0, round(positive_count * negative_ratio / (1.0 - negative_ratio)))


def _pack_pair_batches(pairs: list[EmbeddingPair], batch_size: int) -> list[ScheduledEmbeddingBatch]:
    batches: list[ScheduledEmbeddingBatch] = []
    current_pairs: list[EmbeddingPair] = []
    current_indices: list[int] = []
    current_set: set[int] = set()
    for pair in pairs:
        pair_indices = {pair.i, pair.j}
        if current_pairs and len(current_set | pair_indices) > batch_size:
            batches.append(_make_scheduled_batch(current_indices, current_pairs))
            current_pairs = []
            current_indices = []
            current_set = set()
        for index in (pair.i, pair.j):
            if index not in current_set:
                current_indices.append(index)
                current_set.add(index)
        current_pairs.append(pair)
    if current_pairs:
        batches.append(_make_scheduled_batch(current_indices, current_pairs))
    return batches


def _make_scheduled_batch(sample_indices: list[int], pairs: list[EmbeddingPair]) -> ScheduledEmbeddingBatch:
    local_index = {sample_index: offset for offset, sample_index in enumerate(sample_indices)}
    local_pairs = tuple(
        EmbeddingPair(
            i=local_index[pair.i],
            j=local_index[pair.j],
            same=pair.same,
            source=pair.source,
            pair_id=pair.pair_id,
        )
        for pair in pairs
    )
    return ScheduledEmbeddingBatch(sample_indices=tuple(sample_indices), pairs=local_pairs)


def _repeat_rate(pair_ids: list[str]) -> float:
    return 0.0 if not pair_ids else 1.0 - len(set(pair_ids)) / len(pair_ids)


def _sample_root(sample: object) -> str:
    return str(getattr(sample, "root"))


def _sample_roi_id(sample: object) -> str:
    return f"{_sample_root(sample)}::{getattr(sample, 'sample_id')}"


def _pair_id(samples: Sequence[object], i: int, j: int, same: bool, source: str) -> str:
    left = samples[i]
    right = samples[j]
    return "|".join(
        (
            _sample_root(left),
            str(getattr(left, "sample_id")),
            str(getattr(right, "sample_id")),
            "same" if same else "diff",
            source,
        )
    )


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
