from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from subfast_roi_data.data import RoiSample

_EMBEDDING_SAMPLES_PER_SEGMENT = 2


class RoiBalancedBatchSampler(Sampler[list[int]]):
    """Build full, reproducible ROI batches without removing source samples."""

    def __init__(
        self,
        samples: Sequence[RoiSample],
        *,
        batch_size: int,
        negative_ratio: float,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= negative_ratio <= 1.0:
            raise ValueError("negative_ratio must be in [0, 1]")
        self.samples = samples
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.seed = seed
        self.epoch = 0
        self.positive_indices = [index for index, sample in enumerate(samples) if sample.has_subtitle]
        self.negative_indices = [index for index, sample in enumerate(samples) if not sample.has_subtitle]

        if self.positive_indices and self.negative_indices:
            self.negative_slots = round(batch_size * negative_ratio)
            self.positive_slots = batch_size - self.negative_slots
            if self.positive_slots < 2 or (negative_ratio > 0.0 and self.negative_slots < 1):
                raise ValueError(
                    "balanced ROI batches require at least two subtitle-present slots "
                    "and one empty slot when both classes exist"
                )
        elif self.negative_indices:
            self.negative_slots = batch_size
            self.positive_slots = 0
        else:
            self.negative_slots = 0
            self.positive_slots = batch_size

        self._pairs = self._build_positive_pairs()
        self._segment_groups = self._build_segment_groups()
        counts = []
        if self.positive_indices and self.positive_slots > 0:
            counts.append(math.ceil(len(self.positive_indices) / self.positive_slots))
        if self.negative_indices and self.negative_slots > 0:
            counts.append(math.ceil(len(self.negative_indices) / self.negative_slots))
        self._batch_count = max(counts, default=0)

    def _build_positive_pairs(self) -> list[tuple[int, int]]:
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index in self.positive_indices:
            sample = self.samples[index]
            groups[(str(sample.root.resolve()), sample.segment_id)].append(index)
        pairs: list[tuple[int, int]] = []
        for indices in groups.values():
            for offset, left_index in enumerate(indices):
                for right_index in indices[offset + 1 :]:
                    pairs.append((left_index, right_index))
        return pairs

    def _build_segment_groups(self) -> list[list[int]]:
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index in self.positive_indices:
            sample = self.samples[index]
            groups[(str(sample.root.resolve()), sample.segment_id)].append(index)
        return [indices for indices in groups.values() if len(indices) >= 2]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._batch_count

    @staticmethod
    def _take(
        queue: list[int],
        pool: list[int],
        count: int,
        selected: set[int],
        rng: random.Random,
    ) -> list[int]:
        result: list[int] = []
        attempts = 0
        while len(result) < count:
            if not queue:
                queue.extend(pool)
                rng.shuffle(queue)
            value = queue.pop()
            attempts += 1
            if value not in selected or len(pool) < count:
                result.append(value)
                selected.add(value)
                attempts = 0
            elif attempts > len(pool) * 2:
                result.append(value)
        return result

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        positive_queue = list(self.positive_indices)
        negative_queue = list(self.negative_indices)
        pairs = list(self._pairs)
        segment_groups = [list(group) for group in self._segment_groups]
        rng.shuffle(positive_queue)
        rng.shuffle(negative_queue)
        rng.shuffle(pairs)
        rng.shuffle(segment_groups)
        unseen_positive = set(self.positive_indices)

        for batch_number in range(self._batch_count):
            positive_batch: list[int] = []
            batches_left = self._batch_count - batch_number
            if segment_groups and self.positive_slots >= _EMBEDDING_SAMPLES_PER_SEGMENT:
                segments_per_batch = max(1, self.positive_slots // _EMBEDDING_SAMPLES_PER_SEGMENT)
                selected: set[int] = set()
                for _ in range(min(segments_per_batch, len(segment_groups))):
                    group = segment_groups.pop(0)
                    segment_groups.append(group)
                    group_queue = [index for index in group if index in unseen_positive and index not in selected]
                    if len(group_queue) < _EMBEDDING_SAMPLES_PER_SEGMENT:
                        group_queue = [index for index in group if index not in selected]
                    rng.shuffle(group_queue)
                    picks = group_queue[:_EMBEDDING_SAMPLES_PER_SEGMENT]
                    if len(picks) < _EMBEDDING_SAMPLES_PER_SEGMENT:
                        picks.extend(rng.choices(group, k=_EMBEDDING_SAMPLES_PER_SEGMENT - len(picks)))
                    positive_batch.extend(picks)
                    selected.update(picks)
                    unseen_positive.difference_update(picks)
                    if len(positive_batch) + _EMBEDDING_SAMPLES_PER_SEGMENT > self.positive_slots:
                        break
            elif pairs and self.positive_slots >= 2:
                # Do not sacrifice source coverage merely to repeat a pair.
                pair_capacity_left = (batches_left - 1) * self.positive_slots + (self.positive_slots - 2)
                if len(unseen_positive) <= pair_capacity_left + 2:
                    best_offset = max(
                        range(len(pairs)),
                        key=lambda offset: sum(index in unseen_positive for index in pairs[offset]),
                    )
                    pair = pairs.pop(best_offset)
                    pairs.append(pair)
                    positive_batch.extend(pair)
                    unseen_positive.difference_update(pair)

            selected = set(positive_batch)
            if len(positive_batch) < self.positive_slots:
                unseen_queue = [index for index in positive_queue if index in unseen_positive and index not in selected]
                for index in unseen_queue[: self.positive_slots - len(positive_batch)]:
                    positive_queue.remove(index)
                    positive_batch.append(index)
                    selected.add(index)
                    unseen_positive.discard(index)
            positive_batch.extend(
                self._take(
                    positive_queue,
                    self.positive_indices,
                    self.positive_slots - len(positive_batch),
                    selected,
                    rng,
                )
            )
            negative_batch = self._take(
                negative_queue,
                self.negative_indices,
                self.negative_slots,
                set(),
                rng,
            )
            batch = positive_batch + negative_batch
            rng.shuffle(batch)
            yield batch
