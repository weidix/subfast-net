from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from subfast_roi_data.data import RoiSample


class PresenceBalancedBatchSampler(Sampler[list[int]]):
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
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.positive_indices = [index for index, sample in enumerate(samples) if sample.has_subtitle]
        self.negative_indices = [index for index, sample in enumerate(samples) if not sample.has_subtitle]
        if self.positive_indices and self.negative_indices:
            self.negative_slots = round(batch_size * negative_ratio)
            self.positive_slots = batch_size - self.negative_slots
            if negative_ratio > 0.0 and self.negative_slots == 0:
                raise ValueError("batch_size is too small to realize the requested negative ratio")
            if negative_ratio < 1.0 and self.positive_slots == 0:
                raise ValueError("batch_size is too small to realize the requested positive ratio")
        elif self.negative_indices:
            self.negative_slots = batch_size
            self.positive_slots = 0
        else:
            self.negative_slots = 0
            self.positive_slots = batch_size
        self.batch_count = (len(samples) + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self.batch_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _take(queue: list[int], pool: list[int], count: int, rng: random.Random) -> list[int]:
        selected: list[int] = []
        while len(selected) < count:
            if not queue:
                queue.extend(pool)
                rng.shuffle(queue)
            selected.append(queue.pop())
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        positives = list(self.positive_indices)
        negatives = list(self.negative_indices)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        for _ in range(self.batch_count):
            batch = self._take(positives, self.positive_indices, self.positive_slots, rng)
            batch.extend(self._take(negatives, self.negative_indices, self.negative_slots, rng))
            rng.shuffle(batch)
            yield batch
