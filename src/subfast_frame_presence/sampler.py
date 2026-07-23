from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler

from .dataset import FramePresenceDataset


class MixedMacroBatchSampler(Sampler[list[int]]):
    """Use every sample once while forcing each macro batch to mix domains and labels."""

    _REQUIRED_TYPES = ("full_frame", "roi", "random_crop")

    def __init__(
        self,
        dataset: FramePresenceDataset,
        *,
        batch_size: int,
        seed: int,
        epoch: int,
    ) -> None:
        if batch_size < len(self._REQUIRED_TYPES) * 2:
            raise ValueError("macro batch_size must be at least 6 to mix all domains and labels")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.batch_count = max(1, len(dataset) // batch_size)

    def __len__(self) -> int:
        return self.batch_count

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(f"{self.seed}:macro-batches:{self.epoch}")
        groups: dict[tuple[str, bool], list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            groups[(self.dataset.sample_type_for_index(index), self.dataset.presence_for_index(index))].append(index)
        required_keys = [(sample_type, label) for sample_type in self._REQUIRED_TYPES for label in (False, True)]
        for key in required_keys:
            if len(groups[key]) < self.batch_count:
                raise RuntimeError(
                    f"cannot construct mixed macro batches: group={key} samples={len(groups[key])} "
                    f"batches={self.batch_count}"
                )
            rng.shuffle(groups[key])

        capacities = [self.batch_size] * self.batch_count
        for index in range(len(self.dataset) - self.batch_count * self.batch_size):
            capacities[index % self.batch_count] += 1
        batches = [[] for _ in range(self.batch_count)]
        reserved: set[int] = set()
        for key in required_keys:
            for batch in batches:
                index = groups[key].pop()
                batch.append(index)
                reserved.add(index)

        remaining = [index for index in range(len(self.dataset)) if index not in reserved]
        rng.shuffle(remaining)
        cursor = 0
        for batch, capacity in zip(batches, capacities, strict=True):
            take = capacity - len(batch)
            batch.extend(remaining[cursor : cursor + take])
            cursor += take
            rng.shuffle(batch)
            labels = {self.dataset.presence_for_index(index) for index in batch}
            sample_types = {self.dataset.sample_type_for_index(index) for index in batch}
            output_sizes = {self.dataset.output_size_for_index(index) for index in batch}
            if labels != {False, True} or not set(self._REQUIRED_TYPES).issubset(sample_types):
                raise AssertionError("macro batch lost its required domain or label mix")
            if len(output_sizes) < 2:
                raise AssertionError("macro batch must contain at least two execution sizes")
        if cursor != len(remaining):
            raise AssertionError("macro batch construction did not consume every sample")
        rng.shuffle(batches)
        yield from batches


__all__ = ["MixedMacroBatchSampler"]
