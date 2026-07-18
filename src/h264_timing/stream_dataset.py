from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import FeatureCache, LoadedRecord, ManifestRecord, load_records
from .labels import SubtitleInterval
from .stream_labels import (
    causal_boundary_event_targets_from_intervals,
    presence_targets_from_intervals,
)


@dataclass
class StreamingLoadedRecord:
    record: ManifestRecord
    cache: FeatureCache
    intervals: list[SubtitleInterval]
    presence_targets: np.ndarray
    boundary_event_targets: np.ndarray


def load_streaming_records(
    records: list[ManifestRecord],
    *,
    boundary_event_sigma_seconds: float = 0.05,
) -> list[StreamingLoadedRecord]:
    """Load validated feature caches and derive causal streaming targets."""
    validated: list[LoadedRecord] = load_records(
        records,
        boundary_event_sigma_seconds=boundary_event_sigma_seconds,
    )
    loaded: list[StreamingLoadedRecord] = []
    for item in validated:
        timestamps = np.asarray(item.cache.timestamps, dtype=np.float64)
        loaded.append(
            StreamingLoadedRecord(
                record=item.record,
                cache=item.cache,
                intervals=item.intervals,
                presence_targets=presence_targets_from_intervals(
                    timestamps, item.intervals
                ),
                boundary_event_targets=(
                    causal_boundary_event_targets_from_intervals(
                        timestamps,
                        item.intervals,
                        sigma_seconds=boundary_event_sigma_seconds,
                    )
                ),
            )
        )
    return loaded


class StreamingTimingWindowDataset(Dataset):
    """Causal windows whose prefix is context and whose suffix receives loss."""

    def __init__(
        self,
        records: list[StreamingLoadedRecord],
        *,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        window_frames: int,
        stride_frames: int,
        max_windows: int | None = None,
    ) -> None:
        if window_frames <= 0 or stride_frames <= 0:
            raise ValueError("window and stride must be positive")
        if stride_frames > window_frames:
            raise ValueError("stride must not exceed window length")
        if max_windows is not None and max_windows <= 0:
            raise ValueError("max_windows must be positive when provided")
        self.records = records
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.history_frames = window_frames - stride_frames
        self.items = [
            (record_index, core_start)
            for record_index, record in enumerate(records)
            for core_start in range(0, len(record.cache.timestamps), stride_frames)
        ]
        if max_windows is not None and len(self.items) > max_windows:
            selected = np.linspace(
                0, len(self.items) - 1, num=max_windows, dtype=np.int64
            )
            self.items = [self.items[int(index)] for index in selected]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, core_start = self.items[index]
        record = self.records[record_index]
        core_stop = min(core_start + self.stride_frames, len(record.cache.timestamps))
        source_start = max(0, core_start - self.history_frames)
        source_stop = core_stop
        source_length = source_stop - source_start
        left_padding = self.history_frames - (core_start - source_start)
        destination_stop = left_padding + source_length
        core_stop_in_window = self.history_frames + (core_stop - core_start)

        feature_count = record.cache.features.shape[1]
        token_count = record.cache.tokens.shape[1]
        features = np.zeros((self.window_frames, feature_count), dtype=np.float32)
        tokens = np.zeros((self.window_frames, token_count), dtype=np.int64)
        presence_targets = np.zeros((self.window_frames,), dtype=np.float32)
        boundary_targets = np.zeros((self.window_frames, 2), dtype=np.float32)
        mask = np.zeros((self.window_frames,), dtype=np.float32)

        features[left_padding:destination_stop] = (
            np.asarray(
                record.cache.features[source_start:source_stop], dtype=np.float32
            )
            - self.feature_mean
        ) / self.feature_std
        tokens[left_padding:destination_stop] = np.asarray(
            record.cache.tokens[source_start:source_stop], dtype=np.int64
        )
        presence_targets[left_padding:destination_stop] = record.presence_targets[
            source_start:source_stop
        ]
        boundary_targets[left_padding:destination_stop] = record.boundary_event_targets[
            source_start:source_stop
        ]
        mask[self.history_frames : core_stop_in_window] = 1.0
        return {
            "features": torch.from_numpy(features),
            "tokens": torch.from_numpy(tokens),
            "presence_targets": torch.from_numpy(presence_targets),
            "boundary_event_targets": torch.from_numpy(boundary_targets),
            "mask": torch.from_numpy(mask),
        }
