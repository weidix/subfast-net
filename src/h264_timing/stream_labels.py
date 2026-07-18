from __future__ import annotations

import numpy as np

from .labels import SubtitleInterval


def presence_targets_from_intervals(
    timestamps: np.ndarray,
    intervals: list[SubtitleInterval],
) -> np.ndarray:
    """Mark samples whose presentation timestamp is inside a subtitle interval."""
    _validate_timestamps(timestamps)
    targets = np.zeros((len(timestamps),), dtype=np.float32)
    for interval in intervals:
        left = int(np.searchsorted(timestamps, interval.start_seconds, side="left"))
        right = int(np.searchsorted(timestamps, interval.end_seconds, side="left"))
        targets[left:right] = 1.0
    return targets


def causal_boundary_event_targets_from_intervals(
    timestamps: np.ndarray,
    intervals: list[SubtitleInterval],
    *,
    sigma_seconds: float,
) -> np.ndarray:
    """Create one-sided start/end heatmaps with no target before an event occurs."""
    _validate_timestamps(timestamps)
    if not np.isfinite(sigma_seconds) or sigma_seconds <= 0.0:
        raise ValueError("boundary event sigma must be finite and positive")
    targets = np.zeros((len(timestamps), 2), dtype=np.float32)
    radius = 3.5 * sigma_seconds
    for interval in intervals:
        for channel, center in (
            (0, interval.start_seconds),
            (1, interval.end_seconds),
        ):
            left = int(np.searchsorted(timestamps, center, side="left"))
            right = int(np.searchsorted(timestamps, center + radius, side="right"))
            if right <= left:
                continue
            distance = (timestamps[left:right] - center) / sigma_seconds
            values = np.exp(-0.5 * distance * distance).astype(np.float32)
            targets[left:right, channel] = np.maximum(
                targets[left:right, channel], values
            )
    return targets


def _validate_timestamps(timestamps: np.ndarray) -> None:
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0.0):
        raise ValueError("timestamps must be finite and sorted")
