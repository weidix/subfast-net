"""Small metrics helpers shared by binary similarity tasks."""

from __future__ import annotations


def similarity_percentile(values: list[float], percentile: float) -> float:
    """Return a linearly interpolated percentile, or zero for no values."""

    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


__all__ = ["similarity_percentile"]
