"""Full-frame visual subtitle timing detection."""

from subtitle_timing_core.formats import (
    FULL_FRAME_TIMING_FEATURE_FORMAT as FEATURE_FORMAT,
    FULL_FRAME_TIMING_FEATURE_VERSION as FEATURE_VERSION,
)


CHECKPOINT_FORMAT = "subfast-net.full-frame-timing-stream-model"
CHECKPOINT_VERSION = 1
INPUT_DOMAIN = "full_frame_visual"

__all__ = [
    "FEATURE_FORMAT",
    "FEATURE_VERSION",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "INPUT_DOMAIN",
]
