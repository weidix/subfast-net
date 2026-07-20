"""Direct H.264 subtitle-timing detector and feature extraction family."""

from subtitle_timing_core.formats import (
    H264_TIMING_FEATURE_FORMAT as FEATURE_FORMAT,
    H264_TIMING_FEATURE_VERSION as FEATURE_VERSION,
)


CHECKPOINT_FORMAT = "subfast-net.h264-timing-model"
CHECKPOINT_VERSION = 5
__all__ = [
    "FEATURE_FORMAT",
    "FEATURE_VERSION",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
]
