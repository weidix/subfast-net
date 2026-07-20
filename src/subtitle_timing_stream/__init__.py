"""Generic causal subtitle-timing model and streaming runtime."""

from subtitle_timing_core.formats import H264_TIMING_FEATURE_VERSION as FEATURE_VERSION


STREAM_CHECKPOINT_FORMAT = "subfast-net.h264-timing-stream-model"
STREAM_CHECKPOINT_VERSION = 1

__all__ = [
    "FEATURE_VERSION",
    "STREAM_CHECKPOINT_FORMAT",
    "STREAM_CHECKPOINT_VERSION",
]
