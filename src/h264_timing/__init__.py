"""H.264 compressed-domain subtitle timing detection.

The package contains two intentionally independent model families:

``train``
    A windowed segment proposal model that scores complete subtitle intervals.
``train-stream``
    A causal model and stateful decoder for low-latency streaming inference.

Both families share feature extraction, labels, dataset validation, and
post-processing code, but their checkpoints are not interchangeable.
"""

FEATURE_FORMAT = "subfast-net.h264-timing-features"
FEATURE_VERSION = 2
CHECKPOINT_FORMAT = "subfast-net.h264-timing-model"
CHECKPOINT_VERSION = 5
STREAM_CHECKPOINT_FORMAT = "subfast-net.h264-timing-stream-model"
STREAM_CHECKPOINT_VERSION = 1

__all__ = [
    "FEATURE_FORMAT",
    "FEATURE_VERSION",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "STREAM_CHECKPOINT_FORMAT",
    "STREAM_CHECKPOINT_VERSION",
]
