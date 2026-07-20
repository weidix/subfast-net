"""Compressed-domain H.264 causal subtitle-timing training family."""


COMPRESSED_STREAM_CHECKPOINT_FORMAT = (
    "subfast-net.h264-timing-compressed-stream-model"
)
COMPRESSED_STREAM_CHECKPOINT_VERSION = 1

__all__ = [
    "COMPRESSED_STREAM_CHECKPOINT_FORMAT",
    "COMPRESSED_STREAM_CHECKPOINT_VERSION",
    "CompressedStreamingPrepareSettings",
    "CompressedStreamingSegmentDetector",
    "CompressedStreamingTrainSettings",
    "prepare_compressed_streaming_features",
    "train_compressed_streaming",
]


def __getattr__(name: str):
    if name in {
        "CompressedStreamingPrepareSettings",
        "prepare_compressed_streaming_features",
    }:
        from .compressed_stream_prepare import (
            CompressedStreamingPrepareSettings,
            prepare_compressed_streaming_features,
        )

        return {
            "CompressedStreamingPrepareSettings": CompressedStreamingPrepareSettings,
            "prepare_compressed_streaming_features": prepare_compressed_streaming_features,
        }[name]
    if name in {"CompressedStreamingTrainSettings", "train_compressed_streaming"}:
        from .compressed_stream_train import (
            CompressedStreamingTrainSettings,
            train_compressed_streaming,
        )

        return {
            "CompressedStreamingTrainSettings": CompressedStreamingTrainSettings,
            "train_compressed_streaming": train_compressed_streaming,
        }[name]
    if name == "CompressedStreamingSegmentDetector":
        from .compressed_streaming import CompressedStreamingSegmentDetector

        return CompressedStreamingSegmentDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
