from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from . import (
    COMPRESSED_STREAM_CHECKPOINT_FORMAT,
    COMPRESSED_STREAM_CHECKPOINT_VERSION,
)
from subtitle_timing_core.formats import H264_TIMING_FEATURE_VERSION as FEATURE_VERSION
from subtitle_timing_stream.stream_model import (
    StreamingH264SubtitleModel,
    StreamingModelConfig,
)
from subtitle_timing_stream.stream_postprocess import StreamingDecoderConfig
from subtitle_timing_stream.streaming import StreamingSegmentDetector


class CompressedStreamingSegmentDetector(StreamingSegmentDetector):
    """Streaming detector with a strict compressed-domain input checkpoint."""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "auto",
    ) -> CompressedStreamingSegmentDetector:
        path = Path(checkpoint_path).expanduser().resolve()
        checkpoint: Any = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"invalid compressed streaming checkpoint: {path}")
        if (
            checkpoint.get("format") != COMPRESSED_STREAM_CHECKPOINT_FORMAT
            or checkpoint.get("version") != COMPRESSED_STREAM_CHECKPOINT_VERSION
            or checkpoint.get("feature_version") != FEATURE_VERSION
            or checkpoint.get("input_domain") != "h264_compressed_only"
            or checkpoint.get("pixel_decode_required") is not False
        ):
            raise ValueError(f"unsupported compressed streaming checkpoint: {path}")
        if checkpoint.get("model_output_contract") not in {
            "causal_compressed_streaming_presence_events",
            "causal_compressed_streaming_presence_events_anchor",
        }:
            raise ValueError(
                f"checkpoint does not contain compressed streaming output: {path}"
            )
        required = (
            "model_config",
            "model",
            "compressed_feature_names",
            "feature_mean",
            "feature_std",
            "streaming_decoder_config",
        )
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(
                "compressed streaming checkpoint is missing required fields: "
                + ", ".join(missing)
            )
        if "visual_feature_settings" in checkpoint:
            raise ValueError(
                "compressed streaming checkpoint must not define visual inference features"
            )
        try:
            model_config = StreamingModelConfig(**dict(checkpoint["model_config"]))
            decoder_config = StreamingDecoderConfig(
                **dict(checkpoint["streaming_decoder_config"])
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid compressed streaming checkpoint configuration: {path}"
            ) from error
        feature_names = list(checkpoint["compressed_feature_names"])
        if model_config.feature_count != len(feature_names):
            raise ValueError(
                "compressed feature schema width does not match the model configuration"
            )
        model = StreamingH264SubtitleModel(model_config)
        model.load_state_dict(checkpoint["model"], strict=True)
        return cls(
            model,
            checkpoint["feature_mean"],
            checkpoint["feature_std"],
            decoder_config,
            device=device,
            feature_names=feature_names,
        )
