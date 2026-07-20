from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from h264_timing.dataset import FeatureCache, intervals_inside_cache
from h264_timing.labels import read_intervals
from h264_timing.metrics import interval_metrics
from h264_timing.postprocess import SegmentPrediction, write_segment_predictions
from h264_timing.stream_model import StreamingH264SubtitleModel, StreamingModelConfig
from h264_timing.stream_postprocess import StreamingDecoderConfig
from h264_timing.streaming import (
    StreamSample,
    StreamingSegmentDetector,
)

from . import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    FEATURE_VERSION,
    INPUT_DOMAIN,
)
from .features import (
    FullFrameFeatureSettings,
    extract_full_frame_feature_cache,
)


def load_full_frame_detector(
    checkpoint_path: Path,
    *,
    device: str | torch.device = "auto",
) -> tuple[StreamingSegmentDetector, dict[str, object]]:
    path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid full-frame checkpoint: {path}")
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("version") != CHECKPOINT_VERSION
        or checkpoint.get("feature_version") != FEATURE_VERSION
        or checkpoint.get("input_domain") != INPUT_DOMAIN
        or checkpoint.get("pixel_decode_required") is not True
    ):
        raise ValueError(f"unsupported full-frame checkpoint: {path}")
    if checkpoint.get("model_output_contract") not in {
        "causal_streaming_presence_events",
        "causal_streaming_presence_events_anchor",
    }:
        raise ValueError(f"checkpoint has an unsupported output contract: {path}")
    required = (
        "model_config",
        "model",
        "feature_names",
        "feature_mean",
        "feature_std",
        "streaming_decoder_config",
        "full_frame_feature_settings",
    )
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ValueError(
            "full-frame checkpoint is missing required fields: "
            + ", ".join(missing)
        )
    try:
        model_config = StreamingModelConfig(**dict(checkpoint["model_config"]))
        decoder_config = StreamingDecoderConfig(
            **dict(checkpoint["streaming_decoder_config"])
        )
        FullFrameFeatureSettings.from_dict(
            dict(checkpoint["full_frame_feature_settings"])
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid full-frame checkpoint configuration: {path}"
        ) from error
    if model_config.use_byte_branch:
        raise ValueError("full-frame checkpoints cannot use H.264 byte tokens")
    expected_output_contract = (
        "causal_streaming_presence_events_anchor"
        if model_config.use_segment_head
        else "causal_streaming_presence_events"
    )
    if checkpoint["model_output_contract"] != expected_output_contract:
        raise ValueError("full-frame checkpoint model and output contracts differ")
    model = StreamingH264SubtitleModel(model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    detector = StreamingSegmentDetector(
        model,
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
        decoder_config,
        device=device,
        feature_names=checkpoint["feature_names"],
    )
    return detector, checkpoint


def infer_full_frame_video(
    video: Path,
    checkpoint_path: Path,
    output_csv: Path,
    *,
    labels_path: Path | None = None,
    device: str = "auto",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict[str, object]:
    source = video.expanduser().resolve()
    checkpoint_file = checkpoint_path.expanduser().resolve()
    output = output_csv.expanduser().resolve()
    if output in {source, checkpoint_file}:
        raise ValueError("inference output must not replace the video or checkpoint")
    detector, checkpoint = load_full_frame_detector(
        checkpoint_file,
        device=device,
    )
    settings = FullFrameFeatureSettings.from_dict(
        dict(checkpoint["full_frame_feature_settings"])
    )
    chunk_frames = int(checkpoint.get("inference_chunk_frames", 128))
    if chunk_frames <= 0:
        raise ValueError("checkpoint inference chunk size must be positive")
    predictions: list[SegmentPrediction] = []
    with tempfile.TemporaryDirectory(prefix="full-frame-timing-infer-") as temporary:
        feature_dir = Path(temporary) / "features"
        extract_full_frame_feature_cache(
            source,
            feature_dir,
            settings=settings,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
        cache = FeatureCache(feature_dir)
        try:
            if cache.feature_names != list(detector.feature_names):
                raise ValueError(
                    "video full-frame feature schema does not match checkpoint"
                )
            if (
                cache.meta.get("full_frame_feature_settings")
                != checkpoint["full_frame_feature_settings"]
            ):
                raise ValueError(
                    "video full-frame feature settings do not match checkpoint"
                )
            for start in range(0, len(cache.timestamps), chunk_frames):
                stop = min(start + chunk_frames, len(cache.timestamps))
                samples = (
                    StreamSample(
                        timestamp_seconds=float(cache.timestamps[index]),
                        duration_seconds=float(cache.durations[index]),
                        features=np.asarray(cache.features[index]),
                        tokens=None,
                    )
                    for index in range(start, stop)
                )
                predictions.extend(detector.push_many(samples))
            predictions.extend(detector.close())
            write_segment_predictions(output, predictions)
            result: dict[str, object] = {
                "output": str(output),
                "interval_count": len(predictions),
                "frame_count": len(cache.timestamps),
                "input_domain": INPUT_DOMAIN,
                "pixel_decode": True,
                "spatial_contract": "full_frame",
                "model_output_contract": checkpoint["model_output_contract"],
            }
            if labels_path is not None:
                labels = labels_path.expanduser().resolve()
                target = intervals_inside_cache(cache, read_intervals(labels))
                timestamps = np.asarray(cache.timestamps, dtype=np.float64)
                positive_steps = np.diff(timestamps)
                positive_steps = positive_steps[positive_steps > 0.0]
                frame_tolerance = (
                    float(np.median(positive_steps))
                    if positive_steps.size
                    else 1.0 / 30.0
                )
                cache_start, cache_end = cache.coverage_range_seconds
                result["metrics"] = interval_metrics(
                    [prediction.to_interval() for prediction in predictions],
                    target,
                    video_duration_seconds=cache_end - cache_start,
                    frame_tolerance_seconds=frame_tolerance,
                )
            return result
        finally:
            cache.release()
