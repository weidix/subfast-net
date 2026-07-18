from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import CHECKPOINT_FORMAT, CHECKPOINT_VERSION, FEATURE_VERSION
from .bitstream import (
    EXACT_BOTTOM_SLICES,
    PAYLOAD_TAIL_PROXY,
    FeatureSettings,
    extract_feature_cache,
    probe_stream,
)
from .dataset import FeatureCache, intervals_inside_cache
from .labels import parse_srt_timing, read_intervals, write_intervals
from .metrics import interval_metrics
from .model import H264SubtitleSegmentModel, ModelConfig
from .postprocess import (
    SegmentPrediction,
    SegmentSelectionConfig,
    select_segments,
    write_segment_predictions,
)
from .prepare import PrepareSettings, prepare_dataset
from .predict import predict_cache
from .synthesis import CueScheduleSettings, SynthesisSettings, synthesize_segment
from .stream_train import StreamingTrainSettings, train_streaming
from .streaming import StreamSample, StreamingSegmentDetector
from .train import TrainSettings, select_device, train
from .visual import VisualFeatureSettings, extract_visual_feature_cache


def build_parser(prog: str = "h264-timing") -> argparse.ArgumentParser:
    """Build the H.264 command parser.

    The parser is intentionally independent from the project's top-level
    parser.  The public entry point is ``h264-timing ...``; keeping the parser
    local lets this independent model family retain its own command options.
    """

    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Detect subtitle segments from H.264 statistics plus compact bottom-ROI "
            "visual features."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="Read container/stream metadata only")
    probe.add_argument("video", type=Path)
    probe.add_argument("--ffprobe", default="ffprobe")

    extract = subparsers.add_parser(
        "extract",
        help="Build a disk-backed compressed feature cache without pixel decode",
    )
    extract.add_argument("video", type=Path)
    extract.add_argument("output_dir", type=Path)
    extract.add_argument("--ffprobe", default="ffprobe")
    extract.add_argument("--start", type=float)
    extract.add_argument("--duration", type=float)
    extract.add_argument("--max-packets", type=int)
    extract.add_argument("--tokens", type=int, default=256)
    extract.add_argument("--histogram-bins", type=int, default=16)
    extract.add_argument("--payload-segments", type=int, default=4)
    extract.add_argument(
        "--spatial-mode",
        choices=(EXACT_BOTTOM_SLICES, PAYLOAD_TAIL_PROXY),
        default=EXACT_BOTTOM_SLICES,
    )
    extract.add_argument("--overwrite", action="store_true")

    extract_visual = subparsers.add_parser(
        "extract-visual",
        help="Decode the bottom ROI into compact edge features beside a packet cache",
    )
    extract_visual.add_argument("feature_dir", type=Path)
    extract_visual.add_argument("--ffmpeg", default="ffmpeg")
    extract_visual.add_argument("--width", type=int, default=256)
    extract_visual.add_argument("--height", type=int, default=32)
    extract_visual.add_argument("--overwrite", action="store_true")

    normalize_srt = subparsers.add_parser(
        "normalize-srt",
        help="Convert SRT timing lines to strict CSV without using subtitle text",
    )
    normalize_srt.add_argument("input_srt", type=Path)
    normalize_srt.add_argument("output_csv", type=Path)
    normalize_srt.add_argument("--drop-invalid", action="store_true")

    synthesize = subparsers.add_parser(
        "synthesize",
        help="Burn an SRT-aligned subtitle segment into the bottom region for dataset creation",
    )
    synthesize.add_argument("video", type=Path)
    synthesize.add_argument("srt", type=Path)
    synthesize.add_argument("output_video", type=Path)
    synthesize.add_argument("output_labels", type=Path)
    synthesize.add_argument("--start", type=float, required=True)
    synthesize.add_argument("--end", type=float, required=True)
    synthesize.add_argument("--ffmpeg", default="ffmpeg")
    synthesize.add_argument("--ffprobe", default="ffprobe")
    synthesize.add_argument("--font", type=Path)
    synthesize.add_argument("--font-size", type=int, default=54)
    synthesize.add_argument("--minimum-font-size", type=int, default=24)
    synthesize.add_argument(
        "--encoder",
        choices=("h264_videotoolbox", "libx264"),
        default="libx264",
        help="libx264 is required for the five-slice exact-ROI P1 contract",
    )
    synthesize.add_argument("--bit-rate", type=int, default=6_000_000)
    synthesize.add_argument(
        "--schedule",
        choices=("source_timing", "randomized_signal", "none"),
        default="source_timing",
    )
    synthesize.add_argument("--seed", type=int, default=2026)
    synthesize.add_argument("--minimum-cue-duration", type=float, default=0.5)
    synthesize.add_argument("--maximum-cue-duration", type=float, default=5.0)
    synthesize.add_argument("--minimum-cue-gap", type=float, default=0.5)
    synthesize.add_argument("--maximum-cue-gap", type=float, default=4.0)
    synthesize.add_argument("--pair-id")
    synthesize.add_argument("--overwrite", action="store_true")

    prepare = subparsers.add_parser(
        "prepare", help="Build paired P1 samples, features, manifest, and dataset audit"
    )
    prepare.add_argument("video", type=Path)
    prepare.add_argument("srt", type=Path)
    prepare.add_argument("sample_plan", type=Path)
    prepare.add_argument("output_root", type=Path)
    prepare.add_argument("--source-group", default="p1-signal-validation")
    prepare.add_argument(
        "--signal-schedule",
        choices=("randomized_signal", "source_timing"),
        default="randomized_signal",
    )
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--font", type=Path)
    prepare.add_argument("--font-size", type=int, default=54)
    prepare.add_argument("--minimum-font-size", type=int, default=24)
    prepare.add_argument("--minimum-cue-duration", type=float, default=0.5)
    prepare.add_argument("--maximum-cue-duration", type=float, default=5.0)
    prepare.add_argument("--minimum-cue-gap", type=float, default=0.5)
    prepare.add_argument("--maximum-cue-gap", type=float, default=4.0)
    prepare.add_argument("--maximum-source-characters", type=int, default=72)
    prepare.add_argument("--maximum-source-lines", type=int, default=2)
    prepare.add_argument("--minimum-split-active-ratio", type=float, default=0.35)
    prepare.add_argument("--maximum-split-active-ratio", type=float, default=0.70)
    prepare.add_argument("--temporal-guard", type=float, default=10.0)
    prepare.add_argument("--ffmpeg", default="ffmpeg")
    prepare.add_argument("--ffprobe", default="ffprobe")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--no-resume", action="store_true")

    training = subparsers.add_parser(
        "train", help="Train direct scored start/end segment proposals"
    )
    training.add_argument("manifest", type=Path)
    training.add_argument("output_dir", type=Path)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--window-frames", type=int, default=512)
    training.add_argument("--stride-frames", type=int, default=256)
    training.add_argument("--boundary-event-sigma", type=float, default=0.05)
    training.add_argument("--target-recall", type=float, default=1.0)
    training.add_argument("--minimum-duration", type=float, default=0.20)
    training.add_argument("--maximum-duration", type=float, default=8.0)
    training.add_argument("--width", type=int, default=64)
    training.add_argument("--temporal-layers", type=int, default=7)
    training.add_argument("--recurrent-layers", type=int, default=1)
    training.add_argument("--dropout", type=float, default=0.10)
    training.add_argument("--use-byte-branch", action="store_true")
    training.add_argument("--max-train-windows", type=int)
    training.add_argument("--max-val-windows", type=int)
    training.add_argument("--seed", type=int, default=2026)
    training.add_argument("--device", default="auto")
    training.add_argument(
        "--validation-mode",
        choices=("held_out", "diagnostic_temporal"),
        default="held_out",
    )
    training.add_argument("--temporal-guard", type=float, default=10.0)

    stream_training = subparsers.add_parser(
        "train-stream", help="Train the independent causal streaming model"
    )
    stream_training.add_argument("manifest", type=Path)
    stream_training.add_argument("output_dir", type=Path)
    stream_training.add_argument("--epochs", type=int, default=20)
    stream_training.add_argument("--batch-size", type=int, default=8)
    stream_training.add_argument("--lr", type=float, default=1e-3)
    stream_training.add_argument("--weight-decay", type=float, default=1e-4)
    stream_training.add_argument("--window-frames", type=int, default=512)
    stream_training.add_argument("--stride-frames", type=int, default=256)
    stream_training.add_argument("--boundary-event-sigma", type=float, default=0.05)
    stream_training.add_argument("--target-recall", type=float, default=1.0)
    stream_training.add_argument("--minimum-duration", type=float, default=0.20)
    stream_training.add_argument("--maximum-duration", type=float, default=8.0)
    stream_training.add_argument("--width", type=int, default=64)
    stream_training.add_argument("--temporal-layers", type=int, default=6)
    stream_training.add_argument("--recurrent-layers", type=int, default=1)
    stream_training.add_argument("--dropout", type=float, default=0.10)
    stream_training.add_argument("--use-byte-branch", action="store_true")
    stream_training.add_argument("--max-train-windows", type=int)
    stream_training.add_argument("--max-val-windows", type=int)
    stream_training.add_argument("--inference-chunk-frames", type=int, default=128)
    stream_training.add_argument("--seed", type=int, default=2026)
    stream_training.add_argument("--device", default="auto")
    stream_training.add_argument(
        "--validation-mode",
        choices=("held_out", "diagnostic_temporal"),
        default="held_out",
    )
    stream_training.add_argument("--temporal-guard", type=float, default=10.0)

    infer = subparsers.add_parser(
        "infer", help="Emit subtitle intervals from a feature cache"
    )
    infer.add_argument("feature_dir", type=Path)
    infer.add_argument("checkpoint", type=Path)
    infer.add_argument("output_csv", type=Path)
    infer.add_argument("--batch-size", type=int, default=8)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--score-threshold", type=float)
    infer.add_argument("--nms-iou", type=float)
    infer.add_argument("--minimum-duration", type=float)
    infer.add_argument("--maximum-duration", type=float)
    infer.add_argument("--boundary-event-threshold", type=float)
    infer.add_argument("--start-boundary-refinement", type=float)
    infer.add_argument("--end-boundary-refinement", type=float)
    infer.add_argument("--end-event-relative-threshold", type=float)
    infer.add_argument(
        "--require-boundary-events",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    infer.add_argument(
        "--labels", type=Path, help="Optional reference CSV for evaluation"
    )

    stream_infer = subparsers.add_parser(
        "stream-infer", help="Consume model-ready feature samples as JSON Lines"
    )
    stream_infer.add_argument("checkpoint", type=Path)
    stream_infer.add_argument("--device", default="auto")
    return parser


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[H264SubtitleSegmentModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format") != CHECKPOINT_FORMAT
        or checkpoint.get("version") != CHECKPOINT_VERSION
        or checkpoint.get("feature_version") != FEATURE_VERSION
    ):
        raise ValueError(f"unsupported checkpoint: {checkpoint_path}")
    if checkpoint.get("model_output_contract") != "direct_scored_start_end_segments":
        raise ValueError(
            f"checkpoint does not contain direct segment output: {checkpoint_path}"
        )
    model = H264SubtitleSegmentModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def _infer(args: argparse.Namespace) -> dict:
    feature_dir = args.feature_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    labels_path = (
        args.labels.expanduser().resolve() if args.labels is not None else None
    )
    if (
        output_csv == checkpoint_path
        or output_csv == feature_dir
        or output_csv.is_relative_to(feature_dir)
    ):
        raise ValueError(
            "inference output must not replace checkpoint, labels, or feature cache"
        )
    device = select_device(args.device)
    cache = FeatureCache(feature_dir)
    model, checkpoint = _load_model(checkpoint_path, device)
    if cache.feature_names != list(checkpoint["feature_names"]):
        raise ValueError("feature cache schema does not match checkpoint")
    if dict(cache.meta["feature_settings"]) != dict(checkpoint["feature_settings"]):
        raise ValueError("feature cache and checkpoint use different feature settings")
    if cache.visual_feature_settings != checkpoint.get("visual_feature_settings"):
        raise ValueError("feature cache and checkpoint use different visual features")
    if cache.meta["spatial_contract"] != checkpoint["spatial_contract"]:
        raise ValueError("feature cache and checkpoint use different spatial contracts")
    proposals = predict_cache(
        model,
        cache,
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
        window_frames=int(checkpoint["window_frames"]),
        hop_frames=int(checkpoint["hop_frames"]),
        batch_size=args.batch_size,
        device=device,
    )
    calibrated = SegmentSelectionConfig(**checkpoint["segment_selection_config"])
    selection = SegmentSelectionConfig(
        score_threshold=(
            calibrated.score_threshold
            if args.score_threshold is None
            else args.score_threshold
        ),
        nms_iou_threshold=(
            calibrated.nms_iou_threshold if args.nms_iou is None else args.nms_iou
        ),
        minimum_duration_seconds=(
            calibrated.minimum_duration_seconds
            if args.minimum_duration is None
            else args.minimum_duration
        ),
        maximum_duration_seconds=(
            calibrated.maximum_duration_seconds
            if args.maximum_duration is None
            else args.maximum_duration
        ),
        peak_radius_frames=calibrated.peak_radius_frames,
        boundary_event_threshold=(
            calibrated.boundary_event_threshold
            if args.boundary_event_threshold is None
            else args.boundary_event_threshold
        ),
        start_boundary_refinement_seconds=(
            calibrated.start_boundary_refinement_seconds
            if args.start_boundary_refinement is None
            else args.start_boundary_refinement
        ),
        end_boundary_refinement_seconds=(
            calibrated.end_boundary_refinement_seconds
            if args.end_boundary_refinement is None
            else args.end_boundary_refinement
        ),
        end_event_relative_threshold=(
            calibrated.end_event_relative_threshold
            if args.end_event_relative_threshold is None
            else args.end_event_relative_threshold
        ),
        boundary_event_peak_radius_frames=(
            calibrated.boundary_event_peak_radius_frames
        ),
        require_boundary_events=(
            calibrated.require_boundary_events
            if args.require_boundary_events is None
            else args.require_boundary_events
        ),
    )
    predicted_segments = select_segments(
        proposals,
        np.asarray(cache.timestamps),
        config=selection,
    )
    write_segment_predictions(output_csv, predicted_segments)
    result: dict[str, object] = {
        "output": str(output_csv),
        "interval_count": len(predicted_segments),
        "packet_count": len(cache.timestamps),
        "device": str(device),
        "pixel_decode": True,
        "model_output_contract": "direct_scored_start_end_segments",
        "segment_selection_config": selection.to_dict(),
    }
    if labels_path is not None:
        target = intervals_inside_cache(cache, read_intervals(labels_path))
        timestamps = np.asarray(cache.timestamps)
        cache_start, cache_end = cache.coverage_range_seconds
        duration = cache_end - cache_start
        positive_steps = np.diff(timestamps)
        positive_steps = positive_steps[positive_steps > 0]
        frame_tolerance = (
            float(np.median(positive_steps)) if positive_steps.size else 1 / 30
        )
        result["metrics"] = interval_metrics(
            [segment.to_interval() for segment in predicted_segments],
            target,
            video_duration_seconds=duration,
            frame_tolerance_seconds=frame_tolerance,
        )
    return result


def _stream_infer(args: argparse.Namespace) -> None:
    detector = StreamingSegmentDetector.from_checkpoint(
        args.checkpoint,
        device=args.device,
    )
    sample_count = 0
    interrupted = False
    try:
        for line_number, line in enumerate(sys.stdin, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid stream JSON at input line {line_number}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(
                    f"stream input line {line_number} must contain a JSON object"
                )
            if payload.get("type") == "close":
                break
            sample = StreamSample(
                timestamp_seconds=payload["timestamp_seconds"],
                duration_seconds=payload["duration_seconds"],
                features=payload["features"],
                tokens=payload.get("tokens"),
            )
            sample_count += 1
            _write_stream_segments(detector.push(sample))
    except KeyboardInterrupt:
        interrupted = True
    _write_stream_segments(detector.close())
    print(
        json.dumps(
            {
                "type": "closed",
                "sample_count": sample_count,
                "interrupted": interrupted,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


def _write_stream_segments(segments: tuple[SegmentPrediction, ...]) -> None:
    for segment in segments:
        print(
            json.dumps(
                {
                    "type": "segment",
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "confidence": segment.confidence,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            flush=True,
        )


def main(argv: list[str] | None = None, *, prog: str = "h264-timing") -> None:
    """Run one H.264 timing command and print its JSON result."""

    args = build_parser(prog=prog).parse_args(argv)
    if args.command == "stream-infer":
        _stream_infer(args)
        return
    if args.command == "probe":
        info = probe_stream(args.video, ffprobe=args.ffprobe)
        result = {
            **info.__dict__,
            "source": str(args.video.expanduser().resolve()),
            "decode_contract": {"pixel_decode": False, "operation": "container_probe"},
        }
    elif args.command == "extract":
        meta = extract_feature_cache(
            args.video,
            args.output_dir,
            settings=FeatureSettings(
                token_count=args.tokens,
                histogram_bins=args.histogram_bins,
                payload_segments=args.payload_segments,
                payload_tail_ratio=0.20,
                spatial_mode=args.spatial_mode,
            ),
            ffprobe=args.ffprobe,
            start_seconds=args.start,
            duration_seconds=args.duration,
            max_packets=args.max_packets,
            overwrite=args.overwrite,
        )
        result = {
            "output": str(args.output_dir.expanduser().resolve()),
            "packet_count": meta["packet_count"],
            "time_range_seconds": meta["time_range_seconds"],
            "feature_count": len(meta["feature_names"]),
            "byte_totals": meta["byte_totals"],
            "decode_contract": meta["decode_contract"],
            "spatial_contract": meta["spatial_contract"],
        }
    elif args.command == "extract-visual":
        meta = extract_visual_feature_cache(
            args.feature_dir,
            settings=VisualFeatureSettings(width=args.width, height=args.height),
            ffmpeg=args.ffmpeg,
            overwrite=args.overwrite,
        )
        result = {
            "output": str(args.feature_dir.expanduser().resolve()),
            "frame_count": meta["frame_count"],
            "feature_count": len(meta["feature_names"]),
            "decode_contract": meta["decode_contract"],
        }
    elif args.command == "normalize-srt":
        input_srt = args.input_srt.expanduser().resolve()
        output_csv = args.output_csv.expanduser().resolve()
        if input_srt == output_csv:
            raise ValueError("normalized CSV output must not replace the source SRT")
        intervals, invalid = parse_srt_timing(input_srt, drop_invalid=args.drop_invalid)
        write_intervals(output_csv, intervals)
        result = {
            "output": str(output_csv),
            "interval_count": len(intervals),
            "dropped_invalid_count": len(invalid),
            "dropped_invalid_lines": [item.line_number for item in invalid],
            "subtitle_text_used": False,
        }
    elif args.command == "synthesize":
        role = {
            "source_timing": "source_timing",
            "randomized_signal": "subtitle_signal",
            "none": "clean_control",
        }[args.schedule]
        result = synthesize_segment(
            SynthesisSettings(
                source_video=args.video,
                source_srt=args.srt,
                output_video=args.output_video,
                output_labels=args.output_labels,
                start_seconds=args.start,
                end_seconds=args.end,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                font_path=args.font,
                font_size=args.font_size,
                minimum_font_size=args.minimum_font_size,
                encoder=args.encoder,
                bit_rate=args.bit_rate,
                overwrite=args.overwrite,
                cue_schedule=CueScheduleSettings(
                    mode=args.schedule,
                    random_seed=args.seed,
                    minimum_duration_seconds=args.minimum_cue_duration,
                    maximum_duration_seconds=args.maximum_cue_duration,
                    minimum_gap_seconds=args.minimum_cue_gap,
                    maximum_gap_seconds=args.maximum_cue_gap,
                ),
                signal_validation_role=role,
                pair_id=args.pair_id,
            )
        )
    elif args.command == "prepare":
        result = prepare_dataset(
            PrepareSettings(
                source_video=args.video,
                source_srt=args.srt,
                sample_plan=args.sample_plan,
                output_root=args.output_root,
                source_group=args.source_group,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                font_path=args.font,
                font_size=args.font_size,
                minimum_font_size=args.minimum_font_size,
                base_random_seed=args.seed,
                signal_schedule_mode=args.signal_schedule,
                minimum_cue_duration_seconds=args.minimum_cue_duration,
                maximum_cue_duration_seconds=args.maximum_cue_duration,
                minimum_cue_gap_seconds=args.minimum_cue_gap,
                maximum_cue_gap_seconds=args.maximum_cue_gap,
                maximum_source_characters=args.maximum_source_characters,
                maximum_source_lines=args.maximum_source_lines,
                minimum_split_signal_active_ratio=args.minimum_split_active_ratio,
                maximum_split_signal_active_ratio=args.maximum_split_active_ratio,
                temporal_guard_seconds=args.temporal_guard,
                overwrite=args.overwrite,
                resume=not args.no_resume and not args.overwrite,
            )
        )
    elif args.command == "train-stream":
        result = train_streaming(
            StreamingTrainSettings(
                manifest=args.manifest,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                window_frames=args.window_frames,
                stride_frames=args.stride_frames,
                boundary_event_sigma_seconds=args.boundary_event_sigma,
                target_recall=args.target_recall,
                minimum_duration_seconds=args.minimum_duration,
                maximum_duration_seconds=args.maximum_duration,
                width=args.width,
                temporal_layers=args.temporal_layers,
                recurrent_layers=args.recurrent_layers,
                dropout=args.dropout,
                use_byte_branch=args.use_byte_branch,
                max_train_windows=args.max_train_windows,
                max_val_windows=args.max_val_windows,
                inference_chunk_frames=args.inference_chunk_frames,
                seed=args.seed,
                device=args.device,
                validation_mode=args.validation_mode,
                temporal_guard_seconds=args.temporal_guard,
            )
        )
    elif args.command == "train":
        result = train(
            TrainSettings(
                manifest=args.manifest,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                window_frames=args.window_frames,
                stride_frames=args.stride_frames,
                boundary_event_sigma_seconds=args.boundary_event_sigma,
                target_recall=args.target_recall,
                minimum_duration_seconds=args.minimum_duration,
                maximum_duration_seconds=args.maximum_duration,
                width=args.width,
                temporal_layers=args.temporal_layers,
                recurrent_layers=args.recurrent_layers,
                dropout=args.dropout,
                use_byte_branch=args.use_byte_branch,
                max_train_windows=args.max_train_windows,
                max_val_windows=args.max_val_windows,
                seed=args.seed,
                device=args.device,
                validation_mode=args.validation_mode,
                temporal_guard_seconds=args.temporal_guard,
            )
        )
    else:
        result = _infer(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
