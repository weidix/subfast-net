from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import FullFrameFeatureSettings, extract_full_frame_feature_cache
from .inference import infer_full_frame_video
from .prepare import FullFramePrepareSettings, prepare_full_frame_features
from .train import FullFrameTrainSettings, train_full_frame


def build_parser(prog: str = "full-frame-timing") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Detect subtitle time intervals from decoded full-frame visual features."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="Decode one video into a compact full-frame feature cache"
    )
    extract.add_argument("video", type=Path)
    extract.add_argument("output_dir", type=Path)
    extract.add_argument("--width", type=int, default=256)
    extract.add_argument("--height", type=int, default=144)
    extract.add_argument("--tile-rows", type=int, default=8)
    extract.add_argument("--tile-columns", type=int, default=8)
    extract.add_argument("--row-bins", type=int, default=18)
    extract.add_argument("--fine-column-bins", type=int, default=32)
    extract.add_argument("--temporal-lags", type=int, nargs="+", default=(1, 4, 8))
    extract.add_argument("--ffmpeg", default="ffmpeg")
    extract.add_argument("--ffprobe", default="ffprobe")
    extract.add_argument("--overwrite", action="store_true")

    prepare = subparsers.add_parser(
        "prepare",
        help="Rebuild a timing manifest with decoded full-frame features",
    )
    prepare.add_argument("manifest", type=Path)
    prepare.add_argument("output_root", type=Path)
    prepare.add_argument("--video-dir", type=Path)
    prepare.add_argument("--width", type=int, default=256)
    prepare.add_argument("--height", type=int, default=144)
    prepare.add_argument("--tile-rows", type=int, default=8)
    prepare.add_argument("--tile-columns", type=int, default=8)
    prepare.add_argument("--row-bins", type=int, default=18)
    prepare.add_argument("--fine-column-bins", type=int, default=32)
    prepare.add_argument("--temporal-lags", type=int, nargs="+", default=(1, 4, 8))
    prepare.add_argument("--workers", type=int, default=2)
    prepare.add_argument("--ffmpeg", default="ffmpeg")
    prepare.add_argument("--overwrite", action="store_true")

    training = subparsers.add_parser(
        "train", help="Train and calibrate the causal full-frame timing model"
    )
    training.add_argument("manifest", type=Path)
    training.add_argument("output_dir", type=Path)
    training.add_argument("--epochs", type=int, default=24)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--window-frames", type=int, default=512)
    training.add_argument("--stride-frames", type=int, default=256)
    training.add_argument("--boundary-event-sigma", type=float, default=0.05)
    training.add_argument("--target-recall", type=float, default=1.0)
    training.add_argument("--minimum-duration", type=float, default=0.20)
    training.add_argument("--maximum-duration", type=float, default=8.0)
    training.add_argument("--width", type=int, default=64)
    training.add_argument("--temporal-layers", type=int, default=6)
    training.add_argument("--recurrent-layers", type=int, default=1)
    training.add_argument("--dropout", type=float, default=0.10)
    training.add_argument("--use-segment-head", action="store_true")
    training.add_argument("--segment-boundary-weight", type=float, default=2.0)
    training.add_argument("--segment-loss-weight", type=float, default=0.25)
    training.add_argument("--negative-weight", type=float, default=1.0)
    training.add_argument("--boundary-event-loss-weight", type=float, default=1.5)
    training.add_argument("--clean-negative-weight", type=float, default=4.0)
    training.add_argument("--short-segment-weight", type=float, default=2.0)
    training.add_argument("--max-train-windows", type=int)
    training.add_argument("--max-val-windows", type=int)
    training.add_argument("--inference-chunk-frames", type=int, default=128)
    training.add_argument("--initial-checkpoint", type=Path)
    training.add_argument(
        "--calibration-profile",
        choices=("fast", "full"),
        default="fast",
        help="Use a compact grid for short experiments or the exhaustive grid",
    )
    training.add_argument("--seed", type=int, default=2026)
    training.add_argument("--device", default="auto")
    training.add_argument(
        "--validation-mode",
        choices=("held_out", "diagnostic_temporal"),
        default="held_out",
    )
    training.add_argument("--temporal-guard", type=float, default=10.0)

    infer = subparsers.add_parser(
        "infer", help="Decode a full video and write detected subtitle intervals"
    )
    infer.add_argument("video", type=Path)
    infer.add_argument("checkpoint", type=Path)
    infer.add_argument("output_csv", type=Path)
    infer.add_argument("--labels", type=Path)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--ffmpeg", default="ffmpeg")
    infer.add_argument("--ffprobe", default="ffprobe")
    return parser


def _feature_settings(args: argparse.Namespace) -> FullFrameFeatureSettings:
    return FullFrameFeatureSettings(
        width=args.width,
        height=args.height,
        tile_rows=args.tile_rows,
        tile_columns=args.tile_columns,
        row_bins=args.row_bins,
        fine_column_bins=args.fine_column_bins,
        temporal_lags=tuple(args.temporal_lags),
    )


def main(argv: list[str] | None = None, *, prog: str = "full-frame-timing") -> None:
    args = build_parser(prog=prog).parse_args(argv)
    if args.command == "extract":
        meta = extract_full_frame_feature_cache(
            args.video,
            args.output_dir,
            settings=_feature_settings(args),
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            overwrite=args.overwrite,
        )
        result: dict[str, object] = {
            "output": str(args.output_dir.expanduser().resolve()),
            "frame_count": meta["packet_count"],
            "feature_count": len(meta["feature_names"]),
            "decode_contract": meta["decode_contract"],
            "spatial_contract": meta["spatial_contract"],
        }
    elif args.command == "prepare":
        result = prepare_full_frame_features(
            FullFramePrepareSettings(
                manifest=args.manifest,
                output_root=args.output_root,
                video_dir=args.video_dir,
                width=args.width,
                height=args.height,
                tile_rows=args.tile_rows,
                tile_columns=args.tile_columns,
                row_bins=args.row_bins,
                fine_column_bins=args.fine_column_bins,
                temporal_lags=tuple(args.temporal_lags),
                workers=args.workers,
                ffmpeg=args.ffmpeg,
                overwrite=args.overwrite,
            )
        )
    elif args.command == "train":
        result = train_full_frame(
            FullFrameTrainSettings(
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
                use_segment_head=args.use_segment_head,
                segment_boundary_weight=args.segment_boundary_weight,
                segment_loss_weight=args.segment_loss_weight,
                negative_weight=args.negative_weight,
                boundary_event_loss_weight=args.boundary_event_loss_weight,
                clean_negative_weight=args.clean_negative_weight,
                short_segment_weight=args.short_segment_weight,
                max_train_windows=args.max_train_windows,
                max_val_windows=args.max_val_windows,
                inference_chunk_frames=args.inference_chunk_frames,
                initial_checkpoint=args.initial_checkpoint,
                calibration_profile=args.calibration_profile,
                seed=args.seed,
                device=args.device,
                validation_mode=args.validation_mode,
                temporal_guard_seconds=args.temporal_guard,
            )
        )
    else:
        result = infer_full_frame_video(
            args.video,
            args.checkpoint,
            args.output_csv,
            labels_path=args.labels,
            device=args.device,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
