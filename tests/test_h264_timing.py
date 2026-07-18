import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from h264_timing import (
    FEATURE_FORMAT,
    FEATURE_VERSION,
    STREAM_CHECKPOINT_FORMAT,
    STREAM_CHECKPOINT_VERSION,
    cli,
)
from h264_timing.bitstream import (
    FeatureSettings,
    PacketInfo,
    StreamInfo,
    _select_spatial_payload,
)
from h264_timing.dataset import FeatureCache, read_manifest, window_starts
from h264_timing.hashing import file_sha256
from h264_timing.labels import (
    SrtCue,
    SubtitleInterval,
    segment_targets_from_intervals,
)
from h264_timing.loss import segment_detection_loss
from h264_timing.metrics import interval_metrics
from h264_timing.model import H264SubtitleSegmentModel, ModelConfig
from h264_timing.postprocess import (
    SegmentSelectionConfig,
    select_segments,
)
from h264_timing.prepare import ClipPlan, _validate_resumed_source_range
from h264_timing.predict import predict_cache
from h264_timing.synthesis import (
    CueScheduleSettings,
    _partition_randomized_cues,
    _payload_sha256,
)
from h264_timing.stream_labels import (
    causal_boundary_event_targets_from_intervals,
    presence_targets_from_intervals,
)
from h264_timing.stream_model import (
    StreamingH264SubtitleModel,
    StreamingModelConfig,
)
from h264_timing.stream_postprocess import (
    StreamingDecoderConfig,
    StreamingSegmentDecoder,
)
from h264_timing.streaming import StreamSample, StreamingSegmentDetector
from h264_timing.train import TrainSettings, _checkpoint


def _unsigned_exp_golomb(value: int) -> str:
    payload = f"{value + 1:b}"
    return "0" * (len(payload) - 1) + payload


def _slice_nal(first_macroblock: int, payload_size: int) -> memoryview:
    bits = _unsigned_exp_golomb(first_macroblock) + _unsigned_exp_golomb(2) + "1"
    bits += "0" * (-len(bits) % 8)
    rbsp = bytes(int(bits[index : index + 8], 2) for index in range(0, len(bits), 8))
    ebsp = bytearray()
    zero_count = 0
    for value in rbsp:
        if zero_count >= 2 and value <= 3:
            ebsp.append(3)
            zero_count = 0
        ebsp.append(value)
        zero_count = zero_count + 1 if value == 0 else 0
    return memoryview(bytes([0x41]) + bytes(ebsp) + bytes([0x80]) * payload_size)


def _stream() -> StreamInfo:
    return StreamInfo(
        codec_name="h264",
        profile="Main",
        level=40,
        width=1920,
        height=1080,
        nominal_frame_rate="30000/1001",
        average_frame_rate="30000/1001",
        time_base="1/30000",
        start_time_seconds=0.0,
        duration_seconds=1.0,
        frame_count=30,
        bit_rate=6_000_000,
        is_avc=True,
        nal_length_size=4,
        format_name="mov,mp4",
        file_size=1,
        field_order="progressive",
    )


class H264TimingTests(unittest.TestCase):
    def test_exact_bottom_slice_selects_roi_and_rejects_single_slice(self):
        starts = [0, 1680, 3240, 4920, 6480]
        units = [_slice_nal(start, 10 + index) for index, start in enumerate(starts)]
        packet = PacketInfo(0.0, 0.0, 1 / 30, 0, 100, True, 0)

        spatial = _select_spatial_payload(
            units,
            packet=packet,
            stream=_stream(),
            settings=FeatureSettings(),
        )

        self.assertEqual(spatial.roi_vcl_bytes, len(units[-1]) - 1)
        self.assertEqual(
            spatial.above_roi_vcl_bytes, sum(len(unit) - 1 for unit in units[:-1])
        )
        with self.assertRaisesRegex(ValueError, "ROI boundary"):
            _select_spatial_payload(
                [_slice_nal(0, 20)],
                packet=packet,
                stream=_stream(),
                settings=FeatureSettings(),
            )

    def test_randomized_payload_partitions_are_content_disjoint(self):
        cues = [
            SrtCue(index + 1, index * 2.0, index * 2.0 + 1.0, (f"text-{index}",))
            for index in range(60)
        ]
        cues.append(SrtCue(1000, 200.0, 201.0, ("text-0",)))
        partitions = []
        for index in range(3):
            selected, audit = _partition_randomized_cues(
                cues,
                CueScheduleSettings(
                    mode="randomized_signal",
                    payload_partition_index=index,
                    payload_partition_count=3,
                ),
            )
            self.assertEqual(audit["index"], index)
            partitions.append({_payload_sha256(cue) for cue in selected})
        self.assertTrue(partitions[0].isdisjoint(partitions[1]))
        self.assertTrue(partitions[0].isdisjoint(partitions[2]))
        self.assertTrue(partitions[1].isdisjoint(partitions[2]))
        self.assertEqual(len(set().union(*partitions)), 60)

    def test_manifest_requires_pair_metadata_to_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.jsonl"
            records = [
                {
                    "video_id": "pair-signal",
                    "source_group": "source",
                    "features": "features/signal",
                    "labels": "labels/signal.csv",
                    "split": "train",
                    "source_time_offset_seconds": 12.0,
                    "pair_id": "pair",
                    "signal_validation_role": "subtitle_signal",
                },
                {
                    "video_id": "pair-clean",
                    "source_group": "source",
                    "features": "features/clean",
                    "labels": "labels/clean.csv",
                    "split": "train",
                    "source_time_offset_seconds": 12.0,
                    "pair_id": "pair",
                    "signal_validation_role": "clean_control",
                },
            ]
            manifest.write_text("\n".join(json.dumps(item) for item in records) + "\n")
            self.assertEqual(len(read_manifest(manifest)), 2)

            records[1]["source_time_offset_seconds"] = 13.0
            manifest.write_text("\n".join(json.dumps(item) for item in records) + "\n")
            with self.assertRaisesRegex(ValueError, "different source offsets"):
                read_manifest(manifest)

    def test_checkpoint_calibration_is_used_by_infer_defaults(self):
        model_config = ModelConfig(
            feature_count=1, token_count=1, width=16, temporal_layers=1
        )
        model = H264SubtitleSegmentModel(model_config)
        calibrated = SegmentSelectionConfig(
            score_threshold=0.35,
            nms_iou_threshold=0.70,
            minimum_duration_seconds=0.20,
            maximum_duration_seconds=6.0,
            boundary_event_threshold=0.65,
            start_boundary_refinement_seconds=0.25,
            end_boundary_refinement_seconds=0.75,
            end_event_relative_threshold=0.85,
            require_boundary_events=True,
        )
        settings = TrainSettings(
            manifest=Path("manifest.jsonl"),
            output_dir=Path("output"),
            epochs=1,
            window_frames=16,
            stride_frames=8,
        )
        feature_settings = {"schema": "test"}
        visual_feature_settings = {"width": 256, "height": 32}
        spatial_contract = {"spatial_mode": "exact_bottom_slices"}
        checkpoint = _checkpoint(
            model,
            model_config=model_config,
            settings=settings,
            feature_names=["feature"],
            feature_mean=np.zeros((1,), dtype=np.float32),
            feature_std=np.ones((1,), dtype=np.float32),
            feature_settings=feature_settings,
            visual_feature_settings=visual_feature_settings,
            payload_tail_ratio=0.2,
            spatial_contract=spatial_contract,
            proposal_positive_weight=4.0,
            boundary_event_positive_weights=np.ones((2,), dtype=np.float32),
            segment_selection_config=calibrated,
            calibration_metrics={"recall_target_met": 1.0},
            epoch=1,
            metrics={},
        )
        self.assertEqual(checkpoint["segment_selection_calibration_split"], "train")

        cache = SimpleNamespace(
            feature_names=["feature"],
            meta={
                "feature_settings": feature_settings,
                "spatial_contract": spatial_contract,
            },
            visual_feature_settings=visual_feature_settings,
            timestamps=np.asarray([0.0, 1 / 30], dtype=np.float64),
            durations=np.asarray([1 / 30, 1 / 30], dtype=np.float32),
        )
        arguments = Namespace(
            feature_dir=Path("features"),
            checkpoint=Path("best.pt"),
            output_csv=Path("prediction.csv"),
            labels=None,
            batch_size=1,
            device="cpu",
            score_threshold=None,
            nms_iou=None,
            minimum_duration=None,
            maximum_duration=None,
            boundary_event_threshold=None,
            start_boundary_refinement=None,
            end_boundary_refinement=None,
            end_event_relative_threshold=None,
            require_boundary_events=None,
        )
        with (
            patch.object(cli, "FeatureCache", return_value=cache),
            patch.object(cli, "_load_model", return_value=(model, checkpoint)),
            patch.object(
                cli,
                "predict_cache",
                return_value=np.zeros((2, 5), dtype=np.float32),
            ),
            patch.object(cli, "select_segments", return_value=[]) as select,
            patch.object(cli, "write_segment_predictions"),
        ):
            result = cli._infer(arguments)

        self.assertEqual(select.call_args.kwargs["config"], calibrated)
        np.testing.assert_array_equal(select.call_args.args[1], cache.timestamps)
        self.assertEqual(result["segment_selection_config"], calibrated.to_dict())

    def test_cli_defaults_expose_exact_prepare_and_numeric_training(self):
        parser = cli.build_parser()
        self.assertEqual(parser.prog, "h264-timing")
        extract = parser.parse_args(["extract", "input.mp4", "features"])
        proxy = parser.parse_args(
            [
                "extract",
                "input.mp4",
                "features",
                "--spatial-mode",
                "payload_tail_proxy",
            ]
        )
        training = parser.parse_args(["train", "manifest.jsonl", "output"])
        prepare = parser.parse_args(
            [
                "prepare",
                "input.mp4",
                "input.srt",
                "plan.csv",
                "dataset",
                "--signal-schedule",
                "source_timing",
            ]
        )

        self.assertEqual(extract.spatial_mode, "exact_bottom_slices")
        self.assertEqual(proxy.spatial_mode, "payload_tail_proxy")
        self.assertFalse(training.use_byte_branch)
        self.assertEqual(prepare.signal_schedule, "source_timing")

        inference = parser.parse_args(
            [
                "infer",
                "features",
                "best.pt",
                "prediction.csv",
                "--boundary-event-threshold",
                "0.7",
                "--start-boundary-refinement",
                "0.3",
                "--end-boundary-refinement",
                "1.1",
                "--end-event-relative-threshold",
                "0.75",
                "--no-require-boundary-events",
            ]
        )
        self.assertEqual(inference.boundary_event_threshold, 0.7)
        self.assertEqual(inference.start_boundary_refinement, 0.3)
        self.assertEqual(inference.end_boundary_refinement, 1.1)
        self.assertEqual(inference.end_event_relative_threshold, 0.75)
        self.assertFalse(inference.require_boundary_events)

        stream_inference = parser.parse_args(["stream-infer", "stream.pt"])
        stream_training = parser.parse_args(
            ["train-stream", "manifest.jsonl", "stream-output"]
        )
        self.assertEqual(stream_inference.checkpoint, Path("stream.pt"))
        self.assertEqual(stream_training.inference_chunk_frames, 128)

    def test_segment_targets_reconstruct_exact_adjacent_segments(self):
        timestamps = np.arange(11, dtype=np.float64) / 10.0
        intervals = [
            SubtitleInterval(0.2, 0.5),
            SubtitleInterval(0.5, 0.8),
        ]

        targets = segment_targets_from_intervals(
            timestamps,
            intervals,
        )

        for candidate_indices, interval in (
            (range(2, 5), intervals[0]),
            (range(5, 8), intervals[1]),
        ):
            anchor_index = max(candidate_indices, key=lambda index: targets[index, 0])
            self.assertGreater(float(targets[anchor_index, 0]), 0.5)
            self.assertAlmostEqual(
                timestamps[anchor_index] + float(targets[anchor_index, 1]),
                interval.start_seconds,
            )
            self.assertAlmostEqual(
                timestamps[anchor_index] + float(targets[anchor_index, 2]),
                interval.end_seconds,
            )

    def test_segment_selection_removes_duplicates_and_preserves_adjacent_segments(self):
        timestamps = np.arange(5, dtype=np.float64) / 10.0
        proposals = np.asarray(
            [
                [0.90, 0.10, 1.00, 0.0, 0.0],
                [0.10, 0.11, 1.01, 0.0, 0.0],
                [0.80, 0.12, 1.02, 0.0, 0.0],
                [0.10, 1.00, 2.00, 0.0, 0.0],
                [0.85, 1.00, 2.00, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        selected = select_segments(
            proposals,
            timestamps,
            config=SegmentSelectionConfig(
                score_threshold=0.5,
                nms_iou_threshold=0.7,
                peak_radius_frames=1,
                start_boundary_refinement_seconds=0.0,
                end_boundary_refinement_seconds=0.0,
                require_boundary_events=False,
            ),
        )

        self.assertEqual(len(selected), 2)
        self.assertAlmostEqual(selected[0].start_seconds, 0.10, places=5)
        self.assertAlmostEqual(selected[1].start_seconds, 1.00, places=5)

    def test_segment_selection_refines_direct_boundaries_from_event_heads(self):
        timestamps = np.arange(8, dtype=np.float64) / 10.0
        proposals = np.asarray(
            [
                [0.1, 0.08, 0.45, 0.1, 0.1],
                [0.2, 0.08, 0.45, 0.9, 0.1],
                [0.2, 0.08, 0.45, 0.1, 0.1],
                [0.9, 0.08, 0.45, 0.1, 0.1],
                [0.2, 0.08, 0.45, 0.1, 0.8],
                [0.1, 0.08, 0.45, 0.1, 0.1],
                [0.1, 0.08, 0.45, 0.1, 0.9],
                [0.1, 0.08, 0.45, 0.1, 0.1],
            ],
            dtype=np.float32,
        )

        selected = select_segments(
            proposals,
            timestamps,
            config=SegmentSelectionConfig(
                score_threshold=0.5,
                peak_radius_frames=1,
                boundary_event_threshold=0.5,
                start_boundary_refinement_seconds=0.2,
                end_boundary_refinement_seconds=0.3,
                end_event_relative_threshold=0.8,
                boundary_event_peak_radius_frames=1,
                require_boundary_events=True,
            ),
        )

        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0].start_seconds, 0.1)
        self.assertAlmostEqual(selected[0].end_seconds, 0.4)

    def test_interval_metrics_require_both_boundaries_from_the_same_segment(self):
        target = [SubtitleInterval(0.0, 1.0), SubtitleInterval(1.0, 2.0)]
        exact = [SubtitleInterval(0.0, 1.0), SubtitleInterval(1.0, 2.0)]
        imprecise = [SubtitleInterval(0.02, 1.02), SubtitleInterval(1.0, 2.20)]

        exact_metrics = interval_metrics(
            exact,
            target,
            video_duration_seconds=2.0,
            frame_tolerance_seconds=1.0 / 30.0,
        )
        imprecise_metrics = interval_metrics(
            imprecise,
            target,
            video_duration_seconds=2.0,
            frame_tolerance_seconds=1.0 / 30.0,
        )

        self.assertEqual(exact_metrics["interval_recall_iou50"], 1.0)
        self.assertEqual(exact_metrics["segment_recall_1frame"], 1.0)
        self.assertEqual(exact_metrics["missed_segment_1frame_count"], 0.0)
        self.assertEqual(imprecise_metrics["interval_recall_iou50"], 1.0)
        self.assertEqual(imprecise_metrics["segment_recall_1frame"], 0.5)
        self.assertEqual(imprecise_metrics["missed_segment_1frame_count"], 1.0)

    def test_segment_loss_regresses_both_boundaries_and_handles_clean_windows(self):
        model = H264SubtitleSegmentModel(
            ModelConfig(
                feature_count=2,
                token_count=1,
                width=16,
                temporal_layers=1,
                dropout=0.0,
            )
        )
        features = torch.zeros((2, 8, 2), dtype=torch.float32)
        tokens = torch.zeros((2, 8, 1), dtype=torch.int64)
        targets = torch.zeros((2, 8, 3), dtype=torch.float32)
        targets[0, 3, :] = torch.tensor([1.0, 0.0, 0.8])
        boundary_event_targets = torch.zeros((2, 8, 2), dtype=torch.float32)
        boundary_event_targets[0, 3, 0] = 1.0
        boundary_event_targets[0, 6, 1] = 1.0
        mask = torch.ones((2, 8), dtype=torch.float32)
        regression_mask = (targets[..., 0] > 0.0).to(torch.float32)

        output = model(features, tokens)
        loss, components = segment_detection_loss(
            output,
            targets,
            boundary_event_targets,
            mask,
            regression_mask,
            proposal_positive_weight=4.0,
            boundary_event_positive_weights=torch.ones((2,)),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(np.isfinite(value) for value in components.values()))
        self.assertIsNotNone(model.boundary_head[-1].weight.grad)

    def test_predict_windows_cover_short_and_tail_frames(self):
        frame_count = 21
        cache = SimpleNamespace(
            features=np.zeros((frame_count, 4), dtype=np.float32),
            tokens=np.zeros((frame_count, 3), dtype=np.uint8),
            timestamps=np.arange(frame_count, dtype=np.float64) / 30.0,
        )
        model = H264SubtitleSegmentModel(
            ModelConfig(
                feature_count=4,
                token_count=3,
                width=16,
                temporal_layers=1,
                dropout=0.0,
            )
        ).eval()

        proposals = predict_cache(
            model,
            cache,
            feature_mean=np.zeros((4,), dtype=np.float32),
            feature_std=np.ones((4,), dtype=np.float32),
            window_frames=8,
            hop_frames=6,
            batch_size=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(window_starts(frame_count, 8, 6), [0, 6, 12, 13])
        self.assertEqual(proposals.shape, (frame_count, 5))
        self.assertTrue(np.isfinite(proposals).all())
        self.assertTrue(np.all((proposals[:, [0, 3, 4]] >= 0.0)))
        self.assertTrue(np.all((proposals[:, [0, 3, 4]] <= 1.0)))

    def test_streaming_model_is_invariant_to_chunk_boundaries(self):
        torch.manual_seed(7)
        model = StreamingH264SubtitleModel(
            StreamingModelConfig(
                feature_count=3,
                token_count=2,
                width=8,
                temporal_layers=3,
                dropout=0.0,
            )
        ).eval()
        features = torch.randn((1, 19, 3))
        tokens = torch.zeros((1, 19, 2), dtype=torch.int64)

        complete = model(features, tokens)
        state = None
        presence_chunks = []
        boundary_chunks = []
        for start, stop in ((0, 1), (1, 6), (6, 8), (8, 19)):
            output, state = model.forward_stream(
                features[:, start:stop], tokens[:, start:stop], state
            )
            presence_chunks.append(output.presence_logits)
            boundary_chunks.append(output.boundary_event_logits)

        torch.testing.assert_close(
            torch.cat(presence_chunks, dim=1), complete.presence_logits
        )
        torch.testing.assert_close(
            torch.cat(boundary_chunks, dim=1), complete.boundary_event_logits
        )

    def test_streaming_decoder_emits_adjacent_segments_and_flushes_tail(self):
        config = StreamingDecoderConfig(
            score_threshold=0.0,
            presence_on_threshold=0.6,
            presence_off_threshold=0.3,
            boundary_event_threshold=0.6,
            minimum_duration_seconds=0.05,
            maximum_duration_seconds=2.0,
            confirmation_samples=2,
        )
        decoder = StreamingSegmentDecoder(config)
        self.assertEqual(decoder.push(0.0, 0.1, 0.9, 0.9, 0.0), ())
        self.assertEqual(decoder.push(0.1, 0.1, 0.9, 0.0, 0.0), ())

        first = decoder.push(0.2, 0.1, 0.9, 0.9, 0.9)
        self.assertEqual(len(first), 1)
        self.assertAlmostEqual(first[0].start_seconds, 0.0)
        self.assertAlmostEqual(first[0].end_seconds, 0.2)

        self.assertEqual(decoder.push(0.3, 0.1, 0.9, 0.0, 0.8), ())
        self.assertEqual(decoder.push(0.4, 0.1, 0.9, 0.0, 0.0), ())
        second = decoder.push(0.5, 0.1, 0.1, 0.0, 0.9)
        self.assertEqual(len(second), 1)
        self.assertAlmostEqual(second[0].start_seconds, 0.2)
        self.assertAlmostEqual(second[0].end_seconds, 0.5)

        tail = StreamingSegmentDecoder(config)
        tail.push(1.0, 0.1, 0.9, 0.9, 0.0)
        tail.push(1.1, 0.1, 0.9, 0.0, 0.0)
        flushed = tail.close()
        self.assertEqual(len(flushed), 1)
        self.assertAlmostEqual(flushed[0].end_seconds, 1.2)
        self.assertEqual(tail.close(), ())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            tail.push(1.2, 0.1, 0.0, 0.0, 0.0)

    def test_streaming_detector_discards_samples_and_keeps_bounded_state(self):
        model = StreamingH264SubtitleModel(
            StreamingModelConfig(
                feature_count=2,
                token_count=1,
                width=8,
                temporal_layers=2,
                dropout=0.0,
            )
        )
        detector = StreamingSegmentDetector(
            model,
            np.zeros((2,), dtype=np.float32),
            np.ones((2,), dtype=np.float32),
            StreamingDecoderConfig(score_threshold=1.0),
        )
        detector.push_many(
            StreamSample(index / 30.0, 1 / 30.0, [0.0, 0.0])
            for index in range(10)
        )
        retained = detector.retained_state_elements
        self.assertGreater(retained, 0)
        detector.push_many(
            StreamSample(index / 30.0, 1 / 30.0, [0.0, 0.0])
            for index in range(10, 210)
        )
        self.assertEqual(detector.retained_state_elements, retained)
        detector.close()
        self.assertEqual(detector.retained_state_elements, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            detector.push(StreamSample(7.0, 1 / 30.0, [0.0, 0.0]))

    def test_streaming_checkpoint_is_separate_from_legacy_v5(self):
        config = StreamingModelConfig(
            feature_count=2,
            token_count=1,
            width=8,
            temporal_layers=1,
            dropout=0.0,
        )
        model = StreamingH264SubtitleModel(config)
        decoder_config = StreamingDecoderConfig()
        checkpoint = {
            "format": STREAM_CHECKPOINT_FORMAT,
            "version": STREAM_CHECKPOINT_VERSION,
            "feature_version": FEATURE_VERSION,
            "model_output_contract": "causal_streaming_presence_events",
            "model_config": config.to_dict(),
            "model": model.state_dict(),
            "feature_names": ["first", "second"],
            "feature_mean": torch.zeros((2,)),
            "feature_std": torch.ones((2,)),
            "streaming_decoder_config": decoder_config.to_dict(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stream.pt"
            torch.save(checkpoint, path)
            detector = StreamingSegmentDetector.from_checkpoint(path, device="cpu")
            self.assertEqual(detector.feature_names, ("first", "second"))
            with self.assertRaisesRegex(ValueError, "unsupported checkpoint"):
                cli._load_model(path, torch.device("cpu"))

    def test_stream_cli_stops_and_flushes_on_manual_close_record(self):
        pushed = []
        close_count = 0

        class FakeDetector:
            def push(self, sample):
                pushed.append(sample)
                return ()

            def close(self):
                nonlocal close_count
                close_count += 1
                return ()

        stream_input = io.StringIO(
            json.dumps(
                {
                    "timestamp_seconds": 0.0,
                    "duration_seconds": 1 / 30,
                    "features": [0.0, 0.0],
                }
            )
            + "\n"
            + json.dumps({"type": "close"})
            + "\n"
            + "not parsed after close\n"
        )
        output = io.StringIO()
        with (
            patch.object(
                cli.StreamingSegmentDetector,
                "from_checkpoint",
                return_value=FakeDetector(),
            ),
            patch.object(cli.sys, "stdin", stream_input),
            redirect_stdout(output),
        ):
            cli._stream_infer(Namespace(checkpoint=Path("stream.pt"), device="cpu"))

        self.assertEqual(len(pushed), 1)
        self.assertEqual(close_count, 1)
        self.assertEqual(json.loads(output.getvalue())["type"], "closed")

    def test_stream_targets_never_place_boundary_mass_before_the_event(self):
        timestamps = np.arange(8, dtype=np.float64) / 10.0
        interval = SubtitleInterval(0.2, 0.5)
        presence = presence_targets_from_intervals(timestamps, [interval])
        events = causal_boundary_event_targets_from_intervals(
            timestamps,
            [interval],
            sigma_seconds=0.1,
        )

        np.testing.assert_array_equal(
            presence, np.asarray([0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)
        )
        self.assertTrue(np.all(events[timestamps < interval.start_seconds, 0] == 0.0))
        self.assertTrue(np.all(events[timestamps < interval.end_seconds, 1] == 0.0))

    def test_file_hash_changes_for_same_size_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.mp4"
            path.write_bytes(b"first")
            first = file_sha256(path)
            path.write_bytes(b"other")
            second = file_sha256(path)
            self.assertNotEqual(first, second)

    def test_feature_cache_rejects_corrupted_array(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            arrays = {
                "features.npy": np.zeros((2, 1), dtype=np.float32),
                "tokens.npy": np.zeros((2, 1), dtype=np.uint8),
                "timestamps.npy": np.asarray([0.0, 1 / 30], dtype=np.float64),
                "durations.npy": np.asarray([1 / 30, 1 / 30], dtype=np.float32),
            }
            for name, values in arrays.items():
                np.save(directory / name, values)
            meta = {
                "format": FEATURE_FORMAT,
                "version": FEATURE_VERSION,
                "completed": True,
                "packet_count": 2,
                "feature_names": ["feature"],
                "feature_settings": {"token_count": 1},
                "source_sha256": "0" * 64,
                "artifact_sha256": {
                    name: file_sha256(directory / name) for name in arrays
                },
            }
            (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            FeatureCache(directory)

            features_path = directory / "features.npy"
            contents = bytearray(features_path.read_bytes())
            contents[-1] ^= 1
            features_path.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, "artifact fingerprint mismatch"):
                FeatureCache(directory)

    def test_resume_requires_exact_source_frame_range(self):
        output = {"frame_rate": "30/1", "frame_count": 30}
        source_timeline = {
            "start_frame": 0,
            "end_frame": 30,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
        }
        original = ClipPlan("clip", "train", 0.0, 1.0, 2)
        _validate_resumed_source_range(
            source_timeline, output, original, Path("clip.mp4")
        )

        changed_inside_one_frame = ClipPlan("clip", "train", 0.01, 1.0, 2)
        with self.assertRaisesRegex(ValueError, "source range changed"):
            _validate_resumed_source_range(
                source_timeline,
                output,
                changed_inside_one_frame,
                Path("clip.mp4"),
            )


if __name__ == "__main__":
    unittest.main()
