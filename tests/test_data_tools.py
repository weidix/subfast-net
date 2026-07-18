import contextlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from tools import build_samples, synthesize_samples, via_labels


class DataToolsTests(unittest.TestCase):
    def test_independent_cli_dispatches_tools(self):
        from tools.cli import main

        calls: list[tuple[str, list[str], str]] = []

        def fake_import(module_name: str) -> SimpleNamespace:
            return SimpleNamespace(
                main=lambda argv: calls.append((module_name, argv, sys.argv[0])),
            )

        with patch("tools.cli.import_module", side_effect=fake_import):
            main(["build-samples", "--help"])
            main(["labels-to-via", "--help"])

        self.assertEqual(
            calls,
            [
                ("tools.build_samples", ["--help"], "subfast-tools build-samples"),
                ("tools.via_labels", ["labels-to-via", "--help"], "subfast-tools"),
            ],
        )

    def test_independent_cli_rejects_unknown_command(self):
        from tools.cli import main

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(["unknown"])

        self.assertEqual(status, 2)
        self.assertIn("unknown command 'unknown'", stderr.getvalue())

    def test_via_round_trip_uses_non_jpeg_image_and_preserves_box(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images = root / "images"
            labels = root / "labels"
            restored = root / "restored"
            images.mkdir()
            labels.mkdir()
            Image.new("RGB", (200, 100), (12, 34, 56)).save(images / "sample.png")
            (labels / "sample.txt").write_text(
                "0 0.500000 0.500000 0.250000 0.400000\n", encoding="utf-8"
            )

            via = via_labels.labels_to_via(labels, images)
            self.assertEqual(len(via), 1)
            item = next(iter(via.values()))
            self.assertEqual(item["filename"], "sample.png")
            self.assertEqual(item["file_attributes"], {"width": 200, "height": 100})

            via_labels.via_to_labels(via, restored)
            self.assertEqual(
                (restored / "sample.txt").read_text(encoding="utf-8"),
                "0 0.500000 0.500000 0.250000 0.400000\n",
            )

    def test_via_dimensions_can_come_from_annotation_for_missing_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            labels = root / "labels"
            labels.mkdir()
            annotations = root / "annotations.jsonl"
            (labels / "sample.txt").write_text(
                "0 0.500000 0.500000 0.100000 0.200000\n", encoding="utf-8"
            )
            annotations.write_text(
                json.dumps({"image": "images/sample.webp", "image_width": 80, "image_height": 60})
                + "\n",
                encoding="utf-8",
            )

            via = via_labels.labels_to_via(labels, root / "images", annotations)
            item = next(iter(via.values()))
            self.assertEqual(item["file_attributes"], {"width": 80, "height": 60})

    def test_srt_parser_ignores_sequence_and_timestamps(self):
        subtitles = synthesize_samples.parse_srt_text(
            "1\n00:00:00,000 --> 00:00:01,000\n第一行\n\n"
            "2\n00:00:02,000 --> 00:00:03,000\nsecond line\n"
        )
        self.assertEqual(subtitles, ["第一行", "second line"])

    def test_synthesize_rejects_negative_margin_before_loading_pillow(self):
        args = Namespace(
            count=1,
            font_size_min=10,
            font_size_max=10,
            margin=-1,
            subtitle_file=Path("missing.srt"),
            image_dir=Path("missing-images"),
            output=Path("out"),
            seed=None,
            font=None,
            placement_region=None,
            boxed_images=False,
        )
        with self.assertRaisesRegex(ValueError, "margin"):
            synthesize_samples.synthesize_samples(args)

    def test_build_sample_geometry_helpers(self):
        detection = build_samples.Detection(
            polygon=[[10, 20], [30, 20], [30, 40], [10, 40]], score=0.9
        )
        self.assertEqual(build_samples.detection_bbox(detection), (10, 20, 30, 40))
        self.assertEqual(build_samples.detection_center(detection), (20.0, 30.0))
        self.assertTrue(build_samples.bbox_intersects_region((10, 20, 30, 40), (0, 0, 15, 25)))
        self.assertFalse(build_samples.bbox_intersects_region((10, 20, 30, 40), (31, 41, 50, 60)))


if __name__ == "__main__":
    unittest.main()
