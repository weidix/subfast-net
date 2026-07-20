import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from subfast_detector.model import SubtitleDetector


class CliTests(unittest.TestCase):
    def test_export_unified_subcommand_writes_model_directory(self):
        from subfast_detector.cli import main

        model = nn.Conv2d(3, 2, kernel_size=3, stride=2, padding=1).eval()
        exported = torch.export.export(model, (torch.randn(1, 3, 8, 8),))

        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = Path(tmp) / "tiny.pt2"
            output_dir = Path(tmp) / "unified"
            torch.export.save(exported, pt2_path)

            main(["export", "unified", str(pt2_path), str(output_dir)])

            manifest = json.loads((output_dir / "model.json").read_text())
            self.assertEqual(manifest["format"], "subfast-net.unified-model")
            self.assertTrue((output_dir / "weights.bin").is_file())

    def test_export_unified_subcommand_accepts_training_checkpoint(self):
        from subfast_detector.cli import main

        model = SubtitleDetector().eval()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            output_dir = Path(tmp) / "unified"
            torch.save(
                {
                    "model": model.state_dict(),
                    "settings": {"image_size": 32},
                },
                checkpoint_path,
            )

            main(["export", "unified", str(checkpoint_path), str(output_dir)])

            manifest = json.loads((output_dir / "model.json").read_text())
            self.assertEqual(manifest["format"], "subfast-net.unified-model")
            self.assertEqual(manifest["source"]["checkpoint"], str(checkpoint_path))
            self.assertTrue((output_dir / "weights.bin").is_file())

    def test_export_unified_subcommand_accepts_batch_size_and_head_output(self):
        from subfast_detector.cli import main

        model = SubtitleDetector().eval()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            output_dir = Path(tmp) / "unified"
            torch.save(
                {
                    "model": model.state_dict(),
                    "settings": {"image_size": 32},
                },
                checkpoint_path,
            )

            main([
                "export",
                "unified",
                "--batch-size",
                "4",
                "--head-output",
                str(checkpoint_path),
                str(output_dir),
            ])

            manifest = json.loads((output_dir / "model.json").read_text())
            self.assertEqual(manifest["inputs"][0]["shape"], [4, 3, 32, 32])
            self.assertEqual(manifest["outputs"][0]["name"], "conv2d_9")
            self.assertEqual(manifest["outputs"][0]["shape"], [4, 2, 8, 8])
            self.assertNotIn("aten.upsample_bilinear2d.vec", [node["op"] for node in manifest["nodes"]])

    def test_export_coreml_subcommand_writes_model_package(self):
        from subfast_detector.cli import main

        class FakeTensorType:
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape

        class FakeCoreMLModel:
            def save(self, path):
                Path(path).mkdir()

        fake_coremltools = SimpleNamespace(
            TensorType=FakeTensorType,
            convert=lambda model, *, inputs, source: FakeCoreMLModel(),
        )
        previous_coremltools = sys.modules.get("coremltools")
        sys.modules["coremltools"] = fake_coremltools

        try:
            model = SubtitleDetector().eval()
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint_path = Path(tmp) / "best.pt"
                output_dir = Path(tmp) / "coreml"
                output_path = output_dir / "model.mlpackage"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "settings": {"image_size": 32},
                    },
                    checkpoint_path,
                )

                main(["export", "coreml", str(checkpoint_path), str(output_dir)])

                self.assertTrue(output_path.is_dir())
        finally:
            if previous_coremltools is None:
                sys.modules.pop("coremltools", None)
            else:
                sys.modules["coremltools"] = previous_coremltools

    def test_export_safetensors_subcommand_writes_loadable_bundle(self):
        from safetensors import safe_open
        from safetensors.torch import load_file

        from subfast_detector.cli import main

        model = SubtitleDetector().eval()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.pt"
            output_dir = Path(tmp) / "safetensors"
            torch.save(
                {
                    "model": model.state_dict(),
                    "settings": {
                        "image_size": 96,
                        "stride": 32,
                        "region_threshold": 0.6,
                        "kernel_threshold": 0.4,
                    },
                },
                checkpoint_path,
            )

            main(["export", "safetensors", str(checkpoint_path), str(output_dir)])

            weights_path = output_dir / "model.safetensors"
            config = json.loads((output_dir / "config.json").read_text())
            loaded = load_file(weights_path)
            self.assertEqual(set(loaded), set(model.state_dict()))
            torch.testing.assert_close(loaded["stem.0.weight"], model.state_dict()["stem.0.weight"])
            self.assertEqual(config["format"], "subfast-net.safetensors")
            self.assertEqual(config["model_type"], "subtitle_detector")
            self.assertEqual(config["model"]["kwargs"]["width"], 32)
            self.assertEqual(config["postprocessing"]["region_threshold"], 0.6)
            with safe_open(weights_path, framework="pt", device="cpu") as archive:
                self.assertEqual(archive.metadata()["model_type"], "subtitle_detector")

    def test_training_validation_and_benchmark_commands_dispatch(self):
        from subfast_detector.cli import main

        calls: list[tuple[str, str, list[str]]] = []

        def fake_import(module_name: str) -> SimpleNamespace:
            def record(function_name: str):
                return lambda argv: calls.append((module_name, function_name, argv))

            return SimpleNamespace(
                main=record("main"),
                main_validate=record("main_validate"),
                main_benchmark=record("main_benchmark"),
            )

        cases = [
            (
                ["train", "detector", "--epochs", "2"],
                ("subfast_detector.train", "main", ["--epochs", "2"]),
            ),
            (
                ["train", "presence", "--epochs", "3"],
                ("subfast_roi_presence.train", "main", ["--epochs", "3"]),
            ),
            (
                ["train", "matcher", "--epochs", "4"],
                ("subfast_roi_matcher.train", "main", ["--epochs", "4"]),
            ),
            (
                ["validate", "matcher", "--checkpoint", "best.pt"],
                ("subfast_roi_matcher.train", "main_validate", ["--checkpoint", "best.pt"]),
            ),
            (
                ["benchmark", "presence", "--device", "mps"],
                ("subfast_roi_presence.train", "main_benchmark", ["--device", "mps"]),
            ),
        ]

        with patch("subfast_detector.cli.import_module", side_effect=fake_import):
            for command, expected in cases:
                with self.subTest(command=command):
                    main(command)
                    self.assertEqual(calls[-1], expected)

    def test_tools_are_not_routed_through_model_cli(self):
        from subfast_detector.cli import main

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(["data", "build-samples"])

        self.assertEqual(status, 2)
        self.assertIn("unknown command 'data'", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
