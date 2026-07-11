import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from src.model import SubtitleDetector


class CliTests(unittest.TestCase):
    def test_export_unified_subcommand_writes_model_directory(self):
        from src.cli import main

        model = nn.Conv2d(3, 2, kernel_size=3, stride=2, padding=1).eval()
        exported = torch.export.export(model, (torch.randn(1, 3, 8, 8),))

        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = Path(tmp) / "tiny.pt2"
            output_dir = Path(tmp) / "unified"
            torch.export.save(exported, pt2_path)

            main(["export-unified", str(pt2_path), str(output_dir)])

            manifest = json.loads((output_dir / "model.json").read_text())
            self.assertEqual(manifest["format"], "subfast-net.unified-model")
            self.assertTrue((output_dir / "weights.bin").is_file())

    def test_export_unified_subcommand_accepts_training_checkpoint(self):
        from src.cli import main

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

            main(["export-unified", str(checkpoint_path), str(output_dir)])

            manifest = json.loads((output_dir / "model.json").read_text())
            self.assertEqual(manifest["format"], "subfast-net.unified-model")
            self.assertEqual(manifest["source"]["checkpoint"], str(checkpoint_path))
            self.assertTrue((output_dir / "weights.bin").is_file())

    def test_export_unified_subcommand_accepts_batch_size_and_head_output(self):
        from src.cli import main

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
                "export-unified",
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
        from src.cli import main

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

                main(["export-coreml", str(checkpoint_path), str(output_dir)])

                self.assertTrue(output_path.is_dir())
        finally:
            if previous_coremltools is None:
                sys.modules.pop("coremltools", None)
            else:
                sys.modules["coremltools"] = previous_coremltools

    def test_train_roi_subcommand_dispatches_to_roi_trainer(self):
        from src import train_roi
        from src.cli import main

        calls = []
        original = train_roi.main
        train_roi.main = lambda argv: calls.append(argv)
        try:
            main(["train-roi", "--presence-epochs", "3", "--embedding-epochs", "0", "--joint-epochs", "0"])
        finally:
            train_roi.main = original

        self.assertEqual(calls, [["--presence-epochs", "3", "--embedding-epochs", "0", "--joint-epochs", "0"]])

    def test_benchmark_presence_subcommand_dispatches(self):
        from src import train_presence
        from src.cli import main

        calls = []
        original = train_presence.main_benchmark
        train_presence.main_benchmark = lambda argv: calls.append(argv)
        try:
            main(["benchmark-presence", "--device", "mps"])
        finally:
            train_presence.main_benchmark = original

        self.assertEqual(calls, [["--device", "mps"]])


if __name__ == "__main__":
    unittest.main()
