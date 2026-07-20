import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from subfast_detector.model import SubtitleDetector
from subfast_export.unified import export_pt2_to_unified_model


class TinyExportModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.conv(x))


class UnsupportedExportModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class ExtendedOperatorExportModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x[:, 0:1], x[:, 1:2])


class ConvBatchNormExportModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.bn(self.conv(x)))


class UnifiedModelExportTests(unittest.TestCase):
    def test_exported_program_writes_graph_metadata_and_weight_blob(self):
        model = TinyExportModel().eval()
        exported = torch.export.export(model, (torch.randn(1, 3, 8, 8),))

        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = Path(tmp) / "tiny.pt2"
            output_dir = Path(tmp) / "unified"
            torch.export.save(exported, pt2_path)

            manifest_path = export_pt2_to_unified_model(pt2_path, output_dir)

            self.assertEqual(manifest_path, output_dir / "model.json")
            manifest = json.loads(manifest_path.read_text())
            weight_blob = output_dir / "weights.bin"
            self.assertTrue(weight_blob.exists())

            self.assertEqual(manifest["format"], "subfast-net.unified-model")
            self.assertEqual(manifest["format_version"], 1)
            self.assertEqual(manifest["weights"]["file"], "weights.bin")
            self.assertEqual(manifest["inputs"], [{"name": "x", "shape": [1, 3, 8, 8], "dtype": "float32", "layout": "dense_row_major"}])
            self.assertEqual(manifest["outputs"], [{"name": "silu", "shape": [1, 2, 4, 4], "dtype": "float32", "layout": "dense_row_major"}])

            conv_node = next(node for node in manifest["nodes"] if node["name"] == "conv2d")
            self.assertEqual(conv_node["op"], "aten.conv2d.default")
            self.assertEqual(conv_node["inputs"][:3], ["x", "p_conv_weight", "p_conv_bias"])
            self.assertEqual(conv_node["attrs"]["stride"], [2, 2])
            self.assertEqual(conv_node["attrs"]["padding"], [1, 1])
            self.assertEqual(conv_node["outputs"], [{"name": "conv2d", "shape": [1, 2, 4, 4], "dtype": "float32", "layout": "dense_row_major"}])

            silu_node = next(node for node in manifest["nodes"] if node["name"] == "silu")
            self.assertEqual(silu_node["op"], "aten.silu.default")
            self.assertEqual(silu_node["inputs"], ["conv2d"])

            tensors = manifest["tensors"]
            self.assertEqual(tensors["p_conv_weight"]["kind"], "parameter")
            self.assertEqual(tensors["p_conv_weight"]["target"], "conv.weight")
            self.assertEqual(tensors["p_conv_weight"]["shape"], [2, 3, 3, 3])
            self.assertEqual(tensors["p_conv_weight"]["dtype"], "float32")
            self.assertEqual(tensors["p_conv_weight"]["layout"], "dense_row_major")
            self.assertEqual(tensors["p_conv_weight"]["data"]["file"], "weights.bin")
            self.assertEqual(tensors["p_conv_weight"]["data"]["offset"], 0)
            self.assertEqual(tensors["p_conv_weight"]["data"]["byte_length"], model.conv.weight.numel() * 4)

            bias_meta = tensors["p_conv_bias"]["data"]
            self.assertEqual(bias_meta["offset"], tensors["p_conv_weight"]["data"]["byte_length"])
            self.assertEqual(bias_meta["byte_length"], model.conv.bias.numel() * 4)
            self.assertEqual(weight_blob.stat().st_size, bias_meta["offset"] + bias_meta["byte_length"])

    def test_export_cli_writes_unified_model_directory(self):
        from subfast_export.unified import main

        model = TinyExportModel().eval()
        exported = torch.export.export(model, (torch.randn(1, 3, 8, 8),))

        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = Path(tmp) / "tiny.pt2"
            output_dir = Path(tmp) / "cli-output"
            torch.export.save(exported, pt2_path)

            main([str(pt2_path), str(output_dir)])

            manifest = json.loads((output_dir / "model.json").read_text())
            self.assertEqual(manifest["format"], "subfast-net.unified-model")
            self.assertTrue((output_dir / "weights.bin").is_file())

    def test_extended_operators_use_format_version_two(self):
        model = ExtendedOperatorExportModel().eval()
        exported = torch.export.export(model, (torch.randn(1, 3, 8, 8),))

        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = Path(tmp) / "extended.pt2"
            output_dir = Path(tmp) / "unified"
            torch.export.save(exported, pt2_path)

            manifest_path = export_pt2_to_unified_model(pt2_path, output_dir)
            manifest = json.loads(manifest_path.read_text())

            self.assertEqual(manifest["format_version"], 2)
            self.assertIn("aten.maximum.default", {node["op"] for node in manifest["nodes"]})

    def test_export_rejects_unsupported_operator_with_node_name(self):
        exported = torch.export.export(UnsupportedExportModel().eval(), (torch.randn(1, 3, 8, 8),))

        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = Path(tmp) / "unsupported.pt2"
            output_dir = Path(tmp) / "unified"
            torch.export.save(exported, pt2_path)

            with self.assertRaisesRegex(RuntimeError, "unsupported operator.*relu"):
                export_pt2_to_unified_model(pt2_path, output_dir)

    def test_export_folds_conv_batch_norm_for_inference_graph(self):
        model = ConvBatchNormExportModel().eval()
        with torch.no_grad():
            model.conv.weight.copy_(torch.arange(model.conv.weight.numel(), dtype=torch.float32).reshape_as(model.conv.weight) / 100.0)
            model.bn.weight.copy_(torch.tensor([1.25, -0.75]))
            model.bn.bias.copy_(torch.tensor([0.5, -0.25]))
            model.bn.running_mean.copy_(torch.tensor([0.2, -0.4]))
            model.bn.running_var.copy_(torch.tensor([0.5, 2.0]))
        exported = torch.export.export(model, (torch.randn(1, 3, 8, 8),))

        with tempfile.TemporaryDirectory() as tmp:
            pt2_path = Path(tmp) / "conv-bn.pt2"
            output_dir = Path(tmp) / "unified"
            torch.export.save(exported, pt2_path)

            manifest_path = export_pt2_to_unified_model(pt2_path, output_dir)
            manifest = json.loads(manifest_path.read_text())

            self.assertNotIn("aten.batch_norm.default", [node["op"] for node in manifest["nodes"]])
            conv_node = next(node for node in manifest["nodes"] if node["op"] == "aten.conv2d.default")
            self.assertEqual(conv_node["outputs"][0]["name"], "batch_norm")
            self.assertEqual(conv_node["inputs"][2], "conv2d_batch_norm_fused_bias")

            scale = model.bn.weight / torch.sqrt(model.bn.running_var + model.bn.eps)
            expected_weight = model.conv.weight * scale.reshape(-1, 1, 1, 1)
            expected_bias = model.bn.bias + (torch.zeros_like(model.bn.running_mean) - model.bn.running_mean) * scale

            tensors = manifest["tensors"]
            blob = (output_dir / "weights.bin").read_bytes()
            weight_meta = tensors["conv2d_batch_norm_fused_weight"]["data"]
            bias_meta = tensors["conv2d_batch_norm_fused_bias"]["data"]
            actual_weight = torch.frombuffer(
                bytearray(blob[weight_meta["offset"] : weight_meta["offset"] + weight_meta["byte_length"]]),
                dtype=torch.float32,
            ).reshape_as(expected_weight)
            actual_bias = torch.frombuffer(
                bytearray(blob[bias_meta["offset"] : bias_meta["offset"] + bias_meta["byte_length"]]),
                dtype=torch.float32,
            )
            torch.testing.assert_close(actual_weight, expected_weight)
            torch.testing.assert_close(actual_bias, expected_bias)

    def test_export_checkpoint_to_coreml_saves_mlpackage(self):
        from subfast_export.unified import export_model_to_coreml_model

        calls = {}

        class FakeTensorType:
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape

        class FakeCoreMLModel:
            def save(self, path):
                calls["saved_path"] = Path(path)
                Path(path).mkdir()

        def fake_convert(model, *, inputs, source):
            calls["model"] = model
            calls["inputs"] = inputs
            calls["source"] = source
            return FakeCoreMLModel()

        fake_coremltools = SimpleNamespace(TensorType=FakeTensorType, convert=fake_convert)
        previous_coremltools = sys.modules.get("coremltools")
        sys.modules["coremltools"] = fake_coremltools

        try:
            model = SubtitleDetector().eval()
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint_path = Path(tmp) / "best.pt"
                output_path = Path(tmp) / "detector.mlpackage"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "settings": {"image_size": 32},
                    },
                    checkpoint_path,
                )

                result = export_model_to_coreml_model(checkpoint_path, output_path)

                self.assertEqual(result, output_path)
                self.assertEqual(calls["saved_path"], output_path)
                self.assertEqual(calls["source"], "pytorch")
                self.assertEqual(calls["inputs"][0].name, "x")
                self.assertEqual(calls["inputs"][0].shape, (1, 3, 32, 32))
        finally:
            if previous_coremltools is None:
                sys.modules.pop("coremltools", None)
            else:
                sys.modules["coremltools"] = previous_coremltools


if __name__ == "__main__":
    unittest.main()
