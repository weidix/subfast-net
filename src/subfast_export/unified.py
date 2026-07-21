from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.fx import Node

from subfast_detector.model import SubtitleDetector
from subfast_frame_presence.model import FramePresenceModel
from subfast_roi_matcher.model import RoiPairInference, RoiPairMatcher, fuse_pair_matcher_for_inference
from subfast_roi_presence.model import RoiPresenceModel

from .checkpoint import checkpoint_model_type


FORMAT_NAME = "subfast-net.unified-model"
FORMAT_VERSION = 1
EXTENDED_FORMAT_VERSION = 2
FRAME_PRESENCE_FORMAT_VERSION = 3
WEIGHTS_FILE = "weights.bin"
COREML_OUTPUT_SUFFIXES = {".mlmodel", ".mlpackage"}
FORMAT_V2_OPERATORS = {"aten.maximum.default", "aten.slice.Tensor"}
FORMAT_V3_OPERATORS = {
    "aten.adaptive_avg_pool2d.default",
    "aten.detach_.default",
    "aten.flatten.using_ints",
    "aten.lift_fresh_copy.default",
    "aten.max_pool2d.default",
    "aten.minimum.default",
    "aten.pad.default",
    "aten.rsub.Scalar",
    "aten.unsqueeze.default",
    "aten.view.default",
    "aten.where.self",
}
SUPPORTED_OPERATORS = {
    "aten.abs.default",
    "aten.add.Tensor",
    "aten.adaptive_avg_pool2d.default",
    "aten.amax.default",
    "aten.avg_pool2d.default",
    "aten.batch_norm.default",
    "aten.bitwise_not.default",
    "aten.cat.default",
    "aten.clamp_min.default",
    "aten.conv2d.default",
    "aten.div.Tensor",
    "aten.detach_.default",
    "aten.flatten.using_ints",
    "aten.gt.Scalar",
    "aten.linear.default",
    "aten.lift_fresh_copy.default",
    "aten.masked_fill.Scalar",
    "aten.maximum.default",
    "aten.max_pool2d.default",
    "aten.mean.dim",
    "aten.mul.Tensor",
    "aten.minimum.default",
    "aten.pad.default",
    "aten.rsub.Scalar",
    "aten.sigmoid.default",
    "aten.silu.default",
    "aten.silu_.default",
    "aten.slice.Tensor",
    "aten.softplus.default",
    "aten.squeeze.dim",
    "aten.sub.Tensor",
    "aten.to.dtype",
    "aten.unsqueeze.default",
    "aten.upsample_bilinear2d.vec",
    "aten.view.default",
    "aten.where.self",
}


def dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
        torch.bool: "bool",
    }
    return names.get(dtype, str(dtype).removeprefix("torch."))


def tensor_meta_from_node(node: Node) -> dict[str, Any] | None:
    meta = node.meta.get("tensor_meta")
    if meta is not None:
        return {
            "shape": [int(dim) for dim in meta.shape],
            "dtype": dtype_name(meta.dtype),
            "layout": "dense_row_major",
        }
    value = node.meta.get("val")
    if isinstance(value, torch.Tensor):
        return tensor_meta_from_tensor(value)
    return None


def tensor_meta_from_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": dtype_name(tensor.dtype),
        "layout": "dense_row_major",
    }


def op_name(target: Any) -> str:
    return str(target).removeprefix("OpOverload(").removesuffix(")")


def is_fx_node(value: Any) -> bool:
    return isinstance(value, Node)


def serialize_value(value: Any) -> Any:
    if isinstance(value, Node):
        return {"tensor": value.name}
    if isinstance(value, torch.dtype):
        return dtype_name(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": [int(dim) for dim in value.shape],
            "dtype": dtype_name(value.dtype),
            "values": value.detach().cpu().reshape(-1).tolist(),
        }
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    return value


def collect_node_inputs(value: Any) -> list[str]:
    if isinstance(value, Node):
        return [value.name]
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(collect_node_inputs(item))
        return result
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(collect_node_inputs(item))
        return result
    return []


def argument_attr_names(operator: str) -> dict[int, str]:
    if operator == "aten.conv2d.default":
        return {
            2: "bias",
            3: "stride",
            4: "padding",
            5: "dilation",
            6: "groups",
        }
    if operator == "aten.batch_norm.default":
        return {
            5: "training",
            6: "momentum",
            7: "eps",
            8: "cudnn_enabled",
        }
    if operator == "aten.upsample_bilinear2d.vec":
        return {
            1: "output_size",
            2: "align_corners",
            3: "scale_factors",
        }
    return {}


def collect_node_attrs(node: Node, operator: str) -> dict[str, Any]:
    named_args = argument_attr_names(operator)
    attrs: dict[str, Any] = {}
    for index, arg in enumerate(node.args):
        if is_fx_node(arg):
            continue
        name = named_args.get(index, f"arg{index}")
        attrs[name] = serialize_value(arg)
    for key, value in node.kwargs.items():
        if is_fx_node(value):
            continue
        attrs[str(key)] = serialize_value(value)
    return attrs


def graph_signature_maps(exported_program: torch.export.ExportedProgram) -> tuple[dict[str, tuple[str, str | None]], list[str], list[str]]:
    input_kinds: dict[str, tuple[str, str | None]] = {}
    user_inputs: list[str] = []
    user_outputs: list[str] = []

    for spec in exported_program.graph_signature.input_specs:
        name = spec.arg.name
        kind = spec.kind.name.lower()
        input_kinds[name] = (kind, spec.target)
        if kind == "user_input":
            user_inputs.append(name)

    for spec in exported_program.graph_signature.output_specs:
        if spec.kind.name.lower() == "user_output":
            user_outputs.append(spec.arg.name)

    return input_kinds, user_inputs, user_outputs


def write_weight_tensor(weights_file, tensor: torch.Tensor) -> tuple[int, int]:
    contiguous = tensor.detach().cpu().contiguous()
    offset = weights_file.tell()
    data = contiguous.numpy().tobytes(order="C")
    weights_file.write(data)
    return offset, len(data)


def is_operator(node: Node, operator: str) -> bool:
    return node.op == "call_function" and op_name(node.target) == operator


def node_bool_arg(node: Node, index: int, name: str, default: bool) -> bool:
    if name in node.kwargs:
        return bool(node.kwargs[name])
    if len(node.args) > index:
        return bool(node.args[index])
    return default


def foldable_conv_batch_norm_pairs(nodes: list[Node]) -> dict[str, Node]:
    pairs: dict[str, Node] = {}
    for node in nodes:
        if not is_operator(node, "aten.conv2d.default"):
            continue
        users = list(node.users)
        if len(users) != 1:
            continue
        batch_norm = users[0]
        if not is_operator(batch_norm, "aten.batch_norm.default"):
            continue
        if not batch_norm.args or batch_norm.args[0] is not node:
            continue
        if node_bool_arg(batch_norm, 5, "training", False):
            continue
        pairs[node.name] = batch_norm
    return pairs


def placeholder_target(
    value: Any,
    input_kinds: dict[str, tuple[str, str | None]],
    expected_kind: str,
) -> str:
    if not isinstance(value, Node):
        raise RuntimeError(f"expected {expected_kind} placeholder, got {value!r}")
    kind, target = input_kinds.get(value.name, ("placeholder", None))
    if kind != expected_kind or target is None:
        raise RuntimeError(f"expected {expected_kind} placeholder for {value.name}, got kind={kind} target={target}")
    return target


def optional_parameter_tensor(
    value: Any,
    state_dict: dict[str, torch.Tensor],
    input_kinds: dict[str, tuple[str, str | None]],
    like: torch.Tensor,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((like.shape[0],), dtype=like.dtype, device=like.device)
    target = placeholder_target(value, input_kinds, "parameter")
    return state_dict[target]


def fold_conv_batch_norm_tensors(
    conv: Node,
    batch_norm: Node,
    state_dict: dict[str, torch.Tensor],
    input_kinds: dict[str, tuple[str, str | None]],
) -> tuple[torch.Tensor, torch.Tensor]:
    conv_weight = state_dict[placeholder_target(conv.args[1], input_kinds, "parameter")]
    conv_bias = optional_parameter_tensor(
        conv.args[2] if len(conv.args) > 2 else None,
        state_dict,
        input_kinds,
        conv_weight,
    )
    bn_weight = state_dict[placeholder_target(batch_norm.args[1], input_kinds, "parameter")]
    bn_bias = state_dict[placeholder_target(batch_norm.args[2], input_kinds, "parameter")]
    running_mean = state_dict[placeholder_target(batch_norm.args[3], input_kinds, "buffer")]
    running_var = state_dict[placeholder_target(batch_norm.args[4], input_kinds, "buffer")]
    eps = float(batch_norm.args[7] if len(batch_norm.args) > 7 else batch_norm.kwargs.get("eps", 1e-5))

    scale = bn_weight / torch.sqrt(running_var + eps)
    fused_weight = conv_weight * scale.reshape((-1,) + (1,) * (conv_weight.ndim - 1))
    fused_bias = bn_bias + (conv_bias - running_mean) * scale
    return fused_weight.contiguous(), fused_bias.contiguous()


def folded_placeholder_names(
    pairs: dict[str, Node],
    input_kinds: dict[str, tuple[str, str | None]],
) -> set[str]:
    names: set[str] = set()
    for conv_name, batch_norm in pairs.items():
        conv = batch_norm.args[0]
        for value in [conv.args[1], conv.args[2] if len(conv.args) > 2 else None, *batch_norm.args[1:5]]:
            if isinstance(value, Node):
                kind, _target = input_kinds.get(value.name, ("placeholder", None))
                if kind in {"parameter", "buffer"}:
                    names.add(value.name)
        names.add(conv_name)
        names.add(batch_norm.name)
    return names


def write_tensor_record(
    tensors: dict[str, dict[str, Any]],
    weights_file,
    name: str,
    kind: str,
    tensor: torch.Tensor,
) -> None:
    offset, byte_length = write_weight_tensor(weights_file, tensor)
    tensors[name] = {
        "name": name,
        "kind": kind,
        **tensor_meta_from_tensor(tensor),
        "data": {
            "file": WEIGHTS_FILE,
            "offset": offset,
            "byte_length": byte_length,
        },
    }


def load_training_checkpoint_model(checkpoint_path: Path) -> tuple[SubtitleDetector, int]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError(f"invalid training checkpoint: {checkpoint_path}")

    settings = checkpoint.get("settings") or {}
    image_size = int(settings.get("image_size", 256)) if isinstance(settings, dict) else 256

    model = SubtitleDetector().eval()
    model.load_state_dict(checkpoint["model"])
    return model, image_size


def checkpoint_resize_roi(checkpoint: dict[str, Any], checkpoint_path: Path) -> tuple[int, int]:
    settings = checkpoint.get("settings") or {}
    resize_roi = checkpoint.get("resize_roi")
    if resize_roi is None and isinstance(settings, dict):
        resize_roi = settings.get("resize_roi")
    if not isinstance(resize_roi, (list, tuple)) or len(resize_roi) != 2:
        raise RuntimeError(f"checkpoint does not define resize_roi: {checkpoint_path}")
    width, height = (int(value) for value in resize_roi)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid resize_roi in checkpoint {checkpoint_path}: {resize_roi}")
    return width, height


def load_checkpoint_export_model(
    checkpoint_path: Path,
    batch_size: int,
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"invalid training checkpoint: {checkpoint_path}")

    model_type = checkpoint_model_type(checkpoint)
    if model_type == "frame_presence":
        model_config = checkpoint.get("model")
        preprocessing = checkpoint.get("preprocessing")
        state_dict = checkpoint.get("model_state")
        if not isinstance(model_config, Mapping):
            raise RuntimeError(
                f"frame_presence checkpoint does not define model config: {checkpoint_path}"
            )
        if not isinstance(preprocessing, Mapping):
            raise RuntimeError(
                f"frame_presence checkpoint does not define preprocessing: {checkpoint_path}"
            )
        if not isinstance(state_dict, Mapping):
            raise RuntimeError(
                f"frame_presence checkpoint does not contain model_state: {checkpoint_path}"
            )
        model = FramePresenceModel(
            width=int(model_config.get("width", 16)),
            kernel_size=int(model_config.get("kernel_size", 5)),
        ).eval()
        architecture_version = int(checkpoint.get("architecture_version", 1))
        if architecture_version != model.architecture_version:
            raise RuntimeError(
                "unsupported frame_presence "
                f"architecture_version={architecture_version}; "
                f"runtime=v{model.architecture_version}"
            )
        model.load_state_dict(state_dict)
        input_width = int(preprocessing.get("input_width", 256))
        input_height = int(preprocessing.get("input_height", 144))
        focus_width = int(preprocessing.get("focus_width", 256))
        focus_height = int(preprocessing.get("focus_height", 32))
        return model, (
            torch.randn(batch_size, 3, input_height, input_width),
            torch.randn(batch_size, 3, focus_height, focus_width),
            torch.zeros(batch_size, dtype=torch.float32),
        )

    if model_type == "roi_presence":
        width, height = checkpoint_resize_roi(checkpoint, checkpoint_path)
        settings = checkpoint.get("settings") or {}
        model = RoiPresenceModel(
            width=int(checkpoint.get("width", settings.get("width", 16))),
            evidence_kernel_size=int(
                checkpoint.get("evidence_kernel_size", settings.get("evidence_kernel_size", 5))
            ),
        ).eval()
        version = int(checkpoint.get("architecture_version", 1))
        if version != model.architecture_version:
            raise RuntimeError(
                f"unsupported roi_presence architecture_version={version}; runtime=v{model.architecture_version}"
            )
        model.load_state_dict(checkpoint["model"])
        return model, (torch.randn(batch_size, 3, height, width),)

    if model_type == "roi_pair_matcher":
        width, height = checkpoint_resize_roi(checkpoint, checkpoint_path)
        pair_model = RoiPairMatcher().eval()
        architecture_version = int(checkpoint.get("architecture_version", 1))
        if architecture_version != pair_model.architecture_version:
            raise RuntimeError(
                f"unsupported roi_pair_matcher architecture_version={architecture_version}; "
                f"runtime=v{pair_model.architecture_version}"
            )
        pooling_version = int(checkpoint.get("pooling_version", 1))
        if pooling_version != pair_model.pooling_version:
            raise RuntimeError(
                f"unsupported roi_pair_matcher pooling_version={pooling_version}; runtime=v{pair_model.pooling_version}"
            )
        pair_model.load_state_dict(checkpoint["model"])
        model = RoiPairInference(fuse_pair_matcher_for_inference(pair_model)).eval()
        shape = (batch_size, 3, height, width)
        return model, (torch.randn(shape), torch.randn(shape))

    model, image_size = load_training_checkpoint_model(checkpoint_path)
    return model, (torch.randn(batch_size, 3, image_size, image_size),)


def export_pt2_to_unified_model(pt2_path: Path, output_dir: Path) -> Path:
    pt2_path = Path(pt2_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_program = torch.export.load(pt2_path)
    input_kinds, user_input_names, user_output_names = graph_signature_maps(exported_program)
    state_dict = exported_program.state_dict
    constants = exported_program.constants
    graph_nodes = list(exported_program.graph.nodes)
    conv_batch_norm_pairs = foldable_conv_batch_norm_pairs(graph_nodes)
    folded_names = folded_placeholder_names(conv_batch_norm_pairs, input_kinds)

    tensors: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []

    weights_path = output_dir / WEIGHTS_FILE
    with weights_path.open("wb") as weights_file:
        for node in graph_nodes:
            if node.op == "placeholder":
                meta = tensor_meta_from_node(node)
                kind, target = input_kinds.get(node.name, ("placeholder", None))
                stored_kinds = {"parameter", "buffer", "constant_tensor"}
                if node.name in folded_names or (kind in stored_kinds and not node.users):
                    continue
                tensor_record: dict[str, Any] = {
                    "name": node.name,
                    "kind": kind,
                    "target": target,
                }
                if meta is not None:
                    tensor_record.update(meta)
                if kind in stored_kinds:
                    if target is None:
                        raise RuntimeError(
                            f"stored placeholder has no target: {node.name} kind={kind}"
                        )
                    state_tensor = state_dict.get(target)
                    if state_tensor is None:
                        state_tensor = constants.get(target)
                    if not isinstance(state_tensor, torch.Tensor):
                        raise RuntimeError(
                            f"missing state tensor for placeholder {node.name}: "
                            f"kind={kind} target={target}"
                        )
                    tensor_record.update(tensor_meta_from_tensor(state_tensor))
                    offset, byte_length = write_weight_tensor(weights_file, state_tensor)
                    tensor_record["data"] = {
                        "file": WEIGHTS_FILE,
                        "offset": offset,
                        "byte_length": byte_length,
                    }
                tensors[node.name] = tensor_record
                if node.name in user_input_names:
                    if meta is None:
                        raise RuntimeError(f"user input has no tensor metadata: {node.name}")
                    inputs.append({"name": node.name, **meta})
                continue

            if node.op == "call_function":
                if op_name(node.target) == "aten._assert_tensor_metadata.default":
                    continue
                if node.name in folded_names and node.name not in conv_batch_norm_pairs:
                    continue
                operator = op_name(node.target)
                if operator not in SUPPORTED_OPERATORS:
                    raise RuntimeError(
                        f"unsupported operator for unified runtime export: node={node.name} op={operator}"
                    )
                meta = tensor_meta_from_node(node)
                if meta is None:
                    raise RuntimeError(f"call_function node has no tensor metadata: {node.name}")

                output_name = node.name
                inputs_for_node = collect_node_inputs(node.args) + collect_node_inputs(node.kwargs)
                if node.name in conv_batch_norm_pairs:
                    batch_norm = conv_batch_norm_pairs[node.name]
                    bn_meta = tensor_meta_from_node(batch_norm)
                    if bn_meta is None:
                        raise RuntimeError(f"batch_norm node has no tensor metadata: {batch_norm.name}")
                    weight_name = f"{node.name}_{batch_norm.name}_fused_weight"
                    bias_name = f"{node.name}_{batch_norm.name}_fused_bias"
                    fused_weight, fused_bias = fold_conv_batch_norm_tensors(
                        node,
                        batch_norm,
                        state_dict,
                        input_kinds,
                    )
                    write_tensor_record(tensors, weights_file, weight_name, "parameter", fused_weight)
                    write_tensor_record(tensors, weights_file, bias_name, "parameter", fused_bias)
                    output_name = batch_norm.name
                    meta = bn_meta
                    inputs_for_node = [node.args[0].name, weight_name, bias_name]
                tensors[node.name] = {
                    "name": output_name,
                    "kind": "intermediate",
                    **meta,
                }
                if output_name != node.name:
                    tensors[output_name] = tensors.pop(node.name)
                nodes.append(
                    {
                        "name": node.name,
                        "op": operator,
                        "inputs": inputs_for_node,
                        "attrs": collect_node_attrs(node, operator),
                        "outputs": [{"name": output_name, **meta}],
                    }
                )
                continue

            if node.op == "output":
                continue

            raise RuntimeError(f"unsupported FX node kind: {node.op} name={node.name} target={node.target}")

    for output_name in user_output_names:
        if output_name not in tensors:
            raise RuntimeError(f"user output tensor not found in graph: {output_name}")
        output_tensor = tensors[output_name]
        outputs.append(
            {
                "name": output_name,
                "shape": output_tensor["shape"],
                "dtype": output_tensor["dtype"],
                "layout": output_tensor["layout"],
            }
        )

    operators = {node["op"] for node in nodes}
    if operators & FORMAT_V3_OPERATORS:
        format_version = FRAME_PRESENCE_FORMAT_VERSION
    elif operators & FORMAT_V2_OPERATORS:
        format_version = EXTENDED_FORMAT_VERSION
    else:
        format_version = FORMAT_VERSION
    manifest = {
        "format": FORMAT_NAME,
        "format_version": format_version,
        "producer": {
            "name": "subfast_export.unified",
            "torch_version": torch.__version__,
        },
        "graph": {"node_count": len(nodes)},
        "source": {"pt2": str(pt2_path)},
        "weights": {"file": WEIGHTS_FILE, "byte_order": "little", "layout": "dense_row_major"},
        "inputs": inputs,
        "outputs": outputs,
        "tensors": tensors,
        "nodes": nodes,
    }

    manifest_path = output_dir / "model.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def export_checkpoint_to_pt2(checkpoint_path: Path, pt2_path: Path, batch_size: int = 1) -> Path:
    checkpoint_path = Path(checkpoint_path)
    pt2_path = Path(pt2_path)
    if batch_size <= 0:
        raise RuntimeError(f"batch_size must be positive, got {batch_size}")
    model, example_inputs = load_checkpoint_export_model(checkpoint_path, batch_size)
    exported = torch.export.export(model, example_inputs)
    pt2_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, pt2_path)
    return pt2_path


def load_coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("coremltools is required for CoreML export. Install it with `uv sync`.") from exc
    return ct


def coreml_output_path(path: Path) -> Path:
    path = Path(path)
    if path.suffix in COREML_OUTPUT_SUFFIXES:
        return path
    return path / "model.mlpackage"


def user_input_shapes(
    exported_program: torch.export.ExportedProgram,
) -> list[tuple[str, tuple[int, ...]]]:
    _input_kinds, user_input_names, _user_output_names = graph_signature_maps(exported_program)
    node_metadata = {
        node.name: tensor_meta_from_node(node) for node in exported_program.graph.nodes
    }
    result: list[tuple[str, tuple[int, ...]]] = []
    for input_name in user_input_names:
        meta = node_metadata.get(input_name)
        if meta is None:
            raise RuntimeError(f"user input has no tensor metadata: {input_name}")
        result.append((input_name, tuple(meta["shape"])))
    if not result:
        raise RuntimeError("CoreML export requires at least one user input")
    return result


class CoreMLSubtitleDetector(torch.nn.Module):
    def __init__(self, model: SubtitleDetector, image_size: int) -> None:
        super().__init__()
        self.model = model
        self.output_size = (image_size, image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model.head(self.model.stem(x))
        return torch.nn.functional.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)


def export_model_to_coreml_model(
    model_path: Path,
    output_path: Path,
    batch_size: int = 1,
) -> Path:
    model_path = Path(model_path)
    output_path = coreml_output_path(output_path)
    if batch_size <= 0:
        raise RuntimeError(f"batch_size must be positive, got {batch_size}")

    ct = load_coremltools()
    converted_outputs = None
    if model_path.suffix == ".pt2":
        if batch_size != 1:
            raise RuntimeError("--batch-size can only be used when exporting CoreML from a training checkpoint")
        exported_program = torch.export.load(model_path)
        exported_inputs = user_input_shapes(exported_program)
        model_for_conversion = exported_program.run_decompositions({})
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model, example_inputs = load_checkpoint_export_model(model_path, batch_size)
        model_type = (
            checkpoint_model_type(checkpoint) if isinstance(checkpoint, Mapping) else None
        )
        if model_type is None:
            image_size = int(example_inputs[0].shape[-1])
            model = CoreMLSubtitleDetector(model, image_size).eval()
            converted_inputs = [ct.TensorType(name="x", shape=tuple(example_inputs[0].shape))]
        elif model_type == "roi_presence":
            converted_inputs = [ct.TensorType(name="images", shape=tuple(example_inputs[0].shape))]
        elif model_type == "roi_pair_matcher":
            converted_inputs = [
                ct.TensorType(name="left", shape=tuple(example_inputs[0].shape)),
                ct.TensorType(name="right", shape=tuple(example_inputs[1].shape)),
            ]
        elif model_type == "frame_presence":
            converted_inputs = [
                ct.TensorType(name="images", shape=tuple(example_inputs[0].shape)),
                ct.TensorType(name="focus", shape=tuple(example_inputs[1].shape)),
                ct.TensorType(name="focus_mode", shape=tuple(example_inputs[2].shape)),
            ]
            converted_outputs = [
                ct.TensorType(name="presence_logits"),
                ct.TensorType(name="region_logits"),
            ]
        else:
            raise RuntimeError(f"unsupported checkpoint model_type for CoreML export: {model_type}")
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.jit")
            model_for_conversion = torch.jit.trace(model, example_inputs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    conversion_options: dict[str, Any] = {
        "inputs": (
            [ct.TensorType(name=name, shape=shape) for name, shape in exported_inputs]
            if model_path.suffix == ".pt2"
            else converted_inputs
        ),
        "source": "pytorch",
    }
    if converted_outputs is not None:
        conversion_options["outputs"] = converted_outputs
    if model_path.suffix != ".pt2" and model_type in {"roi_presence", "roi_pair_matcher"}:
        conversion_options["compute_precision"] = ct.precision.FLOAT32
    coreml_model = ct.convert(model_for_conversion, **conversion_options)
    coreml_model.save(output_path)
    return output_path


def replace_manifest_source(manifest_path: Path, source: dict[str, Any]) -> None:
    manifest = json.loads(manifest_path.read_text())
    manifest["source"] = source
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def annotate_checkpoint_contract(manifest_path: Path, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        return
    model_type = checkpoint_model_type(checkpoint)
    if model_type != "frame_presence":
        return
    manifest = json.loads(manifest_path.read_text())
    semantic_outputs = ("presence_logits", "region_logits")
    if len(manifest["outputs"]) != len(semantic_outputs):
        raise RuntimeError(
            "frame_presence unified export must have presence and region outputs"
        )
    manifest["model_type"] = model_type
    for output, semantic_name in zip(
        manifest["outputs"], semantic_outputs, strict=True
    ):
        output["semantic_name"] = semantic_name
    manifest["deployment_contract"] = {
        "preprocessing": checkpoint.get("preprocessing", {}),
        "outputs": checkpoint.get("outputs", {}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def use_head_output(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["outputs"]) != 1:
        raise RuntimeError("--head-output requires exactly one model output")
    output_name = manifest["outputs"][0]["name"]
    if not manifest["nodes"]:
        raise RuntimeError("--head-output requires a non-empty graph")
    final_node = manifest["nodes"][-1]
    if final_node["op"] != "aten.upsample_bilinear2d.vec" or final_node["outputs"][0]["name"] != output_name:
        raise RuntimeError("--head-output requires the final graph node to be the output upsample")
    head_name = final_node["inputs"][0]
    head_tensor = manifest["tensors"][head_name]
    manifest["nodes"] = manifest["nodes"][:-1]
    manifest["graph"]["node_count"] = len(manifest["nodes"])
    manifest["outputs"] = [
        {
            "name": head_name,
            "shape": head_tensor["shape"],
            "dtype": head_tensor["dtype"],
            "layout": head_tensor["layout"],
        }
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def export_model_to_unified_model(
    model_path: Path,
    output_dir: Path,
    batch_size: int = 1,
    head_output: bool = False,
) -> Path:
    model_path = Path(model_path)
    if batch_size <= 0:
        raise RuntimeError(f"batch_size must be positive, got {batch_size}")
    if model_path.suffix == ".pt2":
        if batch_size != 1:
            raise RuntimeError("--batch-size can only be used when exporting from a training checkpoint")
        manifest_path = export_pt2_to_unified_model(model_path, output_dir)
        if head_output:
            use_head_output(manifest_path)
        return manifest_path

    with tempfile.TemporaryDirectory(prefix="subfastnet-export-") as tmp:
        pt2_path = export_checkpoint_to_pt2(model_path, Path(tmp) / "model.pt2", batch_size=batch_size)
        manifest_path = export_pt2_to_unified_model(pt2_path, output_dir)
    if head_output:
        use_head_output(manifest_path)
    source: dict[str, Any] = {"checkpoint": str(model_path)}
    if batch_size != 1:
        source["batch_size"] = batch_size
    if head_output:
        source["head_output"] = True
    replace_manifest_source(manifest_path, source)
    annotate_checkpoint_contract(manifest_path, model_path)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a training checkpoint (.pt) or torch.export ExportedProgram (.pt2) into the subfast-net unified model format."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch dimension to use when exporting from a training checkpoint.",
    )
    parser.add_argument(
        "--head-output",
        action="store_true",
        help="Use the detector head logits as the runtime output instead of the final input-resolution upsample.",
    )
    parser.add_argument("model_path", type=Path, help="Input training checkpoint .pt or torch.export .pt2 file.")
    parser.add_argument("output_dir", type=Path, help="Output directory for model.json and weights.bin.")
    return parser.parse_args(argv)


def parse_coreml_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a training checkpoint (.pt) or torch.export ExportedProgram (.pt2) into a CoreML model package."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch dimension to use when exporting from a training checkpoint.",
    )
    parser.add_argument("model_path", type=Path, help="Input training checkpoint .pt or torch.export .pt2 file.")
    parser.add_argument("output_path", type=Path, help="Output CoreML .mlpackage or .mlmodel path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = export_model_to_unified_model(
        args.model_path,
        args.output_dir,
        batch_size=args.batch_size,
        head_output=args.head_output,
    )
    print(manifest_path)


def main_coreml(argv: list[str] | None = None) -> None:
    args = parse_coreml_args(argv)
    output_path = export_model_to_coreml_model(
        args.model_path,
        args.output_path,
        batch_size=args.batch_size,
    )
    print(output_path)


if __name__ == "__main__":
    main()
