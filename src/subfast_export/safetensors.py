from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from subfast_shared.vision import IMAGENET_MEAN_VALUES, IMAGENET_STD_VALUES

from .checkpoint import checkpoint_model_type, checkpoint_state_dict


FORMAT_NAME = "subfast-net.safetensors"
FORMAT_VERSION = 1
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
IMAGENET_MEAN = list(IMAGENET_MEAN_VALUES)
IMAGENET_STD = list(IMAGENET_STD_VALUES)


def _checkpoint_settings(checkpoint: Mapping[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    raw_settings = checkpoint.get("settings")
    if raw_settings is None:
        return {}
    if not isinstance(raw_settings, Mapping):
        raise RuntimeError(f"checkpoint settings must be a mapping: {checkpoint_path}")
    return dict(raw_settings)


def _checkpoint_value(
    checkpoint: Mapping[str, Any],
    settings: Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    value = checkpoint.get(name)
    if value is not None:
        return value
    value = settings.get(name)
    return default if value is None else value


def _positive_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be a positive integer, got {value!r}") from exc
    if result <= 0:
        raise RuntimeError(f"{name} must be a positive integer, got {value!r}")
    return result


def _number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be numeric, got {value!r}") from exc


def _roi_size(
    checkpoint: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    default: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    value = _checkpoint_value(checkpoint, settings, "resize_roi", default)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeError(f"resize_roi must be [width, height], got {value!r}")
    return _positive_int(value[0], "resize_roi width"), _positive_int(value[1], "resize_roi height")


def _roi_shape(resize_roi: tuple[int, int] | None) -> list[Any]:
    if resize_roi is None:
        return ["batch", 3, "height", "width"]
    width, height = resize_roi
    return ["batch", 3, height, width]


def _imagenet_preprocessing() -> dict[str, Any]:
    return {
        "color_space": "RGB",
        "source_value_range": [0.0, 1.0],
        "tensor_layout": "NCHW",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
    }


def _infer_width(state_dict: Mapping[str, torch.Tensor], key: str, default: int) -> int:
    tensor = state_dict.get(key)
    if tensor is None or tensor.ndim == 0:
        return default
    return int(tensor.shape[0])


def _detector_config(
    checkpoint: Mapping[str, Any],
    settings: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    image_size = _positive_int(_checkpoint_value(checkpoint, settings, "image_size", 256), "image_size")
    stride = _positive_int(_checkpoint_value(checkpoint, settings, "stride", 32), "stride")
    return {
        "model": {
            "type": "subtitle_detector",
            "class_name": "SubtitleDetector",
            "module": "subfast_detector.model",
            "architecture_version": None,
            "kwargs": {"width": _infer_width(state_dict, "stem.0.weight", 32)},
        },
        "input": {
            "tensors": [
                {
                    "name": "images",
                    "shape": ["batch", 3, "padded_height", "padded_width"],
                    "layout": "NCHW",
                    "coordinate_space": "letterboxed_image",
                }
            ]
        },
        "preprocessing": {
            **_imagenet_preprocessing(),
            "resize": {
                "mode": "aspect_ratio_scale",
                "longest_side": image_size,
                "rounding": "round",
            },
            "padding": {
                "placement": "right_bottom",
                "value": 0.0,
                "align_to_stride": stride,
                "coordinate_space": "normalized_nchw",
            },
        },
        "output": {
            "tensors": [
                {
                    "name": "logits",
                    "shape": ["batch", 2, "padded_height", "padded_width"],
                    "layout": "NCHW",
                    "coordinate_space": "letterboxed_image",
                    "channels": ["region_logit", "kernel_logit"],
                }
            ]
        },
        "postprocessing": {
            "algorithm": "subfast_detector.postprocess.logits_to_boxes",
            "region_threshold": _number(
                _checkpoint_value(checkpoint, settings, "region_threshold", 0.5),
                "region_threshold",
            ),
            "kernel_threshold": _number(
                _checkpoint_value(checkpoint, settings, "kernel_threshold", 0.5),
                "kernel_threshold",
            ),
            "max_detection_width_ratio": _number(
                _checkpoint_value(checkpoint, settings, "max_detection_width_ratio", 1.0),
                "max_detection_width_ratio",
            ),
            "min_size": 3.0,
        },
    }


def _frame_presence_config(
    checkpoint: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    model_config = checkpoint.get("model")
    preprocessing = checkpoint.get("preprocessing")
    outputs = checkpoint.get("outputs")
    if not isinstance(model_config, Mapping):
        raise RuntimeError("frame_presence checkpoint model config must be a mapping")
    if not isinstance(preprocessing, Mapping):
        raise RuntimeError("frame_presence checkpoint preprocessing must be a mapping")
    if not isinstance(outputs, Mapping):
        raise RuntimeError("frame_presence checkpoint outputs must be a mapping")

    input_width = _positive_int(preprocessing.get("input_width", 256), "input_width")
    input_height = _positive_int(preprocessing.get("input_height", 144), "input_height")
    focus_width = _positive_int(preprocessing.get("focus_width", 256), "focus_width")
    focus_height = _positive_int(preprocessing.get("focus_height", 32), "focus_height")
    heatmap_width = _positive_int(
        preprocessing.get("heatmap_width", 64), "heatmap_width"
    )
    heatmap_height = _positive_int(
        preprocessing.get("heatmap_height", 72), "heatmap_height"
    )
    stride_x = _positive_int(
        preprocessing.get("heatmap_stride_x", 4), "heatmap_stride_x"
    )
    stride_y = _positive_int(
        preprocessing.get("heatmap_stride_y", 2), "heatmap_stride_y"
    )
    width = _positive_int(
        model_config.get(
            "width",
            _infer_width(state_dict, "focus_backbone.0.weight", 16),
        ),
        "width",
    )
    kernel_size = _positive_int(model_config.get("kernel_size", 5), "kernel_size")
    return {
        "model": {
            "type": "frame_presence",
            "class_name": "FramePresenceModel",
            "module": "subfast_frame_presence.model",
            "architecture_version": _positive_int(
                checkpoint.get("architecture_version", 1),
                "architecture_version",
            ),
            "kwargs": {"width": width, "kernel_size": kernel_size},
        },
        "input": {
            "tensors": [
                {
                    "name": "images",
                    "shape": ["batch", 3, input_height, input_width],
                    "layout": "NCHW",
                    "coordinate_space": "full_frame_stretch",
                    "channels": ["luma", "x_position", "y_position"],
                },
                {
                    "name": "focus",
                    "shape": ["batch", 3, focus_height, focus_width],
                    "layout": "NCHW",
                    "coordinate_space": "automatic_subtitle_focus",
                    "color_space": "RGB",
                },
                {
                    "name": "focus_mode",
                    "shape": ["batch"],
                    "values": {"0": "normal", "1": "wide", "2": "legacy"},
                },
            ]
        },
        "preprocessing": {
            "input_kind": preprocessing.get(
                "input_kind",
                "full_frame_luma_xy_plus_automatic_rgb_focus",
            ),
            "full_frame": {
                "resize": {
                    "mode": preprocessing.get("resize_mode", "full_frame_stretch"),
                    "target_size": {"width": input_width, "height": input_height},
                    "interpolation": "bilinear",
                },
                "luma_normalization": "uint8 / 127.5 - 1.0",
                "position_channels": "linear -1.0 to 1.0 in source axes",
            },
            "focus": {
                "resize": {
                    "mode": preprocessing.get(
                        "focus_resize_mode", "aspect_preserving_letterbox"
                    ),
                    "target_size": {"width": focus_width, "height": focus_height},
                    "interpolation": "bilinear",
                },
                "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                "normalized_padding_value": 0.0,
                "automatic_boxes": {
                    "normal": [0.16, 0.86, 0.84, 1.0],
                    "wide": [0.12, 0.70, 0.88, 1.0],
                    "legacy": [0.08, 0.88, 0.92, 1.0],
                },
            },
        },
        "output": {
            "tensors": [
                {
                    "name": "presence_logits",
                    "shape": ["batch"],
                    "score_transform": "sigmoid",
                },
                {
                    "name": "region_logits",
                    "shape": ["batch", 1, heatmap_height, heatmap_width],
                    "layout": "NCHW",
                    "coordinate_space": "full_source_frame",
                    "stride": {"x": stride_x, "y": stride_y},
                    "score_transform": "sigmoid",
                    "semantics": "interleaved subtitle-enclosing contour samples",
                },
            ]
        },
        "postprocessing": {
            "decision_threshold": _number(
                outputs.get("decision_threshold", 0.5), "decision_threshold"
            ),
            "heatmap_threshold": _number(
                outputs.get("heatmap_threshold", 0.5), "heatmap_threshold"
            ),
            "coordinate_mapping": {
                "x": "heatmap_x / heatmap_width * source_width",
                "y": "heatmap_y / heatmap_height * source_height",
            },
        },
    }


def _roi_preprocessing(resize_roi: tuple[int, int] | None, resize_mode: str) -> dict[str, Any]:
    resize: dict[str, Any]
    if resize_roi is None:
        resize = {"mode": "none"}
    else:
        width, height = resize_roi
        resize = {
            "mode": resize_mode,
            "target_size": {"width": width, "height": height},
            "interpolation": "bilinear",
            "align_corners": False,
        }
    return {
        **_imagenet_preprocessing(),
        "resize": resize,
        "padding": (
            {
                "value": 0.0,
                "coordinate_space": "normalized_nchw",
                "valid_mask_value": 0.0,
            }
            if resize_mode == "letterbox" and resize_roi is not None
            else None
        ),
    }


def _roi_presence_config(
    checkpoint: Mapping[str, Any],
    settings: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    resize_roi = _roi_size(checkpoint, settings)
    resize_mode = str(_checkpoint_value(checkpoint, settings, "resize_mode", "letterbox"))
    if resize_mode not in {"letterbox", "stretch"}:
        raise RuntimeError(f"unsupported roi_presence resize_mode: {resize_mode}")
    score_contract = checkpoint.get("score_contract")
    if score_contract is not None and not isinstance(score_contract, Mapping):
        raise RuntimeError("roi_presence score_contract must be a mapping")
    postprocessing: dict[str, Any] = {
        "decision_threshold": _number(
            _checkpoint_value(checkpoint, settings, "decision_threshold", 0.5),
            "decision_threshold",
        ),
        "score_transform": "sigmoid",
    }
    if score_contract is not None:
        postprocessing["score_contract"] = _json_value(score_contract)
    return {
        "model": {
            "type": "roi_presence",
            "class_name": "RoiPresenceModel",
            "module": "subfast_roi_presence.model",
            "architecture_version": _positive_int(
                _checkpoint_value(checkpoint, settings, "architecture_version", 1),
                "architecture_version",
            ),
            "kwargs": {
                "width": _positive_int(
                    _checkpoint_value(
                        checkpoint,
                        settings,
                        "width",
                        _infer_width(state_dict, "backbone.0.weight", 16),
                    ),
                    "width",
                ),
                "evidence_kernel_size": _positive_int(
                    _checkpoint_value(checkpoint, settings, "evidence_kernel_size", 5),
                    "evidence_kernel_size",
                ),
            },
        },
        "input": {
            "tensors": [
                {
                    "name": "images",
                    "shape": _roi_shape(resize_roi),
                    "layout": "NCHW",
                    "coordinate_space": "resized_roi",
                },
                {
                    "name": "valid_mask",
                    "shape": ["batch", 1, *_roi_shape(resize_roi)[2:]],
                    "layout": "NCHW",
                    "coordinate_space": "resized_roi",
                    "optional": True,
                    "semantics": "one for source pixels and zero for letterbox padding",
                },
            ]
        },
        "preprocessing": _roi_preprocessing(resize_roi, resize_mode),
        "output": {
            "tensors": [
                {
                    "name": "presence_logit",
                    "shape": ["batch"],
                    "score_transform": "sigmoid",
                },
                {
                    "name": "region_logits",
                    "shape": ["batch", 1, "ceil(height/4)", "ceil(width/4)"],
                    "coordinate_space": "stride_4_roi",
                    "available_via": "RoiPresenceModel.forward_with_presence_map",
                },
            ]
        },
        "postprocessing": postprocessing,
    }


def _roi_embedding_config(
    checkpoint: Mapping[str, Any],
    settings: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    resize_roi = _roi_size(checkpoint, settings)
    aggregation = str(_checkpoint_value(checkpoint, settings, "embedding_aggregation", "width_tokens"))
    if aggregation not in {"width_tokens", "local_alignment"}:
        raise RuntimeError(f"unsupported roi_presence_embedding aggregation: {aggregation}")
    return {
        "model": {
            "type": "roi_presence_embedding",
            "class_name": "RoiPresenceEmbeddingModel",
            "module": "subfast_roi_embedding.model",
            "architecture_version": None,
            "kwargs": {
                "width": _positive_int(
                    _checkpoint_value(
                        checkpoint,
                        settings,
                        "width",
                        _infer_width(state_dict, "backbone.0.weight", 32),
                    ),
                    "width",
                ),
                "embedding_dim": _positive_int(
                    _checkpoint_value(checkpoint, settings, "embedding_dim", 256),
                    "embedding_dim",
                ),
                "presence_topk_ratio": _number(
                    _checkpoint_value(checkpoint, settings, "presence_topk_ratio", 0.05),
                    "presence_topk_ratio",
                ),
                "embedding_width_tokens": _positive_int(
                    _checkpoint_value(checkpoint, settings, "embedding_width_tokens", 32),
                    "embedding_width_tokens",
                ),
                "embedding_aggregation": aggregation,
            },
        },
        "input": {
            "tensors": [
                {
                    "name": "images",
                    "shape": _roi_shape(resize_roi),
                    "layout": "NCHW",
                    "coordinate_space": "resized_roi",
                }
            ]
        },
        "preprocessing": _roi_preprocessing(resize_roi, "stretch"),
        "output": {
            "tensors": [
                {"name": "presence_logit", "shape": ["batch"], "score_transform": "sigmoid"},
                {
                    "name": "embedding",
                    "shape": ["batch", "embedding_features"],
                    "semantics": "normalized descriptor unless embedding_aggregation is local_alignment",
                },
            ]
        },
        "postprocessing": {
            "embedding_similarity_threshold": _number(
                _checkpoint_value(checkpoint, settings, "embedding_similarity_threshold", 0.5),
                "embedding_similarity_threshold",
            )
        },
    }


def _roi_pair_config(
    checkpoint: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    resize_roi = _roi_size(checkpoint, settings, default=(256, 64))
    return {
        "model": {
            "type": "roi_pair_matcher",
            "class_name": "RoiPairMatcher",
            "module": "subfast_roi_matcher.model",
            "architecture_version": _positive_int(
                _checkpoint_value(checkpoint, settings, "architecture_version", 1),
                "architecture_version",
            ),
            "pooling_version": _positive_int(
                _checkpoint_value(checkpoint, settings, "pooling_version", 1),
                "pooling_version",
            ),
            "kwargs": {},
        },
        "input": {
            "tensors": [
                {
                    "name": name,
                    "shape": _roi_shape(resize_roi),
                    "layout": "NCHW",
                    "coordinate_space": "resized_roi",
                }
                for name in ("left", "right")
            ]
        },
        "preprocessing": _roi_preprocessing(resize_roi, "stretch"),
        "output": {
            "tensors": [
                {
                    "name": "same_subtitle_logit",
                    "shape": ["batch"],
                    "score_transform": "sigmoid",
                    "available_via": "RoiPairMatcher.forward or RoiPairInference.forward",
                }
            ]
        },
        "postprocessing": {
            "decision_threshold": _number(
                _checkpoint_value(checkpoint, settings, "threshold", 0.5),
                "threshold",
            ),
            "score_transform": "sigmoid",
        },
    }


def _model_config(
    checkpoint: Mapping[str, Any],
    settings: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    model_type = checkpoint_model_type(checkpoint)
    if model_type is None:
        return _detector_config(checkpoint, settings, state_dict)
    if model_type == "frame_presence":
        return _frame_presence_config(checkpoint, state_dict)
    if model_type == "roi_presence":
        return _roi_presence_config(checkpoint, settings, state_dict)
    if model_type == "roi_presence_embedding":
        return _roi_embedding_config(checkpoint, settings, state_dict)
    if model_type == "roi_pair_matcher":
        return _roi_pair_config(checkpoint, settings)
    raise RuntimeError(f"unsupported checkpoint model_type for safetensors export: {model_type!r}")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise RuntimeError(f"value cannot be represented in config.json: {type(value).__name__}")


def _safetensors_state_dict(state_dict: Mapping[str, Any], checkpoint_path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if not isinstance(name, str):
            raise RuntimeError(f"checkpoint model key is not a string: {checkpoint_path}")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"checkpoint model entry is not a tensor: {name}")
        if value.layout != torch.strided:
            raise RuntimeError(f"safetensors export requires strided tensor storage: {name}")
        tensors[name] = value.detach().cpu().contiguous()
    if not tensors:
        raise RuntimeError(f"checkpoint model state is empty: {checkpoint_path}")
    return tensors


def export_checkpoint_to_safetensors(checkpoint_path: Path, output_dir: Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    if checkpoint_path.suffix == ".pt2":
        raise RuntimeError("safetensors export requires a training checkpoint, not a torch.export .pt2 file")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"invalid training checkpoint: {checkpoint_path}")
    raw_state_dict = checkpoint_state_dict(checkpoint)
    if raw_state_dict is None:
        raise RuntimeError(
            f"checkpoint does not contain a model state dict: {checkpoint_path}"
        )

    state_dict = _safetensors_state_dict(raw_state_dict, checkpoint_path)
    settings = _checkpoint_settings(checkpoint, checkpoint_path)
    contract = _model_config(checkpoint, settings, state_dict)
    config = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "weights": {"file": WEIGHTS_FILE, "format": "safetensors"},
        "source": {"checkpoint": str(checkpoint_path)},
        **contract,
    }
    config["model_type"] = config["model"]["type"]

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / WEIGHTS_FILE
    save_file(
        state_dict,
        str(weights_path),
        metadata={
            "format": FORMAT_NAME,
            "format_version": str(FORMAT_VERSION),
            "model_type": str(config["model_type"]),
        },
    )
    config_path = output_dir / CONFIG_FILE
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a subfast-net training checkpoint as model.safetensors and config.json."
    )
    parser.add_argument("checkpoint_path", type=Path, help="Input training checkpoint .pt file.")
    parser.add_argument("output_dir", type=Path, help="Output directory for model.safetensors and config.json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(export_checkpoint_to_safetensors(args.checkpoint_path, args.output_dir))


if __name__ == "__main__":
    main()
