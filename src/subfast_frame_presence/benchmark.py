from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from subfast_shared.runtime import choose_device

from .model import FramePresenceModel
from .visualization import save_frame_presence_visualization


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_model(
    model: FramePresenceModel,
    *,
    device: torch.device,
    input_width: int,
    input_height: int,
    focus_width: int,
    focus_height: int,
    batch_sizes: list[int],
    warmup: int,
    iterations: int,
) -> list[dict[str, float]]:
    model = model.to(device).eval()
    results: list[dict[str, float]] = []
    with torch.inference_mode():
        for batch_size in batch_sizes:
            images = torch.zeros(
                (batch_size, 3, input_height, input_width),
                dtype=torch.float32,
                device=device,
            )
            focus = torch.zeros(
                (batch_size, 3, focus_height, focus_width),
                dtype=torch.float32,
                device=device,
            )
            focus_mode = torch.zeros((batch_size,), dtype=torch.float32, device=device)
            for _ in range(warmup):
                model(images, focus, focus_mode)
            _synchronize(device)
            durations: list[float] = []
            for _ in range(iterations):
                started = time.perf_counter_ns()
                model(images, focus, focus_mode)
                _synchronize(device)
                durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
            ordered = sorted(durations)
            median = statistics.median(ordered)
            p90 = ordered[round((len(ordered) - 1) * 0.90)]
            results.append(
                {
                    "batch_size": float(batch_size),
                    "median_ms": median,
                    "p90_ms": p90,
                    "median_ms_per_frame": median / batch_size,
                    "frames_per_second": batch_size * 1000.0 / median,
                }
            )
    return results


def export_coreml_model(
    model: FramePresenceModel,
    output: Path,
    *,
    input_width: int,
    input_height: int,
    focus_width: int,
    focus_height: int,
    maximum_batch_size: int,
) -> Path:
    import coremltools as ct

    if maximum_batch_size < 1:
        raise ValueError("maximum_batch_size must be positive")
    output = output.expanduser().resolve()
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    export_model = copy.deepcopy(model).cpu().eval()
    images = torch.zeros((1, 3, input_height, input_width), dtype=torch.float32)
    focus = torch.zeros((1, 3, focus_height, focus_width), dtype=torch.float32)
    focus_mode = torch.zeros((1,), dtype=torch.float32)
    traced = torch.jit.trace(export_model, (images, focus, focus_mode))
    batch = ct.RangeDim(
        lower_bound=1,
        upper_bound=maximum_batch_size,
        default=1,
        symbol="batch",
    )
    batch_shape = lambda dimensions: ct.Shape(shape=(batch, *dimensions))
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="images",
                shape=batch_shape((3, input_height, input_width)),
            ),
            ct.TensorType(
                name="focus",
                shape=batch_shape((3, focus_height, focus_width)),
            ),
            ct.TensorType(name="focus_mode", shape=batch_shape(())),
        ],
        outputs=[
            ct.TensorType(name="presence_logits"),
            ct.TensorType(name="region_logits"),
        ],
        compute_units=ct.ComputeUnit.ALL,
    )
    converted.author = "subfast-net"
    converted.short_description = "Full-frame subtitle presence and enclosing contour samples"
    converted.user_defined_metadata["architecture_version"] = str(
        model.architecture_version
    )
    converted.user_defined_metadata["maximum_batch_size"] = str(maximum_batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(output)
    return output


def benchmark_coreml_model(
    path: Path,
    *,
    input_width: int,
    input_height: int,
    focus_width: int,
    focus_height: int,
    batch_sizes: list[int],
    warmup: int,
    iterations: int,
) -> list[dict[str, float]]:
    import coremltools as ct

    coreml_model = ct.models.MLModel(
        str(path.expanduser().resolve()),
        compute_units=ct.ComputeUnit.ALL,
    )
    results: list[dict[str, float]] = []
    for batch_size in batch_sizes:
        inputs = {
            "images": np.zeros(
                (batch_size, 3, input_height, input_width),
                dtype=np.float32,
            ),
            "focus": np.zeros(
                (batch_size, 3, focus_height, focus_width),
                dtype=np.float32,
            ),
            "focus_mode": np.zeros((batch_size,), dtype=np.float32),
        }
        for _ in range(warmup):
            coreml_model.predict(inputs)
        durations: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            coreml_model.predict(inputs)
            durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
        ordered = sorted(durations)
        median = statistics.median(ordered)
        p90 = ordered[round((len(ordered) - 1) * 0.90)]
        results.append(
            {
                "batch_size": float(batch_size),
                "median_ms": median,
                "p90_ms": p90,
                "median_ms_per_frame": median / batch_size,
                "frames_per_second": batch_size * 1000.0 / median,
            }
        )
    return results


def validate_coreml_model(
    path: Path,
    validation_cache: Path,
    *,
    batch_size: int,
    decision_threshold: float,
    heatmap_threshold: float,
) -> dict[str, float]:
    import coremltools as ct
    from torch.utils.data import DataLoader

    from .data import FramePresenceCacheDataset
    from .metrics import presence_metrics, region_metrics

    dataset = FramePresenceCacheDataset(validation_cache)
    coreml_model = ct.models.MLModel(
        str(path.expanduser().resolve()),
        compute_units=ct.ComputeUnit.ALL,
    )
    presence_logits: list[torch.Tensor] = []
    region_logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    presence: list[torch.Tensor] = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        outputs = coreml_model.predict(
            {
                "images": batch["image"].numpy(),
                "focus": batch["focus"].numpy(),
                "focus_mode": batch["focus_mode"].numpy(),
            }
        )
        presence_logits.append(
            torch.from_numpy(np.asarray(outputs["presence_logits"])).reshape(-1)
        )
        region_logits.append(
            torch.from_numpy(np.asarray(outputs["region_logits"]))
        )
        targets.append(batch["target"])
        presence.append(batch["presence"])
    all_presence_logits = torch.cat(presence_logits)
    all_region_logits = torch.cat(region_logits)
    all_targets = torch.cat(targets)
    all_presence = torch.cat(presence)
    metrics = presence_metrics(
        all_presence_logits,
        all_presence,
        threshold=decision_threshold,
    )
    localization, _ = region_metrics(
        all_region_logits,
        all_targets,
        all_presence,
        threshold=heatmap_threshold,
    )
    metrics.update(localization)
    return metrics


def benchmark_preprocess(
    luma: np.ndarray,
    rgb: np.ndarray,
    *,
    input_width: int,
    input_height: int,
    focus_width: int,
    focus_height: int,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    resized = np.empty((input_height, input_width), dtype=np.uint8)
    output = np.empty((3, input_height, input_width), dtype=np.float32)
    output[1] = np.linspace(-1.0, 1.0, input_width, dtype=np.float32)[None, :]
    output[2] = np.linspace(-1.0, 1.0, input_height, dtype=np.float32)[:, None]
    if luma.shape[1] / luma.shape[0] > 2.0:
        focus_box = (0.12, 0.70, 0.88, 1.0)
    elif luma.shape[1] <= 1280:
        focus_box = (0.08, 0.88, 0.92, 1.0)
    else:
        focus_box = (0.16, 0.86, 0.84, 1.0)
    left = round(focus_box[0] * luma.shape[1])
    top = round(focus_box[1] * luma.shape[0])
    right = round(focus_box[2] * luma.shape[1])
    bottom = round(focus_box[3] * luma.shape[0])
    crop = rgb[top:bottom, left:right]
    scale = min(focus_width / crop.shape[1], focus_height / crop.shape[0])
    resized_focus_width = max(1, min(focus_width, round(crop.shape[1] * scale)))
    resized_focus_height = max(1, min(focus_height, round(crop.shape[0] * scale)))
    resized_focus = np.empty((resized_focus_height, resized_focus_width, 3), dtype=np.uint8)
    focus_output = np.zeros((3, focus_height, focus_width), dtype=np.float32)
    focus_mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
    focus_std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
    focus_left = (focus_width - resized_focus_width) // 2
    focus_top = (focus_height - resized_focus_height) // 2
    focus_view = focus_output[
        :,
        focus_top : focus_top + resized_focus_height,
        focus_left : focus_left + resized_focus_width,
    ]

    def run() -> None:
        cv2.resize(
            luma,
            (input_width, input_height),
            dst=resized,
            interpolation=cv2.INTER_LINEAR,
        )
        np.multiply(resized, 1.0 / 127.5, out=output[0], casting="unsafe")
        output[0] -= 1.0
        cv2.resize(
            crop,
            (resized_focus_width, resized_focus_height),
            dst=resized_focus,
            interpolation=cv2.INTER_LINEAR,
        )
        np.multiply(
            resized_focus.transpose(2, 0, 1),
            1.0 / 255.0,
            out=focus_view,
            casting="unsafe",
        )
        np.subtract(focus_view, focus_mean, out=focus_view)
        np.divide(focus_view, focus_std, out=focus_view)

    for _ in range(warmup):
        run()
    durations: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        run()
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(durations)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[round((len(ordered) - 1) * 0.90)],
        "scope": "full_luma_plus_automatic_rgb_focus_resize_normalize_preallocated",
    }


def preprocess_frame(
    luma: np.ndarray,
    rgb: np.ndarray,
    *,
    input_width: int,
    input_height: int,
    focus_width: int,
    focus_height: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create the exact full-frame and automatic-focus tensors used at inference."""
    if luma.ndim != 2:
        raise ValueError("luma must have shape HxW")
    if rgb.shape != (*luma.shape, 3):
        raise ValueError("rgb must have shape HxWx3 and match luma")

    resized = cv2.resize(luma, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    image = np.empty((3, input_height, input_width), dtype=np.float32)
    np.multiply(resized, 1.0 / 127.5, out=image[0], casting="unsafe")
    image[0] -= 1.0
    image[1] = np.linspace(-1.0, 1.0, input_width, dtype=np.float32)[None, :]
    image[2] = np.linspace(-1.0, 1.0, input_height, dtype=np.float32)[:, None]

    source_height, source_width = luma.shape
    if source_width / source_height > 2.0:
        focus_mode = 1.0
        focus_box = (0.12, 0.70, 0.88, 1.0)
    elif source_width <= 1280:
        focus_mode = 2.0
        focus_box = (0.08, 0.88, 0.92, 1.0)
    else:
        focus_mode = 0.0
        focus_box = (0.16, 0.86, 0.84, 1.0)
    left = round(focus_box[0] * source_width)
    top = round(focus_box[1] * source_height)
    right = round(focus_box[2] * source_width)
    bottom = round(focus_box[3] * source_height)
    crop = rgb[top:bottom, left:right]
    scale = min(focus_width / crop.shape[1], focus_height / crop.shape[0])
    resized_width = max(1, min(focus_width, round(crop.shape[1] * scale)))
    resized_height = max(1, min(focus_height, round(crop.shape[0] * scale)))
    resized_focus = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    focus = np.zeros((3, focus_height, focus_width), dtype=np.float32)
    focus_left = (focus_width - resized_width) // 2
    focus_top = (focus_height - resized_height) // 2
    focus_view = focus[
        :,
        focus_top : focus_top + resized_height,
        focus_left : focus_left + resized_width,
    ]
    np.multiply(
        resized_focus.transpose(2, 0, 1),
        1.0 / 255.0,
        out=focus_view,
        casting="unsafe",
    )
    focus_mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
    focus_std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
    np.subtract(focus_view, focus_mean, out=focus_view)
    np.divide(focus_view, focus_std, out=focus_view)
    return image, focus, focus_mode


def load_model(
    path: Path | None,
    width: int,
    kernel_size: int,
) -> tuple[FramePresenceModel, dict]:
    if path is None:
        return FramePresenceModel(
            width=width,
            kernel_size=kernel_size,
        ), {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model", {})
    width = int(model_config.get("width", width))
    kernel_size = int(model_config.get("kernel_size", kernel_size))
    model = FramePresenceModel(
        width=width,
        kernel_size=kernel_size,
    )
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    return model, checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark subfast_frame_presence.")
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--input-height", type=int, default=144)
    parser.add_argument("--focus-width", type=int, default=256)
    parser.add_argument("--focus-height", type=int, default=32)
    parser.add_argument("--batch-size", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--visualization-output",
        "--visualize-output",
        dest="visualization_output",
        type=Path,
        help="save original, heatmap, and overlay views for --image",
    )
    parser.add_argument("--coreml-output", type=Path)
    parser.add_argument("--coreml-maximum-batch-size", type=int, default=32)
    parser.add_argument("--validation-cache", type=Path)
    args = parser.parse_args(argv)
    if args.visualization_output is not None and args.image is None:
        parser.error("--visualization-output requires --image")
    if args.visualization_output is not None and args.checkpoint is None:
        parser.error("--visualization-output requires a checkpoint")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model, checkpoint = load_model(
        args.checkpoint,
        args.width,
        args.kernel_size,
    )
    preprocessing = checkpoint.get("preprocessing", {})
    input_width = int(preprocessing.get("input_width", args.input_width))
    input_height = int(preprocessing.get("input_height", args.input_height))
    focus_width = int(preprocessing.get("focus_width", args.focus_width))
    focus_height = int(preprocessing.get("focus_height", args.focus_height))
    device = choose_device(args.device)
    batch_sizes = args.batch_size or [1, 2, 4, 8, 16, 32]
    if args.image:
        with Image.open(args.image) as image:
            luma = np.asarray(image.convert("L"), dtype=np.uint8)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        luma = np.zeros((1080, 1920), dtype=np.uint8)
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    payload = {
        "device": str(device),
        "torch_version": torch.__version__,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input": {
            "full_frame": [3, input_height, input_width],
            "focus": [3, focus_height, focus_width],
        },
        "preprocessing": benchmark_preprocess(
            luma,
            rgb,
            input_width=input_width,
            input_height=input_height,
            focus_width=focus_width,
            focus_height=focus_height,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "inference": benchmark_model(
            model,
            device=device,
            input_width=input_width,
            input_height=input_height,
            focus_width=focus_width,
            focus_height=focus_height,
            batch_sizes=batch_sizes,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }
    if args.visualization_output is not None:
        image_input, focus_input, focus_mode = preprocess_frame(
            luma,
            rgb,
            input_width=input_width,
            input_height=input_height,
            focus_width=focus_width,
            focus_height=focus_height,
        )
        model = model.to(device).eval()
        with torch.inference_mode():
            presence_logits, region_logits = model(
                torch.from_numpy(image_input).unsqueeze(0).to(device),
                torch.from_numpy(focus_input).unsqueeze(0).to(device),
                torch.tensor([focus_mode], dtype=torch.float32, device=device),
            )
        presence_score = float(torch.sigmoid(presence_logits)[0].cpu())
        heatmap = torch.sigmoid(region_logits)[0, 0].cpu().numpy()
        outputs = checkpoint.get("outputs", {})
        decision_threshold = float(outputs.get("decision_threshold", 0.5))
        heatmap_threshold = float(outputs.get("heatmap_threshold", 0.5))
        visualization = save_frame_presence_visualization(
            rgb,
            heatmap,
            presence_score=presence_score,
            heatmap_threshold=heatmap_threshold,
            output=args.visualization_output,
        )
        visualization.update(
            {
                "image": str(args.image.expanduser().resolve()),
                "presence": presence_score >= decision_threshold,
                "decision_threshold": decision_threshold,
            }
        )
        payload["visualization"] = visualization
    if args.coreml_output is not None:
        coreml_path = export_coreml_model(
            model,
            args.coreml_output,
            input_width=input_width,
            input_height=input_height,
            focus_width=focus_width,
            focus_height=focus_height,
            maximum_batch_size=args.coreml_maximum_batch_size,
        )
        payload["coreml"] = {
            "model": str(coreml_path),
            "maximum_batch_size": args.coreml_maximum_batch_size,
            "inference": benchmark_coreml_model(
                coreml_path,
                input_width=input_width,
                input_height=input_height,
                focus_width=focus_width,
                focus_height=focus_height,
                batch_sizes=batch_sizes,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
        }
        if args.validation_cache is not None:
            outputs = checkpoint.get("outputs", {})
            payload["coreml"]["validation"] = validate_coreml_model(
                coreml_path,
                args.validation_cache,
                batch_size=args.coreml_maximum_batch_size,
                decision_threshold=float(outputs.get("decision_threshold", 0.5)),
                heatmap_threshold=float(outputs.get("heatmap_threshold", 0.5)),
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
