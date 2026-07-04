from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

DEFAULT_PYCRAFTER_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/d9/56/020f9653668a28d7f969ae339f4fac3c5289bf2cd479c3fdd15600c7ab76/"
    "pycrafter-0.0.7-py3-none-any.whl"
)
DEFAULT_PYCRAFTER_WHEEL_SHA256 = "3f11551ab195c96a6aff71190bbd9465e86a4bb8da218a37bdb180805291bc4b"
PYCRAFTER_CRAFTNET_MEMBER = "crafter/resources/craftnet.onnx"
DEFAULT_CRAFT_ONNX_PATH = Path.home() / ".cache" / "subfast-net" / "craft" / "craftnet.onnx"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save raw CRAFT score maps for ROI samples.")
    parser.add_argument("--root", type=Path, required=True, help="ROI sample folder containing annotations.jsonl.")
    parser.add_argument("--craft-onnx", type=Path, help=f"Path to craftnet.onnx. Defaults to {DEFAULT_CRAFT_ONNX_PATH}.")
    parser.add_argument("--pycrafter-wheel-url", default=DEFAULT_PYCRAFTER_WHEEL_URL, help="Wheel URL used to extract craftnet.onnx.")
    parser.add_argument("--output-subdir", default="craft_outputs", help="Output folder under the ROI sample folder.")
    parser.add_argument("--canvas-size", type=int, default=1280)
    parser.add_argument("--mag-ratio", type=float, default=1.5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-score-text-png", action="store_true", help="Also save score_text grayscale PNG previews.")
    parser.add_argument("--write-score-link-png", action="store_true", help="Also save score_link grayscale PNG previews.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps", "coreml"))
    return parser.parse_args(argv)


def choose_providers(name: str) -> list[str]:
    available = ort.get_available_providers()
    preferred = {
        "auto": ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"),
        "cpu": ("CPUExecutionProvider",),
        "cuda": ("CUDAExecutionProvider",),
        "mps": ("CoreMLExecutionProvider",),
        "coreml": ("CoreMLExecutionProvider",),
    }[name]
    providers = [provider for provider in preferred if provider in available]
    if not providers:
        raise RuntimeError(f"ONNX Runtime provider for --device {name!r} is unavailable. Available providers: {available}")
    if "CPUExecutionProvider" not in providers and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"downloaded pycrafter wheel sha256 mismatch: expected {expected}, got {actual}")


def resolve_craft_onnx(path: Path | None, wheel_url: str) -> Path:
    onnx_path = (path or DEFAULT_CRAFT_ONNX_PATH).expanduser()
    if onnx_path.exists():
        return onnx_path
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wheel_path = onnx_path.with_suffix(".pycrafter.whl.part")
    temp_onnx_path = onnx_path.with_suffix(onnx_path.suffix + ".part")
    print(f"downloading pycrafter CRAFT model: {wheel_url} -> {onnx_path}", flush=True)
    urlretrieve(wheel_url, wheel_path)
    verify_sha256(wheel_path, DEFAULT_PYCRAFTER_WHEEL_SHA256)
    with zipfile.ZipFile(wheel_path) as wheel:
        with wheel.open(PYCRAFTER_CRAFTNET_MEMBER) as source, temp_onnx_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    temp_onnx_path.replace(onnx_path)
    wheel_path.unlink(missing_ok=True)
    return onnx_path


def load_craft_session(craft_onnx: Path, providers: list[str]) -> ort.InferenceSession:
    if not craft_onnx.exists():
        raise FileNotFoundError(f"CRAFT ONNX model missing: {craft_onnx}")
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    return ort.InferenceSession(str(craft_onnx), sess_options=session_options, providers=providers)


def read_annotations(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resize_aspect_ratio(image: np.ndarray, canvas_size: int, mag_ratio: float) -> tuple[np.ndarray, tuple[int, int]]:
    height, width, channels = image.shape
    target_size = min(mag_ratio * max(height, width), canvas_size)
    ratio = target_size / max(height, width)
    target_h, target_w = int(height * ratio), int(width * ratio)
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    padded_h = target_h + ((32 - target_h % 32) % 32)
    padded_w = target_w + ((32 - target_w % 32) % 32)
    canvas = np.zeros((padded_h, padded_w, channels), dtype=np.float32)
    canvas[:target_h, :target_w, :] = resized
    return canvas, (target_w, target_h)


def normalize_mean_variance(image: np.ndarray) -> np.ndarray:
    normalized = image.copy().astype(np.float32)
    normalized -= np.array([0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0], dtype=np.float32)
    normalized /= np.array([0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0], dtype=np.float32)
    return normalized


def load_input(image_path: Path, canvas_size: int, mag_ratio: float) -> tuple[np.ndarray, dict[str, int]]:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        original_w, original_h = rgb.size
        array = np.asarray(rgb, dtype=np.float32)
    resized, (target_w, target_h) = resize_aspect_ratio(array, canvas_size, mag_ratio)
    normalized = normalize_mean_variance(resized)
    tensor = np.transpose(normalized, (2, 0, 1))[None, :, :, :].astype(np.float32)
    metadata = {
        "original_w": original_w,
        "original_h": original_h,
        "target_w": target_w,
        "target_h": target_h,
        "resized_w": int(tensor.shape[3]),
        "resized_h": int(tensor.shape[2]),
    }
    return tensor, metadata


def upsample_score(score: np.ndarray, meta: dict[str, int]) -> np.ndarray:
    resized = cv2.resize(score, (meta["resized_w"], meta["resized_h"]), interpolation=cv2.INTER_LINEAR)
    cropped = resized[: meta["target_h"], : meta["target_w"]]
    full = cv2.resize(cropped, (meta["original_w"], meta["original_h"]), interpolation=cv2.INTER_LINEAR)
    return np.clip(full, 0.0, 1.0).astype(np.float32)


def write_png(path: Path, score: np.ndarray) -> None:
    Image.fromarray((np.clip(score, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode="L").save(path)


def run_sample(
    session: ort.InferenceSession,
    row: dict[str, Any],
    root: Path,
    score_maps_dir: Path,
    score_text_png_dir: Path,
    score_link_png_dir: Path,
    canvas_size: int,
    mag_ratio: float,
    write_score_text_png: bool,
    write_score_link_png: bool,
) -> dict[str, Any]:
    model_input, metadata = load_input(root / str(row["image"]), canvas_size, mag_ratio)
    output, _ = session.run(None, {"image": model_input})
    stem = Path(str(row["image"])).stem
    score_text = upsample_score(output[0, 0, :, :].astype(np.float32), metadata)
    score_link = upsample_score(output[0, 1, :, :].astype(np.float32), metadata)
    npz_path = score_maps_dir / f"{stem}.npz"
    text_png = score_text_png_dir / f"{stem}.png"
    link_png = score_link_png_dir / f"{stem}.png"
    np.savez_compressed(npz_path, score_text=score_text, score_link=score_link)
    manifest_row = {
        "image": row["image"],
        "score_npz": str(npz_path.relative_to(root).as_posix()),
        "score_text_max": float(score_text.max()) if score_text.size else 0.0,
        "score_text_mean": float(score_text.mean()) if score_text.size else 0.0,
        "teacher_method": "craft_raw_score_maps",
    }
    if write_score_text_png:
        write_png(text_png, score_text)
        manifest_row["score_text_png"] = str(text_png.relative_to(root).as_posix())
    if write_score_link_png:
        write_png(link_png, score_link)
        manifest_row["score_link_png"] = str(link_png.relative_to(root).as_posix())
    return manifest_row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    annotations_path = root / "annotations.jsonl"
    if not annotations_path.exists():
        raise FileNotFoundError(f"missing annotations.jsonl: {annotations_path}")
    output_dir = root / args.output_subdir
    score_maps_dir = output_dir / "score_maps"
    score_text_png_dir = output_dir / "score_text_png"
    score_link_png_dir = output_dir / "score_link_png"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_maps_dir.mkdir(parents=True, exist_ok=True)
    if args.write_score_text_png:
        score_text_png_dir.mkdir(parents=True, exist_ok=True)
    if args.write_score_link_png:
        score_link_png_dir.mkdir(parents=True, exist_ok=True)
    rows = read_annotations(annotations_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if args.skip_existing:
        rows = [row for row in rows if not (score_maps_dir / f"{Path(str(row['image'])).stem}.npz").exists()]

    providers = choose_providers(args.device)
    craft_onnx = resolve_craft_onnx(args.craft_onnx, args.pycrafter_wheel_url)
    session = load_craft_session(craft_onnx, providers)
    manifest_path = output_dir / "manifest.jsonl"
    written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in tqdm(rows, desc=f"craft {root}"):
            manifest_row = run_sample(
                session,
                row,
                root,
                score_maps_dir,
                score_text_png_dir,
                score_link_png_dir,
                args.canvas_size,
                args.mag_ratio,
                args.write_score_text_png,
                args.write_score_link_png,
            )
            manifest.write(json.dumps(manifest_row, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    summary = {
        "teacher_method": "craft_raw_score_maps",
        "samples": written,
        "canvas_size": args.canvas_size,
        "mag_ratio": args.mag_ratio,
        "craft_onnx": str(craft_onnx),
        "onnx_providers": providers,
        "manifest": str(manifest_path.relative_to(root).as_posix()),
        "score_maps_dir": str(score_maps_dir.relative_to(root).as_posix()),
        "score_text_png_dir": str(score_text_png_dir.relative_to(root).as_posix()) if args.write_score_text_png else None,
        "score_link_png_dir": str(score_link_png_dir.relative_to(root).as_posix()) if args.write_score_link_png else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved CRAFT outputs: {written} -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
