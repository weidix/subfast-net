from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from subtitle_timing_core.formats import (
    H264_TIMING_VISUAL_FEATURE_FORMAT as VISUAL_FEATURE_FORMAT,
    H264_TIMING_VISUAL_FEATURE_VERSION as VISUAL_FEATURE_VERSION,
)
from subtitle_timing_core.hashing import file_sha256


@dataclass(frozen=True)
class VisualFeatureSettings:
    width: int = 256
    height: int = 32
    row_bins: int = 4
    column_bins: int = 8
    edge_thresholds: tuple[int, ...] = (8, 16, 32, 64)

    def __post_init__(self) -> None:
        if self.width < 16 or self.height < 8:
            raise ValueError("visual feature dimensions are too small")
        if self.row_bins <= 0 or self.height % self.row_bins != 0:
            raise ValueError("row bins must divide the visual feature height")
        if self.column_bins <= 0 or self.width % self.column_bins != 0:
            raise ValueError("column bins must divide the visual feature width")
        if not self.edge_thresholds or any(
            threshold <= 0 or threshold >= 256 for threshold in self.edge_thresholds
        ):
            raise ValueError("edge thresholds must be bytes in (0,256)")

    def to_dict(self) -> dict:
        values = asdict(self)
        values["edge_thresholds"] = list(self.edge_thresholds)
        return values


def _base_feature_names(settings: VisualFeatureSettings) -> list[str]:
    names = [
        "visual_gray_mean",
        "visual_gray_std",
        "visual_gradient_x_mean",
        "visual_gradient_x_std",
        "visual_gradient_y_mean",
        "visual_gradient_y_std",
    ]
    for threshold in settings.edge_thresholds:
        names.extend(
            [
                f"visual_gradient_x_ratio_{threshold:03d}",
                f"visual_gradient_y_ratio_{threshold:03d}",
            ]
        )
    for index in range(settings.row_bins):
        names.extend(
            [
                f"visual_row_{index:02d}_gradient_x_mean",
                f"visual_row_{index:02d}_gradient_y_mean",
                f"visual_row_{index:02d}_gradient_x_ratio_032",
                f"visual_row_{index:02d}_gradient_y_ratio_032",
            ]
        )
    for index in range(settings.column_bins):
        names.extend(
            [
                f"visual_column_{index:02d}_gradient_x_mean",
                f"visual_column_{index:02d}_gradient_y_mean",
                f"visual_column_{index:02d}_gradient_x_ratio_032",
                f"visual_column_{index:02d}_gradient_y_ratio_032",
            ]
        )
    names.extend(
        [
            "visual_center_gradient_x_mean",
            "visual_center_gradient_y_mean",
            "visual_side_gradient_x_mean",
            "visual_side_gradient_y_mean",
        ]
    )
    return names


def visual_feature_names(settings: VisualFeatureSettings) -> list[str]:
    base = _base_feature_names(settings)
    return [
        *base,
        *(f"delta_{name}" for name in base),
        *(f"abs_delta_{name}" for name in base),
    ]


def _frame_features(frame: np.ndarray, settings: VisualFeatureSettings) -> np.ndarray:
    values = frame.astype(np.float32) / 255.0
    gradient_x = np.abs(np.diff(values, axis=1))
    gradient_y = np.abs(np.diff(values, axis=0))
    features = [
        float(values.mean()),
        float(values.std()),
        float(gradient_x.mean()),
        float(gradient_x.std()),
        float(gradient_y.mean()),
        float(gradient_y.std()),
    ]
    for threshold in settings.edge_thresholds:
        normalized = threshold / 255.0
        features.extend(
            [
                float((gradient_x >= normalized).mean()),
                float((gradient_y >= normalized).mean()),
            ]
        )
    ratio_threshold = 32.0 / 255.0
    for x_band, y_band in zip(
        np.array_split(gradient_x, settings.row_bins, axis=0),
        np.array_split(gradient_y, settings.row_bins, axis=0),
        strict=True,
    ):
        features.extend(
            [
                float(x_band.mean()),
                float(y_band.mean()),
                float((x_band >= ratio_threshold).mean()),
                float((y_band >= ratio_threshold).mean()),
            ]
        )
    for x_band, y_band in zip(
        np.array_split(gradient_x, settings.column_bins, axis=1),
        np.array_split(gradient_y, settings.column_bins, axis=1),
        strict=True,
    ):
        features.extend(
            [
                float(x_band.mean()),
                float(y_band.mean()),
                float((x_band >= ratio_threshold).mean()),
                float((y_band >= ratio_threshold).mean()),
            ]
        )
    center_left = settings.width // 4
    center_right = settings.width - center_left
    center_x = gradient_x[:, center_left : center_right - 1]
    center_y = gradient_y[:, center_left:center_right]
    side_x = np.concatenate(
        (gradient_x[:, :center_left], gradient_x[:, center_right - 1 :]), axis=1
    )
    side_y = np.concatenate(
        (gradient_y[:, :center_left], gradient_y[:, center_right:]), axis=1
    )
    features.extend(
        [
            float(center_x.mean()),
            float(center_y.mean()),
            float(side_x.mean()),
            float(side_y.mean()),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_existing(directory: Path, base_meta: dict) -> dict:
    meta_path = directory / "visual_meta.json"
    feature_path = directory / "visual_features.npy"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("format") != VISUAL_FEATURE_FORMAT
        or meta.get("version") != VISUAL_FEATURE_VERSION
        or not meta.get("completed")
    ):
        raise ValueError(f"unsupported visual feature cache: {directory}")
    if meta.get("source_sha256") != base_meta.get("source_sha256"):
        raise ValueError(f"visual and compressed feature sources differ: {directory}")
    if int(meta.get("frame_count", -1)) != int(base_meta["packet_count"]):
        raise ValueError(f"visual and compressed frame counts differ: {directory}")
    if file_sha256(feature_path) != meta.get("artifact_sha256"):
        raise ValueError(f"visual feature fingerprint mismatch: {feature_path}")
    features = np.load(feature_path, mmap_mode="r")
    if features.shape != (int(meta["frame_count"]), len(meta["feature_names"])):
        raise ValueError(f"visual feature shape disagrees with metadata: {directory}")
    return meta


def extract_visual_feature_cache(
    feature_dir: Path,
    *,
    settings: VisualFeatureSettings = VisualFeatureSettings(),
    ffmpeg: str = "ffmpeg",
    overwrite: bool = False,
) -> dict:
    """Decode only the bottom ROI and store compact edge statistics beside H.264 features."""
    directory = feature_dir.expanduser().resolve()
    base_meta_path = directory / "meta.json"
    base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    source = Path(base_meta["source"]).expanduser().resolve()
    if file_sha256(source) != base_meta.get("source_sha256"):
        raise ValueError(f"compressed feature source fingerprint is stale: {source}")
    meta_path = directory / "visual_meta.json"
    feature_path = directory / "visual_features.npy"
    if meta_path.exists() or feature_path.exists():
        if not (meta_path.exists() and feature_path.exists()):
            raise ValueError(f"incomplete visual feature cache: {directory}")
        if not overwrite:
            existing = _validate_existing(directory, base_meta)
            if existing.get("settings") != settings.to_dict():
                raise ValueError(f"visual feature settings changed: {directory}")
            return existing

    frame_count = int(base_meta["packet_count"])
    names = visual_feature_names(settings)
    base_count = len(_base_feature_names(settings))
    partial_path = directory / "visual_features.partial.npy"
    if partial_path.exists():
        partial_path.unlink()
    output = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, len(names)),
    )
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(source),
        "-vf",
        (
            "crop=iw:floor(ih*0.2):0:floor(ih*0.8),"
            f"scale={settings.width}:{settings.height}:flags=area,format=gray"
        ),
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("failed to open FFmpeg pipes")
    frame_bytes = settings.width * settings.height
    previous = np.zeros((base_count,), dtype=np.float32)
    try:
        for index in range(frame_count):
            payload = _read_exact(process.stdout, frame_bytes)
            if len(payload) != frame_bytes:
                raise ValueError(
                    f"decoded {index} visual frames, expected {frame_count}: {source}"
                )
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                settings.height, settings.width
            )
            current = _frame_features(frame, settings)
            delta = current - previous if index else np.zeros_like(current)
            output[index] = np.concatenate((current, delta, np.abs(delta)))
            previous = current
        if process.stdout.read(1):
            raise ValueError(f"decoded more than {frame_count} visual frames: {source}")
        error = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        process.stdout.close()
        process.stderr.close()
        if return_code != 0:
            raise RuntimeError(error or f"FFmpeg exited with status {return_code}")
        output.flush()
        del output
        partial_path.replace(feature_path)
    except BaseException:
        process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()
        del output
        if partial_path.exists():
            partial_path.unlink()
        raise

    meta = {
        "format": VISUAL_FEATURE_FORMAT,
        "version": VISUAL_FEATURE_VERSION,
        "completed": True,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "source_sha256": base_meta["source_sha256"],
        "frame_count": frame_count,
        "feature_names": names,
        "settings": settings.to_dict(),
        "artifact_sha256": file_sha256(feature_path),
        "decode_contract": {
            "pixel_decode": True,
            "roi": "bottom_20_percent",
            "stored_pixels": False,
        },
    }
    temporary_meta = directory / "visual_meta.partial.json"
    temporary_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_meta.replace(meta_path)
    return meta
