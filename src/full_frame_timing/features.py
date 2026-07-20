from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from h264_timing.dataset import FeatureCache
from h264_timing.hashing import file_sha256

from . import FEATURE_FORMAT, FEATURE_VERSION, INPUT_DOMAIN


@dataclass(frozen=True)
class FullFrameFeatureSettings:
    """Compact statistics computed from every pixel in the decoded frame."""

    width: int = 256
    height: int = 144
    tile_rows: int = 8
    tile_columns: int = 8
    edge_thresholds: tuple[int, ...] = (8, 16, 32, 64)
    row_bins: int = 18
    fine_column_bins: int = 32
    temporal_lags: tuple[int, ...] = (1, 4, 8)
    schema_version: int = 5

    def __post_init__(self) -> None:
        if self.width < 32 or self.height < 32:
            raise ValueError("full-frame feature dimensions must be at least 32x32")
        if self.tile_rows < 2 or self.height % self.tile_rows != 0:
            raise ValueError("tile rows must divide the decoded frame height")
        if self.tile_columns < 2 or self.width % self.tile_columns != 0:
            raise ValueError("tile columns must divide the decoded frame width")
        if self.height // self.tile_rows < 3 or self.width // self.tile_columns < 3:
            raise ValueError("each full-frame feature tile must be at least 3x3")
        if not self.edge_thresholds or any(
            threshold <= 0 or threshold >= 256
            for threshold in self.edge_thresholds
        ):
            raise ValueError("edge thresholds must be bytes in (0,256)")
        if self.schema_version not in {1, 2, 3, 4, 5}:
            raise ValueError(
                "full-frame feature schema version must be 1, 2, 3, 4, or 5"
            )
        if self.schema_version >= 2:
            if self.row_bins < 2 or self.height % self.row_bins != 0:
                raise ValueError("row bins must divide the decoded frame height")
            if (
                not self.temporal_lags
                or any(lag <= 0 for lag in self.temporal_lags)
                or tuple(sorted(set(self.temporal_lags))) != self.temporal_lags
            ):
                raise ValueError(
                    "temporal lags must be unique positive integers in ascending order"
                )
        if self.schema_version >= 5 and (
            self.fine_column_bins < self.tile_columns
            or self.width % self.fine_column_bins != 0
            or self.width // self.fine_column_bins < 3
        ):
            raise ValueError(
                "fine column bins must be at least tile columns, divide width, and keep cells at least 3 pixels wide"
            )

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["edge_thresholds"] = list(self.edge_thresholds)
        values["temporal_lags"] = list(self.temporal_lags)
        if self.schema_version == 1:
            # Version-one checkpoints predate these keys. Omitting them keeps
            # their extraction settings byte-for-byte compatible.
            for name in ("row_bins", "temporal_lags", "schema_version"):
                values.pop(name)
        if self.schema_version < 5:
            values.pop("fine_column_bins")
        return values

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> FullFrameFeatureSettings:
        parsed = dict(values)
        parsed.setdefault("schema_version", 1)
        if "edge_thresholds" in parsed:
            parsed["edge_thresholds"] = tuple(parsed["edge_thresholds"])  # type: ignore[arg-type]
        if "temporal_lags" in parsed:
            parsed["temporal_lags"] = tuple(parsed["temporal_lags"])  # type: ignore[arg-type]
        return cls(**parsed)  # type: ignore[arg-type]


@dataclass(frozen=True)
class FrameTimeline:
    timestamps: np.ndarray
    durations: np.ndarray
    stream: dict[str, object]

    def __post_init__(self) -> None:
        if self.timestamps.ndim != 1 or self.durations.ndim != 1:
            raise ValueError("frame timestamps and durations must be one-dimensional")
        if len(self.timestamps) == 0 or len(self.timestamps) != len(self.durations):
            raise ValueError("frame timeline arrays must have the same non-zero length")
        if not np.isfinite(self.timestamps).all() or not np.isfinite(self.durations).all():
            raise ValueError("frame timeline must contain only finite values")
        if np.any(np.diff(self.timestamps) <= 0.0):
            raise ValueError("frame timestamps must be strictly increasing")
        if np.any(self.durations < 0.0):
            raise ValueError("frame durations must be non-negative")


@dataclass(frozen=True)
class _TemporalFrameState:
    values: np.ndarray
    strong_edges: np.ndarray
    text_contrast: np.ndarray


_TILE_METRICS = (
    "gray_std",
    "gradient_x_mean",
    "gradient_y_mean",
    "gradient_x_ratio_032",
    "gradient_y_ratio_032",
    "gradient_x_ratio_064",
    "gradient_y_ratio_064",
)
_POOL_STATISTICS = ("mean", "std", "max", "p50", "p75", "p90", "p95")
_ROW_TILE_METRICS = (
    "gradient_x_mean",
    "gradient_y_mean",
    "gradient_x_ratio_032",
    "gradient_y_ratio_032",
    "text_contrast_ratio",
)
_ROW_TILE_POOL_STATISTICS = ("max", "p75", "p90")
_FINE_ROW_TILE_POOL_STATISTICS = (
    "max",
    "top2_mean",
    "center_max",
    "side_max",
)


def _base_feature_names(settings: FullFrameFeatureSettings) -> list[str]:
    names = [
        "full_frame_gray_mean",
        "full_frame_gray_std",
        "full_frame_gradient_x_mean",
        "full_frame_gradient_x_std",
        "full_frame_gradient_y_mean",
        "full_frame_gradient_y_std",
    ]
    for threshold in settings.edge_thresholds:
        names.extend(
            (
                f"full_frame_gradient_x_ratio_{threshold:03d}",
                f"full_frame_gradient_y_ratio_{threshold:03d}",
            )
        )
    for metric in _TILE_METRICS:
        names.extend(
            f"full_frame_tile_{metric}_{statistic}"
            for statistic in _POOL_STATISTICS
        )
    if settings.schema_version == 1:
        return names
    for index in range(settings.row_bins):
        names.extend(
            (
                f"full_frame_row_{index:02d}_gradient_x_mean",
                f"full_frame_row_{index:02d}_gradient_y_mean",
                f"full_frame_row_{index:02d}_gradient_x_ratio_032",
                f"full_frame_row_{index:02d}_gradient_y_ratio_032",
                f"full_frame_row_{index:02d}_text_contrast_ratio",
            )
        )
    if settings.schema_version >= 3:
        for index in range(settings.row_bins):
            for metric in _ROW_TILE_METRICS:
                names.extend(
                    f"full_frame_row_{index:02d}_tile_{metric}_{statistic}"
                    for statistic in _ROW_TILE_POOL_STATISTICS
                )
    if settings.schema_version >= 4:
        for row_index in range(settings.row_bins):
            for column_index in range(settings.tile_columns):
                names.extend(
                    f"full_frame_row_{row_index:02d}_column_{column_index:02d}_{metric}"
                    for metric in _ROW_TILE_METRICS
                )
    if settings.schema_version >= 5:
        for row_index in range(settings.row_bins):
            for metric in _ROW_TILE_METRICS:
                names.extend(
                    f"full_frame_fine_row_{row_index:02d}_{metric}_{statistic}"
                    for statistic in _FINE_ROW_TILE_POOL_STATISTICS
                )
    names.extend(
        (
            "full_frame_center_gradient_x_mean",
            "full_frame_center_gradient_y_mean",
            "full_frame_side_gradient_x_mean",
            "full_frame_side_gradient_y_mean",
        )
    )
    for lag in settings.temporal_lags:
        prefix = f"full_frame_lag_{lag:03d}"
        names.extend(
            (
                f"{prefix}_pixel_delta_mean",
                f"{prefix}_pixel_delta_ratio_008",
                f"{prefix}_pixel_delta_ratio_016",
                f"{prefix}_pixel_delta_ratio_032",
                f"{prefix}_stable_edge_ratio",
                f"{prefix}_stable_edge_row_max",
                f"{prefix}_stable_edge_row_p90",
                f"{prefix}_stable_text_ratio",
                f"{prefix}_stable_text_row_max",
                f"{prefix}_stable_text_row_p90",
            )
        )
    return names


def full_frame_feature_names(settings: FullFrameFeatureSettings) -> list[str]:
    base = _base_feature_names(settings)
    return [
        *base,
        *(f"delta_{name}" for name in base),
        *(f"abs_delta_{name}" for name in base),
    ]


def _text_contrast_mask(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    neighborhoods = tuple(
        padded[1 + dy : 1 + dy + values.shape[0], 1 + dx : 1 + dx + values.shape[1]]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    )
    local_minimum = np.minimum.reduce(neighborhoods)
    local_maximum = np.maximum.reduce(neighborhoods)
    return ((values >= 0.70) & (local_minimum <= 0.30)) | (
        (values <= 0.30) & (local_maximum >= 0.70)
    )


def _row_distribution(values: np.ndarray, row_bins: int) -> tuple[float, float]:
    ratios = np.asarray(
        [float(band.mean()) for band in np.array_split(values, row_bins, axis=0)],
        dtype=np.float32,
    )
    return float(ratios.max()), float(np.quantile(ratios, 0.90))


def _temporal_frame_state(
    values: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
) -> _TemporalFrameState:
    ratio_threshold = 32.0 / 255.0
    gradient_x_full = np.pad(gradient_x, ((0, 0), (0, 1)), mode="edge")
    gradient_y_full = np.pad(gradient_y, ((0, 1), (0, 0)), mode="edge")
    return _TemporalFrameState(
        values=values,
        strong_edges=(
            np.maximum(gradient_x_full, gradient_y_full) >= ratio_threshold
        ),
        text_contrast=_text_contrast_mask(values),
    )


def _frame_features(
    frame: np.ndarray,
    settings: FullFrameFeatureSettings,
    *,
    history: tuple[_TemporalFrameState, ...] = (),
) -> tuple[np.ndarray, _TemporalFrameState | None]:
    """Extract whole-frame spatial and causal temporal statistics."""

    if frame.shape != (settings.height, settings.width):
        raise ValueError(
            "decoded frame shape does not match full-frame feature settings"
        )
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
            (
                float((gradient_x >= normalized).mean()),
                float((gradient_y >= normalized).mean()),
            )
        )

    tile_height = settings.height // settings.tile_rows
    tile_width = settings.width // settings.tile_columns
    tiles = values.reshape(
        settings.tile_rows,
        tile_height,
        settings.tile_columns,
        tile_width,
    ).transpose(0, 2, 1, 3)
    tile_gradient_x = np.abs(np.diff(tiles, axis=3))
    tile_gradient_y = np.abs(np.diff(tiles, axis=2))
    threshold_32 = 32.0 / 255.0
    threshold_64 = 64.0 / 255.0
    metrics = np.stack(
        (
            tiles.std(axis=(2, 3)),
            tile_gradient_x.mean(axis=(2, 3)),
            tile_gradient_y.mean(axis=(2, 3)),
            (tile_gradient_x >= threshold_32).mean(axis=(2, 3)),
            (tile_gradient_y >= threshold_32).mean(axis=(2, 3)),
            (tile_gradient_x >= threshold_64).mean(axis=(2, 3)),
            (tile_gradient_y >= threshold_64).mean(axis=(2, 3)),
        ),
        axis=0,
    ).reshape(len(_TILE_METRICS), -1)
    ordered = np.sort(metrics, axis=1)
    last = ordered.shape[1] - 1
    quantile_indices = tuple(
        min(last, max(0, int(round(last * quantile))))
        for quantile in (0.50, 0.75, 0.90, 0.95)
    )
    for metric, sorted_metric in zip(metrics, ordered, strict=True):
        features.extend(
            (
                float(metric.mean()),
                float(metric.std()),
                float(sorted_metric[-1]),
                *(float(sorted_metric[index]) for index in quantile_indices),
            )
        )
    if settings.schema_version == 1:
        return np.asarray(features, dtype=np.float32), None

    current_state = _temporal_frame_state(values, gradient_x, gradient_y)
    text_contrast = current_state.text_contrast
    ratio_threshold = 32.0 / 255.0
    for x_band, y_band, text_band in zip(
        np.array_split(gradient_x, settings.row_bins, axis=0),
        np.array_split(gradient_y, settings.row_bins, axis=0),
        np.array_split(text_contrast, settings.row_bins, axis=0),
        strict=True,
    ):
        features.extend(
            (
                float(x_band.mean()),
                float(y_band.mean()),
                float((x_band >= ratio_threshold).mean()),
                float((y_band >= ratio_threshold).mean()),
                float(text_band.mean()),
            )
        )

    if settings.schema_version >= 3:
        row_height = settings.height // settings.row_bins
        column_width = settings.width // settings.tile_columns
        row_tiles = values.reshape(
            settings.row_bins,
            row_height,
            settings.tile_columns,
            column_width,
        ).transpose(0, 2, 1, 3)
        row_tile_gradient_x = np.abs(np.diff(row_tiles, axis=3))
        row_tile_gradient_y = np.abs(np.diff(row_tiles, axis=2))
        text_tiles = text_contrast.reshape(
            settings.row_bins,
            row_height,
            settings.tile_columns,
            column_width,
        ).transpose(0, 2, 1, 3)
        row_tile_metrics = np.stack(
            (
                row_tile_gradient_x.mean(axis=(2, 3)),
                row_tile_gradient_y.mean(axis=(2, 3)),
                (row_tile_gradient_x >= ratio_threshold).mean(axis=(2, 3)),
                (row_tile_gradient_y >= ratio_threshold).mean(axis=(2, 3)),
                text_tiles.mean(axis=(2, 3)),
            ),
            axis=-1,
        )
        ordered_row_metrics = np.sort(row_tile_metrics, axis=1)
        last_column = ordered_row_metrics.shape[1] - 1
        p75_column = min(last_column, max(0, int(round(last_column * 0.75))))
        p90_column = min(last_column, max(0, int(round(last_column * 0.90))))
        pooled_row_metrics = np.stack(
            (
                ordered_row_metrics[:, -1, :],
                ordered_row_metrics[:, p75_column, :],
                ordered_row_metrics[:, p90_column, :],
            ),
            axis=-1,
        )
        features.extend(float(value) for value in pooled_row_metrics.reshape(-1))
        if settings.schema_version >= 4:
            # Retain cell position as well as the location-neutral row pools.
            # A narrow subtitle can otherwise be hidden by stronger unrelated
            # edges elsewhere in the same row band.
            features.extend(float(value) for value in row_tile_metrics.reshape(-1))
        if settings.schema_version >= 5:
            fine_column_width = settings.width // settings.fine_column_bins
            fine_row_tiles = values.reshape(
                settings.row_bins,
                row_height,
                settings.fine_column_bins,
                fine_column_width,
            ).transpose(0, 2, 1, 3)
            fine_gradient_x = np.abs(np.diff(fine_row_tiles, axis=3))
            fine_gradient_y = np.abs(np.diff(fine_row_tiles, axis=2))
            fine_text_tiles = text_contrast.reshape(
                settings.row_bins,
                row_height,
                settings.fine_column_bins,
                fine_column_width,
            ).transpose(0, 2, 1, 3)
            fine_metrics = np.stack(
                (
                    fine_gradient_x.mean(axis=(2, 3)),
                    fine_gradient_y.mean(axis=(2, 3)),
                    (fine_gradient_x >= ratio_threshold).mean(axis=(2, 3)),
                    (fine_gradient_y >= ratio_threshold).mean(axis=(2, 3)),
                    fine_text_tiles.mean(axis=(2, 3)),
                ),
                axis=-1,
            )
            ordered_fine_metrics = np.sort(fine_metrics, axis=1)
            center_left = settings.fine_column_bins // 4
            center_right = settings.fine_column_bins - center_left
            center_fine_metrics = fine_metrics[:, center_left:center_right, :]
            side_fine_metrics = np.concatenate(
                (
                    fine_metrics[:, :center_left, :],
                    fine_metrics[:, center_right:, :],
                ),
                axis=1,
            )
            pooled_fine_metrics = np.stack(
                (
                    ordered_fine_metrics[:, -1, :],
                    ordered_fine_metrics[:, -2:, :].mean(axis=1),
                    center_fine_metrics.max(axis=1),
                    side_fine_metrics.max(axis=1),
                ),
                axis=-1,
            )
            features.extend(
                float(value) for value in pooled_fine_metrics.reshape(-1)
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
        (
            float(center_x.mean()),
            float(center_y.mean()),
            float(side_x.mean()),
            float(side_y.mean()),
        )
    )

    for lag in settings.temporal_lags:
        if len(history) < lag:
            features.extend((0.0,) * 10)
            continue
        previous = history[-lag]
        pixel_delta = np.abs(values - previous.values)
        stable_edges = current_state.strong_edges & previous.strong_edges
        stable_text = text_contrast & previous.text_contrast
        stable_edge_max, stable_edge_p90 = _row_distribution(
            stable_edges, settings.row_bins
        )
        stable_text_max, stable_text_p90 = _row_distribution(
            stable_text, settings.row_bins
        )
        features.extend(
            (
                float(pixel_delta.mean()),
                float((pixel_delta >= 8.0 / 255.0).mean()),
                float((pixel_delta >= 16.0 / 255.0).mean()),
                float((pixel_delta >= 32.0 / 255.0).mean()),
                float(stable_edges.mean()),
                stable_edge_max,
                stable_edge_p90,
                float(stable_text.mean()),
                stable_text_max,
                stable_text_p90,
            )
        )
    return np.asarray(features, dtype=np.float32), current_state


def frame_features(
    frame: np.ndarray,
    settings: FullFrameFeatureSettings,
    *,
    history: tuple[np.ndarray, ...] = (),
) -> np.ndarray:
    """Extract features directly; extraction caches temporal state separately."""

    states: list[_TemporalFrameState] = []
    if settings.schema_version >= 2:
        for previous in history:
            values = previous.astype(np.float32) / 255.0
            states.append(
                _temporal_frame_state(
                    values,
                    np.abs(np.diff(values, axis=1)),
                    np.abs(np.diff(values, axis=0)),
                )
            )
    features, _ = _frame_features(frame, settings, history=tuple(states))
    return features


def _required_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise FileNotFoundError(f"required executable not found: {name}")
    return binary


def _optional_float(value: object) -> float | None:
    if value in {None, "", "N/A"}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def probe_frame_timeline(video: Path, *, ffprobe: str = "ffprobe") -> FrameTimeline:
    """Read presentation-order frame timing for any FFmpeg-supported video codec."""

    source = video.expanduser().resolve()
    command = [
        _required_binary(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        (
            "stream=codec_name,profile,level,width,height,r_frame_rate,avg_frame_rate,"
            "time_base,start_time,duration,nb_frames,bit_rate,field_order:"
            "frame=best_effort_timestamp_time,pkt_duration_time,duration_time:"
            "format=format_name,start_time,duration,size"
        ),
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    frames = payload.get("frames", [])
    if not streams or not frames:
        raise ValueError(f"video contains no decodable frames: {source}")
    raw_timestamps = [
        _optional_float(frame.get("best_effort_timestamp_time")) for frame in frames
    ]
    if any(value is None for value in raw_timestamps):
        raise ValueError(f"video frame timeline has missing timestamps: {source}")
    timestamps = np.asarray(raw_timestamps, dtype=np.float64)
    timestamps -= timestamps[0]
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"video frame timeline is not strictly increasing: {source}")
    positive_steps = np.diff(timestamps)
    fallback_duration = (
        float(np.median(positive_steps)) if positive_steps.size else 1.0 / 30.0
    )
    durations = np.asarray(
        [
            _optional_float(frame.get("pkt_duration_time"))
            or _optional_float(frame.get("duration_time"))
            or (
                float(positive_steps[index])
                if index < len(positive_steps)
                else fallback_duration
            )
            for index, frame in enumerate(frames)
        ],
        dtype=np.float64,
    )
    stream = dict(streams[0])
    format_info = dict(payload.get("format", {}))
    stream.update(
        {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "start_time_seconds": (
                _optional_float(stream.get("start_time"))
                or _optional_float(format_info.get("start_time"))
                or 0.0
            ),
            "duration_seconds": (
                _optional_float(stream.get("duration"))
                or _optional_float(format_info.get("duration"))
            ),
            "frame_count": len(frames),
            "format_name": format_info.get("format_name"),
            "file_size": int(format_info.get("size") or source.stat().st_size),
        }
    )
    return FrameTimeline(timestamps=timestamps, durations=durations, stream=stream)


def timeline_from_cache(cache: FeatureCache) -> FrameTimeline:
    return FrameTimeline(
        timestamps=np.asarray(cache.timestamps, dtype=np.float64).copy(),
        durations=np.asarray(cache.durations, dtype=np.float64).copy(),
        stream=dict(cache.meta["stream"]),
    )


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


def _validate_existing(
    directory: Path,
    *,
    source_sha256: str,
    settings: FullFrameFeatureSettings,
) -> dict[str, object]:
    cache = FeatureCache(directory)
    try:
        if cache.meta.get("format") != FEATURE_FORMAT:
            raise ValueError(f"cache is not a full-frame feature cache: {directory}")
        if cache.meta.get("source_sha256") != source_sha256:
            raise ValueError(f"full-frame cache source changed: {directory}")
        if cache.meta.get("full_frame_feature_settings") != settings.to_dict():
            raise ValueError(f"full-frame feature settings changed: {directory}")
        if cache.feature_names != full_frame_feature_names(settings):
            raise ValueError(f"full-frame feature schema changed: {directory}")
        return dict(cache.meta)
    finally:
        cache.release()


def extract_full_frame_feature_cache(
    video: Path,
    output_dir: Path,
    *,
    settings: FullFrameFeatureSettings = FullFrameFeatureSettings(),
    timeline: FrameTimeline | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    overwrite: bool = False,
) -> dict[str, object]:
    """Decode every source pixel through an aspect-preserving full-frame transform."""

    source = video.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source video not found: {source}")
    directory = output_dir.expanduser().resolve()
    source_sha256 = file_sha256(source)
    meta_path = directory / "meta.json"
    if meta_path.exists() and not overwrite:
        return _validate_existing(
            directory,
            source_sha256=source_sha256,
            settings=settings,
        )
    if directory.exists() and overwrite:
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    frame_timeline = timeline or probe_frame_timeline(source, ffprobe=ffprobe)
    frame_count = len(frame_timeline.timestamps)
    names = full_frame_feature_names(settings)
    base_count = len(_base_feature_names(settings))
    feature_partial = directory / "features.partial.npy"
    features = np.lib.format.open_memmap(
        feature_partial,
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, len(names)),
    )
    filter_graph = (
        f"scale={settings.width}:{settings.height}:"
        "force_original_aspect_ratio=decrease:flags=area,"
        f"pad={settings.width}:{settings.height}:(ow-iw)/2:(oh-ih)/2:black,"
        "setsar=1,format=gray"
    )
    command = [
        _required_binary(ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        filter_graph,
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("failed to open FFmpeg full-frame decode pipes")
    frame_bytes = settings.width * settings.height
    previous = np.zeros((base_count,), dtype=np.float32)
    history: list[_TemporalFrameState] = []
    maximum_lag = max(settings.temporal_lags, default=0)
    try:
        for index in range(frame_count):
            payload = _read_exact(process.stdout, frame_bytes)
            if len(payload) != frame_bytes:
                raise ValueError(
                    f"decoded {index} frames, expected {frame_count}: {source}"
                )
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(
                settings.height, settings.width
            )
            current, current_state = _frame_features(
                frame, settings, history=tuple(history)
            )
            delta = current - previous if index else np.zeros_like(current)
            features[index] = np.concatenate((current, delta, np.abs(delta)))
            previous = current
            if maximum_lag and current_state is not None:
                history.append(current_state)
                if len(history) > maximum_lag:
                    history.pop(0)
        if process.stdout.read(1):
            raise ValueError(f"decoded more than {frame_count} frames: {source}")
        error = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        process.stdout.close()
        process.stderr.close()
        if return_code != 0:
            raise RuntimeError(error or f"FFmpeg exited with status {return_code}")
        features.flush()
        feature_path = directory / "features.npy"
        feature_partial.replace(feature_path)

        token_path = directory / "tokens.npy"
        np.save(token_path, np.zeros((frame_count, 1), dtype=np.uint8))
        timestamp_path = directory / "timestamps.npy"
        duration_path = directory / "durations.npy"
        np.save(timestamp_path, frame_timeline.timestamps.astype(np.float64))
        np.save(duration_path, frame_timeline.durations.astype(np.float64))
        del features
    except BaseException:
        process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()
        del features
        if feature_partial.exists():
            feature_partial.unlink()
        raise

    spatial_contract = {
        "requested_roi": "full_frame",
        "spatial_mode": "decoded_full_frame",
        "implemented_feature_scope": "all_display_pixels",
        "exact_pixel_roi": True,
        "source_coordinate_space": {
            "x": 0,
            "y": 0,
            "width": int(frame_timeline.stream["width"]),
            "height": int(frame_timeline.stream["height"]),
        },
        "model_coordinate_space": {
            "width": settings.width,
            "height": settings.height,
            "resize": "preserve_aspect_ratio_then_letterbox",
        },
        "spatial_pooling": (
            "all_position_row_column_cells_with_fine_local_column_pools"
            if settings.schema_version >= 5
            else (
                "all_position_row_column_cells_with_local_column_pools"
                if settings.schema_version >= 4
                else (
                    "all_position_row_bands_with_local_column_pools"
                    if settings.schema_version >= 3
                    else (
                        "location_neutral_sorted_tiles_plus_all_position_row_bands"
                        if settings.schema_version >= 2
                        else "location_neutral_sorted_tile_statistics"
                    )
                )
            )
        ),
    }
    meta: dict[str, object] = {
        "format": FEATURE_FORMAT,
        "version": FEATURE_VERSION,
        "completed": True,
        "created_at": datetime.now(UTC).isoformat(),
        "input_domain": INPUT_DOMAIN,
        "source": str(source),
        "source_sha256": source_sha256,
        "source_id": f"sha256:{source_sha256}",
        "frame_count": frame_count,
        # The shared temporal dataset reader uses this compatibility key.
        "packet_count": frame_count,
        "feature_names": names,
        "feature_settings": {
            "token_count": 1,
            "payload_tail_ratio": 1.0,
            "input_domain": INPUT_DOMAIN,
        },
        "full_frame_feature_settings": settings.to_dict(),
        "stream": frame_timeline.stream,
        "time_range_seconds": [
            float(frame_timeline.timestamps[0]),
            float(frame_timeline.timestamps[-1]),
        ],
        "spatial_contract": spatial_contract,
        "decode_contract": {
            "pixel_decode": True,
            "roi": "full_frame",
            "stored_pixels": False,
        },
        "artifact_sha256": {
            "features.npy": file_sha256(directory / "features.npy"),
            "tokens.npy": file_sha256(directory / "tokens.npy"),
            "timestamps.npy": file_sha256(directory / "timestamps.npy"),
            "durations.npy": file_sha256(directory / "durations.npy"),
        },
    }
    temporary_meta = directory / "meta.partial.json"
    temporary_meta.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_meta.replace(meta_path)
    return meta
