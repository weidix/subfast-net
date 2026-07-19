from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from . import FEATURE_FORMAT, FEATURE_VERSION
from .hashing import file_sha256
from .labels import (
    SubtitleInterval,
    boundary_event_targets_from_intervals,
    read_intervals,
    segment_targets_from_intervals,
)
from .visual import VISUAL_FEATURE_FORMAT, VISUAL_FEATURE_VERSION


@dataclass(frozen=True)
class ManifestRecord:
    video_id: str
    source_group: str
    feature_dir: Path
    labels_path: Path
    split: str
    source_time_offset_seconds: float = 0.0
    synthesis_audit_path: Path | None = None
    pair_id: str | None = None
    signal_validation_role: Literal[
        "source_timing", "subtitle_signal", "clean_control"
    ] = "source_timing"


class CombinedFeatureArray:
    def __init__(self, arrays: tuple[np.ndarray, ...]) -> None:
        if not arrays or len({len(array) for array in arrays}) != 1:
            raise ValueError("combined feature arrays must have the same non-empty length")
        self.arrays = arrays
        self.shape = (len(arrays[0]), sum(array.shape[1] for array in arrays))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        return np.concatenate(
            tuple(np.asarray(array[key]) for array in self.arrays), axis=-1
        )


class FeatureCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.expanduser().resolve()
        self.meta = json.loads((self.directory / "meta.json").read_text(encoding="utf-8"))
        if self.meta.get("format") != FEATURE_FORMAT or self.meta.get("version") != FEATURE_VERSION:
            raise ValueError(f"unsupported feature cache: {self.directory}")
        if not self.meta.get("completed"):
            raise ValueError(f"incomplete feature cache: {self.directory}")
        source_sha256 = str(self.meta.get("source_sha256", ""))
        if len(source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_sha256
        ):
            raise ValueError(f"feature cache lacks a full source fingerprint: {self.directory}")
        expected_artifacts = {
            "features.npy",
            "tokens.npy",
            "timestamps.npy",
            "durations.npy",
        }
        artifact_sha256 = self.meta.get("artifact_sha256", {})
        if not isinstance(artifact_sha256, dict) or set(artifact_sha256) != expected_artifacts:
            raise ValueError(f"feature cache lacks complete artifact fingerprints: {self.directory}")
        for name in sorted(expected_artifacts):
            if file_sha256(self.directory / name) != artifact_sha256[name]:
                raise ValueError(f"feature cache artifact fingerprint mismatch: {self.directory / name}")
        compressed_features = np.load(self.directory / "features.npy", mmap_mode="r")
        self.compressed_features = compressed_features
        self.tokens = np.load(self.directory / "tokens.npy", mmap_mode="r")
        self.timestamps = np.load(self.directory / "timestamps.npy", mmap_mode="r")
        self.durations = np.load(self.directory / "durations.npy", mmap_mode="r")
        count = int(self.meta["packet_count"])
        if count <= 0:
            raise ValueError(f"feature cache contains no packets: {self.directory}")
        if not (
            len(compressed_features)
            == len(self.tokens)
            == len(self.timestamps)
            == len(self.durations)
            == count
        ):
            raise ValueError(f"feature cache arrays disagree on packet count: {self.directory}")
        if compressed_features.shape[1] != len(self.meta["feature_names"]):
            raise ValueError(f"feature array width disagrees with metadata: {self.directory}")
        if self.tokens.shape[1] != int(self.meta["feature_settings"]["token_count"]):
            raise ValueError(f"token array width disagrees with metadata: {self.directory}")
        if np.any(np.diff(self.timestamps) <= 0.0):
            raise ValueError(f"feature-cache timestamps must be strictly increasing: {self.directory}")
        self.visual_meta: dict | None = None
        self.visual_features: np.ndarray | None = None
        arrays: tuple[np.ndarray, ...] = (compressed_features,)
        visual_meta_path = self.directory / "visual_meta.json"
        visual_feature_path = self.directory / "visual_features.npy"
        if visual_meta_path.exists() != visual_feature_path.exists():
            raise ValueError(f"incomplete visual feature sidecar: {self.directory}")
        if visual_meta_path.exists():
            visual_meta = json.loads(visual_meta_path.read_text(encoding="utf-8"))
            if (
                visual_meta.get("format") != VISUAL_FEATURE_FORMAT
                or visual_meta.get("version") != VISUAL_FEATURE_VERSION
                or not visual_meta.get("completed")
            ):
                raise ValueError(f"unsupported visual feature sidecar: {self.directory}")
            if visual_meta.get("source_sha256") != source_sha256:
                raise ValueError(
                    f"visual and compressed feature sources differ: {self.directory}"
                )
            if int(visual_meta.get("frame_count", -1)) != count:
                raise ValueError(
                    f"visual and compressed frame counts differ: {self.directory}"
                )
            if file_sha256(visual_feature_path) != visual_meta.get("artifact_sha256"):
                raise ValueError(
                    f"visual feature fingerprint mismatch: {visual_feature_path}"
                )
            visual_features = np.load(visual_feature_path, mmap_mode="r")
            if visual_features.shape != (count, len(visual_meta["feature_names"])):
                raise ValueError(
                    f"visual feature shape disagrees with metadata: {self.directory}"
                )
            self.visual_meta = visual_meta
            self.visual_features = visual_features
            arrays = (compressed_features, visual_features)
        self.features = CombinedFeatureArray(arrays)

    @property
    def compressed_feature_names(self) -> list[str]:
        return list(self.meta["feature_names"])

    @property
    def visual_feature_names(self) -> list[str]:
        return (
            list(self.visual_meta["feature_names"])
            if self.visual_meta is not None
            else []
        )

    @property
    def feature_names(self) -> list[str]:
        visual_names = (
            list(self.visual_meta["feature_names"])
            if self.visual_meta is not None
            else []
        )
        return [*self.meta["feature_names"], *visual_names]

    @property
    def visual_feature_settings(self) -> dict | None:
        return (
            dict(self.visual_meta["settings"])
            if self.visual_meta is not None
            else None
        )

    @property
    def source_id(self) -> str:
        return str(self.meta["source_id"])

    def release(self) -> None:
        """Release mmap-backed arrays while retaining metadata needed by audits."""
        self.timestamps = np.asarray(self.timestamps).copy()
        self.durations = np.asarray(self.durations).copy()
        arrays: list[np.ndarray] = [self.tokens, *self.features.arrays]
        self._close_mmaps(arrays)

    def materialize(self) -> None:
        """Copy all cache arrays into process memory and close mmap handles."""
        arrays = tuple(np.asarray(array).copy() for array in self.features.arrays)
        tokens = np.asarray(self.tokens).copy()
        timestamps = np.asarray(self.timestamps).copy()
        durations = np.asarray(self.durations).copy()
        self._close_mmaps([self.tokens, *self.features.arrays])
        self.compressed_features = arrays[0]
        self.visual_features = arrays[1] if len(arrays) > 1 else None
        self.features = CombinedFeatureArray(arrays)
        self.tokens = tokens
        self.timestamps = timestamps
        self.durations = durations

    @staticmethod
    def _close_mmaps(arrays: list[np.ndarray]) -> None:
        for array in arrays:
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    @property
    def coverage_range_seconds(self) -> tuple[float, float]:
        if len(self.timestamps) == 0:
            return 0.0, 0.0
        positive_steps = np.diff(self.timestamps)
        positive_steps = positive_steps[positive_steps > 0]
        fallback = float(np.median(positive_steps)) if positive_steps.size else 1 / 30
        last_duration = float(self.durations[-1])
        end = float(self.timestamps[-1] + (last_duration if last_duration > 0 else fallback))
        return float(self.timestamps[0]), end


@dataclass
class LoadedRecord:
    record: ManifestRecord
    cache: FeatureCache
    intervals: list[SubtitleInterval]
    segment_targets: np.ndarray
    boundary_event_targets: np.ndarray


def _resolve_manifest_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def read_manifest(path: Path, *, split: str | None = None) -> list[ManifestRecord]:
    path = path.expanduser().resolve()
    all_records: list[ManifestRecord] = []
    video_ids: set[str] = set()
    feature_dirs: set[Path] = set()
    pair_roles: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            required = {"video_id", "source_group", "features", "labels", "split"}
            missing = sorted(required.difference(item))
            if missing:
                raise ValueError(
                    f"manifest line {line_number} is missing required fields: {', '.join(missing)}"
                )
            video_id = str(item["video_id"]).strip()
            source_group = str(item["source_group"]).strip()
            pair_id = str(item["pair_id"]).strip() if item.get("pair_id") else None
            role = str(item.get("signal_validation_role", "source_timing"))
            if not video_id or not source_group:
                raise ValueError(f"manifest line {line_number} has an empty identifier")
            if role not in {"source_timing", "subtitle_signal", "clean_control"}:
                raise ValueError(
                    f"invalid signal_validation_role on line {line_number}: {role}"
                )
            if role in {"subtitle_signal", "clean_control"} and pair_id is None:
                raise ValueError(
                    f"manifest line {line_number} requires pair_id for role {role}"
                )
            record = ManifestRecord(
                video_id=video_id,
                source_group=source_group,
                feature_dir=_resolve_manifest_path(path.parent, str(item["features"])),
                labels_path=_resolve_manifest_path(path.parent, str(item["labels"])),
                split=str(item["split"]),
                source_time_offset_seconds=float(item.get("source_time_offset_seconds", 0.0)),
                synthesis_audit_path=(
                    _resolve_manifest_path(path.parent, str(item["synthesis_audit"]))
                    if item.get("synthesis_audit")
                    else None
                ),
                pair_id=pair_id,
                signal_validation_role=role,  # type: ignore[arg-type]
            )
            if (
                not np.isfinite(record.source_time_offset_seconds)
                or record.source_time_offset_seconds < 0.0
            ):
                raise ValueError(
                    f"invalid source_time_offset_seconds on line {line_number}"
                )
            if record.split not in {"train", "val", "test"}:
                raise ValueError(f"invalid split on line {line_number}: {record.split}")
            if record.video_id in video_ids:
                raise ValueError(f"duplicate video_id in manifest: {record.video_id}")
            if record.feature_dir in feature_dirs:
                raise ValueError(f"duplicate feature cache in manifest: {record.feature_dir}")
            if record.pair_id is not None:
                pair_role = (
                    record.pair_id,
                    record.split,
                    record.signal_validation_role,
                )
                if pair_role in pair_roles:
                    raise ValueError(
                        "duplicate pair role in manifest: "
                        f"{record.pair_id}/{record.split}/{record.signal_validation_role}"
                    )
                pair_roles.add(pair_role)
            video_ids.add(record.video_id)
            feature_dirs.add(record.feature_dir)
            all_records.append(record)
    paired_records: dict[str, list[ManifestRecord]] = {}
    for record in all_records:
        if record.pair_id is not None:
            paired_records.setdefault(record.pair_id, []).append(record)
    if any(
        record.signal_validation_role == "clean_control" for record in all_records
    ):
        for pair_id, records in paired_records.items():
            roles = {record.signal_validation_role for record in records}
            if roles != {"subtitle_signal", "clean_control"} or len(records) != 2:
                raise ValueError(
                    f"paired manifest requires one signal and one clean record: {pair_id}"
                )
            if len({record.split for record in records}) != 1:
                raise ValueError(f"paired records use different splits: {pair_id}")
            if len({record.source_group for record in records}) != 1:
                raise ValueError(f"paired records use different source groups: {pair_id}")
            offsets = [record.source_time_offset_seconds for record in records]
            if abs(offsets[0] - offsets[1]) > 1e-6:
                raise ValueError(f"paired records use different source offsets: {pair_id}")
    return [record for record in all_records if split is None or record.split == split]


def load_records(
    records: list[ManifestRecord],
    *,
    boundary_event_sigma_seconds: float = 0.05,
    materialize: bool = False,
) -> list[LoadedRecord]:
    loaded: list[LoadedRecord] = []
    expected_names: list[str] | None = None
    for record in records:
        cache = FeatureCache(record.feature_dir)
        if expected_names is None:
            expected_names = cache.feature_names
        elif cache.feature_names != expected_names:
            raise ValueError("all feature caches in a run must use the same feature schema")
        all_intervals = read_intervals(record.labels_path)
        if record.signal_validation_role == "subtitle_signal" and not all_intervals:
            raise ValueError(f"subtitle-signal sample has no labels: {record.video_id}")
        if record.signal_validation_role == "clean_control" and all_intervals:
            raise ValueError(f"clean-control sample contains labels: {record.video_id}")
        if (
            record.signal_validation_role in {"subtitle_signal", "clean_control"}
            and record.synthesis_audit_path is None
        ):
            raise ValueError(f"paired signal-validation sample lacks audit: {record.video_id}")
        _validate_synthesis_audit(record, cache, all_intervals)
        intervals = intervals_inside_cache(cache, all_intervals)
        segment_targets = segment_targets_from_intervals(
            np.asarray(cache.timestamps),
            intervals,
        )
        boundary_event_targets = boundary_event_targets_from_intervals(
            np.asarray(cache.timestamps),
            intervals,
            sigma_seconds=boundary_event_sigma_seconds,
        )
        loaded.append(
            LoadedRecord(
                record,
                cache,
                intervals,
                segment_targets,
                boundary_event_targets,
            )
        )
        if materialize:
            cache.materialize()
    return loaded


def _validate_synthesis_audit(
    record: ManifestRecord,
    cache: FeatureCache,
    intervals: list[SubtitleInterval],
) -> None:
    if record.synthesis_audit_path is None:
        return
    audit = json.loads(record.synthesis_audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("format") != "h264_timing_synthetic_subtitle_segment"
        or audit.get("version") != 2
    ):
        raise ValueError(f"unsupported synthesis audit: {record.synthesis_audit_path}")
    audited_video = Path(audit["output"]["video"]).expanduser().resolve()
    audited_labels = Path(audit["output"]["labels"]).expanduser().resolve()
    cache_source = Path(cache.meta["source"]).expanduser().resolve()
    if audited_video != cache_source or audited_labels != record.labels_path:
        raise ValueError(f"synthesis audit paths do not match manifest/cache: {record.video_id}")
    if audit["output"].get("video_sha256") != cache.meta.get("source_sha256"):
        raise ValueError(f"synthesis audit and feature cache fingerprints differ: {record.video_id}")
    if audit["output"].get("labels_sha256") != file_sha256(record.labels_path):
        raise ValueError(f"synthesis audit label fingerprint mismatch: {record.video_id}")
    audited_offset = float(audit["source_timeline"]["start_seconds"])
    if abs(audited_offset - record.source_time_offset_seconds) > 1e-6:
        raise ValueError(f"synthesis audit source offset mismatch: {record.video_id}")
    if int(audit["output"]["frame_count"]) != int(cache.meta["packet_count"]):
        raise ValueError(f"synthesis audit frame count mismatch: {record.video_id}")
    if (
        not audit["source_timeline"].get("cfr_packet_grid_verified")
        or int(audit["source_timeline"].get("verified_packet_count", -1))
        != int(audit["output"]["frame_count"])
    ):
        raise ValueError(f"synthesis audit lacks CFR packet-grid verification: {record.video_id}")
    if int(audit["labels"]["cue_count"]) != len(intervals):
        raise ValueError(f"synthesis audit label count mismatch: {record.video_id}")
    rendering = audit["labels"]["rendering"]
    for interval, rendered in zip(intervals, rendering, strict=True):
        if (
            abs(interval.start_seconds - float(rendered["start_seconds"])) > 1e-6
            or abs(interval.end_seconds - float(rendered["end_seconds"])) > 1e-6
        ):
            raise ValueError(f"synthesis audit label timing mismatch: {record.video_id}")
    stream = cache.meta["stream"]
    roi = audit["subtitle_roi"]
    frame_width = int(stream["width"])
    frame_height = int(stream["height"])
    if (
        int(roi["x"]) < 0
        or int(roi["y"]) < int(np.ceil(frame_height * 0.80))
        or int(roi["x"]) + int(roi["width"]) > frame_width
        or int(roi["y"]) + int(roi["height"]) > frame_height
        or not roi.get("all_nontransparent_pixels_confined_to_roi")
    ):
        raise ValueError(f"synthesis audit violates bottom-20% ROI: {record.video_id}")
    contracts = audit["contracts"]
    if (
        not contracts.get("offline_generation_decodes_and_reencodes_pixels")
        or contracts.get("agent_visually_inspected_video_or_frames")
        or contracts.get("training_feature_extraction_requires_pixel_decode")
    ):
        raise ValueError(f"unexpected synthesis audit contract: {record.video_id}")
    signal_validation = audit.get("signal_validation", {})
    if (
        signal_validation.get("role") != record.signal_validation_role
        or signal_validation.get("pair_id") != record.pair_id
    ):
        raise ValueError(f"synthesis audit pair metadata mismatch: {record.video_id}")
    slice_contract = audit["output"].get("slice_contract", {})
    spatial_contract = cache.meta.get("spatial_contract", {})
    if spatial_contract.get("spatial_mode") == "exact_bottom_slices" and (
        slice_contract.get("mode") != "fixed_horizontal_slices"
        or not slice_contract.get("verified_all_video_packets")
        or int(slice_contract.get("verified_packet_count", -1))
        != int(cache.meta["packet_count"])
    ):
        raise ValueError(f"synthesis audit lacks exact slice verification: {record.video_id}")


def intervals_inside_cache(
    cache: FeatureCache, intervals: list[SubtitleInterval]
) -> list[SubtitleInterval]:
    timestamps = np.asarray(cache.timestamps)
    if len(timestamps) == 0:
        return []
    positive_steps = np.diff(timestamps)
    positive_steps = positive_steps[positive_steps > 0]
    tolerance = float(np.median(positive_steps)) if positive_steps.size else 1 / 30
    cache_start, cache_end = cache.coverage_range_seconds
    selected: list[SubtitleInterval] = []
    for interval in intervals:
        if interval.end_seconds <= cache_start or interval.start_seconds >= cache_end:
            continue
        if (
            interval.start_seconds < cache_start - tolerance
            or interval.end_seconds > cache_end + tolerance
        ):
            raise ValueError(
                f"subtitle interval {interval.start_seconds:.3f}-{interval.end_seconds:.3f} "
                f"crosses feature-cache boundary {cache_start:.3f}-{cache_end:.3f}; "
                "choose a subtitle-free split gap"
            )
        selected.append(
            SubtitleInterval(
                start_seconds=max(interval.start_seconds, cache_start),
                end_seconds=min(interval.end_seconds, cache_end),
                label=interval.label,
            )
        )
    return selected


def ensure_source_disjoint(
    train: list[LoadedRecord],
    val: list[LoadedRecord],
    *,
    allow_same_source_temporal: bool = False,
    temporal_guard_seconds: float = 10.0,
) -> None:
    contracts = {
        json.dumps(
            {
                "feature_settings": item.cache.meta["feature_settings"],
                "visual_feature_settings": item.cache.visual_feature_settings,
                "spatial_contract": item.cache.meta["spatial_contract"],
            },
            sort_keys=True,
        )
        for item in [*train, *val]
    }
    if len(contracts) != 1:
        raise ValueError("all training and validation caches must use the same feature/spatial contract")
    train_video_ids = {item.record.video_id for item in train}
    video_id_overlap = train_video_ids.intersection(item.record.video_id for item in val)
    if video_id_overlap:
        names = ", ".join(sorted(video_id_overlap))
        raise ValueError(f"training and validation share video_id values: {names}")
    train_groups = {item.record.source_group for item in train}
    group_overlap = train_groups.intersection(item.record.source_group for item in val)
    if group_overlap and not allow_same_source_temporal:
        names = ", ".join(sorted(group_overlap))
        raise ValueError(
            f"training and validation share original source_group values: {names}"
        )
    train_ids = {item.cache.source_id for item in train}
    overlap = train_ids.intersection(item.cache.source_id for item in val)
    if overlap and not allow_same_source_temporal:
        raise ValueError(
            "training and validation contain the same source video; split by source video, not windows"
        )
    train_origins = {
        origin
        for item in train
        if (origin := _synthesis_origin_id(item.record)) is not None
    }
    origin_overlap = train_origins.intersection(
        origin
        for item in val
        if (origin := _synthesis_origin_id(item.record)) is not None
    )
    if origin_overlap and not allow_same_source_temporal:
        raise ValueError(
            "training and validation contain synthetic shards derived from the same audited "
            "origin video"
        )
    if allow_same_source_temporal:
        _ensure_temporal_guard(train, val, temporal_guard_seconds)


def _ensure_temporal_guard(
    train: list[LoadedRecord], val: list[LoadedRecord], guard_seconds: float
) -> None:
    if guard_seconds < 0.0:
        raise ValueError("temporal validation guard must be non-negative")
    checked_shared_source = False
    for train_item in train:
        for val_item in val:
            same_source = (
                train_item.record.source_group == val_item.record.source_group
                or train_item.cache.source_id == val_item.cache.source_id
                or (
                    _synthesis_origin_id(train_item.record) is not None
                    and _synthesis_origin_id(train_item.record)
                    == _synthesis_origin_id(val_item.record)
                )
            )
            if not same_source:
                continue
            checked_shared_source = True
            train_start, train_end = train_item.cache.coverage_range_seconds
            val_start, val_end = val_item.cache.coverage_range_seconds
            train_start += train_item.record.source_time_offset_seconds
            train_end += train_item.record.source_time_offset_seconds
            val_start += val_item.record.source_time_offset_seconds
            val_end += val_item.record.source_time_offset_seconds
            separated = (
                train_end + guard_seconds <= val_start
                or val_end + guard_seconds <= train_start
            )
            if not separated:
                raise ValueError(
                    "diagnostic temporal validation requires non-overlapping cache ranges with "
                    f"at least {guard_seconds:.3f}s guard"
                )
    if not checked_shared_source:
        raise ValueError(
            "diagnostic temporal validation was requested but train/val do not share a source"
        )


def _synthesis_origin_id(record: ManifestRecord) -> str | None:
    if record.synthesis_audit_path is None:
        return None
    audit = json.loads(record.synthesis_audit_path.read_text(encoding="utf-8"))
    source = audit["source"]
    video_sha256 = str(source.get("video_sha256", ""))
    if len(video_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in video_sha256
    ):
        raise ValueError(f"synthesis audit lacks source video SHA-256: {record.video_id}")
    return json.dumps(
        {
            "video_sha256": video_sha256,
            "video_size_bytes": int(source["video_size_bytes"]),
        },
        sort_keys=True,
    )


def compute_feature_stats(records: list[LoadedRecord], *, chunk_size: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("cannot compute feature statistics without training records")
    feature_count = records[0].cache.features.shape[1]
    total = 0
    sums = np.zeros((feature_count,), dtype=np.float64)
    squares = np.zeros((feature_count,), dtype=np.float64)
    for record in records:
        features = record.cache.features
        for start in range(0, len(features), chunk_size):
            batch = np.asarray(features[start : start + chunk_size], dtype=np.float64)
            sums += batch.sum(axis=0)
            squares += np.square(batch).sum(axis=0)
            total += len(batch)
    mean = sums / max(1, total)
    variance = np.maximum(0.0, squares / max(1, total) - np.square(mean))
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def window_starts(length: int, window_frames: int, stride_frames: int) -> list[int]:
    if length <= window_frames:
        return [0]
    starts = list(range(0, length - window_frames + 1, stride_frames))
    tail = length - window_frames
    if starts[-1] != tail:
        starts.append(tail)
    return starts


class TimingWindowDataset(Dataset):
    def __init__(
        self,
        records: list[LoadedRecord],
        *,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        window_frames: int,
        stride_frames: int,
        max_windows: int | None = None,
    ) -> None:
        if window_frames <= 0 or stride_frames <= 0:
            raise ValueError("window and stride must be positive")
        if max_windows is not None and max_windows <= 0:
            raise ValueError("max_windows must be positive when provided")
        self.records = records
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.window_frames = window_frames
        self.window_weight = np.maximum(
            np.hanning(window_frames + 2)[1:-1].astype(np.float32), 0.05
        )
        self.items: list[tuple[int, int]] = []
        for record_index, record in enumerate(records):
            self.items.extend(
                (record_index, start)
                for start in window_starts(
                    len(record.cache.timestamps), window_frames, stride_frames
                )
            )
        if max_windows is not None and len(self.items) > max_windows:
            selected = np.linspace(
                0, len(self.items) - 1, num=max_windows, dtype=np.int64
            )
            self.items = [self.items[int(index)] for index in selected]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, start = self.items[index]
        record = self.records[record_index]
        stop = min(start + self.window_frames, len(record.cache.timestamps))
        valid = stop - start
        feature_count = record.cache.features.shape[1]
        token_count = record.cache.tokens.shape[1]
        features = np.zeros((self.window_frames, feature_count), dtype=np.float32)
        tokens = np.zeros((self.window_frames, token_count), dtype=np.int64)
        segment_targets = np.zeros((self.window_frames, 3), dtype=np.float32)
        boundary_event_targets = np.zeros((self.window_frames, 2), dtype=np.float32)
        regression_mask = np.zeros((self.window_frames,), dtype=np.float32)
        mask = np.zeros((self.window_frames,), dtype=np.float32)
        features[:valid] = (
            np.asarray(record.cache.features[start:stop], dtype=np.float32)
            - self.feature_mean
        ) / self.feature_std
        tokens[:valid] = np.asarray(record.cache.tokens[start:stop], dtype=np.int64)
        segment_targets[:valid] = record.segment_targets[start:stop]
        boundary_event_targets[:valid] = record.boundary_event_targets[start:stop]
        timestamps = np.asarray(record.cache.timestamps[start:stop], dtype=np.float64)
        if valid:
            positive_steps = np.diff(timestamps)
            positive_steps = positive_steps[positive_steps > 0.0]
            fallback_duration = (
                float(np.median(positive_steps))
                if positive_steps.size
                else 1.0 / 30.0
            )
            last_duration = float(record.cache.durations[stop - 1])
            window_start = float(timestamps[0])
            window_end = float(
                timestamps[-1]
                + (last_duration if last_duration > 0.0 else fallback_duration)
            )
            target_starts = timestamps + segment_targets[:valid, 1]
            target_ends = timestamps + segment_targets[:valid, 2]
            complete_segments = (
                (target_starts >= window_start - 1e-6)
                & (target_ends <= window_end + 1e-6)
                & (segment_targets[:valid, 0] > 0.0)
            )
            regression_mask[:valid] = complete_segments.astype(np.float32)
        mask[:valid] = self.window_weight[:valid]
        return {
            "features": torch.from_numpy(features),
            "tokens": torch.from_numpy(tokens),
            "segment_targets": torch.from_numpy(segment_targets),
            "boundary_event_targets": torch.from_numpy(boundary_event_targets),
            "regression_mask": torch.from_numpy(regression_mask),
            "mask": torch.from_numpy(mask),
        }
