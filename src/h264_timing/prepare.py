from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .bitstream import EXACT_BOTTOM_SLICES, FeatureSettings, extract_feature_cache
from .dataset import FeatureCache
from .hashing import file_sha256
from .labels import SubtitleInterval, read_intervals
from .synthesis import (
    CueScheduleSettings,
    SynthesisSettings,
    _frame_ceiling,
    _frame_floor,
    synthesize_segment,
)
from .visual import extract_visual_feature_cache


_AUDIT_FORMAT = "h264_timing_dataset_audit"
_AUDIT_VERSION = 2
_SAFE_CLIP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PAYLOAD_PARTITIONS = {"train": 0, "val": 1, "test": 2}


class PrepareSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_video: Path
    source_srt: Path
    sample_plan: Path
    output_root: Path
    source_group: str = "p1-signal-validation"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    font_path: Path | None = None
    font_size: int = Field(default=54, gt=0)
    minimum_font_size: int = Field(default=24, gt=0)
    base_random_seed: int = 2026
    signal_schedule_mode: Literal["randomized_signal", "source_timing"] = (
        "randomized_signal"
    )
    minimum_cue_duration_seconds: float = Field(default=0.5, ge=0.5, le=5.0)
    maximum_cue_duration_seconds: float = Field(default=5.0, ge=0.5, le=5.0)
    minimum_cue_gap_seconds: float = Field(default=0.5, ge=0.5)
    maximum_cue_gap_seconds: float = Field(default=4.0, ge=0.5)
    maximum_source_characters: int = Field(default=72, gt=0)
    maximum_source_lines: int = Field(default=2, gt=0)
    minimum_split_signal_active_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    maximum_split_signal_active_ratio: float = Field(default=0.70, ge=0.0, le=1.0)
    temporal_guard_seconds: float = Field(default=10.0, ge=0.0)
    token_count: int = Field(default=256, gt=0)
    histogram_bins: int = Field(default=16, gt=1)
    payload_segments: int = Field(default=4, gt=0)
    payload_tail_ratio: float = Field(default=0.20, gt=0.0, le=1.0)
    spatial_mode: Literal["exact_bottom_slices", "payload_tail_proxy"] = (
        EXACT_BOTTOM_SLICES
    )
    paired_clean_controls: bool = True
    overwrite: bool = False
    resume: bool = True

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if not self.source_group.strip():
            raise ValueError("source_group must be non-empty")
        if self.minimum_font_size > self.font_size:
            raise ValueError("minimum_font_size must not exceed font_size")
        if self.minimum_cue_duration_seconds > self.maximum_cue_duration_seconds:
            raise ValueError("cue duration bounds are reversed")
        if self.minimum_cue_gap_seconds > self.maximum_cue_gap_seconds:
            raise ValueError("cue gap bounds are reversed")
        if (
            self.minimum_split_signal_active_ratio
            > self.maximum_split_signal_active_ratio
        ):
            raise ValueError("split active-ratio bounds are reversed")
        if self.overwrite and self.resume:
            raise ValueError("overwrite and resume are mutually exclusive")
        if 256 % self.histogram_bins != 0:
            raise ValueError("histogram_bins must divide 256")
        if (
            self.spatial_mode == EXACT_BOTTOM_SLICES
            and abs(self.payload_tail_ratio - 0.20) > 1e-9
        ):
            raise ValueError(
                "the five-slice exact ROI contract requires payload_tail_ratio=0.20"
            )
        return self


@dataclass(frozen=True)
class ClipPlan:
    clip_id: str
    split: Literal["train", "val", "test"]
    start_seconds: float
    end_seconds: float
    line_number: int


@dataclass(frozen=True)
class PreparedVariant:
    video_id: str
    pair_id: str
    role: Literal["subtitle_signal", "clean_control"]
    split: str
    video_path: Path
    labels_path: Path
    audit_path: Path
    feature_dir: Path
    audit: dict[str, object]
    cache: FeatureCache
    intervals: list[SubtitleInterval]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_clip_seed(base_seed: int, clip_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{clip_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _container_source_id(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    sample_size = 1024 * 1024
    with path.open("rb") as file:
        digest.update(file.read(sample_size))
        if size > sample_size:
            file.seek(max(0, size - sample_size))
            digest.update(file.read(sample_size))
    return f"container-edge-sha256:{digest.hexdigest()}"


def _read_sample_plan(path: Path) -> list[ClipPlan]:
    path = path.expanduser().resolve()
    records: list[ClipPlan] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"clip_id", "split", "start_seconds", "end_seconds"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"sample plan must contain columns: {','.join(sorted(required))}"
            )
        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            clip_id = (row.get("clip_id") or "").strip()
            split = (row.get("split") or "").strip()
            if not _SAFE_CLIP_ID.fullmatch(clip_id):
                raise ValueError(f"unsafe clip_id on sample-plan line {line_number}: {clip_id}")
            if clip_id in seen_ids:
                raise ValueError(f"duplicate clip_id on sample-plan line {line_number}: {clip_id}")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"invalid split on sample-plan line {line_number}: {split}")
            try:
                start_seconds = float(row["start_seconds"])
                end_seconds = float(row["end_seconds"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid segment time on sample-plan line {line_number}"
                ) from error
            if (
                not math.isfinite(start_seconds)
                or not math.isfinite(end_seconds)
                or start_seconds < 0.0
                or end_seconds <= start_seconds
            ):
                raise ValueError(
                    f"sample-plan line {line_number} must satisfy 0 <= start < end"
                )
            records.append(
                ClipPlan(
                    clip_id=clip_id,
                    split=cast(Literal["train", "val", "test"], split),
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    line_number=line_number,
                )
            )
            seen_ids.add(clip_id)
    if not records:
        raise ValueError("sample plan contains no clips")
    present_splits = {record.split for record in records}
    if not {"train", "val"}.issubset(present_splits):
        raise ValueError("sample plan must contain at least one train and one val clip")
    _validate_plan_ranges(records)
    return records


def _validate_plan_ranges(records: list[ClipPlan]) -> None:
    ordered = sorted(records, key=lambda item: (item.start_seconds, item.end_seconds))
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.start_seconds < previous.end_seconds:
            raise ValueError(
                f"sample-plan clips overlap on the source timeline: "
                f"{previous.clip_id} and {current.clip_id}"
            )


def _validate_cross_split_guard(records: list[ClipPlan], guard_seconds: float) -> None:
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if left.split == right.split:
                continue
            separated = (
                left.end_seconds + guard_seconds <= right.start_seconds
                or right.end_seconds + guard_seconds <= left.start_seconds
            )
            if not separated:
                raise ValueError(
                    "source-timeline clips in different splits require at least "
                    f"{guard_seconds:.3f}s guard: {left.clip_id} and {right.clip_id}"
                )


def _variant_paths(
    output_root: Path, plan: ClipPlan, role: Literal["subtitle_signal", "clean_control"]
) -> tuple[str, Path, Path, Path, Path]:
    suffix = "signal" if role == "subtitle_signal" else "clean"
    video_id = f"{plan.clip_id}-{suffix}"
    video_path = output_root / "composite" / "videos" / f"{video_id}.mp4"
    labels_path = output_root / "composite" / "labels" / f"{video_id}.csv"
    audit_path = video_path.with_suffix(".audit.json")
    feature_dir = output_root / "features" / video_id
    return video_id, video_path, labels_path, audit_path, feature_dir


def _expected_synthesis_paths(video_path: Path, labels_path: Path) -> set[Path]:
    return {
        video_path,
        labels_path,
        video_path.with_suffix(".audit.json"),
        video_path.with_name(f"{video_path.stem}.assets"),
    }


def _schedule_for_variant(
    settings: PrepareSettings,
    plan: ClipPlan,
    role: Literal["subtitle_signal", "clean_control"],
) -> CueScheduleSettings:
    if role == "clean_control":
        return CueScheduleSettings(mode="none")
    return CueScheduleSettings(
        mode=settings.signal_schedule_mode,
        random_seed=_stable_clip_seed(settings.base_random_seed, plan.clip_id),
        minimum_duration_seconds=settings.minimum_cue_duration_seconds,
        maximum_duration_seconds=settings.maximum_cue_duration_seconds,
        minimum_gap_seconds=settings.minimum_cue_gap_seconds,
        maximum_gap_seconds=settings.maximum_cue_gap_seconds,
        maximum_source_characters=settings.maximum_source_characters,
        maximum_source_lines=settings.maximum_source_lines,
        payload_partition_index=_PAYLOAD_PARTITIONS[plan.split],
        payload_partition_count=len(_PAYLOAD_PARTITIONS),
    )


def _validate_resumed_source_range(
    source_timeline: dict[str, object],
    output: dict[str, object],
    plan: ClipPlan,
    video_path: Path,
) -> None:
    frame_rate = Fraction(str(output["frame_rate"]))
    expected_start_frame = _frame_ceiling(plan.start_seconds, frame_rate)
    expected_end_frame = _frame_floor(plan.end_seconds, frame_rate)
    if not (
        int(source_timeline.get("start_frame", -1)) == expected_start_frame
        and int(source_timeline.get("end_frame", -1)) == expected_end_frame
        and int(output.get("frame_count", -1))
        == expected_end_frame - expected_start_frame
        and abs(
            float(source_timeline["start_seconds"])
            - expected_start_frame / float(frame_rate)
        )
        <= 1e-9
        and abs(
            float(source_timeline["end_seconds"])
            - expected_end_frame / float(frame_rate)
        )
        <= 1e-9
    ):
        raise ValueError(f"resumed clip source range changed: {video_path}")


def _validate_resumed_synthesis(
    audit: dict[str, object],
    *,
    settings: PrepareSettings,
    plan: ClipPlan,
    role: Literal["subtitle_signal", "clean_control"],
    video_path: Path,
    labels_path: Path,
    schedule: CueScheduleSettings,
) -> None:
    if (
        audit.get("format") != "h264_timing_synthetic_subtitle_segment"
        or audit.get("version") != 2
    ):
        raise ValueError(f"unsupported synthesis audit for resumed clip: {video_path}")
    source = audit["source"]
    output = audit["output"]
    signal_validation = audit.get("signal_validation", {})
    label_data = audit["labels"]
    if Path(source["video"]).expanduser().resolve() != settings.source_video:
        raise ValueError(f"resumed clip source video changed: {video_path}")
    if Path(source["srt"]).expanduser().resolve() != settings.source_srt:
        raise ValueError(f"resumed clip source SRT changed: {video_path}")
    if int(source.get("video_size_bytes", -1)) != settings.source_video.stat().st_size:
        raise ValueError(f"resumed clip source video size changed: {video_path}")
    if source.get("video_sha256") != file_sha256(settings.source_video):
        raise ValueError(f"resumed clip source video contents changed: {video_path}")
    if source.get("srt_sha256") != _sha256(settings.source_srt):
        raise ValueError(f"resumed clip source SRT contents changed: {video_path}")
    if Path(output["video"]).expanduser().resolve() != video_path:
        raise ValueError(f"resumed clip output video path changed: {video_path}")
    if Path(output["labels"]).expanduser().resolve() != labels_path:
        raise ValueError(f"resumed clip output labels path changed: {video_path}")
    if int(output.get("video_size_bytes", -1)) != video_path.stat().st_size:
        raise ValueError(f"resumed clip output video size changed: {video_path}")
    if output.get("video_sha256") != file_sha256(video_path):
        raise ValueError(f"resumed clip output video contents changed: {video_path}")
    if output.get("labels_sha256") != file_sha256(labels_path):
        raise ValueError(f"resumed clip output labels changed: {labels_path}")
    if output.get("encoder") != "libx264":
        raise ValueError(f"P1 paired samples require libx264: {video_path}")
    slice_contract = output.get("slice_contract", {})
    if (
        slice_contract.get("mode") != "fixed_horizontal_slices"
        or int(slice_contract.get("slices_per_frame", 0)) != 5
        or not slice_contract.get("verified_all_video_packets")
    ):
        raise ValueError(f"resumed clip lacks the verified five-slice contract: {video_path}")
    if (
        signal_validation.get("role") != role
        or signal_validation.get("pair_id") != plan.clip_id
    ):
        raise ValueError(f"resumed clip pair metadata changed: {video_path}")
    schedule_audit = label_data.get("schedule", {})
    if schedule_audit.get("mode") != schedule.mode:
        raise ValueError(f"resumed clip cue schedule mode changed: {video_path}")
    if schedule.mode == "randomized_signal":
        if int(schedule_audit.get("random_seed", -1)) != schedule.random_seed:
            raise ValueError(f"resumed clip random seed changed: {video_path}")
        if schedule_audit.get("duration_bounds_seconds") != [
            schedule.minimum_duration_seconds,
            schedule.maximum_duration_seconds,
        ]:
            raise ValueError(f"resumed clip cue duration contract changed: {video_path}")
        if schedule_audit.get("gap_bounds_seconds") != [
            schedule.minimum_gap_seconds,
            schedule.maximum_gap_seconds,
        ]:
            raise ValueError(f"resumed clip cue gap contract changed: {video_path}")
        payload_filter = schedule_audit.get("payload_filter", {})
        if (
            int(payload_filter.get("maximum_source_characters", -1))
            != schedule.maximum_source_characters
            or int(payload_filter.get("maximum_source_lines", -1))
            != schedule.maximum_source_lines
            or int(payload_filter.get("minimum_rendered_font_size", -1))
            != settings.minimum_font_size
        ):
            raise ValueError(f"resumed clip source-payload filter changed: {video_path}")
        partition = schedule_audit.get("payload_partition", {})
        if (
            partition.get("method") != "normalized_payload_sha256_modulo_v1"
            or int(partition.get("index", -1)) != schedule.payload_partition_index
            or int(partition.get("count", -1)) != schedule.payload_partition_count
        ):
            raise ValueError(f"resumed clip payload partition changed: {video_path}")
        rendering = label_data.get("rendering", [])
        if len(rendering) != int(label_data.get("cue_count", -1)) or any(
            len(str(item.get("source_payload_sha256", ""))) != 64
            for item in rendering
        ):
            raise ValueError(f"resumed clip lacks payload content hashes: {video_path}")
    rendering_contract = label_data.get("rendering_contract", {})
    expected_font = str(settings.font_path) if settings.font_path is not None else None
    if (
        rendering_contract.get("requested_font") != expected_font
        or int(rendering_contract.get("requested_font_size", -1))
        != settings.font_size
        or int(rendering_contract.get("minimum_font_size", -1))
        != settings.minimum_font_size
    ):
        raise ValueError(f"resumed clip rendering contract changed: {video_path}")
    if role == "clean_control" and int(label_data.get("cue_count", -1)) != 0:
        raise ValueError(f"clean control contains subtitle cues: {video_path}")
    source_timeline = audit["source_timeline"]
    _validate_resumed_source_range(source_timeline, output, plan, video_path)


def _ensure_synthesis(
    settings: PrepareSettings,
    plan: ClipPlan,
    role: Literal["subtitle_signal", "clean_control"],
    *,
    video_path: Path,
    labels_path: Path,
    audit_path: Path,
) -> dict[str, object]:
    schedule = _schedule_for_variant(settings, plan, role)
    expected_paths = _expected_synthesis_paths(video_path, labels_path)
    existing_paths = {path for path in expected_paths if path.exists()}
    if settings.overwrite:
        synthesize_segment(
            SynthesisSettings(
                source_video=settings.source_video,
                source_srt=settings.source_srt,
                output_video=video_path,
                output_labels=labels_path,
                start_seconds=plan.start_seconds,
                end_seconds=plan.end_seconds,
                ffmpeg=settings.ffmpeg,
                ffprobe=settings.ffprobe,
                font_path=settings.font_path,
                font_size=settings.font_size,
                minimum_font_size=settings.minimum_font_size,
                encoder="libx264",
                overwrite=True,
                cue_schedule=schedule,
                signal_validation_role=(
                    "subtitle_signal" if role == "subtitle_signal" else "clean_control"
                ),
                pair_id=plan.clip_id,
            )
        )
    elif existing_paths:
        if not settings.resume or existing_paths != expected_paths:
            missing = sorted(str(path) for path in expected_paths - existing_paths)
            raise FileExistsError(
                f"incomplete or non-resumable synthesis outputs for {video_path}; "
                f"missing={missing}; use --overwrite"
            )
    else:
        synthesize_segment(
            SynthesisSettings(
                source_video=settings.source_video,
                source_srt=settings.source_srt,
                output_video=video_path,
                output_labels=labels_path,
                start_seconds=plan.start_seconds,
                end_seconds=plan.end_seconds,
                ffmpeg=settings.ffmpeg,
                ffprobe=settings.ffprobe,
                font_path=settings.font_path,
                font_size=settings.font_size,
                minimum_font_size=settings.minimum_font_size,
                encoder="libx264",
                cue_schedule=schedule,
                signal_validation_role=(
                    "subtitle_signal" if role == "subtitle_signal" else "clean_control"
                ),
                pair_id=plan.clip_id,
            )
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _validate_resumed_synthesis(
        audit,
        settings=settings,
        plan=plan,
        role=role,
        video_path=video_path,
        labels_path=labels_path,
        schedule=schedule,
    )
    return audit


def _ensure_feature_cache(
    settings: PrepareSettings,
    *,
    video_path: Path,
    feature_dir: Path,
) -> FeatureCache:
    feature_settings = FeatureSettings(
        token_count=settings.token_count,
        histogram_bins=settings.histogram_bins,
        payload_segments=settings.payload_segments,
        payload_tail_ratio=settings.payload_tail_ratio,
        spatial_mode=settings.spatial_mode,
    )
    if feature_dir.exists() and not settings.overwrite:
        if not settings.resume:
            raise FileExistsError(f"feature cache already exists: {feature_dir}")
        cache = FeatureCache(feature_dir)
        if Path(cache.meta["source"]).expanduser().resolve() != video_path:
            raise ValueError(f"resumed feature cache source changed: {feature_dir}")
        if cache.source_id != _container_source_id(video_path):
            raise ValueError(
                f"resumed feature cache does not match current video bytes: {feature_dir}"
            )
        if cache.meta.get("source_sha256") != file_sha256(video_path):
            raise ValueError(
                f"resumed feature cache has a stale full-video fingerprint: {feature_dir}"
            )
        if cache.meta["feature_settings"] != asdict(feature_settings):
            raise ValueError(f"resumed feature settings changed: {feature_dir}")
    else:
        extract_feature_cache(
            video_path,
            feature_dir,
            settings=feature_settings,
            ffprobe=settings.ffprobe,
            overwrite=settings.overwrite,
        )
    extract_visual_feature_cache(
        feature_dir,
        ffmpeg=settings.ffmpeg,
        overwrite=settings.overwrite,
    )
    return FeatureCache(feature_dir)


def _prepare_variant(
    settings: PrepareSettings,
    plan: ClipPlan,
    role: Literal["subtitle_signal", "clean_control"],
) -> PreparedVariant:
    video_id, video_path, labels_path, audit_path, feature_dir = _variant_paths(
        settings.output_root, plan, role
    )
    audit = _ensure_synthesis(
        settings,
        plan,
        role,
        video_path=video_path,
        labels_path=labels_path,
        audit_path=audit_path,
    )
    cache = _ensure_feature_cache(
        settings,
        video_path=video_path,
        feature_dir=feature_dir,
    )
    intervals = read_intervals(labels_path)
    if len(cache.timestamps) != int(audit["output"]["frame_count"]):
        raise ValueError(f"feature-cache frame count differs from synthesis audit: {video_id}")
    if len(intervals) != int(audit["labels"]["cue_count"]):
        raise ValueError(f"label count differs from synthesis audit: {video_id}")
    if settings.spatial_mode == EXACT_BOTTOM_SLICES and not cache.meta[
        "spatial_contract"
    ].get("exact_pixel_roi"):
        raise ValueError(f"feature cache is not using the exact bottom ROI: {video_id}")
    return PreparedVariant(
        video_id=video_id,
        pair_id=plan.clip_id,
        role=role,
        split=plan.split,
        video_path=video_path,
        labels_path=labels_path,
        audit_path=audit_path,
        feature_dir=feature_dir,
        audit=audit,
        cache=cache,
        intervals=intervals,
    )


def _validate_pair(signal: PreparedVariant, clean: PreparedVariant) -> dict[str, object]:
    if signal.pair_id != clean.pair_id or signal.split != clean.split:
        raise ValueError("paired signal/control records disagree on identity or split")
    signal_output = signal.audit["output"]
    clean_output = clean.audit["output"]
    for key in ("frame_count", "duration_seconds", "frame_rate", "encoder", "rate_control"):
        if signal_output[key] != clean_output[key]:
            raise ValueError(f"paired encoder contract differs for {signal.pair_id}: {key}")
    if signal_output["slice_contract"] != clean_output["slice_contract"]:
        raise ValueError(f"paired slice contract differs for {signal.pair_id}")
    if signal.audit["source_timeline"] != clean.audit["source_timeline"]:
        raise ValueError(f"paired source timeline differs for {signal.pair_id}")
    if len(clean.intervals) != 0:
        raise ValueError(f"clean control labels are not empty: {clean.labels_path}")
    signal_timestamps = np.asarray(signal.cache.timestamps)
    clean_timestamps = np.asarray(clean.cache.timestamps)
    if not np.array_equal(signal_timestamps, clean_timestamps):
        raise ValueError(f"paired feature timestamps differ for {signal.pair_id}")
    if signal.cache.meta["feature_settings"] != clean.cache.meta["feature_settings"]:
        raise ValueError(f"paired feature settings differ for {signal.pair_id}")
    active_seconds = sum(
        interval.end_seconds - interval.start_seconds for interval in signal.intervals
    )
    duration_seconds = float(signal_output["duration_seconds"])
    return {
        "pair_id": signal.pair_id,
        "split": signal.split,
        "signal_video_id": signal.video_id,
        "clean_video_id": clean.video_id,
        "frame_count": int(signal_output["frame_count"]),
        "duration_seconds": duration_seconds,
        "signal_cue_count": len(signal.intervals),
        "signal_active_seconds": active_seconds,
        "signal_active_ratio": active_seconds / duration_seconds,
        "clean_cue_count": 0,
        "clean_active_ratio": 0.0,
        "paired_combined_active_ratio": active_seconds / (2.0 * duration_seconds),
    }


def _duration_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
        "mean_seconds": sum(values) / len(values),
    }


def _validate_payload_split_isolation(
    variants: list[PreparedVariant], settings: PrepareSettings
) -> dict[str, object]:
    if settings.signal_schedule_mode != "randomized_signal":
        return {"enforced": False, "reason": "source_timing_schedule"}
    payloads_by_split: dict[str, set[str]] = {}
    for variant in variants:
        if variant.role != "subtitle_signal":
            continue
        schedule = variant.audit["labels"]["schedule"]
        partition = schedule.get("payload_partition", {})
        expected_index = _PAYLOAD_PARTITIONS[variant.split]
        if (
            partition.get("method") != "normalized_payload_sha256_modulo_v1"
            or int(partition.get("index", -1)) != expected_index
            or int(partition.get("count", -1)) != len(_PAYLOAD_PARTITIONS)
        ):
            raise ValueError(f"unexpected payload partition: {variant.video_id}")
        payload_hashes = {
            str(item["source_payload_sha256"])
            for item in variant.audit["labels"]["rendering"]
        }
        if not payload_hashes:
            raise ValueError(f"subtitle signal has no audited payload hashes: {variant.video_id}")
        payloads_by_split.setdefault(variant.split, set()).update(payload_hashes)
    split_names = sorted(payloads_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = payloads_by_split[left].intersection(payloads_by_split[right])
            if overlap:
                raise ValueError(
                    f"randomized subtitle payloads overlap across splits: {left}/{right}"
                )
    return {
        "enforced": True,
        "method": "normalized_payload_sha256_modulo_v1",
        "partition_count": len(_PAYLOAD_PARTITIONS),
        "split_partition_indices": {
            split: _PAYLOAD_PARTITIONS[split] for split in split_names
        },
        "unique_payload_count_by_split": {
            split: len(payloads_by_split[split]) for split in split_names
        },
        "cross_split_overlap_count": 0,
    }


def _split_statistics(
    variants: list[PreparedVariant],
    pair_audits: list[dict[str, object]],
    settings: PrepareSettings,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for split in sorted({variant.split for variant in variants}):
        split_variants = [variant for variant in variants if variant.split == split]
        signal_variants = [
            variant for variant in split_variants if variant.role == "subtitle_signal"
        ]
        clean_variants = [
            variant for variant in split_variants if variant.role == "clean_control"
        ]
        signal_duration = sum(
            float(variant.audit["output"]["duration_seconds"])
            for variant in signal_variants
        )
        clean_duration = sum(
            float(variant.audit["output"]["duration_seconds"])
            for variant in clean_variants
        )
        signal_intervals = [
            interval for variant in signal_variants for interval in variant.intervals
        ]
        cue_durations = [
            interval.end_seconds - interval.start_seconds
            for interval in signal_intervals
        ]
        cue_gaps = [
            current.start_seconds - previous.end_seconds
            for variant in signal_variants
            for previous, current in zip(
                variant.intervals, variant.intervals[1:], strict=False
            )
        ]
        active_seconds = sum(cue_durations)
        signal_active_ratio = active_seconds / max(signal_duration, 1e-12)
        outside_contract = [
            duration
            for duration in cue_durations
            if not (
                settings.minimum_cue_duration_seconds - 1e-6
                <= duration
                <= settings.maximum_cue_duration_seconds + 1e-6
            )
        ]
        if outside_contract:
            raise ValueError(
                f"{split} contains cue durations outside the configured "
                "decoder/context contract"
            )
        if not (
            settings.minimum_split_signal_active_ratio
            <= signal_active_ratio
            <= settings.maximum_split_signal_active_ratio
        ):
            raise ValueError(
                f"{split} signal active ratio {signal_active_ratio:.6f} is outside "
                f"[{settings.minimum_split_signal_active_ratio:.6f}, "
                f"{settings.maximum_split_signal_active_ratio:.6f}]"
            )
        result[split] = {
            "pair_count": sum(1 for pair in pair_audits if pair["split"] == split),
            "signal_clip_count": len(signal_variants),
            "clean_clip_count": len(clean_variants),
            "signal_duration_seconds": signal_duration,
            "clean_duration_seconds": clean_duration,
            "signal_cue_count": len(signal_intervals),
            "signal_active_seconds": active_seconds,
            "signal_active_ratio": signal_active_ratio,
            "clean_active_ratio": 0.0,
            "paired_combined_active_ratio": active_seconds
            / max(signal_duration + clean_duration, 1e-12),
            "cue_duration_summary": _duration_summary(cue_durations),
            "between_cue_gap_summary": _duration_summary(cue_gaps),
        }
    return result


def _relative_path(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def _manifest_text(
    variants: list[PreparedVariant], settings: PrepareSettings
) -> str:
    lines: list[str] = []
    for variant in variants:
        source_offset = float(variant.audit["source_timeline"]["start_seconds"])
        record = {
            "video_id": variant.video_id,
            "source_group": settings.source_group,
            "features": _relative_path(variant.feature_dir, settings.output_root),
            "labels": _relative_path(variant.labels_path, settings.output_root),
            "split": variant.split,
            "source_time_offset_seconds": source_offset,
            "synthesis_audit": _relative_path(
                variant.audit_path, settings.output_root
            ),
            "pair_id": variant.pair_id,
            "signal_validation_role": variant.role,
        }
        lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines) + "\n"


def _stage_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _write_metadata_atomically(
    manifest_path: Path,
    manifest_text: str,
    audit_path: Path,
    audit_text: str,
) -> None:
    manifest_temporary = _stage_text(manifest_path, manifest_text)
    audit_temporary = _stage_text(audit_path, audit_text)
    try:
        os.replace(audit_temporary, audit_path)
        os.replace(manifest_temporary, manifest_path)
    finally:
        manifest_temporary.unlink(missing_ok=True)
        audit_temporary.unlink(missing_ok=True)


def prepare_dataset(settings: PrepareSettings) -> dict[str, object]:
    settings = settings.model_copy(
        update={
            "source_video": settings.source_video.expanduser().resolve(),
            "source_srt": settings.source_srt.expanduser().resolve(),
            "sample_plan": settings.sample_plan.expanduser().resolve(),
            "output_root": settings.output_root.expanduser().resolve(),
            "font_path": (
                settings.font_path.expanduser().resolve()
                if settings.font_path is not None
                else None
            ),
        }
    )
    for source_path in (settings.source_video, settings.source_srt, settings.sample_plan):
        if not source_path.is_file():
            raise FileNotFoundError(f"required P1 preparation input not found: {source_path}")
    plans = _read_sample_plan(settings.sample_plan)
    _validate_cross_split_guard(plans, settings.temporal_guard_seconds)

    variants: list[PreparedVariant] = []
    pair_audits: list[dict[str, object]] = []
    for plan in plans:
        signal = _prepare_variant(settings, plan, "subtitle_signal")
        variants.append(signal)
        if settings.paired_clean_controls:
            clean = _prepare_variant(settings, plan, "clean_control")
            variants.append(clean)
            pair_audits.append(_validate_pair(signal, clean))
    split_statistics = _split_statistics(variants, pair_audits, settings)
    payload_split_isolation = _validate_payload_split_isolation(variants, settings)
    manifest_path = settings.output_root / "manifest.jsonl"
    dataset_audit_path = settings.output_root / "dataset-audit.json"
    manifest = _manifest_text(variants, settings)
    manifest_sha256 = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    schedule_distribution_contract = {
        "mode": settings.signal_schedule_mode,
        "duration_distribution": "discrete_uniform_integer_frames",
        "duration_bounds_seconds": [
            settings.minimum_cue_duration_seconds,
            settings.maximum_cue_duration_seconds,
        ],
        "gap_distribution": "discrete_uniform_integer_frames",
        "gap_bounds_seconds": [
            settings.minimum_cue_gap_seconds,
            settings.maximum_cue_gap_seconds,
        ],
        "identical_for_all_splits": True,
    }
    dataset_audit: dict[str, object] = {
        "format": _AUDIT_FORMAT,
        "version": _AUDIT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "video": str(settings.source_video),
            "video_size_bytes": settings.source_video.stat().st_size,
            "video_sha256": file_sha256(settings.source_video),
            "srt": str(settings.source_srt),
            "srt_sha256": _sha256(settings.source_srt),
            "source_group": settings.source_group,
        },
        "sample_plan": {
            "path": str(settings.sample_plan),
            "sha256": _sha256(settings.sample_plan),
            "clip_count": len(plans),
        },
        "preparation": {
            "paired_clean_controls": settings.paired_clean_controls,
            "base_random_seed": settings.base_random_seed,
            "schedule_distribution_contract": schedule_distribution_contract,
            "payload_split_isolation": payload_split_isolation,
            "font_size": settings.font_size,
            "minimum_font_size": settings.minimum_font_size,
            "maximum_source_characters": settings.maximum_source_characters,
            "maximum_source_lines": settings.maximum_source_lines,
            "encoder": "libx264",
            "encoder_settings": {
                "crf": 20,
                "preset": "veryfast",
                "gop_frames": 60,
                "horizontal_slices_per_frame": 5,
            },
            "feature_settings": {
                "token_count": settings.token_count,
                "histogram_bins": settings.histogram_bins,
                "payload_segments": settings.payload_segments,
                "payload_tail_ratio": settings.payload_tail_ratio,
                "spatial_mode": settings.spatial_mode,
            },
        },
        "pairs": pair_audits,
        "paired_counts": {
            "pair_count": len(pair_audits),
            "signal_clip_count": sum(
                variant.role == "subtitle_signal" for variant in variants
            ),
            "clean_clip_count": sum(
                variant.role == "clean_control" for variant in variants
            ),
        },
        "features": {
            "cache_count": len(variants),
            "packet_count": sum(len(variant.cache.timestamps) for variant in variants),
            "exact_pixel_roi": settings.spatial_mode == EXACT_BOTTOM_SLICES,
            "spatial_mode": settings.spatial_mode,
            "full_video_sha256_verified": all(
                variant.audit["output"].get("video_sha256")
                == variant.cache.meta.get("source_sha256")
                for variant in variants
            ),
            "cache_artifact_sha256_verified": all(
                bool(variant.cache.meta.get("artifact_sha256")) for variant in variants
            ),
        },
        "splits": split_statistics,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "record_count": len(variants),
        },
        "validation": {
            "all_pairs_share_source_timeline_and_encoder_contract": bool(
                pair_audits or not settings.paired_clean_controls
            ),
            "clean_labels_are_empty": all(
                not variant.intervals
                for variant in variants
                if variant.role == "clean_control"
            ),
            "split_signal_active_ratios_within_contract": True,
            "randomized_duration_and_gap_distribution_shared_across_splits": True,
            "randomized_payloads_disjoint_across_splits": bool(
                payload_split_isolation.get("enforced")
            ),
            "cross_split_temporal_guard_seconds": settings.temporal_guard_seconds,
            "required_training_validation_mode": "diagnostic_temporal",
            "validation_contract": "same_source_paired_signal_control_diagnostic",
        },
    }
    audit_text = (
        json.dumps(dataset_audit, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    _write_metadata_atomically(
        manifest_path,
        manifest,
        dataset_audit_path,
        audit_text,
    )
    return {
        "manifest": str(manifest_path),
        "dataset_audit": str(dataset_audit_path),
        "clip_count": len(variants),
        "pair_count": len(pair_audits),
        "splits": split_statistics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired P1 H.264 signal/control samples, feature caches, manifest, and audit."
        )
    )
    parser.add_argument("source_video", type=Path)
    parser.add_argument("source_srt", type=Path)
    parser.add_argument("sample_plan", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--source-group", default="p1-signal-validation")
    parser.add_argument(
        "--signal-schedule",
        choices=("randomized_signal", "source_timing"),
        default="randomized_signal",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-size", type=int, default=54)
    parser.add_argument("--minimum-font-size", type=int, default=24)
    parser.add_argument("--minimum-cue-duration", type=float, default=0.5)
    parser.add_argument("--maximum-cue-duration", type=float, default=5.0)
    parser.add_argument("--minimum-cue-gap", type=float, default=0.5)
    parser.add_argument("--maximum-cue-gap", type=float, default=4.0)
    parser.add_argument("--maximum-source-characters", type=int, default=72)
    parser.add_argument("--maximum-source-lines", type=int, default=2)
    parser.add_argument("--minimum-split-active-ratio", type=float, default=0.35)
    parser.add_argument("--maximum-split-active-ratio", type=float, default=0.70)
    parser.add_argument("--temporal-guard", type=float, default=10.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = PrepareSettings(
        source_video=args.source_video,
        source_srt=args.source_srt,
        sample_plan=args.sample_plan,
        output_root=args.output_root,
        source_group=args.source_group,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        font_path=args.font,
        font_size=args.font_size,
        minimum_font_size=args.minimum_font_size,
        base_random_seed=args.seed,
        signal_schedule_mode=args.signal_schedule,
        minimum_cue_duration_seconds=args.minimum_cue_duration,
        maximum_cue_duration_seconds=args.maximum_cue_duration,
        minimum_cue_gap_seconds=args.minimum_cue_gap,
        maximum_cue_gap_seconds=args.maximum_cue_gap,
        maximum_source_characters=args.maximum_source_characters,
        maximum_source_lines=args.maximum_source_lines,
        minimum_split_signal_active_ratio=args.minimum_split_active_ratio,
        maximum_split_signal_active_ratio=args.maximum_split_active_ratio,
        temporal_guard_seconds=args.temporal_guard,
        overwrite=args.overwrite,
        resume=not args.no_resume and not args.overwrite,
    )
    result = prepare_dataset(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
