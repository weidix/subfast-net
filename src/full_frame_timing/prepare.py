from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from h264_timing.dataset import FeatureCache, read_manifest

from . import INPUT_DOMAIN
from .features import (
    FullFrameFeatureSettings,
    extract_full_frame_feature_cache,
    timeline_from_cache,
)


class FullFramePrepareSettings(BaseModel):
    manifest: Path
    output_root: Path
    video_dir: Path | None = None
    width: int = Field(default=256, ge=32)
    height: int = Field(default=144, ge=32)
    tile_rows: int = Field(default=8, ge=2)
    tile_columns: int = Field(default=8, ge=2)
    row_bins: int = Field(default=18, ge=2)
    fine_column_bins: int = Field(default=32, ge=2)
    temporal_lags: tuple[int, ...] = (1, 4, 8)
    workers: int = Field(default=2, gt=0)
    ffmpeg: str = "ffmpeg"
    overwrite: bool = False


def _read_raw_records(manifest: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with manifest.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"manifest line {line_number} must contain a JSON object"
                )
            records.append(value)
    if not records:
        raise ValueError("source manifest contains no records")
    return records


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def prepare_full_frame_features(
    settings: FullFramePrepareSettings,
) -> dict[str, object]:
    """Rebuild an existing timing manifest with decoded full-frame features."""

    manifest = settings.manifest.expanduser().resolve()
    output_root = settings.output_root.expanduser().resolve()
    source_records = read_manifest(manifest)
    raw_records = _read_raw_records(manifest)
    if len(source_records) != len(raw_records):
        raise ValueError("source manifest validation changed its record count")
    source_by_id = {record.video_id: record for record in source_records}
    output_root.mkdir(parents=True, exist_ok=True)
    feature_root = output_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    feature_settings = FullFrameFeatureSettings(
        width=settings.width,
        height=settings.height,
        tile_rows=settings.tile_rows,
        tile_columns=settings.tile_columns,
        row_bins=settings.row_bins,
        fine_column_bins=settings.fine_column_bins,
        temporal_lags=settings.temporal_lags,
    )
    video_dir = (
        settings.video_dir.expanduser().resolve()
        if settings.video_dir is not None
        else None
    )

    def convert(item: dict[str, object]) -> dict[str, object]:
        video_id = str(item["video_id"])
        source_record = source_by_id[video_id]
        source_cache = FeatureCache(source_record.feature_dir)
        try:
            timeline = timeline_from_cache(source_cache)
            source_video = (
                video_dir / f"{video_id}.mp4"
                if video_dir is not None
                else Path(str(source_cache.meta["source"])).expanduser().resolve()
            )
        finally:
            source_cache.release()
        feature_dir = feature_root / video_id
        extract_full_frame_feature_cache(
            source_video,
            feature_dir,
            settings=feature_settings,
            timeline=timeline,
            ffmpeg=settings.ffmpeg,
            overwrite=settings.overwrite,
        )
        converted = dict(item)
        converted["features"] = str(feature_dir.relative_to(output_root))
        converted["labels"] = str(_resolve(manifest.parent, item["labels"]))
        if item.get("synthesis_audit"):
            converted["synthesis_audit"] = str(
                _resolve(manifest.parent, item["synthesis_audit"])
            )
        return converted

    with ThreadPoolExecutor(max_workers=settings.workers) as executor:
        converted_records = list(executor.map(convert, raw_records))

    output_manifest = output_root / "manifest.jsonl"
    partial_manifest = output_root / "manifest.partial.jsonl"
    with partial_manifest.open("w", encoding="utf-8") as file:
        for record in converted_records:
            file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    partial_manifest.replace(output_manifest)
    return {
        "manifest": str(output_manifest),
        "record_count": len(converted_records),
        "feature_count": len(
            json.loads(
                (
                    feature_root
                    / str(converted_records[0]["video_id"])
                    / "meta.json"
                ).read_text(encoding="utf-8")
            )["feature_names"]
        ),
        "input_domain": INPUT_DOMAIN,
        "pixel_decode_required": True,
        "spatial_contract": "full_frame",
    }
