from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .bitstream import EXACT_BOTTOM_SLICES, FeatureSettings, extract_feature_cache
from .dataset import FeatureCache


class CompressedStreamingPrepareSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: Path
    output_root: Path
    video_dir: Path | None = None
    token_count: int = Field(default=512, gt=0)
    histogram_bins: int = Field(default=64, gt=1)
    payload_segments: int = Field(default=6, gt=0)
    workers: int = Field(default=4, gt=0)
    ffprobe: str = "ffprobe"
    overwrite: bool = False

    @model_validator(mode="after")
    def validate_feature_width(self) -> "CompressedStreamingPrepareSettings":
        if 256 % self.histogram_bins != 0:
            raise ValueError("histogram_bins must divide 256")
        return self


def _resolve_source_path(base: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def prepare_compressed_streaming_features(
    settings: CompressedStreamingPrepareSettings,
) -> dict[str, object]:
    manifest = settings.manifest.expanduser().resolve()
    output_root = settings.output_root.expanduser().resolve()
    video_dir = (
        settings.video_dir.expanduser().resolve()
        if settings.video_dir is not None
        else (manifest.parent / "composite" / "videos").resolve()
    )
    records: list[dict[str, object]] = []
    with manifest.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not item.get("video_id"):
                raise ValueError(f"invalid source manifest record on line {line_number}")
            records.append(item)
    if not records:
        raise ValueError("source manifest contains no records")

    feature_settings = FeatureSettings(
        token_count=settings.token_count,
        histogram_bins=settings.histogram_bins,
        payload_segments=settings.payload_segments,
        payload_tail_ratio=0.20,
        spatial_mode=EXACT_BOTTOM_SLICES,
    )
    feature_settings.validate()
    features_root = output_root / "features"
    features_root.mkdir(parents=True, exist_ok=True)

    def extract(item: dict[str, object]) -> dict[str, object]:
        video_id = str(item["video_id"])
        source_video = video_dir / f"{video_id}.mp4"
        if not source_video.is_file():
            raise FileNotFoundError(f"source video not found: {source_video}")
        feature_dir = features_root / video_id
        if not feature_dir.exists() or settings.overwrite:
            extract_feature_cache(
                source_video,
                feature_dir,
                settings=feature_settings,
                ffprobe=settings.ffprobe,
                overwrite=settings.overwrite,
            )
        else:
            cache = FeatureCache(feature_dir)
            try:
                if cache.visual_feature_settings is not None:
                    raise ValueError(
                        f"compressed feature cache contains visual data: {feature_dir}"
                    )
                if dict(cache.meta["feature_settings"]) != asdict(feature_settings):
                    raise ValueError(
                        f"compressed feature cache settings differ: {feature_dir}"
                    )
            finally:
                cache.release()
        converted = dict(item)
        converted["features"] = str(feature_dir.relative_to(output_root))
        converted["labels"] = str(
            _resolve_source_path(manifest.parent, item["labels"])
        )
        if item.get("synthesis_audit"):
            converted["synthesis_audit"] = str(
                _resolve_source_path(manifest.parent, item["synthesis_audit"])
            )
        return converted

    with ThreadPoolExecutor(max_workers=settings.workers) as executor:
        converted_records = list(executor.map(extract, records))

    output_manifest = output_root / "manifest.jsonl"
    partial_manifest = output_root / ".manifest.jsonl.partial"
    with partial_manifest.open("w", encoding="utf-8") as file:
        for item in converted_records:
            file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    partial_manifest.replace(output_manifest)

    first_meta = json.loads(
        (features_root / str(converted_records[0]["video_id"]) / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "manifest": str(output_manifest),
        "record_count": len(converted_records),
        "feature_count": len(first_meta["feature_names"]),
        "byte_token_count": settings.token_count,
        "pixel_decode_required": False,
        "input_domain": "h264_compressed_only",
    }
