from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from .bitstream import (
    MACROBLOCK_SIZE,
    VCL_NAL_TYPES,
    iter_packets,
    probe_stream,
    slice_header_prefix_from_nal,
    split_nals,
)
from subtitle_timing_core.hashing import file_sha256
from subtitle_timing_core.labels import (
    InvalidSrtInterval,
    SrtCue,
    SubtitleInterval,
    parse_srt_cues,
    write_intervals,
)


_HTML_TAG = re.compile(r"</?[^>]+>")
_ASS_TAG = re.compile(r"\{\\[^{}]*\}")
_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)
_LIBX264_SLICE_COUNT = 5
_MINIMUM_SIGNAL_DURATION_SECONDS = 0.5
_MAXIMUM_SIGNAL_DURATION_SECONDS = 5.0
_MINIMUM_SIGNAL_GAP_SECONDS = 0.5


@dataclass(frozen=True)
class CueScheduleSettings:
    mode: Literal["source_timing", "randomized_signal", "none"] = "source_timing"
    random_seed: int = 2026
    minimum_duration_seconds: float = 0.5
    maximum_duration_seconds: float = 5.0
    minimum_gap_seconds: float = 0.5
    maximum_gap_seconds: float = 4.0
    maximum_source_characters: int = 72
    maximum_source_lines: int = 2
    payload_partition_index: int = 0
    payload_partition_count: int = 1


@dataclass(frozen=True)
class SynthesisSettings:
    source_video: Path
    source_srt: Path
    output_video: Path
    output_labels: Path
    start_seconds: float
    end_seconds: float
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    font_path: Path | None = None
    font_size: int = 54
    minimum_font_size: int = 24
    encoder: str = "libx264"
    bit_rate: int = 6_000_000
    overwrite: bool = False
    cue_schedule: CueScheduleSettings = field(default_factory=CueScheduleSettings)
    signal_validation_role: Literal[
        "source_timing", "subtitle_signal", "clean_control"
    ] = "source_timing"
    pair_id: str | None = None


@dataclass(frozen=True)
class _PreparedCue:
    source: SrtCue
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    overlay_path: Path


def _required_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(f"required executable not found: {name}")
    return resolved


def _frame_ceiling(seconds: float, frame_rate: Fraction) -> int:
    return math.ceil(seconds * float(frame_rate) - 1e-9)


def _frame_floor(seconds: float, frame_rate: Fraction) -> int:
    return math.floor(seconds * float(frame_rate) + 1e-9)


def _resolve_font(path: Path | None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"subtitle font not found: {resolved}")
        return resolved
    for candidate in _FONT_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("no CJK-capable subtitle font found; pass --font")


def _clean_payload(lines: tuple[str, ...]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for line in lines:
        value = html.unescape(_HTML_TAG.sub("", _ASS_TAG.sub("", line)))
        for logical_line in value.replace(r"\N", "\n").splitlines():
            stripped = logical_line.strip()
            if stripped:
                cleaned.append(stripped)
    if not cleaned:
        raise ValueError("SRT cue payload becomes empty after removing style markup")
    return tuple(cleaned)


def _wrap_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    font: ImageFont.FreeTypeFont,
    maximum_width: int,
) -> list[str]:
    wrapped: list[str] = []
    current = ""
    for character in line:
        candidate = current + character
        if not current or draw.textlength(candidate, font=font) <= maximum_width:
            current = candidate
            continue
        wrapped.append(current.rstrip())
        current = character.lstrip() if character.isspace() else character
    if current:
        wrapped.append(current.rstrip())
    return wrapped or [""]


@lru_cache(maxsize=64)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size)


def _select_overlay_layout(
    cue: SrtCue,
    *,
    width: int,
    band_height: int,
    font_path: Path,
    font_size: int,
    minimum_font_size: int,
) -> tuple[ImageFont.FreeTypeFont, str, int, tuple[int, int, int, int]]:
    source_lines = _clean_payload(cue.payload_lines)
    draw = ImageDraw.Draw(Image.new("RGBA", (width, band_height), (0, 0, 0, 0)))
    maximum_width = width - 160
    maximum_height = band_height - 28
    for size in range(font_size, minimum_font_size - 1, -2):
        font = _load_font(str(font_path), size)
        wrapped = [
            item
            for source_line in source_lines
            for item in _wrap_line(draw, source_line, font, maximum_width)
        ]
        text = "\n".join(wrapped)
        spacing = max(4, size // 8)
        stroke_width = max(2, size // 18)
        box = draw.multiline_textbbox(
            (0, 0),
            text,
            font=font,
            spacing=spacing,
            align="center",
            stroke_width=stroke_width,
        )
        if box[2] - box[0] <= maximum_width and box[3] - box[1] <= maximum_height:
            return font, text, spacing, box
    raise ValueError(
        f"subtitle cue at line {cue.line_number} does not fit inside the bottom-20% band"
    )


def _render_overlay(
    cue: SrtCue,
    path: Path,
    *,
    width: int,
    band_height: int,
    font_path: Path,
    font_size: int,
    minimum_font_size: int,
) -> dict[str, object]:
    canvas = Image.new("RGBA", (width, band_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font, text, spacing, box = _select_overlay_layout(
        cue,
        width=width,
        band_height=band_height,
        font_path=font_path,
        font_size=font_size,
        minimum_font_size=minimum_font_size,
    )
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    x = (width - text_width) / 2 - box[0]
    y = (band_height - text_height) / 2 - box[1]
    stroke_width = max(2, font.size // 18)
    draw.multiline_text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=spacing,
        align="center",
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0, 255),
    )
    alpha_bounds = canvas.getchannel("A").getbbox()
    if alpha_bounds is None:
        raise ValueError(f"subtitle cue at line {cue.line_number} rendered no visible pixels")
    if not (
        0 <= alpha_bounds[0] < alpha_bounds[2] <= width
        and 0 <= alpha_bounds[1] < alpha_bounds[3] <= band_height
    ):
        raise AssertionError("rendered subtitle escaped its ROI canvas")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", compress_level=3)
    return {
        "font_size": font.size,
        "alpha_bounds_in_band": list(alpha_bounds),
    }


def _payload_sha256(cue: SrtCue) -> str:
    normalized = "\n".join(_clean_payload(cue.payload_lines))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_cue_schedule(settings: SynthesisSettings) -> None:
    schedule = settings.cue_schedule
    if schedule.mode not in {"source_timing", "randomized_signal", "none"}:
        raise ValueError(f"unsupported cue schedule mode: {schedule.mode}")
    allowed_roles = {
        "source_timing": {"source_timing", "subtitle_signal"},
        "randomized_signal": {"subtitle_signal"},
        "none": {"clean_control"},
    }[schedule.mode]
    if settings.signal_validation_role not in allowed_roles:
        roles = ", ".join(sorted(allowed_roles))
        raise ValueError(
            f"cue schedule {schedule.mode} requires signal_validation_role in: {roles}"
        )
    if settings.pair_id is not None and not settings.pair_id.strip():
        raise ValueError("pair_id must be non-empty when provided")
    if schedule.mode != "randomized_signal":
        return
    numeric_values = (
        schedule.minimum_duration_seconds,
        schedule.maximum_duration_seconds,
        schedule.minimum_gap_seconds,
        schedule.maximum_gap_seconds,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("randomized cue schedule bounds must be finite")
    if not (
        _MINIMUM_SIGNAL_DURATION_SECONDS
        <= schedule.minimum_duration_seconds
        <= schedule.maximum_duration_seconds
        <= _MAXIMUM_SIGNAL_DURATION_SECONDS
    ):
        raise ValueError(
            "randomized cue durations must stay inside the 0.5-5.0 second decoder/context contract"
        )
    if not (
        _MINIMUM_SIGNAL_GAP_SECONDS
        <= schedule.minimum_gap_seconds
        <= schedule.maximum_gap_seconds
    ):
        raise ValueError("randomized cue gaps must be ordered and at least 0.5 seconds")
    if schedule.maximum_source_characters <= 0 or schedule.maximum_source_lines <= 0:
        raise ValueError("randomized source-text complexity limits must be positive")
    if schedule.payload_partition_count <= 0:
        raise ValueError("payload_partition_count must be positive")
    if not 0 <= schedule.payload_partition_index < schedule.payload_partition_count:
        raise ValueError("payload_partition_index must be inside the partition count")


def _eligible_randomized_cues(
    cues: list[SrtCue],
    *,
    width: int,
    band_height: int,
    font_path: Path,
    font_size: int,
    minimum_font_size: int,
    maximum_source_characters: int,
    maximum_source_lines: int,
) -> tuple[list[SrtCue], dict[str, object]]:
    eligible: list[SrtCue] = []
    empty_lines: list[int] = []
    complexity_lines: list[int] = []
    unrenderable_lines: list[int] = []
    for cue in cues:
        try:
            source_lines = _clean_payload(cue.payload_lines)
        except ValueError:
            empty_lines.append(cue.line_number)
            continue
        character_count = sum(len(line) for line in source_lines)
        if (
            len(source_lines) > maximum_source_lines
            or character_count > maximum_source_characters
        ):
            complexity_lines.append(cue.line_number)
            continue
        try:
            _select_overlay_layout(
                cue,
                width=width,
                band_height=band_height,
                font_path=font_path,
                font_size=font_size,
                minimum_font_size=minimum_font_size,
            )
        except ValueError:
            unrenderable_lines.append(cue.line_number)
            continue
        eligible.append(cue)
    return eligible, {
        "source_cue_count": len(cues),
        "eligible_source_cue_count": len(eligible),
        "rejected_empty_payload_count": len(empty_lines),
        "rejected_empty_payload_lines": empty_lines,
        "rejected_complexity_count": len(complexity_lines),
        "rejected_complexity_lines": complexity_lines,
        "rejected_unrenderable_count": len(unrenderable_lines),
        "rejected_unrenderable_lines": unrenderable_lines,
        "maximum_source_characters": maximum_source_characters,
        "maximum_source_lines": maximum_source_lines,
        "minimum_rendered_font_size": minimum_font_size,
    }


def _partition_randomized_cues(
    cues: list[SrtCue], settings: CueScheduleSettings
) -> tuple[list[SrtCue], dict[str, object]]:
    selected = [
        cue
        for cue in cues
        if int(_payload_sha256(cue), 16) % settings.payload_partition_count
        == settings.payload_partition_index
    ]
    if not selected:
        raise ValueError(
            "randomized cue payload partition contains no eligible source payloads"
        )
    return selected, {
        "method": "normalized_payload_sha256_modulo_v1",
        "index": settings.payload_partition_index,
        "count": settings.payload_partition_count,
        "eligible_before_partition": len(cues),
        "eligible_after_partition": len(selected),
        "unique_payloads_after_partition": len({_payload_sha256(cue) for cue in selected}),
    }


def _frame_bounds(
    minimum_seconds: float,
    maximum_seconds: float,
    frame_rate: Fraction,
    *,
    name: str,
) -> tuple[int, int]:
    minimum_frames = _frame_ceiling(minimum_seconds, frame_rate)
    maximum_frames = _frame_floor(maximum_seconds, frame_rate)
    if minimum_frames < 0 or maximum_frames < minimum_frames:
        raise ValueError(f"{name} bounds contain no valid integer-frame values")
    return minimum_frames, maximum_frames


def _randomized_cue_schedule(
    cues: list[SrtCue],
    *,
    frame_count: int,
    frame_rate: Fraction,
    settings: CueScheduleSettings,
) -> tuple[list[tuple[SrtCue, int, int]], dict[str, object]]:
    if not cues:
        raise ValueError("randomized cue scheduling has no eligible source payloads")
    minimum_duration, maximum_duration = _frame_bounds(
        settings.minimum_duration_seconds,
        settings.maximum_duration_seconds,
        frame_rate,
        name="cue duration",
    )
    minimum_gap, maximum_gap = _frame_bounds(
        settings.minimum_gap_seconds,
        settings.maximum_gap_seconds,
        frame_rate,
        name="cue gap",
    )
    randomizer = random.Random(settings.random_seed)
    deck: list[int] = []

    def next_cue() -> SrtCue:
        nonlocal deck
        if not deck:
            deck = list(range(len(cues)))
            randomizer.shuffle(deck)
        return cues[deck.pop()]

    scheduled: list[tuple[SrtCue, int, int]] = []
    cursor = 0
    while True:
        gap_frames = randomizer.randint(minimum_gap, maximum_gap)
        start_frame = cursor + gap_frames
        if start_frame + minimum_duration > frame_count:
            break
        available_duration = min(maximum_duration, frame_count - start_frame)
        duration_frames = randomizer.randint(minimum_duration, available_duration)
        end_frame = start_frame + duration_frames
        scheduled.append((next_cue(), start_frame, end_frame))
        cursor = end_frame
    if not scheduled:
        raise ValueError("segment is too short for the randomized cue schedule contract")
    return scheduled, {
        "algorithm": "seeded_shuffled_payload_deck_with_discrete_uniform_frame_gaps_v1",
        "random_seed": settings.random_seed,
        "duration_bounds_seconds": [
            settings.minimum_duration_seconds,
            settings.maximum_duration_seconds,
        ],
        "duration_bounds_frames": [minimum_duration, maximum_duration],
        "gap_bounds_seconds": [
            settings.minimum_gap_seconds,
            settings.maximum_gap_seconds,
        ],
        "gap_bounds_frames": [minimum_gap, maximum_gap],
    }


def _schedule_statistics(
    selected_cues: list[tuple[SrtCue, int, int]],
    *,
    frame_count: int,
    frame_rate: Fraction,
) -> dict[str, object]:
    duration_frames = [end - start for _, start, end in selected_cues]
    between_gap_frames = [
        current[1] - previous[2]
        for previous, current in zip(selected_cues, selected_cues[1:], strict=False)
    ]
    active_frames = sum(duration_frames)

    def summary(values: list[int]) -> dict[str, float | int] | None:
        if not values:
            return None
        return {
            "minimum_frames": min(values),
            "maximum_frames": max(values),
            "mean_frames": sum(values) / len(values),
            "minimum_seconds": min(values) / float(frame_rate),
            "maximum_seconds": max(values) / float(frame_rate),
            "mean_seconds": sum(values) / len(values) / float(frame_rate),
        }

    return {
        "generated_cue_count": len(selected_cues),
        "active_frames": active_frames,
        "active_ratio": active_frames / frame_count,
        "cue_duration_summary": summary(duration_frames),
        "between_cue_gap_summary": summary(between_gap_frames),
        "leading_gap_frames": selected_cues[0][1] if selected_cues else frame_count,
        "trailing_gap_frames": (
            frame_count - selected_cues[-1][2] if selected_cues else frame_count
        ),
    }


def _verify_cfr_packet_grid(
    video_path: Path,
    *,
    ffprobe: str,
    timestamp_origin_seconds: float,
    source_start_frame: int,
    source_end_frame: int,
    frame_rate: Fraction,
) -> int:
    frame_seconds = 1.0 / float(frame_rate)
    source_start = source_start_frame * frame_seconds
    source_end = source_end_frame * frame_seconds
    padding = 2.0
    scan_start = max(0.0, source_start - padding)
    scan_duration = source_end - scan_start + padding
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        f"{timestamp_origin_seconds + scan_start:.9f}%+{scan_duration:.9f}",
        "-show_packets",
        "-show_entries",
        "packet=pts_time",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    packets = json.loads(result.stdout).get("packets", [])
    tolerance = max(2e-6, frame_seconds * 1e-4)
    observed: set[int] = set()
    for packet in packets:
        raw_pts = packet.get("pts_time")
        if raw_pts in (None, "", "N/A"):
            continue
        relative_pts = float(raw_pts) - timestamp_origin_seconds
        frame_index = round(relative_pts * float(frame_rate))
        if frame_index < source_start_frame or frame_index >= source_end_frame:
            continue
        expected_pts = frame_index * frame_seconds
        if abs(relative_pts - expected_pts) > tolerance:
            raise ValueError(
                "synthesis requires CFR packet PTS on the declared frame-rate grid"
            )
        if frame_index in observed:
            raise ValueError("synthesis source has duplicate video PTS inside the segment")
        observed.add(frame_index)
    expected_count = source_end_frame - source_start_frame
    if len(observed) != expected_count:
        raise ValueError(
            "synthesis source has missing or variable-rate video PTS inside the segment: "
            f"observed {len(observed)} of {expected_count} expected frames"
        )
    return len(observed)


def _verify_fixed_vcl_slices(
    video_path: Path,
    *,
    ffprobe: str,
    expected_slices: int,
    expected_packets: int,
) -> tuple[int, list[int], int, int]:
    stream = probe_stream(video_path, ffprobe=ffprobe)
    roi_start_y = math.ceil(stream.height * 0.80 - 1e-9)
    if roi_start_y % MACROBLOCK_SIZE != 0:
        raise ValueError(
            "five-slice P1 encoding requires a macroblock-aligned bottom-20% boundary: "
            f"height={stream.height}, y={roi_start_y}"
        )
    macroblocks_per_row = math.ceil(stream.width / MACROBLOCK_SIZE)
    roi_start_macroblock = (roi_start_y // MACROBLOCK_SIZE) * macroblocks_per_row
    verified_packets = 0
    verified_slice_starts: list[int] | None = None
    with video_path.open("rb") as video_file:
        for packet in iter_packets(
            video_path,
            ffprobe=ffprobe,
            timestamp_origin_seconds=stream.start_time_seconds,
        ):
            video_file.seek(packet.position)
            payload = video_file.read(packet.size)
            if len(payload) != packet.size:
                raise EOFError(
                    f"truncated synthesized packet at {packet.position}: "
                    f"expected {packet.size}, got {len(payload)}"
                )
            units = split_nals(
                payload,
                is_avc=stream.is_avc,
                nal_length_size=stream.nal_length_size,
            )
            vcl_units = [
                unit for unit in units if unit and (int(unit[0]) & 0x1F) in VCL_NAL_TYPES
            ]
            slice_count = len(vcl_units)
            if slice_count != expected_slices:
                raise ValueError(
                    "libx264 slice contract failed: "
                    f"packet {verified_packets} has {slice_count} VCL slices, "
                    f"expected {expected_slices}"
                )
            prefixes = [slice_header_prefix_from_nal(unit) for unit in vcl_units]
            if any(prefix is None for prefix in prefixes):
                raise ValueError(
                    f"cannot parse synthesized slice headers in packet {verified_packets}"
                )
            slice_starts = [
                prefix.first_mb_in_slice for prefix in prefixes if prefix is not None
            ]
            if slice_starts != sorted(set(slice_starts)) or slice_starts[0] != 0:
                raise ValueError(
                    "libx264 slices are not in unique raster order: "
                    f"packet {verified_packets}, first_mb_in_slice={slice_starts}"
                )
            if roi_start_macroblock not in slice_starts:
                raise ValueError(
                    "libx264 five-slice layout does not start a slice at the bottom-20% ROI: "
                    f"packet {verified_packets}, expected first_mb={roi_start_macroblock}, "
                    f"observed={slice_starts}"
                )
            if verified_slice_starts is None:
                verified_slice_starts = slice_starts
            elif slice_starts != verified_slice_starts:
                raise ValueError(
                    "libx264 slice boundaries changed between frames: "
                    f"expected={verified_slice_starts}, observed={slice_starts}"
                )
            verified_packets += 1
    if verified_packets != expected_packets:
        raise ValueError(
            "libx264 slice verification packet count mismatch: "
            f"{verified_packets} != {expected_packets}"
        )
    if verified_slice_starts is None:
        raise ValueError("libx264 slice verification found no video packets")
    return (
        verified_packets,
        verified_slice_starts,
        roi_start_y,
        roi_start_macroblock,
    )


def _prepare_output_paths(settings: SynthesisSettings) -> tuple[Path, Path, Path, Path, Path]:
    output_video = settings.output_video.expanduser().resolve()
    output_labels = settings.output_labels.expanduser().resolve()
    source_video = settings.source_video.expanduser().resolve()
    source_srt = settings.source_srt.expanduser().resolve()
    if output_video == source_video or output_labels == source_srt:
        raise ValueError("synthesis outputs must not replace source files")
    assets = output_video.with_name(f"{output_video.stem}.assets")
    audit = output_video.with_suffix(".audit.json")
    partial = output_video.with_name(f"{output_video.stem}.partial{output_video.suffix}")
    output_paths = (output_video, output_labels, assets, audit, partial)
    source_paths = {source_video, source_srt}
    if source_paths.intersection(output_paths) or any(
        source.is_relative_to(assets) for source in source_paths
    ):
        raise ValueError("synthesis outputs and asset directory must not overlap source files")
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("synthesis output paths collide with each other")
    existing = [path for path in output_paths if path.exists()]
    if existing and not settings.overwrite:
        raise FileExistsError(f"synthesis output already exists: {existing[0]}")
    if settings.overwrite:
        for path in (output_video, output_labels, audit, partial):
            if path.exists():
                path.unlink()
        if assets.exists():
            shutil.rmtree(assets)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_labels.parent.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=False)
    return source_video, source_srt, output_video, output_labels, partial


def synthesize_segment(settings: SynthesisSettings) -> dict[str, object]:
    if not math.isfinite(settings.start_seconds) or not math.isfinite(settings.end_seconds):
        raise ValueError("segment times must be finite")
    if settings.start_seconds < 0 or settings.end_seconds <= settings.start_seconds:
        raise ValueError("segment must satisfy 0 <= start < end")
    if settings.font_size < settings.minimum_font_size or settings.minimum_font_size <= 0:
        raise ValueError("font sizes are invalid")
    if settings.bit_rate <= 0:
        raise ValueError("bit rate must be positive")
    if settings.encoder not in {"h264_videotoolbox", "libx264"}:
        raise ValueError(f"unsupported synthesis encoder: {settings.encoder}")
    _validate_cue_schedule(settings)

    source_video, source_srt, output_video, output_labels, partial = _prepare_output_paths(
        settings
    )
    assets = output_video.with_name(f"{output_video.stem}.assets")
    audit_path = output_video.with_suffix(".audit.json")
    ffmpeg = _required_binary(settings.ffmpeg)
    ffprobe = _required_binary(settings.ffprobe)
    info = probe_stream(source_video, ffprobe=ffprobe)
    frame_rate = Fraction(info.average_frame_rate)
    if frame_rate <= 0:
        raise ValueError(f"invalid source frame rate: {info.average_frame_rate}")
    if Fraction(info.nominal_frame_rate) != frame_rate:
        raise ValueError(
            "synthesis requires a constant-frame-rate source with matching nominal and "
            "average frame rates"
        )
    roi_y = math.ceil(info.height * 0.80 - 1e-9)
    if settings.encoder == "libx264" and roi_y % MACROBLOCK_SIZE != 0:
        raise ValueError(
            "five-slice P1 encoding requires a macroblock-aligned bottom-20% boundary: "
            f"height={info.height}, y={roi_y}"
        )
    if info.duration_seconds is not None and settings.end_seconds > info.duration_seconds + 1e-6:
        raise ValueError("requested synthesis segment exceeds source duration")
    source_start_frame = _frame_ceiling(settings.start_seconds, frame_rate)
    source_end_frame = _frame_floor(settings.end_seconds, frame_rate)
    if source_end_frame <= source_start_frame:
        raise ValueError("requested synthesis segment contains no complete video frames")
    frame_count = source_end_frame - source_start_frame
    source_start = source_start_frame / float(frame_rate)
    source_end = source_end_frame / float(frame_rate)
    duration = frame_count / float(frame_rate)
    verified_source_packets = _verify_cfr_packet_grid(
        source_video,
        ffprobe=ffprobe,
        timestamp_origin_seconds=info.start_time_seconds,
        source_start_frame=source_start_frame,
        source_end_frame=source_end_frame,
        frame_rate=frame_rate,
    )

    band_height = info.height - roi_y
    selected_cues: list[tuple[SrtCue, int, int]] = []
    invalid: list[InvalidSrtInterval] = []
    schedule_details: dict[str, object] = {"mode": settings.cue_schedule.mode}
    if settings.cue_schedule.mode == "none":
        schedule_details["source_srt_parsed"] = False
        font_path: Path | None = None
    else:
        cues, invalid = parse_srt_cues(source_srt, drop_invalid=True)
        font_path = _resolve_font(settings.font_path)
        if settings.cue_schedule.mode == "source_timing":
            one_frame = 1.0 / float(frame_rate)
            for cue in cues:
                if cue.end_seconds <= source_start or cue.start_seconds >= source_end:
                    continue
                if (
                    cue.start_seconds < source_start - one_frame
                    or cue.end_seconds > source_end + one_frame
                ):
                    raise ValueError(
                        f"SRT cue {cue.start_seconds:.3f}-{cue.end_seconds:.3f} crosses segment "
                        f"boundary {source_start:.3f}-{source_end:.3f}; "
                        "choose subtitle-free boundaries"
                    )
                start_frame = max(
                    0,
                    _frame_ceiling(cue.start_seconds, frame_rate) - source_start_frame,
                )
                end_frame = min(
                    frame_count,
                    _frame_ceiling(cue.end_seconds, frame_rate) - source_start_frame,
                )
                if end_frame > start_frame:
                    selected_cues.append((cue, start_frame, end_frame))
            schedule_details.update(
                {
                    "source_srt_parsed": True,
                    "source_cue_count": len(cues),
                    "payload_selection": "cues_whose_source_timing_is_inside_the_segment",
                }
            )
        else:
            eligible_cues, payload_audit = _eligible_randomized_cues(
                cues,
                width=info.width,
                band_height=band_height,
                font_path=font_path,
                font_size=settings.font_size,
                minimum_font_size=settings.minimum_font_size,
                maximum_source_characters=(
                    settings.cue_schedule.maximum_source_characters
                ),
                maximum_source_lines=settings.cue_schedule.maximum_source_lines,
            )
            partitioned_cues, partition_audit = _partition_randomized_cues(
                eligible_cues, settings.cue_schedule
            )
            selected_cues, randomized_details = _randomized_cue_schedule(
                partitioned_cues,
                frame_count=frame_count,
                frame_rate=frame_rate,
                settings=settings.cue_schedule,
            )
            schedule_details.update(
                {
                    "source_srt_parsed": True,
                    "payload_filter": payload_audit,
                    "payload_partition": partition_audit,
                    **randomized_details,
                }
            )
    schedule_details.update(
        _schedule_statistics(
            selected_cues,
            frame_count=frame_count,
            frame_rate=frame_rate,
        )
    )

    prepared: list[_PreparedCue] = []
    render_audit: list[dict[str, object]] = []
    for index, (cue, start_frame, end_frame) in enumerate(selected_cues):
        if font_path is None:
            raise RuntimeError("a cue schedule with rendered cues requires a font")
        overlay_path = assets / f"cue-{index:04d}-line-{cue.line_number}.png"
        rendering = _render_overlay(
            cue,
            overlay_path,
            width=info.width,
            band_height=band_height,
            font_path=font_path,
            font_size=settings.font_size,
            minimum_font_size=settings.minimum_font_size,
        )
        start = start_frame / float(frame_rate)
        end = end_frame / float(frame_rate)
        prepared.append(_PreparedCue(cue, start_frame, end_frame, start, end, overlay_path))
        render_audit.append(
            {
                "srt_timing_line": cue.line_number,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": start,
                "end_seconds": end,
                "source_srt_start_seconds": cue.start_seconds,
                "source_srt_end_seconds": cue.end_seconds,
                "source_payload_sha256": _payload_sha256(cue),
                **rendering,
            }
        )

    graph_lines = [
        f"[0:v]settb=expr={frame_rate.denominator}/{frame_rate.numerator},setpts=N[base0]"
    ]
    for index, cue in enumerate(prepared, start=1):
        graph_lines.append(
            f"[{index}:v]format=rgba,settb=expr={frame_rate.denominator}/"
            f"{frame_rate.numerator},setpts=0[overlay{index}]"
        )
        graph_lines.append(
            f"[base{index - 1}][overlay{index}]overlay=x=0:y={roi_y}:"
            f"enable='gte(n,{cue.start_frame})*lt(n,{cue.end_frame})':"
            f"eof_action=repeat:repeatlast=1:shortest=0[base{index}]"
        )
    graph_lines.append(f"[base{len(prepared)}]format=yuv420p[outv]")
    filter_graph = ";\n".join(graph_lines) + "\n"
    filter_path = assets / "filtergraph.txt"
    filter_path.write_text(filter_graph, encoding="utf-8")

    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "warning",
        "-stats",
        "-stats_period",
        "10",
        "-y",
        "-ss",
        f"{source_start:.9f}",
        "-i",
        str(source_video),
    ]
    for cue in prepared:
        command.extend(["-framerate", str(frame_rate), "-i", str(cue.overlay_path)])
    command.extend(
        [
            "-/filter_complex",
            str(filter_path),
            "-map",
            "[outv]",
            "-an",
            "-frames:v",
            str(frame_count),
            "-fps_mode",
            "passthrough",
            "-c:v",
            settings.encoder,
            "-profile:v",
            "main",
            "-g",
            "60",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if settings.encoder == "h264_videotoolbox":
        command.extend(
            [
                "-coder",
                "cabac",
                "-b:v",
                str(settings.bit_rate),
                "-maxrate",
                str(int(settings.bit_rate * 1.5)),
                "-bufsize",
                str(settings.bit_rate * 2),
                "-prio_speed",
                "1",
            ]
        )
    else:
        command.extend(
            [
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-x264-params",
                f"slices={_LIBX264_SLICE_COUNT}",
            ]
        )
    command.extend(["-movflags", "+faststart", str(partial)])
    subprocess.run(command, check=True)

    output_info = probe_stream(partial, ffprobe=ffprobe)
    if output_info.width != info.width or output_info.height != info.height:
        raise ValueError("synthesized video dimensions do not match the source")
    if output_info.frame_count is not None and output_info.frame_count != frame_count:
        raise ValueError(
            f"synthesized frame count mismatch: {output_info.frame_count} != {frame_count}"
        )
    if output_info.duration_seconds is not None:
        if abs(output_info.duration_seconds - duration) > 2.0 / float(frame_rate):
            raise ValueError("synthesized video duration differs by more than two frames")
    if settings.encoder == "libx264":
        (
            verified_slice_packets,
            verified_slice_starts,
            verified_roi_start_y,
            verified_roi_start_macroblock,
        ) = _verify_fixed_vcl_slices(
            partial,
            ffprobe=ffprobe,
            expected_slices=_LIBX264_SLICE_COUNT,
            expected_packets=frame_count,
        )
        slice_contract: dict[str, object] = {
            "mode": "fixed_horizontal_slices",
            "encoder_parameter": f"slices={_LIBX264_SLICE_COUNT}",
            "slices_per_frame": _LIBX264_SLICE_COUNT,
            "raster_scan_order": "top_to_bottom_horizontal_bands",
            "first_mb_in_slice": verified_slice_starts,
            "roi_start_y": verified_roi_start_y,
            "roi_start_macroblock": verified_roi_start_macroblock,
            "verified_all_video_packets": True,
            "verified_packet_count": verified_slice_packets,
        }
    else:
        slice_contract = {
            "mode": "encoder_default_unverified",
            "slices_per_frame": None,
            "verified_all_video_packets": False,
            "verified_packet_count": 0,
        }
    partial.replace(output_video)

    labels = [
        SubtitleInterval(cue.start_seconds, cue.end_seconds) for cue in prepared
    ]
    write_intervals(output_labels, labels)
    output_video_sha256 = file_sha256(output_video)
    output_labels_sha256 = file_sha256(output_labels)
    audit: dict[str, object] = {
        "format": "h264_timing_synthetic_subtitle_segment",
        "version": 2,
        "source": {
            "video": str(source_video),
            "video_size_bytes": source_video.stat().st_size,
            "video_sha256": file_sha256(source_video),
            "srt": str(source_srt),
            "srt_sha256": file_sha256(source_srt),
        },
        "output": {
            "video": str(output_video),
            "labels": str(output_labels),
            "video_size_bytes": output_video.stat().st_size,
            "video_sha256": output_video_sha256,
            "labels_sha256": output_labels_sha256,
            "frame_count": frame_count,
            "duration_seconds": duration,
            "frame_rate": str(frame_rate),
            "codec": output_info.codec_name,
            "encoder": settings.encoder,
            "measured_bit_rate": output_info.bit_rate,
            "rate_control": (
                {
                    "mode": "average_bit_rate",
                    "target_bit_rate": settings.bit_rate,
                    "maximum_bit_rate": int(settings.bit_rate * 1.5),
                }
                if settings.encoder == "h264_videotoolbox"
                else {"mode": "constant_rate_factor", "crf": 20, "preset": "veryfast"}
            ),
            "slice_contract": slice_contract,
            "audio_included": False,
        },
        "source_timeline": {
            "start_seconds": source_start,
            "end_seconds": source_end,
            "start_frame": source_start_frame,
            "end_frame": source_end_frame,
            "cfr_packet_grid_verified": True,
            "verified_packet_count": verified_source_packets,
        },
        "subtitle_roi": {
            "x": 0,
            "y": roi_y,
            "width": info.width,
            "height": band_height,
            "frame_height_fraction": band_height / info.height,
            "all_nontransparent_pixels_confined_to_roi": True,
        },
        "labels": {
            "cue_count": len(prepared),
            "invalid_source_cue_count": len(invalid),
            "invalid_source_timing_lines": [item.line_number for item in invalid],
            "time_quantization": "first output frame with pts >= SRT boundary",
            "schedule": schedule_details,
            "rendering_contract": {
                "requested_font": (
                    str(settings.font_path.expanduser().resolve())
                    if settings.font_path is not None
                    else None
                ),
                "resolved_font": str(font_path) if font_path is not None else None,
                "requested_font_size": settings.font_size,
                "minimum_font_size": settings.minimum_font_size,
            },
            "rendering": render_audit,
        },
        "signal_validation": {
            "role": settings.signal_validation_role,
            "pair_id": settings.pair_id,
        },
        "contracts": {
            "offline_generation_decodes_and_reencodes_pixels": True,
            "agent_visually_inspected_video_or_frames": False,
            "training_feature_extraction_requires_pixel_decode": False,
        },
        "filter_graph_sha256": hashlib.sha256(filter_graph.encode("utf-8")).hexdigest(),
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "video": str(output_video),
        "labels": str(output_labels),
        "audit": str(audit_path),
        "source_time_offset_seconds": source_start,
        "duration_seconds": duration,
        "frame_count": frame_count,
        "cue_count": len(prepared),
        "invalid_source_cue_count": len(invalid),
        "cue_schedule": schedule_details,
        "signal_validation": audit["signal_validation"],
        "slice_contract": slice_contract,
        "subtitle_roi": audit["subtitle_roi"],
        "contracts": audit["contracts"],
    }
