from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from . import FEATURE_FORMAT, FEATURE_VERSION
from subtitle_timing_core.hashing import file_sha256


VCL_NAL_TYPES = {1, 2, 3, 4, 5}
SLICE_HEADER_NAL_TYPES = {1, 2, 5}
EXACT_SLICE_NAL_TYPES = {1, 5}
SEI_NAL_TYPE = 6
FILLER_NAL_TYPE = 12
EXACT_BOTTOM_SLICES = "exact_bottom_slices"
PAYLOAD_TAIL_PROXY = "payload_tail_proxy"
SPATIAL_MODES = {EXACT_BOTTOM_SLICES, PAYLOAD_TAIL_PROXY}
MACROBLOCK_SIZE = 16
TEMPORAL_SIGNAL_NAMES = (
    "log_packet_bytes",
    "log_global_vcl_bytes",
    "log_roi_vcl_bytes",
    "log_above_roi_vcl_bytes",
    "roi_within_global_vcl_ratio",
    "roi_to_above_vcl_ratio",
    "roi_above_vcl_contrast",
)
SCALAR_FEATURE_NAMES = (
    "log_packet_bytes",
    "log_global_vcl_bytes",
    "log_roi_vcl_bytes",
    "log_above_roi_vcl_bytes",
    "log_filler_bytes",
    "log_sei_bytes",
    "global_vcl_to_packet_ratio",
    "roi_vcl_to_packet_ratio",
    "above_roi_vcl_to_packet_ratio",
    "roi_within_global_vcl_ratio",
    "above_within_global_vcl_ratio",
    "roi_to_above_vcl_ratio",
    "roi_above_vcl_contrast",
    "filler_to_packet_ratio",
    "sei_to_packet_ratio",
    "duration_seconds",
    "pts_minus_dts_seconds",
    "presentation_step_seconds",
    "keyframe",
    "decode_order_offset",
    "log_nal_count",
    "log_vcl_nal_count",
    "slice_p",
    "slice_b",
    "slice_i",
    "slice_other",
    "has_idr",
    "has_sps",
    "has_pps",
    "has_aud",
)
PACKET_RECORD = struct.Struct("<ddfqqB")
PACKET_DTYPE = np.dtype(
    [
        ("pts", "<f8"),
        ("dts", "<f8"),
        ("duration", "<f4"),
        ("position", "<i8"),
        ("size", "<i8"),
        ("keyframe", "u1"),
    ]
)


@dataclass(frozen=True)
class StreamInfo:
    codec_name: str
    profile: str | None
    level: int | None
    width: int
    height: int
    nominal_frame_rate: str
    average_frame_rate: str
    time_base: str
    start_time_seconds: float
    duration_seconds: float | None
    frame_count: int | None
    bit_rate: int | None
    is_avc: bool
    nal_length_size: int
    format_name: str | None
    file_size: int
    field_order: str | None = None


@dataclass(frozen=True)
class PacketInfo:
    pts: float
    dts: float
    duration: float
    position: int
    size: int
    keyframe: bool
    decode_index: int


@dataclass(frozen=True)
class FeatureSettings:
    token_count: int = 256
    histogram_bins: int = 16
    payload_segments: int = 4
    payload_tail_ratio: float = 0.20
    spatial_mode: str = EXACT_BOTTOM_SLICES

    def validate(self) -> None:
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.histogram_bins <= 1 or 256 % self.histogram_bins != 0:
            raise ValueError("histogram_bins must be a divisor of 256 and greater than 1")
        if self.payload_segments <= 0:
            raise ValueError("payload_segments must be positive")
        if not 0.0 < self.payload_tail_ratio <= 1.0:
            raise ValueError("payload_tail_ratio must be in (0, 1]")
        if self.spatial_mode not in SPATIAL_MODES:
            modes = ", ".join(sorted(SPATIAL_MODES))
            raise ValueError(f"spatial_mode must be one of: {modes}")
        if self.spatial_mode == EXACT_BOTTOM_SLICES and self.payload_tail_ratio >= 1.0:
            raise ValueError(
                "exact_bottom_slices requires a proper bottom ROI smaller than the frame"
            )


def _required_binary(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        raise FileNotFoundError(f"required executable not found: {binary}")
    return resolved


def _optional_int(value: object) -> int | None:
    if value in (None, "", "N/A"):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    return float(value)


def probe_stream(video_path: Path, *, ffprobe: str = "ffprobe") -> StreamInfo:
    video_path = video_path.expanduser().resolve()
    binary = _required_binary(ffprobe)
    command = [
        binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,profile,level,width,height,r_frame_rate,avg_frame_rate,time_base,"
            "start_time,duration,nb_frames,bit_rate,is_avc,nal_length_size,field_order:"
            "format=format_name,start_time,size"
        ),
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"no video stream found in {video_path}")
    stream = streams[0]
    codec_name = str(stream.get("codec_name", ""))
    if codec_name != "h264":
        raise ValueError(f"expected H.264 video, found {codec_name or 'unknown'}")
    format_info = payload.get("format", {})
    nal_length_size = int(stream.get("nal_length_size") or 4)
    stream_start = _optional_float(stream.get("start_time"))
    format_start = _optional_float(format_info.get("start_time"))
    return StreamInfo(
        codec_name=codec_name,
        profile=stream.get("profile"),
        level=_optional_int(stream.get("level")),
        width=int(stream["width"]),
        height=int(stream["height"]),
        nominal_frame_rate=str(stream.get("r_frame_rate", "0/0")),
        average_frame_rate=str(stream.get("avg_frame_rate", "0/0")),
        time_base=str(stream.get("time_base", "0/0")),
        start_time_seconds=float(
            stream_start if stream_start is not None else format_start or 0.0
        ),
        duration_seconds=_optional_float(stream.get("duration")),
        frame_count=_optional_int(stream.get("nb_frames")),
        bit_rate=_optional_int(stream.get("bit_rate")),
        is_avc=str(stream.get("is_avc", "true")).lower() in {"1", "true"},
        nal_length_size=nal_length_size,
        format_name=format_info.get("format_name"),
        file_size=int(format_info.get("size") or video_path.stat().st_size),
        field_order=stream.get("field_order"),
    )


def _read_interval(start_seconds: float | None, duration_seconds: float | None) -> str | None:
    if start_seconds is None and duration_seconds is None:
        return None
    start = 0.0 if start_seconds is None else start_seconds
    if duration_seconds is None:
        return f"{start}%"
    if start_seconds is None:
        return f"0%+{duration_seconds}"
    return f"{start}%{start + duration_seconds}"


def iter_packets(
    video_path: Path,
    *,
    ffprobe: str = "ffprobe",
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    max_packets: int | None = None,
    timestamp_origin_seconds: float = 0.0,
    reorder_padding_seconds: float = 2.0,
) -> Iterator[PacketInfo]:
    binary = _required_binary(ffprobe)
    command = [binary, "-v", "error", "-select_streams", "v:0"]
    scan_duration = (
        None
        if duration_seconds is None
        else duration_seconds + max(0.0, reorder_padding_seconds)
    )
    interval = _read_interval(start_seconds, scan_duration)
    if interval is not None:
        command.extend(["-read_intervals", interval])
    command.extend(
        [
            "-show_packets",
            "-show_entries",
            "packet=pts_time,dts_time,duration_time,pos,size,flags",
            "-of",
            "compact=p=0:nk=0",
            str(video_path),
        ]
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    emitted = 0
    decode_index = 0
    end_seconds = (
        None
        if duration_seconds is None
        else (0.0 if start_seconds is None else start_seconds) + duration_seconds
    )
    stopped_early = False
    try:
        for line in process.stdout:
            fields = {}
            for item in line.strip().split("|"):
                key, separator, value = item.partition("=")
                if separator:
                    fields[key] = value
            if not {"pts_time", "pos", "size"}.issubset(fields) or any(
                fields[key] == "N/A" for key in ("pts_time", "pos", "size")
            ):
                decode_index += 1
                continue
            pts = float(fields["pts_time"])
            if start_seconds is not None and pts < start_seconds:
                decode_index += 1
                continue
            if end_seconds is not None and pts >= end_seconds:
                decode_index += 1
                continue
            dts_text = fields.get("dts_time", fields["pts_time"])
            duration_text = fields.get("duration_time", "0")
            yield PacketInfo(
                pts=pts - timestamp_origin_seconds,
                dts=(float(dts_text) if dts_text != "N/A" else pts)
                - timestamp_origin_seconds,
                duration=float(duration_text) if duration_text != "N/A" else 0.0,
                position=int(fields["pos"]),
                size=int(fields["size"]),
                keyframe="K" in fields.get("flags", ""),
                decode_index=decode_index,
            )
            emitted += 1
            decode_index += 1
            if max_packets is not None and emitted >= max_packets:
                stopped_early = True
                break
    finally:
        if stopped_early:
            process.terminate()
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else ""
        return_code = process.wait()
        if process.stderr is not None:
            process.stderr.close()
        if not stopped_early and return_code != 0:
            raise RuntimeError(f"ffprobe packet scan failed: {stderr.strip()}")


def split_avcc_nals(payload: bytes, length_size: int) -> list[memoryview]:
    if length_size not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported AVCC NAL length size: {length_size}")
    view = memoryview(payload)
    offset = 0
    units: list[memoryview] = []
    while offset < len(view):
        if offset + length_size > len(view):
            raise ValueError("truncated AVCC NAL length")
        unit_size = int.from_bytes(view[offset : offset + length_size], "big")
        offset += length_size
        if unit_size <= 0 or offset + unit_size > len(view):
            raise ValueError("invalid AVCC NAL size")
        units.append(view[offset : offset + unit_size])
        offset += unit_size
    return units


def split_annex_b_nals(payload: bytes) -> list[memoryview]:
    starts: list[tuple[int, int]] = []
    index = 0
    while index + 3 <= len(payload):
        if payload[index : index + 3] == b"\x00\x00\x01":
            starts.append((index, 3))
            index += 3
        elif index + 4 <= len(payload) and payload[index : index + 4] == b"\x00\x00\x00\x01":
            starts.append((index, 4))
            index += 4
        else:
            index += 1
    if not starts:
        raise ValueError("packet is neither valid AVCC nor Annex B")
    view = memoryview(payload)
    units: list[memoryview] = []
    for unit_index, (start, prefix_size) in enumerate(starts):
        end = starts[unit_index + 1][0] if unit_index + 1 < len(starts) else len(payload)
        unit = view[start + prefix_size : end]
        while unit and unit[-1] == 0:
            unit = unit[:-1]
        if unit:
            units.append(unit)
    return units


def split_nals(payload: bytes, *, is_avc: bool, nal_length_size: int) -> list[memoryview]:
    if is_avc:
        return split_avcc_nals(payload, nal_length_size)
    return split_annex_b_nals(payload)


def _rbsp_from_ebsp(ebsp: memoryview) -> bytes:
    output = bytearray()
    zero_count = 0
    for value in ebsp:
        if zero_count >= 2 and value == 3:
            continue
        output.append(value)
        zero_count = zero_count + 1 if value == 0 else 0
    return bytes(output)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.bit_offset = 0

    def read_bit(self) -> int:
        if self.bit_offset >= len(self.data) * 8:
            raise EOFError("end of RBSP")
        byte = self.data[self.bit_offset // 8]
        shift = 7 - self.bit_offset % 8
        self.bit_offset += 1
        return (byte >> shift) & 1

    def read_ue(self) -> int:
        leading_zero_bits = 0
        while self.read_bit() == 0:
            leading_zero_bits += 1
            if leading_zero_bits > 31:
                raise ValueError("invalid Exp-Golomb code")
        suffix = 0
        for _ in range(leading_zero_bits):
            suffix = (suffix << 1) | self.read_bit()
        return (1 << leading_zero_bits) - 1 + suffix


@dataclass(frozen=True)
class SliceHeaderPrefix:
    first_mb_in_slice: int
    slice_type: int


@dataclass(frozen=True)
class _ExactRoiLayout:
    start_y: int
    start_macroblock: int
    macroblocks_per_row: int
    coded_macroblock_rows: int


@dataclass(frozen=True)
class _SpatialPayload:
    global_vcl_bytes: int
    roi_vcl_bytes: int
    above_roi_vcl_bytes: int
    roi_values: np.ndarray


def slice_header_prefix_from_nal(unit: memoryview) -> SliceHeaderPrefix | None:
    if not unit or (int(unit[0]) & 0x1F) not in SLICE_HEADER_NAL_TYPES:
        return None
    try:
        reader = _BitReader(_rbsp_from_ebsp(unit[1:]))
        first_mb_in_slice = reader.read_ue()
        slice_type = reader.read_ue() % 5
        return SliceHeaderPrefix(first_mb_in_slice, slice_type)
    except (EOFError, ValueError):
        return None


def slice_type_from_nal(unit: memoryview) -> int | None:
    prefix = slice_header_prefix_from_nal(unit)
    return None if prefix is None else prefix.slice_type


def _exact_roi_layout(stream: StreamInfo, settings: FeatureSettings) -> _ExactRoiLayout:
    if stream.width <= 0 or stream.height <= 0:
        raise ValueError("exact bottom-slice extraction requires positive stream dimensions")
    if stream.field_order != "progressive":
        raise ValueError(
            "exact bottom-slice extraction requires a stream reported as progressive; "
            f"ffprobe field_order={stream.field_order!r}"
        )
    start_y = math.ceil(stream.height * (1.0 - settings.payload_tail_ratio) - 1e-9)
    if start_y % MACROBLOCK_SIZE != 0:
        raise ValueError(
            "exact bottom-slice extraction requires a macroblock-aligned ROI boundary: "
            f"y={start_y} for height={stream.height} and ratio={settings.payload_tail_ratio}"
        )
    macroblocks_per_row = math.ceil(stream.width / MACROBLOCK_SIZE)
    coded_macroblock_rows = math.ceil(stream.height / MACROBLOCK_SIZE)
    return _ExactRoiLayout(
        start_y=start_y,
        start_macroblock=(start_y // MACROBLOCK_SIZE) * macroblocks_per_row,
        macroblocks_per_row=macroblocks_per_row,
        coded_macroblock_rows=coded_macroblock_rows,
    )


def _spatial_contract(stream: StreamInfo, settings: FeatureSettings) -> dict[str, object]:
    requested_roi = f"bottom_{settings.payload_tail_ratio * 100:g}_percent"
    if settings.spatial_mode == PAYLOAD_TAIL_PROXY:
        return {
            "requested_roi": requested_roi,
            "spatial_mode": PAYLOAD_TAIL_PROXY,
            "implemented_feature_scope": "last_fraction_of_global_VCL_payload_proxy",
            "payload_fraction": settings.payload_tail_ratio,
            "exact_pixel_roi": False,
            "warning": (
                "VCL byte offset is not an image-row mapping. This opt-in mode is only a "
                "non-spatial baseline."
            ),
        }
    layout = _exact_roi_layout(stream, settings)
    return {
        "requested_roi": requested_roi,
        "spatial_mode": EXACT_BOTTOM_SLICES,
        "implemented_feature_scope": "VCL_slice_payloads_at_or_below_ROI_macroblock_boundary",
        "exact_pixel_roi": True,
        "roi_height_fraction": settings.payload_tail_ratio,
        "roi_start_y": layout.start_y,
        "macroblock_size": MACROBLOCK_SIZE,
        "macroblocks_per_row": layout.macroblocks_per_row,
        "coded_macroblock_rows": layout.coded_macroblock_rows,
        "roi_start_macroblock": layout.start_macroblock,
        "slice_requirement": (
            "every packet has unique raster-ordered VCL slices starting at macroblocks 0 and "
            f"{layout.start_macroblock}"
        ),
        "payload_definition": "VCL NAL payload bytes excluding the one-byte NAL header",
        "display_padding_note": (
            "the selected bottom macroblock rows may include coded padding below display height"
        ),
    }


def feature_names(settings: FeatureSettings) -> list[str]:
    names = list(SCALAR_FEATURE_NAMES)
    for source in TEMPORAL_SIGNAL_NAMES:
        names.extend((f"delta_{source}", f"abs_delta_{source}"))
    names.extend(f"roi_hist_{index:02d}" for index in range(settings.histogram_bins))
    for segment in range(settings.payload_segments):
        names.extend(
            f"roi_segment_{segment:02d}_hist_{index:02d}"
            for index in range(settings.histogram_bins)
        )
        names.append(f"roi_segment_{segment:02d}_entropy")
    return names


def _normalized_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((bins,), dtype=np.float32)
    bucket_width = 256 // bins
    counts = np.bincount(values // bucket_width, minlength=bins)
    return counts.astype(np.float32) / float(values.size)


def _normalized_entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    counts = np.bincount(values, minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / float(values.size)
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    return entropy / 8.0


def _uniform_tokens(values: np.ndarray, token_count: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((token_count,), dtype=np.uint8)
    indices = np.linspace(0, values.size - 1, num=token_count, dtype=np.int64)
    return values[indices]


def _payload_tail(vcl_payload: bytes, ratio: float) -> np.ndarray:
    values = np.frombuffer(vcl_payload, dtype=np.uint8)
    if values.size == 0:
        return values
    tail_size = max(1, math.ceil(values.size * ratio))
    return values[-tail_size:]


def _select_spatial_payload(
    vcl_units: list[memoryview],
    *,
    packet: PacketInfo,
    stream: StreamInfo,
    settings: FeatureSettings,
) -> _SpatialPayload:
    if not vcl_units:
        raise ValueError(f"H.264 packet at PTS {packet.pts:.6f} contains no VCL NAL units")
    global_payload = b"".join(bytes(unit[1:]) for unit in vcl_units if len(unit) > 1)
    if not global_payload:
        raise ValueError(f"H.264 packet at PTS {packet.pts:.6f} has empty VCL payload")
    if settings.spatial_mode == PAYLOAD_TAIL_PROXY:
        roi_values = _payload_tail(global_payload, settings.payload_tail_ratio)
        return _SpatialPayload(
            global_vcl_bytes=len(global_payload),
            roi_vcl_bytes=int(roi_values.size),
            above_roi_vcl_bytes=len(global_payload) - int(roi_values.size),
            roi_values=roi_values,
        )

    layout = _exact_roi_layout(stream, settings)
    prefixes: list[SliceHeaderPrefix] = []
    for unit in vcl_units:
        nal_type = int(unit[0]) & 0x1F
        if nal_type not in EXACT_SLICE_NAL_TYPES:
            raise ValueError(
                "exact bottom-slice extraction does not support H.264 data partitioning: "
                f"NAL type {nal_type} at PTS {packet.pts:.6f}"
            )
        prefix = slice_header_prefix_from_nal(unit)
        if prefix is None:
            raise ValueError(
                f"cannot parse first_mb_in_slice at PTS {packet.pts:.6f}"
            )
        prefixes.append(prefix)

    slice_starts = [prefix.first_mb_in_slice for prefix in prefixes]
    maximum_macroblocks = layout.macroblocks_per_row * layout.coded_macroblock_rows
    if slice_starts != sorted(set(slice_starts)):
        raise ValueError(
            "exact bottom-slice extraction requires unique slices in increasing raster order: "
            f"PTS {packet.pts:.6f}, first_mb_in_slice={slice_starts}"
        )
    if slice_starts[0] != 0 or any(start >= maximum_macroblocks for start in slice_starts):
        raise ValueError(
            "exact bottom-slice extraction found an invalid slice layout: "
            f"PTS {packet.pts:.6f}, first_mb_in_slice={slice_starts}, "
            f"coded_macroblocks={maximum_macroblocks}"
        )
    if layout.start_macroblock not in slice_starts:
        raise ValueError(
            "exact bottom-slice extraction requires every frame to start a slice at the ROI "
            f"boundary: PTS {packet.pts:.6f}, y={layout.start_y}, "
            f"first_mb={layout.start_macroblock}, observed={slice_starts}"
        )

    roi_payload = b"".join(
        bytes(unit[1:])
        for unit, prefix in zip(vcl_units, prefixes, strict=True)
        if prefix.first_mb_in_slice >= layout.start_macroblock and len(unit) > 1
    )
    above_roi_vcl_bytes = sum(
        len(unit) - 1
        for unit, prefix in zip(vcl_units, prefixes, strict=True)
        if prefix.first_mb_in_slice < layout.start_macroblock
    )
    if not roi_payload or above_roi_vcl_bytes <= 0:
        raise ValueError(
            "exact bottom-slice extraction produced an empty spatial partition at "
            f"PTS {packet.pts:.6f}"
        )
    return _SpatialPayload(
        global_vcl_bytes=len(global_payload),
        roi_vcl_bytes=len(roi_payload),
        above_roi_vcl_bytes=above_roi_vcl_bytes,
        roi_values=np.frombuffer(roi_payload, dtype=np.uint8),
    )


def packet_features(
    packet: PacketInfo,
    payload: bytes,
    *,
    stream: StreamInfo,
    settings: FeatureSettings,
    presentation_step: float,
    decode_order_offset: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    units = split_nals(payload, is_avc=stream.is_avc, nal_length_size=stream.nal_length_size)
    unit_types = [int(unit[0]) & 0x1F for unit in units if unit]
    vcl_units = [
        unit
        for unit, unit_type in zip(units, unit_types, strict=True)
        if unit_type in VCL_NAL_TYPES
    ]
    filler_bytes = sum(
        len(unit)
        for unit, unit_type in zip(units, unit_types, strict=True)
        if unit_type == FILLER_NAL_TYPE
    )
    sei_bytes = sum(
        len(unit)
        for unit, unit_type in zip(units, unit_types, strict=True)
        if unit_type == SEI_NAL_TYPE
    )
    packet_bytes = max(1, len(payload))
    spatial = _select_spatial_payload(
        vcl_units,
        packet=packet,
        stream=stream,
        settings=settings,
    )
    slice_type = next(
        (
            parsed
            for unit in vcl_units
            if (parsed := slice_type_from_nal(unit)) is not None
        ),
        None,
    )
    global_vcl_bytes = spatial.global_vcl_bytes
    roi_vcl_bytes = spatial.roi_vcl_bytes
    above_roi_vcl_bytes = spatial.above_roi_vcl_bytes
    roi_to_above = roi_vcl_bytes / max(1, above_roi_vcl_bytes)
    roi_above_contrast = (roi_vcl_bytes - above_roi_vcl_bytes) / global_vcl_bytes

    scalars = np.asarray(
        [
            math.log1p(len(payload)),
            math.log1p(global_vcl_bytes),
            math.log1p(roi_vcl_bytes),
            math.log1p(above_roi_vcl_bytes),
            math.log1p(filler_bytes),
            math.log1p(sei_bytes),
            global_vcl_bytes / packet_bytes,
            roi_vcl_bytes / packet_bytes,
            above_roi_vcl_bytes / packet_bytes,
            roi_vcl_bytes / global_vcl_bytes,
            above_roi_vcl_bytes / global_vcl_bytes,
            roi_to_above,
            roi_above_contrast,
            filler_bytes / packet_bytes,
            sei_bytes / packet_bytes,
            packet.duration,
            packet.pts - packet.dts,
            presentation_step,
            float(packet.keyframe),
            float(np.clip(decode_order_offset, -8, 8)) / 8.0,
            math.log1p(len(units)),
            math.log1p(len(vcl_units)),
            float(slice_type == 0),
            float(slice_type == 1),
            float(slice_type == 2),
            float(slice_type not in {0, 1, 2}),
            float(5 in unit_types),
            float(7 in unit_types),
            float(8 in unit_types),
            float(9 in unit_types),
        ],
        dtype=np.float32,
    )
    temporal_placeholders = np.zeros((2 * len(TEMPORAL_SIGNAL_NAMES),), dtype=np.float32)
    parts = [
        scalars,
        temporal_placeholders,
        _normalized_histogram(spatial.roi_values, settings.histogram_bins),
    ]
    for segment in np.array_split(spatial.roi_values, settings.payload_segments):
        parts.append(_normalized_histogram(segment, settings.histogram_bins))
        parts.append(np.asarray([_normalized_entropy(segment)], dtype=np.float32))
    features = np.concatenate(parts)
    tokens = _uniform_tokens(spatial.roi_values, settings.token_count)
    byte_stats = {
        "packet_bytes": len(payload),
        "global_vcl_bytes": global_vcl_bytes,
        "roi_vcl_bytes": roi_vcl_bytes,
        "above_roi_vcl_bytes": above_roi_vcl_bytes,
        "filler_bytes": filler_bytes,
        "sei_bytes": sei_bytes,
    }
    return features, tokens, byte_stats


def _populate_temporal_signals(features: np.ndarray, names: list[str]) -> None:
    if len(features) == 0:
        return
    for source_name in TEMPORAL_SIGNAL_NAMES:
        source_index = names.index(source_name)
        delta_index = names.index(f"delta_{source_name}")
        absolute_delta_index = names.index(f"abs_delta_{source_name}")
        values = np.asarray(features[:, source_index], dtype=np.float32)
        delta = np.empty_like(values)
        delta[0] = 0.0
        delta[1:] = values[1:] - values[:-1]
        features[:, delta_index] = delta
        features[:, absolute_delta_index] = np.abs(delta)


def _source_id(path: Path) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    sample_size = 1024 * 1024
    with path.open("rb") as file:
        digest.update(file.read(sample_size))
        if stat.st_size > sample_size:
            file.seek(max(0, stat.st_size - sample_size))
            digest.update(file.read(sample_size))
    return f"container-edge-sha256:{digest.hexdigest()}"


def extract_feature_cache(
    video_path: Path,
    output_dir: Path,
    *,
    settings: FeatureSettings = FeatureSettings(),
    ffprobe: str = "ffprobe",
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    max_packets: int | None = None,
    overwrite: bool = False,
) -> dict:
    settings.validate()
    video_path = video_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    partial_dir = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if (
        video_path == output_dir
        or video_path.is_relative_to(output_dir)
        or video_path == partial_dir
        or video_path.is_relative_to(partial_dir)
    ):
        raise ValueError("feature-cache output directories must not contain the source video")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    stream = probe_stream(video_path, ffprobe=ffprobe)
    format_names = set((stream.format_name or "").split(","))
    if not stream.is_avc or not format_names.intersection(
        {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}
    ):
        raise NotImplementedError(
            "direct packet pread is currently verified only for AVCC samples in MP4/MOV-family "
            "containers; TS/MKV require a real demuxer-backed extractor"
        )
    spatial_contract = _spatial_contract(stream, settings)
    packet_table_path = partial_dir / "packet-table.bin"
    packet_count = 0
    with packet_table_path.open("wb") as packet_file:
        relative_start = 0.0 if start_seconds is None else start_seconds
        absolute_start = (
            None
            if start_seconds is None and duration_seconds is None
            else stream.start_time_seconds + relative_start
        )
        for packet in iter_packets(
            video_path,
            ffprobe=ffprobe,
            start_seconds=absolute_start,
            duration_seconds=duration_seconds,
            max_packets=max_packets,
            timestamp_origin_seconds=stream.start_time_seconds,
        ):
            packet_file.write(
                PACKET_RECORD.pack(
                    packet.pts,
                    packet.dts,
                    packet.duration,
                    packet.position,
                    packet.size,
                    int(packet.keyframe),
                )
            )
            packet_count += 1
    if packet_count == 0:
        raise ValueError("no H.264 packets selected")
    packet_table = np.memmap(
        packet_table_path, mode="r", dtype=PACKET_DTYPE, shape=(packet_count,)
    )

    presentation_order = np.lexsort((packet_table["dts"], packet_table["pts"]))
    presentation_index = np.empty((packet_count,), dtype=np.int64)
    for rank, packet_index in enumerate(presentation_order):
        presentation_index[packet_index] = rank
    ordered_pts = np.asarray(packet_table["pts"][presentation_order], dtype=np.float64)
    ordered_duration = np.asarray(packet_table["duration"][presentation_order], dtype=np.float32)
    steps = np.diff(ordered_pts, prepend=ordered_pts[0])
    if len(steps) > 1:
        positive_steps = steps[steps > 0]
        default_step = float(np.median(positive_steps)) if positive_steps.size else 0.0
        steps[0] = default_step

    names = feature_names(settings)
    features = open_memmap(
        partial_dir / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(packet_count, len(names)),
    )
    tokens = open_memmap(
        partial_dir / "tokens.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(packet_count, settings.token_count),
    )
    timestamps = open_memmap(
        partial_dir / "timestamps.npy", mode="w+", dtype=np.float64, shape=(packet_count,)
    )
    durations = open_memmap(
        partial_dir / "durations.npy", mode="w+", dtype=np.float32, shape=(packet_count,)
    )
    timestamps[:] = ordered_pts
    durations[:] = ordered_duration

    totals = {
        "packet_bytes": 0,
        "global_vcl_bytes": 0,
        "roi_vcl_bytes": 0,
        "above_roi_vcl_bytes": 0,
        "filler_bytes": 0,
        "sei_bytes": 0,
    }
    with video_path.open("rb") as video_file:
        for packet_index, row in enumerate(
            tqdm(packet_table, desc="H.264 packets", unit="packet", total=packet_count)
        ):
            packet = PacketInfo(
                pts=float(row["pts"]),
                dts=float(row["dts"]),
                duration=float(row["duration"]),
                position=int(row["position"]),
                size=int(row["size"]),
                keyframe=bool(row["keyframe"]),
                decode_index=packet_index,
            )
            payload = _read_packet(video_file, packet)
            rank = int(presentation_index[packet_index])
            vector, packet_tokens, byte_stats = packet_features(
                packet,
                payload,
                stream=stream,
                settings=settings,
                presentation_step=float(steps[rank]),
                decode_order_offset=packet_index - rank,
            )
            features[rank] = vector
            tokens[rank] = packet_tokens
            for key, value in byte_stats.items():
                totals[key] += value

    _populate_temporal_signals(features, names)
    features.flush()
    tokens.flush()
    timestamps.flush()
    durations.flush()
    artifact_sha256 = {
        name: file_sha256(partial_dir / name)
        for name in ("features.npy", "tokens.npy", "timestamps.npy", "durations.npy")
    }
    meta = {
        "format": FEATURE_FORMAT,
        "version": FEATURE_VERSION,
        "completed": True,
        "source": str(video_path),
        "source_id": _source_id(video_path),
        "source_id_kind": "container_size_plus_first_last_1MiB_sha256",
        "source_sha256": file_sha256(video_path),
        "source_sha256_kind": "full_file_sha256",
        "stream": asdict(stream),
        "selection": {
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
            "max_packets": max_packets,
        },
        "packet_count": packet_count,
        "time_range_seconds": [float(ordered_pts[0]), float(ordered_pts[-1])],
        "feature_names": names,
        "feature_settings": asdict(settings),
        "byte_totals": totals,
        "decode_contract": {
            "pixel_decode": False,
            "pixel_reconstruction": False,
            "entropy_decode": False,
            "input_operations": [
                "container_demux",
                "packet_pread",
                "NAL_split",
                "slice_header_prefix",
            ],
        },
        "spatial_contract": spatial_contract,
        "temporal_feature_contract": {
            "ordering": "presentation_timestamp",
            "delta_definition": "current_frame_minus_previous_frame",
            "absolute_delta_definition": "absolute_current_frame_minus_previous_frame",
            "first_frame_delta": 0.0,
            "cross_cache_context": False,
            "source_features": list(TEMPORAL_SIGNAL_NAMES),
        },
        "ordering": "presentation_timestamp",
        "timeline": "seconds relative to video stream start_time",
        "artifact_sha256": artifact_sha256,
        "container_contract": (
            "AVCC H.264 in MP4/MOV-family with packet.pos pointing to sample payload"
        ),
    }
    (partial_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del packet_table
    packet_table_path.unlink()
    partial_dir.rename(output_dir)
    return meta


def _read_packet(video_file: BinaryIO, packet: PacketInfo) -> bytes:
    video_file.seek(packet.position)
    payload = video_file.read(packet.size)
    if len(payload) != packet.size:
        raise EOFError(
            f"truncated packet at position {packet.position}: expected {packet.size}, "
            f"got {len(payload)}"
        )
    return payload
