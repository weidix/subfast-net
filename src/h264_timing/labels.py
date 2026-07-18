from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, order=True)
class SubtitleInterval:
    start_seconds: float
    end_seconds: float
    label: str = "subtitle"

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_seconds) or not math.isfinite(self.end_seconds):
            raise ValueError("subtitle interval times must be finite")
        if self.start_seconds < 0.0:
            raise ValueError("subtitle interval start must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("subtitle interval end must be greater than start")


@dataclass(frozen=True)
class InvalidSrtInterval:
    line_number: int
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class SrtCue:
    line_number: int
    start_seconds: float
    end_seconds: float
    payload_lines: tuple[str, ...]


def read_intervals(path: Path) -> list[SubtitleInterval]:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".srt":
        return _read_srt_intervals(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"unsupported subtitle label format: {path.suffix or '<none>'}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"start_seconds", "end_seconds"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: start_seconds,end_seconds")
        intervals = [
            SubtitleInterval(
                start_seconds=float(row["start_seconds"]),
                end_seconds=float(row["end_seconds"]),
                label=(row.get("label") or "subtitle").strip(),
            )
            for row in reader
            if row.get("start_seconds") and row.get("end_seconds")
        ]
    return _validate_non_overlapping(sorted(intervals), source=path)


_SRT_TIMING = re.compile(
    r"^\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d+):(\d{2}):(\d{2})[,.](\d{3})(?:\s+.*)?$"
)


def _srt_seconds(groups: tuple[str, ...]) -> float:
    hours, minutes, seconds, milliseconds = (int(value) for value in groups)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("invalid SRT timestamp")
    return hours * 3600.0 + minutes * 60.0 + seconds + milliseconds / 1000.0


def _read_srt_intervals(path: Path) -> list[SubtitleInterval]:
    intervals, invalid = parse_srt_timing(path, drop_invalid=False)
    if invalid:
        raise AssertionError("strict SRT parsing cannot return invalid intervals")
    return intervals


def parse_srt_timing(
    path: Path, *, drop_invalid: bool
) -> tuple[list[SubtitleInterval], list[InvalidSrtInterval]]:
    cues, invalid = parse_srt_cues(path, drop_invalid=drop_invalid)
    return [
        SubtitleInterval(cue.start_seconds, cue.end_seconds) for cue in cues
    ], invalid


def parse_srt_cues(
    path: Path, *, drop_invalid: bool
) -> tuple[list[SrtCue], list[InvalidSrtInterval]]:
    path = path.expanduser().resolve()
    cues: list[SrtCue] = []
    invalid: list[InvalidSrtInterval] = []
    for block in _iter_srt_blocks(path):
        line_number, timing_line, payload_lines = _cue_from_block(block, path)
        match = _SRT_TIMING.match(timing_line)
        if match is None:
            raise ValueError(f"invalid SRT timing line {line_number} in {path}")
        groups = match.groups()
        start_seconds = _srt_seconds(groups[:4])
        end_seconds = _srt_seconds(groups[4:])
        if end_seconds <= start_seconds:
            issue = InvalidSrtInterval(line_number, start_seconds, end_seconds)
            if not drop_invalid:
                raise ValueError(
                    f"SRT timing line {line_number} ends before it starts: "
                    f"{start_seconds:.3f} -> {end_seconds:.3f}"
                )
            invalid.append(issue)
            continue
        cues.append(
            SrtCue(
                line_number=line_number,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                payload_lines=payload_lines,
            )
        )
    if not cues:
        raise ValueError(f"no SRT timing intervals found in {path}")
    cues.sort(key=lambda cue: (cue.start_seconds, cue.end_seconds, cue.line_number))
    _validate_non_overlapping(
        [SubtitleInterval(cue.start_seconds, cue.end_seconds) for cue in cues],
        source=path,
    )
    return cues, invalid


def _validate_non_overlapping(
    intervals: list[SubtitleInterval], *, source: Path
) -> list[SubtitleInterval]:
    for previous, current in zip(intervals, intervals[1:], strict=False):
        if current.start_seconds < previous.end_seconds:
            raise ValueError(
                "overlapping subtitle intervals are not supported by the non-overlapping "
                f"interval decoder ({source}): "
                f"{previous.start_seconds:.3f}-{previous.end_seconds:.3f} overlaps "
                f"{current.start_seconds:.3f}-{current.end_seconds:.3f}"
            )
    return intervals


def _iter_srt_blocks(path: Path):
    with path.open("r", encoding="utf-8-sig", newline=None) as file:
        block: list[tuple[int, str]] = []
        for line_number, line in enumerate(file, start=1):
            stripped = line.rstrip("\r\n")
            if stripped.strip():
                block.append((line_number, stripped))
                continue
            if block:
                yield block
            block = []
        if block:
            yield block


def _cue_from_block(
    block: list[tuple[int, str]], path: Path
) -> tuple[int, str, tuple[str, ...]]:
    candidate_index = 1 if block[0][1].strip().isdigit() else 0
    if candidate_index >= len(block):
        raise ValueError(f"SRT cue at line {block[0][0]} has no timing line in {path}")
    line_number, candidate = block[candidate_index]
    if "-->" not in candidate:
        raise ValueError(f"SRT cue at line {block[0][0]} has no timing arrow in {path}")
    payload = block[candidate_index + 1 :]
    if not payload:
        raise ValueError(f"SRT cue at line {block[0][0]} has no subtitle payload in {path}")
    for index, (extra_line_number, extra_line) in enumerate(payload):
        previous_is_index = index > 0 and payload[index - 1][1].strip().isdigit()
        if _SRT_TIMING.match(extra_line) or (previous_is_index and "-->" in extra_line):
            raise ValueError(
                "SRT cues must be separated by a blank line; "
                f"found another timing line at {extra_line_number} in {path}"
            )
    return line_number, candidate, tuple(line for _, line in payload)


def write_intervals(path: Path, intervals: list[SubtitleInterval]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["start_seconds", "end_seconds", "duration_seconds", "label"])
        for interval in intervals:
            writer.writerow(
                [
                    f"{interval.start_seconds:.6f}",
                    f"{interval.end_seconds:.6f}",
                    f"{interval.end_seconds - interval.start_seconds:.6f}",
                    interval.label,
                ]
            )


def segment_targets_from_intervals(
    timestamps: np.ndarray,
    intervals: list[SubtitleInterval],
) -> np.ndarray:
    """Encode FCOS-style centerness and paired boundaries for direct segments."""
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0.0):
        raise ValueError("timestamps must be finite and sorted")
    targets = np.zeros((len(timestamps), 3), dtype=np.float32)
    positive_steps = np.diff(timestamps)
    positive_steps = positive_steps[positive_steps > 0.0]
    half_step = (
        float(np.median(positive_steps)) / 2.0
        if positive_steps.size
        else 1.0 / 60.0
    )
    for interval in intervals:
        left = int(np.searchsorted(timestamps, interval.start_seconds, side="left"))
        right = int(np.searchsorted(timestamps, interval.end_seconds, side="left"))
        if right <= left:
            continue
        anchors = timestamps[left:right]
        start_distances = anchors - interval.start_seconds
        end_distances = interval.end_seconds - anchors
        centered_start_distances = start_distances + half_step
        centered_end_distances = np.maximum(end_distances - half_step, half_step)
        centerness = np.sqrt(
            np.minimum(centered_start_distances, centered_end_distances)
            / np.maximum(centered_start_distances, centered_end_distances)
        ).astype(np.float32)
        targets[left:right, 0] = centerness
        targets[left:right, 1] = (-start_distances).astype(np.float32)
        targets[left:right, 2] = end_distances.astype(np.float32)
    return targets


def boundary_event_targets_from_intervals(
    timestamps: np.ndarray,
    intervals: list[SubtitleInterval],
    *,
    sigma_seconds: float,
) -> np.ndarray:
    """Build auxiliary start/end event heatmaps without defining output segments."""
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if sigma_seconds <= 0.0:
        raise ValueError("boundary event sigma must be positive")
    targets = np.zeros((len(timestamps), 2), dtype=np.float32)
    for interval in intervals:
        for channel, center in (
            (0, interval.start_seconds),
            (1, interval.end_seconds),
        ):
            radius = 3.5 * sigma_seconds
            left = int(np.searchsorted(timestamps, center - radius, side="left"))
            right = int(np.searchsorted(timestamps, center + radius, side="right"))
            if right <= left:
                continue
            distance = (timestamps[left:right] - center) / sigma_seconds
            values = np.exp(-0.5 * distance * distance).astype(np.float32)
            targets[left:right, channel] = np.maximum(
                targets[left:right, channel], values
            )
    return targets
