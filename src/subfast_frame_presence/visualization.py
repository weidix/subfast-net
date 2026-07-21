from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _source_bounds(
    active: np.ndarray,
    *,
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int] | None:
    rows, columns = np.nonzero(active)
    if not len(rows):
        return None
    heatmap_height, heatmap_width = active.shape
    return (
        int(np.floor(columns.min() / heatmap_width * source_width)),
        int(np.floor(rows.min() / heatmap_height * source_height)),
        int(np.ceil((columns.max() + 1) / heatmap_width * source_width)),
        int(np.ceil((rows.max() + 1) / heatmap_height * source_height)),
    )


def _zoom_crop(
    bounds: tuple[int, int, int, int] | None,
    *,
    source_width: int,
    source_height: int,
    aspect_ratio: float,
) -> tuple[int, int, int, int]:
    if bounds is None:
        bounds = (
            round(source_width * 0.25),
            round(source_height * 0.75),
            round(source_width * 0.75),
            source_height,
        )
    left, top, right, bottom = bounds
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    crop_width = max((right - left) * 1.5, source_width * 0.5)
    crop_height = max((bottom - top) * 3.0, source_height * 0.25)
    if crop_width / crop_height < aspect_ratio:
        crop_width = crop_height * aspect_ratio
    else:
        crop_height = crop_width / aspect_ratio
    crop_width = min(float(source_width), crop_width)
    crop_height = min(float(source_height), crop_height)
    crop_left = min(max(0.0, center_x - crop_width / 2.0), source_width - crop_width)
    crop_top = min(max(0.0, center_y - crop_height / 2.0), source_height - crop_height)
    return (
        round(crop_left),
        round(crop_top),
        round(crop_left + crop_width),
        round(crop_top + crop_height),
    )


def save_frame_presence_visualization(
    frame_rgb: np.ndarray,
    heatmap: np.ndarray,
    *,
    presence_score: float,
    heatmap_threshold: float,
    output: Path,
) -> dict[str, object]:
    """Save original, raw-heatmap, and overlay views for one inference frame."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must have shape HxWx3")
    if heatmap.ndim != 2:
        raise ValueError("heatmap must have shape HxW")
    if not 0.0 < heatmap_threshold < 1.0:
        raise ValueError("heatmap_threshold must be between zero and one")

    frame = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    source_height, source_width = frame.shape[:2]
    active = heatmap >= heatmap_threshold
    denominator = max(1e-6, 1.0 - heatmap_threshold)
    active_strength = np.clip(
        (heatmap - heatmap_threshold) / denominator,
        0.0,
        1.0,
    )
    color_small = cv2.applyColorMap(
        np.round(active_strength * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    strength = cv2.resize(
        active_strength,
        (source_width, source_height),
        interpolation=cv2.INTER_NEAREST,
    )
    active_source = cv2.resize(
        active.astype(np.uint8),
        (source_width, source_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    color = cv2.resize(
        color_small,
        (source_width, source_height),
        interpolation=cv2.INTER_NEAREST,
    )
    alpha = np.where(active_source, 0.42 + 0.48 * strength, 0.0).astype(
        np.float32
    )[..., None]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dark = np.round(dark.astype(np.float32) * 0.19).astype(np.uint8)
    raw_panel = np.round(
        dark.astype(np.float32) * (1.0 - alpha)
        + color.astype(np.float32) * alpha
    ).astype(np.uint8)
    overlay_panel = np.round(
        frame.astype(np.float32) * (1.0 - 0.72 * alpha)
        + color.astype(np.float32) * (0.72 * alpha)
    ).astype(np.uint8)

    predicted_bounds = _source_bounds(
        active,
        source_width=source_width,
        source_height=source_height,
    )
    if predicted_bounds is not None:
        left, top, right, bottom = predicted_bounds
        cv2.rectangle(
            overlay_panel,
            (left, top),
            (right, bottom),
            (70, 235, 115),
            4,
            cv2.LINE_AA,
        )

    heatmap_height, heatmap_width = active.shape
    for row, column in np.argwhere(active):
        left = int(np.floor(column / heatmap_width * source_width))
        right = int(np.ceil((column + 1) / heatmap_width * source_width)) - 1
        top = int(np.floor(row / heatmap_height * source_height))
        bottom = int(np.ceil((row + 1) / heatmap_height * source_height)) - 1
        for panel in (raw_panel, overlay_panel):
            cv2.rectangle(
                panel,
                (left, top),
                (right, bottom),
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )

    panel_width = 640
    full_height = 360
    zoom_height = 180
    header_height = 76
    footer_height = 70
    gap = 6
    canvas_width = panel_width * 3 + gap * 2
    canvas_height = header_height + full_height + gap + zoom_height + footer_height
    canvas = np.full((canvas_height, canvas_width, 3), (16, 18, 23), dtype=np.uint8)
    panels = (frame, raw_panel, overlay_panel)
    labels = (
        "Original frame",
        f"Raw {heatmap_width} x {heatmap_height} model heatmap",
        "Heatmap overlay + enclosing bounds",
    )
    crop_left, crop_top, crop_right, crop_bottom = _zoom_crop(
        predicted_bounds,
        source_width=source_width,
        source_height=source_height,
        aspect_ratio=panel_width / zoom_height,
    )
    for column, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        panel_left = column * (panel_width + gap)
        full = cv2.resize(panel, (panel_width, full_height), interpolation=cv2.INTER_AREA)
        canvas[
            header_height : header_height + full_height,
            panel_left : panel_left + panel_width,
        ] = full
        crop = panel[crop_top:crop_bottom, crop_left:crop_right]
        zoom = cv2.resize(crop, (panel_width, zoom_height), interpolation=cv2.INTER_AREA)
        zoom_top = header_height + full_height + gap
        canvas[
            zoom_top : zoom_top + zoom_height,
            panel_left : panel_left + panel_width,
        ] = zoom
        cv2.putText(
            canvas,
            label,
            (panel_left + 18, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (238, 241, 247),
            2,
            cv2.LINE_AA,
        )

    active_count = int(active.sum())
    footer_top = header_height + full_height + gap + zoom_height
    cv2.putText(
        canvas,
        (
            f"Presence={presence_score:.6f}    Active heatmap cells={active_count}"
            f"    Threshold={heatmap_threshold:g}"
        ),
        (18, footer_top + 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (222, 226, 234),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            "Color = predicted contour   |   Green = enclosing bounds"
            "   |   Lower row = predicted-area zoom"
        ),
        (18, footer_top + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (161, 170, 186),
        1,
        cv2.LINE_AA,
    )

    output = output.expanduser().resolve()
    if output.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("visualization output must use .jpg, .jpeg, .png, or .webp")
    output.parent.mkdir(parents=True, exist_ok=True)
    options = [cv2.IMWRITE_JPEG_QUALITY, 94] if output.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(output), canvas, options):
        raise RuntimeError(f"failed to write visualization: {output}")
    heatmap_bounds = None
    if predicted_bounds is not None:
        rows, columns = np.nonzero(active)
        heatmap_bounds = [
            int(columns.min()),
            int(rows.min()),
            int(columns.max() + 1),
            int(rows.max() + 1),
        ]
    return {
        "output": str(output),
        "presence_score": presence_score,
        "heatmap_threshold": heatmap_threshold,
        "active_cells": active_count,
        "heatmap_bounds": heatmap_bounds,
        "source_bounds": list(predicted_bounds) if predicted_bounds is not None else None,
    }


__all__ = ["save_frame_presence_visualization"]
