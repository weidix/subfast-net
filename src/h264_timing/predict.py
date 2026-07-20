from __future__ import annotations

import numpy as np
import torch

from subtitle_timing_core.dataset import FeatureCache, window_starts
from .model import H264SubtitleSegmentModel


@torch.inference_mode()
def predict_cache(
    model: H264SubtitleSegmentModel,
    cache: FeatureCache,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window_frames: int,
    hop_frames: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Return direct segments plus auxiliary start/end event confidence."""
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if hop_frames <= 0 or hop_frames > window_frames:
        raise ValueError("hop_frames must be in [1, window_frames]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    feature_count = cache.features.shape[1]
    token_count = cache.tokens.shape[1]
    if feature_count != model.config.feature_count:
        raise ValueError(
            f"feature cache has {feature_count} numeric features, "
            f"model expects {model.config.feature_count}"
        )
    if token_count != model.config.token_count:
        raise ValueError(
            f"feature cache has {token_count} byte tokens, model expects {model.config.token_count}"
        )
    if feature_mean.shape != (feature_count,) or feature_std.shape != (feature_count,):
        raise ValueError("feature statistics must match the cache feature width")
    if not np.isfinite(feature_mean).all() or not np.isfinite(feature_std).all():
        raise ValueError("feature statistics must be finite")
    if np.any(feature_std <= 0):
        raise ValueError("feature standard deviations must be positive")
    frame_count = len(cache.timestamps)
    if frame_count == 0:
        return np.empty((0, 5), dtype=np.float32)
    starts = window_starts(frame_count, window_frames, hop_frames)
    score_sum = np.zeros((frame_count,), dtype=np.float32)
    start_offset_sum = np.zeros((frame_count,), dtype=np.float32)
    end_offset_sum = np.zeros((frame_count,), dtype=np.float32)
    boundary_event_sum = np.zeros((frame_count, 2), dtype=np.float32)
    score_weight = np.zeros((frame_count,), dtype=np.float32)
    window_weight = np.hanning(window_frames + 2)[1:-1].astype(np.float32)
    window_weight = np.maximum(window_weight, 0.05)
    model.eval()
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start : batch_start + batch_size]
        features = np.zeros(
            (len(batch_starts), window_frames, feature_count), dtype=np.float32
        )
        tokens = (
            np.zeros((len(batch_starts), window_frames, token_count), dtype=np.int64)
            if model.config.use_byte_branch
            else None
        )
        valid_lengths: list[int] = []
        for index, start in enumerate(batch_starts):
            stop = min(start + window_frames, frame_count)
            valid = stop - start
            valid_lengths.append(valid)
            features[index, :valid] = (
                np.asarray(cache.features[start:stop], dtype=np.float32) - feature_mean
            ) / feature_std
            if tokens is not None:
                tokens[index, :valid] = np.asarray(cache.tokens[start:stop], dtype=np.int64)
        token_tensor = (
            torch.from_numpy(tokens).to(device)
            if tokens is not None
            else torch.empty(
                (len(batch_starts), window_frames, token_count), dtype=torch.uint8
            )
        )
        output = model(torch.from_numpy(features).to(device), token_tensor)
        scores = torch.sigmoid(output.score_logits).cpu().numpy()
        start_offsets = output.start_offsets_seconds.cpu().numpy()
        end_offsets = output.end_offsets_seconds.cpu().numpy()
        boundary_events = torch.sigmoid(output.boundary_event_logits).cpu().numpy()
        for index, (start, valid) in enumerate(
            zip(batch_starts, valid_lengths, strict=True)
        ):
            weight = window_weight[:valid]
            score_sum[start : start + valid] += scores[index, :valid] * weight
            start_offset_sum[start : start + valid] += start_offsets[index, :valid] * weight
            end_offset_sum[start : start + valid] += end_offsets[index, :valid] * weight
            boundary_event_sum[start : start + valid] += (
                boundary_events[index, :valid] * weight[:, None]
            )
            score_weight[start : start + valid] += weight
    if np.any(score_weight <= 0):
        raise RuntimeError("inference windows did not cover every compressed frame")
    anchors = np.asarray(cache.timestamps, dtype=np.float64)
    scores = score_sum / score_weight
    predicted_starts = anchors + start_offset_sum / score_weight
    predicted_ends = anchors + end_offset_sum / score_weight
    boundary_events = boundary_event_sum / score_weight[:, None]
    return np.column_stack(
        (scores, predicted_starts, predicted_ends, boundary_events)
    ).astype(np.float32)
