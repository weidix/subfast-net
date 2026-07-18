from __future__ import annotations

import numpy as np
import torch

from .dataset import FeatureCache
from .stream_model import StreamingH264SubtitleModel, StreamingModelState


@torch.inference_mode()
def predict_stream_cache(
    model: StreamingH264SubtitleModel,
    cache: FeatureCache,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    chunk_frames: int,
    device: torch.device,
) -> np.ndarray:
    """Evaluate a cache causally; chunk boundaries do not change the result."""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    feature_count = cache.features.shape[1]
    token_count = cache.tokens.shape[1]
    if feature_count != model.config.feature_count:
        raise ValueError(
            f"feature cache has {feature_count} numeric features, "
            f"model expects {model.config.feature_count}"
        )
    if token_count != model.config.token_count:
        raise ValueError(
            f"feature cache has {token_count} byte tokens, "
            f"model expects {model.config.token_count}"
        )
    if feature_mean.shape != (feature_count,) or feature_std.shape != (feature_count,):
        raise ValueError("feature statistics must match the cache feature width")
    if not np.isfinite(feature_mean).all() or not np.isfinite(feature_std).all():
        raise ValueError("feature statistics must be finite")
    if np.any(feature_std <= 0.0):
        raise ValueError("feature standard deviations must be positive")
    timestamps = np.asarray(cache.timestamps, dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("cache timestamps must be finite and strictly increasing")
    frame_count = len(timestamps)
    predictions = np.empty((frame_count, 3), dtype=np.float32)
    state: StreamingModelState | None = None
    model.eval()
    for start in range(0, frame_count, chunk_frames):
        stop = min(start + chunk_frames, frame_count)
        features = (
            np.asarray(cache.features[start:stop], dtype=np.float32) - feature_mean
        ) / feature_std
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        if model.config.use_byte_branch:
            tokens = (
                torch.from_numpy(np.asarray(cache.tokens[start:stop], dtype=np.int64))
                .unsqueeze(0)
                .to(device)
            )
        else:
            tokens = torch.empty(
                (1, stop - start, token_count),
                dtype=torch.int64,
                device=device,
            )
        output, state = model.forward_stream(feature_tensor, tokens, state)
        predictions[start:stop, 0] = (
            torch.sigmoid(output.presence_logits[0]).cpu().numpy()
        )
        predictions[start:stop, 1:] = (
            torch.sigmoid(output.boundary_event_logits[0]).cpu().numpy()
        )
    return predictions
