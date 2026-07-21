from __future__ import annotations

from collections.abc import Mapping
from typing import Any


FRAME_PRESENCE_CHECKPOINT_KIND = "subfast_frame_presence_training_checkpoint"


def checkpoint_model_type(checkpoint: Mapping[str, Any]) -> str | None:
    model_type = checkpoint.get("model_type")
    if model_type is not None:
        return str(model_type)
    if checkpoint.get("kind") == FRAME_PRESENCE_CHECKPOINT_KIND:
        return "frame_presence"
    return None


def checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any] | None:
    key = "model_state" if checkpoint_model_type(checkpoint) == "frame_presence" else "model"
    state_dict = checkpoint.get(key)
    return state_dict if isinstance(state_dict, Mapping) else None


__all__ = ["checkpoint_model_type", "checkpoint_state_dict"]
