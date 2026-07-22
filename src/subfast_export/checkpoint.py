from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def checkpoint_model_type(checkpoint: Mapping[str, Any]) -> str | None:
    model_type = checkpoint.get("model_type")
    if model_type is not None:
        return str(model_type)
    return None


def checkpoint_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any] | None:
    state_dict = checkpoint.get("model")
    return state_dict if isinstance(state_dict, Mapping) else None


__all__ = ["checkpoint_model_type", "checkpoint_state_dict"]
