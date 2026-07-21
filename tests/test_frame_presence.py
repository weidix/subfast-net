from __future__ import annotations

import torch

from subfast_frame_presence.data import rasterize_boxes
from subfast_frame_presence.metrics import region_metrics


def test_frame_presence_box_rasterization_is_conservative() -> None:
    target, bounds = rasterize_boxes(
        [(357.2, 930.1, 641.1, 994.2)],
        source_width=1920,
        source_height=1080,
        heatmap_width=64,
        heatmap_height=72,
    )

    assert bounds.tolist() == [11, 62, 22, 67]
    assert target[62:67, 11:22].all()


def test_full_frame_region_cannot_pass_area_guard() -> None:
    target = torch.zeros((1, 1, 72, 64))
    target[:, :, 66:72, 29:35] = 1.0
    full_frame_logits = torch.full_like(target, 12.0)

    metrics, _ = region_metrics(
        full_frame_logits,
        target,
        torch.ones((1,)),
        threshold=0.5,
    )

    assert metrics["region_bbox_containment_rate"] == 1.0
    assert metrics["region_area_limit_pass_rate"] == 0.0
    assert metrics["region_bbox_full_frame_ratio_max"] == 1.0
