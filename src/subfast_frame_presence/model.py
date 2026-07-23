from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=dilation * (kernel_size // 2),
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class FramePresenceModel(nn.Module):
    """Context-fused full-frame subtitle-presence model with dense supervision."""

    architecture_version = 4
    feature_stride = 8

    def __init__(self, *, width: int = 24) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        stem_width = max(8, width * 2 // 3)
        region_width = width + width // 3
        context_width = width * 2
        self.stem = ConvNormAct(3, stem_width, kernel_size=5, stride=2)
        self.detail = ConvNormAct(stem_width, width, kernel_size=3, stride=2)
        self.region_encoder = ConvNormAct(width, region_width, kernel_size=3, stride=2)
        self.context_encoder = ConvNormAct(region_width, context_width, kernel_size=3, stride=2)
        self.context_refine = ConvNormAct(
            context_width,
            context_width,
            kernel_size=3,
            stride=1,
            dilation=3,
        )
        self.context_projection = nn.Conv2d(context_width, region_width, 1, bias=False)
        self.region_head = nn.Conv2d(region_width, 1, 1)
        self.global_head = nn.Linear(context_width, 1)

    def encode_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        region = self.region_encoder(self.detail(self.stem(images)))
        context = self.context_refine(self.context_encoder(region))
        region = region + F.interpolate(
            self.context_projection(context),
            size=region.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return region, context

    def forward_with_presence_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        region_features, context = self.encode_map(images)
        region_logits = self.region_head(region_features)
        coherent_logits = F.avg_pool2d(
            region_logits,
            3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        local_evidence = coherent_logits.amax(dim=(1, 2, 3))
        global_evidence = self.global_head(context.mean(dim=(2, 3))).squeeze(1)
        return local_evidence + global_evidence, region_logits

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        presence_logits, _ = self.forward_with_presence_map(images)
        return presence_logits


def fuse_frame_presence_for_inference(model: FramePresenceModel) -> FramePresenceModel:
    """Return an eval-only copy with every Conv-BatchNorm pair folded."""
    device = next(model.parameters()).device
    optimized = copy.deepcopy(model).to("cpu").eval()
    for block in (
        optimized.stem,
        optimized.detail,
        optimized.region_encoder,
        optimized.context_encoder,
        optimized.context_refine,
    ):
        convolution = block[0]
        normalization = block[1]
        if not isinstance(convolution, nn.Conv2d) or not isinstance(normalization, nn.BatchNorm2d):
            raise TypeError("frame presence feature blocks must contain Conv2d-BatchNorm2d")
        block[0] = fuse_conv_bn_eval(convolution, normalization)
        block[1] = nn.Identity()
    return optimized.to(device)


__all__ = ["FramePresenceModel", "fuse_frame_presence_for_inference"]
