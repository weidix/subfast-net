from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        normalization: str = "none",
    ) -> None:
        if normalization == "none":
            norm: nn.Module = nn.Identity()
        elif normalization == "group_norm":
            groups = next(group for group in range(min(8, output_channels), 0, -1) if output_channels % group == 0)
            norm = nn.GroupNorm(groups, output_channels)
        else:
            raise ValueError(f"unsupported normalization: {normalization}")
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=dilation * (kernel_size // 2),
                dilation=dilation,
                bias=normalization == "none",
            ),
            norm,
            nn.ReLU(inplace=True),
        )


class FramePresenceModel(nn.Module):
    """Context-fused full-frame subtitle-presence model with dense supervision."""

    model_name = "Frame Presence V5"
    architecture_version = 5
    feature_stride = 8

    def __init__(self, *, width: int = 24, normalization: str = "none") -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if normalization not in {"none", "group_norm"}:
            raise ValueError("normalization must be 'none' or 'group_norm'")
        self.normalization = normalization
        stem_width = max(8, width * 2 // 3)
        region_width = width + width // 3
        context_width = width * 2
        self.stem = ConvNormAct(3, stem_width, kernel_size=5, stride=2, normalization=normalization)
        self.detail = ConvNormAct(stem_width, width, kernel_size=3, stride=2, normalization=normalization)
        self.region_encoder = ConvNormAct(
            width, region_width, kernel_size=3, stride=2, normalization=normalization
        )
        self.context_encoder = ConvNormAct(
            region_width, context_width, kernel_size=3, stride=2, normalization=normalization
        )
        self.context_refine = ConvNormAct(
            context_width,
            context_width,
            kernel_size=3,
            stride=1,
            dilation=3,
            normalization=normalization,
        )
        self.context_projection = nn.Conv2d(context_width, region_width, 1)
        self.region_head = nn.Conv2d(region_width, 1, 1)
        self.global_head = nn.Linear(context_width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5**0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
    """Return an eval-only deployment copy; V5 has no foldable BatchNorm."""
    device = next(model.parameters()).device
    return copy.deepcopy(model).to("cpu").eval().to(device)


__all__ = ["FramePresenceModel", "fuse_frame_presence_for_inference"]
