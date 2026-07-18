from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def resize_valid_mask(valid_mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize validity with the same majority-area rule used by dense supervision."""
    if valid_mask.shape[-2:] == size:
        resized = valid_mask
    else:
        input_height, input_width = valid_mask.shape[-2:]
        output_height, output_width = size
        if (
            input_height % output_height == 0
            and input_width % output_width == 0
            and input_height >= output_height
            and input_width >= output_width
        ):
            kernel = (input_height // output_height, input_width // output_width)
            resized = F.avg_pool2d(valid_mask, kernel_size=kernel, stride=kernel)
        else:
            resized = F.interpolate(valid_mask, size=size, mode="area")
    return (resized > 0.5).to(valid_mask.dtype)


class CoherentEvidencePooling(nn.Module):
    """Select the strongest locally-supported response inside the valid ROI."""

    def __init__(self, *, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer greater than 1")
        self.support = nn.AvgPool2d(
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        region_logits: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is not None:
            if valid_mask.shape != region_logits.shape:
                raise ValueError("valid_mask must use the region-logit coordinate space")
            valid_mask = (valid_mask > 0.5).to(region_logits.dtype)
            supported_valid = self.support(valid_mask)
            coherent_logits = self.support(region_logits * valid_mask) / supported_valid.clamp_min(1e-6)
            valid_locations = valid_mask > 0.5
            coherent_logits = coherent_logits.masked_fill(
                ~valid_locations,
                torch.finfo(coherent_logits.dtype).min,
            )
        else:
            coherent_logits = self.support(region_logits)
        evidence = coherent_logits.amax(dim=(1, 2, 3)).clamp_min(-1e4)
        positive_scale = F.softplus(self.log_scale) + 0.5
        return positive_scale * evidence + self.bias


class SubtitleRegionPresenceHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        *,
        evidence_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.local = nn.Conv2d(
            feature_dim,
            1,
            kernel_size=3,
            padding=2,
            dilation=2,
        )
        self.evidence_pool = CoherentEvidencePooling(kernel_size=evidence_kernel_size)

    def region_logits(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.local(feature_map)

    def forward(
        self,
        feature_map: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.evidence_pool(self.region_logits(feature_map), valid_mask)


class RoiPresenceModel(nn.Module):
    architecture_version = 3
    feature_stride = 4

    def __init__(
        self,
        width: int = 16,
        *,
        evidence_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width * 2, 5, stride=2, padding=2),
            nn.SiLU(inplace=True),
        )
        self.presence_head = SubtitleRegionPresenceHead(
            width * 2,
            evidence_kernel_size=evidence_kernel_size,
        )

    def encode_map(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    @staticmethod
    def input_valid_mask(images: torch.Tensor) -> torch.Tensor:
        """Recover the exact zero padding emitted by the letterbox preprocessor."""
        return (images.abs().amax(dim=1, keepdim=True) > 1e-8).to(images.dtype)

    def feature_valid_mask(self, images: torch.Tensor) -> torch.Tensor:
        """Recover validity directly at the model's stride-4 output locations."""
        feature_size = (
            (images.shape[-2] + self.feature_stride - 1) // self.feature_stride,
            (images.shape[-1] + self.feature_stride - 1) // self.feature_stride,
        )
        return resize_valid_mask(self.input_valid_mask(images), feature_size)

    def downsample_valid_mask(
        self,
        valid_mask: torch.Tensor,
        region_logits: torch.Tensor,
    ) -> torch.Tensor:
        return resize_valid_mask(valid_mask, region_logits.shape[-2:]).to(dtype=region_logits.dtype)

    def _region_logits_and_valid_mask(
        self,
        images: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        region_logits = self.presence_head.region_logits(self.encode_map(images))
        region_valid_mask = (
            self.feature_valid_mask(images)
            if valid_mask is None
            else self.downsample_valid_mask(valid_mask, region_logits)
        )
        return region_logits, region_valid_mask

    def forward_with_presence_map(
        self,
        images: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        region_logits, region_valid_mask = self._region_logits_and_valid_mask(images, valid_mask)
        return self.presence_head.evidence_pool(region_logits, region_valid_mask), region_logits

    def forward(
        self,
        images: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        region_logits, region_valid_mask = self._region_logits_and_valid_mask(images, valid_mask)
        return self.presence_head.evidence_pool(region_logits, region_valid_mask)
