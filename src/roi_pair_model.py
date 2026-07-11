from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class RoiPairMatcher(nn.Module):
    """Directly predicts whether two aligned subtitle ROIs contain the same subtitle.

    The pair representation is symmetric by construction. It keeps the shared
    appearance, the raw change, and the local high-frequency change as separate
    signals, so the network can ignore changing video backgrounds without losing a
    one-character subtitle difference.
    """

    pooling_version = 2

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvNormAct(9, 16, stride=2)
        self.downsample = ConvNormAct(16, 32, stride=2)
        self.detail = DepthwiseSeparableBlock(32, 32)
        self.vertical_downsample = DepthwiseSeparableBlock(32, 48, stride=(2, 1))
        self.context = DepthwiseSeparableBlock(48, 48)
        self.text_mask = nn.Conv2d(48, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(48 * 3, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )

    @staticmethod
    def _pair_features_impl(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        delta = left - right
        delta_background = F.avg_pool2d(delta, 5, stride=1, padding=2, count_include_pad=False)
        shared_appearance = 0.5 * (left + right)
        raw_change = delta.abs()
        local_change = (delta - delta_background).abs()
        return torch.cat((shared_appearance, raw_change, local_change), dim=1)

    @staticmethod
    def pair_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.ndim != 4 or left.shape != right.shape or left.shape[1] != 3:
            raise ValueError("ROI pair inputs must have matching [batch, 3, height, width] shapes")
        return RoiPairMatcher._pair_features_impl(left, right)

    def _forward_impl(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.context(
            self.vertical_downsample(
                self.detail(
                    self.downsample(
                        self.stem(self._pair_features_impl(left, right))
                    )
                )
            )
        )
        text_mask_logits = self.text_mask(features)
        gated = features * (0.25 + 0.75 * torch.sigmoid(text_mask_logits))
        spatial_mean = gated.mean(dim=(2, 3))
        spatial_peak = gated.amax(dim=(2, 3))
        width_evidence = gated.mean(dim=2)
        width_tail = width_evidence.amax(dim=-1)
        pair_logit = self.classifier(torch.cat((spatial_mean, spatial_peak, width_tail), dim=1)).squeeze(1)
        return pair_logit, text_mask_logits

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if left.ndim != 4 or left.shape != right.shape or left.shape[1] != 3:
            raise ValueError("ROI pair inputs must have matching [batch, 3, height, width] shapes")
        return self._forward_impl(left, right)

    def score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(left, right)[0])


def fuse_pair_matcher_for_inference(model: RoiPairMatcher) -> RoiPairMatcher:
    """Return an eval-only copy with every Conv-BatchNorm pair folded."""
    device = next(model.parameters()).device
    optimized = copy.deepcopy(model).to("cpu").eval()
    for block in (optimized.stem, optimized.downsample):
        block[0] = fuse_conv_bn_eval(block[0], block[1])
        block[1] = nn.Identity()
    for block in (optimized.detail, optimized.vertical_downsample, optimized.context):
        block[0] = fuse_conv_bn_eval(block[0], block[1])
        block[1] = nn.Identity()
        block[3] = fuse_conv_bn_eval(block[3], block[4])
        block[4] = nn.Identity()
    return optimized.to(device)


class RoiPairInference(nn.Module):
    """Trace-friendly inference wrapper without Python shape checks."""

    def __init__(self, model: RoiPairMatcher) -> None:
        super().__init__()
        self.model = model

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.model._forward_impl(left, right)[0]


@torch.inference_mode()
def trace_pair_matcher_for_inference(
    model: RoiPairMatcher,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.jit.ScriptModule:
    """Fold inference BatchNorm and trace the warm fixed-shape MPS graph."""
    if left.ndim != 4 or left.shape != right.shape or left.shape[1] != 3:
        raise ValueError("ROI pair inputs must have matching [batch, 3, height, width] shapes")
    optimized = RoiPairInference(fuse_pair_matcher_for_inference(model)).eval()
    return torch.jit.trace(
        optimized,
        (left, right),
        strict=False,
        check_trace=False,
    )
