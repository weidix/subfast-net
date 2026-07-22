from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .model import FramePresenceModel, ResidualDepthwiseBlock


_MPS_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void static_affine_silu(
    device float* output [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* scale [[buffer(2)]],
    const device float* bias [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& spatial [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  const uint channel = (index / spatial) % channels;
  const float value = input[index] * scale[channel] + bias[channel];
  output[index] = value / (1.0f + exp(-value));
}

kernel void static_depthwise_silu(
    device float* output [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* depthwise_weight [[buffer(2)]],
    const device float* first_scale [[buffer(3)]],
    const device float* first_bias [[buffer(4)]],
    const device float* second_bias [[buffer(5)]],
    constant uint& channels [[buffer(6)]],
    constant uint& height [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& kernel_size [[buffer(9)]],
    uint index [[thread_position_in_grid]]) {
  const uint spatial = height * width;
  const uint channel = (index / spatial) % channels;
  const uint batch = index / (channels * spatial);
  const uint location = index % spatial;
  const uint row = location / width;
  const uint column = location % width;
  const int padding = int(kernel_size / 2);
  float total = 0.0f;

  for (uint kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
    const int input_row = int(row) + int(kernel_row) - padding;
    if (input_row < 0 || input_row >= int(height)) {
      continue;
    }
    for (uint kernel_column = 0; kernel_column < kernel_size; ++kernel_column) {
      const int input_column = int(column) + int(kernel_column) - padding;
      if (input_column < 0 || input_column >= int(width)) {
        continue;
      }
      const uint input_index =
          (batch * channels + channel) * spatial + uint(input_row) * width + uint(input_column);
      const float affine = input[input_index] * first_scale[channel] + first_bias[channel];
      const float activated = affine / (1.0f + exp(-affine));
      const uint weight_index =
          (channel * kernel_size + kernel_row) * kernel_size + kernel_column;
      total += activated * depthwise_weight[weight_index];
    }
  }
  const float affine = total + second_bias[channel];
  output[index] = affine / (1.0f + exp(-affine));
}

kernel void bilinear_add(
    device float* output [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* skip [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& input_height [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& output_height [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  const uint output_spatial = output_height * output_width;
  const uint channel = (index / output_spatial) % channels;
  const uint batch = index / (channels * output_spatial);
  const uint location = index % output_spatial;
  const uint row = location / output_width;
  const uint column = location % output_width;
  const float input_row =
      ((float(row) + 0.5f) * float(input_height) / float(output_height)) - 0.5f;
  const float input_column =
      ((float(column) + 0.5f) * float(input_width) / float(output_width)) - 0.5f;
  int row0 = int(floor(input_row));
  int column0 = int(floor(input_column));
  int row1 = row0 + 1;
  int column1 = column0 + 1;
  const float row_weight = input_row - float(row0);
  const float column_weight = input_column - float(column0);
  row0 = clamp(row0, 0, int(input_height) - 1);
  row1 = clamp(row1, 0, int(input_height) - 1);
  column0 = clamp(column0, 0, int(input_width) - 1);
  column1 = clamp(column1, 0, int(input_width) - 1);
  const uint input_base = (batch * channels + channel) * input_height * input_width;
  const float top_left = input[input_base + uint(row0) * input_width + uint(column0)];
  const float top_right = input[input_base + uint(row0) * input_width + uint(column1)];
  const float bottom_left = input[input_base + uint(row1) * input_width + uint(column0)];
  const float bottom_right = input[input_base + uint(row1) * input_width + uint(column1)];
  const float top = top_left * (1.0f - column_weight) + top_right * column_weight;
  const float bottom = bottom_left * (1.0f - column_weight) + bottom_right * column_weight;
  output[index] = skip[index] + top * (1.0f - row_weight) + bottom * row_weight;
}
"""


_MPS_LIBRARY: Any | None = None


def _mps_library() -> Any:
    global _MPS_LIBRARY
    if _MPS_LIBRARY is None:
        _MPS_LIBRARY = torch.mps.compile_shader(_MPS_SOURCE)
    return _MPS_LIBRARY


class StaticAffineSiLU(nn.Module):
    """Inference-only per-channel affine normalization followed by SiLU."""

    def __init__(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("bias", bias.detach().clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.device.type == "mps" and features.dtype == torch.float32 and features.is_contiguous():
            output = torch.empty_like(features)
            _mps_library().static_affine_silu(
                output,
                features,
                self.scale,
                self.bias,
                features.shape[1],
                features.shape[-2] * features.shape[-1],
            )
            return output
        scale = self.scale.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        return F.silu(features * scale + bias)


class StaticDepthwiseSiLU(nn.Module):
    """Fused static affine, depthwise convolution, static affine, and SiLU."""

    def __init__(
        self,
        depthwise: nn.Conv2d,
        first_scale: torch.Tensor,
        first_bias: torch.Tensor,
        second_scale: torch.Tensor,
        second_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        if depthwise.groups != depthwise.in_channels or depthwise.in_channels != depthwise.out_channels:
            raise ValueError("StaticDepthwiseSiLU requires a depthwise convolution")
        if depthwise.bias is not None:
            raise ValueError("StaticDepthwiseSiLU requires a bias-free depthwise convolution")
        if depthwise.stride != (1, 1) or depthwise.dilation != (1, 1):
            raise ValueError("StaticDepthwiseSiLU requires stride-one, non-dilated convolution")
        if depthwise.padding != (depthwise.kernel_size[0] // 2, depthwise.kernel_size[1] // 2):
            raise ValueError("StaticDepthwiseSiLU requires same-padding convolution")
        if depthwise.kernel_size[0] != depthwise.kernel_size[1]:
            raise ValueError("StaticDepthwiseSiLU requires square kernels")
        self.kernel_size = depthwise.kernel_size[0]
        self.register_buffer("depthwise_weight", depthwise.weight.detach().clone() * second_scale.reshape(-1, 1, 1, 1))
        self.register_buffer("first_scale", first_scale.detach().clone())
        self.register_buffer("first_bias", first_bias.detach().clone())
        self.register_buffer("second_bias", second_bias.detach().clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.device.type == "mps" and features.dtype == torch.float32 and features.is_contiguous():
            output = torch.empty_like(features)
            _mps_library().static_depthwise_silu(
                output,
                features,
                self.depthwise_weight,
                self.first_scale,
                self.first_bias,
                self.second_bias,
                features.shape[1],
                features.shape[-2],
                features.shape[-1],
                self.kernel_size,
            )
            return output
        affine = features * self.first_scale.reshape(1, -1, 1, 1) + self.first_bias.reshape(1, -1, 1, 1)
        depthwise = F.conv2d(F.silu(affine), self.depthwise_weight, padding=self.kernel_size // 2, groups=features.shape[1])
        return F.silu(depthwise + self.second_bias.reshape(1, -1, 1, 1))


class GroupStatsBatchNorm(nn.Module):
    """Batch-one GroupNorm expressed as group statistics and MPS BatchNorm."""

    def __init__(self, norm: nn.GroupNorm, *, sampling_stride: int = 1) -> None:
        super().__init__()
        if sampling_stride < 1:
            raise ValueError("sampling_stride must be positive")
        self.groups = norm.num_groups
        self.eps = norm.eps
        self.sampling_stride = sampling_stride
        self.register_buffer("weight", norm.weight.detach().clone())
        self.register_buffer("bias", norm.bias.detach().clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] != 1:
            raise ValueError("optimized frame-presence inference accepts batch size one")
        sampled = features[..., :: self.sampling_stride, :: self.sampling_stride]
        channels = features.shape[1]
        grouped = sampled.reshape(1, self.groups, -1)
        mean = grouped.mean(dim=-1).repeat_interleave(channels // self.groups, dim=1).reshape(-1)
        variance = grouped.var(dim=-1, correction=0).repeat_interleave(channels // self.groups, dim=1).reshape(-1)
        return F.batch_norm(
            features,
            mean,
            variance,
            self.weight,
            self.bias,
            training=False,
            momentum=0.0,
            eps=self.eps,
        )


class BilinearAdd(nn.Module):
    """Bilinear resize with align_corners=False followed by a skip addition."""

    def forward(self, features: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if (
            features.device.type == "mps"
            and features.dtype == torch.float32
            and features.is_contiguous()
            and skip.is_contiguous()
        ):
            output = torch.empty_like(skip)
            _mps_library().bilinear_add(
                output,
                features,
                skip,
                features.shape[1],
                features.shape[-2],
                features.shape[-1],
                skip.shape[-2],
                skip.shape[-1],
            )
            return output
        return skip + F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)


@dataclass(frozen=True)
class StaticNorm:
    scale: torch.Tensor
    bias: torch.Tensor


def _static_norm(norm: nn.GroupNorm, mean: torch.Tensor, variance: torch.Tensor) -> StaticNorm:
    channels_per_group = norm.num_channels // norm.num_groups
    repeated_mean = mean.to(device=norm.weight.device, dtype=norm.weight.dtype).repeat_interleave(channels_per_group)
    repeated_variance = variance.to(device=norm.weight.device, dtype=norm.weight.dtype).repeat_interleave(
        channels_per_group
    )
    scale = norm.weight.detach() / torch.sqrt(repeated_variance + norm.eps)
    bias = norm.bias.detach() - repeated_mean * scale
    return StaticNorm(scale=scale, bias=bias)


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)


def _group_norm_stats(
    model: nn.Module,
    images: Iterable[torch.Tensor],
    names: tuple[str, ...],
    *,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    totals: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}
    hooks: list[Any] = []

    def collect(name: str, groups: int) -> Any:
        def hook(_: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            features = args[0].detach()
            grouped = features.reshape(features.shape[0], groups, -1)
            means = grouped.mean(dim=-1).sum(dim=0).cpu()
            variances = grouped.var(dim=-1, correction=0).sum(dim=0).cpu()
            previous = totals.get(name)
            count = features.shape[0]
            if previous is None:
                totals[name] = (means, variances, count)
            else:
                totals[name] = (previous[0] + means, previous[1] + variances, previous[2] + count)

        return hook

    for name in names:
        module = model.get_submodule(name)
        if not isinstance(module, nn.GroupNorm):
            raise TypeError(f"{name} must be GroupNorm before calibration")
        hooks.append(module.register_forward_pre_hook(collect(name, module.num_groups)))
    try:
        with torch.inference_mode():
            for images_batch in images:
                model(images_batch.to(device=device, dtype=torch.float32, non_blocking=True))
    finally:
        for hook in hooks:
            hook.remove()
    if set(totals) != set(names):
        missing = sorted(set(names) - set(totals))
        raise ValueError(f"calibration data did not execute norms: {missing}")
    return {
        name: (mean / count, variance / count)
        for name, (mean, variance, count) in totals.items()
    }


class OptimizedFramePresenceModel(FramePresenceModel):
    """Frame-presence inference graph with fused FP32 MPS operators."""

    def __init__(self, *, width: int, evidence_kernel_size: int) -> None:
        super().__init__(width=width, evidence_kernel_size=evidence_kernel_size)
        self.up_2_merge = BilinearAdd()
        self.up_1_merge = BilinearAdd()
        self.up_0_merge = BilinearAdd()

    def encode_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.stem(images)
        mid = self.down_1(local)
        high = self.down_2(mid)
        context = self.down_3(high)
        high = self.up_2_merge(self.up_2(context), high)
        mid = self.up_1_merge(self.up_1(high), mid)
        local = self.up_0_merge(self.up_0(mid), local)
        return local, context


_FIRST_STATIC_NORMS = (
    "stem.3.net.0",
    "down_1.0.net.0",
    "down_1.1.net.0",
    "down_2.0.net.0",
    "down_2.1.net.0",
    "down_3.0.net.0",
    "down_3.1.net.0",
    "down_3.2.net.0",
    "down_3.3.net.0",
    "down_3.4.net.0",
    "up_2.1.net.0",
    "up_1.1.net.0",
    "up_0.1.net.0",
)

_SECOND_STATIC_NORMS = (
    "down_1.1.net.3",
    "down_2.1.net.3",
    "down_3.1.net.3",
    "down_3.2.net.3",
    "down_3.3.net.3",
    "down_3.4.net.3",
    "up_2.1.net.3",
    "up_1.1.net.3",
    "up_0.1.net.3",
)


def build_optimized_frame_presence_model(
    model: FramePresenceModel,
    calibration_images: Iterable[torch.Tensor],
    *,
    device: torch.device | str,
) -> FramePresenceModel:
    """Create a calibrated FP32 batch-one inference model without changing input geometry."""

    runtime_device = torch.device(device)
    stem = model.stem[0]
    if not isinstance(stem, nn.Conv2d):
        raise TypeError("stem.0 must be Conv2d")
    kernel_size = model.evidence_pool.support.kernel_size
    evidence_kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
    optimized = OptimizedFramePresenceModel(
        width=stem.out_channels,
        evidence_kernel_size=evidence_kernel_size,
    )
    optimized.load_state_dict(model.state_dict())
    optimized.eval().to(runtime_device)
    first_stats = _group_norm_stats(optimized, calibration_images, _FIRST_STATIC_NORMS, device=runtime_device)
    static: dict[str, StaticNorm] = {}
    for name in _FIRST_STATIC_NORMS:
        norm = optimized.get_submodule(name)
        if not isinstance(norm, nn.GroupNorm):
            raise TypeError(f"{name} must be GroupNorm")
        static[name] = _static_norm(norm, *first_stats[name])
        _replace_module(optimized, name, StaticAffineSiLU(static[name].scale, static[name].bias))
        parent_name, _, child_name = name.rpartition(".")
        parent = optimized.get_submodule(parent_name)
        if not isinstance(parent, nn.Sequential) or child_name != "0":
            raise TypeError(f"{name} must be the first Sequential operation")
        parent[1] = nn.Identity()

    second_stats = _group_norm_stats(optimized, calibration_images, _SECOND_STATIC_NORMS, device=runtime_device)
    for name in _SECOND_STATIC_NORMS:
        norm = optimized.get_submodule(name)
        if not isinstance(norm, nn.GroupNorm):
            raise TypeError(f"{name} must be GroupNorm")
        static[name] = _static_norm(norm, *second_stats[name])

    for block_name, block in optimized.named_modules():
        if not isinstance(block, ResidualDepthwiseBlock):
            continue
        first_name = f"{block_name}.net.0"
        second_name = f"{block_name}.net.3"
        if first_name not in static:
            continue
        if second_name not in static:
            continue
        if not isinstance(block.net[2], nn.Conv2d):
            raise TypeError(f"{block_name}.net.2 must be Conv2d")
        block.net[0] = StaticDepthwiseSiLU(
            block.net[2],
            static[first_name].scale,
            static[first_name].bias,
            static[second_name].scale,
            static[second_name].bias,
        )
        block.net[1] = nn.Identity()
        block.net[2] = nn.Identity()
        block.net[3] = nn.Identity()
        block.net[4] = nn.Identity()

    stem_norm = optimized.stem[1]
    if not isinstance(stem_norm, nn.GroupNorm):
        raise TypeError("stem.1 must be GroupNorm")
    optimized.stem[1] = GroupStatsBatchNorm(stem_norm, sampling_stride=4)
    stem_block = optimized.stem[3]
    if not isinstance(stem_block, ResidualDepthwiseBlock):
        raise TypeError("stem.3 must be ResidualDepthwiseBlock")
    second_norm = stem_block.net[3]
    if not isinstance(second_norm, nn.GroupNorm):
        raise TypeError("stem.3.net.3 must be GroupNorm")
    stem_block.net[3] = GroupStatsBatchNorm(second_norm, sampling_stride=2)
    return optimized.eval()


__all__ = ["OptimizedFramePresenceModel", "build_optimized_frame_presence_model"]
