from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FramePresenceModel(nn.Module):
    """Fast subtitle presence classifier with a conservative enclosing contour."""

    architecture_version = 23
    heatmap_stride = (4, 2)
    compact_activation_threshold = 0.23
    expanded_activation_threshold = 0.19

    def __init__(
        self,
        *,
        width: int = 16,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer greater than one")
        self.width = width
        self.kernel_size = kernel_size
        self.focus_backbone = nn.Sequential(
            nn.Conv2d(3, width, kernel_size, stride=2, padding=kernel_size // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size, stride=2, padding=kernel_size // 2),
            nn.SiLU(inplace=True),
        )
        self.focus_region_head = nn.Conv2d(
            width * 2,
            1,
            kernel_size=3,
            padding=2,
            dilation=2,
        )
        self.evidence_pool = nn.AvgPool2d(
            kernel_size=5,
            stride=1,
        )
        evidence_ones = torch.ones((1, 1, 8, 64), dtype=torch.float32)
        self.register_buffer(
            "evidence_valid_fraction",
            self.evidence_pool(F.pad(evidence_ones, (2, 2, 2, 2))),
            persistent=False,
        )
        self.presence_log_scale = nn.Parameter(torch.zeros(()))
        self.presence_bias = nn.Parameter(torch.zeros(()))
        self.presence_output_scale = nn.Parameter(torch.ones(()))
        self.presence_output_bias = nn.Parameter(torch.zeros(()))
        self.full_context = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=5, stride=8, padding=2),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.context_projection = nn.Linear(4, 1, bias=False)
        nn.init.zeros_(self.context_projection.weight)
        self.region_envelope = nn.MaxPool2d(kernel_size=3, stride=1)
        self.region_support = nn.MaxPool2d(kernel_size=3, stride=1)
        contour_y = torch.arange(72).view(72, 1)
        contour_x = torch.arange(64).view(1, 64)
        self.register_buffer(
            "contour_sampling_mask",
            ((contour_x + contour_y) % 2 == 0).view(1, 1, 72, 64),
            persistent=False,
        )

    @staticmethod
    def _embed_focus_region(
        focus_region_logits: torch.Tensor,
        focus_mode: torch.Tensor,
        *,
        expanded: bool,
    ) -> torch.Tensor:
        if expanded:
            normal_size = (12, 48)
            normal_padding = (8, 8, 60, 0)
        else:
            normal_size = (11, 44)
            normal_padding = (10, 10, 61, 0)
        normal = F.pad(
            F.interpolate(
                focus_region_logits,
                size=normal_size,
                mode="bilinear",
                align_corners=False,
            ),
            normal_padding,
            value=-12.0,
        )
        wide = F.pad(
            F.interpolate(
                focus_region_logits,
                size=(22, 50),
                mode="bilinear",
                align_corners=False,
            ),
            (7, 7, 50, 0),
            value=-12.0,
        )
        legacy = F.pad(
            F.interpolate(
                focus_region_logits,
                size=(9, 54),
                mode="bilinear",
                align_corners=False,
            ),
            (5, 5, 63, 0),
            value=-12.0,
        )
        mode = focus_mode.view(-1, 1, 1, 1)
        return torch.where(mode > 1.5, legacy, torch.where(mode > 0.5, wide, normal))

    @staticmethod
    def _threshold_logit(value: float) -> float:
        return math.log(value / (1.0 - value))

    @staticmethod
    def _safe_max_pool(values: torch.Tensor, pool: nn.MaxPool2d) -> torch.Tensor:
        # Core ML's reduction kernel clamps an all-negative reduction to zero.
        # The bounded positive shift preserves the mathematical max exactly in
        # the score range that can affect either region threshold.
        return pool(values.clamp_min(-12.0) + 16.0) - 16.0

    def _enclosing_contour(
        self,
        focus_region_logits: torch.Tensor,
        focus_mode: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compact = self._safe_max_pool(
            F.pad(
                self._embed_focus_region(
                    focus_region_logits,
                    focus_mode,
                    expanded=False,
                ),
                (1, 1, 1, 1),
                value=-12.0,
            ),
            self.region_envelope,
        )
        expanded = self._safe_max_pool(
            F.pad(
                self._embed_focus_region(
                    focus_region_logits,
                    focus_mode,
                    expanded=True,
                ),
                (1, 1, 1, 1),
                value=-12.0,
            ),
            self.region_envelope,
        )
        compact_centered = compact - self._threshold_logit(
            self.compact_activation_threshold
        )
        expanded_centered = expanded - self._threshold_logit(
            self.expanded_activation_threshold
        )

        # Expanded evidence may add at most one cell around compact evidence.
        supported_expansion = torch.minimum(
            expanded_centered,
            self._safe_max_pool(
                F.pad(compact_centered, (1, 1, 1, 1), value=-12.0),
                self.region_support,
            ),
        )
        filled_envelope = torch.maximum(compact_centered, supported_expansion)

        # A signed morphological gradient emits a closed, one-cell contour. Its
        # active area cannot grow by filling the entire subtitle band.
        filled_probability = torch.sigmoid(filled_envelope)
        exterior_neighbor = F.max_pool2d(
            F.pad(1.0 - filled_probability, (1, 1, 1, 1), value=1.0),
            kernel_size=3,
            stride=1,
        ) - 0.5
        contour = 8.0 * torch.minimum(
            filled_probability - 0.5,
            exterior_neighbor,
        )
        contour = torch.where(
            self.contour_sampling_mask,
            contour,
            contour.new_tensor(-4.0),
        )
        return contour, compact

    def forward_with_components(
        self,
        images: torch.Tensor,
        focus: torch.Tensor,
        focus_mode: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        focus_features = self.focus_backbone(focus)
        focus_evidence_logits = self.focus_region_head(focus_features)
        coherent_map = self.evidence_pool(
            F.pad(focus_evidence_logits, (2, 2, 2, 2), value=0.0)
        ) / self.evidence_valid_fraction
        coherent_evidence = (coherent_map.clamp_min(-15.0) + 16.0).amax(
            dim=(1, 2, 3)
        ) - 16.0
        scale = F.softplus(self.presence_log_scale) + 0.5
        context = self.context_projection(self.full_context(images)).squeeze(1)
        uncalibrated_presence = (
            scale * coherent_evidence + self.presence_bias + 0.1 * context
        )
        presence_logits = (
            self.presence_output_scale * uncalibrated_presence
            + self.presence_output_bias
        )
        contour_logits, compact_region_logits = self._enclosing_contour(
            focus_evidence_logits,
            focus_mode,
        )
        contour_logits = torch.minimum(
            contour_logits,
            presence_logits[:, None, None, None],
        )
        compact_region_logits = torch.minimum(
            compact_region_logits,
            presence_logits[:, None, None, None],
        )
        return presence_logits, contour_logits, compact_region_logits

    def forward_with_heatmap(
        self,
        images: torch.Tensor,
        focus: torch.Tensor,
        focus_mode: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        presence_logits, region_logits, _ = self.forward_with_components(
            images,
            focus,
            focus_mode,
        )
        return presence_logits, region_logits

    def forward(
        self,
        images: torch.Tensor,
        focus: torch.Tensor,
        focus_mode: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_with_heatmap(images, focus, focus_mode)
