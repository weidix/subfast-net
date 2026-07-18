from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    feature_count: int
    token_count: int
    width: int = 64
    byte_embedding_dim: int = 8
    temporal_layers: int = 7
    recurrent_layers: int = 1
    dropout: float = 0.10
    use_byte_branch: bool = False
    initial_boundary_distance_seconds: float = 1.25

    def to_dict(self) -> dict:
        return asdict(self)


class SegmentModelOutput(NamedTuple):
    """A complete temporal segment proposal at every compressed-frame anchor."""

    score_logits: torch.Tensor
    start_offsets_seconds: torch.Tensor
    end_offsets_seconds: torch.Tensor
    boundary_event_logits: torch.Tensor


class BytePayloadEncoder(nn.Module):
    def __init__(self, embedding_dim: int, output_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, embedding_dim)
        self.convolution = nn.Sequential(
            nn.Conv1d(embedding_dim, output_dim, 7, stride=2, padding=3),
            nn.SiLU(inplace=True),
            nn.Conv1d(output_dim, output_dim, 5, stride=2, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv1d(output_dim, output_dim, 3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Linear(output_dim * 2, output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        encoded = self.convolution(embedded)
        pooled = torch.cat([encoded.mean(dim=-1), encoded.amax(dim=-1)], dim=-1)
        return self.output(pooled)


class TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.scale * self.network(inputs)


class H264SubtitleSegmentModel(nn.Module):
    """Anchor-based detector that predicts complete subtitle time segments."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.feature_count <= 0 or config.token_count <= 0:
            raise ValueError("feature_count and token_count must be positive")
        if config.width < 2:
            raise ValueError("width must be at least 2")
        if config.byte_embedding_dim <= 0:
            raise ValueError("byte_embedding_dim must be positive")
        if config.temporal_layers <= 0:
            raise ValueError("temporal_layers must be positive")
        if config.recurrent_layers <= 0:
            raise ValueError("recurrent_layers must be positive")
        if config.width % 2 != 0:
            raise ValueError("width must be even for the bidirectional recurrent layer")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not math.isfinite(config.initial_boundary_distance_seconds) or (
            config.initial_boundary_distance_seconds <= 0.0
        ):
            raise ValueError("initial_boundary_distance_seconds must be positive")
        self.config = config
        numeric_width = config.width // 2 if config.use_byte_branch else config.width
        self.numeric_encoder = nn.Sequential(
            nn.Linear(config.feature_count, numeric_width),
            nn.LayerNorm(numeric_width),
            nn.SiLU(inplace=True),
            nn.Linear(numeric_width, numeric_width),
            nn.SiLU(inplace=True),
        )
        if config.use_byte_branch:
            self.byte_encoder: BytePayloadEncoder | None = BytePayloadEncoder(
                config.byte_embedding_dim, numeric_width
            )
            self.fusion: nn.Module = nn.Sequential(
                nn.Linear(numeric_width * 2, config.width),
                nn.LayerNorm(config.width),
                nn.SiLU(inplace=True),
            )
        else:
            self.byte_encoder = None
            self.fusion = nn.Identity()
        self.temporal = nn.Sequential(
            *[
                TemporalResidualBlock(config.width, 2**layer, config.dropout)
                for layer in range(config.temporal_layers)
            ]
        )
        self.recurrent = nn.GRU(
            input_size=config.width,
            hidden_size=config.width // 2,
            num_layers=config.recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(config.width)
        self.score_head = nn.Conv1d(config.width, 1, kernel_size=1)
        self.boundary_head = nn.Sequential(
            nn.Conv1d(config.width, config.width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(config.width, 2, kernel_size=1),
        )
        self.boundary_event_head = nn.Conv1d(config.width, 2, kernel_size=1)
        nn.init.constant_(self.score_head.bias, -2.0)
        boundary_output = self.boundary_head[-1]
        if not isinstance(boundary_output, nn.Conv1d):
            raise TypeError("boundary head must end with Conv1d")
        initial_boundary_bias = math.log(
            math.expm1(config.initial_boundary_distance_seconds)
        )
        nn.init.constant_(boundary_output.bias, initial_boundary_bias)
        nn.init.constant_(self.boundary_event_head.bias, -2.0)

    def forward(
        self, features: torch.Tensor, tokens: torch.Tensor
    ) -> SegmentModelOutput:
        if features.ndim != 3 or tokens.ndim != 3:
            raise ValueError("features and tokens must have shape [batch, time, channels]")
        batch, frames, feature_count = features.shape
        token_batch, token_frames, token_count = tokens.shape
        if (token_batch, token_frames) != (batch, frames):
            raise ValueError("features and tokens must have matching batch and time dimensions")
        if feature_count != self.config.feature_count:
            raise ValueError(
                f"expected {self.config.feature_count} numeric features, got {feature_count}"
            )
        if token_count != self.config.token_count:
            raise ValueError(f"expected {self.config.token_count} byte tokens, got {token_count}")
        numeric = self.numeric_encoder(features)
        if self.byte_encoder is None:
            fused = numeric
        else:
            byte_features = self.byte_encoder(tokens.reshape(batch * frames, -1)).reshape(
                batch, frames, -1
            )
            fused = self.fusion(torch.cat([numeric, byte_features], dim=-1))
        encoded = self.temporal(fused.transpose(1, 2)).transpose(1, 2)
        recurrent, _ = self.recurrent(encoded)
        encoded = self.temporal_norm(encoded + recurrent).transpose(1, 2)
        score_logits = self.score_head(encoded).squeeze(1)
        raw_boundaries = self.boundary_head(encoded).transpose(1, 2)
        return SegmentModelOutput(
            score_logits=score_logits,
            start_offsets_seconds=-F.softplus(raw_boundaries[..., 0]),
            end_offsets_seconds=F.softplus(raw_boundaries[..., 1]),
            boundary_event_logits=self.boundary_event_head(encoded).transpose(1, 2),
        )
