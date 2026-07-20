from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StreamingModelConfig:
    feature_count: int
    token_count: int
    width: int = 64
    byte_embedding_dim: int = 8
    temporal_layers: int = 6
    recurrent_layers: int = 1
    dropout: float = 0.10
    use_byte_branch: bool = False
    use_segment_head: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class StreamingSegmentModelOutput(NamedTuple):
    """Per-sample subtitle presence and boundary-event logits."""

    presence_logits: torch.Tensor
    boundary_event_logits: torch.Tensor
    segment_anchor_logits: torch.Tensor
    segment_start_offsets_seconds: torch.Tensor
    segment_end_offsets_seconds: torch.Tensor


class StreamingTemporalBlockState(NamedTuple):
    """Fixed-size histories for the two causal convolutions in one block."""

    first_history: torch.Tensor
    second_history: torch.Tensor


class StreamingModelState(NamedTuple):
    """Immutable state required to continue one batch of streams."""

    temporal_blocks: tuple[StreamingTemporalBlockState, ...]
    recurrent_hidden: torch.Tensor


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


class CausalConvolution(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.history_frames = 2 * dilation
        self.convolution = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=0,
            dilation=dilation,
        )
        self.register_buffer("_step_weight", None, persistent=False)
        self._step_weight_source_version = -1

    def forward_stream(
        self, inputs: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((history, inputs), dim=-1)
        outputs = self.convolution(combined)
        next_history = combined[..., -self.history_frames :].clone()
        return outputs, next_history

    def forward_step(
        self, inputs: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate one frame without the slow dilated-Conv1d CPU kernel."""

        dilation = self.history_frames // 2
        taps = torch.cat(
            (history[..., 0], history[..., dilation], inputs[..., 0]), dim=-1
        )
        outputs = F.linear(
            taps,
            self._linear_step_weight(),
            self.convolution.bias,
        ).unsqueeze(-1)
        next_history = torch.cat((history[..., 1:], inputs), dim=-1)
        return outputs, next_history

    def prepare_step_inference(self) -> None:
        """Pack the three causal taps for the single-frame inference kernel."""

        self._linear_step_weight()

    def _linear_step_weight(self) -> torch.Tensor:
        source = self.convolution.weight
        cached = self._step_weight
        source_version = source._version
        if (
            cached is None
            or self._step_weight_source_version != source_version
            or cached.device != source.device
            or cached.dtype != source.dtype
        ):
            cached = (
                source.detach()
                .permute(0, 2, 1)
                .reshape(source.shape[0], -1)
                .contiguous()
            )
            self._step_weight = cached
            self._step_weight_source_version = source_version
        return cached


class CausalTemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.first_convolution = CausalConvolution(width, dilation)
        self.second_convolution = CausalConvolution(width, dilation)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(0.1))

    @property
    def history_frames(self) -> int:
        return self.first_convolution.history_frames

    def forward_stream(
        self, inputs: torch.Tensor, state: StreamingTemporalBlockState
    ) -> tuple[torch.Tensor, StreamingTemporalBlockState]:
        encoded, first_history = self.first_convolution.forward_stream(
            inputs, state.first_history
        )
        encoded = self.dropout(F.silu(encoded))
        encoded, second_history = self.second_convolution.forward_stream(
            encoded, state.second_history
        )
        encoded = self.dropout(F.silu(encoded))
        return (
            inputs + self.scale * encoded,
            StreamingTemporalBlockState(first_history, second_history),
        )

    def forward_step(
        self, inputs: torch.Tensor, state: StreamingTemporalBlockState
    ) -> tuple[torch.Tensor, StreamingTemporalBlockState]:
        encoded, first_history = self.first_convolution.forward_step(
            inputs, state.first_history
        )
        encoded = F.silu(encoded)
        encoded, second_history = self.second_convolution.forward_step(
            encoded, state.second_history
        )
        encoded = F.silu(encoded)
        return (
            inputs + self.scale * encoded,
            StreamingTemporalBlockState(first_history, second_history),
        )

    def prepare_step_inference(self) -> None:
        self.first_convolution.prepare_step_inference()
        self.second_convolution.prepare_step_inference()


class StreamingH264SubtitleModel(nn.Module):
    """Causal subtitle segment detector with explicit bounded stream state."""

    def __init__(self, config: StreamingModelConfig) -> None:
        super().__init__()
        if config.feature_count <= 0 or config.token_count <= 0:
            raise ValueError("feature_count and token_count must be positive")
        if config.width <= 0:
            raise ValueError("width must be positive")
        if config.byte_embedding_dim <= 0:
            raise ValueError("byte_embedding_dim must be positive")
        if config.temporal_layers <= 0:
            raise ValueError("temporal_layers must be positive")
        if config.recurrent_layers <= 0:
            raise ValueError("recurrent_layers must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.config = config
        numeric_width = config.width // 2 if config.use_byte_branch else config.width
        if numeric_width <= 0:
            raise ValueError("width must be at least 2 when the byte branch is enabled")
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
        self.temporal = nn.ModuleList(
            CausalTemporalResidualBlock(config.width, 2**layer, config.dropout)
            for layer in range(config.temporal_layers)
        )
        self.recurrent = nn.GRU(
            input_size=config.width,
            hidden_size=config.width,
            num_layers=config.recurrent_layers,
            batch_first=True,
            bidirectional=False,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(config.width)
        self.presence_head = nn.Linear(config.width, 1)
        self.boundary_event_head = nn.Linear(config.width, 2)
        if config.use_segment_head:
            self.segment_anchor_head: nn.Linear | None = nn.Linear(config.width, 1)
            self.segment_boundary_head: nn.Linear | None = nn.Linear(config.width, 2)
        else:
            self.segment_anchor_head = None
            self.segment_boundary_head = None
        nn.init.constant_(self.presence_head.bias, -2.0)
        nn.init.constant_(self.boundary_event_head.bias, -2.0)
        if self.segment_anchor_head is not None:
            nn.init.constant_(self.segment_anchor_head.bias, -4.5)
        if self.segment_boundary_head is not None:
            # Start offsets are negative and usually span a few seconds; end
            # offsets are close to zero at the first observed end frame.
            nn.init.constant_(self.segment_boundary_head.bias[0], 2.0)
            nn.init.constant_(self.segment_boundary_head.bias[1], -2.0)

    def forward(
        self, features: torch.Tensor, tokens: torch.Tensor
    ) -> StreamingSegmentModelOutput:
        output, _ = self.forward_stream(features, tokens)
        return output

    def forward_stream(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
        state: StreamingModelState | None = None,
    ) -> tuple[StreamingSegmentModelOutput, StreamingModelState]:
        """Process one non-empty chunk and return its outputs plus continuation state."""

        batch, frames = self._validate_inputs(features, tokens)
        fused = self._encode_inputs(features, tokens, batch=batch, frames=frames)

        if state is None:
            state = self._initial_state(fused)
        else:
            self._validate_state(state, fused)

        encoded = fused.transpose(1, 2)
        block_states: list[StreamingTemporalBlockState] = []
        for block, block_state in zip(
            self.temporal, state.temporal_blocks, strict=True
        ):
            encoded, next_block_state = block.forward_stream(encoded, block_state)
            block_states.append(next_block_state)
        encoded = encoded.transpose(1, 2)
        recurrent, recurrent_hidden = self.recurrent(encoded, state.recurrent_hidden)
        encoded = self.temporal_norm(encoded + recurrent)
        output = self._output_from_encoded(encoded, batch=batch, frames=frames)
        next_state = StreamingModelState(
            temporal_blocks=tuple(block_states),
            recurrent_hidden=recurrent_hidden,
        )
        return output, next_state

    def prepare_step_inference(self) -> None:
        """Prepare immutable weights used by the low-latency one-frame path."""

        if self.training:
            raise RuntimeError("step inference preparation requires eval mode")
        for block in self.temporal:
            block.prepare_step_inference()

    def forward_step(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
        state: StreamingModelState | None = None,
    ) -> tuple[StreamingSegmentModelOutput, StreamingModelState]:
        """Process exactly one frame with kernels specialized for streaming latency."""

        if self.training:
            raise RuntimeError("step inference requires eval mode")
        batch, frames = self._validate_inputs(features, tokens)
        if frames != 1:
            raise ValueError("step inference requires exactly one frame")

        fused = self._encode_inputs(features, tokens, batch=batch, frames=frames)

        if state is None:
            state = self._initial_state(fused)
        else:
            self._validate_state(state, fused)

        encoded = fused.transpose(1, 2)
        block_states: list[StreamingTemporalBlockState] = []
        for block, block_state in zip(
            self.temporal, state.temporal_blocks, strict=True
        ):
            encoded, next_block_state = block.forward_step(encoded, block_state)
            block_states.append(next_block_state)

        encoded = encoded.transpose(1, 2)
        recurrent_input = encoded[:, 0]
        recurrent_hidden: list[torch.Tensor] = []
        for layer in range(self.config.recurrent_layers):
            previous_hidden = state.recurrent_hidden[layer]
            input_gates = F.linear(
                recurrent_input,
                getattr(self.recurrent, f"weight_ih_l{layer}"),
                getattr(self.recurrent, f"bias_ih_l{layer}"),
            )
            hidden_gates = F.linear(
                previous_hidden,
                getattr(self.recurrent, f"weight_hh_l{layer}"),
                getattr(self.recurrent, f"bias_hh_l{layer}"),
            )
            input_reset, input_update, input_new = input_gates.chunk(3, dim=-1)
            hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, dim=-1)
            reset = torch.sigmoid(input_reset + hidden_reset)
            update = torch.sigmoid(input_update + hidden_update)
            new = torch.tanh(input_new + reset * hidden_new)
            recurrent_input = (1.0 - update) * new + update * previous_hidden
            recurrent_hidden.append(recurrent_input)

        next_recurrent_hidden = torch.stack(recurrent_hidden)
        encoded = self.temporal_norm(encoded + recurrent_input.unsqueeze(1))
        output = self._output_from_encoded(encoded, batch=batch, frames=frames)
        return output, StreamingModelState(
            temporal_blocks=tuple(block_states),
            recurrent_hidden=next_recurrent_hidden,
        )

    def _encode_inputs(
        self,
        features: torch.Tensor,
        tokens: torch.Tensor,
        *,
        batch: int,
        frames: int,
    ) -> torch.Tensor:
        numeric = self.numeric_encoder(features)
        if self.byte_encoder is None:
            return numeric
        byte_features = self.byte_encoder(
            tokens.reshape(batch * frames, -1)
        ).reshape(batch, frames, -1)
        return self.fusion(torch.cat((numeric, byte_features), dim=-1))

    def _output_from_encoded(
        self, encoded: torch.Tensor, *, batch: int, frames: int
    ) -> StreamingSegmentModelOutput:
        segment_boundaries = (
            self.segment_boundary_head(encoded)
            if self.segment_boundary_head is not None
            else None
        )
        return StreamingSegmentModelOutput(
            presence_logits=self.presence_head(encoded).squeeze(-1),
            boundary_event_logits=self.boundary_event_head(encoded),
            segment_anchor_logits=(
                self.segment_anchor_head(encoded).squeeze(-1)
                if self.segment_anchor_head is not None
                else encoded.new_zeros((batch, frames))
            ),
            segment_start_offsets_seconds=(
                -F.softplus(segment_boundaries[..., 0])
                if segment_boundaries is not None
                else encoded.new_zeros((batch, frames))
            ),
            segment_end_offsets_seconds=(
                F.softplus(segment_boundaries[..., 1])
                if segment_boundaries is not None
                else encoded.new_zeros((batch, frames))
            ),
        )

    def _validate_inputs(
        self, features: torch.Tensor, tokens: torch.Tensor
    ) -> tuple[int, int]:
        if features.ndim != 3 or tokens.ndim != 3:
            raise ValueError(
                "features and tokens must have shape [batch, time, channels]"
            )
        batch, frames, feature_count = features.shape
        token_batch, token_frames, token_count = tokens.shape
        if batch <= 0 or frames <= 0:
            raise ValueError(
                "features and tokens must contain a non-empty batch and time axis"
            )
        if (token_batch, token_frames) != (batch, frames):
            raise ValueError(
                "features and tokens must have matching batch and time dimensions"
            )
        if feature_count != self.config.feature_count:
            raise ValueError(
                f"expected {self.config.feature_count} numeric features, got {feature_count}"
            )
        if token_count != self.config.token_count:
            raise ValueError(
                f"expected {self.config.token_count} byte tokens, got {token_count}"
            )
        if self.byte_encoder is not None and tokens.device != features.device:
            raise ValueError(
                "features and tokens must use the same device when byte input is enabled"
            )
        return batch, frames

    def _initial_state(self, encoded: torch.Tensor) -> StreamingModelState:
        batch = encoded.shape[0]
        temporal_blocks = tuple(
            StreamingTemporalBlockState(
                encoded.new_zeros((batch, self.config.width, block.history_frames)),
                encoded.new_zeros((batch, self.config.width, block.history_frames)),
            )
            for block in self.temporal
        )
        recurrent_hidden = encoded.new_zeros(
            (self.config.recurrent_layers, batch, self.config.width)
        )
        return StreamingModelState(temporal_blocks, recurrent_hidden)

    def _validate_state(
        self, state: StreamingModelState, encoded: torch.Tensor
    ) -> None:
        if len(state.temporal_blocks) != len(self.temporal):
            raise ValueError(
                f"stream state has {len(state.temporal_blocks)} temporal blocks, "
                f"expected {len(self.temporal)}"
            )
        batch = encoded.shape[0]
        for index, (block, block_state) in enumerate(
            zip(self.temporal, state.temporal_blocks, strict=True)
        ):
            expected_shape = (batch, self.config.width, block.history_frames)
            for name, history in (
                ("first", block_state.first_history),
                ("second", block_state.second_history),
            ):
                if history.shape != expected_shape:
                    raise ValueError(
                        f"stream state temporal block {index} {name} history must have "
                        f"shape {expected_shape}"
                    )
                self._validate_state_tensor(history, encoded, f"temporal block {index}")
        expected_recurrent_shape = (
            self.config.recurrent_layers,
            batch,
            self.config.width,
        )
        if state.recurrent_hidden.shape != expected_recurrent_shape:
            raise ValueError(
                f"stream recurrent state must have shape {expected_recurrent_shape}"
            )
        self._validate_state_tensor(
            state.recurrent_hidden, encoded, "recurrent hidden state"
        )

    @staticmethod
    def _validate_state_tensor(
        value: torch.Tensor, reference: torch.Tensor, name: str
    ) -> None:
        if value.device != reference.device:
            raise ValueError(f"stream {name} must use device {reference.device}")
        if value.dtype != reference.dtype:
            raise ValueError(f"stream {name} must use dtype {reference.dtype}")
