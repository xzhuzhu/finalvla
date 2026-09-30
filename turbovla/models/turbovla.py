from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .action_head import StateProjection, TurboVLAActionHead
from .components.myvla_action_head import MyVLAFlowMatchingActionHead
from .components.fusion import BiAttentionBlock
from .components.transformer import TransformerEncoderLayer
from .components.utils import _get_clones
from .configuration import (
    ActionHeadConfig,
    HistoryConfig,
    InteractionConfig,
    R3MEncoderConfig,
    TextEncoderConfig,
    TurboVLAConfig,
    VisionEncoderConfig,
)
from .history_encoder import HistoryEncoder, MambaBlock
from .r3m_encoder import R3MResNet18Encoder
from .text_encoder import TurboVLATextEncoder
from .vision_encoder import DINOv3VisionEncoder


def apply_rotary_position_encoding(
    tokens: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply parameter-free RoPE to a ``[B,T,D]`` token sequence."""
    if tokens.ndim != 3:
        raise ValueError(f"RoPE tokens must be [B,T,D], got {tuple(tokens.shape)}")
    if positions.ndim != 1 or positions.shape[0] != tokens.shape[1]:
        raise ValueError(
            f"RoPE positions must have shape ({tokens.shape[1]},), got {tuple(positions.shape)}"
        )
    if tokens.shape[-1] % 2:
        raise ValueError("RoPE token dimension must be even")
    if base <= 0:
        raise ValueError("RoPE base must be positive")

    rotary_dim = tokens.shape[-1]
    frequency_indices = torch.arange(
        0,
        rotary_dim,
        2,
        device=tokens.device,
        dtype=torch.float32,
    )
    inverse_frequencies = base ** (-frequency_indices / rotary_dim)
    angles = positions.to(device=tokens.device, dtype=torch.float32)[:, None] * inverse_frequencies[None]
    cos = angles.cos().to(dtype=tokens.dtype)[None]
    sin = angles.sin().to(dtype=tokens.dtype)[None]
    even = tokens[..., 0::2]
    odd = tokens[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class VisionProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.skip(tokens) + self.mlp(self.input_norm(tokens)))


class VisionLanguageInteraction(nn.Module):
    def __init__(self, config: InteractionConfig) -> None:
        super().__init__()
        text_layer = TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=max(1, config.nheads // 2),
            dim_feedforward=config.enhancer_inner_dim,
            dropout=config.text_dropout,
        )
        fusion_layer = BiAttentionBlock(
            v_dim=config.hidden_dim,
            l_dim=config.hidden_dim,
            embed_dim=config.enhancer_inner_dim,
            num_heads=max(1, config.nheads // 2),
            dropout=config.fusion_dropout,
            drop_path=config.fusion_droppath,
            residual_style=config.residual_style,
            attention_backend=config.attention_backend,
        )
        self.text_layers = _get_clones(text_layer, config.num_layers)
        self.fusion_layers = _get_clones(fusion_layer, config.num_layers)
        self.padding_strategy = config.padding_strategy

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        text_self_attention_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero_fill = self.padding_strategy == "zero_fill"
        if zero_fill:
            text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)

        for fusion_layer, text_layer in zip(self.fusion_layers, self.text_layers):
            visual_tokens, text_tokens = fusion_layer(
                v=visual_tokens,
                l=text_tokens,
                attention_mask_v=None,
                attention_mask_l=text_key_padding_mask,
            )
            source_mask = None if text_self_attention_masks is None else ~text_self_attention_masks
            text_tokens = text_layer(
                src=text_tokens.transpose(0, 1),
                src_mask=source_mask,
                src_key_padding_mask=None if zero_fill else text_key_padding_mask,
                pos=None,
            ).transpose(0, 1)
            if zero_fill:
                text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
        return visual_tokens, text_tokens


class TurboVLA(nn.Module):
    """Shared TurboVLA architecture for LIBERO and RoboTwin."""

    def __init__(
        self,
        config: TurboVLAConfig,
        action_head: str = "act",
        flow_num_heads: int = 4,
        flow_condition_layers: int = 4,
        flow_dit_layers: int = 16,
        flow_target_tokens: int = 16,
        flow_static_tokens: int = 8,
        flow_state_dim: int = 0,
        flow_bijection_blocks: int = 6,
        flow_state_encoding: str = "native_mlp",
    ) -> None:
        super().__init__()
        self.config = config
        self.action_head_type = str(action_head)
        if self.action_head_type not in {"act", "flow_matching"}:
            raise ValueError(
                f"Unsupported action_head={self.action_head_type!r}; expected 'act' or 'flow_matching'"
            )
        hidden_dim = config.interaction.hidden_dim
        self.action_dim = int(config.action.action_dim)
        self.chunk_size = int(config.action.horizon)
        self.state_dim = int(config.action.state_dim)
        self.flow_state_dim = int(flow_state_dim)
        self.flow_bijection_blocks = int(flow_bijection_blocks)
        requested_state_encoding = str(flow_state_encoding)
        if self.action_head_type == "flow_matching" and self.flow_state_dim not in (0, self.state_dim):
            raise ValueError(
                "flow_state_dim must be 0 for legacy state tokens or equal "
                f"to action.state_dim ({self.state_dim}); got {self.flow_state_dim}"
            )
        if self.flow_state_dim == 0:
            if requested_state_encoding not in {"native_mlp", "state_tokens"}:
                raise ValueError("legacy flow_state_dim=0 requires state_tokens encoding")
            self.flow_state_encoding = "state_tokens"
        else:
            if requested_state_encoding not in {"native_mlp", "zero_pad", "state_tokens_zero_pad"}:
                raise ValueError("direct flow state requires native_mlp, zero_pad, or state_tokens_zero_pad encoding")
            self.flow_state_encoding = requested_state_encoding
        self.num_views = int(config.vision.num_views)
        self.history_encoder = None
        if config.history.enabled:
            if config.history.hidden_dim != hidden_dim:
                raise ValueError("history.hidden_dim must equal interaction.hidden_dim")
            self.history_encoder = HistoryEncoder(
                state_dim=config.history.state_dim,
                hidden_dim=config.history.hidden_dim, num_layers=config.history.num_layers,
                dropout=config.history.dropout, encoder_type=config.history.encoder_type,
                history_length=config.history.length,
            )

        self.text_encoder = TurboVLATextEncoder(config.text, hidden_dim=hidden_dim)
        self.vision_encoder = DINOv3VisionEncoder(config.vision)
        self.vision_projection = VisionProjection(
            in_dim=self.vision_encoder.hidden_size,
            out_dim=hidden_dim,
            hidden_dim=max(hidden_dim * 4, self.vision_encoder.hidden_size // 2),
            dropout=config.vision.dropout,
        )
        self.r3m_encoder = None
        self.r3m_projection = None
        self.r3m_history_cross_attention = None
        self.r3m_history_relation = None
        self.r3m_history_memory_dynamic_gate = None
        self.r3m_history_memory_residual_scale = None
        self.r3m_history_intentional_goal_factors = None
        self.r3m_history_intentional_current_factors = None
        self.r3m_history_intentional_relation_out = None
        self.r3m_history_dynamics_encoder = None
        self.r3m_history_dynamics_attention = None
        self.r3m_history_state_delta_projection = None
        self.r3m_history_goal_projection = None
        self.r3m_history_belief_fusion = None
        self.r3m_history_current_hidden_predictor = None
        self.r3m_history_innovation_projection = None
        self.r3m_history_innovation_gate = None
        self.r3m_history_future_horizon_embedding = None
        self.r3m_history_future_type_embedding = None
        self.r3m_history_future_context = None
        self.r3m_history_future_predictor = None
        self.r3m_history_tacit_memory_queries = None
        self.r3m_history_tacit_memory_type_embedding = None
        self.r3m_history_tacit_cross_attention = None
        self.r3m_history_tacit_relation = None
        self.r3m_history_tacit_gate_logit = None
        self.r3m_history_belief_to_tacit_attention = None
        self.r3m_history_belief_to_tacit_relation = None
        self.r3m_history_belief_to_tacit_gate = None
        if config.r3m.enabled:
            self.r3m_encoder = R3MResNet18Encoder(config.r3m)
            self.r3m_projection = VisionProjection(
                in_dim=config.r3m.output_dim,
                out_dim=hidden_dim,
                hidden_dim=max(hidden_dim * 2, config.r3m.output_dim),
                dropout=config.r3m.dropout,
            )
            self.r3m_view_embedding = nn.Parameter(
                torch.zeros(1, config.r3m.num_views, hidden_dim)
            )
            nn.init.trunc_normal_(self.r3m_view_embedding, std=0.02)
            if config.history.r3m_enabled:
                num_memory_queries = (
                    config.history.r3m_belief_num_slots
                    if config.history.r3m_predictive_belief
                    else config.history.r3m_memory_num_queries
                )
                self.r3m_history_memory_queries = nn.Parameter(
                    torch.zeros(num_memory_queries, hidden_dim)
                )
                self.r3m_history_memory_type_embedding = nn.Parameter(
                    torch.zeros(1, 1, num_memory_queries, hidden_dim)
                )
                self.r3m_history_cross_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=config.history.r3m_memory_num_heads,
                    dropout=config.history.r3m_memory_dropout,
                    batch_first=True,
                )
                self.r3m_history_relation = nn.Sequential(
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(config.history.r3m_memory_dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                gate_init = torch.tensor(
                    float(config.history.r3m_memory_gate_init), dtype=torch.float32
                )
                gate_logit = torch.logit(gate_init)
                if config.history.r3m_intentional_memory:
                    # Intentional memory is a single normalized path: the goal
                    # organizes retrieval itself instead of gating memory after
                    # it has been formed.
                    pass
                elif (
                    config.history.r3m_memory_dynamic_gate
                    or config.history.r3m_memory_residual_gate
                ):
                    self.r3m_history_memory_dynamic_gate = nn.Sequential(
                        nn.LayerNorm(hidden_dim * 3),
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.SiLU(),
                        nn.Dropout(config.history.r3m_memory_dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                    final_gate = self.r3m_history_memory_dynamic_gate[-1]
                    if config.history.r3m_memory_residual_gate:
                        # Preserve the scalar-gate function exactly at step 0.
                        # The residual network starts receiving gradients after
                        # its zero-initialized scalar scale begins to move.
                        self.r3m_history_memory_gate_logit = nn.Parameter(gate_logit)
                        self.r3m_history_memory_residual_scale = nn.Parameter(
                            torch.zeros((), dtype=torch.float32)
                        )
                        nn.init.zeros_(final_gate.bias)
                    else:
                        # Legacy dynamic gate: start from the previous scalar
                        # behavior, then predict each view/query gate directly.
                        nn.init.zeros_(final_gate.weight)
                        nn.init.constant_(final_gate.bias, float(gate_logit))
                else:
                    self.r3m_history_memory_gate_logit = nn.Parameter(gate_logit)
                nn.init.trunc_normal_(self.r3m_history_memory_queries, std=0.02)
                nn.init.trunc_normal_(self.r3m_history_memory_type_embedding, std=0.02)
                if config.history.r3m_intentional_memory:
                    operator_rank = config.history.r3m_intentional_operator_rank
                    self.r3m_history_intentional_goal_factors = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, operator_rank),
                        nn.Tanh(),
                    )
                    self.r3m_history_intentional_current_factors = nn.Linear(
                        hidden_dim,
                        operator_rank,
                        bias=False,
                    )
                    self.r3m_history_intentional_relation_out = nn.Linear(
                        operator_rank,
                        hidden_dim,
                        bias=False,
                    )
                elif config.history.r3m_predictive_belief:
                    # Shared MambaBlock dispatches CUDA BF16 dynamics to matched scan.
                    self.r3m_history_dynamics_encoder = nn.ModuleList(
                        [MambaBlock(hidden_dim) for _ in range(config.history.num_layers)]
                    )
                    self.r3m_history_dynamics_attention = nn.MultiheadAttention(
                        embed_dim=hidden_dim,
                        num_heads=config.history.r3m_memory_num_heads,
                        dropout=config.history.r3m_memory_dropout,
                        batch_first=True,
                    )
                    self.r3m_history_state_delta_projection = nn.Linear(hidden_dim, hidden_dim)
                    if not config.history.r3m_controlled_causal_belief:
                        self.r3m_history_goal_projection = nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.SiLU(),
                        )
                    self.r3m_history_belief_fusion = nn.Sequential(
                        nn.LayerNorm(hidden_dim * 4),
                        nn.Linear(hidden_dim * 4, hidden_dim * 2),
                        nn.SiLU(),
                        nn.Dropout(config.history.r3m_memory_dropout),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                    )
                    self.r3m_history_current_hidden_predictor = nn.Sequential(
                        nn.LayerNorm(hidden_dim),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    self.r3m_history_innovation_projection = nn.Linear(hidden_dim, hidden_dim)
                    self.r3m_history_innovation_gate = nn.Sequential(
                        nn.LayerNorm(hidden_dim * 3),
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.SiLU(),
                        nn.Dropout(config.history.r3m_memory_dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                    nn.init.zeros_(self.r3m_history_innovation_gate[-1].weight)
                    nn.init.zeros_(self.r3m_history_innovation_gate[-1].bias)
                    if not config.history.r3m_tacit_belief_fusion:
                        horizons = tuple(config.history.r3m_belief_future_horizons)
                        self.r3m_history_future_horizon_embedding = nn.Parameter(
                            torch.zeros(1, len(horizons), hidden_dim)
                        )
                        self.r3m_history_future_type_embedding = nn.Parameter(
                            torch.zeros(1, len(horizons), hidden_dim)
                        )
                        self.r3m_history_future_context = nn.Sequential(
                            nn.LayerNorm(hidden_dim * 2),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                            nn.SiLU(),
                        )
                        self.r3m_history_future_predictor = nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.SiLU(),
                            nn.Dropout(config.history.r3m_memory_dropout),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                        )
                        nn.init.trunc_normal_(self.r3m_history_future_horizon_embedding, std=0.02)
                        nn.init.trunc_normal_(self.r3m_history_future_type_embedding, std=0.02)
                    if config.history.r3m_tacit_belief_fusion:
                        tacit_queries = config.history.r3m_memory_num_queries
                        self.r3m_history_tacit_memory_queries = nn.Parameter(
                            torch.zeros(tacit_queries, hidden_dim)
                        )
                        self.r3m_history_tacit_memory_type_embedding = nn.Parameter(
                            torch.zeros(1, 1, tacit_queries, hidden_dim)
                        )
                        self.r3m_history_tacit_cross_attention = nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=config.history.r3m_memory_num_heads,
                            dropout=config.history.r3m_memory_dropout,
                            batch_first=True,
                        )
                        self.r3m_history_tacit_relation = nn.Sequential(
                            nn.LayerNorm(hidden_dim * 2),
                            nn.Linear(hidden_dim * 2, hidden_dim * 2),
                            nn.GELU(),
                            nn.Dropout(config.history.r3m_memory_dropout),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                        )
                        self.r3m_history_tacit_gate_logit = nn.Parameter(gate_logit.clone())
                        self.r3m_history_belief_to_tacit_attention = nn.MultiheadAttention(
                            embed_dim=hidden_dim,
                            num_heads=config.history.r3m_memory_num_heads,
                            dropout=config.history.r3m_memory_dropout,
                            batch_first=True,
                        )
                        self.r3m_history_belief_to_tacit_relation = nn.Sequential(
                            nn.LayerNorm(hidden_dim * 2),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                            nn.SiLU(),
                            nn.Dropout(config.history.r3m_memory_dropout),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                        )
                        self.r3m_history_belief_to_tacit_gate = nn.Sequential(
                            nn.LayerNorm(hidden_dim * 3),
                            nn.Linear(hidden_dim * 3, hidden_dim),
                            nn.SiLU(),
                            nn.Dropout(config.history.r3m_memory_dropout),
                            nn.Linear(hidden_dim, 1),
                        )
                        correction_gate = self.r3m_history_belief_to_tacit_gate[-1]
                        nn.init.zeros_(correction_gate.weight)
                        nn.init.constant_(
                            correction_gate.bias,
                            float(
                                torch.logit(
                                    torch.tensor(
                                        config.history.r3m_tacit_belief_gate_init,
                                        dtype=torch.float32,
                                    )
                                )
                            ),
                        )
                        nn.init.trunc_normal_(self.r3m_history_tacit_memory_queries, std=0.02)
                        nn.init.trunc_normal_(
                            self.r3m_history_tacit_memory_type_embedding,
                            std=0.02,
                        )
        else:
            self.register_parameter("r3m_view_embedding", None)
        if config.history.visual_enabled:
            self.history_visual_view_embedding = nn.Parameter(
                torch.zeros(1, config.vision.num_views, hidden_dim)
            )
            self.history_visual_time_embedding = nn.Parameter(
                torch.zeros(1, config.history.length, 1, hidden_dim)
            )
            nn.init.trunc_normal_(self.history_visual_view_embedding, std=0.02)
            nn.init.trunc_normal_(self.history_visual_time_embedding, std=0.02)
        else:
            self.register_parameter("history_visual_view_embedding", None)
            self.register_parameter("history_visual_time_embedding", None)

        if config.vision.position_embedding == "learned_patch":
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, 1, hidden_dim))
            self.patch_position_embedding = nn.Parameter(
                torch.zeros(1, self.num_views, self.vision_encoder.num_patches, hidden_dim)
            )
            self.patch_position_scale = nn.Parameter(
                torch.full((1, self.num_views, 1, 1), float(config.vision.position_scale_init))
            )
            nn.init.trunc_normal_(self.patch_position_embedding, std=config.vision.position_init_std)
        else:
            self.view_embedding = nn.Parameter(torch.zeros(1, self.num_views, hidden_dim))
            self.register_parameter("patch_position_embedding", None)
            self.register_parameter("patch_position_scale", None)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)

        self.vision_language_interaction = VisionLanguageInteraction(config.interaction)
        if self.action_head_type == "act":
            self.action_head = TurboVLAActionHead(
                config=config.action,
                hidden_dim=hidden_dim,
                nheads=config.interaction.nheads,
                dim_feedforward=config.interaction.dim_feedforward,
            )
        else:
            # Legacy flow checkpoints route state through two condition tokens.
            # The direct route deliberately has no StateProjection parameters.
            if self.flow_state_dim == 0 or self.flow_state_encoding == "state_tokens_zero_pad":
                self.state_proj_module = StateProjection(config.action, hidden_dim=hidden_dim)
            self.flow_action_policy = MyVLAFlowMatchingActionHead(
                hidden_size=hidden_dim,
                action_dim=self.action_dim,
                chunk_size=self.chunk_size,
                num_heads=int(flow_num_heads),
                condition_layers=int(flow_condition_layers),
                dit_layers=int(flow_dit_layers),
                num_target_vision_tokens=int(flow_target_tokens),
                num_static_future_tokens=int(flow_static_tokens),
                state_dim=self.flow_state_dim,
                bijection_blocks=self.flow_bijection_blocks,
                state_encoding=(
                    "native_mlp" if self.flow_state_encoding == "state_tokens"
                    else "zero_pad" if self.flow_state_encoding == "state_tokens_zero_pad"
                    else self.flow_state_encoding
                ),
            )

    def _normalize_samples(self, samples: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(samples, Mapping):
            if "dinov3" not in samples:
                raise ValueError("samples mapping must contain 'dinov3'")
            pixel_values = samples["dinov3"]
        else:
            pixel_values = samples
        if pixel_values.ndim == 6:
            pixel_values = pixel_values[:, -1]
        if pixel_values.ndim != 5:
            raise ValueError(f"samples must be [B,V,3,H,W] or [B,T,V,3,H,W], got {tuple(pixel_values.shape)}")
        return pixel_values

    def _position_visual_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.config.vision.position_embedding == "learned_patch":
            if tokens.shape[2] != self.patch_position_embedding.shape[2]:
                raise ValueError(
                    f"configured patch position length {self.patch_position_embedding.shape[2]} "
                    f"does not match encoded length {tokens.shape[2]}"
                )
            position = self.patch_position_embedding.to(device=tokens.device, dtype=tokens.dtype)
            scale = self.patch_position_scale.to(device=tokens.device, dtype=tokens.dtype)
            view = self.view_embedding.to(device=tokens.device, dtype=tokens.dtype)
            return tokens + scale * position + view
        view = self.view_embedding[:, :, None, :].to(device=tokens.device, dtype=tokens.dtype)
        return tokens + view

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = self.vision_encoder(pixel_values)
        tokens = tokens.to(dtype=self.vision_projection.skip.weight.dtype)
        tokens = self.vision_projection(tokens)
        return self._position_visual_tokens(tokens).flatten(1, 2)

    def _encode_r3m_current_outputs(
        self,
        samples: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.r3m_encoder is None or self.r3m_projection is None:
            raise RuntimeError("R3M current-image encoder is not enabled")
        cached_features = (
            samples.get("r3m_features") if isinstance(samples, Mapping) else None
        )
        if cached_features is not None:
            if not getattr(self.config.r3m, "frozen", True):
                raise ValueError("r3m_features cache requires frozen R3M; it cannot provide backbone gradients")
            expected_features = (
                cached_features.shape[0], self.config.r3m.num_views, self.config.r3m.output_dim
            )
            if tuple(cached_features.shape) != expected_features:
                raise ValueError(
                    f"r3m_features must be [B,{self.config.r3m.num_views},{self.config.r3m.output_dim}], "
                    f"got {tuple(cached_features.shape)}"
                )
            raw_embeddings = cached_features
        else:
            pixel_values = self._mapping_tensor(samples, "r3m")
            expected = (
                self.config.r3m.num_views,
                3,
                self.config.r3m.image_size,
                self.config.r3m.image_size,
            )
            if pixel_values.ndim != 5 or tuple(pixel_values.shape[1:]) != expected:
                raise ValueError(
                    f"r3m must be [B,{','.join(str(v) for v in expected)}], "
                    f"got {tuple(pixel_values.shape)}"
                )
            raw_embeddings = self.r3m_encoder(pixel_values)
        tokens = raw_embeddings.to(dtype=self.r3m_projection.skip.weight.dtype)
        tokens = self.r3m_projection(tokens)
        view = self.r3m_view_embedding.to(device=tokens.device, dtype=tokens.dtype)
        return tokens + view, raw_embeddings

    def encode_r3m_current(
        self,
        samples: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        tokens, _ = self._encode_r3m_current_outputs(samples)
        return tokens

    def _project_r3m_history_frames(
        self,
        values: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        cached_features: bool = False,
    ) -> torch.Tensor:
        if self.r3m_encoder is None or self.r3m_projection is None:
            raise RuntimeError("R3M history requires the current R3M encoder")
        batch_size, history_length = values.shape[:2]
        flat_frames = values.flatten(0, 1)
        valid_indices = history_mask.flatten().nonzero(as_tuple=False).flatten()
        if valid_indices.numel():
            valid_frames = flat_frames.index_select(0, valid_indices)
            valid_embeddings = valid_frames if cached_features else self.r3m_encoder(valid_frames)
            valid_embeddings = valid_embeddings.to(dtype=self.r3m_projection.skip.weight.dtype)
            valid_tokens = self.r3m_projection(valid_embeddings)
            history_tokens = valid_tokens.new_zeros(
                batch_size * history_length,
                self.config.r3m.num_views,
                self.config.interaction.hidden_dim,
            )
            history_tokens = history_tokens.index_copy(0, valid_indices, valid_tokens)
        else:
            history_tokens = self.r3m_projection.skip.weight.new_zeros(
                batch_size * history_length,
                self.config.r3m.num_views,
                self.config.interaction.hidden_dim,
            )
        return history_tokens.view(
            batch_size,
            history_length,
            self.config.r3m.num_views,
            self.config.interaction.hidden_dim,
        )

    def encode_r3m_history(
        self,
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        current_r3m_tokens: torch.Tensor,
        history_mask: torch.Tensor,
        history_state_tokens: torch.Tensor | None = None,
        goal_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve compact, current-conditioned memories from each R3M view."""
        if not self.config.history.r3m_enabled:
            raise RuntimeError("R3M history is not enabled")
        if self.r3m_encoder is None or self.r3m_projection is None:
            raise RuntimeError("R3M history requires the current R3M encoder")
        if self.r3m_history_cross_attention is None or self.r3m_history_relation is None:
            raise RuntimeError("R3M current-conditioned history memory is not constructed")
        cached_features = isinstance(samples, Mapping) and "r3m_history_features" in samples
        pixel_values = self._mapping_tensor(samples, "r3m_history_features" if cached_features else "r3m_history")
        expected = ((self.config.history.length, self.config.r3m.num_views, self.config.r3m.output_dim)
                    if cached_features else (self.config.history.length, self.config.r3m.num_views, 3,
                                             self.config.r3m.image_size, self.config.r3m.image_size))
        expected_ndim = 4 if cached_features else 6
        if pixel_values.ndim != expected_ndim or tuple(pixel_values.shape[1:]) != expected:
            raise ValueError(
                f"r3m_history must be [B,{','.join(str(v) for v in expected)}], "
                f"got {tuple(pixel_values.shape)}"
            )
        history_mask = history_mask.to(device=pixel_values.device, dtype=torch.bool)
        if history_mask.shape != pixel_values.shape[:2]:
            raise ValueError(
                f"history_mask must be {tuple(pixel_values.shape[:2])}, "
                f"got {tuple(history_mask.shape)}"
            )
        expected_current = (
            pixel_values.shape[0],
            self.config.r3m.num_views,
            self.config.interaction.hidden_dim,
        )
        if tuple(current_r3m_tokens.shape) != expected_current:
            raise ValueError(
                f"current_r3m_tokens must be {expected_current}, "
                f"got {tuple(current_r3m_tokens.shape)}"
            )

        batch_size, history_length = pixel_values.shape[:2]
        history_tokens = self._project_r3m_history_frames(pixel_values, history_mask, cached_features=cached_features)
        num_views = self.config.r3m.num_views
        hidden_dim = self.config.interaction.hidden_dim
        num_queries = self.config.history.r3m_memory_num_queries
        view = self.r3m_view_embedding.to(
            device=history_tokens.device,
            dtype=history_tokens.dtype,
        )
        current_base = current_r3m_tokens.to(dtype=history_tokens.dtype) - view
        intentional_memory = bool(
            getattr(self.config.history, "r3m_intentional_memory", False)
        )

        # Preserve absolute historical features as values and encode relative
        # positions (-12..-1) only in the attention keys for the v10 recipe.
        # The legacy switch can still rotate values when loading older recipes.
        # Views are folded into the batch and current-frame queries are at p=0.
        history_values_by_view = history_tokens.permute(0, 2, 1, 3)
        history_values_by_view = history_values_by_view.reshape(
            batch_size * num_views,
            history_length,
            hidden_dim,
        )
        relative_positions = torch.arange(
            -history_length,
            0,
            device=history_tokens.device,
        )
        history_keys_by_view = apply_rotary_position_encoding(
            history_values_by_view,
            relative_positions,
            base=self.config.history.r3m_rope_base,
        )
        if self.config.history.r3m_memory_rope_values:
            history_values_by_view = history_keys_by_view

        learned_queries = self.r3m_history_memory_queries.to(
            device=history_tokens.device,
            dtype=history_tokens.dtype,
        )
        query_current = current_base
        if intentional_memory:
            required_modules = (
                self.r3m_history_intentional_goal_factors,
                self.r3m_history_intentional_current_factors,
                self.r3m_history_intentional_relation_out,
            )
            if any(module is None for module in required_modules):
                raise RuntimeError("intentional R3M query operator is not constructed")
            expected_goal = (batch_size, hidden_dim)
            if goal_summary is None or tuple(goal_summary.shape) != expected_goal:
                actual = None if goal_summary is None else tuple(goal_summary.shape)
                raise ValueError(
                    f"intentional R3M memory requires goal_summary {expected_goal}, got {actual}"
                )
            goal_factors = self.r3m_history_intentional_goal_factors(
                goal_summary.to(device=history_tokens.device, dtype=history_tokens.dtype)
            )
            current_factors = self.r3m_history_intentional_current_factors(current_base)
            from_to_relation = self.r3m_history_intentional_relation_out(
                current_factors * goal_factors[:, None, :]
            )
            query_current = current_base + from_to_relation
        queries = query_current[:, :, None, :] + learned_queries[None, None]
        queries = queries.reshape(batch_size * num_views, num_queries, hidden_dim)

        valid_by_view = history_mask[:, None, :].expand(-1, num_views, -1)
        valid_by_view = valid_by_view.reshape(batch_size * num_views, history_length)
        key_padding_mask = ~valid_by_view
        empty_rows = ~valid_by_view.any(dim=1)
        if empty_rows.any():
            # MultiheadAttention returns NaNs when every key is masked. Expose
            # one zero/dummy slot and mask the resulting memory again below.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[empty_rows, 0] = False

        memory, _ = self.r3m_history_cross_attention(
            query=queries,
            key=history_keys_by_view,
            value=history_values_by_view,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        memory = memory.view(batch_size, num_views, num_queries, hidden_dim)
        current_for_memory = current_base[:, :, None, :].expand_as(memory)
        memory_content = self.r3m_history_relation(
            torch.cat([memory, current_for_memory], dim=-1)
        )

        memory = memory_content + view[:, :, None, :]
        memory = memory + self.r3m_history_memory_type_embedding.to(
            device=memory.device,
            dtype=memory.dtype,
        )
        if intentional_memory:
            # The focal goal has already organized retrieval.  Keeping this
            # path ungated avoids constructing a second, post-hoc authority
            # over the same tacit memory.
            pass
        elif (
            self.config.history.r3m_memory_dynamic_gate
            or self.config.history.r3m_memory_residual_gate
        ):
            expected_state_tokens = (batch_size, history_length, hidden_dim)
            if history_state_tokens is None or tuple(history_state_tokens.shape) != expected_state_tokens:
                actual = None if history_state_tokens is None else tuple(history_state_tokens.shape)
                raise ValueError(
                    f"dynamic R3M gate requires history_state_tokens {expected_state_tokens}, got {actual}"
                )
            state_tokens = history_state_tokens.to(device=memory.device, dtype=memory.dtype)
            state_valid = history_mask.unsqueeze(-1).to(dtype=memory.dtype)
            state_summary = (state_tokens * state_valid).sum(dim=1)
            state_summary = state_summary / state_valid.sum(dim=1).clamp_min(1.0)
            state_for_memory = state_summary[:, None, None, :].expand_as(memory_content)
            gate_features = torch.cat(
                [current_for_memory, memory_content, state_for_memory],
                dim=-1,
            )
            if self.r3m_history_memory_dynamic_gate is None:
                raise RuntimeError("dynamic R3M history gate is not constructed")
            predicted_gate = self.r3m_history_memory_dynamic_gate(gate_features)
            if self.config.history.r3m_memory_residual_gate:
                if self.r3m_history_memory_residual_scale is None:
                    raise RuntimeError("residual R3M history gate scale is not constructed")
                base_gate_logit = self.r3m_history_memory_gate_logit.to(
                    device=memory.device,
                    dtype=memory.dtype,
                )
                residual_scale = torch.tanh(
                    self.r3m_history_memory_residual_scale.to(
                        device=memory.device,
                        dtype=memory.dtype,
                    )
                )
                gate = torch.sigmoid(base_gate_logit + residual_scale * predicted_gate)
            else:
                gate = torch.sigmoid(predicted_gate)
        else:
            gate = torch.sigmoid(self.r3m_history_memory_gate_logit).to(
                device=memory.device,
                dtype=memory.dtype,
            )
        if not intentional_memory:
            memory = gate * memory

        memory_mask = history_mask.any(dim=1)[:, None, None].expand(
            -1, num_views, num_queries
        )
        token_dropout = float(self.config.history.r3m_memory_token_dropout)
        if self.training and token_dropout > 0.0:
            keep_tokens = torch.rand(
                memory_mask.shape,
                device=memory_mask.device,
            ) >= token_dropout
            memory_mask = memory_mask & keep_tokens
        history_visual_tokens = memory.flatten(1, 2)
        history_visual_mask = memory_mask.flatten(1, 2)
        history_visual_tokens = history_visual_tokens.masked_fill(
            ~history_visual_mask.unsqueeze(-1),
            0.0,
        )
        return history_visual_tokens, history_visual_mask

    def encode_r3m_intentional_memory(
        self,
        samples: Mapping[str, torch.Tensor],
        current_r3m_tokens: torch.Tensor,
        history_mask: torch.Tensor,
        history_state_tokens: torch.Tensor,
        goal_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Form one goal-operated tacit memory."""
        if not bool(getattr(self.config.history, "r3m_intentional_memory", False)):
            raise RuntimeError("intentional R3M memory is not enabled")
        tokens, token_mask = self.encode_r3m_history(
            samples,
            current_r3m_tokens,
            history_mask,
            history_state_tokens,
            goal_summary,
        )
        return tokens, token_mask, {}

    def _encode_r3m_tacit_belief_fusion(
        self,
        history_by_view: torch.Tensor,
        history_mask: torch.Tensor,
        current_base: torch.Tensor,
        posterior: torch.Tensor,
        state_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Let explicit belief correct, but never replace, v8-style tacit memory."""
        required_modules = (
            self.r3m_history_tacit_cross_attention,
            self.r3m_history_tacit_relation,
            self.r3m_history_belief_to_tacit_attention,
            self.r3m_history_belief_to_tacit_relation,
            self.r3m_history_belief_to_tacit_gate,
        )
        if any(module is None for module in required_modules):
            raise RuntimeError("tacit-belief fusion modules are not fully constructed")
        if (
            self.r3m_history_tacit_memory_queries is None
            or self.r3m_history_tacit_memory_type_embedding is None
            or self.r3m_history_tacit_gate_logit is None
        ):
            raise RuntimeError("tacit-belief fusion parameters are not fully constructed")

        batch_size, num_views, hidden_dim = current_base.shape
        history_length = self.config.history.length
        num_queries = self.config.history.r3m_memory_num_queries
        expected_history = (batch_size * num_views, history_length, hidden_dim)
        if tuple(history_by_view.shape) != expected_history:
            raise ValueError(
                f"tacit history must be {expected_history}, got {tuple(history_by_view.shape)}"
            )
        if tuple(history_mask.shape) != (batch_size, history_length):
            raise ValueError("tacit memory history mask has an invalid shape")
        if posterior.shape[:2] != (batch_size, num_views) or posterior.shape[-1] != hidden_dim:
            raise ValueError("tacit memory posterior has an invalid shape")
        if tuple(state_summary.shape) != (batch_size, hidden_dim):
            raise ValueError("tacit memory state summary has an invalid shape")

        # This branch deliberately preserves v8's full RoPE behavior. Rotating
        # both keys and values keeps the temporal trace available to the action
        # path, while the predictive branch continues to use key-only RoPE.
        positions = torch.arange(-history_length, 0, device=history_by_view.device)
        temporal_history = apply_rotary_position_encoding(
            history_by_view,
            positions,
            base=self.config.history.r3m_rope_base,
        )
        learned_queries = self.r3m_history_tacit_memory_queries.to(
            device=history_by_view.device,
            dtype=history_by_view.dtype,
        )
        queries = current_base[:, :, None, :] + learned_queries[None, None, :, :]
        queries = queries.reshape(batch_size * num_views, num_queries, hidden_dim)

        valid = history_mask[:, None, :].expand(-1, num_views, -1).reshape(
            batch_size * num_views,
            history_length,
        )
        empty = ~valid.any(dim=1)
        padding = ~valid
        if empty.any():
            padding = padding.clone()
            padding[empty, 0] = False
        tacit_memory, _ = self.r3m_history_tacit_cross_attention(
            query=queries,
            key=temporal_history,
            value=temporal_history,
            key_padding_mask=padding,
            need_weights=False,
        )
        tacit_memory = tacit_memory.masked_fill(empty[:, None, None], 0.0)
        current_for_memory = current_base[:, :, None, :].expand(
            -1,
            -1,
            num_queries,
            -1,
        ).reshape(batch_size * num_views, num_queries, hidden_dim)
        tacit_content = self.r3m_history_tacit_relation(
            torch.cat([tacit_memory, current_for_memory], dim=-1)
        )

        posterior_by_view = posterior.reshape(
            batch_size * num_views,
            posterior.shape[2],
            hidden_dim,
        )
        belief_correction, _ = self.r3m_history_belief_to_tacit_attention(
            query=tacit_content,
            key=posterior_by_view,
            value=posterior_by_view,
            need_weights=False,
        )
        belief_correction = self.r3m_history_belief_to_tacit_relation(
            torch.cat([belief_correction, tacit_content], dim=-1)
        )
        state_for_memory = state_summary[:, None, None, :].expand(
            -1,
            num_views,
            num_queries,
            -1,
        ).reshape(batch_size * num_views, num_queries, hidden_dim)
        correction_gate = torch.sigmoid(
            self.r3m_history_belief_to_tacit_gate(
                torch.cat([tacit_content, belief_correction, state_for_memory], dim=-1)
            )
        )

        view = self.r3m_view_embedding.to(
            device=tacit_content.device,
            dtype=tacit_content.dtype,
        )
        tacit_tokens = tacit_content.view(
            batch_size,
            num_views,
            num_queries,
            hidden_dim,
        )
        tacit_tokens = tacit_tokens + view[:, :, None, :]
        tacit_tokens = tacit_tokens + self.r3m_history_tacit_memory_type_embedding.to(
            device=tacit_tokens.device,
            dtype=tacit_tokens.dtype,
        )
        tacit_gate = torch.sigmoid(self.r3m_history_tacit_gate_logit).to(
            device=tacit_tokens.device,
            dtype=tacit_tokens.dtype,
        )
        fused = tacit_gate * tacit_tokens
        fused = fused + correction_gate.view(
            batch_size,
            num_views,
            num_queries,
            1,
        ) * belief_correction.view(batch_size, num_views, num_queries, hidden_dim)

        token_mask = history_mask.any(dim=1)[:, None, None].expand(
            -1,
            num_views,
            num_queries,
        )
        fused = fused.flatten(1, 2)
        token_mask = token_mask.flatten(1, 2)
        fused = fused.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        valid_gate = correction_gate.view(batch_size, num_views, num_queries)
        gate_mean = self._masked_mean_loss(valid_gate.flatten(1, 2), token_mask)
        return fused, token_mask, gate_mean

    @staticmethod
    def _masked_mean_loss(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=values.device, dtype=values.dtype)
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        denominator = mask.expand_as(values).sum().clamp_min(1.0)
        return (values * mask).sum() / denominator

    def encode_r3m_predictive_belief(
        self,
        samples: Mapping[str, torch.Tensor],
        current_r3m_tokens: torch.Tensor,
        history_mask: torch.Tensor,
        history_state_tokens: torch.Tensor,
        goal_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Build a belief from semantic history, dynamics, and current observation."""
        if not self.config.history.r3m_predictive_belief:
            raise RuntimeError("predictive R3M belief is not enabled")
        required_modules = (
            self.r3m_history_cross_attention,
            self.r3m_history_relation,
            self.r3m_history_dynamics_encoder,
            self.r3m_history_dynamics_attention,
            self.r3m_history_state_delta_projection,
            self.r3m_history_belief_fusion,
            self.r3m_history_current_hidden_predictor,
            self.r3m_history_innovation_projection,
            self.r3m_history_innovation_gate,
        )
        if any(module is None for module in required_modules):
            raise RuntimeError("predictive R3M belief modules are not fully constructed")
        controlled_causal = bool(
            getattr(self.config.history, "r3m_controlled_causal_belief", False)
        )
        if not controlled_causal and self.r3m_history_goal_projection is None:
            raise RuntimeError("predictive R3M goal projection is not constructed")
        if not bool(getattr(self.config.history, "r3m_tacit_belief_fusion", False)):
            future_modules = (
                self.r3m_history_future_horizon_embedding,
                self.r3m_history_future_type_embedding,
                self.r3m_history_future_context,
                self.r3m_history_future_predictor,
            )
            if any(module is None for module in future_modules):
                raise RuntimeError("future condition tokens are not fully constructed")

        cached_history_features = isinstance(samples, Mapping) and "r3m_history_features" in samples
        pixel_values = self._mapping_tensor(
            samples, "r3m_history_features" if cached_history_features else "r3m_history"
        )
        batch_size = pixel_values.shape[0]
        history_length = self.config.history.length
        num_views = self.config.r3m.num_views
        hidden_dim = self.config.interaction.hidden_dim
        num_slots = self.config.history.r3m_belief_num_slots
        expected_pixels = (
            (batch_size, history_length, num_views, self.config.r3m.output_dim)
            if cached_history_features
            else (batch_size, history_length, num_views, 3, self.config.r3m.image_size, self.config.r3m.image_size)
        )
        if tuple(pixel_values.shape) != expected_pixels:
            raise ValueError(
                f"r3m_history must be {expected_pixels}, got {tuple(pixel_values.shape)}"
            )
        history_mask = history_mask.to(device=pixel_values.device, dtype=torch.bool)
        if tuple(history_mask.shape) != (batch_size, history_length):
            raise ValueError("predictive belief history mask has an invalid shape")
        if tuple(history_state_tokens.shape) != (batch_size, history_length, hidden_dim):
            raise ValueError("predictive belief state tokens have an invalid shape")
        if tuple(current_r3m_tokens.shape) != (batch_size, num_views, hidden_dim):
            raise ValueError("predictive belief current R3M tokens have an invalid shape")
        if tuple(goal_summary.shape) != (batch_size, hidden_dim):
            raise ValueError("predictive belief goal summary has an invalid shape")

        history_tokens = self._project_r3m_history_frames(
            pixel_values, history_mask, cached_features=cached_history_features
        )
        history_by_view = history_tokens.permute(0, 2, 1, 3).reshape(
            batch_size * num_views,
            history_length,
            hidden_dim,
        )
        positions = torch.arange(-history_length, 0, device=history_tokens.device)
        semantic_keys = apply_rotary_position_encoding(
            history_by_view,
            positions,
            base=self.config.history.r3m_rope_base,
        )

        learned_slots = self.r3m_history_memory_queries.to(
            device=history_tokens.device,
            dtype=history_tokens.dtype,
        )
        if controlled_causal:
            goal = torch.zeros_like(goal_summary, device=history_tokens.device).to(
                dtype=history_tokens.dtype
            )
            queries = learned_slots[None, None].expand(batch_size, num_views, -1, -1)
        else:
            goal = self.r3m_history_goal_projection(
                goal_summary.to(device=history_tokens.device, dtype=history_tokens.dtype)
            )
            queries = learned_slots[None, None] + goal[:, None, None, :]
        queries = queries.expand(-1, num_views, -1, -1).reshape(
            batch_size * num_views,
            num_slots,
            hidden_dim,
        )
        valid_by_view = history_mask[:, None, :].expand(-1, num_views, -1).reshape(
            batch_size * num_views,
            history_length,
        )

        def attend(
            module: nn.MultiheadAttention,
            keys: torch.Tensor,
            values: torch.Tensor,
            valid: torch.Tensor,
        ) -> torch.Tensor:
            empty = ~valid.any(dim=1)
            padding = ~valid
            if empty.any():
                padding = padding.clone()
                padding[empty, 0] = False
            output, _ = module(
                query=queries,
                key=keys,
                value=values,
                key_padding_mask=padding,
                need_weights=False,
            )
            return output.masked_fill(empty[:, None, None], 0.0)

        semantic_memory = attend(
            self.r3m_history_cross_attention,
            semantic_keys,
            history_by_view,
            valid_by_view,
        ).view(batch_size, num_views, num_slots, hidden_dim)
        view = self.r3m_view_embedding.to(
            device=history_tokens.device,
            dtype=history_tokens.dtype,
        )
        current_base = current_r3m_tokens.to(dtype=history_tokens.dtype) - view
        current_for_slots = current_base[:, :, None, :].expand_as(semantic_memory)
        goal_for_slots = goal[:, None, None, :].expand_as(semantic_memory)
        semantic_memory = self.r3m_history_relation(
            torch.cat(
                [
                    semantic_memory,
                    current_for_slots if controlled_causal else goal_for_slots,
                ],
                dim=-1,
            )
        )

        visual_delta = torch.zeros_like(history_tokens)
        state_delta = torch.zeros_like(history_state_tokens)
        pair_mask = torch.zeros_like(history_mask)
        pair_mask[:, 1:] = history_mask[:, 1:] & history_mask[:, :-1]
        visual_delta[:, 1:] = history_tokens[:, 1:] - history_tokens[:, :-1]
        state_delta[:, 1:] = history_state_tokens[:, 1:] - history_state_tokens[:, :-1]
        projected_state_delta = self.r3m_history_state_delta_projection(
            state_delta.to(dtype=history_tokens.dtype)
        )
        dynamics = visual_delta + projected_state_delta[:, :, None, :]
        dynamics = dynamics.masked_fill(~pair_mask[:, :, None, None], 0.0)
        dynamics_by_view = dynamics.permute(0, 2, 1, 3).reshape(
            batch_size * num_views,
            history_length,
            hidden_dim,
        )
        pair_valid_by_view = pair_mask[:, None, :].expand(-1, num_views, -1).reshape(
            batch_size * num_views,
            history_length,
        )
        for layer in self.r3m_history_dynamics_encoder:
            dynamics_by_view = layer(dynamics_by_view, pair_valid_by_view)
        dynamics_keys = apply_rotary_position_encoding(
            dynamics_by_view,
            positions,
            base=self.config.history.r3m_rope_base,
        )
        dynamics_memory = attend(
            self.r3m_history_dynamics_attention,
            dynamics_keys,
            dynamics_by_view,
            pair_valid_by_view,
        ).view(batch_size, num_views, num_slots, hidden_dim)

        state_valid = history_mask.unsqueeze(-1).to(dtype=history_state_tokens.dtype)
        state_summary = (history_state_tokens * state_valid).sum(dim=1)
        state_summary = state_summary / state_valid.sum(dim=1).clamp_min(1.0)
        state_for_slots = state_summary[:, None, None, :].expand_as(semantic_memory)
        prior = self.r3m_history_belief_fusion(
            torch.cat(
                [
                    semantic_memory,
                    dynamics_memory,
                    state_for_slots,
                    current_for_slots if controlled_causal else goal_for_slots,
                ],
                dim=-1,
            )
        )
        history_any = history_mask.any(dim=1)
        prior = prior.masked_fill(~history_any[:, None, None, None], 0.0)
        prior_summary = prior.mean(dim=2)

        predicted_current_hidden = self.r3m_history_current_hidden_predictor(prior_summary)
        innovation = current_base - predicted_current_hidden
        innovation_for_slots = innovation[:, :, None, :].expand_as(prior)
        gain_features = torch.cat(
            [
                prior,
                innovation_for_slots,
                state_for_slots if controlled_causal else goal_for_slots,
            ],
            dim=-1,
        )
        gain_delta = self.r3m_history_innovation_gate(gain_features)
        gain_logit = self.r3m_history_memory_gate_logit.to(
            device=prior.device,
            dtype=prior.dtype,
        )
        gain = torch.sigmoid(gain_logit + gain_delta)
        posterior = prior + gain * self.r3m_history_innovation_projection(innovation_for_slots)
        posterior = posterior.masked_fill(~history_any[:, None, None, None], 0.0)

        belief_tokens = posterior + view[:, :, None, :]
        belief_tokens = belief_tokens + self.r3m_history_memory_type_embedding.to(
            device=posterior.device,
            dtype=posterior.dtype,
        )
        belief_mask = history_any[:, None, None].expand(-1, num_views, num_slots)
        belief_tokens = belief_tokens.flatten(1, 2)
        belief_mask = belief_mask.flatten(1, 2)
        belief_tokens = belief_tokens.masked_fill(~belief_mask.unsqueeze(-1), 0.0)

        tacit_gate_mean = posterior.new_zeros(())
        if bool(getattr(self.config.history, "r3m_tacit_belief_fusion", False)):
            tokens, token_mask, tacit_gate_mean = self._encode_r3m_tacit_belief_fusion(
                history_by_view,
                history_mask,
                current_base,
                posterior,
                state_summary,
            )
        else:
            horizon_embedding = self.r3m_history_future_horizon_embedding.to(
                device=posterior.device,
                dtype=posterior.dtype,
            )
            future_context = self.r3m_history_future_context(
                torch.cat([posterior.mean(dim=(1, 2)), goal], dim=-1)
            )[:, None, :]
            future_tokens = self.r3m_history_future_predictor(
                future_context + horizon_embedding
            )
            future_tokens = future_tokens + self.r3m_history_future_type_embedding.to(
                device=posterior.device,
                dtype=posterior.dtype,
            )
            future_mask = history_any[:, None].expand(-1, future_tokens.shape[1])
            future_tokens = future_tokens.masked_fill(~future_mask.unsqueeze(-1), 0.0)
            tokens = torch.cat([belief_tokens, future_tokens], dim=1)
            token_mask = torch.cat([belief_mask, future_mask], dim=1)
        auxiliary = {
            "belief_gain_mean": gain.mean(),
        }
        if bool(getattr(self.config.history, "r3m_tacit_belief_fusion", False)):
            auxiliary["belief_tacit_gate_mean"] = tacit_gate_mean
        return tokens, token_mask, auxiliary

    @staticmethod
    def _mapping_tensor(
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        key: str,
    ) -> torch.Tensor:
        if not isinstance(samples, Mapping) or key not in samples:
            raise ValueError(f"samples mapping must contain {key!r}")
        return samples[key]

    def encode_dinov3_history(
        self,
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.history.visual_enabled:
            raise RuntimeError("DINOv3 visual history is not enabled")
        pixel_values = self._mapping_tensor(samples, "dinov3_history")
        expected = (
            self.config.history.length,
            self.num_views,
            3,
            self.config.vision.image_size,
            self.config.vision.image_size,
        )
        if pixel_values.ndim != 6 or tuple(pixel_values.shape[1:]) != expected:
            raise ValueError(
                f"dinov3_history must be [B,{','.join(str(v) for v in expected)}], "
                f"got {tuple(pixel_values.shape)}"
            )
        history_mask = history_mask.to(device=pixel_values.device, dtype=torch.bool)
        if history_mask.shape != pixel_values.shape[:2]:
            raise ValueError(
                f"history_mask must be {tuple(pixel_values.shape[:2])}, "
                f"got {tuple(history_mask.shape)}"
            )

        batch_size, history_length = pixel_values.shape[:2]
        flat_frames = pixel_values.flatten(0, 1)
        valid_indices = history_mask.flatten().nonzero(as_tuple=False).flatten()
        if valid_indices.numel():
            valid_frames = flat_frames.index_select(0, valid_indices)
            encoded_chunks = []
            chunk_size = self.config.history.visual_encode_chunk_size
            grad_context = nullcontext() if self.config.history.visual_backprop else torch.no_grad()
            with grad_context:
                for start in range(0, valid_frames.shape[0], chunk_size):
                    patch_tokens = self.vision_encoder(valid_frames[start : start + chunk_size])
                    encoded_chunks.append(patch_tokens.mean(dim=2))
            valid_embeddings = torch.cat(encoded_chunks, dim=0)
            valid_embeddings = valid_embeddings.to(dtype=self.vision_projection.skip.weight.dtype)
            all_embeddings = valid_embeddings.new_zeros(
                batch_size * history_length,
                self.num_views,
                self.vision_encoder.hidden_size,
            )
            all_embeddings = all_embeddings.index_copy(0, valid_indices, valid_embeddings)
        else:
            all_embeddings = self.vision_projection.skip.weight.new_zeros(
                batch_size * history_length,
                self.num_views,
                self.vision_encoder.hidden_size,
            )
        all_embeddings = all_embeddings.view(
            batch_size,
            history_length,
            self.num_views,
            self.vision_encoder.hidden_size,
        )
        tokens = self.vision_projection(all_embeddings)
        view = self.history_visual_view_embedding[:, None].to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        time = self.history_visual_time_embedding.to(device=tokens.device, dtype=tokens.dtype)
        tokens = tokens + view + time
        token_mask = history_mask[:, :, None].expand(-1, -1, self.num_views)
        tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        return tokens.flatten(1, 2), token_mask.flatten(1, 2)

    def _encode_condition_with_mask(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        cached_visual_tokens = (
            samples.get("dinov3_tokens") if isinstance(samples, Mapping) else None
        )
        if cached_visual_tokens is not None:
            if cached_visual_tokens.ndim != 3:
                raise ValueError(
                    "dinov3_tokens must be [B,N,D], got "
                    f"{tuple(cached_visual_tokens.shape)}"
                )
            device = cached_visual_tokens.device
            batch_size = cached_visual_tokens.shape[0]
            pixel_values = None
        else:
            pixel_values = self._normalize_samples(samples)
            device = pixel_values.device
            batch_size = pixel_values.shape[0]
        precision_context = nullcontext()
        if self.config.interaction.compute_precision == "bf16_autocast" and device.type == "cuda":
            precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with precision_context:
            text_tokens, text_key_padding_mask, text_self_attention_masks = self.text_encoder(
                instructions,
                device=device,
            )
            if text_tokens.shape[0] != batch_size:
                raise ValueError("instruction batch size does not match image batch size")
            visual_tokens = (
                cached_visual_tokens
                if cached_visual_tokens is not None
                else self.encode_vision(pixel_values)
            )
            current_r3m_tokens = None
            current_r3m_embeddings = None
            if self.r3m_encoder is not None:
                current_r3m_tokens, current_r3m_embeddings = self._encode_r3m_current_outputs(
                    samples
                )
                current_r3m_tokens = current_r3m_tokens.to(dtype=visual_tokens.dtype)
                visual_tokens = torch.cat([visual_tokens, current_r3m_tokens], dim=1)
            visual_tokens, text_tokens = self.vision_language_interaction(
                visual_tokens=visual_tokens,
                text_tokens=text_tokens,
                text_key_padding_mask=text_key_padding_mask,
                text_self_attention_masks=text_self_attention_masks,
            )
            return (
                torch.cat([visual_tokens, text_tokens], dim=1),
                text_key_padding_mask,
                current_r3m_tokens,
                current_r3m_embeddings,
            )

    def encode_condition(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        condition, _, _, _ = self._encode_condition_with_mask(instructions, samples)
        return condition

    def forward(
        self,
        instructions: Sequence[str],
        samples: torch.Tensor | Mapping[str, torch.Tensor],
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
        history_states: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        ablate_history_r3m_tokens: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        encoded_condition = self._encode_condition_with_mask(instructions, samples)
        if len(encoded_condition) == 3:
            condition, text_key_padding_mask, current_r3m_tokens = encoded_condition
            current_r3m_embeddings = None
        else:
            (
                condition,
                text_key_padding_mask,
                current_r3m_tokens,
                current_r3m_embeddings,
            ) = encoded_condition
        if self.action_head_type == "flow_matching":
            state = state.to(device=condition.device, dtype=condition.dtype)
            use_direct_flow_state = (
                int(getattr(self, "flow_state_dim", 0)) > 0
                and int(getattr(self, "flow_state_dim", 0)) == int(getattr(self, "state_dim", state.shape[-1]))
            )
            use_state_tokens = not use_direct_flow_state or getattr(self, "flow_state_encoding", None) == "state_tokens_zero_pad"
            state_tokens = self.state_proj_module(state) if use_state_tokens else None
            text_len = text_key_padding_mask.shape[1]
            visual_len = condition.shape[1] - text_len
            visual_padding = torch.zeros(
                condition.shape[0],
                visual_len,
                dtype=torch.bool,
                device=condition.device,
            )
            state_padding = torch.zeros(
                state_tokens.shape[0], state_tokens.shape[1], dtype=torch.bool,
                device=condition.device,
            ) if use_state_tokens else None
            valid_text = ~text_key_padding_mask
            text_condition = condition[:, -text_len:]
            goal_summary = (
                text_condition * valid_text.unsqueeze(-1).to(dtype=condition.dtype)
            ).sum(dim=1)
            goal_summary = goal_summary / valid_text.sum(dim=1, keepdim=True).clamp_min(1)
            auxiliary_losses: dict[str, torch.Tensor] = {}
            if self.history_encoder is not None:
                if history_states is None or history_mask is None:
                    raise ValueError("history_states and history_mask are required when history is enabled")
                history_mask = history_mask.to(device=condition.device, dtype=torch.bool)
                history_tokens = self.history_encoder(
                    history_states.to(device=condition.device, dtype=condition.dtype),
                    history_mask,
                )
                extra_tokens = []
                extra_padding = []
                if self.config.history.r3m_enabled and not ablate_history_r3m_tokens:
                    predictive_belief = bool(
                        getattr(self.config.history, "r3m_predictive_belief", False)
                    )
                    intentional_memory = bool(
                        getattr(self.config.history, "r3m_intentional_memory", False)
                    )
                    if current_r3m_tokens is None:
                        raise RuntimeError("R3M history requires current R3M outputs")
                    if predictive_belief or intentional_memory:
                        if not isinstance(samples, Mapping):
                            raise TypeError("predictive R3M history requires mapped samples")
                    if intentional_memory:
                        (
                            history_r3m_tokens,
                            history_r3m_mask,
                            auxiliary_losses,
                        ) = self.encode_r3m_intentional_memory(
                            samples,
                            current_r3m_tokens,
                            history_mask,
                            history_tokens,
                            goal_summary,
                        )
                    elif predictive_belief:
                        (
                            history_r3m_tokens,
                            history_r3m_mask,
                            auxiliary_losses,
                        ) = self.encode_r3m_predictive_belief(
                            samples,
                            current_r3m_tokens,
                            history_mask,
                            history_tokens,
                            goal_summary,
                        )
                    else:
                        history_r3m_tokens, history_r3m_mask = self.encode_r3m_history(
                            samples,
                            current_r3m_tokens,
                            history_mask,
                            history_tokens,
                        )
                    extra_tokens.append(history_r3m_tokens.to(dtype=condition.dtype))
                    extra_padding.append(~history_r3m_mask)
                if self.config.history.visual_enabled:
                    history_visual_tokens, history_visual_mask = self.encode_dinov3_history(
                        samples,
                        history_mask,
                    )
                    extra_tokens.append(history_visual_tokens.to(dtype=condition.dtype))
                    extra_padding.append(~history_visual_mask)
                if (
                    not bool(getattr(self.config.history, "r3m_predictive_belief", False))
                    or bool(getattr(self.config.history, "r3m_tacit_belief_fusion", False))
                ):
                    extra_tokens.append(history_tokens)
                    extra_padding.append(~history_mask)
                condition = torch.cat(
                    [condition, *extra_tokens] + ([state_tokens] if use_state_tokens else []), dim=1
                )
            else:
                if use_state_tokens:
                    condition = torch.cat([condition, state_tokens], dim=1)
                extra_padding = []
            condition_padding_mask = torch.cat(
                [visual_padding, text_key_padding_mask, *extra_padding]
                + ([state_padding] if use_state_tokens else []),
                dim=1,
            )
            outputs = self.flow_action_policy(
                condition=condition,
                condition_padding_mask=condition_padding_mask,
                states=state if use_direct_flow_state else None,
                actions=actions,
                action_masks=action_masks,
            )
            if auxiliary_losses and isinstance(outputs, dict) and "loss" in outputs:
                outputs = dict(outputs)
                outputs.update(auxiliary_losses)
            return outputs

        action_dtype = self.action_head.decoder.action_queries.weight.dtype
        return self.action_head(condition.to(dtype=action_dtype), state.to(dtype=action_dtype))

    # Transitional read-only names used only by legacy checkpoint initialization.
    @property
    def dinov3(self):
        return self.vision_encoder.backbone

    @property
    def text_proj(self):
        return self.text_encoder.text_projection

    @property
    def vision_proj(self):
        return self.vision_projection

    @property
    def feature_enhancer(self):
        return self.vision_language_interaction

    @property
    def state_proj(self):
        if self.action_head_type == "flow_matching":
            return getattr(self, "state_proj_module", None)
        return self.action_head.state_projection

    @property
    def action_policy(self):
        if self.action_head_type == "flow_matching":
            return self.flow_action_policy
        return self.action_head.decoder


def _arg(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _integer_tuple_arg(args: Any, name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = _arg(args, name, default)
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def build_turbovla(args: TurboVLAConfig | Mapping[str, Any] | Any) -> TurboVLA:
    if isinstance(args, TurboVLAConfig):
        config = args
    elif isinstance(args, Mapping):
        config = TurboVLAConfig.from_mapping(args)
    else:
        config = TurboVLAConfig(
            text=TextEncoderConfig(
                model_name_or_path=_arg(args, "bert_path", "bert-base-uncased"),
                max_length=int(_arg(args, "max_text_len", 256)),
                padding_length=_arg(args, "text_padding_length", None),
                padding_length_by_instruction=dict(_arg(args, "text_padding_length_by_instruction", {})),
                sub_sentence_present=bool(_arg(args, "sub_sentence_present", True)),
                frozen=bool(_arg(args, "freeze_text_encoder", True)),
                force_eval_when_frozen=True,
                zero_padded_tokens=bool(_arg(args, "zero_padded_text", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "text_attention_implementation", None),
                frozen_training_cache=bool(_arg(args, "frozen_text_cache", True)),
            ),
            vision=VisionEncoderConfig(
                model_name_or_path=_arg(args, "dinov3_path", _arg(args, "LOCAL_DINOV3_PATH", "")),
                image_size=int(_arg(args, "image_size", _arg(args, "expected_image_size", 256))),
                num_views=int(_arg(args, "num_views", 2)),
                position_embedding=str(_arg(args, "position_embedding", "view")),
                encode_views_separately=bool(_arg(args, "encode_views_separately", True)),
                frozen=bool(_arg(args, "freeze_vision_encoder", False)),
                local_files_only=bool(_arg(args, "local_files_only", True)),
                attention_implementation=_arg(args, "vision_attention_implementation", None),
                compute_precision=str(_arg(args, "dinov3_precision", "bf16_autocast")),
                dropout=float(_arg(args, "vision_dropout", 0.1)),
            ),
            r3m=R3MEncoderConfig(
                enabled=bool(_arg(args, "use_r3m", False)),
                model_name=str(_arg(args, "r3m_model", "resnet18")),
                checkpoint_path=str(_arg(args, "r3m_path", "")),
                image_size=int(_arg(args, "r3m_image_size", 224)),
                output_dim=int(_arg(args, "r3m_output_dim", 512)),
                frozen=bool(_arg(args, "freeze_r3m", True)),
                num_views=int(_arg(args, "num_views", 2)),
                encode_chunk_size=int(_arg(args, "r3m_encode_chunk_size", 32)),
                dropout=float(_arg(args, "r3m_dropout", 0.0)),
            ),
            interaction=InteractionConfig(
                hidden_dim=int(_arg(args, "hidden_dim", 256)),
                nheads=int(_arg(args, "nheads", 8)),
                num_layers=int(_arg(args, "vla_feature_enhancer_layers", 6)),
                dim_feedforward=int(_arg(args, "dim_feedforward", 2048)),
                enhancer_inner_dim=int(_arg(args, "enhancer_inner_dim", 1024)),
                text_dropout=float(_arg(args, "text_dropout", 0.0)),
                fusion_dropout=float(_arg(args, "fusion_dropout", 0.0)),
                fusion_droppath=float(_arg(args, "fusion_droppath", 0.1)),
                padding_strategy=str(_arg(args, "padding_strategy", "key_padding_mask")),
                residual_style=str(_arg(args, "residual_style", "normalized")),
                attention_backend=str(_arg(args, "attention_backend", "manual")),
                compute_precision=str(_arg(args, "interaction_precision", "fp32")),
            ),
            action=ActionHeadConfig(
                action_dim=int(_arg(args, "action_dim", 7)),
                state_dim=int(_arg(args, "state_dim", 8)),
                horizon=int(_arg(args, "chunk_size", _arg(args, "action_horizon", 12))),
                num_state_tokens=int(_arg(args, "num_state_tokens", 2)),
                state_hidden_dim=int(_arg(args, "act_state_hidden_dim", 256)),
                dropout=float(_arg(args, "act_dropout", 0.1)),
            ),
            history=HistoryConfig(
                enabled=int(_arg(args, "history_length", 0)) > 0,
                length=int(_arg(args, "history_length", 4)),
                state_dim=int(_arg(args, "state_dim", 8)),
                hidden_dim=int(_arg(args, "history_hidden_dim", _arg(args, "hidden_dim", 256))),
                num_layers=int(_arg(args, "history_layers", 2)), dropout=float(_arg(args, "history_dropout", 0.0)),
                encoder_type=str(_arg(args, "history_encoder", "mamba")),
                r3m_enabled=bool(_arg(args, "history_r3m", False)),
                r3m_rope_base=float(_arg(args, "history_r3m_rope_base", 10000.0)),
                r3m_memory_num_queries=int(
                    _arg(args, "history_r3m_memory_num_queries", 2)
                ),
                r3m_memory_num_heads=int(_arg(args, "history_r3m_memory_num_heads", 4)),
                r3m_memory_dropout=float(_arg(args, "history_r3m_memory_dropout", 0.1)),
                r3m_memory_gate_init=float(_arg(args, "history_r3m_memory_gate_init", 0.1)),
                r3m_belief_num_slots=int(
                    _arg(args, "history_r3m_belief_num_slots", 4)
                ),
                r3m_tacit_belief_gate_init=float(
                    _arg(args, "history_r3m_tacit_belief_gate_init", 0.1)
                ),
            ),
        )
    return TurboVLA(
        config,
        action_head=str(_arg(args, "action_head", "act")),
        flow_num_heads=int(_arg(args, "flow_num_heads", 4)),
        flow_condition_layers=int(_arg(args, "flow_condition_layers", 4)),
        flow_dit_layers=int(_arg(args, "flow_dit_layers", 16)),
        flow_target_tokens=int(_arg(args, "flow_target_tokens", 16)),
        flow_static_tokens=int(_arg(args, "flow_static_tokens", 8)),
        flow_state_dim=int(_arg(args, "flow_state_dim", 0)),
        flow_bijection_blocks=int(_arg(args, "flow_bijection_blocks", 6)),
        flow_state_encoding=str(_arg(args, "flow_state_encoding", "native_mlp")),
    )
