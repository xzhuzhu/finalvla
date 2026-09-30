from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar, Mapping


@dataclass
class TextEncoderConfig:
    model_name_or_path: str = "bert-base-uncased"
    max_length: int = 256
    padding_length: int | None = None
    padding_length_by_instruction: dict[str, int] = field(default_factory=dict)
    sub_sentence_present: bool = True
    frozen: bool = True
    force_eval_when_frozen: bool = True
    zero_padded_tokens: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None
    frozen_training_cache: bool = True


@dataclass
class VisionEncoderConfig:
    model_name_or_path: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    image_size: int = 256
    num_views: int = 2
    position_embedding: str = "view"
    encode_views_separately: bool = True
    frozen: bool = False
    local_files_only: bool = True
    attention_implementation: str | None = None
    compute_precision: str = "bf16_autocast"
    position_init_std: float = 0.01
    position_scale_init: float = 0.01
    dropout: float = 0.1


@dataclass
class R3MEncoderConfig:
    enabled: bool = False
    model_name: str = "resnet18"
    checkpoint_path: str = ""
    image_size: int = 224
    output_dim: int = 512
    frozen: bool = True
    num_views: int = 2
    encode_chunk_size: int = 32
    dropout: float = 0.0


@dataclass
class InteractionConfig:
    hidden_dim: int = 256
    nheads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    enhancer_inner_dim: int = 1024
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    padding_strategy: str = "key_padding_mask"
    residual_style: str = "normalized"
    attention_backend: str = "manual"
    compute_precision: str = "fp32"


@dataclass
class ActionHeadConfig:
    action_dim: int = 7
    state_dim: int = 8
    horizon: int = 12
    num_state_tokens: int = 2
    num_layers: ClassVar[int] = 3
    mlp_hidden_dim: ClassVar[int] = 512
    state_hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class HistoryConfig:
    enabled: bool = False
    length: int = 4
    state_dim: int = 8
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    encoder_type: str = "mamba"
    visual_enabled: ClassVar[bool] = False
    visual_encoder: ClassVar[str] = "dinov3"
    visual_encode_chunk_size: ClassVar[int] = 32
    visual_backprop: ClassVar[bool] = False
    r3m_enabled: bool = False
    r3m_rope_base: float = 10000.0
    r3m_memory_num_queries: int = 2
    r3m_memory_num_heads: int = 4
    r3m_memory_dropout: float = 0.1
    r3m_memory_gate_init: float = 0.1
    r3m_memory_dynamic_gate: ClassVar[bool] = False
    r3m_memory_residual_gate: ClassVar[bool] = False
    r3m_memory_token_dropout: ClassVar[float] = 0.0
    r3m_memory_rope_values: ClassVar[bool] = False
    r3m_intentional_memory: ClassVar[bool] = False
    r3m_intentional_operator_rank: ClassVar[int] = 64
    r3m_belief_num_slots: int = 4
    r3m_belief_future_horizons: ClassVar[tuple[int, ...]] = (1, 4, 8, 12)

    @property
    def r3m_predictive_belief(self) -> bool:
        return self.r3m_enabled

    @property
    def r3m_controlled_causal_belief(self) -> bool:
        return self.r3m_enabled

    @property
    def r3m_tacit_belief_fusion(self) -> bool:
        return self.r3m_enabled

    r3m_tacit_belief_gate_init: float = 0.1


@dataclass
class TurboVLAConfig:
    name: str = "TurboVLA"
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    r3m: R3MEncoderConfig = field(default_factory=R3MEncoderConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    action: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)

    def __post_init__(self) -> None:
        if self.name != "TurboVLA":
            raise ValueError(f"model name must be 'TurboVLA', got {self.name!r}")
        if self.vision.num_views < 1:
            raise ValueError("vision.num_views must be positive")
        if self.vision.position_embedding not in {"view", "learned_patch"}:
            raise ValueError("vision.position_embedding must be 'view' or 'learned_patch'")
        if self.vision.compute_precision not in {"fp32", "bf16", "bf16_autocast"}:
            raise ValueError("vision.compute_precision must be fp32, bf16, or bf16_autocast")
        if self.r3m.enabled:
            if self.r3m.model_name != "resnet18":
                raise ValueError("r3m.model_name must be 'resnet18'")
            if not self.r3m.checkpoint_path:
                raise ValueError("r3m.checkpoint_path is required when R3M is enabled")
            if self.r3m.image_size != 224 or self.r3m.output_dim != 512:
                raise ValueError("R3M ResNet-18 requires image_size=224 and output_dim=512")
            if self.r3m.num_views != self.vision.num_views:
                raise ValueError("R3M and DINOv3 must use the same number of camera views")
            if self.r3m.encode_chunk_size < 1:
                raise ValueError("r3m.encode_chunk_size must be positive")
        if self.text.padding_length is not None:
            if self.text.padding_length < 1:
                raise ValueError("text.padding_length must be positive")
            if self.text.padding_length > self.text.max_length:
                raise ValueError("text.padding_length cannot exceed text.max_length")
        for instruction, length in self.text.padding_length_by_instruction.items():
            if not instruction:
                raise ValueError("text.padding_length_by_instruction cannot contain an empty instruction")
            if length < 1 or length > self.text.max_length:
                raise ValueError(f"invalid text padding length {length} for instruction {instruction!r}")
        if self.interaction.hidden_dim % self.interaction.nheads != 0:
            raise ValueError("interaction.hidden_dim must be divisible by interaction.nheads")
        if self.interaction.padding_strategy not in {"key_padding_mask", "zero_fill"}:
            raise ValueError("interaction.padding_strategy must be key_padding_mask or zero_fill")
        if self.interaction.residual_style not in {"normalized", "pre_norm"}:
            raise ValueError("interaction.residual_style must be normalized or pre_norm")
        if self.interaction.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("interaction.attention_backend must be manual or sdpa")
        if self.interaction.compute_precision not in {"fp32", "bf16_autocast"}:
            raise ValueError("interaction.compute_precision must be fp32 or bf16_autocast")
        if self.action.action_dim < 1 or self.action.state_dim < 1 or self.action.horizon < 1:
            raise ValueError("action dimensions and horizon must be positive")
        if self.history.encoder_type != "mamba":
            raise ValueError("history.encoder_type must be 'mamba'")
        if self.history.visual_encoder != "dinov3":
            raise ValueError("history.visual_encoder must be 'dinov3'")
        if self.history.visual_enabled and not self.history.enabled:
            raise ValueError("visual history requires state history to be enabled")
        if self.history.r3m_enabled and not self.history.enabled:
            raise ValueError("R3M history requires state history to be enabled")
        if self.history.r3m_enabled and not self.r3m.enabled:
            raise ValueError("R3M history requires the current R3M encoder to be enabled")
        if self.history.r3m_enabled and self.history.length != 12:
            raise ValueError("R3M history requires exactly 12 history steps")
        if self.history.r3m_memory_num_queries < 1:
            raise ValueError("history.r3m_memory_num_queries must be positive")
        if self.history.r3m_memory_num_heads < 1:
            raise ValueError("history.r3m_memory_num_heads must be positive")
        if self.interaction.hidden_dim % self.history.r3m_memory_num_heads:
            raise ValueError(
                "interaction.hidden_dim must be divisible by history.r3m_memory_num_heads"
            )
        if not 0.0 <= self.history.r3m_memory_dropout < 1.0:
            raise ValueError("history.r3m_memory_dropout must be in [0, 1)")
        if not 0.0 < self.history.r3m_memory_gate_init < 1.0:
            raise ValueError("history.r3m_memory_gate_init must be in (0, 1)")
        if not 0.0 <= self.history.r3m_memory_token_dropout < 1.0:
            raise ValueError("history.r3m_memory_token_dropout must be in [0, 1)")
        if self.history.r3m_memory_dynamic_gate and not self.history.r3m_enabled:
            raise ValueError("dynamic R3M memory gating requires R3M history")
        if self.history.r3m_memory_residual_gate and not self.history.r3m_enabled:
            raise ValueError("residual R3M memory gating requires R3M history")
        if (
            self.history.r3m_memory_dynamic_gate
            and self.history.r3m_memory_residual_gate
        ):
            raise ValueError("dynamic and residual R3M memory gating are mutually exclusive")
        if self.history.r3m_memory_token_dropout and not self.history.r3m_enabled:
            raise ValueError("R3M memory token dropout requires R3M history")
        if self.history.r3m_intentional_memory and not self.history.r3m_enabled:
            raise ValueError("intentional R3M memory requires R3M history")
        if self.history.r3m_intentional_operator_rank < 1:
            raise ValueError("history.r3m_intentional_operator_rank must be positive")
        if self.history.r3m_intentional_operator_rank > self.interaction.hidden_dim:
            raise ValueError(
                "history.r3m_intentional_operator_rank cannot exceed interaction.hidden_dim"
            )
        if (
            self.history.r3m_intentional_memory
            and not self.history.r3m_memory_rope_values
        ):
            raise ValueError("intentional R3M memory requires full key/value RoPE")
        if (
            self.history.r3m_intentional_memory
            and self.history.r3m_predictive_belief
        ):
            raise ValueError(
                "intentional R3M memory and predictive belief are mutually exclusive"
            )
        if self.history.r3m_intentional_memory and (
            self.history.r3m_memory_dynamic_gate
            or self.history.r3m_memory_residual_gate
            or self.history.r3m_memory_token_dropout
            or self.history.r3m_tacit_belief_fusion
        ):
            raise ValueError(
                "intentional R3M memory is a single ungated path and cannot use "
                "memory gates, token dropout, or tacit-belief fusion"
            )
        if self.history.r3m_predictive_belief and not self.history.r3m_enabled:
            raise ValueError("predictive R3M belief requires R3M history")
        if self.history.r3m_predictive_belief and self.history.r3m_memory_rope_values:
            raise ValueError("predictive R3M belief requires key-only semantic RoPE")
        if (
            self.history.r3m_tacit_belief_fusion
            and not self.history.r3m_predictive_belief
        ):
            raise ValueError("tacit-belief fusion requires predictive R3M belief")
        if self.history.r3m_controlled_causal_belief:
            if not self.history.r3m_predictive_belief:
                raise ValueError("controlled causal belief requires predictive R3M belief")
            if not self.history.r3m_tacit_belief_fusion:
                raise ValueError("controlled causal belief requires tacit-belief fusion")
            if self.history.r3m_intentional_memory:
                raise ValueError(
                    "controlled causal belief and intentional R3M memory are mutually exclusive"
                )
        if not 0.0 < self.history.r3m_tacit_belief_gate_init < 1.0:
            raise ValueError("history.r3m_tacit_belief_gate_init must be in (0, 1)")
        if self.history.r3m_belief_num_slots < 1:
            raise ValueError("history.r3m_belief_num_slots must be positive")
        horizons = tuple(self.history.r3m_belief_future_horizons)
        if not horizons or any(int(value) < 1 for value in horizons):
            raise ValueError("history.r3m_belief_future_horizons must be positive")
        if tuple(sorted(set(int(value) for value in horizons))) != horizons:
            raise ValueError(
                "history.r3m_belief_future_horizons must be sorted and unique"
            )
        if self.history.r3m_rope_base <= 0:
            raise ValueError("history.r3m_rope_base must be positive")
        if self.history.visual_encode_chunk_size < 1:
            raise ValueError("history.visual_encode_chunk_size must be positive")
        if self.history.length < 1 or self.history.state_dim < 1:
            raise ValueError("history dimensions and length must be positive")
        if self.history.enabled and (
            self.history.length not in {4, 12}
            or self.history.state_dim != 8
            or self.history.hidden_dim != 256
            or self.history.num_layers != 2
            or self.history.dropout != 0.0
        ):
            raise ValueError(
                "history encoder requires H in {4, 12}, state_dim=8, "
                "hidden_dim=256, layers=2, dropout=0"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TurboVLAConfig":
        data = dict(payload)
        history_data = dict(data.get("history", {}))
        # Accept checkpoints from the original training run while exposing only
        # parameters that affect the fixed FinalVLA architecture.
        history_data = {k: v for k, v in history_data.items()
                        if k in {f.name for f in fields(HistoryConfig)}}
        action_data = {k: v for k, v in dict(data.get("action", {})).items()
                       if k in {f.name for f in fields(ActionHeadConfig)}}
        return cls(
            name=str(data.get("name", "TurboVLA")),
            text=TextEncoderConfig(**dict(data.get("text", {}))),
            vision=VisionEncoderConfig(**dict(data.get("vision", {}))),
            r3m=R3MEncoderConfig(**dict(data.get("r3m", {}))),
            interaction=InteractionConfig(**dict(data.get("interaction", {}))),
            action=ActionHeadConfig(**action_data),
            history=HistoryConfig(**history_data),
        )
