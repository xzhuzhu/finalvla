"""Adapter for the vendored MyVLA ``FlowMatching7DHead``.

The two required MyVLA source files live under ``turbovla/_vendor/myvla``.
MyVLA pins Transformers 5.3 while this TurboVLA environment intentionally uses
Transformers 4.56, so importing the complete ``fluxvla`` package normally
fails before the head is reached.  This module creates only the small import
surface required by those source files, then loads them unchanged.  Set
``TURBOVLA_MYVLA_ROOT`` only when intentionally testing another MyVLA checkout.

The adapter can either retain MyVLA's single state-token pathway or disable it
for state-free ablations.  State-conditioned future queries remain disabled:
this keeps the experiment to exactly one proprioceptive token.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F


_VENDORED_MYVLA_ROOT = Path(__file__).resolve().parents[2] / "_vendor/myvla"
_MYVLA_ROOT = Path(
    os.environ.get("TURBOVLA_MYVLA_ROOT", str(_VENDORED_MYVLA_ROOT))
).expanduser().resolve()
_MYVLA_BLOCK_PATH = _MYVLA_ROOT / "fluxvla/models/blocks/cross_attention_dit.py"
_MYVLA_HEAD_PATH = _MYVLA_ROOT / "fluxvla/models/heads/flow_matching_7d_head.py"


class _HeadRegistry:
    """Minimal registry needed by MyVLA's registration decorator."""

    def register_module(self):
        return lambda cls: cls


def _reduce_action_bc_loss(losses: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
    if action_mask is None:
        return losses.mean()
    valid = action_mask.to(device=losses.device, dtype=losses.dtype)
    while valid.ndim < losses.ndim:
        valid = valid.unsqueeze(-1)
    return (losses * valid.expand_as(losses)).sum() / valid.expand_as(losses).sum().clamp_min(1e-8)


def _install_module(name: str, module: types.ModuleType) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    sys.modules[name] = module
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return _install_module(name, module)


def _load_source(name: str, path: Path) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    if not path.is_file():
        raise FileNotFoundError(f"MyVLA source file is required but missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load MyVLA source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_myvla_head_classes():
    # Avoid importing fluxvla.__init__, which enforces MyVLA's incompatible
    # Transformers pin.  The native action head only needs this narrow set of
    # package-level objects.
    _package("fluxvla")
    engines = _package("fluxvla.engines")
    engines.HEADS = _HeadRegistry()
    losses = _package("fluxvla.engines.losses")
    losses.reduce_action_bc_loss = _reduce_action_bc_loss
    utils = _package("fluxvla.engines.utils")
    overwatch = _package("fluxvla.engines.utils.overwatch")
    overwatch.initialize_overwatch = lambda _name: None
    utils.overwatch = overwatch

    _package("fluxvla.models")
    blocks = _package("fluxvla.models.blocks")
    block_module = _load_source("fluxvla.models.blocks.cross_attention_dit", _MYVLA_BLOCK_PATH)
    blocks.SelfAttentionTransformer = block_module.SelfAttentionTransformer
    _package("fluxvla.models.heads")
    head_module = _load_source("fluxvla.models.heads.flow_matching_7d_head", _MYVLA_HEAD_PATH)
    return head_module.FlowMatching7DHead, head_module.BijectionNet


_MyVLAFlowMatching7DHead, _MyVLABijectionNet = _load_myvla_head_classes()


class LinearExpansionBijection(nn.Module):
    """Injectively expand actions before a latent-space bijection.

    A reduced QR factorization constructs a trainable semi-orthogonal 35x7
    expansion Q.  The down-projection is Q^T, so Q^T Q = I by construction
    throughout training instead of relying on a separately learned decoder or
    a runtime matrix inverse.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_blocks: int = 6,
        num_hidden: int = 256,
    ) -> None:
        super().__init__()
        if latent_dim <= input_dim:
            raise ValueError(
                f"latent_dim={latent_dim} must be greater than input_dim={input_dim}"
            )
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_blocks = int(num_blocks)
        # ``expansion.weight`` is the unconstrained trainable seed.  The actual
        # linear map is its reduced-QR Q factor, whose columns are orthonormal.
        self.expansion = nn.Linear(self.input_dim, self.latent_dim, bias=False)
        nn.init.orthogonal_(self.expansion.weight)
        self.bijection = _MyVLABijectionNet(
            self.latent_dim,
            self.num_blocks,
            int(num_hidden),
        )

    def expansion_weight(self) -> torch.Tensor:
        # QR runs in FP32 because CUDA linalg does not support BF16 reliably.
        weight, _ = torch.linalg.qr(self.expansion.weight.float(), mode="reduced")
        return weight

    def inverse_expansion_weight(self) -> torch.Tensor:
        return self.expansion_weight().transpose(0, 1)

    @staticmethod
    def _linear_fp32(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Keep construction and projection in FP32 even under AMP, then return
        # the original activation dtype for the surrounding reversible blocks.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            output = F.linear(inputs.float(), weight)
        return output.to(dtype=inputs.dtype)

    def _inverse_expansion(self, latent: torch.Tensor) -> torch.Tensor:
        left_inverse = self.inverse_expansion_weight()
        return self._linear_fp32(latent, left_inverse)

    def forward(self, inputs: torch.Tensor, mode: str = "direct") -> torch.Tensor:
        if mode == "direct":
            expanded = self._linear_fp32(inputs, self.expansion_weight())
            return self.bijection(expanded, mode="direct")
        if mode == "inverse":
            expanded = self.bijection(inputs, mode="inverse")
            return self._inverse_expansion(expanded)
        raise ValueError(f"mode must be 'direct' or 'inverse', got {mode!r}")


class MyVLAFlowMatchingActionHead(_MyVLAFlowMatching7DHead):
    """MyVLA flow head with optional native state conditioning and QR flow."""

    def __init__(
        self,
        hidden_size: int = 256,
        action_dim: int = 7,
        chunk_size: int = 9,
        num_heads: int = 4,
        condition_layers: int = 4,
        dit_layers: int = 16,
        num_target_vision_tokens: int = 16,
        num_static_future_tokens: int = 8,
        state_dim: int = 0,
        bijection_blocks: int = 6,
        state_encoding: str = "native_mlp",
    ) -> None:
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        head_dim = hidden_size // num_heads
        self.state_dim = int(state_dim)
        self.state_encoding = str(state_encoding)
        if self.state_dim < 0:
            raise ValueError(f"state_dim must be non-negative, got {self.state_dim}")
        if self.state_encoding not in {"native_mlp", "zero_pad"}:
            raise ValueError(f"unsupported state_encoding={self.state_encoding!r}")
        if self.state_encoding == "zero_pad" and self.state_dim <= 0:
            raise ValueError("zero_pad state encoding requires a positive state_dim")
        if self.state_encoding == "zero_pad" and self.state_dim > hidden_size:
            raise ValueError(
                f"zero_pad state_dim={self.state_dim} exceeds hidden_size={hidden_size}"
            )
        if int(bijection_blocks) < 1:
            raise ValueError("bijection_blocks must be positive")
        self.bijection_blocks = int(bijection_blocks)
        use_state_condition = self.state_dim > 0

        super().__init__(
            hidden_size=hidden_size,
            state_dim=max(self.state_dim, 1),
            input_embedding_dim=hidden_size,
            action_dim=action_dim,
            use_vlln=True,
            num_target_vision_tokens=num_target_vision_tokens,
            num_static_future_tokens=num_static_future_tokens,
            num_state_future_tokens=0,
            static_future_token_dropout=0.0,
            backbone_embedding_dim=hidden_size,
            vl_self_attention_cfg=dict(
                attention_head_dim=head_dim,
                dropout=0.0,
                final_dropout=False,
                num_attention_heads=num_heads,
                num_layers=condition_layers,
                positional_embeddings=None,
            ),
            add_positional_embeddings=True,
            max_seq_len=max(1024, chunk_size),
            num_timestep_buckets=1000,
            noise_s=0.999,
            noise_beta_alpha=1.5,
            noise_beta_beta=1.0,
            num_steps=chunk_size,
            diffusion_model_cfg=dict(
                attention_head_dim=head_dim,
                cross_attention_dim=hidden_size,
                dropout=0.0,
                final_dropout=False,
                interleave_self_attention=True,
                norm_type="ada_norm",
                num_attention_heads=num_heads,
                num_layers=dit_layers,
                output_dim=hidden_size,
                positional_embeddings=None,
            ),
            use_state_condition=use_state_condition,
        )
        # Shared by both training and inference:
        # 7D action -> 35D linear expansion -> configurable coupling blocks ->
        # F.normalize -> inverse coupling blocks -> inverse expansion -> 7D.
        self.flow = LinearExpansionBijection(
            input_dim=self.action_dim,
            latent_dim=35,
            num_blocks=self.bijection_blocks,
            num_hidden=hidden_size,
        )
        # MyVLA creates this module unconditionally even when state
        # conditioning is off.  Remove it so state-free ablations contain no
        # unused state-token parameters.
        if not use_state_condition or self.state_encoding == "zero_pad":
            del self.state_encoder

    def _encode_state_condition(self, states: torch.Tensor | None) -> torch.Tensor | None:
        """Use the native MLP or the canonical parameter-free 8D zero pad."""
        if self.state_encoding != "zero_pad":
            return super()._encode_state_condition(states)
        if states is None:
            raise ValueError("states must be provided when state_encoding='zero_pad'")
        if states.ndim != 2 or states.shape[-1] != self.state_dim:
            raise ValueError(
                f"states must be [B, {self.state_dim}], got {tuple(states.shape)}"
            )
        return F.pad(states, (0, self.hidden_size - self.state_dim)).unsqueeze(1)

    @staticmethod
    def _valid_attention_mask(
        condition_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if condition_padding_mask is None:
            return None
        return ~condition_padding_mask.to(dtype=torch.bool)

    def forward(
        self,
        condition: torch.Tensor,
        condition_padding_mask: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
    ):
        attention_mask = self._valid_attention_mask(condition_padding_mask)
        if self.use_state_condition:
            if states is None:
                raise ValueError("states are required when state_dim is positive")
            if states.ndim != 2 or states.shape[-1] != self.state_dim:
                raise ValueError(
                    f"states must be [B, {self.state_dim}], got {tuple(states.shape)}"
                )
        if actions is None:
            return self.predict_action(
                input_features=condition,
                states=states,
                attention_mask=attention_mask,
            )
        return super().forward(
            input_features=condition,
            states=states,
            attention_mask=attention_mask,
            actions=actions,
            action_masks=action_masks,
        )
