from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from turbovla.models.components.myvla_action_head import MyVLAFlowMatchingActionHead
from turbovla.models.flow_checkpoint import (
    assert_flow_checkpoint_compatible,
    infer_flow_checkpoint_architecture,
)
from turbovla.models.turbovla import TurboVLA


class _CaptureFlow(nn.Module):
    def forward(self, **kwargs):
        return kwargs


def _direct_route_stub() -> TurboVLA:
    model = TurboVLA.__new__(TurboVLA)
    nn.Module.__init__(model)
    model.action_head_type = "flow_matching"
    model.flow_state_dim = 8
    model.flow_state_encoding = "zero_pad"
    model.state_dim = 8
    model.config = SimpleNamespace(history=SimpleNamespace(r3m_enabled=False, visual_enabled=False))
    model.history_encoder = None
    model.flow_action_policy = _CaptureFlow()

    def encode_condition(self, instructions, samples):
        return torch.zeros(2, 3, 256), torch.tensor([[False], [True]]), None

    model._encode_condition_with_mask = encode_condition.__get__(model, TurboVLA)
    return model


def test_direct_state_route_omits_state_projection_tokens_and_passes_raw_state() -> None:
    model = _direct_route_stub()
    states = torch.randn(2, 8)
    result = model(["a", "b"], {}, states)

    assert not hasattr(model, "state_proj_module")
    assert result["states"] is states
    assert result["condition"].shape == (2, 3, 256)
    assert result["condition_padding_mask"].shape == (2, 3)


def test_state_tokens_zero_pad_adds_both_state_conditions() -> None:
    model = _direct_route_stub()
    model.flow_state_encoding = "state_tokens_zero_pad"
    model.state_proj_module = nn.Sequential(nn.Linear(8, 256, bias=False), nn.Unflatten(-1, (1, 256)))
    states = torch.randn(2, 8)
    result = model(["a", "b"], {}, states)

    assert result["states"] is states
    assert result["condition"].shape == (2, 4, 256)
    assert torch.equal(result["condition"][:, 3:4], model.state_proj_module(states))
    assert result["condition_padding_mask"].shape == (2, 4)
    assert not result["condition_padding_mask"][:, 3:].any()


def test_zero_pad_state_token_is_exact_parameter_free_and_differentiable() -> None:
    torch.manual_seed(4)
    head = MyVLAFlowMatchingActionHead(
        hidden_size=256, action_dim=7, chunk_size=2, num_heads=4,
        condition_layers=1, dit_layers=1, num_target_vision_tokens=2,
        num_static_future_tokens=1, state_dim=8, state_encoding="zero_pad",
        bijection_blocks=8,
    )
    states = torch.randn(1, 8, requires_grad=True)
    token = head._encode_state_condition(states)
    assert token.shape == (1, 1, 256)
    assert torch.equal(token[..., :8], states.unsqueeze(1))
    assert torch.count_nonzero(token[..., 8:]) == 0
    assert not any("state_encoder" in name for name, _ in head.named_parameters())
    assert head.flow.num_blocks == 8 and len(head.flow.bijection) == 8

    result = head(
        condition=torch.randn(1, 4, 256),
        condition_padding_mask=torch.zeros(1, 4, dtype=torch.bool),
        states=states,
        actions=torch.randn(1, 2, 7),
        action_masks=torch.ones(1, 2, dtype=torch.bool),
    )
    result["loss"].backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    gradients = dict(head.named_parameters())
    assert gradients["flow.bijection.6.translate_net.network.0.weight"].grad is not None
    assert gradients["flow.bijection.7.translate_net.network.0.weight"].grad is not None


def test_zero_pad_state_changes_fixed_noise_prediction() -> None:
    head = MyVLAFlowMatchingActionHead(
        hidden_size=256, action_dim=7, chunk_size=2, num_heads=4,
        condition_layers=1, dit_layers=16, num_target_vision_tokens=2,
        num_static_future_tokens=1, state_dim=8, state_encoding="zero_pad",
        bijection_blocks=8,
    ).eval()
    condition = torch.randn(1, 4, 256)
    padding = torch.zeros(1, 4, dtype=torch.bool)
    torch.manual_seed(19)
    first = head(condition, padding, states=torch.zeros(1, 8))
    torch.manual_seed(19)
    second = head(condition, padding, states=torch.ones(1, 8))
    # A one-layer random test head can have a very small initial state effect;
    # exact fixed-noise equality would still indicate that the state token was
    # dropped before DiT.
    assert torch.max((first - second).abs()).item() > 0.0


def _checkpoint_from_head(head: nn.Module, **metadata):
    return {
        "model_state_dict": {f"flow_action_policy.{key}": value for key, value in head.state_dict().items()},
        **metadata,
    }


def test_checkpoint_route_block_and_encoding_inference_and_mismatch() -> None:
    direct = MyVLAFlowMatchingActionHead(
        hidden_size=256, action_dim=7, chunk_size=2, condition_layers=1, dit_layers=1,
        state_dim=8, state_encoding="zero_pad", bijection_blocks=8,
    )
    direct_payload = _checkpoint_from_head(
        direct, flow_state_dim=8, flow_state_encoding="zero_pad", flow_bijection_blocks=8,
    )
    assert infer_flow_checkpoint_architecture(direct_payload, state_dim=8) == (8, 8, "zero_pad")
    target = SimpleNamespace(
        action_head_type="flow_matching", state_dim=8, flow_state_dim=8,
        flow_state_encoding="zero_pad", flow_bijection_blocks=8,
    )
    assert_flow_checkpoint_compatible(direct_payload, target)
    target.flow_state_encoding = "native_mlp"
    with pytest.raises(RuntimeError, match="incompatible flow checkpoint architecture"):
        assert_flow_checkpoint_compatible(direct_payload, target)

    legacy = MyVLAFlowMatchingActionHead(
        hidden_size=256, action_dim=7, chunk_size=2, condition_layers=1, dit_layers=1,
        state_dim=0, bijection_blocks=6,
    )
    legacy_payload = _checkpoint_from_head(legacy)
    legacy_payload["model_state_dict"]["state_proj_module.layers.0.weight"] = torch.empty(1)
    assert infer_flow_checkpoint_architecture(legacy_payload, state_dim=8) == (0, 6, "state_tokens")

    inconsistent = dict(direct_payload)
    inconsistent["flow_config"] = {"state_encoding": "native_mlp"}
    with pytest.raises(RuntimeError, match="inconsistent flow_state_encoding"):
        infer_flow_checkpoint_architecture(inconsistent, state_dim=8)

    combined = dict(direct_payload)
    combined["flow_state_encoding"] = "state_tokens_zero_pad"
    combined["model_state_dict"] = dict(direct_payload["model_state_dict"])
    combined["model_state_dict"]["state_proj_module.net.1.weight"] = torch.empty(1)
    assert infer_flow_checkpoint_architecture(combined, state_dim=8) == (8, 8, "state_tokens_zero_pad")
