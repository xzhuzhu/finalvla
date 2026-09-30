from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn

from turbovla.data.libero_rlds import LiberoRLDSDataset
from turbovla.models.turbovla import TurboVLA, apply_rotary_position_encoding


class _FakeR3M(nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pooled = pixels.float().mean(dim=(-3, -2, -1))
        return pooled[..., None].expand(*pooled.shape, 4)


class _IdentityProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.skip = nn.Linear(4, 4, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _FirstHalf(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., : values.shape[-1] // 2]


class _LastStateGate(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., -1:]


class _CaptureAttention(nn.Module):
    def forward(self, query, key, value, **kwargs):
        self.key = key.detach().clone()
        self.value = value.detach().clone()
        return torch.zeros_like(query), None


def test_rope_preserves_norm_and_distinguishes_past_positions() -> None:
    repeated = torch.ones(1, 3, 8)
    positions = torch.tensor([-3, -2, -1])

    encoded = apply_rotary_position_encoding(repeated, positions)

    assert torch.allclose(encoded.norm(dim=-1), repeated.norm(dim=-1), atol=1e-6)
    assert not torch.allclose(encoded[:, 0], encoded[:, 1])
    assert not torch.allclose(encoded[:, 1], encoded[:, 2])


def _memory_model(
    *,
    dynamic_gate: bool = False,
    residual_gate: bool = False,
    token_dropout: float = 0.0,
    rope_values: bool = True,
) -> TurboVLA:
    model = TurboVLA.__new__(TurboVLA)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        history=SimpleNamespace(
            r3m_enabled=True,
            length=12,
            r3m_memory_num_queries=2,
            r3m_rope_base=10000.0,
            r3m_memory_dynamic_gate=dynamic_gate,
            r3m_memory_residual_gate=residual_gate,
            r3m_memory_token_dropout=token_dropout,
            r3m_memory_rope_values=rope_values,
        ),
        r3m=SimpleNamespace(num_views=2, image_size=2),
        interaction=SimpleNamespace(hidden_dim=4),
    )
    model.r3m_encoder = _FakeR3M()
    model.r3m_projection = _IdentityProjection()
    model.r3m_view_embedding = nn.Parameter(
        torch.tensor([[[1.0] * 4, [2.0] * 4]])
    )
    model.r3m_history_memory_queries = nn.Parameter(torch.zeros(2, 4))
    model.r3m_history_memory_type_embedding = nn.Parameter(
        torch.tensor([[[[0.0] * 4, [1.0] * 4]]])
    )
    if dynamic_gate:
        model.r3m_history_memory_dynamic_gate = _LastStateGate()
        model.r3m_history_memory_residual_scale = None
    elif residual_gate:
        model.r3m_history_memory_dynamic_gate = _LastStateGate()
        model.r3m_history_memory_gate_logit = nn.Parameter(torch.tensor(0.0))
        model.r3m_history_memory_residual_scale = nn.Parameter(torch.tensor(0.0))
    else:
        model.r3m_history_memory_gate_logit = nn.Parameter(torch.tensor(0.0))
        model.r3m_history_memory_dynamic_gate = None
        model.r3m_history_memory_residual_scale = None
    model.r3m_history_cross_attention = nn.MultiheadAttention(
        4, 1, dropout=0.0, batch_first=True
    )
    model.r3m_history_relation = _FirstHalf()
    with torch.no_grad():
        attention = model.r3m_history_cross_attention
        attention.in_proj_weight.zero_()
        for offset in (0, 4, 8):
            attention.in_proj_weight[offset : offset + 4].copy_(torch.eye(4))
        attention.in_proj_bias.zero_()
        attention.out_proj.weight.copy_(torch.eye(4))
        attention.out_proj.bias.zero_()
    return model


def test_r3m_history_builds_four_current_conditioned_view_separated_memories() -> None:
    model = _memory_model()

    pixels = torch.zeros(1, 12, 2, 3, 2, 2)
    pixels[:, -2, 0] = 3.0
    pixels[:, -2, 1] = 4.0
    pixels[:, -1, 0] = 7.0
    pixels[:, -1, 1] = 20.0
    # Current tokens already contain their corresponding view embeddings.
    current = torch.tensor([[[6.0] * 4, [12.0] * 4]])
    mask = torch.tensor([[False] * 10 + [True, True]])

    tokens, token_mask = model.encode_r3m_history(
        {"r3m_history": pixels},
        current,
        mask,
    )

    assert tokens.shape == (1, 4, 4)
    assert torch.equal(token_mask, torch.ones(1, 4, dtype=torch.bool))
    assert torch.isfinite(tokens).all()
    # Query type and camera view remain distinguishable after retrieval.
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])
    assert not torch.allclose(tokens[:, 0], tokens[:, 2])


def test_key_only_rope_preserves_raw_history_values() -> None:
    model = _memory_model(rope_values=False)
    capture = _CaptureAttention()
    model.r3m_history_cross_attention = capture
    pixels = torch.zeros(1, 12, 2, 3, 2, 2)
    pixels[:, -2, 0] = 3.0
    pixels[:, -2, 1] = 4.0
    pixels[:, -1, 0] = 7.0
    pixels[:, -1, 1] = 20.0
    current = torch.tensor([[[6.0] * 4, [12.0] * 4]])
    mask = torch.tensor([[False] * 10 + [True, True]])

    model.encode_r3m_history({"r3m_history": pixels}, current, mask)

    assert not torch.allclose(capture.key, capture.value)
    assert torch.all(capture.value[0, -2] == 3.0)
    assert torch.all(capture.value[0, -1] == 7.0)
    assert torch.all(capture.value[1, -2] == 4.0)
    assert torch.all(capture.value[1, -1] == 20.0)


def test_intentional_goal_operator_changes_how_current_queries_history() -> None:
    model = _memory_model()
    model.config.history.r3m_intentional_memory = True
    model.r3m_history_intentional_goal_factors = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    model.r3m_history_intentional_current_factors = nn.Linear(4, 4, bias=False)
    model.r3m_history_intentional_relation_out = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.r3m_history_intentional_goal_factors[0].weight.copy_(torch.eye(4))
        model.r3m_history_intentional_goal_factors[0].bias.zero_()
        model.r3m_history_intentional_current_factors.weight.copy_(torch.eye(4))
        model.r3m_history_intentional_relation_out.weight.copy_(torch.eye(4))

    pixels = torch.zeros(1, 12, 2, 3, 2, 2)
    pixels[:, -2] = 1.0
    pixels[:, -1, 0] = 4.0
    pixels[:, -1, 1] = 7.0
    current = torch.tensor([[[2.0] * 4, [3.0] * 4]])
    mask = torch.tensor([[False] * 10 + [True, True]])

    positive, positive_mask = model.encode_r3m_history(
        {"r3m_history": pixels},
        current,
        mask,
        goal_summary=torch.ones(1, 4),
    )
    negative, negative_mask = model.encode_r3m_history(
        {"r3m_history": pixels},
        current,
        mask,
        goal_summary=-torch.ones(1, 4),
    )

    assert torch.equal(positive_mask, negative_mask)
    assert not torch.allclose(positive, negative)


def test_intentional_memory_uses_four_tokens_without_auxiliary_losses() -> None:
    model = _memory_model()
    model.config.history.r3m_intentional_memory = True
    model.config.history.r3m_belief_future_horizons = (1, 4, 8, 12)
    model.config.history.state_dim = 8
    model.config.r3m.output_dim = 4
    model.r3m_history_intentional_goal_factors = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    model.r3m_history_intentional_current_factors = nn.Linear(4, 4, bias=False)
    model.r3m_history_intentional_relation_out = nn.Linear(4, 4, bias=False)

    samples = {
        "r3m_history": torch.ones(2, 12, 2, 3, 2, 2),
    }
    tokens, token_mask, losses = model.encode_r3m_intentional_memory(
        samples,
        current_r3m_tokens=torch.ones(2, 2, 4),
        history_mask=torch.ones(2, 12, dtype=torch.bool),
        history_state_tokens=torch.zeros(2, 12, 4),
        goal_summary=torch.ones(2, 4),
    )

    assert tokens.shape == (2, 4, 4)
    assert torch.equal(token_mask, torch.ones(2, 4, dtype=torch.bool))
    assert losses == {}
    tokens.mean().backward()
    assert model.r3m_history_intentional_relation_out.weight.grad is not None


def test_r3m_history_all_padding_returns_zero_memories_without_nans() -> None:
    model = _memory_model()
    pixels = torch.zeros(1, 12, 2, 3, 2, 2)
    current = torch.tensor([[[6.0] * 4, [12.0] * 4]])

    tokens, token_mask = model.encode_r3m_history(
        {"r3m_history": pixels},
        current,
        torch.zeros(1, 12, dtype=torch.bool),
    )

    assert torch.equal(token_mask, torch.zeros(1, 4, dtype=torch.bool))
    assert torch.equal(tokens, torch.zeros_like(tokens))


def test_dynamic_gate_uses_state_history_per_view_and_query() -> None:
    model = _memory_model(dynamic_gate=True)
    model.eval()
    pixels = torch.zeros(1, 12, 2, 3, 2, 2)
    pixels[:, -1] = 3.0
    current = torch.tensor([[[6.0] * 4, [12.0] * 4]])
    mask = torch.tensor([[False] * 11 + [True]])

    low_gate, low_mask = model.encode_r3m_history(
        {"r3m_history": pixels}, current, mask, torch.zeros(1, 12, 4)
    )
    high_gate, high_mask = model.encode_r3m_history(
        {"r3m_history": pixels}, current, mask, torch.full((1, 12, 4), 2.0)
    )

    assert torch.equal(low_mask, high_mask)
    assert low_gate.shape == (1, 4, 4)
    assert not torch.allclose(low_gate, high_gate)
    assert high_gate.norm() > low_gate.norm()


def test_residual_gate_starts_at_scalar_gate_then_learns_state_correction() -> None:
    model = _memory_model(residual_gate=True)
    model.eval()
    pixels = torch.zeros(1, 12, 2, 3, 2, 2)
    pixels[:, -1] = 3.0
    current = torch.tensor([[[6.0] * 4, [12.0] * 4]])
    mask = torch.tensor([[False] * 11 + [True]])

    low_state = torch.zeros(1, 12, 4)
    high_state = torch.full((1, 12, 4), 2.0)
    baseline_low, _ = model.encode_r3m_history(
        {"r3m_history": pixels}, current, mask, low_state
    )
    baseline_high, _ = model.encode_r3m_history(
        {"r3m_history": pixels}, current, mask, high_state
    )
    assert torch.allclose(baseline_low, baseline_high)

    with torch.no_grad():
        model.r3m_history_memory_residual_scale.fill_(1.0)
    adapted_low, _ = model.encode_r3m_history(
        {"r3m_history": pixels}, current, mask, low_state
    )
    adapted_high, _ = model.encode_r3m_history(
        {"r3m_history": pixels}, current, mask, high_state
    )
    assert not torch.allclose(adapted_low, adapted_high)
    assert adapted_high.norm() > adapted_low.norm()


def test_history_token_dropout_masks_complete_tokens_only_during_training(monkeypatch) -> None:
    model = _memory_model(token_dropout=0.15)
    pixels = torch.zeros(1, 12, 2, 3, 2, 2)
    pixels[:, -1] = 3.0
    current = torch.tensor([[[6.0] * 4, [12.0] * 4]])
    mask = torch.tensor([[False] * 11 + [True]])

    model.eval()
    _, eval_mask = model.encode_r3m_history({"r3m_history": pixels}, current, mask)
    assert torch.equal(eval_mask, torch.ones(1, 4, dtype=torch.bool))

    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, **kwargs: torch.zeros(shape, device=kwargs.get("device")),
    )
    model.train()
    train_tokens, train_mask = model.encode_r3m_history(
        {"r3m_history": pixels}, current, mask
    )

    assert not train_mask.any()
    assert torch.equal(train_tokens, torch.zeros_like(train_tokens))


def test_dataset_right_aligns_prior_r3m_frames_without_current_leakage() -> None:
    dataset = LiberoRLDSDataset.__new__(LiberoRLDSDataset)
    dataset.history_length = 12
    dataset.history_visual_dinov3 = False
    dataset.history_r3m = True
    dataset.chunk_size = 12
    dataset.expected_image_size = 256
    dataset._normalize_state = lambda value: value
    dataset._normalize_action_chunk = lambda value: value
    dataset._process_image_pair = lambda image: {
        "dinov3": torch.zeros(3, 256, 256),
        "r3m": torch.full((3, 224, 224), int(image[0, 0, 0]), dtype=torch.uint8),
    }
    steps = []
    history_cache = []
    for value in range(4):
        image = np.full((256, 256, 3), value, dtype=np.uint8)
        steps.append(
            {
                "observation": {
                    "image": image,
                    "wrist_image": image,
                    "state": np.full(8, value, dtype=np.float32),
                },
                "language_instruction": b"instruction",
                "action": np.full(7, value, dtype=np.float32),
            }
        )
        history_cache.append(
            (
                {"r3m": torch.full((3, 224, 224), value, dtype=torch.uint8)},
                {"r3m": torch.full((3, 224, 224), value, dtype=torch.uint8)},
            )
        )

    item = dataset._build_step_sample(steps, t=3, episode_len=4, image_cache=history_cache)
    history_mask = item[6]
    history_r3m = item[7]["r3m_history"]

    assert torch.equal(history_mask, torch.tensor([False] * 9 + [True] * 3))
    assert torch.all(history_r3m[-3] == 0)
    assert torch.all(history_r3m[-2] == 1)
    assert torch.all(history_r3m[-1] == 2)
    assert not torch.any(history_r3m == 3)


def test_dataset_builds_history_without_predictive_targets() -> None:
    dataset = LiberoRLDSDataset.__new__(LiberoRLDSDataset)
    dataset.history_length = 12
    dataset.history_visual_dinov3 = False
    dataset.history_r3m = True
    dataset.chunk_size = 12
    dataset.expected_image_size = 256
    dataset._normalize_state = lambda value: value
    dataset._normalize_action_chunk = lambda value: value
    dataset._process_image_pair = lambda image: {
        "dinov3": torch.zeros(3, 256, 256),
        "r3m": torch.full((3, 224, 224), int(image[0, 0, 0]), dtype=torch.uint8),
    }
    steps = []
    image_cache = []
    for value in range(5):
        image = np.full((256, 256, 3), value, dtype=np.uint8)
        steps.append(
            {
                "observation": {
                    "image": image,
                    "wrist_image": image,
                    "state": np.full(8, value, dtype=np.float32),
                },
                "language_instruction": b"instruction",
                "action": np.full(7, value, dtype=np.float32),
            }
        )
        image_cache.append(
            (
                {"r3m": torch.full((3, 224, 224), value, dtype=torch.uint8)},
                {"r3m": torch.full((3, 224, 224), value, dtype=torch.uint8)},
            )
        )

    item = dataset._build_step_sample(steps, t=1, episode_len=5, image_cache=image_cache)
    targets = item[7]

    assert "r3m_history" in targets
    assert "r3m_future" not in targets
    assert "future_states" not in targets
    assert "future_mask" not in targets


class _MaskedIdentity(nn.Module):
    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return values.masked_fill(~mask.unsqueeze(-1), 0.0)


class _FirstQuarter(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[..., : values.shape[-1] // 4]


def test_belief_uses_history_without_auxiliary_prediction_targets() -> None:
    model = TurboVLA.__new__(TurboVLA)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        history=SimpleNamespace(
            r3m_predictive_belief=True,
            r3m_tacit_belief_fusion=False,
            length=12,
            r3m_belief_num_slots=4,
            r3m_belief_future_horizons=(1, 4, 8, 12),
            r3m_rope_base=10000.0,
            state_dim=8,
        ),
        r3m=SimpleNamespace(num_views=2, image_size=2, output_dim=4),
        interaction=SimpleNamespace(hidden_dim=4),
    )
    model.r3m_encoder = _FakeR3M()
    model.r3m_projection = _IdentityProjection()
    model.r3m_view_embedding = nn.Parameter(torch.zeros(1, 2, 4))
    model.r3m_history_memory_queries = nn.Parameter(torch.zeros(4, 4))
    model.r3m_history_memory_type_embedding = nn.Parameter(torch.zeros(1, 1, 4, 4))
    model.r3m_history_cross_attention = nn.MultiheadAttention(4, 1, batch_first=True)
    model.r3m_history_relation = _FirstHalf()
    model.r3m_history_dynamics_encoder = nn.ModuleList([_MaskedIdentity()])
    model.r3m_history_dynamics_attention = nn.MultiheadAttention(4, 1, batch_first=True)
    model.r3m_history_state_delta_projection = nn.Identity()
    model.r3m_history_goal_projection = nn.Identity()
    model.r3m_history_belief_fusion = _FirstQuarter()
    model.r3m_history_current_hidden_predictor = nn.Identity()
    model.r3m_history_innovation_projection = nn.Identity()
    model.r3m_history_innovation_gate = nn.Linear(12, 1)
    model.r3m_history_memory_gate_logit = nn.Parameter(torch.tensor(-2.1972246))
    model.r3m_history_future_horizon_embedding = nn.Parameter(torch.zeros(1, 4, 4))
    model.r3m_history_future_type_embedding = nn.Parameter(torch.zeros(1, 4, 4))
    model.r3m_history_future_context = _FirstHalf()
    model.r3m_history_future_predictor = nn.Linear(4, 4)

    samples = {"r3m_history": torch.ones(2, 12, 2, 3, 2, 2)}
    tokens, token_mask, losses = model.encode_r3m_predictive_belief(
        samples,
        current_r3m_tokens=torch.ones(2, 2, 4),
        history_mask=torch.ones(2, 12, dtype=torch.bool),
        history_state_tokens=torch.zeros(2, 12, 4),
        goal_summary=torch.zeros(2, 4),
    )

    assert tokens.shape == (2, 12, 4)
    assert torch.equal(token_mask, torch.ones(2, 12, dtype=torch.bool))
    assert all(torch.isfinite(value) for value in losses.values())
    assert not any(key.endswith("loss") or "kl" in key for key in losses)
    tokens.mean().backward()
    assert model.r3m_history_future_predictor.weight.grad is not None

    model.config.history.r3m_controlled_causal_belief = True
    model.config.history.r3m_tacit_belief_fusion = True
    model.r3m_history_future_horizon_embedding = None
    model.r3m_history_future_type_embedding = None
    model.r3m_history_future_context = None
    model.r3m_history_future_predictor = None

    def fake_tacit(self, history_by_view, history_mask, current_base, posterior, state_summary):
        mask = history_mask.any(dim=1)[:, None].expand(-1, 8)
        return posterior.flatten(1, 2), mask, posterior.new_tensor(0.1)

    model._encode_r3m_tacit_belief_fusion = MethodType(fake_tacit, model)
    causal_tokens, causal_mask, causal_metrics = model.encode_r3m_predictive_belief(
        samples,
        current_r3m_tokens=torch.ones(2, 2, 4),
        history_mask=torch.ones(2, 12, dtype=torch.bool),
        history_state_tokens=torch.zeros(2, 12, 4),
        goal_summary=torch.zeros(2, 4),
    )
    assert causal_tokens.shape == (2, 8, 4)
    assert torch.equal(causal_mask, torch.ones(2, 8, dtype=torch.bool))
    assert not any(key.endswith("loss") or "kl" in key for key in causal_metrics)


def test_tacit_belief_fusion_outputs_v8_sized_memory_with_low_initial_correction() -> None:
    model = TurboVLA.__new__(TurboVLA)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        history=SimpleNamespace(length=12, r3m_memory_num_queries=2, r3m_rope_base=10000.0),
    )
    model.r3m_view_embedding = nn.Parameter(torch.zeros(1, 2, 4))
    model.r3m_history_tacit_memory_queries = nn.Parameter(torch.zeros(2, 4))
    model.r3m_history_tacit_memory_type_embedding = nn.Parameter(torch.zeros(1, 1, 2, 4))
    model.r3m_history_tacit_cross_attention = nn.MultiheadAttention(4, 1, batch_first=True)
    model.r3m_history_tacit_relation = _FirstHalf()
    model.r3m_history_tacit_gate_logit = nn.Parameter(torch.tensor(-2.1972246))
    model.r3m_history_belief_to_tacit_attention = nn.MultiheadAttention(
        4,
        1,
        batch_first=True,
    )
    model.r3m_history_belief_to_tacit_relation = _FirstHalf()
    model.r3m_history_belief_to_tacit_gate = nn.Linear(12, 1)
    nn.init.zeros_(model.r3m_history_belief_to_tacit_gate.weight)
    nn.init.constant_(model.r3m_history_belief_to_tacit_gate.bias, -2.1972246)

    tokens, token_mask, gate_mean = model._encode_r3m_tacit_belief_fusion(
        history_by_view=torch.randn(4, 12, 4),
        history_mask=torch.ones(2, 12, dtype=torch.bool),
        current_base=torch.randn(2, 2, 4),
        posterior=torch.randn(2, 2, 4, 4),
        state_summary=torch.randn(2, 4),
    )

    assert tokens.shape == (2, 4, 4)
    assert torch.equal(token_mask, torch.ones(2, 4, dtype=torch.bool))
    assert torch.allclose(gate_mean, torch.tensor(0.1), atol=1e-6)


class _HistoryEncoder(nn.Module):
    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.zeros(states.shape[0], states.shape[1], 4, device=states.device)


class _StateProjection(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(state.shape[0], 2, 4, device=state.device)


class _CaptureFlow(nn.Module):
    def forward(self, **kwargs):
        self.kwargs = kwargs
        return kwargs


def test_forward_appends_r3m_history_before_state_history() -> None:
    model = TurboVLA.__new__(TurboVLA)
    nn.Module.__init__(model)
    model.action_head_type = "flow_matching"
    model.config = SimpleNamespace(
        history=SimpleNamespace(r3m_enabled=True, visual_enabled=False),
    )
    model.history_encoder = _HistoryEncoder()
    model.state_proj_module = _StateProjection()
    model.flow_action_policy = _CaptureFlow()

    def encode_condition(self, instructions, samples):
        return (
            torch.zeros(1, 3, 4),
            torch.tensor([[False]]),
            torch.zeros(1, 2, 4),
        )

    def encode_history(self, samples, current, mask, state_tokens):
        return torch.ones(1, 4, 4), mask.any(dim=1)[:, None].expand(-1, 4)

    model._encode_condition_with_mask = MethodType(encode_condition, model)
    model.encode_r3m_history = MethodType(encode_history, model)
    history_mask = torch.ones(1, 12, dtype=torch.bool)

    result = model(
        ["instruction"],
        {},
        torch.zeros(1, 8),
        history_states=torch.zeros(1, 12, 8),
        history_mask=history_mask,
    )

    assert result["condition"].shape == (1, 21, 4)
    assert result["condition_padding_mask"].shape == (1, 21)
    assert torch.all(result["condition"][:, 3:7] == 1)


def test_forward_ablation_removes_only_four_r3m_history_tokens() -> None:
    model = TurboVLA.__new__(TurboVLA)
    nn.Module.__init__(model)
    model.action_head_type = "flow_matching"
    model.config = SimpleNamespace(
        history=SimpleNamespace(r3m_enabled=True, visual_enabled=False),
    )
    model.history_encoder = _HistoryEncoder()
    model.state_proj_module = _StateProjection()
    model.flow_action_policy = _CaptureFlow()

    def encode_condition(self, instructions, samples):
        return (
            torch.zeros(1, 3, 4),
            torch.tensor([[False]]),
            torch.zeros(1, 2, 4),
        )

    def fail_if_encoded(self, samples, current, mask):
        raise AssertionError("ablated R3M history must not be encoded")

    model._encode_condition_with_mask = MethodType(encode_condition, model)
    model.encode_r3m_history = MethodType(fail_if_encoded, model)
    history_mask = torch.ones(1, 12, dtype=torch.bool)

    result = model(
        ["instruction"],
        {},
        torch.zeros(1, 8),
        history_states=torch.zeros(1, 12, 8),
        history_mask=history_mask,
        ablate_history_r3m_tokens=True,
    )

    # 3 current vision/text + 12 state-history + 2 current-state tokens.
    assert result["condition"].shape == (1, 17, 4)
    assert result["condition_padding_mask"].shape == (1, 17)
