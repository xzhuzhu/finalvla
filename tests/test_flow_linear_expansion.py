import torch
import torch.nn.functional as F

from turbovla.models.components.myvla_action_head import (
    LinearExpansionBijection,
    MyVLAFlowMatchingActionHead,
)


def test_linear_expansion_bijection_uses_six_latent_blocks() -> None:
    flow = LinearExpansionBijection(input_dim=7, latent_dim=35)

    assert flow.expansion.in_features == 7
    assert flow.expansion.out_features == 35
    assert flow.num_blocks == 6
    assert len(flow.bijection) == 6


def test_direct_inverse_round_trip_before_normalization() -> None:
    torch.manual_seed(0)
    flow = LinearExpansionBijection(input_dim=7, latent_dim=35)
    actions = torch.randn(2, 12, 7)

    latent = flow(actions, mode="direct")
    reconstructed = flow(latent, mode="inverse")

    assert latent.shape == (2, 12, 35)
    assert reconstructed.shape == actions.shape
    assert torch.allclose(reconstructed, actions, atol=2e-5, rtol=2e-5)


def test_expansion_and_down_projection_multiply_to_identity() -> None:
    torch.manual_seed(4)
    flow = LinearExpansionBijection(input_dim=7, latent_dim=35)
    with torch.no_grad():
        flow.expansion.weight.add_(0.1 * torch.randn_like(flow.expansion.weight))

    up = flow.expansion_weight()
    down = flow.inverse_expansion_weight()

    assert up.shape == (35, 7)
    assert down.shape == (7, 35)
    assert torch.allclose(down @ up, torch.eye(7), atol=1e-6, rtol=1e-6)


def test_normalized_latent_path_is_finite_and_differentiable() -> None:
    torch.manual_seed(1)
    flow = LinearExpansionBijection(input_dim=7, latent_dim=35)
    actions = torch.randn(2, 12, 7, requires_grad=True)

    latent = flow(actions, mode="direct")
    normalized = F.normalize(latent, dim=-1)
    reconstructed = flow(normalized, mode="inverse")
    loss = reconstructed.square().mean()
    loss.backward()

    assert reconstructed.shape == actions.shape
    assert torch.isfinite(reconstructed).all()
    assert actions.grad is not None and torch.isfinite(actions.grad).all()
    assert flow.expansion.weight.grad is not None


def test_action_head_replaces_original_twelve_block_flow() -> None:
    head = MyVLAFlowMatchingActionHead(
        hidden_size=256,
        action_dim=7,
        chunk_size=12,
        dit_layers=1,
        condition_layers=1,
    )

    assert isinstance(head.flow, LinearExpansionBijection)
    assert head.flow.expansion.out_features == 35
    assert head.flow.num_blocks == 6
    assert len(head.flow.bijection) == 6


def test_action_head_training_reduces_expanded_latent_loss() -> None:
    torch.manual_seed(2)
    head = MyVLAFlowMatchingActionHead(
        hidden_size=256,
        action_dim=7,
        chunk_size=2,
        dit_layers=1,
        condition_layers=1,
        num_target_vision_tokens=2,
        num_static_future_tokens=1,
        state_dim=0,
    )
    condition = torch.randn(1, 4, 256)
    padding_mask = torch.zeros(1, 4, dtype=torch.bool)
    actions = torch.randn(1, 2, 7)
    action_masks = torch.ones(1, 2, dtype=torch.bool)

    result = head(
        condition=condition,
        condition_padding_mask=padding_mask,
        actions=actions,
        action_masks=action_masks,
    )
    result["loss"].backward()

    assert result["pred_actions"].shape == actions.shape
    assert torch.isfinite(result["loss"])
    assert head.flow.expansion.weight.grad is not None
    assert torch.isfinite(head.flow.expansion.weight.grad).all()
