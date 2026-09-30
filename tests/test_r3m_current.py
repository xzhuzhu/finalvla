from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from turbovla.data.libero_rlds import vla_collate_fn
from turbovla.evaluation.policy import build_r3m_manual_processor
from turbovla.models.configuration import HistoryConfig, R3MEncoderConfig, TurboVLAConfig
from turbovla.models.turbovla import TurboVLA


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


def test_finalvla_history_config_has_only_active_parameters() -> None:
    config = TurboVLAConfig(
        r3m=R3MEncoderConfig(enabled=True, checkpoint_path="r3m.pth"),
        history=HistoryConfig(enabled=True, length=12, r3m_enabled=True),
    )
    assert config.history.r3m_predictive_belief
    assert config.history.r3m_controlled_causal_belief
    assert config.history.r3m_tacit_belief_fusion
    assert not config.history.visual_enabled
    assert "r3m_belief_future_horizons" not in config.to_dict()["history"]
    assert "r3m_intentional_memory" not in config.to_dict()["history"]


def test_collator_adds_current_r3m_without_image_history() -> None:
    current = (
        {
            "dinov3": torch.zeros(3, 256, 256),
            "r3m": torch.zeros(3, 224, 224, dtype=torch.uint8),
        },
        {
            "dinov3": torch.ones(3, 256, 256),
            "r3m": torch.ones(3, 224, 224, dtype=torch.uint8),
        },
    )
    item = (
        current,
        "pick up the object",
        torch.zeros(8),
        torch.zeros(12, 7),
        torch.ones(12),
        torch.zeros(12, 8),
        torch.tensor([False] * 11 + [True]),
    )

    samples, _, _, _, _, history_states, history_mask = vla_collate_fn([item, item])

    assert samples["dinov3"].shape == (2, 2, 3, 256, 256)
    assert samples["r3m"].shape == (2, 2, 3, 224, 224)
    assert samples["r3m"].dtype == torch.uint8
    assert "dinov3_history" not in samples
    assert "r3m_history" not in samples
    assert history_states.shape == (2, 12, 8)
    assert history_mask.shape == (2, 12)


def test_collator_adds_twelve_r3m_history_frames() -> None:
    current = (
        {
            "dinov3": torch.zeros(3, 256, 256),
            "r3m": torch.zeros(3, 224, 224, dtype=torch.uint8),
        },
        {
            "dinov3": torch.ones(3, 256, 256),
            "r3m": torch.ones(3, 224, 224, dtype=torch.uint8),
        },
    )
    item = (
        current,
        "pick up the object",
        torch.zeros(8),
        torch.zeros(12, 7),
        torch.ones(12),
        torch.zeros(12, 8),
        torch.tensor([False] * 10 + [True, True]),
        {"r3m_history": torch.zeros(12, 2, 3, 224, 224, dtype=torch.uint8)},
    )

    samples, _, _, _, _, _, history_mask = vla_collate_fn([item, item])

    assert samples["r3m_history"].shape == (2, 12, 2, 3, 224, 224)
    assert samples["r3m_history"].dtype == torch.uint8
    assert history_mask.shape == (2, 12)


def test_r3m_current_encoder_returns_one_token_per_view() -> None:
    model = TurboVLA.__new__(TurboVLA)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        r3m=SimpleNamespace(num_views=2, image_size=224),
    )
    model.r3m_encoder = _FakeR3M()
    model.r3m_projection = _IdentityProjection()
    model.r3m_view_embedding = nn.Parameter(torch.zeros(1, 2, 4))

    tokens = model.encode_r3m_current(
        {"r3m": torch.ones(3, 2, 3, 224, 224, dtype=torch.uint8)}
    )

    assert tokens.shape == (3, 2, 4)
    assert torch.allclose(tokens, torch.ones_like(tokens))


def test_r3m_processor_center_crops_uint8_images() -> None:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[16:240, 16:240] = 123

    pixels = build_r3m_manual_processor()([image, image])["pixel_values"]

    assert pixels.shape == (2, 3, 224, 224)
    assert pixels.dtype == torch.uint8
    assert torch.all(pixels == 123)


def test_policy_records_synchronized_r3m_history() -> None:
    from turbovla.evaluation.policy import TurboVLAPolicy

    policy = TurboVLAPolicy.__new__(TurboVLAPolicy)
    policy.history_length = 12
    policy.history_visual_dinov3 = False
    policy.history_r3m = True
    policy._state_history = deque(maxlen=12)
    policy._image_history = deque(maxlen=12)
    policy.device = torch.device("cpu")
    policy.model_dtype = torch.float32
    policy.r3m_processor = build_r3m_manual_processor()
    observation = {
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((256, 256, 3), dtype=np.uint8),
    }

    policy.record_history_state(observation)
    history_r3m = policy._history_r3m_tensor()

    assert history_r3m.shape == (1, 12, 2, 3, 224, 224)
    assert history_r3m.dtype == torch.uint8
    assert torch.count_nonzero(history_r3m[:, :-1]) == 0
