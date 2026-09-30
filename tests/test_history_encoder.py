import pytest
import torch

from turbovla.models.configuration import HistoryConfig, TurboVLAConfig
from turbovla.models.history_encoder import HistoryEncoder, MambaBlock


@pytest.mark.parametrize("history_length", [4, 12])
def test_history_encoder_supports_configured_length(history_length: int) -> None:
    encoder = HistoryEncoder(history_length=history_length)
    states = torch.randn(2, history_length, 8)
    mask = torch.ones(2, history_length, dtype=torch.bool)
    mask[0, :2] = False

    output = encoder(states, mask)

    assert output.shape == (2, history_length, 256)
    assert torch.count_nonzero(output[0, :2]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="matched scan requires CUDA")
def test_r3m_dynamics_mamba_dispatches_matched_cuda(monkeypatch) -> None:
    block = MambaBlock(256).cuda()
    calls = []
    matched_scan = block._selective_scan_matched_cuda

    def capture_matched_scan(value, valid_mask):
        calls.append((value.dtype, tuple(value.shape), tuple(valid_mask.shape)))
        return matched_scan(value, valid_mask)

    monkeypatch.setattr(block, "_selective_scan_matched_cuda", capture_matched_scan)
    dynamics = torch.randn(4, 12, 256, device="cuda", requires_grad=True)
    mask = torch.ones(4, 12, device="cuda", dtype=torch.bool)
    mask[0, :3] = False
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = block(dynamics, mask)
    output.float().square().mean().backward()

    assert calls == [(torch.bfloat16, (4, 12, 512), (4, 12))]
    assert torch.count_nonzero(output[0, :3]) == 0
    assert dynamics.grad is not None and torch.isfinite(dynamics.grad).all()


def test_history12_model_config_is_valid() -> None:
    config = TurboVLAConfig(history=HistoryConfig(enabled=True, length=12))

    assert config.history.length == 12


def test_unsupported_history_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="history_length in"):
        HistoryEncoder(history_length=8)
