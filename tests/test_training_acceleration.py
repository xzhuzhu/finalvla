from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from turbovla.data.libero_rlds import vla_collate_fn
from turbovla.data.r3m_feature_cache import R3MFeatureCache, initialize_cache
from turbovla.models.history_encoder import HistoryEncoder
from turbovla.models.configuration import HistoryConfig, TurboVLAConfig
from turbovla.models.text_encoder import TurboVLATextEncoder
import turbovla.models.turbovla as turbovla_module
from turbovla.models.turbovla import TurboVLA, build_turbovla


class _RawR3M(nn.Module):
    def forward(self, values):
        return values.float().mean(dim=(-3, -2, -1), keepdim=False)[..., None].expand(*values.shape[:2], 4)


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip = nn.Linear(4, 4, bias=False)

    def forward(self, values):
        return self.skip(values)


def _r3m_model():
    model = TurboVLA.__new__(TurboVLA)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(r3m=SimpleNamespace(num_views=2, image_size=2, output_dim=4))
    model.r3m_encoder = _RawR3M()
    model.r3m_projection = _Projection()
    model.r3m_view_embedding = nn.Parameter(torch.zeros(1, 2, 4))
    return model


def test_cached_raw_r3m_matches_pixels_and_projection_gradients():
    torch.manual_seed(2)
    model = _r3m_model()
    pixels = torch.randint(0, 255, (3, 2, 3, 2, 2), dtype=torch.uint8)
    tokens, raw = model._encode_r3m_current_outputs({"r3m": pixels})
    tokens.square().sum().backward()
    pixel_grad = model.r3m_projection.skip.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    cached_tokens, cached_raw = model._encode_r3m_current_outputs({"r3m_features": raw.detach()})
    cached_tokens.square().sum().backward()
    assert torch.allclose(tokens, cached_tokens)
    assert torch.allclose(raw, cached_raw)
    assert torch.allclose(pixel_grad, model.r3m_projection.skip.weight.grad)


def test_cached_r3m_collation_keeps_current_history_alignment():
    features = torch.arange(20 * 2 * 512, dtype=torch.float32).view(20, 2, 512)
    # t=12 has a full twelve-frame history.
    t = 12
    current = ({"dinov3": torch.zeros(3, 256, 256), "r3m_features": features[t, 0]},
               {"dinov3": torch.zeros(3, 256, 256), "r3m_features": features[t, 1]})
    item = (current, "task", torch.zeros(8), torch.zeros(12, 7), torch.ones(12),
            torch.zeros(12, 8), torch.ones(12, dtype=torch.bool), {
                "r3m_history_features": features[:t],
            })
    samples, *_ = vla_collate_fn([item])
    assert torch.equal(samples["r3m_features"][0], features[t])
    assert torch.equal(samples["r3m_history_features"][0, -1], features[t - 1])
    assert "r3m_future_features" not in samples


def test_r3m_cache_rejects_bad_manifest_and_episode_shape(tmp_path):
    checkpoint = tmp_path / "r3m.pth"
    checkpoint.write_bytes(b"checkpoint")
    source = tmp_path / "source"
    source.mkdir()
    cache = initialize_cache(tmp_path / "cache", checkpoint, [str(source)])
    manifest = cache.root / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["feature_shape"] = [2, 513]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="feature_shape"):
        R3MFeatureCache(cache.root, checkpoint_path=checkpoint, dataset_dirs=[str(source)])


def test_frozen_training_hidden_cache_precedes_projection_and_invalidates_config():
    encoder = TurboVLATextEncoder.__new__(TurboVLATextEncoder)
    nn.Module.__init__(encoder)
    encoder.config = SimpleNamespace(frozen=True, max_length=21, padding_length=21,
                                     padding_length_by_instruction={}, sub_sentence_present=True)
    encoder.bert = nn.Linear(1, 1)
    encoder.bert.eval()
    encoder.use_frozen_training_cache = True
    encoder._train_hidden_cache = __import__("collections").OrderedDict()
    encoder._frozen_cache_bert_signature = None
    calls = []
    def fake_encode(instructions, device, length):
        calls.append(tuple(instructions))
        return (torch.ones(len(instructions), length, 3, device=device), torch.ones(len(instructions), length, dtype=torch.bool, device=device), torch.ones(len(instructions), length, length, dtype=torch.bool, device=device))
    encoder._encode_group = fake_encode
    first = encoder._encode_frozen_hidden(["a", "b"], torch.device("cpu"))
    second = encoder._encode_frozen_hidden(["a", "b"], torch.device("cpu"))
    assert len(calls) == 1 and torch.equal(first[0], second[0])
    encoder.config.padding_length = 20
    encoder._encode_frozen_hidden(["a", "b"], torch.device("cpu"))
    assert len(calls) == 2
    encoder.config.padding_length = 21
    encoder._encode_frozen_hidden(["a", "b"], torch.device("cpu"))
    calls_before_mutation = len(calls)
    with torch.no_grad():
        encoder.bert.weight.add_(1)
    encoder._encode_frozen_hidden(["a", "b"], torch.device("cpu"))
    assert len(calls) == calls_before_mutation + 1


def test_history_build_has_no_scan_switch(monkeypatch):
    captured = {}
    class Capture:
        def __init__(self, config, **kwargs):
            captured["config"] = config
    monkeypatch.setattr(turbovla_module, "TurboVLA", Capture)
    build_turbovla(SimpleNamespace(history_length=12))
    assert not hasattr(captured["config"].history, "cuda_scan")


def test_checkpoint_evaluation_accepts_legacy_scan_metadata(monkeypatch, tmp_path):
    import turbovla.evaluation.policy as policy_module

    config = TurboVLAConfig(history=HistoryConfig(enabled=True, length=12))
    checkpoint = tmp_path / "checkpoint.pth"
    payload = asdict(config)
    payload["history"]["cuda_scan"] = True
    torch.save({"model_config": payload}, checkpoint)
    captured = {}

    class Capture(nn.Module):
        def __init__(self, loaded_config, **kwargs):
            super().__init__()
            captured["history"] = loaded_config.history
            self.history_encoder = HistoryEncoder(history_length=12)

    monkeypatch.setattr(turbovla_module, "TurboVLA", Capture)
    monkeypatch.setattr(policy_module, "load_turbovla_builder", lambda: (lambda *_: None, "test"))
    monkeypatch.setattr(policy_module.TurboVLAPolicy, "_load_checkpoint", lambda self: None)
    monkeypatch.setattr(policy_module.TurboVLAPolicy, "_set_eval_precision", lambda self: None)
    monkeypatch.setattr(policy_module.TurboVLAPolicy, "_verify_model_precision", lambda self: None)
    monkeypatch.setattr(policy_module, "build_dinov3_manual_processor", lambda *_: object())
    policy = policy_module.TurboVLAPolicy(
        str(checkpoint), dinov3_path="pretrained/dinov3-vitb16",
        bert_path="pretrained/bert-base-uncased", device="cpu", precision="fp32", verbose=False,
    )
    assert not hasattr(captured["history"], "cuda_scan")
    assert isinstance(policy.model.history_encoder, HistoryEncoder)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scan parity requires CUDA")
def test_matched_cuda_scan_masked_output_and_gradient_parity(monkeypatch):
    torch.manual_seed(7)
    reference = HistoryEncoder(history_length=12).cuda()
    accelerated = HistoryEncoder(history_length=12).cuda()
    accelerated.load_state_dict(reference.state_dict())
    for layer in reference.layers:
        monkeypatch.setattr(layer, "_selective_scan_matched_cuda", layer._selective_scan)
    states = torch.randn(3, 12, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    states_fast = states.detach().clone().requires_grad_(True)
    mask = torch.tensor([[False] * 12, [False] * 5 + [True] * 7, [True] * 12], device="cuda")
    cotangent = torch.randn(3, 12, 256, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = reference(states, mask)
        output_fast = accelerated(states_fast, mask)
    (output * cotangent).sum().backward(); (output_fast * cotangent).sum().backward()
    assert torch.allclose(output.float(), output_fast.float(), atol=3e-2, rtol=3e-2)
    relative_gradient_error = (states.grad.float() - states_fast.grad.float()).norm() / states.grad.float().norm().clamp_min(1e-6)
    assert relative_gradient_error < 1e-2
