from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from .configuration import R3MEncoderConfig


class R3MResNet18Encoder(nn.Module):
    """Visual-only R3M ResNet-18 loaded from a compact, locally verified checkpoint."""

    def __init__(self, config: R3MEncoderConfig) -> None:
        super().__init__()
        self.config = config
        checkpoint_path = Path(config.checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"R3M checkpoint not found: {checkpoint_path}")

        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"R3M checkpoint must contain a mapping, got {type(payload)!r}")
        source = payload.get("state_dict", payload.get("r3m", payload))
        if not isinstance(source, dict):
            raise TypeError("R3M checkpoint state_dict must be a mapping")

        expected = backbone.state_dict()
        prefix = "module.convnet."
        state_dict = {}
        for key, value in source.items():
            mapped = key[len(prefix) :] if key.startswith(prefix) else key
            if not mapped.startswith("fc.") and mapped in expected:
                state_dict[mapped] = value
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if not key.startswith("fc.")]
        if missing or unexpected:
            raise RuntimeError(
                f"invalid R3M ResNet-18 checkpoint: missing={missing[:10]}, "
                f"unexpected={unexpected[:10]}"
            )

        self.backbone = backbone
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        if config.frozen:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.frozen:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        expected = (3, self.config.image_size, self.config.image_size)
        if pixel_values.ndim < 4 or tuple(pixel_values.shape[-3:]) != expected:
            raise ValueError(
                f"R3M images must end with {expected}, got {tuple(pixel_values.shape)}"
            )
        leading_shape = pixel_values.shape[:-3]
        flat = pixel_values.reshape(-1, *expected)
        backbone_dtype = next(self.backbone.parameters()).dtype
        flat = flat.to(dtype=backbone_dtype).div(255.0)
        mean = self.image_mean.to(device=flat.device, dtype=flat.dtype)
        std = self.image_std.to(device=flat.device, dtype=flat.dtype)
        flat = (flat - mean) / std

        context = torch.no_grad() if self.config.frozen else torch.enable_grad()
        with context:
            if self.config.frozen and flat.shape[0] > self.config.encode_chunk_size:
                encoded = torch.cat(
                    [
                        self.backbone(flat[start : start + self.config.encode_chunk_size])
                        for start in range(0, flat.shape[0], self.config.encode_chunk_size)
                    ],
                    dim=0,
                )
            else:
                encoded = self.backbone(flat)
        return encoded.reshape(*leading_shape, self.config.output_dim)
