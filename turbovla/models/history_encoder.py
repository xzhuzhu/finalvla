"""Causal Mamba encoder for short proprioception histories."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .short_scan_cuda import short_scan_cuda


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = value.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (value * scale.to(dtype=value.dtype)) * self.weight.to(dtype=value.dtype)


class MambaBlock(nn.Module):
    """Mamba-1 selective SSM block implemented with a short causal scan.

    The supported histories are short, so the dependency-free scan avoids
    custom CUDA kernels without creating a meaningful runtime cost.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        state_size: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        inner_dim = int(expand) * int(hidden_dim)
        dt_rank = int(dt_rank or math.ceil(hidden_dim / 16))
        self.hidden_dim = int(hidden_dim)
        self.inner_dim = inner_dim
        self.state_size = int(state_size)
        self.conv_kernel = int(conv_kernel)

        self.norm = RMSNorm(hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, 2 * inner_dim, bias=False)
        self.conv1d = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_kernel,
            groups=inner_dim,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(inner_dim, dt_rank + 2 * state_size, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner_dim, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_size + 1, dtype=torch.float32)).repeat(inner_dim, 1))
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, hidden_dim, bias=False)
        dt = torch.exp(
            torch.rand(inner_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp_min(1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _selective_scan(self, value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        parameters = self.x_proj(value)
        dt_rank = parameters.shape[-1] - 2 * self.state_size
        dt_raw, B, C = torch.split(parameters, [dt_rank, self.state_size, self.state_size], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log.float()).to(dtype=value.dtype)
        state = value.new_zeros(value.shape[0], self.inner_dim, self.state_size)
        outputs = []
        for index in range(value.shape[1]):
            delta_t = delta[:, index]
            transition = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            candidate = transition * state
            candidate = candidate + delta_t.unsqueeze(-1) * B[:, index].unsqueeze(1) * value[:, index].unsqueeze(-1)
            valid = valid_mask[:, index].view(-1, 1, 1)
            state = torch.where(valid, candidate, state)
            output = (state * C[:, index].unsqueeze(1)).sum(dim=-1)
            output = output + self.D.to(dtype=value.dtype) * value[:, index]
            outputs.append(output * valid_mask[:, index : index + 1].to(dtype=value.dtype))
        return torch.stack(outputs, dim=1)

    def _selective_scan_matched_cuda(self, value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        parameters = self.x_proj(value)
        dt_rank = parameters.shape[-1] - 2 * self.state_size
        dt_raw, B, C = torch.split(parameters, [dt_rank, self.state_size, self.state_size], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw)).to(dtype=value.dtype)
        A = -torch.exp(self.A_log.float()).to(dtype=value.dtype)
        return short_scan_cuda(value, delta, A, B, C, self.D.to(dtype=value.dtype), valid_mask)

    def forward(self, value: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        residual = value
        value, gate = self.in_proj(self.norm(value)).chunk(2, dim=-1)
        value = self.conv1d(value.transpose(1, 2))[..., : value.shape[1]].transpose(1, 2)
        value = F.silu(value)
        if value.is_cuda:
            if value.dtype != torch.bfloat16:
                raise RuntimeError("matched CUDA scan requires BF16 history tensors")
            value = self._selective_scan_matched_cuda(value, valid_mask)
        else:
            value = self._selective_scan(value, valid_mask)
        value = self.out_proj(value * F.silu(gate))
        output = residual + value
        return output.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class HistoryEncoder(nn.Module):
    """Encode a supported sequence of preceding 8D robot states."""

    def __init__(
        self,
        state_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        encoder_type: str = "mamba",
        history_length: int = 4,
    ) -> None:
        super().__init__()
        required = (state_dim, hidden_dim, num_layers, encoder_type)
        if required != (8, 256, 2, "mamba") or history_length not in {4, 12}:
            raise ValueError(
                "history encoder requires state_dim=8, hidden_dim=256, "
                "num_layers=2, history_length in {4, 12}, encoder_type='mamba'"
            )
        if float(dropout) != 0.0:
            raise ValueError("history encoder does not use dropout")
        self.state_dim = 8
        self.hidden_dim = 256
        self.history_length = int(history_length)
        self.input_projection = nn.Linear(8, 256)
        self.layers = nn.ModuleList([MambaBlock(256) for _ in range(2)])
        self.output_norm = RMSNorm(256)

    def forward(self, states: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        expected_states = (states.shape[0], self.history_length, self.state_dim)
        expected_mask = (states.shape[0], self.history_length)
        if tuple(states.shape) != expected_states:
            raise ValueError(f"history_states must have shape {expected_states}, got {tuple(states.shape)}")
        if tuple(valid_mask.shape) != expected_mask:
            raise ValueError(f"history_mask must have shape {expected_mask}, got {tuple(valid_mask.shape)}")
        valid_mask = valid_mask.to(device=states.device, dtype=torch.bool)
        value = self.input_projection(states)
        value = value.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        for layer in self.layers:
            value = layer(value, valid_mask)
        value = self.output_norm(value)
        return value.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
