"""Legacy model components kept for the Qwen3-VL variant.

These classes were previously defined in ``turbovla.models.turbovla`` and are
kept here (unchanged) so ``turbovla_qwen3vl`` still imports cleanly.  The
official TurboVLA model in ``turbovla.models.turbovla`` does not use them.
"""

from __future__ import annotations

import torch
from torch import nn

from .utils import MLP


class ACTActionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nheads,
        action_dim,
        chunk_size=8,
        num_layers=3,
        dim_feedforward=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_queries = nn.Embedding(self.chunk_size, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = MLP(hidden_dim, 512, action_dim, 3)

    def forward(self, memory):
        batch_size = memory.shape[0]
        tgt = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = self.decoder(tgt=tgt, memory=memory)
        pred_actions = torch.tanh(self.action_head(hidden_states))
        return pred_actions, hidden_states


class VisionProjector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=768, hidden_dim=1536, dropout=0.1):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.norm_out(self.skip(x) + self.mlp(self.norm_in(x)))


class StateProjector(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=768, num_tokens=2, proj_hidden=256, dropout=0.1):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, proj_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, self.num_tokens * self.hidden_dim),
        )
        self.pos = nn.Parameter(torch.randn(1, self.num_tokens, self.hidden_dim) * 0.02)
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, state):
        x = self.net(state)
        x = x.view(state.shape[0], self.num_tokens, self.hidden_dim)
        x = self.out_norm(x + self.pos)
        return x


__all__ = ["ACTActionDecoder", "VisionProjector", "StateProjector"]
