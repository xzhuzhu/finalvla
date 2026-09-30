# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy
from torch.distributions import Beta

from fluxvla.engines import HEADS
from fluxvla.engines.losses import reduce_action_bc_loss
from fluxvla.models.blocks import SelfAttentionTransformer
from fluxvla.models.blocks.cross_attention_dit import DiT



class BijectionNet(nn.Sequential):
	"""
	A sequential container of flows based on coupling layers.
	"""
	def __init__(self, num_dims, num_blocks, num_hidden):
		self.num_dims = num_dims
		modules = []
		mask = torch.arange(0, num_dims) % 2  # alternating inputs
		mask = mask.float()
		# mask = mask.to(device).float()
		for _ in range(num_blocks):
			modules += [
				CouplingLayer(
					num_inputs=num_dims, num_hidden=num_hidden, mask=mask),
			]
			mask = 1 - mask  # flipping mask
		super(BijectionNet, self).__init__(*modules)



	def forward(self, inputs, mode='direct'):
		""" Performs a forward or backward pass for flow modules.
		Args:
			inputs: a tuple of inputs and logdets
			mode: to run direct computation or inverse
		"""
		assert mode in ['direct', 'inverse']
		batch_size = inputs.size(0)


		if mode == 'direct':
			for module in self._modules.values():

				inputs = module(inputs, mode)
		else:
			for module in reversed(self._modules.values()):

				inputs = module(inputs, mode)
		return inputs


class CouplingLayer(nn.Module):


	def __init__(self, num_inputs, num_hidden, mask,
				  s_act='elu', t_act='elu'):
		super(CouplingLayer, self).__init__()

		self.num_inputs = num_inputs
		self.register_buffer('mask', mask)

		
		self.scale_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=s_act)
		self.translate_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=t_act)
		
		nn.init.zeros_(self.translate_net.network[-1].weight.data)
		nn.init.zeros_(self.translate_net.network[-1].bias.data)

		nn.init.zeros_(self.scale_net.network[-1].weight.data)
		nn.init.zeros_(self.scale_net.network[-1].bias.data)

		

	def forward(self, inputs, mode='direct'):
		mask = self.mask
		masked_inputs = inputs * mask
		# masked_inputs.requires_grad_(True)

	#	log_s = self.scale_net(masked_inputs) * (1 - mask)
		t = self.translate_net(masked_inputs) * (1 - mask)
      			#s = torch.exp(log_s)
			#return inputs * s + t
                     		#	s = torch.exp(-log_s)
		#	return (inputs - t) * s

		if mode == 'direct':return inputs  + t;

		else:return inputs - t







class FCNN(nn.Module):
	'''
	2-layer fully connected neural network
	'''

	def __init__(self, in_dim, out_dim, hidden_dim, act='tanh'):
		super(FCNN, self).__init__()
		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
					   'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}

		act_func = activations[act]
		self.network = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), nn.ELU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
			nn.Linear(hidden_dim, out_dim)
		)

	def forward(self, x):
		return self.network(x)

def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(
            half_dim, dtype=torch.float, device=device) * (
                torch.log(torch.tensor(10000.0)) / half_dim)
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class MLP(nn.Module):
    """Shared MLP used for state encoding and action decoding."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    """Shared action encoder for DiT."""

    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W1 = nn.Linear(action_dim, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) or (B, T) -- per-sample or per-position
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Accept (B,) or (B, T) timesteps
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        elif timesteps.dim() == 2:
            assert timesteps.shape == (B, T)
        else:
            raise ValueError(f'Expected timesteps shape (B,) or (B, T), got '
                             f'{timesteps.shape}')

        a_emb = self.W1(actions)

        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x))

        return self.W3(x)


@HEADS.register_module()
class FlowMatching7DHead(nn.Module):
    """
    FlowMatching7DHead for DiT, operating directly in 7D action space.

    Args:
        hidden_size: The dimension of the hidden states.
        state_dim: The dimension of the state.
        input_embedding_dim: The dimension of the input embedding.
        action_dim: The dimension of the action.
        use_vlln: Whether to use VLLN.
        num_target_vision_tokens: Number of future query tokens generated from
            visual-language tokens when static future tokens are disabled.
        num_static_future_tokens: Number of StarVLA-style learnable future
            tokens prepended before action tokens in the DiT sequence.
        num_state_future_tokens: Number of future tokens generated from the
            current robot state.
        static_future_token_dropout: Dropout applied only to the learnable
            static future tokens during training.
        backbone_embedding_dim: The dimension of the backbone embedding.
        vl_self_attention_cfg: The configuration for the VL self-attention.
        add_positional_embeddings: Whether to add positional embeddings.
        max_seq_len: The maximum sequence length.
        num_timestep_buckets: The number of timestep buckets.
        noise_s: The noise scale.
        noise_beta_alpha: The alpha for the noise beta distribution.
        noise_beta_beta: The beta for the noise beta distribution.
        num_steps: The number of steps.
        diffusion_model_cfg: The configuration for the diffusion model.
    """

    def __init__(self,
                 hidden_size: int,
                 state_dim: int,
                 input_embedding_dim: int,
                 action_dim: int,
                 use_vlln: bool = True,
                 num_target_vision_tokens: int = 32,
                 num_static_future_tokens: int = 0,
                 num_state_future_tokens: int = 0,
                 static_future_token_dropout: float = 0.0,
                 backbone_embedding_dim: int = 2048,
                 vl_self_attention_cfg: Dict = dict(
                     attention_head_dim=64,
                     dropout=0.2,
                     final_dropout=True,
                     num_attention_heads=32,
                     num_layers=4,
                     positional_embeddings=None),
                 add_positional_embeddings: bool = True,
	                 max_seq_len: int = 1024,
	                 num_timestep_buckets: int = 1000,
	                 noise_s: float = 0.999,
	                 noise_beta_alpha: float = 1.5,
	                 noise_beta_beta: float = 1.0,
	                 num_steps: int = None,
                 traj_length: int = None,
	                 diffusion_model_cfg: Dict = dict(
                     attention_head_dim=48,
                     cross_attention_dim=2048,
                     dropout=0.2,
                     final_dropout=True,
                     interleave_self_attention=True,
                     norm_type='ada_norm',
                     num_attention_heads=32,
                     num_layers=16,
                     output_dim=1024,
                     positional_embeddings=None),
                 use_state_condition: bool = True,
                 rtc_training_config=None,
                 *args,
                 **kwargs):
        super().__init__()
        if action_dim != 7:
            raise ValueError(
                f'FlowMatching7DHead requires action_dim=7, got {action_dim}')
        self.rtc_training_config = rtc_training_config
        self.hidden_size = hidden_size
        self.use_state_condition = use_state_condition
        self.state_encoder = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_size,
            output_dim=input_embedding_dim,
        )
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_size=input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=action_dim,
        )
        self.model = DiT(**diffusion_model_cfg)
        self.input_embedding_dim = input_embedding_dim
        self.action_dim = action_dim
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
        self.num_timestep_buckets = num_timestep_buckets
        self.noise_s = noise_s
        self.add_positional_embeddings = add_positional_embeddings
        self.num_steps = num_steps if num_steps is not None else (
            traj_length if traj_length is not None else 10)
        self.num_static_future_tokens = num_static_future_tokens
        self.num_state_future_tokens = num_state_future_tokens
        self.static_future_token_dropout = static_future_token_dropout
        if num_state_future_tokens > 0 and not use_state_condition:
            raise ValueError('num_state_future_tokens requires '
                             'use_state_condition=True')

        self.vlln = (
            nn.LayerNorm(backbone_embedding_dim)
            if use_vlln else nn.Identity())
        self.vl_self_attention = (
            SelfAttentionTransformer(
                **vl_self_attention_cfg) if use_vlln else nn.Identity())
        if num_static_future_tokens > 0:
            self.static_future_tokens = nn.Embedding(
                num_static_future_tokens, self.input_embedding_dim)
            nn.init.normal_(
                self.static_future_tokens.weight, mean=0.0, std=0.02)
        else:
            self.static_future_tokens = None
        self.future_queries = nn.Embedding(num_target_vision_tokens,
                                           self.input_embedding_dim)
        self.future_cross_attention = nn.MultiheadAttention(
            embed_dim=self.input_embedding_dim,
            num_heads=max(1, self.input_embedding_dim // 64),
            batch_first=True,
        )
        nn.init.normal_(self.future_queries.weight, mean=0.0, std=0.02)

        if num_state_future_tokens > 0:
            self.state_future_queries = nn.Embedding(
                num_state_future_tokens, self.input_embedding_dim)
            self.state_future_key = nn.Sequential(
                nn.LayerNorm(self.input_embedding_dim),
                nn.Linear(
                    self.input_embedding_dim,
                    num_state_future_tokens * self.input_embedding_dim),
            )
            self.state_future_value_attention = nn.MultiheadAttention(
                embed_dim=self.input_embedding_dim,
                num_heads=max(1, self.input_embedding_dim // 64),
                batch_first=True,
            )
            self.state_future_cross_attention = nn.MultiheadAttention(
                embed_dim=self.input_embedding_dim,
                num_heads=max(1, self.input_embedding_dim // 64),
                batch_first=True,
            )
            nn.init.normal_(
                self.state_future_queries.weight, mean=0.0, std=0.02)
        else:
            self.state_future_queries = None
            self.state_future_key = None
            self.state_future_value_attention = None
            self.state_future_cross_attention = None

        if add_positional_embeddings:
            self.position_embedding = nn.Embedding(max_seq_len,
                                                   self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.flow = BijectionNet(self.action_dim,12,256)

    def _padding_mask_from_attention(
            self, attention_mask: Optional[torch.Tensor],
            seq_len: int) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask[:, :seq_len].to(dtype=torch.bool)
        if attention_mask.dim() == 4:
            # Backward compatibility for older Qwen3VL.forward() callers that
            # passed the LM causal mask instead of the padding mask.
            if attention_mask.dtype == torch.bool:
                mask = attention_mask.any(dim=-1)
            else:
                mask = torch.isfinite(attention_mask).any(dim=-1)
            return mask.squeeze(1)[:, :seq_len].to(dtype=torch.bool)
        raise ValueError(
            'attention_mask must be 2D padding mask or 4D LM mask, got '
            f'{tuple(attention_mask.shape)}')

    def _prepare_vl_inputs(self, input_features: torch.Tensor,
                           attention_mask: Optional[torch.Tensor]):
        valid_mask = self._padding_mask_from_attention(
            attention_mask, input_features.shape[1])
        if valid_mask is None:
            return input_features, None

        valid_lengths = valid_mask.sum(dim=1)
        if torch.any(valid_lengths == 0):
            raise ValueError('Each sample must provide at least one VL token')

        max_valid = int(valid_lengths.max().item())
        compact_features = input_features.new_zeros(
            input_features.shape[0], max_valid, input_features.shape[2])
        compact_mask = torch.zeros(
            input_features.shape[0],
            max_valid,
            dtype=torch.bool,
            device=input_features.device)
        for batch_idx in range(input_features.shape[0]):
            keep = valid_mask[batch_idx]
            length = int(valid_lengths[batch_idx].item())
            compact_features[batch_idx, :length] = input_features[
                batch_idx, keep]
            compact_mask[batch_idx, :length] = True
        return compact_features, compact_mask

    def _encode_condition(self, input_features: torch.Tensor,
                          attention_mask: Optional[torch.Tensor]):
        input_features, attention_mask = self._prepare_vl_inputs(
            input_features, attention_mask)
        input_features = self.vlln(input_features)
        input_features = self.vl_self_attention(input_features)
        return input_features, attention_mask

    def _build_state_future_tokens(
            self,
            input_features: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            state_features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.state_future_queries is None:
            return None
        if state_features is None:
            raise ValueError('state_features must be provided when '
                             'num_state_future_tokens > 0')

        batch_size = state_features.shape[0]
        state_context = state_features[:, 0]
        state_keys = self.state_future_key(state_context)
        state_keys = state_keys.view(batch_size, self.num_state_future_tokens,
                                     self.input_embedding_dim)
        queries = self.state_future_queries.weight.unsqueeze(0).expand(
            batch_size, -1, -1)
        queries = queries.to(device=state_features.device,
                             dtype=state_features.dtype)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(dtype=torch.bool)
        vl_values, _ = self.state_future_value_attention(
            queries,
            input_features,
            input_features,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        state_future_tokens, _ = self.state_future_cross_attention(
            queries,
            state_keys,
            vl_values,
            need_weights=False,
        )
        return state_future_tokens

    def _build_future_tokens(self, input_features: torch.Tensor,
                             attention_mask: Optional[torch.Tensor],
                             state_features: Optional[torch.Tensor] = None):
        future_token_parts = []
        if self.static_future_tokens is not None:
            static_future_tokens = self.static_future_tokens.weight.unsqueeze(
                0).expand(input_features.shape[0], -1, -1)
            static_future_tokens = static_future_tokens.to(
                device=input_features.device, dtype=input_features.dtype)
            future_token_parts.append(
                F.dropout(static_future_tokens,
                          p=self.static_future_token_dropout,
                          training=self.training))

        queries = self.future_queries.weight.unsqueeze(0).expand(
            input_features.shape[0], -1, -1)
        queries = queries.to(device=input_features.device,
                             dtype=input_features.dtype)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.to(dtype=torch.bool)
        future_tokens, _ = self.future_cross_attention(
            queries,
            input_features,
            input_features,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        future_token_parts.append(future_tokens)
        state_future_tokens = self._build_state_future_tokens(
            input_features, attention_mask, state_features)
        if state_future_tokens is not None:
            future_token_parts.append(state_future_tokens)
        return torch.cat(future_token_parts, dim=1)

    def _encode_state_condition(self, states: torch.Tensor) -> torch.Tensor:
        if not self.use_state_condition:
            return None
        if states is None:
            raise ValueError(
                'states must be provided when use_state_condition=True')
        return self.state_encoder(states.unsqueeze(1))

    def _build_dit_inputs(self, action_features: torch.Tensor,
                          state_features: torch.Tensor,
                          future_tokens: torch.Tensor) -> torch.Tensor:
        tokens = [action_features]
        if future_tokens is not None:
            tokens.insert(0, future_tokens)
        if state_features is not None:
            tokens.insert(0, state_features)
        return torch.cat(tokens, dim=1)

    def forward(self, input_features: torch.Tensor, states: torch.Tensor,
                attention_mask: torch.Tensor,
                actions: torch.Tensor, action_masks: torch.Tensor, **kwargs):
        input_features, attention_mask = self._encode_condition(
            input_features, attention_mask)
        state_features = self._encode_state_condition(states)
        future_tokens = self._build_future_tokens(input_features,
                                                  attention_mask,
                                                  state_features)
        noise = torch.randn(
            actions.shape, device=actions.device, dtype=actions.dtype)
        t_scalar = self.sample_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype)
        T = actions.shape[1]
        if (self.rtc_training_config
                and self.rtc_training_config.get('enabled', False)):
            from fluxvla.engines.utils.rtc_training import (
                apply_rtc_time_conditioning, sample_training_delay)
            delays = sample_training_delay(
                batch_size=actions.shape[0],
                max_delay=self.rtc_training_config.get('max_delay', 5),
                distribution=self.rtc_training_config.get(
                    'distribution', 'exponential'),
                temperature=self.rtc_training_config.get('temperature', 1.0),
                device=actions.device)
            t, action_masks = apply_rtc_time_conditioning(
                t_scalar, action_masks, delays, T)  # (B, T)
        else:
            t = t_scalar.unsqueeze(1).expand(-1, T)  # (B, T)

        t_encoder = (t * self.num_timestep_buckets).long()  # (B, T)
        noisy_trajectory = (
            1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * actions
        velocity = actions - noise
        action_features = self.action_encoder(noisy_trajectory, t_encoder)

        if self.add_positional_embeddings:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=actions.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = self._build_dit_inputs(action_features, state_features,
                                         future_tokens)
        t_global = (t_scalar * self.num_timestep_buckets).long()
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=input_features,
            encoder_attention_mask=attention_mask,
            timestep=t_global,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1]:]

        goal_pr = noisy_trajectory + (1 - t.unsqueeze(-1)) * pred_actions
        z = self.flow.forward(goal_pr, mode='direct')
        z_norm = F.normalize(z, dim=-1)
        y_back = self.flow(z_norm, mode='inverse')

        with torch.no_grad():
            z_goal = self.flow.forward(actions, mode='direct')

        # The flow may operate in an expanded latent space.  Reduce the
        # latent alignment term before combining it with per-action losses so
        # latent_dim is not required to equal action_dim.
        latent_alignment_loss = F.mse_loss(
            z_goal, z_norm, reduction='none').mean(dim=-1, keepdim=True)
        losses = (0.4 * F.mse_loss(pred_actions, velocity, reduction='none') +
                  F.mse_loss(y_back, actions, reduction='none') +
                  latent_alignment_loss)
        loss = reduce_action_bc_loss(
            losses,
            action_mask=action_masks) * (
                velocity.shape[-1] / actions.shape[-1])

        return dict(
            pred_actions=pred_actions,
            loss=loss,
        )

    def denoise_step(self,
                     actions,
                     input_features,
                     state_features,
                     attention_mask,
                     t_global,
                     t_encoder=None):
        """Single denoising step extracted from predict_action loop.

        Args:
            actions: Current noisy actions (B, T, D).
            input_features: Processed VL features (B, S, H).
            state_features: Encoded state (B, 1, H).
            attention_mask: VL attention mask.
            t_global: Discretized timestep (B,) for DiT AdaLN.
            t_encoder: Per-position timestep (B, T) for action
                encoder. If None, uses t_global.

        Returns:
            Predicted velocity (B, T, D).
        """
        t_enc = t_encoder if t_encoder is not None else t_global
        action_features = self.action_encoder(actions, t_enc)
        if self.add_positional_embeddings:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=actions.device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        future_tokens = self._build_future_tokens(input_features,
                                                  attention_mask,
                                                  state_features)
        sa_embs = self._build_dit_inputs(action_features, state_features,
                                         future_tokens)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=input_features,
            encoder_attention_mask=attention_mask,
            timestep=t_global,
        )
        pred = self.action_decoder(model_output)
        return pred[:, -self.num_steps:]

    def _predict_action_plain(self, actions, denoise, batch_size, device, dt):
        t_global = torch.zeros((batch_size, ), device=device, dtype=torch.long)
        v = denoise(actions, t_global)
        actions = actions + v
        z = self.flow.forward(actions, mode='direct')
        z_norm = F.normalize(z, dim=-1)
        return self.flow(z_norm, mode='inverse')


    def predict_action(self,
                       input_features: torch.Tensor,
                       states: torch.Tensor,
                       attention_mask: torch.Tensor,
                       prev_actions=None,
                       prefix_len: int = 0,
                       rtc_config: dict = None):
        device = input_features.device
        input_features, attention_mask = self._encode_condition(
            input_features, attention_mask)
        batch_size = input_features.shape[0]
        state_features = self._encode_state_condition(states)
        actions = torch.randn(
            size=(batch_size, self.num_steps, self.action_dim),
            dtype=input_features.dtype,
            device=input_features.device,
        )
        def denoise(x, t_global, t_encoder=None):
            return self.denoise_step(x, input_features, state_features,
                                     attention_mask, t_global, t_encoder)


        actions = self._predict_action_plain(
            actions=actions,
            denoise=denoise,
            batch_size=batch_size,
            device=device,
            dt=1.0)

        return actions

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        Returns a function used to determine which modules to wrap with FSDP.
        """
        return partial(
            _module_wrap_policy,
            module_classes=set([SelfAttentionTransformer, DiT]),
        )
