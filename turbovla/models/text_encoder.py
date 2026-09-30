from __future__ import annotations

from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from typing import Sequence

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from ..text.bert import BertModelWarper, generate_masks_with_special_tokens
from .configuration import TextEncoderConfig


def _load_pretrained_model(config: TextEncoderConfig):
    kwargs = {
        "local_files_only": config.local_files_only,
        "trust_remote_code": False,
    }
    if config.attention_implementation:
        try:
            return AutoModel.from_pretrained(
                config.model_name_or_path,
                attn_implementation=config.attention_implementation,
                **kwargs,
            )
        except Exception as error:
            if config.attention_implementation == "flash_attention_2":
                try:
                    return AutoModel.from_pretrained(
                        config.model_name_or_path,
                        attn_implementation="sdpa",
                        **kwargs,
                    )
                except Exception:
                    pass
            print(
                f"[TurboVLA] text attention backend {config.attention_implementation!r} unavailable; "
                f"using model default ({type(error).__name__}).",
                flush=True,
            )
    return AutoModel.from_pretrained(config.model_name_or_path, **kwargs)


class TurboVLATextEncoder(nn.Module):
    """BERT encoder with separate evaluation and frozen-training hidden caches."""

    EVAL_CACHE_MAX_SIZE = 128
    TRAIN_HIDDEN_CACHE_MAX_SIZE = 256

    def __init__(self, config: TextEncoderConfig, hidden_dim: int) -> None:
        super().__init__()
        self.config = config
        self._eval_cache: OrderedDict[
            tuple[str, str, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = OrderedDict()
        # Raw BERT hidden states, never projected outputs: the projection stays
        # trainable and receives gradients on every training iteration.
        self._train_hidden_cache: OrderedDict[
            tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self.use_frozen_training_cache = bool(
            getattr(config, "frozen_training_cache", True)
        )
        self._frozen_cache_bert_signature: tuple[int, ...] | None = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            local_files_only=config.local_files_only,
            use_fast=True,
        )
        bert = _load_pretrained_model(config)
        self.bert = BertModelWarper(bert_model=bert)
        self.text_projection = nn.Linear(self.bert.config.hidden_size, hidden_dim, bias=True)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.constant_(self.text_projection.bias, 0.0)
        self.special_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        if config.frozen:
            self.bert.requires_grad_(False)
            if config.force_eval_when_frozen:
                self.bert.eval()

    def train(self, mode: bool = True):
        if mode:
            self.clear_eval_cache()
        super().train(mode)
        if self.config.frozen and self.config.force_eval_when_frozen:
            self.bert.eval()
        return self

    def clear_eval_cache(self) -> None:
        self._eval_cache.clear()

    def clear_frozen_training_cache(self) -> None:
        self._train_hidden_cache.clear()
        self._frozen_cache_bert_signature = None

    def _apply(self, fn):
        result = super()._apply(fn)
        self.clear_eval_cache()
        self.clear_frozen_training_cache()
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_eval_cache()
        self.clear_frozen_training_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    @property
    def eval_cache_size(self) -> int:
        return len(self._eval_cache)

    def _tokenize_group(
        self,
        instructions: Sequence[str],
        device: torch.device,
        padding_length: int | None,
    ):
        padding = "max_length" if padding_length is not None else "longest"
        effective_max_length = padding_length or self.config.max_length
        tokenized = self.tokenizer(
            [str(item) for item in instructions],
            padding=padding,
            truncation=True,
            max_length=effective_max_length,
            return_tensors="pt",
        ).to(device)
        text_self_attention_masks, position_ids = generate_masks_with_special_tokens(
            tokenized,
            self.special_tokens,
            self.tokenizer,
        )
        return tokenized, text_self_attention_masks, position_ids

    def _encode_group(
        self,
        instructions: Sequence[str],
        device: torch.device,
        padding_length: int | None,
    ):
        tokenized, text_self_attention_masks, position_ids = self._tokenize_group(
            instructions,
            device,
            padding_length,
        )
        if self.config.sub_sentence_present:
            bert_inputs = {key: value for key, value in tokenized.items() if key != "attention_mask"}
            bert_inputs["attention_mask"] = text_self_attention_masks
            bert_inputs["position_ids"] = position_ids
        else:
            bert_inputs = tokenized

        grad_context = torch.no_grad() if self.config.frozen else nullcontext()
        with grad_context:
            bert_output = self.bert(**bert_inputs)

        return (
            bert_output.last_hidden_state,
            tokenized.attention_mask.bool(),
            text_self_attention_masks,
        )

    def encode_bert_hidden(self, instructions: Sequence[str], device: torch.device):
        if not instructions:
            raise ValueError("instructions cannot be empty")
        normalized = [str(item) for item in instructions]
        output_length = self.config.padding_length
        layout = self.config.padding_length_by_instruction
        if not layout:
            return self._encode_group(normalized, device, output_length)

        if output_length is None:
            raise ValueError("text.padding_length is required when instruction-specific lengths are configured")
        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for index, instruction in enumerate(normalized):
            grouped_indices[int(layout.get(instruction, output_length))].append(index)

        hidden = None
        attention_mask = torch.zeros((len(normalized), output_length), device=device, dtype=torch.bool)
        self_attention = torch.eye(output_length, device=device, dtype=torch.bool).expand(
            len(normalized), -1, -1
        ).clone()
        for group_length, indices in grouped_indices.items():
            if group_length > output_length:
                raise ValueError(f"instruction padding length {group_length} exceeds output length {output_length}")
            group_instructions = [normalized[index] for index in indices]
            group_hidden, group_attention, group_self_attention = self._encode_group(
                group_instructions,
                device,
                group_length,
            )
            if hidden is None:
                hidden = group_hidden.new_zeros((len(normalized), output_length, group_hidden.shape[-1]))
            hidden[indices, :group_length] = group_hidden
            attention_mask[indices, :group_length] = group_attention
            self_attention[indices, :group_length, :group_length] = group_self_attention

        return hidden, attention_mask, self_attention

    def _hidden_cache_key(self, instruction: str, device: torch.device, effective_length: int) -> tuple:
        # Include all tokenization/mask-affecting configuration on every lookup,
        # so a mutable config cannot return stale layout tensors.
        layout = tuple(sorted((str(key), int(value)) for key, value in self.config.padding_length_by_instruction.items()))
        bert_dtype = next(self.bert.parameters()).dtype
        return (
            str(instruction), str(device), bert_dtype,
            self.config.max_length, self.config.padding_length, effective_length, layout,
            self.config.sub_sentence_present,
            torch.get_autocast_dtype("cuda") if device.type == "cuda" and torch.is_autocast_enabled("cuda") else None,
        )

    def _encode_frozen_hidden(self, instructions: Sequence[str], device: torch.device):
        use_cache = (self.use_frozen_training_cache and self.config.frozen and not self.bert.training
                     and self.config.padding_length is not None and not torch.is_inference_mode_enabled())
        if not use_cache:
            return self.encode_bert_hidden(instructions, device)
        # Child-module load_state_dict and direct in-place weight edits do not
        # call this module's loader. Tensor versions make either event a hard
        # cache invalidation while frozen BERT remains the only cacheable case.
        bert_signature = tuple(parameter._version for parameter in self.bert.parameters())
        if self._frozen_cache_bert_signature != bert_signature:
            self.clear_frozen_training_cache()
            self._frozen_cache_bert_signature = bert_signature
        normalized = [str(item) for item in instructions]
        output_length = int(self.config.padding_length)
        layout = self.config.padding_length_by_instruction
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, instruction in enumerate(normalized):
            grouped[int(layout.get(instruction, output_length))].append(index)
        hidden = attention = self_attention = None
        for group_length, indices in grouped.items():
            if group_length > output_length:
                raise ValueError(f"instruction padding length {group_length} exceeds output length {output_length}")
            rows: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            misses = []
            for index in indices:
                key = self._hidden_cache_key(normalized[index], device, group_length)
                cached = self._train_hidden_cache.get(key)
                if cached is None:
                    misses.append(index)
                else:
                    self._train_hidden_cache.move_to_end(key)
                    rows[index] = cached
            if misses:
                encoded = self._encode_group([normalized[index] for index in misses], device, group_length)
                for row_index, index in enumerate(misses):
                    # Clone: indexed rows are views of the whole BERT batch and
                    # retaining them would defeat the bounded-cache contract.
                    entry = tuple(tensor[row_index].detach().clone() for tensor in encoded)
                    key = self._hidden_cache_key(normalized[index], device, group_length)
                    self._train_hidden_cache[key] = entry
                    self._train_hidden_cache.move_to_end(key)
                    rows[index] = entry
            if hidden is None:
                sample = rows[indices[0]][0]
                hidden = sample.new_zeros((len(normalized), output_length, sample.shape[-1]))
                attention = torch.zeros((len(normalized), output_length), device=device, dtype=torch.bool)
                self_attention = torch.eye(output_length, device=device, dtype=torch.bool).expand(len(normalized), -1, -1).clone()
            for index in indices:
                row_hidden, row_attention, row_self_attention = rows[index]
                hidden[index, :group_length] = row_hidden
                attention[index, :group_length] = row_attention
                self_attention[index, :group_length, :group_length] = row_self_attention
        while len(self._train_hidden_cache) > self.TRAIN_HIDDEN_CACHE_MAX_SIZE:
            self._train_hidden_cache.popitem(last=False)
        return hidden, attention, self_attention

    def _forward_uncached(self, instructions: Sequence[str], device: torch.device):
        hidden, text_token_mask, text_self_attention_masks = self._encode_frozen_hidden(instructions, device)

        hidden = hidden.to(dtype=self.text_projection.weight.dtype)
        text_tokens = self.text_projection(hidden)
        text_key_padding_mask = ~text_token_mask
        if self.config.zero_padded_tokens:
            text_tokens = text_tokens.masked_fill(text_key_padding_mask.unsqueeze(-1), 0.0)
        return text_tokens, text_key_padding_mask, text_self_attention_masks

    def forward(self, instructions: Sequence[str], device: torch.device):
        use_cache = not self.training and not torch.is_grad_enabled() and len(instructions) == 1
        if not use_cache:
            return self._forward_uncached(instructions, device)

        key = (str(instructions[0]), str(device), self.text_projection.weight.dtype)
        cached = self._eval_cache.get(key)
        if cached is not None:
            self._eval_cache.move_to_end(key)
            return cached

        encoded = tuple(tensor.detach() for tensor in self._forward_uncached(instructions, device))
        self._eval_cache[key] = encoded
        self._eval_cache.move_to_end(key)
        while len(self._eval_cache) > self.EVAL_CACHE_MAX_SIZE:
            self._eval_cache.popitem(last=False)
        return encoded
