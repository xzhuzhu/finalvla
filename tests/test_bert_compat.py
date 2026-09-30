from types import SimpleNamespace

import torch
from torch import nn

from turbovla.text.bert import BertModelWarper


class _Transformers5Bert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.embeddings = SimpleNamespace(word_embeddings=nn.Embedding(8, 4))
        self.encoder = nn.Identity()
        self.pooler = None

    def get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
        return attention_mask.to(dtype=dtype)

    def invert_attention_mask(self, attention_mask):
        return ~attention_mask.bool()


def test_bert_wrapper_supports_transformers5_without_get_head_mask() -> None:
    wrapper = BertModelWarper(_Transformers5Bert())

    assert wrapper._extended_attention_mask_uses_dtype
    assert wrapper.get_head_mask(None, 2) == [None, None]
    expanded = wrapper.get_head_mask(torch.ones(3), 2)
    assert expanded.shape == (2, 1, 3, 1, 1)
