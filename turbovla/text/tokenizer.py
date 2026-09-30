import os
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _resolve_text_encoder_type(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif isinstance(text_encoder_type, dict) and text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(f"Unknown type of text_encoder_type: {type(text_encoder_type)}")
    print(f"final text_encoder_type: {text_encoder_type}")
    return text_encoder_type


def get_tokenlizer(text_encoder_type):
    text_encoder_type = _resolve_text_encoder_type(text_encoder_type)
    local_only = os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)
    tokenizer = AutoTokenizer.from_pretrained(
        text_encoder_type,
        local_files_only=local_only,
        use_fast=True,
    )
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    text_encoder_type = _resolve_text_encoder_type(text_encoder_type)
    local_only = os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)

    # 可选：先读 config 做个白名单校验，避免误载别的模型
    config = AutoConfig.from_pretrained(
        text_encoder_type,
        local_files_only=local_only,
        trust_remote_code=False,
    )

    if config.model_type not in {"bert", "roberta"}:
        raise ValueError(
            f"Unsupported text encoder model_type={config.model_type}, "
            f"expected one of {{'bert', 'roberta'}}"
        )
    model = AutoModel.from_pretrained(
        text_encoder_type,
        local_files_only=local_only,
        trust_remote_code=False,
    )
    return model