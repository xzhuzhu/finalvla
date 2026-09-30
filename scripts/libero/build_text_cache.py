import argparse
import os

import numpy as np
import torch

from turbovla.text.bert import (
    BertModelWarper,
    generate_masks_with_special_tokens,
)
from turbovla.text import tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a small cached BERT text bank for LIBERO language instructions."
    )
    parser.add_argument(
        "--dataset_dir",
        action="append",
        default=[],
        help="TFDS builder directory. Repeat this option to collect instructions from multiple LIBERO suites.",
    )
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--output", type=str, default="data/libero_all4_bert_text_cache.pt")
    parser.add_argument(
        "--text_encoder_type",
        type=str,
        default="bert-base-uncased",
    )
    parser.add_argument("--max_text_len", type=int, default=256)
    parser.add_argument("--expected_count", type=int, default=40)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--instruction", action="append", default=None)
    parser.add_argument("--instructions_file", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.set_defaults(sub_sentence_present=True)
    parser.add_argument("--sub_sentence_present", dest="sub_sentence_present", action="store_true")
    parser.add_argument("--no_sub_sentence_present", dest="sub_sentence_present", action="store_false")
    return parser.parse_args()


def decode_instruction(raw_instruction):
    if isinstance(raw_instruction, bytes):
        return raw_instruction.decode("utf-8")
    if isinstance(raw_instruction, np.ndarray) and raw_instruction.dtype.type is np.bytes_:
        return raw_instruction.item().decode("utf-8")
    return str(raw_instruction)


def collect_instructions_from_datasets(dataset_dirs, split, expected_count=40, max_episodes=0):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    instructions = []
    seen = set()

    for dataset_dir in dataset_dirs:
        builder = tfds.builder_from_directory(builder_dir=dataset_dir)
        dataset = builder.as_dataset(split=split)
        for episode_idx, episode in enumerate(tfds.as_numpy(dataset), start=1):
            for step in episode["steps"]:
                instruction = decode_instruction(step["language_instruction"])
                if instruction not in seen:
                    seen.add(instruction)
                    instructions.append(instruction)
                    if expected_count > 0 and len(instructions) >= expected_count:
                        return instructions
            if max_episodes > 0 and episode_idx >= max_episodes:
                break

    return instructions


def load_instruction_list(args):
    instructions = []
    if args.instructions_file:
        with open(args.instructions_file, "r", encoding="utf-8") as f:
            instructions.extend(line.strip() for line in f if line.strip())
    if args.instruction:
        instructions.extend(args.instruction)
    if not instructions:
        if not args.dataset_dir:
            raise ValueError("provide at least one --dataset_dir or use --instructions_file")
        instructions = collect_instructions_from_datasets(
            args.dataset_dir,
            args.dataset_split,
            expected_count=args.expected_count,
            max_episodes=args.max_episodes,
        )

    deduped = []
    seen = set()
    for instruction in instructions:
        if instruction not in seen:
            seen.add(instruction)
            deduped.append(instruction)
    if not deduped:
        raise RuntimeError("no instructions found for cache")
    return deduped


def truncate_tokenized(tokenized, text_self_attention_masks, position_ids, max_text_len):
    if text_self_attention_masks.shape[1] <= max_text_len:
        return tokenized, text_self_attention_masks, position_ids

    text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
    position_ids = position_ids[:, :max_text_len]
    for key in list(tokenized.keys()):
        if tokenized[key].ndim >= 2:
            tokenized[key] = tokenized[key][:, :max_text_len]
    return tokenized, text_self_attention_masks, position_ids


def build_cache(instructions, args):
    text_tokenizer = tokenizer.get_tokenlizer(args.text_encoder_type)
    bert = tokenizer.get_pretrained_language_model(args.text_encoder_type)
    if hasattr(bert, "pooler") and bert.pooler is not None and hasattr(bert.pooler, "dense"):
        bert.pooler.dense.weight.requires_grad_(False)
        bert.pooler.dense.bias.requires_grad_(False)

    device = torch.device(args.device)
    bert = BertModelWarper(bert_model=bert).to(device)
    bert.eval()

    tokenized = text_tokenizer(instructions, padding="longest", return_tensors="pt")
    special_tokens = text_tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    text_self_attention_masks, position_ids = generate_masks_with_special_tokens(
        tokenized,
        special_tokens,
        text_tokenizer,
    )
    tokenized, text_self_attention_masks, position_ids = truncate_tokenized(
        tokenized,
        text_self_attention_masks,
        position_ids,
        args.max_text_len,
    )

    if args.sub_sentence_present:
        tokenized_for_encoder = {key: value.to(device) for key, value in tokenized.items() if key != "attention_mask"}
        tokenized_for_encoder["attention_mask"] = text_self_attention_masks.to(device)
        tokenized_for_encoder["position_ids"] = position_ids.to(device)
    else:
        tokenized_for_encoder = {key: value.to(device) for key, value in tokenized.items()}

    with torch.inference_mode():
        bert_output = bert(**tokenized_for_encoder)

    last_hidden_state = bert_output["last_hidden_state"].detach().cpu().contiguous()
    attention_mask = tokenized["attention_mask"].detach().cpu().bool().contiguous()
    text_self_attention_masks = text_self_attention_masks.detach().cpu().bool().contiguous()

    return {
        "format_version": 1,
        "text_encoder_type": args.text_encoder_type,
        "max_text_len": int(args.max_text_len),
        "sub_sentence_present": bool(args.sub_sentence_present),
        "instructions": instructions,
        "instruction_to_index": {instruction: idx for idx, instruction in enumerate(instructions)},
        "last_hidden_state": last_hidden_state,
        "attention_mask": attention_mask,
        "text_self_attention_masks": text_self_attention_masks,
    }


def main():
    args = parse_args()
    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(f"output already exists: {args.output} (use --overwrite to replace it)")

    instructions = load_instruction_list(args)
    cache = build_cache(instructions, args)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(cache, args.output)

    print(f"saved text cache: {args.output}")
    print(f"instructions: {len(instructions)}")
    print(f"last_hidden_state: {tuple(cache['last_hidden_state'].shape)}")
    print(f"attention_mask: {tuple(cache['attention_mask'].shape)}")
    print(f"text_self_attention_masks: {tuple(cache['text_self_attention_masks'].shape)}")


if __name__ == "__main__":
    main()
