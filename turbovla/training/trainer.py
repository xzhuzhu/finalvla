import argparse
import glob
import json
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.libero_rlds import (
    LiberoRLDSDataset,
    Qwen3VLCollator,
    vla_collate_fn,
)
from ..models.turbovla import (
    build_turbovla,
)
from ..models.flow_checkpoint import assert_flow_checkpoint_compatible


class DummyArgs:
    pass


def parse_args():
    """Runtime settings for the fixed V15 matched-CUDA architecture."""
    parser = argparse.ArgumentParser(description="Train FinalVLA on four LIBERO suites")
    for name in ("dinov3_path", "bert_path", "r3m_path"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--r3m_feature_cache_path", default="")
    parser.add_argument("--checkpoint_dir", default="outputs/finalvla")
    parser.add_argument("--checkpoint_prefix", default="finalvla_step")
    parser.add_argument("--resume_mode", choices=("none", "model", "all"), default="none")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--dinov3_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--head_weight_decay", type=float, default=1e-10)
    parser.add_argument("--dinov3_weight_decay", type=float, default=1e-10)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--lr_schedule_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=12500)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--log_freq", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--shuffle_buffer", type=int, default=512)
    parser.add_argument("--step_mix_buffer_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(
        dataset_dir="", dataset_split="train", backbone="dinov3", qwen3vl_path="",
        action_head="flow_matching", flow_num_heads=4, flow_condition_layers=4,
        flow_dit_layers=16, flow_target_tokens=16, flow_static_tokens=8,
        flow_state_dim=8, flow_bijection_blocks=6,
        flow_state_encoding="state_tokens_zero_pad", action_dim=7, chunk_size=12,
        state_dim=8, num_state_tokens=2, use_r3m=True, r3m_model="resnet18",
        freeze_r3m=True, r3m_encode_chunk_size=32,
        text_padding_length=21,
        text_layout_path="experiments/libero/configs/online_text_layout.json",
        freeze_text_encoder=True, frozen_text_cache=True,
        pretrained_gdino_ckpt="", qwen3vl_lr=5e-5, qwen3vl_weight_decay=1e-10,
        precision="bf16_amp", keep_latest_ckpt=False, save_final=True,
        expected_image_size=256, hidden_dim=256, nheads=8,
        dim_feedforward=2048, max_text_len=256,
        vla_feature_enhancer_layers=6, enhancer_inner_dim=1024,
        history_length=12, history_hidden_dim=256, history_layers=2,
        history_encoder="mamba", history_dropout=0.0,
        history_visual_dinov3=False, history_visual_encode_chunk_size=32,
        history_visual_backprop=False, history_r3m=True,
        history_r3m_rope_base=10000.0,
        history_r3m_memory_num_queries=2, history_r3m_memory_num_heads=4,
        history_r3m_memory_dropout=0.1, history_r3m_memory_gate_init=0.1,
        history_r3m_memory_dynamic_gate=False,
        history_r3m_memory_residual_gate=False,
        history_r3m_memory_token_dropout=0.0,
        history_r3m_memory_rope_values=False,
        history_r3m_intentional_memory=False,
        history_r3m_intentional_operator_rank=64,
        history_r3m_predictive_belief=True,
        history_r3m_belief_num_slots=4,
        history_r3m_belief_future_horizons="1,4,8,12",
        history_r3m_controlled_causal_belief=True,
        history_r3m_tacit_belief_fusion=True,
        history_r3m_tacit_belief_gate_init=0.1,
        text_dropout=0.0, fusion_dropout=0.0, fusion_droppath=0.1,
        allow_hf_download=False, shuffle_steps_within_episode=True,
        freeze_backbones=False, require_feature_enhancer_preload=False,
        load_text_proj_from_gdino=False, require_text_proj_preload=False,
    )
    args = parser.parse_args()
    args.history_scratch_init = args.resume_mode == "none" and not args.init_checkpoint
    return args


def parse_fusion_layers(value):
    """Parse a comma-separated list of 0-based DINOv3 block indices."""
    if not value:
        return ()
    raw_items = [item.strip() for item in str(value).split(",") if item.strip()]
    try:
        return tuple(int(item) for item in raw_items)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"dinov3_fusion_layers must be comma-separated integers, got {value!r}"
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        use_cuda = torch.cuda.is_available()
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend)
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False
    return is_distributed, rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class CachedTextBank:
    def __init__(self, payload, path):
        if not isinstance(payload, dict):
            raise ValueError(f"text cache at {path} must be a dict")
        for key in ["instructions", "last_hidden_state", "attention_mask"]:
            if key not in payload:
                raise ValueError(f"text cache at {path} missing required key: {key}")

        self.path = path
        self.instructions = [str(item) for item in payload["instructions"]]
        raw_mapping = payload.get("instruction_to_index")
        if raw_mapping is None:
            raw_mapping = {instruction: idx for idx, instruction in enumerate(self.instructions)}
        self.instruction_to_index = {str(key): int(value) for key, value in raw_mapping.items()}
        self.stripped_to_index = {}
        for instruction, idx in self.instruction_to_index.items():
            self.stripped_to_index.setdefault(instruction.strip(), idx)

        self.last_hidden_state = payload["last_hidden_state"].detach().cpu().contiguous()
        self.attention_mask = payload["attention_mask"].detach().cpu().bool().contiguous()
        self.text_self_attention_masks = payload.get("text_self_attention_masks")
        if self.text_self_attention_masks is not None:
            self.text_self_attention_masks = self.text_self_attention_masks.detach().cpu().bool().contiguous()

        if self.last_hidden_state.ndim != 3:
            raise ValueError(f"last_hidden_state should be [N, L, H], got {self.last_hidden_state.shape}")
        if self.attention_mask.ndim != 2:
            raise ValueError(f"attention_mask should be [N, L], got {self.attention_mask.shape}")
        if tuple(self.last_hidden_state.shape[:2]) != tuple(self.attention_mask.shape):
            raise ValueError(
                f"text cache shape mismatch: last_hidden_state={self.last_hidden_state.shape}, "
                f"attention_mask={self.attention_mask.shape}"
            )
        if len(self.instructions) != self.last_hidden_state.shape[0]:
            raise ValueError(
                f"instruction count {len(self.instructions)} does not match cache rows "
                f"{self.last_hidden_state.shape[0]}"
            )

    @classmethod
    def load(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"text cache not found: {path}. Build it first with build_libero_instruction_text_cache.py"
            )
        payload = torch.load(path, map_location="cpu")
        return cls(payload, path)

    @property
    def text_hidden_dim(self):
        return int(self.last_hidden_state.shape[-1])

    def lookup(self, instructions):
        indices = []
        for instruction in instructions:
            instruction = str(instruction)
            idx = self.instruction_to_index.get(instruction)
            if idx is None:
                idx = self.stripped_to_index.get(instruction.strip())
            if idx is None:
                known = "\n".join(f"- {item}" for item in self.instructions[:20])
                raise KeyError(
                    f"instruction not found in text cache: {instruction!r}\nKnown cached instructions:\n{known}"
                )
            indices.append(idx)

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        result = {
            "last_hidden_state": self.last_hidden_state.index_select(0, idx_tensor),
            "attention_mask": self.attention_mask.index_select(0, idx_tensor),
        }
        if self.text_self_attention_masks is not None:
            result["text_self_attention_masks"] = self.text_self_attention_masks.index_select(0, idx_tensor)
        return result


def build_model_architecture(args):
    if getattr(args, "backbone", "dinov3") == "qwen3vl":
        if not args.qwen3vl_path:
            raise ValueError("--qwen3vl_path is required when --backbone=qwen3vl")
        from ..models.turbovla_qwen3vl import build_qwen3vl_vla

        model_args = DummyArgs()
        model_args.QWEN3VL_PATH = args.qwen3vl_path
        model_args.hidden_dim = args.hidden_dim
        model_args.nheads = args.nheads
        model_args.dim_feedforward = args.dim_feedforward
        model_args.action_dim = args.action_dim
        model_args.chunk_size = args.chunk_size
        model_args.state_dim = args.state_dim
        model_args.num_state_tokens = args.num_state_tokens
        model_args.freeze_backbone = args.freeze_backbones
        model_args.local_files_only = not args.allow_hf_download
        model_args.action_head = args.action_head
        model_args.flow_num_heads = args.flow_num_heads
        model_args.flow_condition_layers = args.flow_condition_layers
        model_args.flow_dit_layers = args.flow_dit_layers
        model_args.flow_target_tokens = args.flow_target_tokens
        model_args.flow_static_tokens = args.flow_static_tokens
        model_args.flow_state_dim = args.flow_state_dim
        model_args.flow_bijection_blocks = args.flow_bijection_blocks
        model_args.flow_state_encoding = args.flow_state_encoding
        model = build_qwen3vl_vla(model_args)
        # Keep the Qwen3-VL backbone in BF16 to halve memory and checkpoint size.
        model.qwen3vl.to(dtype=torch.bfloat16)
        return model

    text_layout = {}
    if args.text_layout_path:
        with open(args.text_layout_path, "r", encoding="utf-8") as handle:
            text_layout = json.load(handle)
        configured_length = int(text_layout["output_padding_length"])
        if configured_length != args.text_padding_length:
            raise ValueError(
                f"text layout output length {configured_length} does not match "
                f"--text_padding_length={args.text_padding_length}"
            )

    model_args = DummyArgs()
    model_args.dinov3_path = args.dinov3_path
    model_args.bert_path = args.bert_path
    model_args.hidden_dim = args.hidden_dim
    model_args.nheads = args.nheads
    model_args.dim_feedforward = args.dim_feedforward
    model_args.max_text_len = args.max_text_len
    model_args.text_padding_length = args.text_padding_length
    model_args.text_padding_length_by_instruction = text_layout.get("padding_length_by_instruction", {})
    model_args.vla_feature_enhancer_layers = args.vla_feature_enhancer_layers
    model_args.enhancer_inner_dim = args.enhancer_inner_dim
    model_args.text_dropout = args.text_dropout
    model_args.fusion_dropout = args.fusion_dropout
    model_args.fusion_droppath = args.fusion_droppath
    model_args.action_dim = args.action_dim
    model_args.chunk_size = args.chunk_size
    model_args.state_dim = args.state_dim
    model_args.num_state_tokens = args.num_state_tokens
    model_args.local_files_only = not args.allow_hf_download
    model_args.freeze_vision_encoder = args.freeze_backbones
    model_args.freeze_text_encoder = args.freeze_text_encoder
    model_args.frozen_text_cache = args.frozen_text_cache
    model_args.dinov3_precision = getattr(args, "dinov3_precision", "bf16_autocast")
    model_args.num_views = 2
    model_args.image_size = args.expected_image_size
    model_args.position_embedding = "view"
    model_args.encode_views_separately = True
    model_args.padding_strategy = "key_padding_mask"
    model_args.action_head = args.action_head
    model_args.flow_num_heads = args.flow_num_heads
    model_args.flow_condition_layers = args.flow_condition_layers
    model_args.flow_dit_layers = args.flow_dit_layers
    model_args.flow_target_tokens = args.flow_target_tokens
    model_args.flow_static_tokens = args.flow_static_tokens
    model_args.flow_state_dim = args.flow_state_dim
    model_args.flow_bijection_blocks = args.flow_bijection_blocks
    model_args.flow_state_encoding = args.flow_state_encoding
    model_args.history_length = args.history_length
    model_args.history_hidden_dim = args.history_hidden_dim
    model_args.history_layers = args.history_layers
    model_args.history_encoder = args.history_encoder
    model_args.history_dropout = args.history_dropout
    model_args.history_visual_dinov3 = args.history_visual_dinov3
    model_args.history_visual_encode_chunk_size = args.history_visual_encode_chunk_size
    model_args.history_visual_backprop = args.history_visual_backprop
    model_args.history_r3m = args.history_r3m
    model_args.history_r3m_rope_base = args.history_r3m_rope_base
    model_args.history_r3m_memory_num_queries = args.history_r3m_memory_num_queries
    model_args.history_r3m_memory_num_heads = args.history_r3m_memory_num_heads
    model_args.history_r3m_memory_dropout = args.history_r3m_memory_dropout
    model_args.history_r3m_memory_gate_init = args.history_r3m_memory_gate_init
    model_args.history_r3m_memory_dynamic_gate = args.history_r3m_memory_dynamic_gate
    model_args.history_r3m_memory_residual_gate = args.history_r3m_memory_residual_gate
    model_args.history_r3m_memory_token_dropout = args.history_r3m_memory_token_dropout
    model_args.history_r3m_memory_rope_values = args.history_r3m_memory_rope_values
    model_args.history_r3m_intentional_memory = args.history_r3m_intentional_memory
    model_args.history_r3m_intentional_operator_rank = (
        args.history_r3m_intentional_operator_rank
    )
    model_args.history_r3m_predictive_belief = args.history_r3m_predictive_belief
    model_args.history_r3m_belief_num_slots = args.history_r3m_belief_num_slots
    model_args.history_r3m_belief_future_horizons = args.history_r3m_belief_future_horizons
    model_args.history_r3m_controlled_causal_belief = (
        args.history_r3m_controlled_causal_belief
    )
    model_args.history_r3m_tacit_belief_fusion = args.history_r3m_tacit_belief_fusion
    model_args.history_r3m_tacit_belief_gate_init = args.history_r3m_tacit_belief_gate_init
    model_args.use_r3m = args.use_r3m
    model_args.r3m_path = args.r3m_path
    model_args.r3m_model = args.r3m_model
    model_args.freeze_r3m = args.freeze_r3m
    model_args.r3m_encode_chunk_size = args.r3m_encode_chunk_size
    return build_turbovla(model_args)


def freeze_backbones(model):
    for name, param in model.named_parameters():
        if (
            name.startswith("vision_encoder.backbone")
            or name.startswith("text_encoder.bert")
            or name.startswith("dinov3")
            or name.startswith("qwen3vl")
        ):
            param.requires_grad = False


def get_latest_checkpoint(ckpt_dir, prefix):
    ckpts = glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.pth"))
    if not ckpts:
        return None

    def extract_step(path):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(name.split("_")[-1])
        except ValueError:
            return -1

    return max(ckpts, key=extract_step)


def prune_old_checkpoints(ckpt_dir, prefix, keep_latest_path, logger=print):
    """Remove older checkpoints so only the newest one remains on disk."""
    keep_name = os.path.basename(keep_latest_path)
    for path in glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.pth")):
        if os.path.basename(path) == keep_name:
            continue
        try:
            os.remove(path)
            logger(f"pruned old checkpoint: {path}")
        except OSError as exc:
            logger(f"failed to prune {path}: {exc}")


def masked_l1_loss(pred, target, mask):
    l1 = torch.abs(pred - target)
    mask = mask.unsqueeze(-1).float()
    l1 = l1 * mask
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return l1.sum() / denom


def build_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            warmup_scale = float(step + 1) / float(max(1, warmup_steps))
            return 0.1 + 0.9 * warmup_scale

        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _use_weight_decay(name, param):
    if param.ndim <= 1:
        return False
    lowered = name.lower()
    if lowered.endswith(".bias"):
        return False
    if "norm" in lowered or "layernorm" in lowered:
        return False
    return True


def build_param_group_optimizer(model, args):
    head_lr = args.lr if args.head_lr is None else args.head_lr
    head_wd = args.weight_decay if args.head_weight_decay is None else args.head_weight_decay
    grouped = {
        ("dinov3_decay", args.dinov3_lr, args.dinov3_weight_decay): [],
        ("dinov3_no_decay", args.dinov3_lr, 0.0): [],
        ("qwen3vl_decay", args.qwen3vl_lr, args.qwen3vl_weight_decay): [],
        ("qwen3vl_no_decay", args.qwen3vl_lr, 0.0): [],
        ("head_decay", head_lr, head_wd): [],
        ("head_no_decay", head_lr, 0.0): [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_dino = name.startswith("vision_encoder.backbone") or name.startswith("dinov3")
        is_qwen = name.startswith("qwen3vl")
        decay = _use_weight_decay(name, param)
        if is_dino and decay:
            key = ("dinov3_decay", args.dinov3_lr, args.dinov3_weight_decay)
        elif is_dino:
            key = ("dinov3_no_decay", args.dinov3_lr, 0.0)
        elif is_qwen and decay:
            key = ("qwen3vl_decay", args.qwen3vl_lr, args.qwen3vl_weight_decay)
        elif is_qwen:
            key = ("qwen3vl_no_decay", args.qwen3vl_lr, 0.0)
        elif decay:
            key = ("head_decay", head_lr, head_wd)
        else:
            key = ("head_no_decay", head_lr, 0.0)
        grouped[key].append(param)

    param_groups = []
    summary = []
    for (group_name, lr, weight_decay), params in grouped.items():
        if not params:
            continue
        count = sum(p.numel() for p in params)
        param_groups.append({"params": params, "lr": lr, "weight_decay": weight_decay, "name": group_name})
        summary.append({"name": group_name, "lr": lr, "weight_decay": weight_decay, "params": count})
    return AdamW(param_groups), summary


def reduce_mean(value, device, is_distributed, world_size):
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return tensor.item()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["model_state_dict", "model", "state_dict"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("unsupported checkpoint format")


def load_feature_enhancer_from_groundingdino(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    prefix_pairs = [
        ("transformer.encoder.fusion_layers.", "feature_enhancer.fusion_layers."),
        ("transformer.encoder.text_layers.", "feature_enhancer.text_layers."),
    ]

    for src_key, tensor in source_state.items():
        tgt_key = None
        for src_prefix, tgt_prefix in prefix_pairs:
            if src_key.startswith(src_prefix):
                tgt_key = tgt_prefix + src_key[len(src_prefix):]
                break
        if tgt_key is None:
            continue
        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no feature-enhancer parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def load_text_proj_from_groundingdino(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    source_state = _extract_state_dict(ckpt)
    source_state = {(k[7:] if k.startswith("module.") else k): v for k, v in source_state.items()}

    target_state = model.state_dict()
    mapped = {}
    skipped_missing = 0
    skipped_shape = 0
    missing_keys = []
    shape_mismatch_keys = []

    for src_key, tensor in source_state.items():
        tgt_key = None
        if src_key.startswith("feat_map."):
            tgt_key = "text_proj." + src_key[len("feat_map."):]
        else:
            continue

        if tgt_key not in target_state:
            skipped_missing += 1
            missing_keys.append((src_key, tgt_key))
            continue
        if target_state[tgt_key].shape != tensor.shape:
            skipped_shape += 1
            shape_mismatch_keys.append((src_key, tgt_key, tuple(tensor.shape), tuple(target_state[tgt_key].shape)))
            continue
        mapped[tgt_key] = tensor

    if not mapped:
        raise RuntimeError("no text projection parameters were mapped from checkpoint")

    target_state.update(mapped)
    missing, unexpected = model.load_state_dict(target_state, strict=False)
    return {
        "mapped": len(mapped),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "load_missing_after_update": len(missing),
        "load_unexpected_after_update": len(unexpected),
        "missing_keys": missing_keys,
        "shape_mismatch_keys": shape_mismatch_keys,
    }


def move_samples_to_device(samples, device):
    if isinstance(samples, dict):
        return {k: v.to(device, non_blocking=True) for k, v in samples.items()}
    return samples.to(device, non_blocking=True)


def train_model():
    args = parse_args()
    if args.use_r3m:
        if args.backbone != "dinov3":
            raise ValueError("R3M fusion requires --backbone dinov3")
        if not args.r3m_path or not os.path.isfile(args.r3m_path):
            raise FileNotFoundError(f"R3M checkpoint not found: {args.r3m_path}")
        if not args.freeze_r3m:
            raise ValueError("this R3M fusion recipe requires --freeze_r3m")
        if args.r3m_encode_chunk_size < 1:
            raise ValueError("R3M encode chunk size must be positive")
        if args.history_visual_dinov3:
            raise ValueError("R3M fusion experiment removes image history; use --no_history_visual_dinov3")
        if args.r3m_feature_cache_path and not args.freeze_r3m:
            raise ValueError("r3m_feature_cache_path requires --freeze_r3m; cached raw features cannot train R3M")
    if args.history_r3m:
        if not args.use_r3m or args.history_length != 12:
            raise ValueError("R3M history requires --use_r3m and --history_length 12")
        if args.history_r3m_memory_num_queries < 1:
            raise ValueError("R3M history memory query count must be positive")
        if args.history_r3m_memory_num_heads < 1:
            raise ValueError("R3M history memory head count must be positive")
        if args.history_hidden_dim % args.history_r3m_memory_num_heads:
            raise ValueError("R3M history hidden dimension must be divisible by memory heads")
        if not 0.0 <= args.history_r3m_memory_dropout < 1.0:
            raise ValueError("R3M history memory dropout must be in [0, 1)")
        if not 0.0 < args.history_r3m_memory_gate_init < 1.0:
            raise ValueError("R3M history memory gate init must be in (0, 1)")
        if not 0.0 <= args.history_r3m_memory_token_dropout < 1.0:
            raise ValueError("R3M history memory token dropout must be in [0, 1)")
        if args.history_r3m_memory_dynamic_gate and args.history_r3m_memory_residual_gate:
            raise ValueError("dynamic and residual R3M memory gating are mutually exclusive")
        if args.history_r3m_intentional_operator_rank < 1:
            raise ValueError("intentional R3M operator rank must be positive")
        if args.history_r3m_intentional_operator_rank > args.history_hidden_dim:
            raise ValueError(
                "intentional R3M operator rank cannot exceed the history hidden dimension"
            )
        try:
            belief_horizons = tuple(
                int(value.strip())
                for value in args.history_r3m_belief_future_horizons.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise ValueError("predictive belief horizons must be comma-separated integers") from exc
        if not belief_horizons or tuple(sorted(set(belief_horizons))) != belief_horizons:
            raise ValueError("predictive belief horizons must be positive, sorted, and unique")
        if belief_horizons[0] < 1:
            raise ValueError("predictive belief horizons must be positive")
        args.history_r3m_belief_future_horizons = belief_horizons
        if args.history_r3m_intentional_memory:
            if not args.history_r3m_memory_rope_values:
                raise ValueError("intentional R3M memory requires full key/value RoPE")
            if args.history_r3m_predictive_belief:
                raise ValueError(
                    "intentional R3M memory and predictive belief are mutually exclusive"
                )
            if (
                args.history_r3m_memory_dynamic_gate
                or args.history_r3m_memory_residual_gate
                or args.history_r3m_memory_token_dropout
                or args.history_r3m_tacit_belief_fusion
            ):
                raise ValueError(
                    "intentional R3M memory is a single ungated path and cannot use "
                    "memory gates, token dropout, or tacit-belief fusion"
                )
        if args.history_r3m_predictive_belief or args.history_r3m_intentional_memory:
            if args.history_r3m_memory_rope_values:
                if args.history_r3m_predictive_belief:
                    raise ValueError(
                        "predictive belief requires --no_history_r3m_memory_rope_values"
                    )
            if args.history_r3m_predictive_belief and args.history_r3m_belief_num_slots < 1:
                raise ValueError("predictive belief slot count must be positive")
        if args.history_r3m_tacit_belief_fusion:
            if not args.history_r3m_predictive_belief:
                raise ValueError("tacit-belief fusion requires --history_r3m_predictive_belief")
            if not 0.0 < args.history_r3m_tacit_belief_gate_init < 1.0:
                raise ValueError("tacit-belief fusion gate init must be in (0, 1)")
        if args.history_r3m_controlled_causal_belief:
            if not args.history_r3m_predictive_belief:
                raise ValueError(
                    "controlled causal belief requires --history_r3m_predictive_belief"
                )
            if not args.history_r3m_tacit_belief_fusion:
                raise ValueError(
                    "controlled causal belief requires --history_r3m_tacit_belief_fusion"
                )
            if args.history_r3m_intentional_memory:
                raise ValueError(
                    "controlled causal belief and intentional R3M memory are mutually exclusive"
                )
    elif (
        args.history_r3m_memory_dynamic_gate
        or args.history_r3m_memory_residual_gate
        or args.history_r3m_memory_token_dropout
        or args.history_r3m_intentional_memory
        or args.history_r3m_predictive_belief
        or args.history_r3m_tacit_belief_fusion
        or args.history_r3m_controlled_causal_belief
    ):
        raise ValueError("R3M memory variants require --history_r3m")
    if args.history_length:
        if args.backbone != "dinov3" or args.action_head != "flow_matching":
            raise ValueError("history training requires --backbone dinov3 and --action_head flow_matching")
        if (
            args.history_length not in {4, 12}
            or args.state_dim != 8
            or args.action_dim != 7
            or args.chunk_size != 12
            or args.history_hidden_dim != 256
            or args.history_layers != 2
            or args.history_encoder != "mamba"
            or args.history_dropout != 0.0
        ):
            raise ValueError(
                "history training requires H in {4, 12}, 8D state, "
                "2x256 Mamba, and 12x7 output"
            )
        if args.freeze_backbones:
            raise ValueError("history training requires --no_freeze_backbones so DINOv3 is trainable")
        if not args.freeze_text_encoder:
            raise ValueError("history training requires --freeze_text_encoder so BERT remains frozen")
        if not args.init_checkpoint and args.resume_mode == "none" and not args.history_scratch_init:
            raise ValueError("history training requires --init_checkpoint pointing to the 95k checkpoint")
        if args.history_visual_dinov3 and args.history_visual_encode_chunk_size < 1:
            raise ValueError("DINOv3 visual history encode chunk size must be positive")
        if args.history_scratch_init and (
            args.init_checkpoint or args.resume_mode != "none" or args.pretrained_gdino_ckpt
        ):
            raise ValueError(
                "--history_scratch_init cannot be combined with init, resume, or GroundingDINO weights"
            )
    if args.init_checkpoint and args.resume_mode != "none":
        raise ValueError("--init_checkpoint cannot be combined with --resume_mode")
    if args.init_checkpoint and args.pretrained_gdino_ckpt:
        raise ValueError(
            "--init_checkpoint already supplies all weights and cannot be combined "
            "with --pretrained_gdino_ckpt"
        )
    if args.head_lr is None:
        args.head_lr = args.lr
    if args.head_weight_decay is None:
        args.head_weight_decay = args.weight_decay
    if args.precision == "bf16_amp":
        if not torch.cuda.is_available():
            raise ValueError("bf16_amp precision requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16_amp requested but current CUDA device does not report bf16 support")
    torch.set_float32_matmul_precision("high")

    if args.lr_schedule_steps is None:
        args.lr_schedule_steps = args.max_steps
    if args.lr_schedule_steps <= args.warmup_steps:
        raise ValueError(
            f"lr_schedule_steps={args.lr_schedule_steps} must be greater than warmup_steps={args.warmup_steps}"
        )
    set_seed(args.seed)

    is_distributed, rank, world_size, local_rank, device = setup_distributed()

    try:
        is_qwen3vl = getattr(args, "backbone", "dinov3") == "qwen3vl"
        resume_candidate = None
        if args.resume_mode != "none":
            resume_candidate = get_latest_checkpoint(args.checkpoint_dir, args.checkpoint_prefix)

        if is_qwen3vl:
            if not args.qwen3vl_path:
                raise ValueError("--qwen3vl_path is required when --backbone=qwen3vl")
            text_cache = None
        else:
            if not args.bert_path:
                raise ValueError("--bert_path is required when --backbone=dinov3")
            text_cache = None

        model = build_model_architecture(args)

        if not is_qwen3vl:
            need_any_gdino_preload = (
                args.require_feature_enhancer_preload
                or args.require_text_proj_preload
                or args.load_text_proj_from_gdino
            )
            if (
                resume_candidate is None
                and not args.init_checkpoint
                and not args.pretrained_gdino_ckpt
                and need_any_gdino_preload
            ):
                raise ValueError(
                    "`--pretrained_gdino_ckpt` is required for requested preload options "
                    "(feature-enhancer and/or text_proj)"
                )

            if resume_candidate is None and not args.init_checkpoint and args.pretrained_gdino_ckpt:
                if args.require_feature_enhancer_preload:
                    fe_report = load_feature_enhancer_from_groundingdino(model, args.pretrained_gdino_ckpt)
                    if rank == 0:
                        print("feature-enhancer preload from GroundingDINO:")
                        print(f"  ckpt: {args.pretrained_gdino_ckpt}")
                        print(f"  mapped={fe_report['mapped']}")
                        print(f"  skipped_missing={fe_report['skipped_missing']}")
                        print(f"  skipped_shape={fe_report['skipped_shape']}")
                        print("missing_keys:", fe_report["missing_keys"])
                        print("shape_mismatch_keys:", fe_report["shape_mismatch_keys"])

                if args.load_text_proj_from_gdino:
                    text_proj_report = load_text_proj_from_groundingdino(model, args.pretrained_gdino_ckpt)
                    if rank == 0:
                        print("text projection preload from GroundingDINO:")
                        print(f"  ckpt: {args.pretrained_gdino_ckpt}")
                        print(f"  mapped={text_proj_report['mapped']}")
                        print(f"  skipped_missing={text_proj_report['skipped_missing']}")
                        print(f"  skipped_shape={text_proj_report['skipped_shape']}")
                        print("missing_keys:", text_proj_report["missing_keys"])
                        print("shape_mismatch_keys:", text_proj_report["shape_mismatch_keys"])
                elif args.require_text_proj_preload:
                    raise ValueError(
                        "text projection preload is required, but `--no_load_text_proj_from_gdino` was set"
                    )

        if args.freeze_backbones:
            freeze_backbones(model)

        if rank == 0:
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
            print(f"device={device}, distributed={is_distributed}, world_size={world_size}")
            print(f"backbone={args.backbone}")
            if is_qwen3vl:
                print(f"qwen3vl_path={args.qwen3vl_path}")
            else:
                print(f"dinov3_path={args.dinov3_path}")
                print(f"bert_path={args.bert_path}")
                print(
                    f"r3m_enabled={args.use_r3m}, r3m_path={args.r3m_path or None}, "
                    f"r3m_frozen={args.freeze_r3m}"
                )
                print(f"text_padding_length={args.text_padding_length}")
            print(f"pretrained_gdino_ckpt={args.pretrained_gdino_ckpt}")
            print(f"load_text_proj_from_gdino={args.load_text_proj_from_gdino}")
            print(f"max_steps={args.max_steps}, lr_schedule_steps={args.lr_schedule_steps}")
            print(
                f"precision={args.precision}, head_lr={args.head_lr}, "
                f"dinov3_lr={args.dinov3_lr}, qwen3vl_lr={args.qwen3vl_lr}, "
                f"head_wd={args.head_weight_decay}, dinov3_wd={args.dinov3_weight_decay}, "
                f"qwen3vl_wd={args.qwen3vl_weight_decay}"
            )
            print(f"warmup_steps={args.warmup_steps}, min_lr_ratio={args.min_lr_ratio}")
            print(
                f"shuffle: episode_buffer={args.shuffle_buffer}, "
                f"within_episode={args.shuffle_steps_within_episode}"
            )
            print(f"trainable params: {len(trainable)}")
            print(f"frozen params: {len(frozen)}")
            print("first 30 trainable:", trainable[:30])

        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        if is_distributed:
            if device.type == "cuda":
                dist.barrier(device_ids=[local_rank])
            else:
                dist.barrier()

        global_step = 0
        ckpt = None
        initialization_payload = None
        latest_ckpt_path = resume_candidate
        if latest_ckpt_path is not None and args.resume_mode != "none":
            if rank == 0:
                print(f"resume from {latest_ckpt_path} with mode={args.resume_mode}")
            ckpt = torch.load(latest_ckpt_path, map_location="cpu")
            assert_flow_checkpoint_compatible(ckpt, model)
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if rank == 0:
                print("resume missing keys:", len(missing))
                print("resume unexpected keys:", len(unexpected))
            if args.resume_mode == "all":
                global_step = int(ckpt.get("global_step", 0))
            else:
                global_step = 0
                if rank == 0:
                    print("optimizer/scheduler reset due to resume_mode=model")
        elif args.init_checkpoint:
            if not os.path.isfile(args.init_checkpoint):
                raise FileNotFoundError(f"initialization checkpoint not found: {args.init_checkpoint}")
            initialization_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
            assert_flow_checkpoint_compatible(initialization_payload, model)
            init_state = _extract_state_dict(initialization_payload)
            init_state = {(key[7:] if key.startswith("module.") else key): value for key, value in init_state.items()}
            missing, unexpected = model.load_state_dict(init_state, strict=False)
            allowed_missing = (
                set()
                if args.history_r3m
                else {key for key in model.state_dict() if key.startswith("history_encoder.")}
            )
            if set(missing) != allowed_missing or unexpected:
                raise RuntimeError(
                    "initialization checkpoint mismatch; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            if rank == 0:
                print(f"initialized from {args.init_checkpoint}")
                print(f"new history parameters initialized: {len(missing)} tensors")
        elif rank == 0:
            print("no checkpoint resumed, train from current initialization")

        model.to(device)
        if is_distributed:
            if device.type == "cuda":
                model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            else:
                model = DDP(model, find_unused_parameters=True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer, optimizer_summary = build_param_group_optimizer(unwrap_model(model), args)
        if rank == 0:
            print("optimizer param groups:", optimizer_summary)
        scheduler = build_scheduler(
            optimizer,
            max_steps=args.lr_schedule_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

        if ckpt is not None and args.resume_mode == "all":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if hasattr(optimizer, "load_ema_model_state_dict"):
                if "ema_model_state_dict" not in ckpt:
                    raise KeyError("resume_mode=all requires ema_model_state_dict for the pi0.5 optimizer")
                optimizer.load_ema_model_state_dict(ckpt["ema_model_state_dict"])
                if rank == 0:
                    print("restored EMA model state")

        model_to_load = unwrap_model(model)

        dataset = LiberoRLDSDataset(
            dataset_dir=args.dataset_dir,
            LOCAL_DINOV3_PATH=args.dinov3_path,
            rank=rank,
            world_size=world_size,
            chunk_size=args.chunk_size,
            split=args.dataset_split,
            shuffle_buffer=args.shuffle_buffer,
            shuffle_steps_within_episode=args.shuffle_steps_within_episode,
            step_mix_buffer_size=args.step_mix_buffer_size,
            seed=args.seed,
            local_files_only=not args.allow_hf_download,
            expected_image_size=args.expected_image_size,
            backbone=args.backbone,
            use_r3m=args.use_r3m,
            history_length=args.history_length,
            history_visual_dinov3=args.history_visual_dinov3,
            history_r3m=args.history_r3m,
            r3m_feature_cache_path=args.r3m_feature_cache_path,
            r3m_checkpoint_path=args.r3m_path,
            r3m_cache_dataset_dirs=getattr(args, "dataset_dirs", None),
            r3m_cache_precision="bf16_autocast" if args.precision == "bf16_amp" else "fp32",
        )

        if is_qwen3vl:
            from transformers import Qwen3VLProcessor

            qwen_processor = Qwen3VLProcessor.from_pretrained(
                args.qwen3vl_path,
                local_files_only=not args.allow_hf_download,
            )
            collate_fn = Qwen3VLCollator(qwen_processor)
        else:
            qwen_processor = None
            collate_fn = vla_collate_fn

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=None,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        data_iter = iter(dataloader)

        model.train()
        if args.freeze_backbones:
            if is_qwen3vl:
                model_to_load.qwen3vl.eval()
            else:
                model_to_load.dinov3.eval()

        if rank == 0:
            per_gpu_effective_bs = args.batch_size * args.grad_accum_steps
            global_effective_bs = per_gpu_effective_bs * world_size
            print(f"batch_size(per_gpu)={args.batch_size}, grad_accum_steps={args.grad_accum_steps}")
            print(f"effective_batch_size(per_gpu)={per_gpu_effective_bs}")
            print(f"effective_batch_size(global)={global_effective_bs}")
            pbar = tqdm(total=args.max_steps, initial=global_step, desc="training")
        else:
            pbar = None

        loss_window = []
        while global_step < args.max_steps:
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            last_aux_metrics = {}

            for _ in range(args.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                if args.history_length:
                    (
                        samples,
                        instructions,
                        states,
                        gt_actions,
                        action_chunk_masks,
                        history_states,
                        history_masks,
                    ) = batch
                    history_states = history_states.to(device, non_blocking=True)
                    history_masks = history_masks.to(device, non_blocking=True)
                else:
                    samples, instructions, states, gt_actions, action_chunk_masks = batch
                    history_states = history_masks = None

                samples = move_samples_to_device(samples, device)
                states = states.to(device, non_blocking=True)
                gt_actions = gt_actions.to(device, non_blocking=True)
                action_chunk_masks = action_chunk_masks.to(device, non_blocking=True)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.precision == "bf16_amp",
                ):
                    if is_qwen3vl and args.action_head == "flow_matching":
                        outputs = model(
                            samples,
                            states,
                            actions=gt_actions,
                            action_masks=action_chunk_masks,
                        )
                        if not isinstance(outputs, dict) or "loss" not in outputs:
                            raise TypeError("flow_matching model must return a dict containing loss")
                        loss = outputs["loss"]
                    elif is_qwen3vl:
                        pred_actions = model(samples, states)
                        if pred_actions.shape != gt_actions.shape:
                            raise ValueError(
                                f"pred_actions.shape={pred_actions.shape}, gt_actions.shape={gt_actions.shape} mismatch"
                            )
                        loss = masked_l1_loss(pred_actions, gt_actions, action_chunk_masks)
                    else:
                        if args.action_head == "flow_matching":
                            outputs = model(
                                instructions,
                                samples,
                                states,
                                actions=gt_actions,
                                action_masks=action_chunk_masks,
                                history_states=history_states,
                                history_mask=history_masks,
                            )
                            if not isinstance(outputs, dict) or "loss" not in outputs:
                                raise TypeError("flow_matching model must return a dict containing loss")
                            loss = outputs["loss"]
                            for metric_name in (
                                "belief_gain_mean",
                                "belief_tacit_gate_mean",
                            ):
                                if metric_name in outputs:
                                    last_aux_metrics[metric_name] = float(
                                        outputs[metric_name].detach().item()
                                    )
                        else:
                            pred_actions = model(instructions, samples, states)
                            if pred_actions.shape != gt_actions.shape:
                                raise ValueError(
                                    f"pred_actions.shape={pred_actions.shape}, gt_actions.shape={gt_actions.shape} mismatch"
                                )
                            loss = masked_l1_loss(pred_actions, gt_actions, action_chunk_masks)
                (loss / args.grad_accum_steps).backward()
                loss_accum += loss.detach().item()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            local_loss = loss_accum / args.grad_accum_steps
            global_loss = reduce_mean(local_loss, device, is_distributed, world_size)

            loss_window.append(global_loss)
            if len(loss_window) > args.log_freq:
                loss_window.pop(0)

            global_step += 1

            if rank == 0:
                avg_window_loss = sum(loss_window) / len(loss_window)
                pbar.update(1)
                if global_step % args.log_freq == 0:
                    postfix = {
                        "loss": f"{global_loss:.5f}",
                        "avg": f"{avg_window_loss:.5f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                    if last_aux_metrics:
                        if "belief_gain_mean" in last_aux_metrics:
                            postfix["gain"] = (
                                f"{last_aux_metrics['belief_gain_mean']:.3f}"
                            )
                        if "belief_tacit_gate_mean" in last_aux_metrics:
                            postfix["tacit"] = (
                                f"{last_aux_metrics['belief_tacit_gate_mean']:.3f}"
                            )
                    pbar.set_postfix(postfix)

            should_save = global_step % args.save_steps == 0 or (
                85000 <= global_step <= 99000 and global_step % 1000 == 0
            ) or (
                args.save_final and global_step == args.max_steps
            )
            if rank == 0 and should_save:
                save_path = os.path.join(args.checkpoint_dir, f"{args.checkpoint_prefix}_{global_step}.pth")
                model_state = unwrap_model(model).state_dict()
                flow_model = unwrap_model(model)
                flow_head = getattr(
                    flow_model, "flow_action_policy", getattr(flow_model, "action_policy", None)
                )
                saved_flow_state_dim = int(
                    getattr(flow_model, "flow_state_dim", getattr(flow_head, "state_dim", args.flow_state_dim))
                )
                saved_flow_blocks = int(
                    getattr(flow_model, "flow_bijection_blocks", getattr(flow_head, "bijection_blocks", args.flow_bijection_blocks))
                )
                saved_flow_state_encoding = str(
                    getattr(flow_model, "flow_state_encoding", getattr(flow_head, "state_encoding", args.flow_state_encoding))
                )
                saved_action_head = str(
                    getattr(flow_model, "action_head_type", getattr(flow_model, "action_head", args.action_head))
                )
                tmp_save_path = save_path + ".tmp"
                source_payload = ckpt if ckpt is not None else initialization_payload
                save_payload = {
                        "global_step": global_step,
                        "model_state_dict": model_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": global_loss,
                        "args": vars(args),
                        "model_config": unwrap_model(model).config.to_dict(),
                        "action_head": saved_action_head,
                        "flow_num_heads": args.flow_num_heads,
                        "flow_condition_layers": args.flow_condition_layers,
                        "flow_dit_layers": args.flow_dit_layers,
                        "flow_target_tokens": args.flow_target_tokens,
                        "flow_static_tokens": args.flow_static_tokens,
                        "flow_state_dim": saved_flow_state_dim,
                        "flow_bijection_blocks": saved_flow_blocks,
                        "flow_state_encoding": saved_flow_state_encoding,
                        "flow_config": {
                            "num_heads": args.flow_num_heads,
                            "condition_layers": args.flow_condition_layers,
                            "dit_layers": args.flow_dit_layers,
                            "target_tokens": args.flow_target_tokens,
                            "static_tokens": args.flow_static_tokens,
                            "state_dim": saved_flow_state_dim,
                            "bijection_blocks": saved_flow_blocks,
                            "state_encoding": saved_flow_state_encoding,
                            "bijection_input_dim": args.action_dim,
                            "bijection_latent_dim": 35,
                            "linear_expansion_inverse": "semi_orthogonal_transpose",
                        },
                        "normalization": {
                            "proprio": {
                                "mean": dataset.proprio_mean.detach().cpu().tolist(),
                                "std": dataset.proprio_std.detach().cpu().tolist(),
                            },
                            "action": {
                                "min": dataset.action_min.detach().cpu().tolist(),
                                "max": dataset.action_max.detach().cpu().tolist(),
                            },
                        },
                        "metadata": {
                            "architecture": (
                                "history12_dinov3_r3m_polanyi_from_to_intentional_memory_v14"
                                if args.history_r3m_intentional_memory
                                else "history12_dinov3_r3m_controlled_causal_state_v15"
                                if args.history_r3m_controlled_causal_belief
                                else "history12_dinov3_r3m_polanyi_tacit_belief_v13"
                                if args.history_r3m_tacit_belief_fusion
                                else "history12_dinov3_r3m_predictive_belief_v12"
                                if args.history_r3m_predictive_belief
                                else "history12_dinov3_r3m_current_conditioned_memory_rope_v8"
                                if args.history_r3m
                                else "history12_dinov3_r3m_v5"
                                if args.history_length == 12 and args.use_r3m
                                else
                                "history12_dinov3_v4"
                                if args.history_length == 12 and args.history_visual_dinov3
                                else "history12_v2"
                                if args.history_length == 12
                                else "history_v1"
                                if args.history_length == 4
                                else "turbovla"
                            ),
                            "history_length": args.history_length,
                            "history_input": (
                                "8D state trajectory + goal-operated full-RoPE tacit R3M memory"
                                if args.history_r3m_intentional_memory
                                else "8D state trajectory + goal-independent controlled causal R3M state"
                                if args.history_r3m_controlled_causal_belief
                                else "8D state trajectory + v8 tacit R3M memory with gated predictive correction"
                                if args.history_r3m_tacit_belief_fusion
                                else "task-conditioned predictive R3M belief and multi-horizon future tokens"
                                if args.history_r3m_predictive_belief
                                else "8D robot state + 4 current-conditioned view-separated R3M memory tokens"
                                if args.history_r3m
                                else "8D robot state + 2-view pooled DINOv3 visual history"
                                if args.history_visual_dinov3
                                else "8D robot state"
                                if args.history_length
                                else None
                            ),
                            "history_encoder": "2-layer Mamba, hidden=256" if args.history_length else None,
                            "current_visual_input": (
                                "2-view DINOv3 patch tokens + 2-view R3M ResNet-18 semantic tokens"
                                if args.use_r3m
                                else "2-view DINOv3 patch tokens"
                            ),
                            "r3m_encoder": "frozen_resnet18" if args.use_r3m else None,
                            "r3m_current_tokens": 2 if args.use_r3m else 0,
                            "history_visual_tokens": (
                                args.history_r3m_memory_num_queries * 2
                                if (
                                    args.history_r3m_intentional_memory
                                    or args.history_r3m_tacit_belief_fusion
                                )
                                else
                                (
                                    args.history_r3m_belief_num_slots * 2
                                    + len(args.history_r3m_belief_future_horizons)
                                )
                                if args.history_r3m_predictive_belief
                                else args.history_r3m_memory_num_queries * 2
                                if args.history_r3m
                                else args.history_length * 2
                                if args.history_visual_dinov3
                                else 0
                            ),
                            "history_visual_encoder": (
                                "polanyi_from_to_goal_operator_full_rope_intentional_memory"
                                if args.history_r3m_intentional_memory
                                else "goal_free_controlled_history_belief"
                                if args.history_r3m_controlled_causal_belief
                                else "v8_full_rope_current_memory_plus_gated_predictive_belief_correction"
                                if args.history_r3m_tacit_belief_fusion
                                else "frozen_r3m_semantic_attention_plus_delta_mamba_predictive_belief"
                                if args.history_r3m_predictive_belief
                                else "shared_frozen_r3m_relative_rope_current_query_cross_attention_view_separate"
                                if args.history_r3m
                                else "shared_dinov3_mean_pool"
                                if args.history_visual_dinov3
                                else None
                            ),
                            "history_r3m_memory_num_queries": (
                                args.history_r3m_memory_num_queries if args.history_r3m else None
                            ),
                            "history_r3m_memory_num_heads": (
                                args.history_r3m_memory_num_heads if args.history_r3m else None
                            ),
                            "history_r3m_memory_dropout": (
                                args.history_r3m_memory_dropout if args.history_r3m else None
                            ),
                            "history_r3m_memory_gate_init": (
                                args.history_r3m_memory_gate_init if args.history_r3m else None
                            ),
                            "history_r3m_memory_dynamic_gate": (
                                args.history_r3m_memory_dynamic_gate if args.history_r3m else None
                            ),
                            "history_r3m_memory_residual_gate": (
                                args.history_r3m_memory_residual_gate if args.history_r3m else None
                            ),
                            "history_r3m_memory_token_dropout": (
                                args.history_r3m_memory_token_dropout if args.history_r3m else None
                            ),
                            "history_r3m_memory_rope_values": (
                                args.history_r3m_memory_rope_values if args.history_r3m else None
                            ),
                            "history_r3m_rope_base": (
                                args.history_r3m_rope_base if args.history_r3m else None
                            ),
                            "history_r3m_predictive_belief": (
                                args.history_r3m_predictive_belief if args.history_r3m else None
                            ),
                            "history_r3m_intentional_memory": (
                                args.history_r3m_intentional_memory
                                if args.history_r3m
                                else None
                            ),
                            "history_r3m_intentional_operator_rank": (
                                args.history_r3m_intentional_operator_rank
                                if args.history_r3m_intentional_memory
                                else None
                            ),
                            "history_r3m_tacit_belief_fusion": (
                                args.history_r3m_tacit_belief_fusion
                                if args.history_r3m_predictive_belief
                                else None
                            ),
                            "history_r3m_controlled_causal_belief": (
                                args.history_r3m_controlled_causal_belief
                                if args.history_r3m_predictive_belief
                                else None
                            ),
                            "history_r3m_tacit_belief_gate_init": (
                                args.history_r3m_tacit_belief_gate_init
                                if args.history_r3m_tacit_belief_fusion
                                else None
                            ),
                            "history_r3m_belief_num_slots": (
                                args.history_r3m_belief_num_slots
                                if args.history_r3m_predictive_belief
                                else None
                            ),
                            "history_r3m_belief_future_horizons": (
                                list(args.history_r3m_belief_future_horizons)
                                if args.history_r3m_predictive_belief
                                else None
                            ),
                            "history_visual_backprop": (
                                args.history_visual_backprop if args.history_visual_dinov3 else None
                            ),
                            "history_visual_encode_chunk_size": (
                                args.history_visual_encode_chunk_size
                                if args.history_visual_dinov3
                                else None
                            ),
                            "action_trajectory_shape": [args.chunk_size, args.action_dim],
                            "execute_steps": args.chunk_size if args.history_length else None,
                            "initialization_checkpoint": args.init_checkpoint or None,
                        },
                }
                if isinstance(source_payload, dict) and "metadata" in source_payload:
                    save_payload["initialization_metadata"] = source_payload["metadata"]
                torch.save(save_payload, tmp_save_path)
                os.replace(tmp_save_path, save_path)
                print(f"saved: {save_path}")
                if args.keep_latest_ckpt:
                    prune_old_checkpoints(
                        args.checkpoint_dir,
                        args.checkpoint_prefix,
                        save_path,
                    )

        if pbar is not None:
            pbar.close()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    train_model()
