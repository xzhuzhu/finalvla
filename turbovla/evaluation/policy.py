"""TurboVLA policy adapter for aligned LIBERO evaluation.

This module keeps the GroundingDINO evaluation protocol local to the model:
256px DINOv3 preprocessing, GroundingDINO state normalization, hard min/max
action denormalization, and the original gripper sign rule.
"""

from __future__ import annotations

import json
import math
import os
import random
import types
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
import torch

from ..models.flow_checkpoint import (
    assert_flow_checkpoint_compatible,
    assert_flow_parameters_loaded,
    infer_flow_checkpoint_architecture,
)


EXPECTED_IMAGE_SIZE = 256
DINO_PATCH_SIZE = 16
ACTION_CHUNK_SIZE = 12
ACTION_DIM = 7
STATE_DIM = 8

DEFAULT_DINOV3_PATH = ""
DEFAULT_TEXT_CACHE_PATH = ""

ACTION_MIN = np.asarray(
    [
        -0.9375,
        -0.9375,
        -0.9375,
        -0.23642857372760773,
        -0.3053571283817291,
        -0.3675000071525574,
        -1.0,
    ],
    dtype=np.float32,
)

ACTION_MAX = np.asarray(
    [
        0.9375,
        0.9375,
        0.9375,
        0.30000001192092896,
        0.29357144236564636,
        0.375,
        1.0,
    ],
    dtype=np.float32,
)

PROPRIO_MEAN = np.asarray(
    [
        -0.04190646484494209,
        0.03539437800645828,
        0.8257066607475281,
        2.908315658569336,
        -0.5562158823013306,
        -0.16649103164672852,
        0.02831534668803215,
        -0.028561558574438095,
    ],
    dtype=np.float32,
)

PROPRIO_STD = np.asarray(
    [
        0.10743443667888641,
        0.14424759149551392,
        0.25723373889923096,
        0.34413808584213257,
        1.234430193901062,
        0.35798805952072144,
        0.013308786787092686,
        0.013174591585993767,
    ],
    dtype=np.float32,
)


def configure_transformers_offline(allow_hf_download: bool = False) -> None:
    if allow_hf_download:
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_preprocessor_config(local_model_path: str) -> dict[str, Any]:
    cfg_path = os.path.join(local_model_path, "preprocessor_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_manual_rgb_normalizer(
    image_mean: Sequence[float],
    image_std: Sequence[float],
    rescale_factor: float,
    expected_size: int,
    patch_size: int,
    backbone_name: str,
):
    mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
    rescale_factor = float(rescale_factor)

    def process_one(img: Image.Image | np.ndarray) -> torch.Tensor:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")

        width, height = img.size
        if height != expected_size or width != expected_size:
            raise ValueError(
                f"{backbone_name} expects pre-rotated {expected_size}x{expected_size} RGB input, "
                f"but got {height}x{width}. Do not apply StarVLA/OpenVLA 224px resize here."
            )
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"{backbone_name} input size {(height, width)} must be divisible by patch size {patch_size}."
            )

        arr = np.asarray(img, dtype=np.float32) * rescale_factor
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return (x - mean) / std

    def processor(images: Image.Image | np.ndarray | Sequence[Image.Image | np.ndarray]) -> dict[str, torch.Tensor]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        pixel_values = torch.stack([process_one(im) for im in images], dim=0)
        return {"pixel_values": pixel_values}

    return processor


def build_dinov3_manual_processor(local_dinov3_path: str):
    cfg = load_preprocessor_config(local_dinov3_path)
    return _build_manual_rgb_normalizer(
        image_mean=cfg.get("image_mean", [0.485, 0.456, 0.406]),
        image_std=cfg.get("image_std", [0.229, 0.224, 0.225]),
        rescale_factor=cfg.get("rescale_factor", 1.0 / 255.0),
        expected_size=EXPECTED_IMAGE_SIZE,
        patch_size=DINO_PATCH_SIZE,
        backbone_name="DINOv3",
    )


def build_r3m_manual_processor():
    """Return unnormalized uint8 224px center crops for the R3M encoder."""

    def process_one(image: Image.Image | np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if array.shape != (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE, 3):
            raise ValueError(
                f"R3M input must be {EXPECTED_IMAGE_SIZE}x{EXPECTED_IMAGE_SIZE} RGB, "
                f"got {array.shape}"
            )
        start = (EXPECTED_IMAGE_SIZE - 224) // 2
        crop = np.ascontiguousarray(array[start : start + 224, start : start + 224])
        return torch.from_numpy(crop).permute(2, 0, 1)

    def processor(
        images: Image.Image | np.ndarray | Sequence[Image.Image | np.ndarray],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        return {"pixel_values": torch.stack([process_one(image) for image in images])}

    return processor


def rotate_libero_image(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - float(quat[3]) * float(quat[3])))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def state_from_libero_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def normalize_state(state_np: np.ndarray) -> torch.Tensor:
    state = np.asarray(state_np, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"GroundingDINO state must have {STATE_DIM} dims, got {state.shape}")
    return torch.from_numpy((state - PROPRIO_MEAN) / (PROPRIO_STD + 1e-6)).float()


def denormalize_arm_action(action_norm_np: np.ndarray) -> np.ndarray:
    action = np.asarray(action_norm_np, dtype=np.float32).reshape(-1)
    if action.shape[0] < 6:
        raise ValueError(f"GroundingDINO action must have at least 6 dims, got {action.shape}")
    action = action[:6].copy()
    return 0.5 * (action + 1.0) * (ACTION_MAX[:6] - ACTION_MIN[:6]) + ACTION_MIN[:6]


def gripper_command_from_norm(gripper_norm: float, deadband: float = 0.0) -> float:
    if gripper_norm > deadband:
        return 1.0
    if gripper_norm < -deadband:
        return -1.0
    return 1.0


def normalized_action_to_env_action(
    action_norm_np: np.ndarray,
    gripper_deadband: float = 0.0,
) -> np.ndarray:
    return normalized_action_chunk_to_env_actions(
        np.asarray(action_norm_np, dtype=np.float32).reshape(1, -1),
        gripper_deadband=gripper_deadband,
    )[0]


def normalized_action_chunk_to_env_actions(
    action_norm_np: np.ndarray,
    gripper_deadband: float = 0.0,
    action_min: np.ndarray = ACTION_MIN,
    action_max: np.ndarray = ACTION_MAX,
) -> np.ndarray:
    """Vectorized normalized-to-environment conversion for an action chunk."""
    actions = np.asarray(action_norm_np, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
        raise ValueError(f"action chunk must have shape (N, D>={ACTION_DIM}), got {actions.shape}")

    minimum = np.asarray(action_min, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_max, dtype=np.float32).reshape(-1)
    if minimum.size < 6 or maximum.size < 6:
        raise ValueError("action_min and action_max must contain at least six arm dimensions")

    arm = 0.5 * (actions[:, :6] + 1.0) * (maximum[:6] - minimum[:6]) + minimum[:6]
    gripper_source = actions[:, 6]
    gripper = np.where(
        gripper_source > gripper_deadband,
        1.0,
        np.where(gripper_source < -gripper_deadband, -1.0, 1.0),
    ).astype(np.float32, copy=False)
    return np.concatenate([arm, gripper[:, None]], axis=1).astype(np.float32, copy=False)


def sanitize_pred_chunk(pred_chunk: np.ndarray) -> np.ndarray:
    pred_chunk = np.asarray(pred_chunk, dtype=np.float32)
    if pred_chunk.ndim == 1:
        pred_chunk = pred_chunk[None, :]
    elif pred_chunk.ndim > 2:
        pred_chunk = pred_chunk.reshape(pred_chunk.shape[0], -1)

    valid_actions = []
    for row in pred_chunk:
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.size < ACTION_DIM:
            continue
        if row.size > ACTION_DIM:
            row = row[:ACTION_DIM]
        row = np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=-1.0)
        valid_actions.append(row.astype(np.float32))

    if not valid_actions:
        return np.zeros((1, ACTION_DIM), dtype=np.float32)
    return np.stack(valid_actions, axis=0)


def get_libero_dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def _make_model_args(
    dinov3_path: str,
    bert_path: str,
    hidden_dim: int,
    nheads: int,
    dim_feedforward: int,
    max_text_len: int,
    text_padding_length: int,
    vla_feature_enhancer_layers: int,
    enhancer_inner_dim: int,
    action_dim: int,
    chunk_size: int,
    state_dim: int,
    num_state_tokens: int,
    text_dropout: float,
    fusion_dropout: float,
    fusion_droppath: float,
    sub_sentence_present: bool,
    allow_hf_download: bool,
    precision: str,
    action_head: str,
    flow_num_heads: int,
    flow_condition_layers: int,
    flow_dit_layers: int,
    flow_target_tokens: int,
    flow_static_tokens: int,
    flow_state_dim: int,
    flow_bijection_blocks: int,
    flow_state_encoding: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        dinov3_path=dinov3_path,
        bert_path=bert_path,
        hidden_dim=hidden_dim,
        nheads=nheads,
        dim_feedforward=dim_feedforward,
        max_text_len=max_text_len,
        text_padding_length=text_padding_length,
        sub_sentence_present=sub_sentence_present,
        vla_feature_enhancer_layers=vla_feature_enhancer_layers,
        enhancer_inner_dim=enhancer_inner_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        state_dim=state_dim,
        num_state_tokens=num_state_tokens,
        text_dropout=text_dropout,
        fusion_dropout=fusion_dropout,
        fusion_droppath=fusion_droppath,
        local_files_only=not allow_hf_download,
        freeze_text_encoder=True,
        freeze_vision_encoder=True,
        dinov3_precision="bf16" if precision == "bf16" else "bf16_autocast",
        num_views=2,
        image_size=EXPECTED_IMAGE_SIZE,
        position_embedding="view",
        encode_views_separately=True,
        padding_strategy="key_padding_mask",
        action_head=action_head,
        flow_num_heads=flow_num_heads,
        flow_condition_layers=flow_condition_layers,
        flow_dit_layers=flow_dit_layers,
        flow_target_tokens=flow_target_tokens,
        flow_static_tokens=flow_static_tokens,
        flow_state_dim=flow_state_dim,
        flow_bijection_blocks=flow_bijection_blocks,
        flow_state_encoding=flow_state_encoding,
    )


def load_turbovla_builder():
    from ..models.turbovla import build_turbovla

    return build_turbovla, "turbovla.models.turbovla"


class TurboVLAPolicy:
    def __init__(
        self,
        ckpt_path: str,
        dinov3_path: str = DEFAULT_DINOV3_PATH,
        bert_path: str = "",
        r3m_path: str = "",
        device: str | torch.device | None = None,
        allow_hf_download: bool = False,
        hidden_dim: int = 256,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        max_text_len: int = 256,
        text_padding_length: int = 21,
        vla_feature_enhancer_layers: int = 6,
        enhancer_inner_dim: int = 1024,
        action_dim: int = ACTION_DIM,
        chunk_size: int = ACTION_CHUNK_SIZE,
        state_dim: int = STATE_DIM,
        num_state_tokens: int = 2,
        text_dropout: float = 0.0,
        fusion_dropout: float = 0.0,
        fusion_droppath: float = 0.1,
        sub_sentence_present: bool = True,
        precision: str = "bf16",
        action_head: str = "act",
        flow_num_heads: int = 4,
        flow_condition_layers: int = 4,
        flow_dit_layers: int = 16,
        flow_target_tokens: int = 16,
        flow_static_tokens: int = 8,
        flow_state_dim: int = 0,
        flow_bijection_blocks: int = 6,
        flow_state_encoding: str = "native_mlp",
        dinov3_update_interval: int = 1,
        ablate_history_r3m_tokens: bool = False,
        verbose: bool = True,
    ) -> None:
        configure_transformers_offline(allow_hf_download=allow_hf_download)

        self.ckpt_path = str(ckpt_path)
        self.dinov3_path = str(dinov3_path)
        self.bert_path = str(bert_path)
        self.r3m_path = str(r3m_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.precision = str(precision).lower()
        self.model_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        self.action_head = str(action_head)
        if self.action_head not in {"act", "flow_matching"}:
            raise ValueError(f"Unsupported action_head={self.action_head!r}; expected 'act' or 'flow_matching'.")
        self.flow_num_heads = int(flow_num_heads)
        self.flow_condition_layers = int(flow_condition_layers)
        self.flow_dit_layers = int(flow_dit_layers)
        self.flow_target_tokens = int(flow_target_tokens)
        self.flow_static_tokens = int(flow_static_tokens)
        self.flow_state_dim = int(flow_state_dim)
        self.flow_bijection_blocks = int(flow_bijection_blocks)
        self.flow_state_encoding = str(flow_state_encoding)
        self.dinov3_update_interval = int(dinov3_update_interval)
        if self.dinov3_update_interval < 1:
            raise ValueError("dinov3_update_interval must be positive")
        self.ablate_history_r3m_tokens = bool(ablate_history_r3m_tokens)
        self._dinov3_token_cache: torch.Tensor | None = None
        self._policy_query_count = 0
        self.verbose = bool(verbose)
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported precision={precision!r}; expected 'bf16' or 'fp32'.")

        if not self.dinov3_path:
            raise ValueError("dinov3_path must point to a local DINOv3 checkpoint or a HF model id")
        if not self.bert_path:
            raise ValueError("bert_path must point to BERT base uncased or a compatible model id")
        if not allow_hf_download and not os.path.isdir(self.dinov3_path):
            raise FileNotFoundError(f"local DINOv3 directory not found: {self.dinov3_path}")
        if not allow_hf_download and not os.path.isdir(self.bert_path):
            raise FileNotFoundError(f"local BERT directory not found: {self.bert_path}")

        build_model, loaded_model_path = load_turbovla_builder()
        if self.verbose:
            print(f"[TurboVLAPolicy] model source: {loaded_model_path}", flush=True)
            print(
                "[TurboVLAPolicy] "
                f"precision={self.precision}",
                flush=True,
            )
        self._checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        ckpt_flow_args = {
            "action_head": self.action_head,
            "flow_num_heads": self.flow_num_heads,
            "flow_condition_layers": self.flow_condition_layers,
            "flow_dit_layers": self.flow_dit_layers,
            "flow_target_tokens": self.flow_target_tokens,
            "flow_static_tokens": self.flow_static_tokens,
            "flow_state_dim": self.flow_state_dim,
            "flow_bijection_blocks": self.flow_bijection_blocks,
            "flow_state_encoding": self.flow_state_encoding,
        }
        model_config = self._checkpoint.get("model_config") if isinstance(self._checkpoint, dict) else None
        if model_config is not None:
            from ..models.configuration import TurboVLAConfig
            from ..models.turbovla import TurboVLA

            config = TurboVLAConfig.from_mapping(model_config)
            config.text.model_name_or_path = self.bert_path
            config.text.local_files_only = not allow_hf_download
            config.vision.model_name_or_path = self.dinov3_path
            config.vision.local_files_only = not allow_hf_download
            config.vision.compute_precision = "bf16" if self.precision == "bf16" else "bf16_autocast"
            if config.r3m.enabled:
                configured_r3m_path = self.r3m_path or config.r3m.checkpoint_path
                configured_r3m_path = os.path.abspath(os.path.expanduser(configured_r3m_path))
                if not os.path.isfile(configured_r3m_path):
                    raise FileNotFoundError(f"R3M checkpoint not found: {configured_r3m_path}")
                config.r3m.checkpoint_path = configured_r3m_path
                self.r3m_path = configured_r3m_path
            self.chunk_size = int(config.action.horizon)
            self.action_dim = int(config.action.action_dim)
            ckpt_flow_args["action_head"] = self._checkpoint.get("action_head", self.action_head)
            ckpt_flow_args["flow_num_heads"] = self._checkpoint.get("flow_num_heads", self.flow_num_heads)
            ckpt_flow_args["flow_condition_layers"] = self._checkpoint.get("flow_condition_layers", self.flow_condition_layers)
            ckpt_flow_args["flow_dit_layers"] = self._checkpoint.get("flow_dit_layers", self.flow_dit_layers)
            ckpt_flow_args["flow_target_tokens"] = self._checkpoint.get("flow_target_tokens", self.flow_target_tokens)
            ckpt_flow_args["flow_static_tokens"] = self._checkpoint.get("flow_static_tokens", self.flow_static_tokens)
            ckpt_flow_args["flow_state_dim"] = self._checkpoint.get("flow_state_dim", self.flow_state_dim)
            ckpt_flow_args["flow_bijection_blocks"] = self._checkpoint.get(
                "flow_bijection_blocks", self.flow_bijection_blocks
            )
            ckpt_flow_args["flow_state_encoding"] = self._checkpoint.get(
                "flow_state_encoding", self.flow_state_encoding
            )
            inferred_flow = infer_flow_checkpoint_architecture(
                self._checkpoint, state_dim=int(config.action.state_dim)
            )
            if inferred_flow is not None:
                (
                    ckpt_flow_args["flow_state_dim"],
                    ckpt_flow_args["flow_bijection_blocks"],
                    ckpt_flow_args["flow_state_encoding"],
                ) = inferred_flow
            self.model = TurboVLA(config, **ckpt_flow_args)
            self.history_length = int(config.history.length) if config.history.enabled else 0
            self.history_visual_dinov3 = bool(config.history.visual_enabled)
            self.history_r3m = bool(config.history.r3m_enabled)
            self.r3m_enabled = bool(config.r3m.enabled)
        else:
            checkpoint_action_head = (
                self._checkpoint.get("action_head", self.action_head)
                if isinstance(self._checkpoint, dict)
                else self.action_head
            )
            raw_state = _checkpoint_state_dict(self._checkpoint)
            if any(key.removeprefix("module.").startswith("history_encoder.") for key in raw_state):
                raise RuntimeError(
                    "checkpoint has no model_config but contains history_encoder weights; "
                    "the bare-weight policy path cannot reconstruct full V15 history settings."
                )
            inferred_flow = infer_flow_checkpoint_architecture(
                self._checkpoint, state_dim=int(state_dim)
            )
            if checkpoint_action_head == "flow_matching" and inferred_flow is None:
                raise RuntimeError(
                    "checkpoint has no model_config and no recognizable flow structure; "
                    "cannot safely reconstruct its state route or coupling-block count."
                )
            if inferred_flow is not None:
                (
                    self.flow_state_dim,
                    self.flow_bijection_blocks,
                    self.flow_state_encoding,
                ) = inferred_flow
            model_args = _make_model_args(
                dinov3_path=self.dinov3_path,
                bert_path=self.bert_path,
                hidden_dim=hidden_dim,
                nheads=nheads,
                dim_feedforward=dim_feedforward,
                max_text_len=max_text_len,
                text_padding_length=text_padding_length,
                vla_feature_enhancer_layers=vla_feature_enhancer_layers,
                enhancer_inner_dim=enhancer_inner_dim,
                action_dim=action_dim,
                chunk_size=chunk_size,
                state_dim=state_dim,
                num_state_tokens=num_state_tokens,
                text_dropout=text_dropout,
                fusion_dropout=fusion_dropout,
                fusion_droppath=fusion_droppath,
                sub_sentence_present=sub_sentence_present,
                precision=self.precision,
                allow_hf_download=allow_hf_download,
                action_head=checkpoint_action_head,
                flow_num_heads=self.flow_num_heads,
                flow_condition_layers=self.flow_condition_layers,
                flow_dit_layers=self.flow_dit_layers,
                flow_target_tokens=self.flow_target_tokens,
                flow_static_tokens=self.flow_static_tokens,
                flow_state_dim=self.flow_state_dim,
                flow_bijection_blocks=self.flow_bijection_blocks,
                flow_state_encoding=self.flow_state_encoding,
            )
            self.model = build_model(model_args)
            self.history_length = 0
            self.history_visual_dinov3 = False
            self.history_r3m = False
            self.r3m_enabled = False
        if self.ablate_history_r3m_tokens and not self.history_r3m:
            raise ValueError(
                "ablate_history_r3m_tokens requires a checkpoint with R3M history enabled"
            )
        self._state_history: deque[np.ndarray] = deque(
            maxlen=self.history_length or 1
        )
        self._image_history: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=self.history_length or 1
        )
        self._load_checkpoint()
        self._set_eval_precision()
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self._verify_model_precision()
        self.dinov3_processor = build_dinov3_manual_processor(self.dinov3_path)
        self.r3m_processor = build_r3m_manual_processor() if self.r3m_enabled else None

    def _set_eval_precision(self) -> None:
        if self.precision == "bf16":
            self.model.to(dtype=torch.bfloat16)
        else:
            # FP32 evaluation keeps all parameters FP32 while autocasting only
            # the DINOv3 forward pass to BF16.
            self.model.float()

    def _verify_model_precision(self) -> None:
        floating_dtypes = {
            param.dtype
            for param in self.model.parameters()
            if param.is_floating_point()
        }
        expected = {self.model_dtype}
        if floating_dtypes != expected:
            raise RuntimeError(
                f"precision={self.precision} expected model parameter dtypes {expected}, "
                f"got {floating_dtypes}"
            )
        if self.verbose:
            dtype_name = str(self.model_dtype).removeprefix("torch.")
            print(
                f"[TurboVLAPolicy] precision={self.precision}, model_parameter_dtype={dtype_name}",
                flush=True,
            )

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        flow_architecture = assert_flow_checkpoint_compatible(checkpoint, self.model)
        source_state = _strip_module_prefix(_checkpoint_state_dict(checkpoint))
        target_state = self.model.state_dict()

        loadable = {}
        skipped_shape = []
        skipped_missing = []
        for key, tensor in source_state.items():
            if key not in target_state:
                skipped_missing.append(key)
                continue
            if target_state[key].shape != tensor.shape:
                skipped_shape.append((key, tuple(tensor.shape), tuple(target_state[key].shape)))
                continue
            loadable[key] = tensor

        missing, unexpected = self.model.load_state_dict(loadable, strict=False)
        if flow_architecture is not None:
            assert_flow_parameters_loaded(self.model, list(missing), skipped_missing)
        if self.verbose:
            print(f"[GroundingDINOAlignedPolicy] loaded ckpt: {self.ckpt_path}", flush=True)
            print(
                "[GroundingDINOAlignedPolicy] "
                f"loadable={len(loadable)}, missing_after_load={len(missing)}, "
                f"unexpected_after_load={len(unexpected)}, skipped_missing={len(skipped_missing)}, "
                f"skipped_shape={len(skipped_shape)}",
                flush=True,
            )
            if missing:
                print(f"  first missing keys: {list(missing)[:20]}", flush=True)
            if unexpected:
                print(f"  first unexpected keys: {list(unexpected)[:20]}", flush=True)
            if skipped_shape:
                print(f"  first shape mismatches: {skipped_shape[:10]}", flush=True)

    def _build_batch(
        self,
        primary_images: Sequence[np.ndarray],
        wrist_images: Sequence[np.ndarray],
        states: Sequence[np.ndarray],
        include_dinov3: bool = True,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        flat_images: list[np.ndarray] = []
        for primary, wrist in zip(primary_images, wrist_images):
            flat_images.extend([primary, wrist])

        batch_size = len(primary_images)
        samples = {}
        if include_dinov3:
            dinov3_pixel_values = self.dinov3_processor(flat_images)["pixel_values"]
            samples["dinov3"] = dinov3_pixel_values.view(
                batch_size, 2, *dinov3_pixel_values.shape[1:]
            ).to(self.device)
        if self.r3m_processor is not None:
            r3m_pixel_values = self.r3m_processor(flat_images)["pixel_values"]
            samples["r3m"] = r3m_pixel_values.view(
                batch_size, 2, *r3m_pixel_values.shape[1:]
            ).to(self.device)
        state_tensors = torch.stack([normalize_state(state) for state in states], dim=0).to(self.device)
        return samples, state_tensors

    def _prepare_model_inputs(
        self,
        samples: dict[str, torch.Tensor],
        states: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        samples = {
            key: value.to(dtype=self.model_dtype) if value.is_floating_point() else value
            for key, value in samples.items()
        }
        return samples, states.to(dtype=self.model_dtype)

    def reset_history(self) -> None:
        """Clear synchronized robot-state and visual history."""
        self._state_history.clear()
        self._image_history.clear()
        self._dinov3_token_cache = None
        self._policy_query_count = 0

    def _should_refresh_dinov3(self) -> bool:
        return (
            self._dinov3_token_cache is None
            or self._policy_query_count % self.dinov3_update_interval == 0
        )

    def _attach_dinov3_tokens(
        self,
        samples: dict[str, torch.Tensor],
        refresh: bool,
    ) -> dict[str, torch.Tensor]:
        if refresh:
            pixel_values = samples.pop("dinov3", None)
            if pixel_values is None:
                raise ValueError("dinov3 pixels are required when refreshing the token cache")
            self._dinov3_token_cache = self.model.encode_vision(pixel_values).detach()
        if self._dinov3_token_cache is None:
            raise RuntimeError("DINOv3 token cache was not initialized")
        samples["dinov3_tokens"] = self._dinov3_token_cache
        return samples

    @property
    def history_size(self) -> int:
        return len(self._state_history)

    def record_history_state(
        self,
        state_or_obs: np.ndarray | dict[str, Any],
    ) -> None:
        """Record one observed state and, when enabled, its two camera frames."""
        if not self.history_length:
            return
        state = (
            state_from_libero_obs(state_or_obs)
            if isinstance(state_or_obs, dict)
            else np.asarray(state_or_obs, dtype=np.float32).reshape(-1)
        )
        if state.shape != (STATE_DIM,):
            raise ValueError(f"history state must have shape ({STATE_DIM},), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("history state must be finite")
        self._state_history.append(state.copy())
        if self.history_visual_dinov3 or (
            getattr(self, "history_r3m", False)
            and not getattr(self, "ablate_history_r3m_tokens", False)
        ):
            if not isinstance(state_or_obs, dict):
                raise ValueError("visual history requires a LIBERO observation mapping")
            primary = rotate_libero_image(state_or_obs["agentview_image"])
            wrist = rotate_libero_image(state_or_obs["robot0_eye_in_hand_image"])
            self._image_history.append((primary, wrist))

    def _history_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if not self.history_length:
            return None, None

        states = np.zeros((self.history_length, STATE_DIM), dtype=np.float32)
        mask = np.zeros(self.history_length, dtype=np.bool_)
        count = len(self._state_history)
        if count:
            raw_states = np.stack(self._state_history)
            proprio_mean = np.asarray(getattr(self, "proprio_mean", PROPRIO_MEAN), dtype=np.float32)
            proprio_std = np.asarray(getattr(self, "proprio_std", PROPRIO_STD), dtype=np.float32)
            states[-count:] = (raw_states - proprio_mean) / (proprio_std + 1e-6)
            mask[-count:] = True

        states_tensor = torch.from_numpy(states).unsqueeze(0).to(
            device=self.device, dtype=self.model_dtype
        )
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device=self.device)
        return states_tensor, mask_tensor

    def _history_dinov3_tensor(self) -> torch.Tensor | None:
        if not self.history_visual_dinov3:
            return None
        if len(self._image_history) != len(self._state_history):
            raise RuntimeError("DINOv3 image history is not synchronized with state history")
        output = torch.zeros(
            1,
            self.history_length,
            2,
            3,
            EXPECTED_IMAGE_SIZE,
            EXPECTED_IMAGE_SIZE,
            dtype=torch.float32,
        )
        count = len(self._image_history)
        if count:
            flat_images = []
            for primary, wrist in self._image_history:
                flat_images.extend([primary, wrist])
            pixels = self.dinov3_processor(flat_images)["pixel_values"]
            output[0, -count:] = pixels.view(
                count,
                2,
                3,
                EXPECTED_IMAGE_SIZE,
                EXPECTED_IMAGE_SIZE,
            )
        return output.to(device=self.device)

    def _history_r3m_tensor(self) -> torch.Tensor | None:
        if not self.history_r3m or getattr(self, "ablate_history_r3m_tokens", False):
            return None
        if self.r3m_processor is None:
            raise RuntimeError("R3M history requires the R3M image processor")
        if len(self._image_history) != len(self._state_history):
            raise RuntimeError("R3M image history is not synchronized with state history")
        output = torch.zeros(
            1,
            self.history_length,
            2,
            3,
            224,
            224,
            dtype=torch.uint8,
        )
        count = len(self._image_history)
        if count:
            flat_images = []
            for primary, wrist in self._image_history:
                flat_images.extend([primary, wrist])
            pixels = self.r3m_processor(flat_images)["pixel_values"]
            output[0, -count:] = pixels.view(count, 2, 3, 224, 224)
        return output.to(device=self.device)

    def predict_normalized_action_chunk(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
    ) -> np.ndarray:
        if isinstance(state_or_obs, dict):
            state = state_from_libero_obs(state_or_obs)
        else:
            state = np.asarray(state_or_obs, dtype=np.float32)

        refresh_dinov3 = self._should_refresh_dinov3()
        samples, states = self._build_batch(
            [primary_image],
            [wrist_image],
            [state],
            include_dinov3=refresh_dinov3,
        )
        history_states, history_mask = self._history_tensors()
        history_dinov3 = self._history_dinov3_tensor()
        if history_dinov3 is not None:
            samples["dinov3_history"] = history_dinov3
        history_r3m = self._history_r3m_tensor()
        if history_r3m is not None:
            samples["r3m_history"] = history_r3m
        samples, states = self._prepare_model_inputs(samples, states)
        with torch.inference_mode():
            samples = self._attach_dinov3_tokens(samples, refresh_dinov3)
            pred = self.model(
                [instruction],
                samples,
                states,
                history_states=history_states,
                history_mask=history_mask,
                ablate_history_r3m_tokens=self.ablate_history_r3m_tokens,
            )
        self._policy_query_count += 1
        if pred.dtype != self.model_dtype:
            raise RuntimeError(
                f"precision={self.precision} expected forward output dtype {self.model_dtype}, got {pred.dtype}"
            )
        return sanitize_pred_chunk(pred.detach().float().cpu().numpy()[0])

    def predict_env_action_chunk(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
        execute_steps: int | None = None,
    ) -> np.ndarray:
        pred_norm = self.predict_normalized_action_chunk(primary_image, wrist_image, instruction, state_or_obs)
        if execute_steps is not None:
            pred_norm = pred_norm[: int(execute_steps)]
        return normalized_action_chunk_to_env_actions(pred_norm)

    def normalized_action_chunk_to_env_actions(self, chunk: np.ndarray) -> np.ndarray:
        return normalized_action_chunk_to_env_actions(chunk)

    def predict_env_action_chunk_from_obs(
        self,
        obs: dict[str, Any],
        instruction: str,
        execute_steps: int | None = None,
    ) -> np.ndarray:
        primary = rotate_libero_image(obs["agentview_image"])
        wrist = rotate_libero_image(obs["robot0_eye_in_hand_image"])
        return self.predict_env_action_chunk(primary, wrist, instruction, obs, execute_steps=execute_steps)


def batched(iterable: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


class Qwen3VLVLAPolicy:
    """LIBERO evaluation policy backed by Qwen3-VL (images + instruction in one pass)."""

    def __init__(
        self,
        ckpt_path: str,
        qwen3vl_path: str = "",
        device: str | torch.device | None = None,
        allow_hf_download: bool = False,
        hidden_dim: int = 256,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        action_dim: int = ACTION_DIM,
        chunk_size: int = ACTION_CHUNK_SIZE,
        state_dim: int = STATE_DIM,
        num_state_tokens: int = 2,
        precision: str = "bf16",
        verbose: bool = True,
    ) -> None:
        configure_transformers_offline(allow_hf_download=allow_hf_download)

        from transformers import Qwen3VLProcessor

        from ..models.turbovla_qwen3vl import build_qwen3vl_vla

        self.ckpt_path = str(ckpt_path)
        self.qwen3vl_path = str(qwen3vl_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.precision = str(precision).lower()
        self.model_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        self.verbose = bool(verbose)
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError(f"Unsupported precision={precision!r}; expected 'bf16' or 'fp32'.")
        if not self.qwen3vl_path:
            raise ValueError("qwen3vl_path must point to a local Qwen3-VL directory or a HF model id")
        if not allow_hf_download and not os.path.isdir(self.qwen3vl_path):
            raise FileNotFoundError(f"local Qwen3-VL directory not found: {self.qwen3vl_path}")

        model_args = SimpleNamespace(
            QWEN3VL_PATH=self.qwen3vl_path,
            hidden_dim=hidden_dim,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            action_dim=action_dim,
            chunk_size=chunk_size,
            state_dim=state_dim,
            num_state_tokens=num_state_tokens,
            freeze_backbone=True,
            local_files_only=not allow_hf_download,
        )
        self.model = build_qwen3vl_vla(model_args)
        self._load_checkpoint()
        if self.precision == "bf16":
            self.model.to(dtype=torch.bfloat16)
        else:
            self.model.float()
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.processor = Qwen3VLProcessor.from_pretrained(
            self.qwen3vl_path,
            local_files_only=not allow_hf_download,
        )
        # Keep the 256x256 LIBERO resolution instead of upscaling to Qwen3-VL's
        # default min_pixels budget.
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = EXPECTED_IMAGE_SIZE * EXPECTED_IMAGE_SIZE
            image_processor.max_pixels = EXPECTED_IMAGE_SIZE * EXPECTED_IMAGE_SIZE

        if self.verbose:
            print(f"[Qwen3VLVLAPolicy] model source: turbovla_qwen3vl.Qwen3VLVLA", flush=True)
            print(
                f"[Qwen3VLVLAPolicy] precision={self.precision}, "
                f"qwen3vl_path={self.qwen3vl_path}",
                flush=True,
            )

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        source_state = _strip_module_prefix(_checkpoint_state_dict(checkpoint))
        target_state = self.model.state_dict()

        loadable = {}
        skipped_shape = []
        skipped_missing = []
        for key, tensor in source_state.items():
            if key not in target_state:
                skipped_missing.append(key)
                continue
            if target_state[key].shape != tensor.shape:
                skipped_shape.append((key, tuple(tensor.shape), tuple(target_state[key].shape)))
                continue
            loadable[key] = tensor

        missing, unexpected = self.model.load_state_dict(loadable, strict=False)
        if self.verbose:
            print(f"[Qwen3VLVLAPolicy] loaded ckpt: {self.ckpt_path}", flush=True)
            print(
                "[Qwen3VLVLAPolicy] "
                f"loadable={len(loadable)}, missing_after_load={len(missing)}, "
                f"unexpected_after_load={len(unexpected)}, skipped_missing={len(skipped_missing)}, "
                f"skipped_shape={len(skipped_shape)}",
                flush=True,
            )
            if missing:
                print(f"  first missing keys: {list(missing)[:20]}", flush=True)
            if unexpected:
                print(f"  first unexpected keys: {list(unexpected)[:20]}", flush=True)
            if skipped_shape:
                print(f"  first shape mismatches: {skipped_shape[:10]}", flush=True)

    def _build_batch(
        self,
        primary_images: Sequence[np.ndarray],
        wrist_images: Sequence[np.ndarray],
        states: Sequence[np.ndarray],
        instructions: Sequence[str],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if len(primary_images) != len(wrist_images) or len(primary_images) != len(instructions):
            raise ValueError("primary/wrist image and instruction counts must match")

        image_pairs = []
        for primary, wrist in zip(primary_images, wrist_images):
            image_pairs.append([np.asarray(primary), np.asarray(wrist)])

        texts = []
        for pair, instruction in zip(image_pairs, instructions):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pair[0]},
                        {"type": "image", "image": pair[1]},
                        {"type": "text", "text": str(instruction)},
                    ],
                }
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        processed = self.processor(
            text=texts,
            images=image_pairs,
            padding=True,
            return_tensors="pt",
        )
        samples = {
            key: processed[key].to(self.device)
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
            if key in processed
        }
        state_tensors = torch.stack([normalize_state(state) for state in states], dim=0).to(self.device)
        return samples, state_tensors

    def predict_normalized_action_chunk(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
    ) -> np.ndarray:
        if isinstance(state_or_obs, dict):
            state = state_from_libero_obs(state_or_obs)
        else:
            state = np.asarray(state_or_obs, dtype=np.float32)

        samples, states = self._build_batch([primary_image], [wrist_image], [state], [instruction])
        samples = {
            key: value.to(dtype=self.model_dtype) if value.is_floating_point() else value
            for key, value in samples.items()
        }
        states = states.to(dtype=self.model_dtype)
        with torch.inference_mode():
            pred = self.model(samples, states)
        return sanitize_pred_chunk(pred.detach().float().cpu().numpy()[0])

    def predict_env_action_chunk(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
        execute_steps: int | None = None,
    ) -> np.ndarray:
        pred_norm = self.predict_normalized_action_chunk(primary_image, wrist_image, instruction, state_or_obs)
        if execute_steps is not None:
            pred_norm = pred_norm[: int(execute_steps)]
        return normalized_action_chunk_to_env_actions(pred_norm)

    def normalized_action_chunk_to_env_actions(self, chunk: np.ndarray) -> np.ndarray:
        return normalized_action_chunk_to_env_actions(chunk)


GroundingDINOAlignedPolicy = TurboVLAPolicy
