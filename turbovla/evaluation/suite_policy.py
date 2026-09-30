"""Suite-statistics TurboVLA policy adapter for LIBERO evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import policy as base


def _select_stats(payload: dict[str, Any], stats_key: str | None) -> dict[str, Any]:
    if stats_key:
        if stats_key not in payload:
            raise KeyError(f"stats_key={stats_key!r} not found in stats payload")
        stats = payload[stats_key]
    else:
        keys = [key for key in payload.keys() if key != "metadata"]
        if len(keys) != 1:
            raise KeyError(f"stats_key is required; available keys: {keys}")
        stats = payload[keys[0]]
    if not isinstance(stats, dict):
        raise ValueError("selected stats entry must be a dict")
    return stats


def _array(stats: dict[str, Any], section: str, name: str) -> np.ndarray:
    if section not in stats:
        raise KeyError(f"stats payload missing section {section!r}")
    value = stats[section].get(name)
    if value is None:
        raise KeyError(f"stats payload missing {section}.{name}")
    return np.asarray(value, dtype=np.float32)


class TurboVLAPolicy(base.TurboVLAPolicy):
    def __init__(
        self,
        *args,
        stats_path: str | Path,
        stats_key: str | None = None,
        normalize_binary_gripper: str = "auto",
        **kwargs,
    ) -> None:
        self.stats_path = str(stats_path)
        self.stats_key = stats_key
        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        stats = _select_stats(payload, stats_key)
        state_section = "proprio" if "proprio" in stats else "state"
        self.proprio_mean = _array(stats, state_section, "mean")
        self.proprio_std = _array(stats, state_section, "std")
        self.action_min = _array(stats, "action", "min")
        self.action_max = _array(stats, "action", "max")
        gripper_min = float(self.action_min[6])
        gripper_max = float(self.action_max[6])
        if normalize_binary_gripper == "auto":
            self.normalize_binary_gripper = gripper_min >= -1e-6 and gripper_max <= 1.0 + 1e-6
        elif normalize_binary_gripper in {"1", "true", "yes"}:
            self.normalize_binary_gripper = True
        elif normalize_binary_gripper in {"0", "false", "no"}:
            self.normalize_binary_gripper = False
        else:
            raise ValueError("normalize_binary_gripper must be auto, true, or false")

        super().__init__(*args, **kwargs)
        if self.verbose:
            print(
                "[TurboVLAPolicy] suite stats: "
                f"path={self.stats_path}, key={self.stats_key}, "
                f"normalize_binary_gripper={self.normalize_binary_gripper}",
                flush=True,
            )

    def _build_batch(
        self,
        primary_images,
        wrist_images,
        states,
        include_dinov3: bool = True,
    ):
        flat_images = []
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
        state_tensors = []
        for state in states:
            state_np = np.asarray(state, dtype=np.float32).reshape(-1)
            if state_np.shape[0] != base.STATE_DIM:
                raise ValueError(f"GroundingDINO state must have {base.STATE_DIM} dims, got {state_np.shape}")
            norm = (state_np - self.proprio_mean) / (self.proprio_std + 1e-6)
            state_tensors.append(torch.from_numpy(norm).float())
        return samples, torch.stack(state_tensors, dim=0).to(self.device)

    def normalized_action_to_env_action(self, row: np.ndarray) -> np.ndarray:
        return self.normalized_action_chunk_to_env_actions(row)[0]

    def normalized_action_chunk_to_env_actions(self, chunk: np.ndarray) -> np.ndarray:
        return base.normalized_action_chunk_to_env_actions(
            chunk,
            action_min=self.action_min,
            action_max=self.action_max,
        )

    def predict_env_action_chunk(
        self,
        primary_image,
        wrist_image,
        instruction: str,
        state_or_obs,
        execute_steps: int | None = None,
    ) -> np.ndarray:
        pred_norm = self.predict_normalized_action_chunk(primary_image, wrist_image, instruction, state_or_obs)
        if execute_steps is not None:
            pred_norm = pred_norm[: int(execute_steps)]
        return self.normalized_action_chunk_to_env_actions(pred_norm)


get_libero_dummy_action = base.get_libero_dummy_action
rotate_libero_image = base.rotate_libero_image
set_seed_everywhere = base.set_seed_everywhere
GroundingDINOAlignedPolicy = TurboVLAPolicy


class Qwen3VLVLAPolicy(base.Qwen3VLVLAPolicy):
    def __init__(
        self,
        *args,
        stats_path: str | Path,
        stats_key: str | None = None,
        normalize_binary_gripper: str = "auto",
        **kwargs,
    ) -> None:
        self.stats_path = str(stats_path)
        self.stats_key = stats_key
        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        stats = _select_stats(payload, stats_key)
        state_section = "proprio" if "proprio" in stats else "state"
        self.proprio_mean = _array(stats, state_section, "mean")
        self.proprio_std = _array(stats, state_section, "std")
        self.action_min = _array(stats, "action", "min")
        self.action_max = _array(stats, "action", "max")
        gripper_min = float(self.action_min[6])
        gripper_max = float(self.action_max[6])
        if normalize_binary_gripper == "auto":
            self.normalize_binary_gripper = gripper_min >= -1e-6 and gripper_max <= 1.0 + 1e-6
        elif normalize_binary_gripper in {"1", "true", "yes"}:
            self.normalize_binary_gripper = True
        elif normalize_binary_gripper in {"0", "false", "no"}:
            self.normalize_binary_gripper = False
        else:
            raise ValueError("normalize_binary_gripper must be auto, true, or false")

        super().__init__(*args, **kwargs)
        if self.verbose:
            print(
                "[Qwen3VLVLAPolicy] suite stats: "
                f"path={self.stats_path}, key={self.stats_key}, "
                f"normalize_binary_gripper={self.normalize_binary_gripper}",
                flush=True,
            )

    def _build_batch(
        self,
        primary_images,
        wrist_images,
        states,
        instructions,
    ):
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
        state_tensors = []
        for state in states:
            state_np = np.asarray(state, dtype=np.float32).reshape(-1)
            if state_np.shape[0] != base.STATE_DIM:
                raise ValueError(f"Qwen3-VL state must have {base.STATE_DIM} dims, got {state_np.shape}")
            norm = (state_np - self.proprio_mean) / (self.proprio_std + 1e-6)
            state_tensors.append(torch.from_numpy(norm).float())
        return samples, torch.stack(state_tensors, dim=0).to(self.device)

    def normalized_action_to_env_action(self, row: np.ndarray) -> np.ndarray:
        return self.normalized_action_chunk_to_env_actions(row)[0]

    def normalized_action_chunk_to_env_actions(self, chunk: np.ndarray) -> np.ndarray:
        return base.normalized_action_chunk_to_env_actions(
            chunk,
            action_min=self.action_min,
            action_max=self.action_max,
        )

    def predict_env_action_chunk(
        self,
        primary_image,
        wrist_image,
        instruction: str,
        state_or_obs,
        execute_steps: int | None = None,
    ) -> np.ndarray:
        pred_norm = self.predict_normalized_action_chunk(primary_image, wrist_image, instruction, state_or_obs)
        if execute_steps is not None:
            pred_norm = pred_norm[: int(execute_steps)]
        return self.normalized_action_chunk_to_env_actions(pred_norm)
