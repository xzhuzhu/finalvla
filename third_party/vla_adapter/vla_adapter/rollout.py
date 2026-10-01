"""VLA-Adapter-derived LIBERO evaluation for a TurboVLA checkpoint.

This script intentionally reuses the VLA-Adapter task/episode protocol shape,
but bypasses OpenVLA/Prismatic preprocessing and action unnormalization. The
policy adapter keeps GroundingDINO's 256px DINOv3 inputs, proprio stats, hard
action min/max, and gripper sign rule.

VLA-Adapter is MIT-licensed; see ../LICENSES/VLA-Adapter.txt.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, fields
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Optional

import imageio
import numpy as np
import tqdm


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
    "libero_object_with_mug": 280,
}

LIBERO_PRO_BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
LIBERO_PRO_PERTURBATIONS = ("object", "swap", "lan", "task", "env")
for _base_suite in LIBERO_PRO_BASE_SUITES:
    for _perturbation in LIBERO_PRO_PERTURBATIONS:
        TASK_MAX_STEPS[f"{_base_suite}_{_perturbation}"] = TASK_MAX_STEPS[_base_suite]

LIBERO_SUITES = tuple(TASK_MAX_STEPS.keys())


@dataclass
class GenerateConfig:
    ckpt_path: str = ""
    libero_root: str = ""
    dinov3_path: str = ""
    qwen3vl_path: str = ""
    bert_path: str = ""
    r3m_path: str = ""
    stats_path: str = "experiments/libero/configs/libero_all4_stats.json"
    stats_key: str = "libero_all4_no_noops"
    normalize_binary_gripper: str = "auto"
    allow_hf_download: bool = False

    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 50
    max_tasks: int = -1
    eval_slice_start: int = 0
    eval_slice_stride: int = 1
    eval_episode_slice_start: int = 0
    eval_episode_slice_stride: int = 1
    num_steps_wait: int = 10
    # The policy still predicts 12 actions; execute the validated 10-action
    # prefix before replanning so the final two actions provide overlap.
    num_open_loop_steps: int = 10
    # Refresh DINOv3 on every Nth policy query. R3M remains current on every
    # query; N=2 therefore alternates refresh/reuse while preserving live R3M.
    dinov3_update_interval: int = 1
    # Inference-only ablation: remove the four compact R3M history-memory
    # tokens while preserving current vision and the 12 state-history tokens.
    ablate_history_r3m_tokens: bool = False
    temporal_ensemble: bool = False
    temporal_ensemble_alpha: float = 0.1
    pace: bool = False
    pace_candidate_steps: str = "8,10,12"
    pace_smoothing_window: int = 3
    pace_prominence_threshold: float = 0.2
    pace_gripper_boundaries: bool = True
    env_img_res: int = 256
    seed: int = 42
    control_mode: str = "relative"
    mujoco_gl: str = "osmesa"
    pyopengl_platform: str = "osmesa"
    osmesa_library: str = ""

    save_video: bool = False
    video_out_path: str = "outputs/evaluation"
    result_json_path: str = ""
    log_path: str = ""
    dry_run_model_load: bool = False

    hidden_dim: int = 256
    nheads: int = 8
    dim_feedforward: int = 2048
    max_text_len: int = 256
    vla_feature_enhancer_layers: int = 6
    enhancer_inner_dim: int = 1024
    action_dim: int = 7
    chunk_size: int = 12
    state_dim: int = 8
    num_state_tokens: int = 2
    text_dropout: float = 0.0
    fusion_dropout: float = 0.0
    fusion_droppath: float = 0.1
    sub_sentence_present: bool = True
    precision: str = "bf16"
    text_padding_length: int = 21
    action_head: str = "act"
    flow_num_heads: int = 4
    flow_condition_layers: int = 4
    flow_dit_layers: int = 16
    flow_target_tokens: int = 16
    flow_static_tokens: int = 8
    flow_state_dim: int = 0


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> GenerateConfig:
    parser = argparse.ArgumentParser(
        description="VLA-Adapter-derived aligned LIBERO evaluation for TurboVLA checkpoints."
    )
    for field in fields(GenerateConfig):
        name = field.name
        default = field.default
        arg_name = f"--{name}"
        dashed_arg_name = f"--{name.replace('_', '-')}"
        kwargs = {"default": default, "help": f"default: {default}"}
        if isinstance(default, bool):
            parser.add_argument(arg_name, dashed_arg_name, type=_parse_bool, nargs="?", const=True, **kwargs)
            parser.add_argument(f"--no_{name}", f"--no-{name.replace('_', '-')}", dest=name, action="store_false")
        elif name == "precision":
            parser.add_argument(arg_name, dashed_arg_name, type=str, choices=("bf16", "fp32"), **kwargs)
        elif isinstance(default, int):
            parser.add_argument(arg_name, dashed_arg_name, type=int, **kwargs)
        elif isinstance(default, float):
            parser.add_argument(arg_name, dashed_arg_name, type=float, **kwargs)
        else:
            parser.add_argument(arg_name, dashed_arg_name, type=str, **kwargs)
    return GenerateConfig(**vars(parser.parse_args()))


def _import_turbovla_adapter():
    from turbovla.evaluation.suite_policy import (
        TurboVLAPolicy,
        get_libero_dummy_action,
        rotate_libero_image,
        set_seed_everywhere,
    )

    return TurboVLAPolicy, get_libero_dummy_action, rotate_libero_image, set_seed_everywhere


def _import_qwen3vl_adapter():
    from turbovla.evaluation.suite_policy import (
        Qwen3VLVLAPolicy,
        get_libero_dummy_action,
        rotate_libero_image,
        set_seed_everywhere,
    )

    return Qwen3VLVLAPolicy, get_libero_dummy_action, rotate_libero_image, set_seed_everywhere


def _ensure_libero_import_path(cfg: GenerateConfig) -> None:
    if cfg.mujoco_gl:
        os.environ.setdefault("MUJOCO_GL", cfg.mujoco_gl)
    if cfg.pyopengl_platform:
        os.environ.setdefault("PYOPENGL_PLATFORM", cfg.pyopengl_platform)
    if cfg.osmesa_library:
        os.environ.setdefault("OSMESA_LIBRARY", cfg.osmesa_library)

    candidates = []
    if cfg.libero_root:
        candidates.append(Path(cfg.libero_root))
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            return


def _setup_logging(cfg: GenerateConfig) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.log_path:
        Path(cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def _try_set_control_mode(env, control_mode: str) -> bool:
    candidates = [
        env,
        getattr(env, "env", None),
        getattr(env, "unwrapped", None),
        getattr(getattr(env, "env", None), "env", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        if hasattr(obj, "control_mode"):
            try:
                setattr(obj, "control_mode", control_mode)
                return True
            except Exception:
                pass
        if hasattr(obj, "set_control_mode"):
            try:
                obj.set_control_mode(control_mode)
                return True
            except Exception:
                pass
    return False


def _make_libero_env(task, cfg: GenerateConfig):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if task.problem_folder in LIBERO_SUITES and any(
        task.problem_folder.endswith(f"_{kind}") for kind in LIBERO_PRO_PERTURBATIONS
    ):
        match = re.search(r"\(:language\s+([^)]*?)\s*\)", task_bddl_file.read_text(encoding="utf-8"))
        if match is None:
            raise ValueError(f"Missing language instruction in {task_bddl_file}")
        task_description = match.group(1).strip()
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": cfg.env_img_res,
        "camera_widths": cfg.env_img_res,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(cfg.seed)
    _try_set_control_mode(env, cfg.control_mode)
    return env, task_description


def _save_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [np.asarray(x) for x in frames], fps=fps)


def _result_path(cfg: GenerateConfig) -> Path:
    if cfg.result_json_path:
        return Path(cfg.result_json_path)
    ckpt_tag = Path(cfg.ckpt_path).stem
    return Path(cfg.video_out_path) / f"{ckpt_tag}_{cfg.task_suite_name}_results.json"


class AlignedTemporalEnsembler:
    """Ensemble normalized action chunks aligned to the current environment step."""

    def __init__(self, chunk_size: int, alpha: float) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError(f"temporal_ensemble_alpha must be finite and nonnegative, got {alpha}")
        self.chunk_size = int(chunk_size)
        self.alpha = float(alpha)
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=self.chunk_size)

    def reset(self) -> None:
        self._history.clear()

    @property
    def history_size(self) -> int:
        return len(self._history)

    def add_chunk(self, query_time: int, chunk: np.ndarray) -> None:
        chunk_array = np.asarray(chunk, dtype=np.float32)
        if chunk_array.ndim != 2 or chunk_array.shape[1] != 7 or chunk_array.shape[0] == 0:
            raise ValueError(f"normalized action chunk must have shape (N, 7), N > 0; got {chunk_array.shape}")
        if not np.isfinite(chunk_array).all():
            raise ValueError("normalized action chunk contains non-finite values")
        self._history.append((int(query_time), chunk_array))

    def ensemble(self, current_time: int) -> np.ndarray:
        aligned = []
        for query_time, chunk in self._history:
            row_index = int(current_time) - query_time
            if 0 <= row_index < len(chunk):
                aligned.append(chunk[row_index])
        if not aligned:
            raise RuntimeError(f"no normalized predictions align with time {current_time}")

        predictions = np.stack(aligned, axis=0).astype(np.float64, copy=False)
        reference = next(
            (chunk[0] for query_time, chunk in reversed(self._history) if query_time == current_time),
            None,
        )
        if reference is None:
            raise RuntimeError(f"missing current normalized prediction for time {current_time}")
        reference = reference.astype(np.float64, copy=False)
        epsilon = np.finfo(np.float64).eps
        cosine = (predictions @ reference) / (
            np.maximum(np.linalg.norm(predictions, axis=1) * np.linalg.norm(reference), epsilon)
        )
        logits = self.alpha * cosine
        weights = np.exp(logits - np.max(logits))
        weights /= np.sum(weights)
        output = weights @ predictions
        if not np.isfinite(output).all():
            raise ValueError("temporal ensemble produced non-finite normalized action")
        return output.astype(np.float32, copy=False)


def parse_pace_candidate_steps(value: str, chunk_size: int | None = None) -> tuple[int, ...]:
    """Parse and validate the ordered execution horizons used by PACE."""
    try:
        steps = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"pace_candidate_steps must be comma-separated integers, got {value!r}") from exc
    if not steps:
        raise ValueError("pace_candidate_steps must contain at least one execution horizon")
    if any(step <= 0 for step in steps):
        raise ValueError(f"pace_candidate_steps must be positive, got {steps}")
    if tuple(sorted(set(steps))) != steps:
        raise ValueError(f"pace_candidate_steps must be strictly increasing and unique, got {steps}")
    if chunk_size is not None and steps[-1] > chunk_size:
        raise ValueError(
            f"largest PACE execution horizon {steps[-1]} exceeds chunk_size={chunk_size}"
        )
    return steps


class PACEHorizonSelector:
    """Select a phase-aware execution prefix from a normalized action chunk.

    LIBERO uses relative Cartesian action commands, so the norm of each
    translation/rotation command is the available action-space speed signal.
    A centered moving average suppresses short oscillations.  Pronounced local
    speed valleys and stable gripper transitions become replanning candidates;
    the earliest accepted configured horizon is used, otherwise the longest
    horizon is the conservative fallback.
    """

    def __init__(
        self,
        candidate_steps: tuple[int, ...],
        smoothing_window: int = 3,
        prominence_threshold: float = 0.2,
        gripper_boundaries: bool = True,
    ) -> None:
        if not candidate_steps or tuple(sorted(set(candidate_steps))) != tuple(candidate_steps):
            raise ValueError("candidate_steps must be positive, strictly increasing, and unique")
        if any(step <= 0 for step in candidate_steps):
            raise ValueError("candidate_steps must be positive, strictly increasing, and unique")
        if smoothing_window <= 0 or smoothing_window % 2 == 0:
            raise ValueError(f"smoothing_window must be a positive odd integer, got {smoothing_window}")
        if not np.isfinite(prominence_threshold) or not 0.0 <= prominence_threshold <= 1.0:
            raise ValueError(
                "prominence_threshold must be finite and in [0, 1], "
                f"got {prominence_threshold}"
            )
        self.candidate_steps = tuple(int(step) for step in candidate_steps)
        self.smoothing_window = int(smoothing_window)
        self.prominence_threshold = float(prominence_threshold)
        self.gripper_boundaries = bool(gripper_boundaries)

    def _smoothed_speed(self, chunk: np.ndarray) -> np.ndarray:
        translation_speed = np.linalg.norm(chunk[:, :3], axis=1) / np.sqrt(3.0)
        rotation_speed = np.linalg.norm(chunk[:, 3:6], axis=1) / np.sqrt(3.0)
        speed = np.sqrt(0.5 * (translation_speed**2 + rotation_speed**2))
        if self.smoothing_window == 1 or len(speed) == 1:
            return speed
        radius = self.smoothing_window // 2
        padded = np.pad(speed, (radius, radius), mode="edge")
        kernel = np.full(self.smoothing_window, 1.0 / self.smoothing_window)
        return np.convolve(padded, kernel, mode="valid")

    def _valley_candidates(self, speed: np.ndarray) -> set[int]:
        accepted: set[int] = set()
        epsilon = np.finfo(np.float64).eps
        radius = max(1, self.smoothing_window)
        for horizon in self.candidate_steps[:-1]:
            index = horizon - 1
            if index <= 0 or index >= len(speed) - 1:
                continue
            if speed[index] > speed[index - 1] or speed[index] > speed[index + 1]:
                continue
            left_peak = float(np.max(speed[max(0, index - radius) : index]))
            right_peak = float(np.max(speed[index + 1 : min(len(speed), index + radius + 1)]))
            shoulder = min(left_peak, right_peak)
            prominence = max(0.0, shoulder - float(speed[index])) / max(shoulder, epsilon)
            if prominence >= self.prominence_threshold:
                accepted.add(horizon)
        return accepted

    def _gripper_candidates(self, chunk: np.ndarray) -> set[int]:
        if not self.gripper_boundaries or chunk.shape[1] < 7:
            return set()
        gripper = chunk[:, 6]
        signs = np.where(gripper >= 0.0, 1, -1)
        accepted: set[int] = set()
        for index in range(1, min(len(chunk), self.candidate_steps[-1])):
            if signs[index] == signs[index - 1]:
                continue
            if abs(float(gripper[index - 1])) < 0.25 or abs(float(gripper[index])) < 0.25:
                continue
            stable_end = min(len(chunk), index + 3)
            if not np.all(signs[index:stable_end] == signs[index]):
                continue
            boundary = index + 1
            mapped = next((step for step in self.candidate_steps if step >= boundary), None)
            if mapped is not None and mapped < self.candidate_steps[-1]:
                accepted.add(mapped)
            break
        return accepted

    def select(self, normalized_chunk: np.ndarray) -> int:
        chunk = np.asarray(normalized_chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] < 6 or len(chunk) == 0:
            raise ValueError(f"normalized action chunk must have shape (N, D>=6), got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("normalized action chunk contains non-finite values")
        if self.candidate_steps[-1] > len(chunk):
            raise ValueError(
                f"largest PACE execution horizon {self.candidate_steps[-1]} exceeds chunk length {len(chunk)}"
            )
        speed = self._smoothed_speed(chunk[: self.candidate_steps[-1]])
        accepted = self._valley_candidates(speed) | self._gripper_candidates(chunk)
        return min(accepted) if accepted else self.candidate_steps[-1]


def _run_episode(
    cfg: GenerateConfig,
    env,
    policy,
    task_description: str,
    initial_state: np.ndarray,
    get_libero_dummy_action,
    rotate_libero_image,
) -> tuple[bool, list[np.ndarray], list[int]]:
    env.reset()
    obs = env.set_init_state(initial_state)
    reset_history = getattr(policy, "reset_history", None)
    if callable(reset_history):
        reset_history()
    action_queue: deque[np.ndarray] = deque()
    temporal_ensembler = (
        AlignedTemporalEnsembler(cfg.chunk_size, cfg.temporal_ensemble_alpha)
        if cfg.temporal_ensemble
        else None
    )
    pace_selector = (
        PACEHorizonSelector(
            parse_pace_candidate_steps(cfg.pace_candidate_steps, cfg.chunk_size),
            smoothing_window=cfg.pace_smoothing_window,
            prominence_threshold=cfg.pace_prominence_threshold,
            gripper_boundaries=cfg.pace_gripper_boundaries,
        )
        if cfg.pace
        else None
    )
    replay_images: list[np.ndarray] = []
    query_horizons: list[int] = []
    dummy_action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    success = False
    for t in range(max_steps + cfg.num_steps_wait):
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(dummy_action.tolist())
            continue

        needs_policy_images = temporal_ensembler is not None or not action_queue
        if needs_policy_images or cfg.save_video:
            primary = rotate_libero_image(obs["agentview_image"])
            wrist = rotate_libero_image(obs["robot0_eye_in_hand_image"])
            if cfg.save_video:
                replay_images.append(np.concatenate([primary, wrist], axis=1))

        if temporal_ensembler is not None:
            query_time = t - cfg.num_steps_wait
            normalized_chunk = policy.predict_normalized_action_chunk(primary, wrist, task_description, obs)
            temporal_ensembler.add_chunk(query_time, normalized_chunk)
            normalized_action = temporal_ensembler.ensemble(query_time)
            action = policy.normalized_action_to_env_action(normalized_action)
            query_horizons.append(1)
        else:
            if not action_queue:
                if pace_selector is not None:
                    normalized_chunk = policy.predict_normalized_action_chunk(
                        primary, wrist, task_description, obs
                    )
                    execution_horizon = pace_selector.select(normalized_chunk)
                    env_actions = policy.normalized_action_chunk_to_env_actions(
                        normalized_chunk[:execution_horizon]
                    )
                else:
                    env_actions = policy.predict_env_action_chunk(
                        primary,
                        wrist,
                        task_description,
                        obs,
                        execute_steps=cfg.num_open_loop_steps,
                    )
                query_horizons.append(len(env_actions))
                action_queue.extend(np.asarray(row, dtype=np.float32) for row in env_actions)

            action = action_queue.popleft() if action_queue else dummy_action
        executed_from_obs = obs
        obs, _, done, _ = env.step(action.tolist())
        record_history_state = getattr(policy, "record_history_state", None)
        if callable(record_history_state):
            record_history_state(executed_from_obs)
        else:
            # Compatibility for policies that still track executed actions.
            record_executed = getattr(policy, "record_executed", None)
            if callable(record_executed):
                record_executed(executed_from_obs, action)
        if done:
            success = True
            break

    return success, replay_images, query_horizons


def eval_libero(cfg: GenerateConfig) -> float:
    _setup_logging(cfg)
    if not cfg.ckpt_path:
        raise ValueError("--ckpt_path is required for TurboVLA evaluation.")
    if not Path(cfg.ckpt_path).exists():
        raise FileNotFoundError(f"TurboVLA checkpoint not found: {cfg.ckpt_path}")
    if cfg.qwen3vl_path:
        if not Path(cfg.qwen3vl_path).is_dir():
            raise FileNotFoundError(f"Qwen3-VL directory not found: {cfg.qwen3vl_path}")
    else:
        if not cfg.dinov3_path:
            raise ValueError("--dinov3_path is required (or use --qwen3vl-path)")
        if not cfg.bert_path:
            raise ValueError("--bert_path is required")
        if cfg.r3m_path and not Path(cfg.r3m_path).is_file():
            raise FileNotFoundError(f"R3M checkpoint not found: {cfg.r3m_path}")
    if not Path(cfg.stats_path).is_file():
        raise FileNotFoundError(f"stats file not found: {cfg.stats_path}")
    if cfg.task_suite_name not in LIBERO_SUITES:
        raise ValueError(f"Unknown task suite {cfg.task_suite_name}; choose from {LIBERO_SUITES}")
    if cfg.dinov3_update_interval < 1:
        raise ValueError("dinov3_update_interval must be positive")
    if cfg.qwen3vl_path and cfg.dinov3_update_interval != 1:
        raise ValueError("dinov3_update_interval is only supported by the DINOv3 policy")

    pace_candidate_steps = parse_pace_candidate_steps(
        cfg.pace_candidate_steps,
        cfg.chunk_size if cfg.pace else None,
    )
    cfg.pace_candidate_steps = ",".join(str(step) for step in pace_candidate_steps)

    if cfg.temporal_ensemble:
        if cfg.num_open_loop_steps != cfg.chunk_size:
            raise ValueError(
                "temporal_ensemble requires num_open_loop_steps to equal chunk_size; "
                f"got {cfg.num_open_loop_steps} and {cfg.chunk_size}"
            )
        if not np.isfinite(cfg.temporal_ensemble_alpha) or cfg.temporal_ensemble_alpha < 0:
            raise ValueError(
                "temporal_ensemble_alpha must be finite and nonnegative, "
                f"got {cfg.temporal_ensemble_alpha}"
            )

    if cfg.pace:
        if cfg.temporal_ensemble:
            raise ValueError("pace and temporal_ensemble are mutually exclusive")
        # PACE owns the execution horizon. Canonicalize the otherwise-unused
        # fixed-horizon option to the selector's conservative fallback so
        # summaries and launcher protocol matching remain unambiguous.
        cfg.num_open_loop_steps = pace_candidate_steps[-1]
        # Construct once here to validate all selector hyperparameters before
        # loading the model or creating simulator processes.
        PACEHorizonSelector(
            pace_candidate_steps,
            smoothing_window=cfg.pace_smoothing_window,
            prominence_threshold=cfg.pace_prominence_threshold,
            gripper_boundaries=cfg.pace_gripper_boundaries,
        )

    if not cfg.pace and cfg.num_open_loop_steps != cfg.chunk_size:
        logging.warning(
            "Executing %s actions from each %s-action prediction before replanning.",
            cfg.num_open_loop_steps,
            cfg.chunk_size,
        )

    if cfg.qwen3vl_path:
        (
            PolicyClass,
            get_libero_dummy_action,
            rotate_libero_image,
            set_seed_everywhere,
        ) = _import_qwen3vl_adapter()
    else:
        (
            PolicyClass,
            get_libero_dummy_action,
            rotate_libero_image,
            set_seed_everywhere,
        ) = _import_turbovla_adapter()
    set_seed_everywhere(cfg.seed)

    logging.info("Loading TurboVLA policy from %s", cfg.ckpt_path)
    if cfg.qwen3vl_path:
        policy = PolicyClass(
            ckpt_path=cfg.ckpt_path,
            qwen3vl_path=cfg.qwen3vl_path,
            stats_path=cfg.stats_path,
            stats_key=cfg.stats_key,
            normalize_binary_gripper=cfg.normalize_binary_gripper,
            allow_hf_download=cfg.allow_hf_download,
            hidden_dim=cfg.hidden_dim,
            nheads=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            action_dim=cfg.action_dim,
            chunk_size=cfg.chunk_size,
            state_dim=cfg.state_dim,
            num_state_tokens=cfg.num_state_tokens,
            precision=cfg.precision,
        )
    else:
        policy = PolicyClass(
            ckpt_path=cfg.ckpt_path,
            dinov3_path=cfg.dinov3_path,
            bert_path=cfg.bert_path,
            r3m_path=cfg.r3m_path,
            stats_path=cfg.stats_path,
            stats_key=cfg.stats_key,
            normalize_binary_gripper=cfg.normalize_binary_gripper,
            allow_hf_download=cfg.allow_hf_download,
            hidden_dim=cfg.hidden_dim,
            nheads=cfg.nheads,
            dim_feedforward=cfg.dim_feedforward,
            max_text_len=cfg.max_text_len,
            text_padding_length=cfg.text_padding_length,
            vla_feature_enhancer_layers=cfg.vla_feature_enhancer_layers,
            enhancer_inner_dim=cfg.enhancer_inner_dim,
            action_dim=cfg.action_dim,
            chunk_size=cfg.chunk_size,
            state_dim=cfg.state_dim,
            num_state_tokens=cfg.num_state_tokens,
            text_dropout=cfg.text_dropout,
            fusion_dropout=cfg.fusion_dropout,
            fusion_droppath=cfg.fusion_droppath,
            sub_sentence_present=cfg.sub_sentence_present,
            precision=cfg.precision,
            action_head=cfg.action_head,
            flow_num_heads=cfg.flow_num_heads,
            flow_condition_layers=cfg.flow_condition_layers,
            flow_dit_layers=cfg.flow_dit_layers,
            flow_target_tokens=cfg.flow_target_tokens,
            flow_static_tokens=cfg.flow_static_tokens,
            flow_state_dim=cfg.flow_state_dim,
            dinov3_update_interval=cfg.dinov3_update_interval,
            ablate_history_r3m_tokens=cfg.ablate_history_r3m_tokens,
        )

    if cfg.dry_run_model_load:
        logging.info("dry_run_model_load=True; exiting before LIBERO rollout.")
        return 0.0

    _ensure_libero_import_path(cfg)
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    n_tasks = task_suite.n_tasks if cfg.max_tasks <= 0 else min(cfg.max_tasks, task_suite.n_tasks)

    summary = {
        "script": "turbovla_libero_evaluation",
        "ckpt_path": cfg.ckpt_path,
        "task_suite_name": cfg.task_suite_name,
        "num_trials_per_task": cfg.num_trials_per_task,
        "chunk_size": cfg.chunk_size,
        "num_open_loop_steps": cfg.num_open_loop_steps,
        "dinov3_update_interval": cfg.dinov3_update_interval,
        "r3m_update_interval": 1,
        "ablate_history_r3m_tokens": cfg.ablate_history_r3m_tokens,
        "temporal_ensemble": cfg.temporal_ensemble,
        "temporal_ensemble_alpha": cfg.temporal_ensemble_alpha,
        "pace": cfg.pace,
        "pace_candidate_steps": cfg.pace_candidate_steps,
        "pace_smoothing_window": cfg.pace_smoothing_window,
        "pace_prominence_threshold": cfg.pace_prominence_threshold,
        "pace_gripper_boundaries": cfg.pace_gripper_boundaries,
        "query_interval": "adaptive" if cfg.pace else (1 if cfg.temporal_ensemble else cfg.num_open_loop_steps),
        "seed": cfg.seed,
        "precision": cfg.precision,
        "eval_slice_start": cfg.eval_slice_start,
        "eval_slice_stride": cfg.eval_slice_stride,
        "eval_episode_slice_start": cfg.eval_episode_slice_start,
        "eval_episode_slice_stride": cfg.eval_episode_slice_stride,
        "tasks": [],
        "total_episodes": 0,
        "total_successes": 0,
        "overall_success_rate": 0.0,
        "policy_queries": 0,
        "execution_horizon_counts": {},
        "mean_execution_horizon": 0.0,
    }

    total_episodes = 0
    total_successes = 0
    total_horizon_counts: Counter[int] = Counter()
    video_root = Path(cfg.video_out_path) / Path(cfg.ckpt_path).stem / cfg.task_suite_name
    if cfg.save_video:
        video_root.mkdir(parents=True, exist_ok=True)

    task_ids = list(range(cfg.eval_slice_start, n_tasks, cfg.eval_slice_stride))
    if not task_ids:
        raise ValueError(
            f"eval slice start={cfg.eval_slice_start} stride={cfg.eval_slice_stride} "
            f"selects no tasks out of {n_tasks}"
        )
    episode_ids = list(
        range(
            cfg.eval_episode_slice_start,
            cfg.num_trials_per_task,
            cfg.eval_episode_slice_stride,
        )
    )
    if not episode_ids:
        raise ValueError(
            f"episode slice start={cfg.eval_episode_slice_start} "
            f"stride={cfg.eval_episode_slice_stride} selects no episodes out of "
            f"{cfg.num_trials_per_task}"
        )
    for task_id in tqdm.tqdm(task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if cfg.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"{cfg.task_suite_name} task {task_id} has only {len(initial_states)} initial states, "
                f"but num_trials_per_task={cfg.num_trials_per_task}"
            )

        env, task_description = _make_libero_env(task, cfg)
        task_successes = 0
        task_episodes = 0
        task_horizon_counts: Counter[int] = Counter()
        try:
            for episode_idx in tqdm.tqdm(episode_ids, desc=f"task {task_id}", leave=False):
                success, replay_images, query_horizons = _run_episode(
                    cfg,
                    env,
                    policy,
                    task_description,
                    initial_states[episode_idx],
                    get_libero_dummy_action,
                    rotate_libero_image,
                )
                task_episodes += 1
                total_episodes += 1
                if success:
                    task_successes += 1
                    total_successes += 1
                task_horizon_counts.update(query_horizons)
                total_horizon_counts.update(query_horizons)

                suffix = "success" if success else "failure"
                logging.info(
                    "task=%d episode=%d success=%s running=%d/%d %.1f%%",
                    task_id,
                    episode_idx,
                    success,
                    total_successes,
                    total_episodes,
                    100.0 * total_successes / max(total_episodes, 1),
                )
                if cfg.save_video:
                    _save_video(video_root / f"task{task_id:02d}_ep{episode_idx:03d}_{suffix}.mp4", replay_images)
        finally:
            env.close()

        task_rate = task_successes / max(task_episodes, 1)
        summary["tasks"].append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_rate,
                "policy_queries": sum(task_horizon_counts.values()),
                "execution_horizon_counts": {
                    str(horizon): count for horizon, count in sorted(task_horizon_counts.items())
                },
                "mean_execution_horizon": (
                    sum(horizon * count for horizon, count in task_horizon_counts.items())
                    / max(sum(task_horizon_counts.values()), 1)
                ),
            }
        )
        logging.info("task=%d success_rate=%.4f (%d/%d)", task_id, task_rate, task_successes, task_episodes)

    final_rate = total_successes / max(total_episodes, 1)
    summary["total_episodes"] = total_episodes
    summary["total_successes"] = total_successes
    summary["overall_success_rate"] = final_rate
    summary["policy_queries"] = sum(total_horizon_counts.values())
    summary["execution_horizon_counts"] = {
        str(horizon): count for horizon, count in sorted(total_horizon_counts.items())
    }
    summary["mean_execution_horizon"] = (
        sum(horizon * count for horizon, count in total_horizon_counts.items())
        / max(sum(total_horizon_counts.values()), 1)
    )

    result_path = _result_path(cfg)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Final success rate: %.4f (%d/%d)", final_rate, total_successes, total_episodes)
    logging.info("Saved result json: %s", result_path)
    return final_rate


def main():
    eval_libero(parse_args())


if __name__ == "__main__":
    main()
