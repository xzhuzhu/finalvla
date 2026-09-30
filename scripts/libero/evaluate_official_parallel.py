#!/usr/bin/env python3
"""Parallel episode-sliced LIBERO evaluation for official TurboVLA checkpoints.

Each checkpoint is evaluated by ``len(gpus)`` processes; every process handles
a strided subset of episodes for every task and the slice results are aggregated
into ``summary.json``.  With ``--watch`` the
script monitors the checkpoint directory and evaluates every new checkpoint
that does not yet have a summary (mirroring the old evaluation monitor).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_KEYS = (
    "chunk_size",
    "num_open_loop_steps",
    "dinov3_update_interval",
    "r3m_update_interval",
    "ablate_history_r3m_tokens",
    "temporal_ensemble",
    "temporal_ensemble_alpha",
    "pace",
    "pace_candidate_steps",
    "pace_smoothing_window",
    "pace_prominence_threshold",
    "pace_gripper_boundaries",
    "query_interval",
)
PROTOCOL_DEFAULTS = {
    "pace": False,
    "pace_candidate_steps": "8,10,12",
    "pace_smoothing_window": 3,
    "pace_prominence_threshold": 0.2,
    "pace_gripper_boundaries": True,
    "dinov3_update_interval": 1,
    "r3m_update_interval": 1,
    "ablate_history_r3m_tokens": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs/libero_official")
    parser.add_argument("--checkpoint-prefix", type=str, default="turbovla_libero_official_step")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--qwen3vl-path", type=Path, default=None, help="Enable Qwen3-VL backbone evaluation.")
    parser.add_argument("--gpus", default="1,2,1,2", help="Comma-separated GPU ids; one eval process per entry.")
    parser.add_argument(
        "--egl-device-id",
        type=int,
        default=None,
        help="Host EGL device ID; defaults to each worker's CUDA GPU ID.",
    )
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--bert-path", type=Path, default=ROOT / "pretrained/bert-base-uncased")
    parser.add_argument("--r3m-path", type=Path, default=ROOT / "pretrained/r3m-resnet18/backbone.pth")
    parser.add_argument("--stats-path", type=Path, default=ROOT / "experiments/libero/configs/libero_all4_stats.json")
    parser.add_argument("--stats-key", default="libero_all4_no_noops")
    parser.add_argument("--dinov3-path", type=Path, default=ROOT / "pretrained/dinov3-vitb16")
    parser.add_argument("--action-head", default="act", choices=["act", "flow_matching"])
    parser.add_argument("--flow-num-heads", type=int, default=4)
    parser.add_argument("--flow-condition-layers", type=int, default=4)
    parser.add_argument("--flow-dit-layers", type=int, default=16)
    parser.add_argument("--flow-target-tokens", type=int, default=16)
    parser.add_argument("--flow-static-tokens", type=int, default=8)
    parser.add_argument("--flow-state-dim", type=int, default=0)
    parser.add_argument(
        "--dinov3-update-interval",
        type=int,
        default=1,
        help="Refresh DINOv3 tokens every N policy queries; R3M always refreshes.",
    )
    parser.add_argument(
        "--ablate-history-r3m-tokens",
        action="store_true",
        help="Remove the four R3M history-memory tokens during inference.",
    )
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument(
        "--num-open-loop-steps",
        type=int,
        default=10,
        help="Actions executed before replanning (default: 10; prediction chunk remains 12).",
    )
    parser.add_argument(
        "--num-open-loop-steps-sweep",
        default="",
        help="Sequentially evaluate comma-separated fixed execution horizons, e.g. 8,10,12.",
    )
    parser.add_argument("--temporal-ensemble", action="store_true", help="Ensemble aligned normalized chunks every environment step.")
    parser.add_argument("--temporal-ensemble-alpha", type=float, default=0.1)
    parser.add_argument("--pace", action="store_true", help="Select an execution horizon from phase-aware speed valleys.")
    parser.add_argument("--pace-candidate-steps", default="8,10,12")
    parser.add_argument("--pace-smoothing-window", type=int, default=3)
    parser.add_argument("--pace-prominence-threshold", type=float, default=0.2)
    parser.set_defaults(pace_gripper_boundaries=True)
    parser.add_argument(
        "--pace-gripper-boundaries",
        dest="pace_gripper_boundaries",
        action="store_true",
        help="Treat stable gripper transitions as PACE boundary candidates (default).",
    )
    parser.add_argument(
        "--no-pace-gripper-boundaries",
        dest="pace_gripper_boundaries",
        action="store_false",
    )
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--eval-output-root", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--watch", action="store_true", help="Monitor checkpoint dir and evaluate new checkpoints.")
    parser.add_argument("--sleep-seconds", type=float, default=60.0)
    return parser.parse_args()


def _parse_positive_int_csv(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers, got {value!r}") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers, got {values}")
    if tuple(sorted(set(values))) != values:
        raise ValueError(f"{name} must be strictly increasing and unique, got {values}")
    return values


def _canonical_pace_candidate_steps(args: argparse.Namespace) -> str:
    values = _parse_positive_int_csv(args.pace_candidate_steps, "pace_candidate_steps")
    if args.pace and values[-1] > args.chunk_size:
        raise ValueError(
            f"largest PACE candidate {values[-1]} exceeds chunk_size={args.chunk_size}"
        )
    return ",".join(str(value) for value in values)


def _effective_open_loop_steps(args: argparse.Namespace) -> int:
    """Return the fixed horizon or PACE's canonical fallback horizon."""
    if args.pace:
        return int(_canonical_pace_candidate_steps(args).split(",")[-1])
    return int(args.num_open_loop_steps)


def discovered_checkpoints(args: argparse.Namespace) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(args.checkpoint_prefix)}_(\d+)\.pth$")
    candidates = []
    for path in args.checkpoint_dir.glob(f"{args.checkpoint_prefix}_*.pth"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return sorted(candidates)


def _protocol_suffix(args: argparse.Namespace) -> str:
    suffixes = []
    if args.chunk_size != 12:
        suffixes.append(f"chunk{args.chunk_size}")
    if not args.pace and args.num_open_loop_steps != args.chunk_size:
        suffixes.append(f"openloop{args.num_open_loop_steps}")
    if args.dinov3_update_interval != 1:
        suffixes.append(f"dinov3every{args.dinov3_update_interval}_r3mevery1")
    if args.ablate_history_r3m_tokens:
        suffixes.append("ablatehistoryr3m")
    if args.temporal_ensemble:
        alpha = format(args.temporal_ensemble_alpha, ".12g")
        suffixes.append(f"temporal_ensemble_a{alpha.replace('-', 'm').replace('.', 'p').replace('+', '')}")
    if args.pace:
        candidates = _canonical_pace_candidate_steps(args).replace(",", "-")
        prominence = format(args.pace_prominence_threshold, ".12g").replace("-", "m").replace(".", "p")
        gripper = "gripper" if args.pace_gripper_boundaries else "nogripper"
        suffixes.append(
            f"pace_h{candidates}_w{args.pace_smoothing_window}_p{prominence}_{gripper}"
        )
    return "" if not suffixes else "_" + "_".join(suffixes)


def expected_protocol(args: argparse.Namespace) -> dict[str, int | float | bool | str]:
    open_loop_steps = _effective_open_loop_steps(args)
    return {
        "chunk_size": int(args.chunk_size),
        "num_open_loop_steps": open_loop_steps,
        "dinov3_update_interval": int(args.dinov3_update_interval),
        "r3m_update_interval": 1,
        "ablate_history_r3m_tokens": bool(args.ablate_history_r3m_tokens),
        "temporal_ensemble": bool(args.temporal_ensemble),
        "temporal_ensemble_alpha": float(args.temporal_ensemble_alpha),
        "pace": bool(args.pace),
        "pace_candidate_steps": _canonical_pace_candidate_steps(args),
        "pace_smoothing_window": int(args.pace_smoothing_window),
        "pace_prominence_threshold": float(args.pace_prominence_threshold),
        "pace_gripper_boundaries": bool(args.pace_gripper_boundaries),
        "query_interval": "adaptive" if args.pace else (1 if args.temporal_ensemble else open_loop_steps),
    }


def _protocol_matches(payload: object, expected: dict[str, int | float | bool | str]) -> bool:
    if not isinstance(payload, dict):
        return False
    actual = {
        key: payload[key] if key in payload else PROTOCOL_DEFAULTS.get(key)
        for key in PROTOCOL_KEYS
    }
    if any(value is None for value in actual.values()):
        return False
    return actual == expected and all(type(actual[key]) is type(expected[key]) for key in PROTOCOL_KEYS)


def _step_dir(args: argparse.Namespace, step: int) -> Path:
    slice_count = len([item for item in args.gpus.split(",") if item.strip()])
    return (
        args.eval_output_root
        / f"{args.checkpoint_prefix}_{slice_count}slice_checkpoints{_protocol_suffix(args)}"
        / f"step_{step:06d}"
    )


def _worker_env(
    gpu: str,
    base_env: dict[str, str] | None = None,
    egl_device_id: int | None = None,
) -> dict[str, str]:
    """Build an isolated worker environment for one physical CUDA device.

    ``MUJOCO_EGL_DEVICE_ID`` indexes the host's physical EGL devices; it does
    not follow PyTorch's ``CUDA_VISIBLE_DEVICES`` remapping.  With no explicit
    EGL device, retain the historical one-device CUDA/EGL mapping.  For a
    differing EGL device, append its host ID to CUDA visibility: CUDA keeps the
    requested inference GPU at logical ``cuda:0``, while invalid/non-CUDA EGL
    enumerations (such as a host EGL device 8) are ignored by CUDA and remain
    usable by MuJoCo.
    """
    env = dict(os.environ if base_env is None else base_env)
    cpu_threads = env.get("TURBOVLA_EVAL_CPU_THREADS", "4")
    egl_device = gpu if egl_device_id is None else str(egl_device_id)
    cuda_visible_devices = gpu
    if egl_device_id is not None and egl_device != gpu:
        cuda_visible_devices = f"{gpu},{egl_device}"
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "MUJOCO_EGL_DEVICE_ID": egl_device,
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "PYTHONUNBUFFERED": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            # Multiple rollout workers otherwise each create a machine-wide
            # BLAS/OpenMP pool and oversubscribe large multi-core hosts.
            "OMP_NUM_THREADS": cpu_threads,
            "MKL_NUM_THREADS": cpu_threads,
            "OPENBLAS_NUM_THREADS": cpu_threads,
            "NUMEXPR_NUM_THREADS": cpu_threads,
        }
    )
    return env


def already_evaluated(args: argparse.Namespace, step: int) -> bool:
    summary_path = _step_dir(args, step) / "summary.json"
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return _protocol_matches(payload, expected_protocol(args))


def aggregate(args: argparse.Namespace, ckpt: Path, result_paths: list[Path], output: Path) -> dict:
    task_totals: dict[int, dict] = {}
    total_episodes = 0
    total_successes = 0
    total_policy_queries = 0
    total_horizon_counts: dict[int, int] = {}
    protocol = expected_protocol(args)
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not _protocol_matches(payload, protocol):
            slice_protocol = (
                {key: payload.get(key) for key in PROTOCOL_KEYS}
                if isinstance(payload, dict)
                else payload
            )
            raise ValueError(
                f"slice protocol mismatch: {path} has {slice_protocol}, expected {protocol}"
            )
        total_episodes += int(payload["total_episodes"])
        total_successes += int(payload["total_successes"])
        total_policy_queries += int(payload.get("policy_queries", 0))
        for horizon, count in payload.get("execution_horizon_counts", {}).items():
            horizon_value = int(horizon)
            total_horizon_counts[horizon_value] = total_horizon_counts.get(horizon_value, 0) + int(count)
        for task in payload["tasks"]:
            task_id = int(task["task_id"])
            current = task_totals.setdefault(
                task_id,
                {
                    "task_id": task_id,
                    "task_description": task["task_description"],
                    "episodes": 0,
                    "successes": 0,
                    "policy_queries": 0,
                    "execution_horizon_counts": {},
                },
            )
            current["episodes"] += int(task["episodes"])
            current["successes"] += int(task["successes"])
            current["policy_queries"] += int(task.get("policy_queries", 0))
            for horizon, count in task.get("execution_horizon_counts", {}).items():
                horizon_value = int(horizon)
                horizon_counts = current["execution_horizon_counts"]
                horizon_counts[horizon_value] = horizon_counts.get(horizon_value, 0) + int(count)
    tasks = []
    for task_id in sorted(task_totals):
        task = task_totals[task_id]
        task["success_rate"] = task["successes"] / max(task["episodes"], 1)
        task["mean_execution_horizon"] = (
            sum(horizon * count for horizon, count in task["execution_horizon_counts"].items())
            / max(task["policy_queries"], 1)
        )
        task["execution_horizon_counts"] = {
            str(horizon): count for horizon, count in sorted(task["execution_horizon_counts"].items())
        }
        tasks.append(task)
    summary = {
        "script": "turbovla_official_parallel_evaluation",
        "ckpt_path": str(ckpt.resolve()),
        "slice_results": [str(path.resolve()) for path in result_paths],
        "tasks": tasks,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": total_successes / max(total_episodes, 1),
        "policy_queries": total_policy_queries,
        "execution_horizon_counts": {
            str(horizon): count for horizon, count in sorted(total_horizon_counts.items())
        },
        "mean_execution_horizon": (
            sum(horizon * count for horizon, count in total_horizon_counts.items())
            / max(total_policy_queries, 1)
        ),
    }
    summary.update(protocol)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def evaluate_one(args: argparse.Namespace, ckpt: Path) -> Path:
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    step_dir = _step_dir(args, int(ckpt.stem.split('_')[-1]))
    step_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    result_paths = []
    open_loop_steps = _effective_open_loop_steps(args)
    for slice_idx, gpu in enumerate(gpus):
        result_path = step_dir / f"slice{slice_idx}-of-{len(gpus)}.json"
        log_path = step_dir / f"slice{slice_idx}-gpu{gpu}.log"
        result_paths.append(result_path)
        if args.qwen3vl_path is not None:
            cmd = [
                sys.executable,
                "experiments/libero/evaluate.py",
                "--ckpt_path", str(ckpt),
                "--qwen3vl_path", str(args.qwen3vl_path.resolve()),
                "--no_allow_hf_download",
                "--stats_path", str(args.stats_path.resolve()),
                "--stats_key", args.stats_key,
                "--task_suite_name", args.task_suite_name,
                "--num_trials_per_task", str(args.num_trials_per_task),
                "--num_open_loop_steps", str(open_loop_steps),
                "--chunk_size", str(args.chunk_size),
                "--temporal_ensemble_alpha", str(args.temporal_ensemble_alpha),
                "--pace_candidate_steps", _canonical_pace_candidate_steps(args),
                "--pace_smoothing_window", str(args.pace_smoothing_window),
                "--pace_prominence_threshold", str(args.pace_prominence_threshold),
                "--precision", args.precision,
                "--eval_episode_slice_start", str(slice_idx),
                "--eval_episode_slice_stride", str(len(gpus)),
                "--result_json_path", str(result_path),
            ]
        else:
            cmd = [
                sys.executable,
                "experiments/libero/evaluate.py",
                "--ckpt_path", str(ckpt),
                "--dinov3_path", str(args.dinov3_path.resolve()),
                "--no_allow_hf_download",
                "--bert_path", str(args.bert_path.resolve()),
                "--r3m_path", str(args.r3m_path.resolve()),
                "--stats_path", str(args.stats_path.resolve()),
                "--stats_key", args.stats_key,
                "--action_head", args.action_head,
                "--flow_num_heads", str(args.flow_num_heads),
                "--flow_condition_layers", str(args.flow_condition_layers),
                "--flow_dit_layers", str(args.flow_dit_layers),
                "--flow_target_tokens", str(args.flow_target_tokens),
                "--flow_static_tokens", str(args.flow_static_tokens),
                "--flow_state_dim", str(args.flow_state_dim),
                "--dinov3_update_interval", str(args.dinov3_update_interval),
                "--task_suite_name", args.task_suite_name,
                "--num_trials_per_task", str(args.num_trials_per_task),
                "--num_open_loop_steps", str(open_loop_steps),
                "--chunk_size", str(args.chunk_size),
                "--temporal_ensemble_alpha", str(args.temporal_ensemble_alpha),
                "--pace_candidate_steps", _canonical_pace_candidate_steps(args),
                "--pace_smoothing_window", str(args.pace_smoothing_window),
                "--pace_prominence_threshold", str(args.pace_prominence_threshold),
                "--precision", args.precision,
                "--eval_episode_slice_start", str(slice_idx),
                "--eval_episode_slice_stride", str(len(gpus)),
                "--result_json_path", str(result_path),
            ]
        if args.temporal_ensemble:
            cmd.append("--temporal_ensemble")
        if args.ablate_history_r3m_tokens:
            if args.qwen3vl_path is not None:
                raise ValueError("R3M history-token ablation is not supported by Qwen3-VL")
            cmd.append("--ablate_history_r3m_tokens")
        if args.pace:
            cmd.append("--pace")
        if not args.pace_gripper_boundaries:
            cmd.append("--no_pace_gripper_boundaries")
        env = _worker_env(gpu, egl_device_id=getattr(args, "egl_device_id", None))
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        jobs.append((slice_idx, gpu, process, handle, log_path))
        print(f"[{time.strftime('%F %T')}] started slice={slice_idx}/{len(gpus)} gpu={gpu} pid={process.pid} log={log_path}", flush=True)

    failed = []
    for slice_idx, gpu, process, handle, log_path in jobs:
        returncode = process.wait()
        handle.close()
        print(f"[{time.strftime('%F %T')}] finished slice={slice_idx} gpu={gpu} returncode={returncode}", flush=True)
        if returncode != 0:
            failed.append((slice_idx, gpu, returncode, log_path))
    if failed:
        raise RuntimeError(f"evaluation slices failed: {failed}")

    summary_path = step_dir / "summary.json"
    summary = aggregate(args, ckpt, result_paths, summary_path)
    print(
        f"[{time.strftime('%F %T')}] {ckpt.name}: success={summary['total_successes']}/{summary['total_episodes']} "
        f"rate={summary['overall_success_rate']:.4%} summary={summary_path}",
        flush=True,
    )
    return summary_path


def _evaluate_selected(args: argparse.Namespace) -> int:
    if args.ckpt is not None:
        if not args.ckpt.is_file():
            raise FileNotFoundError(args.ckpt)
        evaluate_one(args, args.ckpt.resolve())
        return 0

    if args.watch:
        print(f"[{time.strftime('%F %T')}] watching {args.checkpoint_dir} for {args.checkpoint_prefix}_*.pth", flush=True)
        while True:
            for step, ckpt in discovered_checkpoints(args):
                if args.max_step is not None and step > args.max_step:
                    continue
                if already_evaluated(args, step):
                    continue
                try:
                    evaluate_one(args, ckpt)
                except Exception as exc:  # keep watching even after a failed eval
                    print(f"[{time.strftime('%F %T')}] evaluation failed for step {step}: {exc}", flush=True)
            time.sleep(args.sleep_seconds)
        return 0

    checkpoints = discovered_checkpoints(args)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {args.checkpoint_dir}")
    step, ckpt = checkpoints[-1]
    if args.max_step is not None and step > args.max_step:
        raise ValueError(f"latest checkpoint step {step} exceeds --max-step {args.max_step}")
    evaluate_one(args, ckpt)
    return 0


def main() -> int:
    args = parse_args()
    if args.num_open_loop_steps_sweep:
        if args.pace or args.temporal_ensemble or args.watch:
            raise ValueError(
                "num_open_loop_steps_sweep supports fixed-horizon evaluation only; "
                "do not combine it with pace, temporal_ensemble, or watch"
            )
        horizons = _parse_positive_int_csv(
            args.num_open_loop_steps_sweep,
            "num_open_loop_steps_sweep",
        )
        if horizons[-1] > args.chunk_size:
            raise ValueError(
                f"largest fixed execution horizon {horizons[-1]} exceeds chunk_size={args.chunk_size}"
            )
        for horizon in horizons:
            sweep_args = argparse.Namespace(**vars(args))
            sweep_args.num_open_loop_steps = horizon
            sweep_args.num_open_loop_steps_sweep = ""
            _evaluate_selected(sweep_args)
        return 0
    return _evaluate_selected(args)


if __name__ == "__main__":
    raise SystemExit(main())
