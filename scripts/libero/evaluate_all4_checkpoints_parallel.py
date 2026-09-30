#!/usr/bin/env python3
"""Evaluate multiple checkpoints over all four LIBERO suites in GPU lanes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_STEPS = ",".join(str(step) for step in range(5_000, 100_001, 5_000))
DEFAULT_GPU_GROUPS = "0,1,3,4,0,1,3,4;2,5,2,5,2,5,2,5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-prefix", required=True)
    parser.add_argument("--steps", default=DEFAULT_STEPS)
    parser.add_argument("--eval-output-root", type=Path, required=True)
    parser.add_argument("--gpu-groups", default=DEFAULT_GPU_GROUPS)
    parser.add_argument(
        "--egl-device-id",
        type=int,
        default=None,
        help="Host EGL device ID forwarded to every official evaluation worker.",
    )
    parser.add_argument(
        "--checkpoint-per-lane",
        action="store_true",
        help=(
            "Keep all four suites for each checkpoint on the same GPU lane. "
            "Checkpoints are assigned round-robin across GPU groups."
        ),
    )
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--num-open-loop-steps", type=int, default=10)
    parser.add_argument("--dinov3-update-interval", type=int, default=1)
    parser.add_argument(
        "--ablate-history-r3m-tokens",
        action="store_true",
        help="Remove the four R3M history-memory tokens during inference.",
    )
    parser.add_argument("--r3m-path", type=Path, default=ROOT / "pretrained/r3m-resnet18/backbone.pth")
    parser.add_argument(
        "--retain-latest-and-best",
        action="store_true",
        help="After every requested checkpoint has been evaluated, delete all but the latest and best weights.",
    )
    return parser.parse_args()


def parse_positive_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"steps must be comma-separated integers, got {value!r}") from exc
    if not steps or any(step <= 0 for step in steps) or len(set(steps)) != len(steps):
        raise ValueError(f"steps must be positive and unique, got {steps}")
    return steps


def parse_gpu_groups(value: str) -> tuple[str, ...]:
    groups = []
    for raw_group in value.split(";"):
        entries = [item.strip() for item in raw_group.split(",") if item.strip()]
        if not entries:
            raise ValueError(f"each GPU group needs at least one task slice, got {raw_group!r}")
        groups.append(",".join(entries))
    if not groups:
        raise ValueError("GPU groups must be non-empty")
    widths = {len(group.split(",")) for group in groups}
    if len(widths) != 1:
        raise ValueError(f"GPU groups must have equal slice counts, got {groups}")
    return tuple(groups)


def _protocol_suffix(
    chunk_size: int,
    open_loop_steps: int,
    dinov3_update_interval: int = 1,
    ablate_history_r3m_tokens: bool = False,
) -> str:
    suffixes = []
    if open_loop_steps != chunk_size:
        suffixes.append(f"openloop{open_loop_steps}")
    if dinov3_update_interval != 1:
        suffixes.append(f"dinov3every{dinov3_update_interval}_r3mevery1")
    if ablate_history_r3m_tokens:
        suffixes.append("ablatehistoryr3m")
    return "" if not suffixes else "_" + "_".join(suffixes)


def suite_summary_path(args: argparse.Namespace, suite: str, step: int) -> Path:
    first_gpu_group = args.gpu_groups.split(";", 1)[0]
    slice_count = len([item for item in first_gpu_group.split(",") if item.strip()])
    return (
        args.eval_output_root
        / suite
        / f"{args.checkpoint_prefix}_{slice_count}slice_checkpoints"
        f"{_protocol_suffix(args.chunk_size, args.num_open_loop_steps, args.dinov3_update_interval, args.ablate_history_r3m_tokens)}"
        / f"step_{step:06d}"
        / "summary.json"
    )


def valid_suite_summary(args: argparse.Namespace, path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("total_episodes") == args.num_trials_per_task * 10
        and payload.get("chunk_size") == args.chunk_size
        and payload.get("num_open_loop_steps") == args.num_open_loop_steps
        and payload.get("dinov3_update_interval", 1) == args.dinov3_update_interval
        and payload.get("r3m_update_interval", 1) == 1
        and payload.get("ablate_history_r3m_tokens", False)
        == args.ablate_history_r3m_tokens
    )


def evaluation_command(
    args: argparse.Namespace,
    checkpoint: Path,
    suite: str,
    gpu_group: str,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "scripts/libero/evaluate_official_parallel.py",
        "--ckpt", str(checkpoint),
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--checkpoint-prefix", args.checkpoint_prefix,
        "--action-head", "flow_matching",
        "--flow-num-heads", "4",
        "--flow-condition-layers", "4",
        "--flow-dit-layers", "16",
        "--flow-target-tokens", "16",
        "--flow-static-tokens", "8",
        "--flow-state-dim", "0",
        "--gpus", gpu_group,
        "--task-suite-name", suite,
        "--num-trials-per-task", str(args.num_trials_per_task),
        "--dinov3-path", str(ROOT / "pretrained/dinov3-vitb16"),
        "--bert-path", str(ROOT / "pretrained/bert-base-uncased"),
        "--r3m-path", str(args.r3m_path.resolve()),
        "--stats-path", str(ROOT / "experiments/libero/configs/libero_all4_stats.json"),
        "--stats-key", "libero_all4_no_noops",
        "--chunk-size", str(args.chunk_size),
        "--num-open-loop-steps", str(args.num_open_loop_steps),
        "--dinov3-update-interval", str(args.dinov3_update_interval),
        "--precision", "bf16",
        "--eval-output-root", str(args.eval_output_root / suite),
    ]
    if getattr(args, "ablate_history_r3m_tokens", False):
        command.append("--ablate-history-r3m-tokens")
    if getattr(args, "egl_device_id", None) is not None:
        command.extend(("--egl-device-id", str(args.egl_device_id)))
    return command


def evaluate_lane(
    args: argparse.Namespace,
    gpu_group: str,
    jobs: list[tuple[int, str, Path]],
) -> None:
    for step, suite, checkpoint in jobs:
        summary_path = suite_summary_path(args, suite, step)
        if valid_suite_summary(args, summary_path):
            print(f"[{time.strftime('%F %T')}] already complete step={step} suite={suite}", flush=True)
            continue
        print(
            f"[{time.strftime('%F %T')}] evaluating step={step} suite={suite} "
            f"gpus={gpu_group}",
            flush=True,
        )
        result = subprocess.run(
            evaluation_command(args, checkpoint, suite, gpu_group),
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0 or not valid_suite_summary(args, summary_path):
            raise RuntimeError(
                f"evaluation failed: step={step}, suite={suite}, gpus={gpu_group}, "
                f"returncode={result.returncode}"
            )


def aggregate_summaries(
    step: int,
    checkpoint: Path,
    summaries: dict[str, dict[str, Any]],
    chunk_size: int,
    open_loop_steps: int,
    dinov3_update_interval: int = 1,
    ablate_history_r3m_tokens: bool = False,
) -> dict[str, Any]:
    total_episodes = sum(int(payload["total_episodes"]) for payload in summaries.values())
    total_successes = sum(int(payload["total_successes"]) for payload in summaries.values())
    return {
        "step": step,
        "checkpoint_path": str(checkpoint.resolve()),
        "suites": {
            suite: {
                "episodes": int(payload["total_episodes"]),
                "successes": int(payload["total_successes"]),
                "success_rate": float(payload["overall_success_rate"]),
            }
            for suite, payload in summaries.items()
        },
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": total_successes / max(total_episodes, 1),
        "chunk_size": chunk_size,
        "num_open_loop_steps": open_loop_steps,
        "dinov3_update_interval": dinov3_update_interval,
        "r3m_update_interval": 1,
        "ablate_history_r3m_tokens": ablate_history_r3m_tokens,
    }


def retain_latest_and_best(
    checkpoints: dict[int, Path],
    best_step: int,
) -> dict[str, Any]:
    """Remove evaluated weights other than the latest and all-suite best."""
    latest_step = max(checkpoints)
    retained_steps = sorted({latest_step, best_step})
    removed_steps = []
    for step, path in checkpoints.items():
        if step in retained_steps:
            continue
        path.unlink(missing_ok=True)
        removed_steps.append(step)
    return {
        "latest_step": latest_step,
        "best_step": best_step,
        "retained_steps": retained_steps,
        "removed_steps": sorted(removed_steps),
        "retained_paths": [str(checkpoints[step].resolve()) for step in retained_steps],
    }


def main() -> int:
    args = parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.eval_output_root = args.eval_output_root.resolve()
    steps = parse_positive_steps(args.steps)
    gpu_groups = parse_gpu_groups(args.gpu_groups)
    checkpoints = {
        step: args.checkpoint_dir / f"{args.checkpoint_prefix}_{step}.pth"
        for step in steps
    }
    # Completed evaluations remain usable after old checkpoint weights are
    # pruned.  A checkpoint is only required when at least one requested suite
    # still needs to be evaluated; aggregation itself reads the saved suite
    # summaries and does not load model weights.
    missing = [
        str(path)
        for step, path in checkpoints.items()
        if not path.is_file()
        and not all(
            valid_suite_summary(args, suite_summary_path(args, suite, step))
            for suite in SUITES
        )
    ]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    if args.checkpoint_per_lane:
        lanes = [
            [
                (step, suite, checkpoints[step])
                for step in steps[index::len(gpu_groups)]
                for suite in SUITES
            ]
            for index in range(len(gpu_groups))
        ]
    else:
        jobs = [
            (step, suite, checkpoints[step])
            for step in steps
            for suite in SUITES
        ]
        lanes = [jobs[index::len(gpu_groups)] for index in range(len(gpu_groups))]
    with ThreadPoolExecutor(max_workers=len(gpu_groups), thread_name_prefix="all4-eval") as executor:
        futures = [
            executor.submit(evaluate_lane, args, gpu_group, lane)
            for gpu_group, lane in zip(gpu_groups, lanes)
        ]
        for future in futures:
            future.result()

    aggregate_paths = []
    for step in steps:
        summaries = {
            suite: json.loads(suite_summary_path(args, suite, step).read_text(encoding="utf-8"))
            for suite in SUITES
        }
        aggregate = aggregate_summaries(
            step,
            checkpoints[step],
            summaries,
            args.chunk_size,
            args.num_open_loop_steps,
            args.dinov3_update_interval,
            args.ablate_history_r3m_tokens,
        )
        output = args.eval_output_root / f"step_{step:06d}" / "summary_all4.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        aggregate_paths.append(output)
        print(
            f"[{time.strftime('%F %T')}] all4 step={step}: "
            f"success={aggregate['total_successes']}/{aggregate['total_episodes']} "
            f"rate={aggregate['overall_success_rate']:.4%} summary={output}",
            flush=True,
        )

    best = max(
        (json.loads(path.read_text(encoding="utf-8")) for path in aggregate_paths),
        key=lambda payload: (payload["overall_success_rate"], payload["step"]),
    )
    (args.eval_output_root / "best_all4.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.retain_latest_and_best:
        retention = retain_latest_and_best(checkpoints, int(best["step"]))
        (args.eval_output_root / "retained_checkpoints.json").write_text(
            json.dumps(retention, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{time.strftime('%F %T')}] retained checkpoint steps="
            f"{retention['retained_steps']}; removed={retention['removed_steps']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
