#!/usr/bin/env python3
"""Watch for checkpoints and evaluate each one on all four LIBERO suites."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATE_ALL4 = ROOT / "scripts/libero/evaluate_all4_checkpoints_parallel.py"


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for checkpoints and evaluate them sequentially on all four "
            "LIBERO suites, using multiple slices on one GPU."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-prefix", required=True)
    parser.add_argument("--eval-output-root", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=80000)
    parser.add_argument("--final-step", type=int, default=100000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument(
        "--gpu",
        default=None,
        help="Single GPU (backward-compatible alias for --gpus).",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU lanes; every lane uses --slices workers.",
    )
    parser.add_argument("--slices", type=int, default=32)
    parser.add_argument(
        "--egl-device-id",
        type=int,
        default=None,
        help="Host EGL device ID forwarded to every evaluation worker.",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Evaluate requested checkpoints from --final-step down to --start-step.",
    )
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--retry-seconds", type=float, default=120.0)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--num-open-loop-steps", type=int, default=10)
    parser.add_argument("--dinov3-update-interval", type=int, default=1)
    parser.add_argument("--r3m-path", type=Path, required=True)
    args = parser.parse_args()

    if args.start_step < 0 or args.final_step < args.start_step:
        parser.error("step range is invalid")
    for name in (
        "save_interval",
        "slices",
        "num_trials_per_task",
        "chunk_size",
        "num_open_loop_steps",
        "dinov3_update_interval",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.poll_seconds <= 0 or args.retry_seconds <= 0:
        parser.error("poll and retry intervals must be positive")
    if args.gpu is not None and args.gpus is not None:
        parser.error("use either --gpu or --gpus, not both")
    gpu_value = args.gpus if args.gpus is not None else (args.gpu or "0")
    args.gpu_list = tuple(item.strip() for item in gpu_value.split(",") if item.strip())
    if not args.gpu_list:
        parser.error("at least one GPU is required")
    return args


def checkpoint_path(args: argparse.Namespace, step: int) -> Path:
    return args.checkpoint_dir / f"{args.checkpoint_prefix}_{step}.pth"


def requested_steps(args: argparse.Namespace) -> range:
    if args.descending:
        return range(args.final_step, args.start_step - 1, -args.save_interval)
    return range(args.start_step, args.final_step + 1, args.save_interval)


def read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def step_is_complete(args: argparse.Namespace, step: int) -> bool:
    summary = read_json(args.eval_output_root / f"step_{step:06d}" / "summary_all4.json")
    if summary is None:
        return False
    expected_episodes = 4 * 10 * args.num_trials_per_task
    return (
        summary.get("step") == step
        and summary.get("total_episodes") == expected_episodes
        and summary.get("chunk_size") == args.chunk_size
        and summary.get("num_open_loop_steps") == args.num_open_loop_steps
    )


def refresh_best_summary(eval_output_root: Path) -> None:
    summaries: list[dict] = []
    for path in sorted(eval_output_root.glob("step_*/summary_all4.json")):
        summary = read_json(path)
        if summary is not None and isinstance(summary.get("overall_success_rate"), (int, float)):
            summaries.append(summary)
    if not summaries:
        return

    best = max(summaries, key=lambda item: (item["overall_success_rate"], item.get("step", -1)))
    output_path = eval_output_root / "best_all4.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_path, output_path)


def evaluate_step(args: argparse.Namespace, step: int) -> int:
    gpu_groups = ";".join(
        ",".join([str(gpu)] * args.slices) for gpu in args.gpu_list
    )
    command = [
        sys.executable,
        "-u",
        str(EVALUATE_ALL4),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--checkpoint-prefix",
        args.checkpoint_prefix,
        "--steps",
        str(step),
        "--eval-output-root",
        str(args.eval_output_root),
        "--gpu-groups",
        gpu_groups,
        "--num-trials-per-task",
        str(args.num_trials_per_task),
        "--chunk-size",
        str(args.chunk_size),
        "--num-open-loop-steps",
        str(args.num_open_loop_steps),
        "--dinov3-update-interval",
        str(args.dinov3_update_interval),
        "--r3m-path",
        str(args.r3m_path),
    ]
    if args.egl_device_id is not None:
        command.extend(("--egl-device-id", str(args.egl_device_id)))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    args = parse_args()
    args.eval_output_root.mkdir(parents=True, exist_ok=True)
    log(
        f"monitoring steps {args.start_step}..{args.final_step} every "
        f"{args.save_interval}; GPU lanes {','.join(args.gpu_list)}, "
        f"{args.slices} slices per lane"
    )

    for step in requested_steps(args):
        while not step_is_complete(args, step):
            checkpoint = checkpoint_path(args, step)
            if not checkpoint.is_file():
                log(f"waiting for checkpoint {step}: {checkpoint}")
                time.sleep(args.poll_seconds)
                continue

            log(f"starting all-four-suite evaluation for checkpoint {step}")
            returncode = evaluate_step(args, step)
            if returncode == 0 and step_is_complete(args, step):
                break
            log(
                f"checkpoint {step} evaluation incomplete (exit={returncode}); "
                f"retrying in {args.retry_seconds:g}s"
            )
            time.sleep(args.retry_seconds)

        refresh_best_summary(args.eval_output_root)
        log(f"checkpoint {step} evaluation complete")

    log("all requested checkpoints are complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("monitor interrupted")
        raise SystemExit(130)
