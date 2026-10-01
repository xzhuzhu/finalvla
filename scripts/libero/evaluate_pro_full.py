#!/usr/bin/env python3
"""Resumable, episode-sliced evaluation of FinalVLA on official LIBERO-PRO suites."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PRO_ROOT = Path(os.environ.get("LIBERO_PRO_ROOT", ROOT.parent / "libero_pro_eval"))
CKPT = Path(os.environ.get("FINALVLA_CKPT", ROOT.parent / "v15_qr6_matched_cuda_retrain/outputs/v15_qr6_matched_cuda_state2_zeropad_nokl_nofuturestate_r2_from0_100k/turbovla_v15_qr6_matched_cuda_state2_zeropad_nokl_nofuturestate_r2_step_95000.pth"))
BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PERTURBATIONS = ("object", "swap", "lan", "task", "env")
ALL_SUITES = tuple(f"{base}_{kind}" for kind in PERTURBATIONS for base in BASE_SUITES)
EXPECTED_EPISODES = 500


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_valid_shard(path: Path, suite: str, shard: int, shards: int) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_per_task = len(range(shard, 50, shards))
    if (
        value.get("task_suite_name") != suite
        or value.get("num_trials_per_task") != 50
        or value.get("eval_episode_slice_start") != shard
        or value.get("eval_episode_slice_stride") != shards
        or value.get("total_episodes") != 10 * expected_per_task
        or len(value.get("tasks", [])) != 10
        or any(task.get("episodes") != expected_per_task for task in value["tasks"])
        or value.get("chunk_size") != 12
        or value.get("num_open_loop_steps") != 10
    ):
        return None
    return value


def run_shard(suite: str, shard: int, shards: int, gpu: int, output_root: Path, timeout: int) -> Path:
    output = output_root / suite / f"shard_{shard:02d}.json"
    if load_valid_shard(output, suite, shard, shards) is not None:
        print(f"skip {suite} shard={shard:02d}", flush=True)
        return output
    python = PRO_ROOT / ".venv/bin/python"
    cmd = [
        str(python), "-m", "vla_adapter.rollout",
        "--ckpt_path", str(CKPT),
        "--libero_root", str(PRO_ROOT),
        "--dinov3_path", str(ROOT / "pretrained/dinov3-vitb16"),
        "--bert_path", str(ROOT / "pretrained/bert-base-uncased"),
        "--r3m_path", str(ROOT / "pretrained/r3m-resnet18/backbone.pth"),
        "--stats_path", str(ROOT / "experiments/libero/configs/libero_all4_stats.json"),
        "--task_suite_name", suite,
        "--num_trials_per_task", "50",
        "--eval_episode_slice_start", str(shard),
        "--eval_episode_slice_stride", str(shards),
        "--action_head", "flow_matching",
        "--flow_state_dim", "8",
        "--num_open_loop_steps", "10",
        "--result_json_path", str(output),
        "--log_path", str(output.with_suffix(".log")),
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "LIBERO_CONFIG_PATH": str(PRO_ROOT / ".libero_config"),
        "MUJOCO_GL": "osmesa",
        "PYOPENGL_PLATFORM": "osmesa",
        "LD_LIBRARY_PATH": str(PRO_ROOT / "local_osmesa/usr/lib/x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", ""),
        "PYTHONPATH": str(ROOT) + ":" + str(ROOT / "third_party/vla_adapter") + ":" + env.get("PYTHONPATH", ""),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    stdout = output.with_suffix(".stdout.log")
    for attempt in range(1, 4):
        try:
            with stdout.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            if proc.returncode == 0 and load_valid_shard(output, suite, shard, shards) is not None:
                print(f"done {suite} shard={shard:02d} gpu={gpu}", flush=True)
                return output
            error = f"exit={proc.returncode}"
        except subprocess.TimeoutExpired:
            error = f"timeout={timeout}s"
        print(f"retry {suite} shard={shard:02d} attempt={attempt} {error}; log={stdout}", flush=True)
        time.sleep(5)
    raise RuntimeError(f"Failed {suite} shard={shard:02d}; inspect {stdout}")


def aggregate_suite(suite: str, shards: int, output_root: Path) -> dict:
    values = []
    for shard in range(shards):
        path = output_root / suite / f"shard_{shard:02d}.json"
        value = load_valid_shard(path, suite, shard, shards)
        if value is None:
            raise RuntimeError(f"Missing or invalid shard: {path}")
        values.append(value)
    tasks = []
    for task_id in range(10):
        rows = [next(t for t in value["tasks"] if t["task_id"] == task_id) for value in values]
        episodes = sum(row["episodes"] for row in rows)
        successes = sum(row["successes"] for row in rows)
        if episodes != 50:
            raise RuntimeError(f"Expected 50 episodes in {suite} task {task_id}, got {episodes}")
        tasks.append({"task_id": task_id, "task_description": rows[0]["task_description"], "episodes": episodes, "successes": successes, "success_rate": successes / episodes})
    total = sum(task["episodes"] for task in tasks)
    successes = sum(task["successes"] for task in tasks)
    if total != EXPECTED_EPISODES:
        raise RuntimeError(f"Expected {EXPECTED_EPISODES} episodes in {suite}, got {total}")
    result = {
        "suite": suite,
        "checkpoint_step": 95000,
        "checkpoint_sha256": "63383e42566ff5f3eef88ef5025b343797ef79d7a5325c3cb6e098e6b0df8710",
        "num_trials_per_task": 50,
        "shards": shards,
        "chunk_size": 12,
        "num_open_loop_steps": 10,
        "tasks": tasks,
        "total_episodes": total,
        "total_successes": successes,
        "overall_success_rate": successes / total,
    }
    save_json(output_root / suite / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", default=",".join(ALL_SUITES))
    parser.add_argument("--gpus", default="1,2,3,4,5,7")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=PRO_ROOT / "results/full_95k")
    parser.add_argument("--timeout-per-shard", type=int, default=3600)
    args = parser.parse_args()
    suites = tuple(s.strip() for s in args.suites.split(",") if s.strip())
    if any(s not in ALL_SUITES for s in suites) or len(set(suites)) != len(suites):
        parser.error("--suites must contain distinct official LIBERO-PRO suites")
    gpus = tuple(int(s) for s in args.gpus.split(",") if s)
    if not gpus or args.workers_per_gpu < 1 or not CKPT.is_file():
        parser.error("No GPUs, invalid workers-per-gpu, or missing checkpoint")
    shards = len(gpus) * args.workers_per_gpu
    if shards > 50:
        parser.error("Shard count cannot exceed 50 initial states per task")
    print(f"start suites={len(suites)} shards={shards} gpus={gpus} checkpoint={CKPT}", flush=True)
    for suite in suites:
        start = time.time()
        print(f"begin {suite}", flush=True)
        with ThreadPoolExecutor(max_workers=shards) as pool:
            futures = {pool.submit(run_shard, suite, shard, shards, gpus[shard % len(gpus)], args.output_root, args.timeout_per_shard): shard for shard in range(shards)}
            for future in as_completed(futures):
                future.result()
        summary = aggregate_suite(suite, shards, args.output_root)
        print(f"complete {suite}: {summary['total_successes']}/{summary['total_episodes']}={summary['overall_success_rate']:.4%}, elapsed={time.time()-start:.1f}s", flush=True)
    print("all requested suites complete", flush=True)


if __name__ == "__main__":
    main()
