#!/usr/bin/env python3
"""Aggregate per-checkpoint and per-task success rates from evaluation summaries
into a single CSV (one row per task per checkpoint, plus an ALL row per checkpoint).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path,
                    default=ROOT / "outputs/evaluation/turbovla_libero10_official_step_4slice_checkpoints")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "outputs/evaluation/success_rates_libero10_official.csv")
    args = ap.parse_args()

    step_dirs = []
    for step_dir in args.eval_root.glob("step_*"):
        m = re.fullmatch(r"step_(\d{6})", step_dir.name)
        if not m:
            continue
        summary = step_dir / "summary.json"
        if summary.is_file():
            step_dirs.append((int(m.group(1)), summary))
    step_dirs.sort()

    if not step_dirs:
        print(f"no summaries found under {args.eval_root}", flush=True)
        return 1

    rows = []
    for step, summary_path in step_dirs:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        for task in data["tasks"]:
            rows.append({
                "step": step,
                "task_id": task["task_id"],
                "task_description": task["task_description"],
                "episodes": task["episodes"],
                "successes": task["successes"],
                "success_rate": task["success_rate"],
            })
        rows.append({
            "step": step,
            "task_id": "ALL",
            "task_description": "overall",
            "episodes": data["total_episodes"],
            "successes": data["total_successes"],
            "success_rate": data["overall_success_rate"],
        })

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "task_id", "task_description",
                                                    "episodes", "successes", "success_rate"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
