import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/libero/evaluate_all4_checkpoints_parallel.py"
SPEC = importlib.util.spec_from_file_location("evaluate_all4_checkpoints_parallel", SCRIPT)
SCHEDULER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCHEDULER)

OFFICIAL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/libero/evaluate_official_parallel.py"
OFFICIAL_SPEC = importlib.util.spec_from_file_location("evaluate_official_parallel", OFFICIAL_SCRIPT)
OFFICIAL = importlib.util.module_from_spec(OFFICIAL_SPEC)
assert OFFICIAL_SPEC.loader is not None
OFFICIAL_SPEC.loader.exec_module(OFFICIAL)

MONITOR_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/libero/monitor_all4_checkpoints_parallel.py"
MONITOR_SPEC = importlib.util.spec_from_file_location("monitor_all4_checkpoints_parallel", MONITOR_SCRIPT)
MONITOR = importlib.util.module_from_spec(MONITOR_SPEC)
assert MONITOR_SPEC.loader is not None
MONITOR_SPEC.loader.exec_module(MONITOR)


class All4EvaluationSchedulerTest(unittest.TestCase):
    def test_parsers(self):
        self.assertEqual(SCHEDULER.parse_positive_steps("5000,10000"), (5000, 10000))
        self.assertEqual(
            SCHEDULER.parse_positive_steps(SCHEDULER.DEFAULT_STEPS),
            tuple(range(5000, 100001, 5000)),
        )
        self.assertEqual(
            SCHEDULER.parse_gpu_groups("0,1,3,4;2,5,2,5"),
            ("0,1,3,4", "2,5,2,5"),
        )
        self.assertEqual(
            SCHEDULER.parse_gpu_groups("0,1,3,4,0,1,3,4;2,5,2,5,2,5,2,5"),
            ("0,1,3,4,0,1,3,4", "2,5,2,5,2,5,2,5"),
        )
        self.assertEqual(
            SCHEDULER.parse_gpu_groups("4,4,4,4,4,4,4,4;4,4,4,4,4,4,4,4"),
            ("4,4,4,4,4,4,4,4", "4,4,4,4,4,4,4,4"),
        )
        with self.assertRaises(ValueError):
            SCHEDULER.parse_positive_steps("5000,5000")
        with self.assertRaises(ValueError):
            SCHEDULER.parse_gpu_groups("0,1;2,3,4")

    def test_aggregate_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model_5000.pth"
            checkpoint.write_bytes(b"weights")
            summaries = {
                suite: {
                    "total_episodes": 500,
                    "total_successes": successes,
                    "overall_success_rate": successes / 500,
                }
                for suite, successes in zip(SCHEDULER.SUITES, (400, 450, 425, 475))
            }
            result = SCHEDULER.aggregate_summaries(5000, checkpoint, summaries, 12, 10, 2)
            self.assertEqual(result["total_episodes"], 2000)
            self.assertEqual(result["total_successes"], 1750)
            self.assertEqual(result["overall_success_rate"], 0.875)
            self.assertEqual(result["dinov3_update_interval"], 2)
            self.assertEqual(result["r3m_update_interval"], 1)

    def test_dinov3_cadence_has_an_isolated_result_namespace(self):
        self.assertEqual(
            SCHEDULER._protocol_suffix(12, 10, 2),
            "_openloop10_dinov3every2_r3mevery1",
        )

    def test_r3m_path_is_forwarded_to_suite_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r3m_path = root / "r3m" / "backbone.pth"
            args = Namespace(
                checkpoint_dir=root / "checkpoints",
                checkpoint_prefix="model_step",
                eval_output_root=root / "evaluation",
                num_trials_per_task=50,
                chunk_size=12,
                num_open_loop_steps=10,
                dinov3_update_interval=1,
                r3m_path=r3m_path,
            )
            command = SCHEDULER.evaluation_command(
                args,
                root / "checkpoints/model_step_99000.pth",
                "libero_spatial",
                "0,0",
            )
            option_index = command.index("--r3m-path")
            self.assertEqual(command[option_index + 1], str(r3m_path.resolve()))

    def test_omitted_egl_device_preserves_worker_mapping(self):
        env = OFFICIAL._worker_env("1", base_env={"TURBOVLA_EVAL_CPU_THREADS": "2"})

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["MUJOCO_EGL_DEVICE_ID"], "1")

    def test_explicit_differing_egl_device_keeps_cuda_gpu_first(self):
        env = OFFICIAL._worker_env("1", egl_device_id=8, base_env={})

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,8")
        self.assertEqual(env["MUJOCO_EGL_DEVICE_ID"], "8")

    def test_egl_device_id_propagates_monitor_to_official_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler_args = Namespace(
                checkpoint_dir=root / "checkpoints",
                checkpoint_prefix="model_step",
                eval_output_root=root / "evaluation",
                num_trials_per_task=50,
                chunk_size=12,
                num_open_loop_steps=10,
                dinov3_update_interval=1,
                r3m_path=root / "r3m" / "backbone.pth",
                egl_device_id=8,
            )
            command = SCHEDULER.evaluation_command(
                scheduler_args,
                root / "checkpoints/model_step_99000.pth",
                "libero_spatial",
                "1,1",
            )
            option_index = command.index("--egl-device-id")
            self.assertEqual(command[option_index + 1], "8")

            monitor_args = Namespace(
                checkpoint_dir=root / "checkpoints",
                checkpoint_prefix="model_step",
                eval_output_root=root / "evaluation",
                gpu_list=("1",),
                slices=2,
                num_trials_per_task=50,
                chunk_size=12,
                num_open_loop_steps=10,
                dinov3_update_interval=1,
                r3m_path=root / "r3m" / "backbone.pth",
                egl_device_id=8,
            )
            with patch.object(MONITOR.subprocess, "run") as run:
                run.return_value.returncode = 0
                self.assertEqual(MONITOR.evaluate_step(monitor_args, 99000), 0)
            monitor_command = run.call_args.args[0]
            monitor_option_index = monitor_command.index("--egl-device-id")
            self.assertEqual(monitor_command[monitor_option_index + 1], "8")

    def test_monitor_descending_steps(self):
        args = Namespace(start_step=80000, final_step=100000, save_interval=1000, descending=True)
        self.assertEqual(list(MONITOR.requested_steps(args)), list(range(100000, 79999, -1000)))

    def test_retain_latest_and_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = {}
            for step in (5000, 10000, 15000):
                path = root / f"model_{step}.pth"
                path.write_bytes(b"weights")
                checkpoints[step] = path

            result = SCHEDULER.retain_latest_and_best(checkpoints, best_step=10000)

            self.assertEqual(result["retained_steps"], [10000, 15000])
            self.assertEqual(result["removed_steps"], [5000])
            self.assertFalse(checkpoints[5000].exists())
            self.assertTrue(checkpoints[10000].exists())
            self.assertTrue(checkpoints[15000].exists())

    def test_retain_one_when_latest_is_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = {}
            for step in (5000, 10000):
                path = root / f"model_{step}.pth"
                path.write_bytes(b"weights")
                checkpoints[step] = path

            result = SCHEDULER.retain_latest_and_best(checkpoints, best_step=10000)

            self.assertEqual(result["retained_steps"], [10000])
            self.assertFalse(checkpoints[5000].exists())
            self.assertTrue(checkpoints[10000].exists())


if __name__ == "__main__":
    unittest.main()
