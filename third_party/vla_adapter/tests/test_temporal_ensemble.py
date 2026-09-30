import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from vla_adapter.rollout import (
    AlignedTemporalEnsembler,
    GenerateConfig,
    PACEHorizonSelector,
    eval_libero,
    parse_pace_candidate_steps,
)


_LAUNCHER_PATH = Path(__file__).resolve().parents[3] / "scripts/libero/evaluate_official_parallel.py"
_LAUNCHER_SPEC = importlib.util.spec_from_file_location("libero_parallel_launcher", _LAUNCHER_PATH)
assert _LAUNCHER_SPEC is not None and _LAUNCHER_SPEC.loader is not None
launcher = importlib.util.module_from_spec(_LAUNCHER_SPEC)
_LAUNCHER_SPEC.loader.exec_module(launcher)


def chunk(values):
    return np.asarray([[value] * 7 for value in values], dtype=np.float32)


class AlignedTemporalEnsemblerTest(unittest.TestCase):
    def test_exact_temporal_alignment(self):
        ensembler = AlignedTemporalEnsembler(chunk_size=3, alpha=0.0)
        ensembler.add_chunk(0, chunk([1.0, 10.0, 30.0]))
        self.assertTrue(np.allclose(ensembler.ensemble(0), 1.0))

        ensembler.add_chunk(1, chunk([20.0, 40.0, 60.0]))
        self.assertTrue(np.allclose(ensembler.ensemble(1), 15.0))

        ensembler.add_chunk(2, chunk([50.0, 70.0, 90.0]))
        self.assertTrue(np.allclose(ensembler.ensemble(2), 40.0))

    def test_history_is_bounded(self):
        ensembler = AlignedTemporalEnsembler(chunk_size=2, alpha=0.0)
        ensembler.add_chunk(0, chunk([1.0, 1.0]))
        ensembler.add_chunk(1, chunk([2.0, 2.0]))
        ensembler.add_chunk(2, chunk([3.0, 3.0]))
        self.assertEqual(ensembler.history_size, 2)
        self.assertTrue(np.allclose(ensembler.ensemble(2), 2.5))

    def test_similarity_weights_favor_current_reference(self):
        ensembler = AlignedTemporalEnsembler(chunk_size=2, alpha=2.0)
        ensembler.add_chunk(0, chunk([-99.0, -1.0]))
        ensembler.add_chunk(1, chunk([1.0]))
        output = ensembler.ensemble(1)
        expected = (np.exp(2.0) - np.exp(-2.0)) / (np.exp(2.0) + np.exp(-2.0))
        self.assertTrue(np.allclose(output, expected, atol=1e-6))

    def test_reset_clears_episode_history_and_zero_norms_are_finite(self):
        ensembler = AlignedTemporalEnsembler(chunk_size=3, alpha=0.1)
        ensembler.add_chunk(0, chunk([2.0]))
        ensembler.reset()
        self.assertEqual(ensembler.history_size, 0)
        ensembler.add_chunk(0, chunk([0.0]))
        output = ensembler.ensemble(0)
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertTrue(np.allclose(output, 0.0))

    def test_invalid_alpha_is_rejected(self):
        for alpha in (-0.1, float("inf"), float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    AlignedTemporalEnsembler(chunk_size=1, alpha=alpha)


class PACEHorizonSelectorTest(unittest.TestCase):
    @staticmethod
    def action_chunk(speeds, gripper=-1.0):
        result = np.zeros((len(speeds), 7), dtype=np.float32)
        result[:, 0] = np.asarray(speeds, dtype=np.float32)
        if np.isscalar(gripper):
            result[:, 6] = float(gripper)
        else:
            result[:, 6] = np.asarray(gripper, dtype=np.float32)
        return result

    def test_candidate_parser_requires_ordered_in_range_steps(self):
        self.assertEqual(parse_pace_candidate_steps("8,10,12", 12), (8, 10, 12))
        for value in ("", "8,8,12", "10,8,12", "0,8", "8,nope"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_pace_candidate_steps(value, 12)
        with self.assertRaisesRegex(ValueError, "exceeds chunk_size"):
            parse_pace_candidate_steps("8,10,13", 12)

    def test_earliest_prominent_speed_valley_is_selected(self):
        selector = PACEHorizonSelector(
            (8, 10, 12),
            smoothing_window=1,
            prominence_threshold=0.2,
            gripper_boundaries=False,
        )
        speeds = [1.0] * 12
        speeds[7] = 0.1
        speeds[9] = 0.05
        self.assertEqual(selector.select(self.action_chunk(speeds)), 8)

    def test_no_accepted_valley_falls_back_to_longest_horizon(self):
        selector = PACEHorizonSelector(
            (8, 10, 12),
            smoothing_window=3,
            prominence_threshold=0.2,
            gripper_boundaries=False,
        )
        self.assertEqual(selector.select(self.action_chunk([0.5] * 12)), 12)

    def test_stable_gripper_transition_maps_to_next_candidate(self):
        selector = PACEHorizonSelector(
            (8, 10, 12),
            smoothing_window=1,
            prominence_threshold=1.0,
            gripper_boundaries=True,
        )
        gripper = [-1.0] * 8 + [1.0] * 4
        self.assertEqual(selector.select(self.action_chunk([0.5] * 12, gripper)), 10)

        unstable = [-1.0] * 8 + [1.0, -1.0, 1.0, -1.0]
        self.assertEqual(selector.select(self.action_chunk([0.5] * 12, unstable)), 12)

    def test_invalid_selector_hyperparameters_are_rejected(self):
        with self.assertRaises(ValueError):
            PACEHorizonSelector((10, 8, 12))
        with self.assertRaisesRegex(ValueError, "positive odd"):
            PACEHorizonSelector((8, 10, 12), smoothing_window=2)
        with self.assertRaisesRegex(ValueError, "in \[0, 1\]"):
            PACEHorizonSelector((8, 10, 12), prominence_threshold=1.1)


def launcher_args(output_root: Path, **overrides):
    values = {
        "eval_output_root": output_root,
        "checkpoint_prefix": "turbovla_libero_official_step",
        "gpus": "1,2,1,2",
        "chunk_size": 12,
        "num_open_loop_steps": 12,
        "dinov3_update_interval": 1,
        "ablate_history_r3m_tokens": False,
        "temporal_ensemble": False,
        "temporal_ensemble_alpha": 0.1,
        "pace": False,
        "pace_candidate_steps": "8,10,12",
        "pace_smoothing_window": 3,
        "pace_prominence_threshold": 0.2,
        "pace_gripper_boundaries": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ParallelLauncherProtocolTest(unittest.TestCase):
    def test_r3m_path_reaches_policy_constructor(self):
        captured = {}

        class FakePolicy:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            dinov3 = root / "dinov3"
            dinov3.mkdir()
            bert = root / "bert"
            bert.mkdir()
            r3m = root / "r3m" / "backbone.pth"
            r3m.parent.mkdir()
            r3m.write_bytes(b"r3m")
            stats = root / "stats.json"
            stats.write_text("{}", encoding="utf-8")
            config = GenerateConfig(
                ckpt_path=str(checkpoint),
                dinov3_path=str(dinov3),
                bert_path=str(bert),
                r3m_path=str(r3m),
                stats_path=str(stats),
                dry_run_model_load=True,
            )
            adapter = (FakePolicy, lambda: [0.0] * 7, lambda image: image, lambda _seed: None)
            with mock.patch("vla_adapter.rollout._import_turbovla_adapter", return_value=adapter):
                self.assertEqual(eval_libero(config), 0.0)

        self.assertEqual(captured["r3m_path"], str(r3m))

    def test_history_r3m_ablation_reaches_policy_constructor(self):
        captured = {}

        class FakePolicy:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            dinov3 = root / "dinov3"
            dinov3.mkdir()
            bert = root / "bert"
            bert.mkdir()
            r3m = root / "r3m.pth"
            r3m.write_bytes(b"r3m")
            stats = root / "stats.json"
            stats.write_text("{}", encoding="utf-8")
            config = GenerateConfig(
                ckpt_path=str(checkpoint),
                dinov3_path=str(dinov3),
                bert_path=str(bert),
                r3m_path=str(r3m),
                stats_path=str(stats),
                ablate_history_r3m_tokens=True,
                dry_run_model_load=True,
            )
            adapter = (FakePolicy, lambda: [0.0] * 7, lambda image: image, lambda _seed: None)
            with mock.patch("vla_adapter.rollout._import_turbovla_adapter", return_value=adapter):
                self.assertEqual(eval_libero(config), 0.0)

        self.assertTrue(captured["ablate_history_r3m_tokens"])

    def test_k10_is_the_fixed_horizon_default(self):
        self.assertEqual(GenerateConfig().chunk_size, 12)
        self.assertEqual(GenerateConfig().num_open_loop_steps, 10)

        with mock.patch("sys.argv", [str(_LAUNCHER_PATH)]):
            args = launcher.parse_args()
        self.assertEqual(args.chunk_size, 12)
        self.assertEqual(args.num_open_loop_steps, 10)

    def test_pace_canonicalizes_the_unused_fixed_horizon_to_its_fallback(self):
        args = launcher_args(Path("/tmp/libero-pace-default-test"), pace=True, num_open_loop_steps=10)
        self.assertEqual(launcher._effective_open_loop_steps(args), 12)
        self.assertEqual(launcher.expected_protocol(args)["num_open_loop_steps"], 12)
        self.assertNotIn("openloop10", launcher._protocol_suffix(args))

    def test_worker_env_keeps_cuda_and_egl_on_the_requested_physical_gpu(self):
        base = {
            "SENTINEL": "kept",
            "MUJOCO_EGL_DEVICE_ID": "7",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
        }
        worker = launcher._worker_env("5", base)

        self.assertEqual(worker["SENTINEL"], "kept")
        self.assertEqual(worker["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(worker["MUJOCO_EGL_DEVICE_ID"], "5")
        self.assertEqual(worker["MUJOCO_GL"], "egl")
        self.assertEqual(worker["PYOPENGL_PLATFORM"], "egl")
        self.assertEqual(worker["OMP_NUM_THREADS"], "4")
        self.assertEqual(worker["MKL_NUM_THREADS"], "4")
        self.assertEqual(worker["OPENBLAS_NUM_THREADS"], "4")
        self.assertEqual(worker["NUMEXPR_NUM_THREADS"], "4")
        self.assertEqual(base["MUJOCO_GL"], "egl")

        gpu_zero_worker = launcher._worker_env("0", base)
        self.assertEqual(gpu_zero_worker["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(gpu_zero_worker["MUJOCO_EGL_DEVICE_ID"], "0")

    def test_protocol_directory_namespaces(self):
        root = Path("/tmp/libero-protocol-test")
        legacy = launcher._step_dir(launcher_args(root), 80000).parent.name
        openloop5 = launcher._step_dir(launcher_args(root, num_open_loop_steps=5), 80000).parent.name
        temporal = launcher._step_dir(launcher_args(root, temporal_ensemble=True), 80000).parent.name
        chunk8 = launcher._step_dir(launcher_args(root, chunk_size=8, num_open_loop_steps=8), 80000).parent.name
        chunk8_temporal = launcher._step_dir(
            launcher_args(root, chunk_size=8, num_open_loop_steps=8, temporal_ensemble=True), 80000
        ).parent.name
        pace = launcher._step_dir(launcher_args(root, pace=True), 80000).parent.name
        dinov3_every2 = launcher._step_dir(
            launcher_args(root, dinov3_update_interval=2), 80000
        ).parent.name
        ablated_history_r3m = launcher._step_dir(
            launcher_args(root, ablate_history_r3m_tokens=True), 80000
        ).parent.name

        self.assertEqual(legacy, "turbovla_libero_official_step_4slice_checkpoints")
        self.assertEqual(openloop5, "turbovla_libero_official_step_4slice_checkpoints_openloop5")
        self.assertEqual(temporal, "turbovla_libero_official_step_4slice_checkpoints_temporal_ensemble_a0p1")
        self.assertEqual(chunk8, "turbovla_libero_official_step_4slice_checkpoints_chunk8")
        self.assertEqual(chunk8_temporal, "turbovla_libero_official_step_4slice_checkpoints_chunk8_temporal_ensemble_a0p1")
        self.assertEqual(
            pace,
            "turbovla_libero_official_step_4slice_checkpoints_pace_h8-10-12_w3_p0p2_gripper",
        )
        self.assertEqual(
            dinov3_every2,
            "turbovla_libero_official_step_4slice_checkpoints_dinov3every2_r3mevery1",
        )
        self.assertEqual(
            ablated_history_r3m,
            "turbovla_libero_official_step_4slice_checkpoints_ablatehistoryr3m",
        )
        self.assertNotIn(chunk8, {legacy, temporal})
        self.assertNotEqual(chunk8_temporal, temporal)

    def test_fixed_horizon_sweep_parser(self):
        self.assertEqual(launcher._parse_positive_int_csv("8,10,12", "sweep"), (8, 10, 12))
        for value in ("", "8,8", "10,8", "0,8"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    launcher._parse_positive_int_csv(value, "sweep")

    def test_skip_requires_matching_summary_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = launcher_args(Path(temp_dir))
            summary_path = launcher._step_dir(args, 80000) / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("not json", encoding="utf-8")
            self.assertFalse(launcher.already_evaluated(args, 80000))

            summary_path.write_text("[]", encoding="utf-8")
            self.assertFalse(launcher.already_evaluated(args, 80000))

            summary_path.write_text(json.dumps(launcher.expected_protocol(args)), encoding="utf-8")
            self.assertTrue(launcher.already_evaluated(args, 80000))

            wrong = launcher.expected_protocol(args)
            wrong["query_interval"] = 1
            summary_path.write_text(json.dumps(wrong), encoding="utf-8")
            self.assertFalse(launcher.already_evaluated(args, 80000))

    def test_aggregate_rejects_consistent_but_wrong_slice_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = launcher_args(root)
            slice_path = root / "slice.json"
            wrong = launcher.expected_protocol(args)
            wrong["chunk_size"] = 8
            slice_path.write_text(
                json.dumps({**wrong, "total_episodes": 0, "total_successes": 0, "tasks": []}),
                encoding="utf-8",
            )
            summary_path = root / "summary.json"
            with self.assertRaisesRegex(ValueError, "slice protocol mismatch"):
                launcher.aggregate(args, root / "checkpoint.pth", [slice_path], summary_path)
            self.assertFalse(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
