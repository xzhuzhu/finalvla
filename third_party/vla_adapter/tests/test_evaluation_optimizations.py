from collections import OrderedDict
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from turbovla.evaluation.policy import (
    ACTION_MAX,
    ACTION_MIN,
    TurboVLAPolicy,
    normalized_action_chunk_to_env_actions,
)
from turbovla.models.text_encoder import TurboVLATextEncoder
from vla_adapter.rollout import GenerateConfig, _run_episode


class ActionConversionTest(unittest.TestCase):
    def test_vectorized_conversion_matches_the_scalar_protocol(self):
        normalized = np.asarray(
            [
                [-1.0, -0.5, 0.0, 0.5, 1.0, 0.25, -0.2],
                [1.0, 0.5, 0.0, -0.5, -1.0, -0.25, 0.3],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        expected_arm = (
            0.5 * (normalized[:, :6] + 1.0) * (ACTION_MAX[:6] - ACTION_MIN[:6])
            + ACTION_MIN[:6]
        )
        expected = np.concatenate(
            [expected_arm, np.asarray([[-1.0], [1.0], [1.0]], dtype=np.float32)],
            axis=1,
        )

        actual = normalized_action_chunk_to_env_actions(normalized)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(actual.dtype, np.float32)

    def test_predict_slices_before_vectorized_conversion(self):
        policy = object.__new__(TurboVLAPolicy)
        prediction = np.zeros((12, 7), dtype=np.float32)
        prediction[10:] = np.nan
        policy.predict_normalized_action_chunk = lambda *_args, **_kwargs: prediction

        output = TurboVLAPolicy.predict_env_action_chunk(
            policy,
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
            "task",
            np.zeros(8, dtype=np.float32),
            execute_steps=10,
        )

        self.assertEqual(output.shape, (10, 7))
        self.assertTrue(np.isfinite(output).all())


class TextEncoderCacheTest(unittest.TestCase):
    @staticmethod
    def make_encoder():
        encoder = object.__new__(TurboVLATextEncoder)
        nn.Module.__init__(encoder)
        encoder.config = SimpleNamespace(
            frozen=True,
            force_eval_when_frozen=True,
            zero_padded_tokens=False,
        )
        encoder.bert = nn.Identity()
        encoder.text_projection = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            encoder.text_projection.weight.copy_(torch.eye(2))
        encoder._eval_cache = OrderedDict()
        encoder.use_frozen_training_cache = False
        encoder.calls = 0

        def encode_bert_hidden(self, instructions, device):
            self.calls += 1
            hidden = torch.full((1, 2, 2), float(self.calls), device=device)
            token_mask = torch.ones((1, 2), dtype=torch.bool, device=device)
            self_attention = torch.eye(2, dtype=torch.bool, device=device).unsqueeze(0)
            return hidden, token_mask, self_attention

        encoder.encode_bert_hidden = MethodType(encode_bert_hidden, encoder)
        encoder.eval()
        return encoder

    def test_repeated_single_instruction_is_cached_only_during_inference(self):
        encoder = self.make_encoder()
        device = torch.device("cpu")

        with torch.inference_mode():
            first = encoder(["pick up the mug"], device)
            second = encoder(["pick up the mug"], device)
            encoder(["open the drawer"], device)

        self.assertEqual(encoder.calls, 2)
        self.assertEqual(encoder.eval_cache_size, 2)
        self.assertEqual(first[0].data_ptr(), second[0].data_ptr())

        encoder.train()
        self.assertEqual(encoder.eval_cache_size, 0)
        encoder.eval()
        with torch.inference_mode():
            encoder(["pick up the mug"], device)
        self.assertEqual(encoder.calls, 3)


class DINOv3CadenceTest(unittest.TestCase):
    class _Processor:
        def __init__(self, size, dtype):
            self.size = size
            self.dtype = dtype
            self.calls = 0

        def __call__(self, images):
            self.calls += 1
            return {
                "pixel_values": torch.zeros(
                    len(images), 3, self.size, self.size, dtype=self.dtype
                )
            }

    class _VisionModel:
        def __init__(self):
            self.calls = 0

        def encode_vision(self, pixels):
            self.calls += 1
            return pixels + 10 * self.calls

    def make_policy(self):
        policy = object.__new__(TurboVLAPolicy)
        policy.dinov3_update_interval = 2
        policy._dinov3_token_cache = None
        policy._policy_query_count = 0
        policy.model = self._VisionModel()
        return policy

    def test_dinov3_refreshes_every_two_queries(self):
        policy = self.make_policy()

        self.assertTrue(policy._should_refresh_dinov3())
        first = policy._attach_dinov3_tokens(
            {"dinov3": torch.ones(1, 2, 3)}, refresh=True
        )["dinov3_tokens"]
        policy._policy_query_count += 1

        self.assertFalse(policy._should_refresh_dinov3())
        second = policy._attach_dinov3_tokens({}, refresh=False)["dinov3_tokens"]
        policy._policy_query_count += 1

        self.assertTrue(policy._should_refresh_dinov3())
        third = policy._attach_dinov3_tokens(
            {"dinov3": torch.full((1, 2, 3), 2.0)}, refresh=True
        )["dinov3_tokens"]

        self.assertEqual(policy.model.calls, 2)
        self.assertIs(first, second)
        self.assertFalse(torch.equal(second, third))

    def test_episode_reset_invalidates_tokens_and_query_phase(self):
        policy = self.make_policy()
        policy._state_history = []
        policy._image_history = []
        policy._dinov3_token_cache = torch.ones(1, 2, 3)
        policy._policy_query_count = 7

        policy.reset_history()

        self.assertIsNone(policy._dinov3_token_cache)
        self.assertEqual(policy._policy_query_count, 0)
        self.assertTrue(policy._should_refresh_dinov3())

    def test_r3m_preprocessing_stays_current_when_dinov3_is_reused(self):
        policy = self.make_policy()
        policy.device = torch.device("cpu")
        policy.dinov3_processor = self._Processor(256, torch.float32)
        policy.r3m_processor = self._Processor(224, torch.uint8)
        images = [np.zeros((256, 256, 3), dtype=np.uint8)]
        states = [np.zeros(8, dtype=np.float32)]

        first, _ = policy._build_batch(images, images, states, include_dinov3=True)
        second, _ = policy._build_batch(images, images, states, include_dinov3=False)

        self.assertIn("dinov3", first)
        self.assertNotIn("dinov3", second)
        self.assertIn("r3m", first)
        self.assertIn("r3m", second)
        self.assertEqual(policy.dinov3_processor.calls, 1)
        self.assertEqual(policy.r3m_processor.calls, 2)


class _FakeEnv:
    def __init__(self, done_after: int = 3):
        self.done_after = done_after
        self.steps = 0
        self.observation = {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((2, 2, 3), dtype=np.uint8),
        }

    def reset(self):
        self.steps = 0

    def set_init_state(self, _initial_state):
        return self.observation

    def step(self, _action):
        self.steps += 1
        return self.observation, 0.0, self.steps >= self.done_after, {}


class _FakePolicy:
    def __init__(self):
        self.queries = 0
        self.resets = 0
        self.executed = []

    def reset_history(self):
        self.resets += 1
        self.executed.clear()

    def record_executed(self, obs, action):
        self.executed.append((obs, np.asarray(action).copy()))

    def predict_env_action_chunk(self, *_args, execute_steps, **_kwargs):
        self.queries += 1
        return np.zeros((execute_steps, 7), dtype=np.float32)


class RolloutImageProcessingTest(unittest.TestCase):
    def run_episode(self, save_video: bool):
        cfg = GenerateConfig(
            task_suite_name="libero_10",
            num_steps_wait=0,
            num_open_loop_steps=2,
            save_video=save_video,
        )
        env = _FakeEnv(done_after=3)
        policy = _FakePolicy()
        rotations = []

        def rotate(image):
            rotations.append(image)
            return image

        result = _run_episode(
            cfg,
            env,
            policy,
            "task",
            np.zeros(1, dtype=np.float32),
            lambda: [0.0] * 7,
            rotate,
        )
        return result, policy, rotations

    def test_no_video_rotates_images_only_when_the_policy_is_queried(self):
        (success, replay_images, horizons), policy, rotations = self.run_episode(save_video=False)

        self.assertTrue(success)
        self.assertEqual(policy.queries, 2)
        self.assertEqual(policy.resets, 1)
        self.assertEqual(len(policy.executed), 3)
        self.assertEqual(horizons, [2, 2])
        self.assertEqual(len(rotations), 4)
        self.assertEqual(replay_images, [])

    def test_video_mode_keeps_one_frame_per_control_step(self):
        (success, replay_images, horizons), policy, rotations = self.run_episode(save_video=True)

        self.assertTrue(success)
        self.assertEqual(policy.queries, 2)
        self.assertEqual(horizons, [2, 2])
        self.assertEqual(len(rotations), 6)
        self.assertEqual(len(replay_images), 3)


if __name__ == "__main__":
    unittest.main()
