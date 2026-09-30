import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoImageProcessor
import tensorflow as tf
import tensorflow_datasets as tfds

from .r3m_feature_cache import R3MFeatureCache, episode_identity

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


class LiberoRLDSDataset(IterableDataset):
    def __init__(
        self,
        dataset_dir,
        LOCAL_DINOV3_PATH,
        rank=0,
        world_size=1,
        chunk_size=8,
        split="train",
        shuffle_buffer=512,
        shuffle_steps_within_episode=False,
        step_mix_buffer_size=0,
        seed=42,
        local_files_only=True,
        expected_image_size=256,
        backbone="dinov3",
        use_r3m=False,
        history_length=0,
        history_visual_dinov3=False,
        history_r3m=False,
        r3m_feature_cache_path="",
        r3m_checkpoint_path="",
        r3m_cache_dataset_dirs=None,
        r3m_cache_precision="bf16_autocast",
    ):
        self.dataset_dir = dataset_dir
        self.backbone = str(backbone)
        self.use_r3m = bool(use_r3m)
        if self.backbone not in ("dinov3", "qwen3vl"):
            raise ValueError(f"unsupported backbone={self.backbone!r}; expected 'dinov3' or 'qwen3vl'")
        if self.use_r3m and self.backbone != "dinov3":
            raise ValueError("R3M current-image processing requires backbone='dinov3'")

        self.dino_processor = None
        if self.backbone == "dinov3":
            self.dino_processor = AutoImageProcessor.from_pretrained(
                LOCAL_DINOV3_PATH,
                local_files_only=bool(local_files_only),
            )
            self._disable_spatial_resize(self.dino_processor)

        self.chunk_size = int(chunk_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.split = split
        self.shuffle_buffer = int(shuffle_buffer)
        self.shuffle_steps_within_episode = bool(shuffle_steps_within_episode)
        self.step_mix_buffer_size = int(step_mix_buffer_size)
        self.seed = int(seed)
        self.expected_image_size = int(expected_image_size)
        self.history_length = int(history_length)
        self.history_visual_dinov3 = bool(history_visual_dinov3)
        self.history_r3m = bool(history_r3m)
        self.r3m_feature_cache_path = str(r3m_feature_cache_path or "")
        self.r3m_checkpoint_path = str(r3m_checkpoint_path or "")
        self.r3m_cache_dataset_dirs = list(r3m_cache_dataset_dirs or [dataset_dir])
        self.r3m_feature_cache = None
        if self.r3m_feature_cache_path:
            if not self.use_r3m:
                raise ValueError("r3m_feature_cache_path requires use_r3m=True")
            if not self.r3m_checkpoint_path:
                raise ValueError("r3m_feature_cache_path requires r3m_checkpoint_path for validation")
            # The cache is CPU-only; data workers never initialize CUDA.
            self.r3m_feature_cache = R3MFeatureCache(
                self.r3m_feature_cache_path,
                checkpoint_path=self.r3m_checkpoint_path,
                dataset_dirs=self.r3m_cache_dataset_dirs,
                compute_precision=r3m_cache_precision,
            )
        if self.history_length < 0:
            raise ValueError("history_length must be non-negative")
        if self.history_visual_dinov3 and (
            self.backbone != "dinov3" or self.history_length == 0
        ):
            raise ValueError(
                "DINOv3 visual history requires backbone='dinov3' and history_length>0"
            )
        if self.history_r3m and (not self.use_r3m or self.history_length != 12):
            raise ValueError("R3M history requires use_r3m=True and history_length=12")

        self.proprio_mean = torch.tensor(
            [
                -0.04190646484494209,
                0.03539437800645828,
                0.8257066607475281,
                2.908315658569336,
                -0.5562158823013306,
                -0.16649103164672852,
                0.02831534668803215,
                -0.028561558574438095,
            ],
            dtype=torch.float32,
        )

        self.proprio_std = torch.tensor(
            [
                0.10743443667888641,
                0.14424759149551392,
                0.25723373889923096,
                0.34413808584213257,
                1.234430193901062,
                0.35798805952072144,
                0.013308786787092686,
                0.013174591585993767,
            ],
            dtype=torch.float32,
        )

        self.action_min = torch.tensor(
            [
                -0.9375,
                -0.9375,
                -0.9375,
                -0.23642857372760773,
                -0.3053571283817291,
                -0.3675000071525574,
                -1.0,
            ],
            dtype=torch.float32,
        )
        self.action_max = torch.tensor(
            [
                0.9375,
                0.9375,
                0.9375,
                0.30000001192092896,
                0.29357144236564636,
                0.375,
                1.0,
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _disable_spatial_resize(processor):
        if hasattr(processor, "do_resize"):
            processor.do_resize = False
        if hasattr(processor, "do_center_crop"):
            processor.do_center_crop = False

    def _normalize_state(self, state):
        return (state - self.proprio_mean) / (self.proprio_std + 1e-6)

    def _normalize_action_chunk(self, action_chunk):
        action_chunk = action_chunk.clone()
        action_chunk[:, :6] = (
            2.0
            * (action_chunk[:, :6] - self.action_min[:6])
            / (self.action_max[:6] - self.action_min[:6] + 1e-6)
            - 1.0
        )
        action_chunk[:, :6] = action_chunk[:, :6].clamp(-1.0, 1.0)
        return action_chunk

    def _ensure_expected_size(self, img_np):
        height, width = img_np.shape[:2]
        if height != self.expected_image_size or width != self.expected_image_size:
            raise ValueError(
                f"Expected raw RLDS image to already be {self.expected_image_size}x{self.expected_image_size}, "
                f"but got {height}x{width}. This dataset loader intentionally does not resize."
            )

    @staticmethod
    def _ensure_processor_preserved_resolution(pixel_values, img_np, backbone_name):
        height, width = img_np.shape[:2]
        if tuple(pixel_values.shape[-2:]) != (height, width):
            raise RuntimeError(
                f"{backbone_name} preprocessor changed image size from {(height, width)} to "
                f"{tuple(pixel_values.shape[-2:])}. Resize/crop should stay disabled."
            )

    def _process_image_pair(self, img_np):
        self._ensure_expected_size(img_np)
        img = Image.fromarray(img_np)

        if self.backbone == "qwen3vl":
            return {"qwen3vl": img}

        dino_pixel_values = self.dino_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        self._ensure_processor_preserved_resolution(dino_pixel_values, img_np, "DINOv3")

        processed = {"dinov3": dino_pixel_values}
        if self.use_r3m and self.r3m_feature_cache is None:
            processed["r3m"] = self._process_r3m_image(img_np)
        return processed

    def _process_r3m_image(self, img_np):
        self._ensure_expected_size(img_np)
        crop_start = (self.expected_image_size - 224) // 2
        crop_end = crop_start + 224
        crop = np.ascontiguousarray(img_np[crop_start:crop_end, crop_start:crop_end])
        if crop.shape != (224, 224, 3):
            raise RuntimeError(f"R3M center crop must be 224x224 RGB, got {crop.shape}")
        return torch.from_numpy(crop).permute(2, 0, 1)

    def _build_episode_image_cache(self, steps):
        if not (self.history_visual_dinov3 or (self.history_r3m and self.r3m_feature_cache is None)):
            return None
        if self.history_visual_dinov3:
            return [
                (
                    self._process_image_pair(step["observation"]["image"]),
                    self._process_image_pair(step["observation"]["wrist_image"]),
                )
                for step in steps
            ]
        return [
            (
                {"r3m": self._process_r3m_image(step["observation"]["image"])},
                {"r3m": self._process_r3m_image(step["observation"]["wrist_image"])},
            )
            for step in steps
        ]

    @staticmethod
    def _decode_instruction(raw_instruction):
        if isinstance(raw_instruction, bytes):
            return raw_instruction.decode("utf-8")
        if isinstance(raw_instruction, np.ndarray) and raw_instruction.dtype.type is np.bytes_:
            return raw_instruction.item().decode("utf-8")
        return str(raw_instruction)

    def _build_step_sample(self, steps, t, episode_len, image_cache=None, r3m_features=None):
        current_step = steps[t]
        if image_cache is None or not self.history_visual_dinov3:
            img1 = self._process_image_pair(current_step["observation"]["image"])
            img2 = self._process_image_pair(current_step["observation"]["wrist_image"])
        else:
            img1, img2 = image_cache[t]
        if r3m_features is not None:
            img1, img2 = dict(img1), dict(img2)
            img1["r3m_features"] = r3m_features[t, 0]
            img2["r3m_features"] = r3m_features[t, 1]

        instruction = self._decode_instruction(current_step["language_instruction"])

        state = torch.tensor(current_step["observation"]["state"], dtype=torch.float32)
        state = self._normalize_state(state)

        future_actions = [steps[i]["action"] for i in range(t, t + min(self.chunk_size, episode_len - t))]
        valid_len = len(future_actions)
        padding_len = self.chunk_size - valid_len

        if padding_len > 0:
            last_action = future_actions[-1]
            future_actions.extend([last_action] * padding_len)

        action_chunk = torch.tensor(np.stack(future_actions), dtype=torch.float32)
        action_chunk = self._normalize_action_chunk(action_chunk)

        action_chunk_mask = torch.zeros(self.chunk_size, dtype=torch.float32)
        action_chunk_mask[:valid_len] = 1.0

        if self.history_length == 0:
            return (img1, img2), instruction, state, action_chunk, action_chunk_mask
        first = max(0, t - self.history_length)
        prior = steps[first:t]  # excludes a_t: no current-target leakage.
        count = len(prior)
        history_states = torch.zeros(self.history_length, state.numel(), dtype=torch.float32)
        history_mask = torch.zeros(self.history_length, dtype=torch.bool)
        if count:
            raw_states = torch.tensor(np.stack([item["observation"]["state"] for item in prior]), dtype=torch.float32)
            history_states[-count:] = self._normalize_state(raw_states)
            history_mask[-count:] = True
        base = (
            (img1, img2),
            instruction,
            state,
            action_chunk,
            action_chunk_mask,
            history_states,
            history_mask,
        )
        if not (self.history_visual_dinov3 or self.history_r3m):
            return base
        if image_cache is None:
            image_cache = self._build_episode_image_cache(steps)
        history_visuals = {}
        if self.history_visual_dinov3:
            history_dinov3 = torch.zeros(
                self.history_length,
                2,
                3,
                self.expected_image_size,
                self.expected_image_size,
                dtype=torch.float32,
            )
            if count:
                history_dinov3[-count:] = torch.stack(
                    [
                        torch.stack(
                            [
                                image_cache[index][0]["dinov3"],
                                image_cache[index][1]["dinov3"],
                            ],
                            dim=0,
                        )
                        for index in range(first, t)
                    ],
                    dim=0,
                )
            history_visuals["dinov3_history"] = history_dinov3
        if self.history_r3m:
            if r3m_features is None:
                history_r3m = torch.zeros(
                    self.history_length, 2, 3, 224, 224, dtype=torch.uint8
                )
            else:
                history_r3m = torch.zeros(self.history_length, 2, 512, dtype=torch.float32)
            if count:
                if r3m_features is None:
                    history_r3m[-count:] = torch.stack(
                        [torch.stack([image_cache[index][0]["r3m"], image_cache[index][1]["r3m"]])
                         for index in range(first, t)]
                    )
                else:
                    history_r3m[-count:] = r3m_features[first:t]
            history_visuals["r3m_history_features" if r3m_features is not None else "r3m_history"] = history_r3m
        return base + (history_visuals,)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        base_seed = self.seed + 1009 * self.rank + 9176 * worker_id
        builder = tfds.builder_from_directory(builder_dir=self.dataset_dir)

        epoch = 0
        while True:
            epoch_seed = base_seed + epoch
            rng = np.random.default_rng(epoch_seed)
            step_buffer = []

            dataset = builder.as_dataset(split=self.split)

            if self.world_size > 1:
                dataset = dataset.shard(num_shards=self.world_size, index=self.rank)

            if worker_info is not None:
                dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer,
                seed=epoch_seed,
                reshuffle_each_iteration=False,
            )

            for episode in tfds.as_numpy(dataset):
                steps = list(episode["steps"])
                episode_len = len(steps)
                r3m_features = None
                if self.r3m_feature_cache is not None:
                    r3m_features = self.r3m_feature_cache.load_episode(episode_identity(steps), episode_len)
                image_cache = self._build_episode_image_cache(steps)
                step_indices = list(range(episode_len))

                if self.shuffle_steps_within_episode:
                    rng.shuffle(step_indices)

                if self.step_mix_buffer_size > 0:
                    for t in step_indices:
                        step_buffer.append(
                            self._build_step_sample(steps, t, episode_len, image_cache=image_cache, r3m_features=r3m_features)
                        )
                        if len(step_buffer) >= self.step_mix_buffer_size:
                            out_idx = int(rng.integers(0, len(step_buffer)))
                            yield step_buffer.pop(out_idx)
                else:
                    for t in step_indices:
                        yield self._build_step_sample(
                            steps,
                            t,
                            episode_len,
                            image_cache=image_cache, r3m_features=r3m_features,
                        )

            while len(step_buffer) > 0:
                out_idx = int(rng.integers(0, len(step_buffer)))
                yield step_buffer.pop(out_idx)

            epoch += 1


def vla_collate_fn(batch):
    if len(batch) == 0:
        raise ValueError("empty batch cannot be collated")

    dino_img1_list = []
    dino_img2_list = []
    r3m_img1_list = []
    r3m_img2_list = []
    instructions = []
    action_chunks = []
    action_chunk_masks = []
    states = []

    sample_length = len(batch[0])
    with_history = sample_length in {7, 8}
    with_visual_history = sample_length == 8
    history_states, history_masks = [], []
    history_visuals = {}
    use_r3m = "r3m" in batch[0][0][0]
    use_r3m_features = "r3m_features" in batch[0][0][0]
    for item in batch:
        if len(item) != sample_length or len(item) not in {5, 7, 8}:
            raise ValueError(
                "samples must uniformly have 5 fields, 7 fields with state history, "
                "or 8 fields with state and DINOv3 visual history"
            )
        images, instruction, state, action_chunk, action_chunk_mask = item[:5]
        if not isinstance(images, (list, tuple)) or len(images) != 2:
            raise ValueError("Each sample must contain two camera views: (img1, img2)")

        img1, img2 = images
        if "dinov3" not in img1 or "dinov3" not in img2:
            raise ValueError("Each view must contain preprocessed tensor for key 'dinov3'")

        dino_img1_list.append(img1["dinov3"])
        dino_img2_list.append(img2["dinov3"])
        if ("r3m" in img1) != use_r3m or ("r3m" in img2) != use_r3m:
            raise ValueError("R3M tensors must be present consistently in both camera views")
        if ("r3m_features" in img1) != use_r3m_features or ("r3m_features" in img2) != use_r3m_features:
            raise ValueError("cached R3M feature tensors must be present consistently in both camera views")
        if use_r3m and use_r3m_features:
            raise ValueError("samples cannot contain both R3M pixels and cached R3M features")
        if use_r3m:
            r3m_img1_list.append(img1["r3m"])
            r3m_img2_list.append(img2["r3m"])
        if use_r3m_features:
            r3m_img1_list.append(img1["r3m_features"])
            r3m_img2_list.append(img2["r3m_features"])
        instructions.append(instruction)
        action_chunks.append(action_chunk)
        action_chunk_masks.append(action_chunk_mask)
        states.append(state)
        if with_history:
            history_states.append(item[5])
            history_masks.append(item[6])
        if with_visual_history:
            item_history = item[7]
            if isinstance(item_history, dict):
                history_mapping = item_history
            else:
                history_mapping = {"dinov3_history": item_history}
            if not history_visuals:
                history_visuals = {key: [] for key in history_mapping}
            if set(history_mapping) != set(history_visuals):
                raise ValueError("visual-history tensors must use consistent keys across the batch")
            for key, tensor in history_mapping.items():
                history_visuals[key].append(tensor)

    dino_img1 = torch.stack(dino_img1_list, dim=0)
    dino_img2 = torch.stack(dino_img2_list, dim=0)

    samples = {"dinov3": torch.stack([dino_img1, dino_img2], dim=1)}
    if use_r3m:
        r3m_img1 = torch.stack(r3m_img1_list, dim=0)
        r3m_img2 = torch.stack(r3m_img2_list, dim=0)
        samples["r3m"] = torch.stack([r3m_img1, r3m_img2], dim=1)
    if use_r3m_features:
        r3m_img1 = torch.stack(r3m_img1_list, dim=0)
        r3m_img2 = torch.stack(r3m_img2_list, dim=0)
        samples["r3m_features"] = torch.stack([r3m_img1, r3m_img2], dim=1)
    if with_visual_history:
        for key, tensors in history_visuals.items():
            samples[key] = torch.stack(tensors)

    action_chunks = torch.stack(action_chunks, dim=0)
    action_chunk_masks = torch.stack(action_chunk_masks, dim=0)
    states = torch.stack(states, dim=0)

    base = samples, instructions, states, action_chunks, action_chunk_masks
    if with_history:
        return base + (torch.stack(history_states), torch.stack(history_masks))
    return base


class Qwen3VLCollator:
    """Batch collator that runs the Qwen3-VL processor on images + instructions."""

    def __init__(self, processor):
        self.processor = processor
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = 256 * 256
            image_processor.max_pixels = 256 * 256

    def __call__(self, batch):
        if len(batch) == 0:
            raise ValueError("empty batch cannot be collated")

        image_pairs = []
        instructions = []
        action_chunks = []
        action_chunk_masks = []
        states = []

        for images, instruction, state, action_chunk, action_chunk_mask in batch:
            if not isinstance(images, (list, tuple)) or len(images) != 2:
                raise ValueError("Each sample must contain two camera views: (img1, img2)")
            img1, img2 = images
            if "qwen3vl" not in img1 or "qwen3vl" not in img2:
                raise ValueError("Each view must contain raw image under key 'qwen3vl'")
            image_pairs.append([img1["qwen3vl"], img2["qwen3vl"]])
            instructions.append(instruction)
            action_chunks.append(action_chunk)
            action_chunk_masks.append(action_chunk_mask)
            states.append(state)

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
            key: processed[key]
            for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
            if key in processed
        }

        action_chunks = torch.stack(action_chunks, dim=0)
        action_chunk_masks = torch.stack(action_chunk_masks, dim=0)
        states = torch.stack(states, dim=0)

        return samples, instructions, states, action_chunks, action_chunk_masks
