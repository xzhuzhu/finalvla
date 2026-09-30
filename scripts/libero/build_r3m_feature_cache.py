#!/usr/bin/env python3
"""Precompute frozen raw R3M features once per TFDS episode.

Run with CUDA_VISIBLE_DEVICES selecting the intended physical GPU; inside that
visibility mask the builder uses cuda:0.  It is resume-safe and never replaces
a validated episode file.
"""

from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

from turbovla.data.r3m_feature_cache import episode_identity, initialize_cache
from turbovla.models.configuration import R3MEncoderConfig
from turbovla.models.r3m_encoder import R3MResNet18Encoder

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def parse_dirs(value: str) -> list[str]:
    result = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("--dataset_dirs must contain at least one TFDS directory")
    return result


def crop_frames(steps) -> torch.Tensor:
    frames = []
    for step in steps:
        observation = step["observation"]
        for key in ("image", "wrist_image"):
            image = np.asarray(observation[key])
            if image.shape != (256, 256, 3):
                raise ValueError(f"R3M cache expects 256x256 RGB frames, got {image.shape}")
            frames.append(np.ascontiguousarray(image[16:240, 16:240]))
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dirs", required=True, type=parse_dirs)
    parser.add_argument("--r3m_path", required=True)
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--precision", choices=["bf16_amp", "fp32"], default="bf16_amp",
                        help="Must match training --precision; cached values are stored as float32.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R3M cache builder requires CUDA; expose a physical GPU then use cuda:0")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")

    cache_precision = "bf16_autocast" if args.precision == "bf16_amp" else "fp32"
    cache = initialize_cache(args.cache_path, args.r3m_path, args.dataset_dirs,
                             compute_precision=cache_precision)
    device = torch.device("cuda:0")
    encoder = R3MResNet18Encoder(R3MEncoderConfig(enabled=True, checkpoint_path=args.r3m_path)).to(device).eval()
    written = skipped = 0
    with torch.inference_mode():
        for dataset_dir in args.dataset_dirs:
            builder = tfds.builder_from_directory(builder_dir=dataset_dir)
            for episode in tfds.as_numpy(builder.as_dataset(split="train")):
                steps = list(episode["steps"])
                identity = episode_identity(steps)
                destination = cache.episode_path(identity)
                if destination.is_file():
                    cache.load_episode(identity, len(steps))
                    skipped += 1
                    continue
                frames = crop_frames(steps)
                features = []
                for start in range(0, frames.shape[0], args.batch_size):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=args.precision == "bf16_amp"):
                        features.append(encoder(frames[start:start + args.batch_size].to(device, non_blocking=True)).cpu())
                wrote = cache.write_episode_atomic(identity, torch.cat(features).view(len(steps), 2, 512))
                written += int(wrote)
    print(f"r3m_feature_cache_complete written={written} skipped={skipped} path={args.cache_path}", flush=True)


if __name__ == "__main__":
    main()
