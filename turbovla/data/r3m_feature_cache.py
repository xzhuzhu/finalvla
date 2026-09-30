"""Validated, CPU-readable offline raw R3M feature caches.

The cache deliberately stores ResNet features *before* TurboVLA's trainable
projection.  It is therefore safe only for frozen R3M, while preserving all
projection gradients and checkpoint keys.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
FEATURE_DTYPE = "float32"
PREPROCESS = {
    "image_size": 224,
    "crop": "deterministic_center_crop_256_to_224",
    "normalization": "imagenet_rgb_0_1",
}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_identity(steps: list[Mapping[str, Any]]) -> str:
    """Content identity independent of TFDS shuffle, worker, or distributed rank."""
    digest = hashlib.sha256()
    for step in steps:
        observation = step["observation"]
        for key in ("image", "wrist_image", "state"):
            value = np.ascontiguousarray(np.asarray(observation[key]))
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
        action = np.ascontiguousarray(np.asarray(step["action"]))
        digest.update(b"action")
        digest.update(str(action.dtype).encode("ascii"))
        digest.update(np.asarray(action.shape, dtype=np.int64).tobytes())
        digest.update(action.tobytes())
        language = step["language_instruction"]
        if isinstance(language, np.ndarray):
            language = language.item()
        if isinstance(language, bytes):
            language = language.decode("utf-8")
        digest.update(str(language).encode("utf-8"))
    return digest.hexdigest()


def source_identity(dataset_dir: str | os.PathLike[str]) -> str:
    """Small source fingerprint, supplementing the per-episode content identity."""
    root = Path(dataset_dir).resolve()
    digest = hashlib.sha256(str(root).encode("utf-8"))
    for name in ("dataset_info.json", "features.json"):
        candidate = root / name
        if candidate.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(candidate.read_bytes())
    return digest.hexdigest()


def cache_manifest(
    checkpoint_path: str | os.PathLike[str], dataset_dirs: list[str], compute_precision: str
) -> dict[str, Any]:
    if compute_precision not in {"bf16_autocast", "fp32"}:
        raise ValueError("R3M cache compute_precision must be bf16_autocast or fp32")
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "preprocess": PREPROCESS,
        "feature_dtype": FEATURE_DTYPE,
        "compute_precision": compute_precision,
        "feature_shape": [2, 512],
        "dataset_sources": {str(Path(path).resolve()): source_identity(path) for path in dataset_dirs},
    }


class R3MFeatureCache:
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        checkpoint_path: str | os.PathLike[str] | None = None,
        dataset_dirs: list[str] | None = None,
        compute_precision: str = "bf16_autocast",
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"R3M feature cache manifest not found: {manifest_path}")
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid R3M feature cache manifest: {manifest_path}") from error
        expected = {
            "schema_version": SCHEMA_VERSION,
            "preprocess": PREPROCESS,
            "feature_dtype": FEATURE_DTYPE,
            "compute_precision": compute_precision,
            "feature_shape": [2, 512],
        }
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise ValueError(f"R3M feature cache mismatch for {key}: {self.manifest.get(key)!r}")
        if not isinstance(self.manifest.get("checkpoint_sha256"), str):
            raise ValueError("R3M feature cache manifest is missing checkpoint_sha256")
        if checkpoint_path:
            actual = sha256_file(checkpoint_path)
            if self.manifest["checkpoint_sha256"] != actual:
                raise ValueError("R3M feature cache checkpoint hash does not match --r3m_path")
        if dataset_dirs is not None:
            expected_sources = {str(Path(path).resolve()): source_identity(path) for path in dataset_dirs}
            if self.manifest.get("dataset_sources") != expected_sources:
                raise ValueError("R3M feature cache dataset identity does not match configured dataset_dirs")

    def episode_path(self, identity: str) -> Path:
        return self.root / "episodes" / f"{identity}.pt"

    def load_episode(self, identity: str, frame_count: int) -> torch.Tensor:
        path = self.episode_path(identity)
        if not path.is_file():
            raise FileNotFoundError(f"R3M feature cache is missing episode {identity}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("episode_identity") != identity:
            raise ValueError(f"invalid R3M feature cache episode payload: {path}")
        features = payload.get("features")
        expected_shape = (frame_count, 2, 512)
        if not isinstance(features, torch.Tensor) or tuple(features.shape) != expected_shape:
            raise ValueError(f"R3M feature cache shape mismatch in {path}: expected {expected_shape}")
        if features.dtype != torch.float32 or not torch.isfinite(features).all():
            raise ValueError(f"R3M feature cache precision/content mismatch in {path}")
        return features.contiguous()

    def write_episode_atomic(self, identity: str, features: torch.Tensor) -> bool:
        if features.ndim != 3 or features.shape[1:] != (2, 512):
            raise ValueError(f"cached R3M features must be [T,2,512], got {tuple(features.shape)}")
        features = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(features).all():
            raise ValueError("refusing to cache non-finite R3M features")
        destination = self.episode_path(identity)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            self.load_episode(identity, features.shape[0])
            return False
        temporary = destination.with_suffix(f".tmp.{os.getpid()}")
        try:
            torch.save({"episode_identity": identity, "features": features}, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return True


def initialize_cache(
    root: str | os.PathLike[str], checkpoint_path: str | os.PathLike[str], dataset_dirs: list[str],
    *, compute_precision: str = "bf16_autocast",
) -> R3MFeatureCache:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    expected = cache_manifest(checkpoint_path, dataset_dirs, compute_precision)
    manifest_path = root_path / MANIFEST_NAME
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("existing R3M feature cache manifest does not match requested inputs")
    else:
        temporary = manifest_path.with_suffix(f".tmp.{os.getpid()}")
        try:
            temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return R3MFeatureCache(
        root_path, checkpoint_path=checkpoint_path, dataset_dirs=dataset_dirs,
        compute_precision=compute_precision,
    )
