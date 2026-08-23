"""Real-data preparation for CINEOS neural flow training.

The pipeline converts approved CINEOS training manifests into deterministic
numeric features and image-derived latent targets. It intentionally performs no
web scraping and consumes only paths explicitly present in approved manifests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .neural_backend import NeuralModelConfig, TorchFlowTrainingRunner, _load_torch
from .training import NativeDatasetManifest, NativeTrainingSample


def _hash_features(payload: bytes, dimensions: int) -> tuple[float, ...]:
    if dimensions <= 0:
        raise ValueError("feature dimensions must be positive")
    digest = hashlib.sha512(payload).digest()
    return tuple(
        ((digest[index % len(digest)] / 255.0) * 2.0) - 1.0
        for index in range(dimensions)
    )


@dataclass(frozen=True, slots=True)
class PreparedNeuralSample:
    sample_id: str
    identity_features: tuple[float, ...]
    scene_features: tuple[float, ...]
    target_latent: tuple[float, ...]


@dataclass(slots=True)
class ApprovedManifestPreprocessor:
    config: NeuralModelConfig
    dataset_root: Path

    def _read_required_file(self, relative_path: str) -> bytes:
        path = (self.dataset_root / relative_path).resolve()
        root = self.dataset_root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("training path escapes configured dataset root")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()

    def prepare(self, sample: NativeTrainingSample) -> PreparedNeuralSample:
        image_bytes = self._read_required_file(sample.image_path)
        reference_bytes = b"".join(
            self._read_required_file(path)
            for path in sample.character_reference_paths
        )
        identity_payload = reference_bytes + "|".join(sample.identity_tags).encode()
        scene_payload = "|".join(
            (sample.caption, sample.scene_description, *sample.continuity_tags)
        ).encode("utf-8")
        return PreparedNeuralSample(
            sample_id=sample.sample_id,
            identity_features=_hash_features(
                identity_payload,
                self.config.feature_dim,
            ),
            scene_features=_hash_features(scene_payload, self.config.feature_dim),
            target_latent=_hash_features(image_bytes, self.config.latent_dim),
        )

    def prepare_dataset(
        self,
        manifest: NativeDatasetManifest,
    ) -> tuple[PreparedNeuralSample, ...]:
        return tuple(self.prepare(sample) for sample in manifest.samples)


@dataclass(slots=True)
class NeuralManifestTrainingPipeline:
    preprocessor: ApprovedManifestPreprocessor
    runner: TorchFlowTrainingRunner

    def train_manifest(self, manifest: NativeDatasetManifest) -> float:
        torch = _load_torch()
        prepared = self.preprocessor.prepare_dataset(manifest)
        if not prepared:
            raise ValueError("cannot train neural model on an empty manifest")
        device = self.runner.model.device
        identity = torch.tensor(
            [item.identity_features for item in prepared],
            dtype=torch.float32,
            device=device,
        )
        scene = torch.tensor(
            [item.scene_features for item in prepared],
            dtype=torch.float32,
            device=device,
        )
        target = torch.tensor(
            [item.target_latent for item in prepared],
            dtype=torch.float32,
            device=device,
        )
        source = torch.zeros_like(target)
        times = torch.linspace(
            0.1,
            0.9,
            steps=len(prepared),
            dtype=torch.float32,
            device=device,
        )
        return self.runner.train_batch(identity, scene, source, target, times)
