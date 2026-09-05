"""First CINEOS real-image mini-training experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .neural_backend import (
    NeuralModelConfig,
    TorchCineosFlowModel,
    TorchFlowTrainingRunner,
    _load_torch,
)
from .neural_encoders import (
    TorchCharacterReferenceEncoder,
    TorchImageLatentEncoder,
    TorchSceneTextEncoder,
)
from .training import NativeDatasetManifest, NativeTrainingSample


@dataclass(frozen=True, slots=True)
class ImageExperimentMetrics:
    training_loss: float
    validation_loss: float
    training_steps: int
    checkpoint_path: str


@dataclass(slots=True)
class RealImageExperimentRunner:
    dataset_root: Path
    config: NeuralModelConfig
    device: str = "cpu"
    learning_rate: float = 1e-4

    def __post_init__(self) -> None:
        self.dataset_root = self.dataset_root.resolve()
        self.model = TorchCineosFlowModel(self.config, device=self.device)
        self.trainer = TorchFlowTrainingRunner(
            self.model,
            learning_rate=self.learning_rate,
        )
        self.image_encoder = TorchImageLatentEncoder(self.config, device=self.device)
        self.identity_encoder = TorchCharacterReferenceEncoder(
            self.config,
            device=self.device,
        )
        self.scene_encoder = TorchSceneTextEncoder(self.config, device=self.device)

    def _resolve(self, relative_path: str) -> Path:
        path = (self.dataset_root / relative_path).resolve()
        if self.dataset_root not in path.parents and path != self.dataset_root:
            raise ValueError("training path escapes configured dataset root")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _encode_sample(self, sample: NativeTrainingSample):
        torch = _load_torch()
        image_path = self._resolve(sample.image_path)
        reference_paths = tuple(
            self._resolve(path) for path in sample.character_reference_paths
        )
        mean, logvar = self.image_encoder.encode_file(image_path)
        target = self.image_encoder.sample(mean, logvar, deterministic=True)
        identity = self.identity_encoder.encode_files(reference_paths)
        scene = self.scene_encoder.encode(
            sample.caption,
            sample.scene_description,
            sample.continuity_tags,
        )
        return (
            identity.to(self.model.device),
            scene.to(self.model.device),
            torch.zeros_like(target).to(self.model.device),
            target.to(self.model.device),
        )

    def _batch(self, samples: list[NativeTrainingSample]):
        torch = _load_torch()
        encoded = [self._encode_sample(sample) for sample in samples]
        identity = torch.stack([item[0] for item in encoded])
        scene = torch.stack([item[1] for item in encoded])
        source = torch.stack([item[2] for item in encoded])
        target = torch.stack([item[3] for item in encoded])
        times = torch.full(
            (len(samples),),
            0.5,
            dtype=torch.float32,
            device=self.model.device,
        )
        return identity, scene, source, target, times

    def evaluate(self, samples: list[NativeTrainingSample]) -> float:
        torch = _load_torch()
        if not samples:
            raise ValueError("validation samples must not be empty")
        identity, scene, source, target, times = self._batch(samples)
        interpolation_time = times.unsqueeze(-1)
        interpolated = ((1.0 - interpolation_time) * source) + (
            interpolation_time * target
        )
        target_velocity = target - source
        with torch.no_grad():
            predicted = self.model.predict_velocity(
                identity,
                scene,
                interpolated,
                times,
            )
            loss = torch.nn.functional.mse_loss(predicted, target_velocity)
        return float(loss.detach().cpu().item())

    def run(
        self,
        manifest: NativeDatasetManifest,
        *,
        checkpoint_path: str | Path,
        epochs: int = 1,
    ) -> ImageExperimentMetrics:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if len(manifest.samples) < 2:
            raise ValueError("real-image experiment requires at least two samples")

        split = max(1, len(manifest.samples) - 1)
        training_samples = manifest.samples[:split]
        validation_samples = manifest.samples[split:]
        training_loss = 0.0
        for _ in range(epochs):
            batch = self._batch(training_samples)
            training_loss = self.trainer.train_batch(*batch)

        validation_loss = self.evaluate(validation_samples)
        checkpoint = self.trainer.save_checkpoint(checkpoint_path)
        return ImageExperimentMetrics(
            training_loss=training_loss,
            validation_loss=validation_loss,
            training_steps=self.trainer.step,
            checkpoint_path=str(checkpoint),
        )
