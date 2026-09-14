"""Joint CINEOS training across image, identity, scene and flow components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .autoencoder import TorchPixelAutoencoder, load_p6_ppm
from .neural_backend import NeuralModelConfig, TorchCineosFlowModel, _load_torch
from .neural_encoders import TorchCharacterReferenceEncoder, TorchSceneTextEncoder
from .training import NativeTrainingSample


@dataclass(frozen=True, slots=True)
class JointTrainingStepResult:
    step: int
    total_loss: float
    reconstruction_loss: float
    flow_loss: float


@dataclass(slots=True)
class JointConditionalImageTrainer:
    dataset_root: Path
    autoencoder: TorchPixelAutoencoder
    config: NeuralModelConfig
    device: str = "cpu"
    learning_rate: float = 1e-4
    flow_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.autoencoder.latent_dim != self.config.latent_dim:
            raise ValueError("autoencoder and flow latent dimensions must match")
        torch = _load_torch()
        self.dataset_root = self.dataset_root.resolve()
        self.flow_model = TorchCineosFlowModel(self.config, device=self.device)
        self.identity_encoder = TorchCharacterReferenceEncoder(
            self.config,
            device=self.device,
        )
        self.scene_encoder = TorchSceneTextEncoder(self.config, device=self.device)
        parameters = [
            *self.autoencoder.encoder.parameters(),
            *self.autoencoder.mean_head.parameters(),
            *self.autoencoder.logvar_head.parameters(),
            *self.autoencoder.decoder.parameters(),
            *self.flow_model.parameters(),
            *self.identity_encoder.network.parameters(),
            *self.scene_encoder.network.parameters(),
        ]
        self.optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)
        self.step = 0

    def _resolve(self, relative_path: str) -> Path:
        path = (self.dataset_root / relative_path).resolve()
        if self.dataset_root not in path.parents and path != self.dataset_root:
            raise ValueError("training path escapes configured dataset root")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def train_sample(self, sample: NativeTrainingSample) -> JointTrainingStepResult:
        torch = _load_torch()
        pixels, width, height = load_p6_ppm(self._resolve(sample.image_path))
        if (width, height) != (self.autoencoder.width, self.autoencoder.height):
            raise ValueError("sample dimensions do not match configured autoencoder")
        pixels = pixels.to(self.autoencoder.device_object)
        mean, logvar = self.autoencoder.encode(pixels)
        target_latent = self.autoencoder.reparameterize(
            mean,
            logvar,
            deterministic=True,
        )
        reconstructed = self.autoencoder.decode(target_latent)
        reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, pixels)

        references = tuple(
            self._resolve(path) for path in sample.character_reference_paths
        )
        identity = self.identity_encoder.encode_files(references).unsqueeze(0)
        scene = self.scene_encoder.encode(
            sample.caption,
            sample.scene_description,
            sample.continuity_tags,
        ).unsqueeze(0)
        source = torch.zeros_like(target_latent).unsqueeze(0)
        target = target_latent.unsqueeze(0)
        times = torch.full(
            (1,),
            0.5,
            dtype=torch.float32,
            device=self.flow_model.device,
        )
        interpolated = 0.5 * source + 0.5 * target
        predicted_velocity = self.flow_model.predict_velocity(
            identity,
            scene,
            interpolated,
            times,
        )
        target_velocity = target - source
        flow_loss = torch.nn.functional.mse_loss(
            predicted_velocity,
            target_velocity,
        )
        total_loss = reconstruction_loss + (self.flow_weight * flow_loss)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self.step += 1
        return JointTrainingStepResult(
            step=self.step,
            total_loss=float(total_loss.detach().cpu().item()),
            reconstruction_loss=float(reconstruction_loss.detach().cpu().item()),
            flow_loss=float(flow_loss.detach().cpu().item()),
        )
