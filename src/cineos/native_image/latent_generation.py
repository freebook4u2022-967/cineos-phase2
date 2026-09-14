"""Conditional latent generation that connects the autoencoder and flow model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .autoencoder import TorchPixelAutoencoder
from .neural_backend import NeuralModelConfig, TorchCineosFlowModel, _load_torch
from .neural_encoders import TorchCharacterReferenceEncoder, TorchSceneTextEncoder


@dataclass(frozen=True, slots=True)
class LatentGenerationResult:
    latent: object
    pixels: object


@dataclass(slots=True)
class ConditionalLatentGenerator:
    """Generate a new image latent from identity and scene conditioning."""

    autoencoder: TorchPixelAutoencoder
    config: NeuralModelConfig
    device: str = "cpu"
    integration_steps: int = 8

    def __post_init__(self) -> None:
        if self.integration_steps < 1:
            raise ValueError("integration_steps must be at least 1")
        if self.autoencoder.latent_dim != self.config.latent_dim:
            raise ValueError("autoencoder and flow latent dimensions must match")
        self.flow_model = TorchCineosFlowModel(self.config, device=self.device)
        self.identity_encoder = TorchCharacterReferenceEncoder(
            self.config,
            device=self.device,
        )
        self.scene_encoder = TorchSceneTextEncoder(self.config, device=self.device)

    def sample_latent(
        self,
        reference_paths: tuple[str | Path, ...],
        caption: str,
        scene_description: str,
        continuity_tags: tuple[str, ...] = (),
        *,
        seed: int = 0,
    ):
        torch = _load_torch()
        torch.manual_seed(seed)
        identity = self.identity_encoder.encode_files(reference_paths).unsqueeze(0)
        scene = self.scene_encoder.encode(
            caption,
            scene_description,
            continuity_tags,
        ).unsqueeze(0)
        latent = torch.randn(
            (1, self.config.latent_dim),
            dtype=torch.float32,
            device=self.flow_model.device,
        )
        dt = 1.0 / self.integration_steps
        for step in range(self.integration_steps):
            time_value = step / self.integration_steps
            times = torch.full(
                (1,),
                time_value,
                dtype=torch.float32,
                device=self.flow_model.device,
            )
            velocity = self.flow_model.predict_velocity(
                identity,
                scene,
                latent,
                times,
            )
            latent = latent + (dt * velocity)
        return latent.squeeze(0)

    def generate(
        self,
        reference_paths: tuple[str | Path, ...],
        caption: str,
        scene_description: str,
        continuity_tags: tuple[str, ...] = (),
        *,
        seed: int = 0,
    ) -> LatentGenerationResult:
        latent = self.sample_latent(
            reference_paths,
            caption,
            scene_description,
            continuity_tags,
            seed=seed,
        )
        pixels = self.autoencoder.decode(latent).detach().cpu()
        return LatentGenerationResult(latent=latent.detach().cpu(), pixels=pixels)
