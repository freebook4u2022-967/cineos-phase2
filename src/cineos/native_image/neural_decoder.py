"""Neural latent decoding and visual comparison artifacts for CINEOS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .neural_backend import NeuralModelConfig, _load_torch


@dataclass(frozen=True, slots=True)
class DecodedRGBFrame:
    width: int
    height: int
    rgb: bytes

    def save_ppm(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            f"P6\n{self.width} {self.height}\n255\n".encode("ascii") + self.rgb
        )
        return destination


@dataclass(slots=True)
class TorchLatentRGBDecoder:
    """Small trainable decoder turning native neural latents into RGB pixels."""

    config: NeuralModelConfig
    width: int = 32
    height: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("decoder dimensions must be positive")
        torch = _load_torch()
        nn = torch.nn
        self.device_object = torch.device(self.device)
        output_dim = self.width * self.height * 3
        self.network = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, output_dim),
            nn.Sigmoid(),
        ).to(self.device_object)

    def decode(self, latent):
        return self.network(latent.to(self.device_object))

    def decode_frame(self, latent) -> DecodedRGBFrame:
        pixels = self.decode(latent).detach().cpu().reshape(-1)
        rgb = bytes(int(max(0.0, min(1.0, float(value))) * 255) for value in pixels)
        return DecodedRGBFrame(self.width, self.height, rgb)


@dataclass(frozen=True, slots=True)
class ImageComparisonArtifacts:
    reconstruction_path: str
    generated_path: str


def save_latent_comparison(
    decoder: TorchLatentRGBDecoder,
    target_latent,
    generated_latent,
    output_dir: str | Path,
) -> ImageComparisonArtifacts:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    reconstruction = decoder.decode_frame(target_latent).save_ppm(
        destination / "reconstruction.ppm"
    )
    generated = decoder.decode_frame(generated_latent).save_ppm(
        destination / "generated.ppm"
    )
    return ImageComparisonArtifacts(str(reconstruction), str(generated))
