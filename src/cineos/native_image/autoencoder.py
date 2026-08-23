"""Trainable RGB autoencoder for CINEOS native image experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .neural_backend import _load_torch


def load_p6_ppm(path: str | Path):
    """Load a simple binary P6 PPM file into a normalized RGB tensor."""
    torch = _load_torch()
    payload = Path(path).read_bytes()
    parts = payload.split(b"\n", 3)
    if len(parts) != 4 or parts[0].strip() != b"P6":
        raise ValueError("autoencoder currently requires binary P6 PPM input")
    dimensions = parts[1].split()
    if len(dimensions) != 2:
        raise ValueError("invalid PPM dimensions")
    width, height = (int(value) for value in dimensions)
    if parts[2].strip() != b"255":
        raise ValueError("PPM max value must be 255")
    expected = width * height * 3
    rgb = parts[3]
    if len(rgb) != expected:
        raise ValueError(f"PPM RGB payload has {len(rgb)} bytes; expected {expected}")
    tensor = torch.tensor(list(rgb), dtype=torch.float32) / 255.0
    return tensor, width, height


@dataclass(frozen=True, slots=True)
class AutoencoderStepResult:
    step: int
    total_loss: float
    reconstruction_loss: float
    kl_loss: float


@dataclass(slots=True)
class TorchPixelAutoencoder:
    """Small VAE-style trainable autoencoder over real RGB pixel vectors."""

    width: int
    height: int
    latent_dim: int = 32
    hidden_dim: int = 256
    device: str = "cpu"
    learning_rate: float = 1e-3
    beta: float = 1e-4

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("autoencoder dimensions must be positive")
        if self.latent_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("autoencoder latent/hidden dimensions must be positive")
        torch = _load_torch()
        nn = torch.nn
        self.device_object = torch.device(self.device)
        self.input_dim = self.width * self.height * 3
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
        ).to(self.device_object)
        self.mean_head = nn.Linear(self.hidden_dim, self.latent_dim).to(
            self.device_object
        )
        self.logvar_head = nn.Linear(self.hidden_dim, self.latent_dim).to(
            self.device_object
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Sigmoid(),
        ).to(self.device_object)
        parameters = [
            *self.encoder.parameters(),
            *self.mean_head.parameters(),
            *self.logvar_head.parameters(),
            *self.decoder.parameters(),
        ]
        self.optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate)
        self.step = 0

    def encode(self, pixels):
        hidden = self.encoder(pixels.to(self.device_object))
        return self.mean_head(hidden), self.logvar_head(hidden)

    def reparameterize(self, mean, logvar, *, deterministic: bool = False):
        torch = _load_torch()
        if deterministic:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def decode(self, latent):
        return self.decoder(latent.to(self.device_object))

    def reconstruct(self, pixels, *, deterministic: bool = True):
        mean, logvar = self.encode(pixels)
        latent = self.reparameterize(mean, logvar, deterministic=deterministic)
        return self.decode(latent), mean, logvar

    def train_pixels(self, pixels) -> AutoencoderStepResult:
        torch = _load_torch()
        target = pixels.to(self.device_object)
        if target.numel() != self.input_dim:
            raise ValueError("pixel vector does not match configured autoencoder size")
        reconstructed, mean, logvar = self.reconstruct(target, deterministic=False)
        reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, target)
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mean.pow(2) - logvar.exp())
        total_loss = reconstruction_loss + (self.beta * kl_loss)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self.step += 1
        return AutoencoderStepResult(
            step=self.step,
            total_loss=float(total_loss.detach().cpu().item()),
            reconstruction_loss=float(reconstruction_loss.detach().cpu().item()),
            kl_loss=float(kl_loss.detach().cpu().item()),
        )

    def train_ppm(self, path: str | Path, *, steps: int = 1) -> AutoencoderStepResult:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        pixels, width, height = load_p6_ppm(path)
        if (width, height) != (self.width, self.height):
            raise ValueError("PPM dimensions do not match configured autoencoder")
        result = None
        for _ in range(steps):
            result = self.train_pixels(pixels)
        assert result is not None
        return result
