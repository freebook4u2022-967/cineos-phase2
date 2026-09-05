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


def save_p6_ppm(pixels, width: int, height: int, path: str | Path) -> Path:
    """Save a normalized RGB tensor as a dependency-free binary P6 PPM."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = pixels.detach().cpu().reshape(-1).clamp(0.0, 1.0)
    rgb = bytes(int(float(value) * 255) for value in values)
    expected = width * height * 3
    if len(rgb) != expected:
        raise ValueError("pixel vector does not match requested PPM dimensions")
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    destination.write_bytes(header + rgb)
    return destination


@dataclass(frozen=True, slots=True)
class AutoencoderStepResult:
    step: int
    total_loss: float
    reconstruction_loss: float
    kl_loss: float


@dataclass(frozen=True, slots=True)
class ReconstructionMetrics:
    mse: float
    mae: float
    psnr: float


@dataclass(frozen=True, slots=True)
class ReconstructionExport:
    original_path: str
    reconstructed_path: str
    checkpoint_path: str
    metrics: ReconstructionMetrics


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

    def save_checkpoint(self, path: str | Path) -> Path:
        torch = _load_torch()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "cineos-pixel-autoencoder/0.1",
                "width": self.width,
                "height": self.height,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "beta": self.beta,
                "step": self.step,
                "encoder": self.encoder.state_dict(),
                "mean_head": self.mean_head.state_dict(),
                "logvar_head": self.logvar_head.state_dict(),
                "decoder": self.decoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            destination,
        )
        return destination

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> TorchPixelAutoencoder:
        torch = _load_torch()
        payload = torch.load(path, map_location=device)
        if payload.get("schema") != "cineos-pixel-autoencoder/0.1":
            raise ValueError("unsupported autoencoder checkpoint schema")
        model = cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            latent_dim=int(payload["latent_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            device=device,
            learning_rate=float(payload["learning_rate"]),
            beta=float(payload["beta"]),
        )
        model.encoder.load_state_dict(payload["encoder"])
        model.mean_head.load_state_dict(payload["mean_head"])
        model.logvar_head.load_state_dict(payload["logvar_head"])
        model.decoder.load_state_dict(payload["decoder"])
        model.optimizer.load_state_dict(payload["optimizer"])
        model.step = int(payload["step"])
        return model

    def reconstruction_metrics(self, pixels) -> ReconstructionMetrics:
        import math

        torch = _load_torch()
        target = pixels.to(self.device_object)
        with torch.no_grad():
            reconstructed, _, _ = self.reconstruct(target, deterministic=True)
            mse = float(torch.nn.functional.mse_loss(reconstructed, target).item())
            mae = float(torch.nn.functional.l1_loss(reconstructed, target).item())
        psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
        return ReconstructionMetrics(mse=mse, mae=mae, psnr=psnr)

    def export_reconstruction(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        *,
        checkpoint_path: str | Path,
    ) -> ReconstructionExport:
        pixels, width, height = load_p6_ppm(source_path)
        if (width, height) != (self.width, self.height):
            raise ValueError("PPM dimensions do not match configured autoencoder")
        torch = _load_torch()
        with torch.no_grad():
            reconstructed, _, _ = self.reconstruct(pixels, deterministic=True)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        original = save_p6_ppm(pixels, width, height, destination / "original.ppm")
        rebuilt = save_p6_ppm(
            reconstructed,
            width,
            height,
            destination / "reconstructed.ppm",
        )
        checkpoint = self.save_checkpoint(checkpoint_path)
        return ReconstructionExport(
            original_path=str(original),
            reconstructed_path=str(rebuilt),
            checkpoint_path=str(checkpoint),
            metrics=self.reconstruction_metrics(pixels),
        )
