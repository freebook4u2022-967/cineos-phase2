"""Trainable neural encoders for CINEOS real-data experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .neural_backend import NeuralModelConfig, _load_torch


def _bytes_to_fixed_tensor(payload: bytes, dimensions: int, *, device: Any):
    torch = _load_torch()
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    if not payload:
        payload = b"\x00"
    values = [payload[index % len(payload)] / 255.0 for index in range(dimensions)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _text_to_fixed_tensor(text: str, dimensions: int, *, device: Any):
    return _bytes_to_fixed_tensor(text.encode("utf-8"), dimensions, device=device)


@dataclass(slots=True)
class TorchImageLatentEncoder:
    """Small VAE-style image encoder producing mean/logvar latent parameters."""

    config: NeuralModelConfig
    device: str = "cpu"

    def __post_init__(self) -> None:
        torch = _load_torch()
        nn = torch.nn
        self.device_object = torch.device(self.device)
        self.input_dim = max(64, self.config.feature_dim * 4)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_dim),
            nn.SiLU(),
        ).to(self.device_object)
        self.mean_head = nn.Linear(
            self.config.hidden_dim,
            self.config.latent_dim,
        ).to(self.device_object)
        self.logvar_head = nn.Linear(
            self.config.hidden_dim,
            self.config.latent_dim,
        ).to(self.device_object)

    def encode_bytes(self, payload: bytes):
        features = _bytes_to_fixed_tensor(
            payload,
            self.input_dim,
            device=self.device_object,
        )
        hidden = self.backbone(features)
        return self.mean_head(hidden), self.logvar_head(hidden)

    def encode_file(self, path: str | Path):
        return self.encode_bytes(Path(path).read_bytes())

    def sample(self, mean, logvar, *, deterministic: bool = False):
        torch = _load_torch()
        if deterministic:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + (torch.randn_like(std) * std)


@dataclass(slots=True)
class TorchCharacterReferenceEncoder:
    """Aggregate one or more approved character references into identity features."""

    config: NeuralModelConfig
    device: str = "cpu"

    def __post_init__(self) -> None:
        torch = _load_torch()
        nn = torch.nn
        self.device_object = torch.device(self.device)
        self.input_dim = max(64, self.config.feature_dim * 4)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.feature_dim),
        ).to(self.device_object)

    def encode_files(self, paths: tuple[str | Path, ...]):
        torch = _load_torch()
        if not paths:
            raise ValueError("character reference encoder requires references")
        encoded = []
        for path in paths:
            features = _bytes_to_fixed_tensor(
                Path(path).read_bytes(),
                self.input_dim,
                device=self.device_object,
            )
            encoded.append(self.network(features))
        return torch.stack(encoded, dim=0).mean(dim=0)


@dataclass(slots=True)
class TorchSceneTextEncoder:
    """Trainable text/scene encoder used by the native flow model."""

    config: NeuralModelConfig
    device: str = "cpu"

    def __post_init__(self) -> None:
        torch = _load_torch()
        nn = torch.nn
        self.device_object = torch.device(self.device)
        self.input_dim = max(64, self.config.feature_dim * 4)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.feature_dim),
        ).to(self.device_object)

    def encode(self, caption: str, scene_description: str, continuity: tuple[str, ...]):
        payload = "|".join((caption, scene_description, *continuity))
        features = _text_to_fixed_tensor(
            payload,
            self.input_dim,
            device=self.device_object,
        )
        return self.network(features)
