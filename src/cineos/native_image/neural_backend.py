"""Optional real neural backend for CINEOS native training.

PyTorch is loaded lazily so the base package and CI remain lightweight. Install
CINEOS with the `neural` extra to enable actual autograd, GPU execution, decoded
image ingestion, and checkpoint-backed flow-matching training.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _load_torch() -> Any:
    if not torch_available():
        raise RuntimeError(
            "PyTorch is required for the neural backend; install cineos[neural]"
        )
    import torch

    return torch


@dataclass(frozen=True, slots=True)
class NeuralModelConfig:
    feature_dim: int = 64
    embedding_dim: int = 128
    latent_dim: int = 256
    hidden_dim: int = 512
    image_size: int = 32

    def __post_init__(self) -> None:
        if (
            min(
                self.feature_dim,
                self.embedding_dim,
                self.latent_dim,
                self.hidden_dim,
                self.image_size,
            )
            <= 0
        ):
            raise ValueError("neural model dimensions must be positive")


class TorchCineosFlowModel:
    """GPU-capable identity/scene conditioned flow network using real autograd."""

    def __init__(
        self,
        config: NeuralModelConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        torch = _load_torch()
        nn = torch.nn
        self.config = config or NeuralModelConfig()
        self.device = torch.device(device)
        self.identity_encoder = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
        ).to(self.device)
        self.scene_encoder = nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
        ).to(self.device)
        flow_input_dim = self.config.latent_dim + (2 * self.config.embedding_dim) + 1
        self.flow_network = nn.Sequential(
            nn.Linear(flow_input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
        ).to(self.device)

    def parameters(self):
        for module in (
            self.identity_encoder,
            self.scene_encoder,
            self.flow_network,
        ):
            yield from module.parameters()

    def predict_velocity(self, identity_features, scene_features, latent, time):
        torch = _load_torch()
        identity = self.identity_encoder(identity_features.to(self.device))
        scene = self.scene_encoder(scene_features.to(self.device))
        latent = latent.to(self.device)
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        combined = torch.cat((latent, identity, scene, time.to(self.device)), dim=-1)
        return self.flow_network(combined)

    def state_dict(self) -> dict[str, Any]:
        return {
            "identity_encoder": self.identity_encoder.state_dict(),
            "scene_encoder": self.scene_encoder.state_dict(),
            "flow_network": self.flow_network.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.identity_encoder.load_state_dict(state["identity_encoder"])
        self.scene_encoder.load_state_dict(state["scene_encoder"])
        self.flow_network.load_state_dict(state["flow_network"])


@dataclass(slots=True)
class TorchFlowTrainingRunner:
    model: TorchCineosFlowModel
    learning_rate: float = 1e-4
    step: int = 0

    def __post_init__(self) -> None:
        torch = _load_torch()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def train_batch(
        self,
        identity_features,
        scene_features,
        source_latents,
        target_latents,
        times,
    ) -> float:
        torch = _load_torch()
        source = source_latents.to(self.model.device)
        target = target_latents.to(self.model.device)
        times = times.to(self.model.device)
        if times.ndim == 1:
            interpolation_time = times.unsqueeze(-1)
        else:
            interpolation_time = times
        interpolated = ((1.0 - interpolation_time) * source) + (
            interpolation_time * target
        )
        target_velocity = target - source
        predicted_velocity = self.model.predict_velocity(
            identity_features,
            scene_features,
            interpolated,
            times,
        )
        loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.step += 1
        return float(loss.detach().cpu().item())

    def save_checkpoint(self, path: str | Path) -> Path:
        torch = _load_torch()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "cineos-torch-flow-checkpoint/0.1",
                "step": self.step,
                "learning_rate": self.learning_rate,
                "config": asdict(self.model.config),
                "model": self.model.state_dict(),
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
    ) -> Self:
        torch = _load_torch()
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        if payload.get("schema") != "cineos-torch-flow-checkpoint/0.1":
            raise ValueError("unsupported torch flow checkpoint schema")
        model = TorchCineosFlowModel(
            NeuralModelConfig(**payload["config"]),
            device=device,
        )
        model.load_state_dict(payload["model"])
        runner = cls(model=model, learning_rate=float(payload["learning_rate"]))
        runner.optimizer.load_state_dict(payload["optimizer"])
        runner.step = int(payload["step"])
        return runner
