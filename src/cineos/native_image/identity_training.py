"""Identity-aware training integration for CINEOS neural models."""

from __future__ import annotations

from dataclasses import dataclass

from .identity_loss import IdentityLossConfig, TorchIdentityConsistencyLoss
from .neural_backend import _load_torch


@dataclass(frozen=True, slots=True)
class IdentityAwareLosses:
    total_loss: float
    flow_loss: float
    identity_loss: float


class TorchIdentityProjection:
    """Trainable projection from generated latent/velocity space to identity space."""

    def __init__(self, input_dim: int, identity_dim: int, device: str = "cpu") -> None:
        if input_dim < 1 or identity_dim < 1:
            raise ValueError("projection dimensions must be positive")
        torch = _load_torch()
        self.module = torch.nn.Linear(input_dim, identity_dim).to(torch.device(device))


class IdentityAwareTrainingStep:
    """Apply flow loss and identity-anchor loss in one optimizer step."""

    def __init__(
        self,
        model,
        identity_projection: TorchIdentityProjection,
        optimizer,
        *,
        identity_loss_config: IdentityLossConfig | None = None,
        flow_weight: float = 1.0,
    ) -> None:
        if flow_weight < 0:
            raise ValueError("flow_weight must be non-negative")
        self.torch = _load_torch()
        self.model = model
        self.identity_projection = identity_projection
        self.optimizer = optimizer
        self.identity_loss = TorchIdentityConsistencyLoss(identity_loss_config)
        self.flow_weight = flow_weight

    def train_batch(
        self,
        identity_features,
        scene_features,
        source_latent,
        target_latent,
        anchor_identity,
    ) -> IdentityAwareLosses:
        torch = self.torch
        time = torch.full(
            (identity_features.shape[0], 1),
            0.5,
            device=identity_features.device,
        )
        interpolated = 0.5 * source_latent + 0.5 * target_latent
        target_velocity = target_latent - source_latent

        self.optimizer.zero_grad(set_to_none=True)
        predicted_velocity = self.model(
            identity_features,
            scene_features,
            interpolated,
            time,
        )
        flow_loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        predicted_identity = self.identity_projection.module(predicted_velocity)
        identity_loss = self.identity_loss(predicted_identity, anchor_identity)
        total = self.flow_weight * flow_loss + identity_loss
        total.backward()
        self.optimizer.step()

        return IdentityAwareLosses(
            total_loss=float(total.detach().cpu()),
            flow_loss=float(flow_loss.detach().cpu()),
            identity_loss=float(identity_loss.detach().cpu()),
        )
