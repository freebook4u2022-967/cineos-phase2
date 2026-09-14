"""Identity consistency loss utilities for CINEOS neural training."""

from __future__ import annotations

from dataclasses import dataclass

from .neural_backend import _load_torch


@dataclass(frozen=True, slots=True)
class IdentityLossConfig:
    weight: float = 1.0
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("identity loss weight must be non-negative")
        if not 0.0 <= self.margin < 1.0:
            raise ValueError("identity loss margin must be within [0, 1)")


class TorchIdentityConsistencyLoss:
    """Cosine-distance loss that pulls generated identity toward a character anchor."""

    def __init__(self, config: IdentityLossConfig | None = None) -> None:
        self.config = config or IdentityLossConfig()
        self.torch = _load_torch()

    def __call__(self, predicted, anchor):
        torch = self.torch
        if predicted.shape != anchor.shape:
            raise ValueError(
                "predicted and anchor identity embeddings must have equal shapes"
            )
        predicted = torch.nn.functional.normalize(predicted, dim=-1)
        anchor = torch.nn.functional.normalize(anchor, dim=-1)
        similarity = (predicted * anchor).sum(dim=-1)
        distance = 1.0 - similarity
        if self.config.margin > 0:
            distance = torch.clamp(distance - self.config.margin, min=0.0)
        return distance.mean() * self.config.weight


def combined_training_loss(
    reconstruction_loss,
    flow_loss,
    identity_loss,
    *,
    reconstruction_weight: float = 1.0,
    flow_weight: float = 1.0,
):
    if reconstruction_weight < 0 or flow_weight < 0:
        raise ValueError("training loss weights must be non-negative")
    return (
        reconstruction_weight * reconstruction_loss
        + flow_weight * flow_loss
        + identity_loss
    )
