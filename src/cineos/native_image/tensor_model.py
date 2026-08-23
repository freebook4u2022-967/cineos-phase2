"""Dependency-light tensor foundation for CINEOS native learned models.

The implementation deliberately uses an owned CPU tensor abstraction first so
CI remains portable. GPU frameworks can implement the same contracts later
without changing dataset, checkpoint, Atlas, QC, or short-drama orchestration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Tensor:
    values: tuple[float, ...]
    shape: tuple[int, ...]
    device: str = "cpu"

    def __post_init__(self) -> None:
        size = math.prod(self.shape)
        if size != len(self.values):
            raise ValueError(f"tensor shape expects {size} values, got {len(self.values)}")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be positive")

    def mse(self, target: Tensor) -> float:
        if self.shape != target.shape:
            raise ValueError("MSE tensors must have identical shapes")
        return sum((a - b) ** 2 for a, b in zip(self.values, target.values)) / len(
            self.values
        )


@dataclass(slots=True)
class LinearTensorLayer:
    input_dim: int
    output_dim: int
    weights: list[float]
    bias: list[float]

    @classmethod
    def initialized(cls, input_dim: int, output_dim: int) -> LinearTensorLayer:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("linear dimensions must be positive")
        weights = [
            (((row + 1) * (column + 3)) % 17 - 8) / 100.0
            for row in range(output_dim)
            for column in range(input_dim)
        ]
        return cls(input_dim, output_dim, weights, [0.0] * output_dim)

    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.shape != (self.input_dim,):
            raise ValueError("linear layer received incompatible tensor shape")
        output = []
        for row in range(self.output_dim):
            offset = row * self.input_dim
            value = self.bias[row]
            for column, item in enumerate(tensor.values):
                value += self.weights[offset + column] * item
            output.append(math.tanh(value))
        return Tensor(tuple(output), (self.output_dim,), tensor.device)


class TensorIdentityEncoder(Protocol):
    def encode_identity_tensor(self, features: Tensor) -> Tensor: ...


class TensorSceneEncoder(Protocol):
    def encode_scene_tensor(self, features: Tensor) -> Tensor: ...


class TensorLatentNetwork(Protocol):
    def predict_latent_tensor(self, identity: Tensor, scene: Tensor) -> Tensor: ...


@dataclass(slots=True)
class CineosTensorModel:
    """First tensor-backed CINEOS model with separate identity and scene paths."""

    identity_encoder: LinearTensorLayer
    scene_encoder: LinearTensorLayer
    latent_network: LinearTensorLayer

    @classmethod
    def initialized(
        cls,
        *,
        feature_dim: int = 8,
        embedding_dim: int = 8,
        latent_dim: int = 16,
    ) -> CineosTensorModel:
        return cls(
            identity_encoder=LinearTensorLayer.initialized(feature_dim, embedding_dim),
            scene_encoder=LinearTensorLayer.initialized(feature_dim, embedding_dim),
            latent_network=LinearTensorLayer.initialized(embedding_dim * 2, latent_dim),
        )

    def encode_identity_tensor(self, features: Tensor) -> Tensor:
        return self.identity_encoder.forward(features)

    def encode_scene_tensor(self, features: Tensor) -> Tensor:
        return self.scene_encoder.forward(features)

    def predict_latent_tensor(self, identity: Tensor, scene: Tensor) -> Tensor:
        if identity.device != scene.device:
            raise ValueError("identity and scene tensors must share a device")
        combined = Tensor(
            identity.values + scene.values,
            (len(identity.values) + len(scene.values),),
            identity.device,
        )
        return self.latent_network.forward(combined)

    def forward(self, identity_features: Tensor, scene_features: Tensor) -> Tensor:
        identity = self.encode_identity_tensor(identity_features)
        scene = self.encode_scene_tensor(scene_features)
        return self.predict_latent_tensor(identity, scene)
