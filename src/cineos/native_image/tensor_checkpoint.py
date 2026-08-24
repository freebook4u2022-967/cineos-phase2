"""Versioned, integrity-checked checkpoints for the tensor training path.

The checkpoint contract is intentionally framework-neutral. It persists the owned
CINEOS tensor model, optimizer state and trainer step atomically, validates shape
metadata before restore, and rejects tampered or incompatible payloads.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tensor_model import CineosTensorModel, LinearTensorLayer
from .tensor_training import TensorBatchTrainer, TensorSGDOptimizer

TENSOR_CHECKPOINT_SCHEMA = "cineos-tensor-training-checkpoint/1"


class TensorCheckpointError(ValueError):
    """Raised when a tensor checkpoint is malformed, incompatible, or corrupted."""


def _layer_payload(layer: LinearTensorLayer) -> dict[str, Any]:
    return {
        "input_dim": layer.input_dim,
        "output_dim": layer.output_dim,
        "weights": list(layer.weights),
        "bias": list(layer.bias),
    }


def _restore_layer(payload: object, *, name: str) -> LinearTensorLayer:
    if not isinstance(payload, dict):
        raise TensorCheckpointError(f"{name} must be an object")
    try:
        input_dim = int(payload["input_dim"])
        output_dim = int(payload["output_dim"])
        weights = [float(value) for value in payload["weights"]]
        bias = [float(value) for value in payload["bias"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise TensorCheckpointError(f"malformed {name}") from exc
    if input_dim <= 0 or output_dim <= 0:
        raise TensorCheckpointError(f"{name} dimensions must be positive")
    if len(weights) != input_dim * output_dim:
        raise TensorCheckpointError(f"{name} weight shape mismatch")
    if len(bias) != output_dim:
        raise TensorCheckpointError(f"{name} bias shape mismatch")
    return LinearTensorLayer(input_dim, output_dim, weights, bias)


@dataclass(frozen=True, slots=True)
class TensorTrainingCheckpoint:
    """Serializable tensor-training state with deterministic integrity metadata."""

    model: CineosTensorModel
    optimizer_learning_rate: float
    trainer_step: int
    schema: str = TENSOR_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TENSOR_CHECKPOINT_SCHEMA:
            raise TensorCheckpointError(f"unsupported checkpoint schema: {self.schema}")
        if self.optimizer_learning_rate <= 0:
            raise TensorCheckpointError("optimizer learning rate must be positive")
        if self.trainer_step < 0:
            raise TensorCheckpointError("trainer step must be non-negative")

    @classmethod
    def capture(cls, trainer: TensorBatchTrainer) -> "TensorTrainingCheckpoint":
        if not isinstance(trainer, TensorBatchTrainer):
            raise TypeError("trainer must be a TensorBatchTrainer")
        return cls(
            model=trainer.model,
            optimizer_learning_rate=trainer.optimizer.learning_rate,
            trainer_step=trainer.step,
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trainer_step": self.trainer_step,
            "optimizer": {"learning_rate": self.optimizer_learning_rate},
            "model": {
                "identity_encoder": _layer_payload(self.model.identity_encoder),
                "scene_encoder": _layer_payload(self.model.scene_encoder),
                "latent_network": _layer_payload(self.model.latent_network),
            },
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["sha256"] = self.sha256
        return payload

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    def restore_trainer(self) -> TensorBatchTrainer:
        return TensorBatchTrainer(
            model=self.model,
            optimizer=TensorSGDOptimizer(self.optimizer_learning_rate),
            step=self.trainer_step,
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        verify_hash: bool = True,
    ) -> "TensorTrainingCheckpoint":
        if not isinstance(payload, dict):
            raise TensorCheckpointError("checkpoint root must be an object")
        if payload.get("schema") != TENSOR_CHECKPOINT_SCHEMA:
            raise TensorCheckpointError(
                f"unsupported checkpoint schema: {payload.get('schema')}"
            )
        model_payload = payload.get("model")
        optimizer_payload = payload.get("optimizer")
        if not isinstance(model_payload, dict) or not isinstance(optimizer_payload, dict):
            raise TensorCheckpointError("checkpoint model and optimizer must be objects")
        try:
            checkpoint = cls(
                model=CineosTensorModel(
                    identity_encoder=_restore_layer(
                        model_payload.get("identity_encoder"), name="identity_encoder"
                    ),
                    scene_encoder=_restore_layer(
                        model_payload.get("scene_encoder"), name="scene_encoder"
                    ),
                    latent_network=_restore_layer(
                        model_payload.get("latent_network"), name="latent_network"
                    ),
                ),
                optimizer_learning_rate=float(optimizer_payload["learning_rate"]),
                trainer_step=int(payload["trainer_step"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, TensorCheckpointError):
                raise
            raise TensorCheckpointError("malformed tensor checkpoint") from exc

        if verify_hash:
            expected = payload.get("sha256")
            if not isinstance(expected, str) or expected != checkpoint.sha256:
                raise TensorCheckpointError("tensor checkpoint hash mismatch")
        return checkpoint

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify_hash: bool = True,
    ) -> "TensorTrainingCheckpoint":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TensorCheckpointError("unable to read tensor checkpoint") from exc
        if not isinstance(payload, dict):
            raise TensorCheckpointError("checkpoint root must be an object")
        return cls.from_dict(payload, verify_hash=verify_hash)
