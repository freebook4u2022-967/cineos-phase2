"""Versioned, integrity-checked checkpoints for learned CINEOS temporal weights.

The bootstrap :class:`NativeTemporalModel` is useful for stabilising sequence and QC
contracts, but production deployment must not silently treat deterministic initial
weights as a trained temporal generator.  This module provides a framework-neutral,
content-addressed checkpoint contract for the recurrent temporal model so learned
weights can be trained externally, promoted through the native-model manifest, and
restored without changing film orchestration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cineos.native_image.tensor_model import LinearTensorLayer

from .temporal_model import NativeTemporalModel

TEMPORAL_MODEL_CHECKPOINT_SCHEMA = "cineos-native-temporal-model-checkpoint/1"


class TemporalModelCheckpointError(ValueError):
    """Raised when a learned temporal checkpoint is malformed or untrusted."""


def _layer_payload(layer: LinearTensorLayer) -> dict[str, Any]:
    return {
        "input_dim": layer.input_dim,
        "output_dim": layer.output_dim,
        "weights": list(layer.weights),
        "bias": list(layer.bias),
    }


def _restore_layer(payload: object, *, name: str) -> LinearTensorLayer:
    if not isinstance(payload, dict):
        raise TemporalModelCheckpointError(f"{name} must be an object")
    try:
        input_dim = int(payload["input_dim"])
        output_dim = int(payload["output_dim"])
        weights = [float(value) for value in payload["weights"]]
        bias = [float(value) for value in payload["bias"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise TemporalModelCheckpointError(f"malformed {name}") from exc
    if input_dim <= 0 or output_dim <= 0:
        raise TemporalModelCheckpointError(f"{name} dimensions must be positive")
    if len(weights) != input_dim * output_dim:
        raise TemporalModelCheckpointError(f"{name} weight shape mismatch")
    if len(bias) != output_dim:
        raise TemporalModelCheckpointError(f"{name} bias shape mismatch")
    return LinearTensorLayer(input_dim, output_dim, weights, bias)


@dataclass(frozen=True, slots=True)
class TemporalModelCheckpoint:
    """Immutable learned temporal-model artifact with explicit training provenance."""

    model: NativeTemporalModel
    training_steps: int
    training_run_id: str
    training_data_fingerprint: str
    schema: str = TEMPORAL_MODEL_CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TEMPORAL_MODEL_CHECKPOINT_SCHEMA:
            raise TemporalModelCheckpointError(
                f"unsupported temporal model checkpoint schema: {self.schema}"
            )
        if self.training_steps <= 0:
            raise TemporalModelCheckpointError(
                "production temporal checkpoint requires positive training_steps"
            )
        if not self.training_run_id.strip():
            raise TemporalModelCheckpointError("training_run_id must not be empty")
        digest = self.training_data_fingerprint.strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise TemporalModelCheckpointError(
                "training_data_fingerprint must be one lowercase SHA-256 digest"
            )
        object.__setattr__(self, "training_data_fingerprint", digest)
        self._validate_model_contract()

    def _validate_model_contract(self) -> None:
        model = self.model
        dims = (
            model.identity_dim,
            model.scene_dim,
            model.motion_dim,
            model.hidden_dim,
            model.latent_dim,
        )
        if min(dims) <= 0:
            raise TemporalModelCheckpointError("temporal model dimensions must be positive")
        recurrent_input = (
            model.identity_dim + model.scene_dim + model.motion_dim + model.hidden_dim
        )
        if (
            model.recurrent.input_dim != recurrent_input
            or model.recurrent.output_dim != model.hidden_dim
        ):
            raise TemporalModelCheckpointError(
                "recurrent layer shape is incompatible with temporal model dimensions"
            )
        if (
            model.decoder.input_dim != model.hidden_dim
            or model.decoder.output_dim != model.latent_dim
        ):
            raise TemporalModelCheckpointError(
                "temporal decoder shape is incompatible with temporal model dimensions"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "training": {
                "steps": self.training_steps,
                "run_id": self.training_run_id,
                "data_fingerprint": self.training_data_fingerprint,
            },
            "model": {
                "identity_dim": self.model.identity_dim,
                "scene_dim": self.model.scene_dim,
                "motion_dim": self.model.motion_dim,
                "hidden_dim": self.model.hidden_dim,
                "latent_dim": self.model.latent_dim,
                "recurrent": _layer_payload(self.model.recurrent),
                "decoder": _layer_payload(self.model.decoder),
            },
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
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

    @classmethod
    def capture(
        cls,
        model: NativeTemporalModel,
        *,
        training_steps: int,
        training_run_id: str,
        training_data_fingerprint: str,
    ) -> "TemporalModelCheckpoint":
        if not isinstance(model, NativeTemporalModel):
            raise TypeError("model must be a NativeTemporalModel")
        return cls(
            model=model,
            training_steps=training_steps,
            training_run_id=training_run_id,
            training_data_fingerprint=training_data_fingerprint,
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        verify_hash: bool = True,
    ) -> "TemporalModelCheckpoint":
        if not isinstance(payload, dict):
            raise TemporalModelCheckpointError("checkpoint root must be an object")
        if payload.get("schema") != TEMPORAL_MODEL_CHECKPOINT_SCHEMA:
            raise TemporalModelCheckpointError(
                f"unsupported temporal model checkpoint schema: {payload.get('schema')}"
            )
        model_payload = payload.get("model")
        training_payload = payload.get("training")
        if not isinstance(model_payload, dict) or not isinstance(training_payload, dict):
            raise TemporalModelCheckpointError(
                "checkpoint model and training provenance must be objects"
            )
        try:
            checkpoint = cls(
                model=NativeTemporalModel(
                    identity_dim=int(model_payload["identity_dim"]),
                    scene_dim=int(model_payload["scene_dim"]),
                    motion_dim=int(model_payload["motion_dim"]),
                    hidden_dim=int(model_payload["hidden_dim"]),
                    latent_dim=int(model_payload["latent_dim"]),
                    recurrent=_restore_layer(
                        model_payload.get("recurrent"), name="recurrent"
                    ),
                    decoder=_restore_layer(model_payload.get("decoder"), name="decoder"),
                ),
                training_steps=int(training_payload["steps"]),
                training_run_id=str(training_payload["run_id"]),
                training_data_fingerprint=str(training_payload["data_fingerprint"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, TemporalModelCheckpointError):
                raise
            raise TemporalModelCheckpointError("malformed temporal model checkpoint") from exc

        if verify_hash:
            expected = payload.get("sha256")
            if not isinstance(expected, str) or expected != checkpoint.sha256:
                raise TemporalModelCheckpointError(
                    "temporal model checkpoint hash mismatch"
                )
        return checkpoint

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify_hash: bool = True,
    ) -> "TemporalModelCheckpoint":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TemporalModelCheckpointError(
                "unable to read temporal model checkpoint"
            ) from exc
        if not isinstance(payload, dict):
            raise TemporalModelCheckpointError("checkpoint root must be an object")
        return cls.from_dict(payload, verify_hash=verify_hash)


__all__ = [
    "TEMPORAL_MODEL_CHECKPOINT_SCHEMA",
    "TemporalModelCheckpoint",
    "TemporalModelCheckpointError",
]
