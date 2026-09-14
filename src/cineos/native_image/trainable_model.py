"""Minimal trainable CINEOS-native model prototype.

This module intentionally avoids claiming production-grade generative quality. It
provides the first model parameters in the native path that are updated by a real
loss/optimizer loop and can be checkpointed and resumed deterministically.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .training import NativeDatasetManifest, NativeTrainingSample


@dataclass(slots=True)
class TrainableParameterSet:
    """Small differentiable parameter set for the first CINEOS training loop."""

    identity_scale: float = 0.6
    scene_scale: float = 0.4
    bias: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    step: int
    sample_id: str
    loss: float
    prediction: float
    target: float


@dataclass(slots=True)
class SGDOptimizer:
    learning_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def step(
        self,
        parameters: TrainableParameterSet,
        *,
        grad_identity: float,
        grad_scene: float,
        grad_bias: float,
    ) -> None:
        parameters.identity_scale -= self.learning_rate * grad_identity
        parameters.scene_scale -= self.learning_rate * grad_scene
        parameters.bias -= self.learning_rate * grad_bias


@dataclass(slots=True)
class NativeTrainableModel:
    """A tiny differentiable model proving CINEOS-owned parameter learning."""

    parameters: TrainableParameterSet = field(default_factory=TrainableParameterSet)

    @staticmethod
    def _stable_feature(text: str) -> float:
        if not text:
            return 0.0
        total = sum((index + 1) * ord(char) for index, char in enumerate(text))
        return ((total % 2001) / 1000.0) - 1.0

    def features(self, sample: NativeTrainingSample) -> tuple[float, float]:
        identity_parts = sample.character_reference_paths + sample.identity_tags
        identity_text = "|".join(identity_parts)
        scene_text = "|".join(
            (sample.caption, sample.scene_description, *sample.continuity_tags)
        )
        return self._stable_feature(identity_text), self._stable_feature(scene_text)

    def target(self, sample: NativeTrainingSample) -> float:
        explicit = sample.metadata.get("training_target")
        if explicit is not None:
            value = float(explicit)
            if not -1.0 <= value <= 1.0:
                raise ValueError("training_target must be between -1 and 1")
            return value
        return self._stable_feature(sample.image_path)

    def predict_features(self, identity_feature: float, scene_feature: float) -> float:
        raw = (
            self.parameters.identity_scale * identity_feature
            + self.parameters.scene_scale * scene_feature
            + self.parameters.bias
        )
        return math.tanh(raw)

    def predict(self, sample: NativeTrainingSample) -> float:
        identity_feature, scene_feature = self.features(sample)
        return self.predict_features(identity_feature, scene_feature)


@dataclass(slots=True)
class NativeTrainingLoop:
    model: NativeTrainableModel = field(default_factory=NativeTrainableModel)
    optimizer: SGDOptimizer = field(default_factory=SGDOptimizer)
    step: int = 0
    history: list[TrainingStepResult] = field(default_factory=list)

    def train_sample(self, sample: NativeTrainingSample) -> TrainingStepResult:
        identity_feature, scene_feature = self.model.features(sample)
        target = self.model.target(sample)
        prediction = self.model.predict_features(identity_feature, scene_feature)
        error = prediction - target
        loss = error * error

        tanh_gradient = 1.0 - (prediction * prediction)
        common = 2.0 * error * tanh_gradient
        self.optimizer.step(
            self.model.parameters,
            grad_identity=common * identity_feature,
            grad_scene=common * scene_feature,
            grad_bias=common,
        )
        self.step += 1
        result = TrainingStepResult(
            step=self.step,
            sample_id=sample.sample_id,
            loss=loss,
            prediction=prediction,
            target=target,
        )
        self.history.append(result)
        return result

    def train_epoch(
        self, dataset: NativeDatasetManifest
    ) -> tuple[TrainingStepResult, ...]:
        if not dataset.samples:
            raise ValueError("cannot train on an empty dataset")
        return tuple(self.train_sample(sample) for sample in dataset.samples)

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema": "cineos-native-trainable-checkpoint/0.1",
            "step": self.step,
            "parameters": asdict(self.model.parameters),
            "optimizer": {"learning_rate": self.optimizer.learning_rate},
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> NativeTrainingLoop:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "cineos-native-trainable-checkpoint/0.1":
            raise ValueError("unsupported trainable checkpoint schema")
        parameters = TrainableParameterSet(**payload["parameters"])
        optimizer = SGDOptimizer(**payload["optimizer"])
        return cls(
            model=NativeTrainableModel(parameters=parameters),
            optimizer=optimizer,
            step=int(payload["step"]),
        )
