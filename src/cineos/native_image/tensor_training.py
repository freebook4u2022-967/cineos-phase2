"""Tensor training primitives for the CINEOS native learned-model path."""

from dataclasses import dataclass

from .tensor_model import CineosTensorModel, LinearTensorLayer, Tensor


SUPPORTED_DEVICES = {"cpu", "cuda", "mps"}


def move_tensor(tensor: Tensor, device: str) -> Tensor:
    """Move a tensor to a logical device without binding a GPU framework yet."""
    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"unsupported tensor device: {device}")
    return Tensor(tensor.values, tensor.shape, device)


@dataclass(frozen=True, slots=True)
class FlowMatchingBatch:
    identity_features: tuple[Tensor, ...]
    scene_features: tuple[Tensor, ...]
    source_latents: tuple[Tensor, ...]
    target_latents: tuple[Tensor, ...]
    times: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.identity_features)
        if size == 0:
            raise ValueError("flow-matching batch must not be empty")
        if not all(
            len(items) == size
            for items in (
                self.scene_features,
                self.source_latents,
                self.target_latents,
                self.times,
            )
        ):
            raise ValueError("flow-matching batch fields must have equal length")
        if any(not 0.0 <= time <= 1.0 for time in self.times):
            raise ValueError("flow-matching times must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class FlowMatchingResult:
    loss: float
    predicted_velocity: Tensor
    target_velocity: Tensor
    interpolated_latent: Tensor


def interpolate_latents(source: Tensor, target: Tensor, time: float) -> Tensor:
    if source.shape != target.shape:
        raise ValueError("source and target latent shapes must match")
    if source.device != target.device:
        raise ValueError("source and target latents must share a device")
    if not 0.0 <= time <= 1.0:
        raise ValueError("flow time must be within [0, 1]")
    values = tuple(
        ((1.0 - time) * start) + (time * end)
        for start, end in zip(source.values, target.values)
    )
    return Tensor(values, source.shape, source.device)


def target_velocity(source: Tensor, target: Tensor) -> Tensor:
    if source.shape != target.shape:
        raise ValueError("source and target latent shapes must match")
    return Tensor(
        tuple(end - start for start, end in zip(source.values, target.values)),
        source.shape,
        source.device,
    )


def flow_matching_objective(
    model: CineosTensorModel,
    identity_features: Tensor,
    scene_features: Tensor,
    source: Tensor,
    target: Tensor,
    time: float,
) -> FlowMatchingResult:
    """Compute the first deterministic rectified-flow-style training objective."""
    interpolated = interpolate_latents(source, target, time)
    conditioning = model.forward(identity_features, scene_features)
    if conditioning.shape != source.shape:
        raise ValueError("model latent shape must match flow latent shape")

    predicted = Tensor(
        tuple(
            condition + (0.1 * latent) + (0.05 * ((2.0 * time) - 1.0))
            for condition, latent in zip(conditioning.values, interpolated.values)
        ),
        source.shape,
        source.device,
    )
    velocity = target_velocity(source, target)
    return FlowMatchingResult(
        loss=predicted.mse(velocity),
        predicted_velocity=predicted,
        target_velocity=velocity,
        interpolated_latent=interpolated,
    )


@dataclass(slots=True)
class TensorSGDOptimizer:
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def step_linear(self, layer: LinearTensorLayer, gradient_scale: float) -> None:
        """Apply a deterministic surrogate gradient update to one tensor layer."""
        for index in range(len(layer.weights)):
            layer.weights[index] -= self.learning_rate * gradient_scale
        for index in range(len(layer.bias)):
            layer.bias[index] -= self.learning_rate * gradient_scale


@dataclass(slots=True)
class TensorBatchTrainer:
    model: CineosTensorModel
    optimizer: TensorSGDOptimizer
    step: int = 0

    def train_batch(self, batch: FlowMatchingBatch) -> float:
        losses = []
        for identity, scene, source, target, time in zip(
            batch.identity_features,
            batch.scene_features,
            batch.source_latents,
            batch.target_latents,
            batch.times,
        ):
            result = flow_matching_objective(
                self.model,
                identity,
                scene,
                source,
                target,
                time,
            )
            losses.append(result.loss)

        mean_loss = sum(losses) / len(losses)
        gradient_scale = min(1.0, mean_loss)
        self.optimizer.step_linear(self.model.identity_encoder, gradient_scale)
        self.optimizer.step_linear(self.model.scene_encoder, gradient_scale)
        self.optimizer.step_linear(self.model.latent_network, gradient_scale)
        self.step += 1
        return mean_loss
