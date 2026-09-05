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


@dataclass(slots=True)
class LinearGradients:
    """Explicit gradients for one owned linear+tanh layer."""

    weights: list[float]
    bias: list[float]

    @classmethod
    def zeros(cls, layer: LinearTensorLayer) -> "LinearGradients":
        return cls([0.0] * len(layer.weights), [0.0] * len(layer.bias))

    def add_(self, other: "LinearGradients") -> None:
        if len(self.weights) != len(other.weights) or len(self.bias) != len(other.bias):
            raise ValueError("gradient shapes must match")
        for index, value in enumerate(other.weights):
            self.weights[index] += value
        for index, value in enumerate(other.bias):
            self.bias[index] += value

    def scale_(self, factor: float) -> None:
        for index in range(len(self.weights)):
            self.weights[index] *= factor
        for index in range(len(self.bias)):
            self.bias[index] *= factor


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


def _backprop_linear_tanh(
    layer: LinearTensorLayer,
    inputs: tuple[float, ...],
    outputs: tuple[float, ...],
    output_gradients: tuple[float, ...],
) -> tuple[LinearGradients, tuple[float, ...]]:
    """Backpropagate through the owned ``tanh(Wx+b)`` primitive."""
    if len(inputs) != layer.input_dim:
        raise ValueError("backprop input dimension does not match layer")
    if len(outputs) != layer.output_dim or len(output_gradients) != layer.output_dim:
        raise ValueError("backprop output dimension does not match layer")

    gradients = LinearGradients.zeros(layer)
    input_gradients = [0.0] * layer.input_dim
    for row in range(layer.output_dim):
        local = output_gradients[row] * (1.0 - (outputs[row] ** 2))
        gradients.bias[row] = local
        offset = row * layer.input_dim
        for column, input_value in enumerate(inputs):
            weight_index = offset + column
            gradients.weights[weight_index] = local * input_value
            input_gradients[column] += layer.weights[weight_index] * local
    return gradients, tuple(input_gradients)


def flow_matching_gradients(
    model: CineosTensorModel,
    identity_features: Tensor,
    scene_features: Tensor,
    source: Tensor,
    target: Tensor,
    time: float,
) -> tuple[FlowMatchingResult, LinearGradients, LinearGradients, LinearGradients]:
    """Return the objective and exact analytic gradients for all trainable layers."""
    result = flow_matching_objective(
        model,
        identity_features,
        scene_features,
        source,
        target,
        time,
    )
    identity_embedding = model.encode_identity_tensor(identity_features)
    scene_embedding = model.encode_scene_tensor(scene_features)
    conditioning = model.predict_latent_tensor(identity_embedding, scene_embedding)

    output_scale = 2.0 / len(result.predicted_velocity.values)
    conditioning_gradients = tuple(
        output_scale * (predicted - expected)
        for predicted, expected in zip(
            result.predicted_velocity.values,
            result.target_velocity.values,
        )
    )
    combined = identity_embedding.values + scene_embedding.values
    latent_gradients, combined_gradients = _backprop_linear_tanh(
        model.latent_network,
        combined,
        conditioning.values,
        conditioning_gradients,
    )

    identity_dim = len(identity_embedding.values)
    identity_output_gradients = combined_gradients[:identity_dim]
    scene_output_gradients = combined_gradients[identity_dim:]
    identity_gradients, _ = _backprop_linear_tanh(
        model.identity_encoder,
        identity_features.values,
        identity_embedding.values,
        identity_output_gradients,
    )
    scene_gradients, _ = _backprop_linear_tanh(
        model.scene_encoder,
        scene_features.values,
        scene_embedding.values,
        scene_output_gradients,
    )
    return result, identity_gradients, scene_gradients, latent_gradients


@dataclass(slots=True)
class TensorSGDOptimizer:
    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def step(self, layer: LinearTensorLayer, gradients: LinearGradients) -> None:
        if len(layer.weights) != len(gradients.weights) or len(layer.bias) != len(
            gradients.bias
        ):
            raise ValueError("optimizer gradient shape does not match layer")
        for index, gradient in enumerate(gradients.weights):
            layer.weights[index] -= self.learning_rate * gradient
        for index, gradient in enumerate(gradients.bias):
            layer.bias[index] -= self.learning_rate * gradient

    def step_linear(self, layer: LinearTensorLayer, gradient_scale: float) -> None:
        """Backward-compatible scalar update retained for older callers."""
        gradients = LinearGradients(
            [gradient_scale] * len(layer.weights),
            [gradient_scale] * len(layer.bias),
        )
        self.step(layer, gradients)


@dataclass(slots=True)
class TensorBatchTrainer:
    model: CineosTensorModel
    optimizer: TensorSGDOptimizer
    step: int = 0

    def train_batch(self, batch: FlowMatchingBatch) -> float:
        identity_gradients = LinearGradients.zeros(self.model.identity_encoder)
        scene_gradients = LinearGradients.zeros(self.model.scene_encoder)
        latent_gradients = LinearGradients.zeros(self.model.latent_network)
        losses = []

        for identity, scene, source, target, time in zip(
            batch.identity_features,
            batch.scene_features,
            batch.source_latents,
            batch.target_latents,
            batch.times,
        ):
            result, identity_gradient, scene_gradient, latent_gradient = (
                flow_matching_gradients(
                    self.model,
                    identity,
                    scene,
                    source,
                    target,
                    time,
                )
            )
            losses.append(result.loss)
            identity_gradients.add_(identity_gradient)
            scene_gradients.add_(scene_gradient)
            latent_gradients.add_(latent_gradient)

        batch_scale = 1.0 / len(losses)
        identity_gradients.scale_(batch_scale)
        scene_gradients.scale_(batch_scale)
        latent_gradients.scale_(batch_scale)
        self.optimizer.step(self.model.identity_encoder, identity_gradients)
        self.optimizer.step(self.model.scene_encoder, scene_gradients)
        self.optimizer.step(self.model.latent_network, latent_gradients)
        self.step += 1
        return sum(losses) / len(losses)
