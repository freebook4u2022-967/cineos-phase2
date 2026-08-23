import pytest

from cineos.native_image.neural_backend import (
    NeuralModelConfig,
    TorchCineosFlowModel,
    TorchFlowTrainingRunner,
    torch_available,
)


def test_neural_backend_dependency_contract():
    if torch_available():
        model = TorchCineosFlowModel(
            NeuralModelConfig(
                feature_dim=4,
                embedding_dim=8,
                latent_dim=6,
                hidden_dim=12,
            )
        )
        assert model.device.type == "cpu"
    else:
        with pytest.raises(RuntimeError, match="cineos\[neural\]"):
            TorchCineosFlowModel()


def test_torch_flow_training_updates_parameters_when_available(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")

    import torch

    config = NeuralModelConfig(
        feature_dim=4,
        embedding_dim=8,
        latent_dim=6,
        hidden_dim=12,
    )
    model = TorchCineosFlowModel(config)
    runner = TorchFlowTrainingRunner(model, learning_rate=1e-3)
    before = next(model.parameters()).detach().clone()
    loss = runner.train_batch(
        torch.randn(2, 4),
        torch.randn(2, 4),
        torch.randn(2, 6),
        torch.randn(2, 6),
        torch.tensor([0.25, 0.75]),
    )
    after = next(model.parameters()).detach().clone()

    assert loss >= 0.0
    assert runner.step == 1
    assert not torch.equal(before, after)

    checkpoint = runner.save_checkpoint(tmp_path / "flow.pt")
    resumed = TorchFlowTrainingRunner.load_checkpoint(checkpoint)
    assert resumed.step == 1
    assert resumed.model.config == config
