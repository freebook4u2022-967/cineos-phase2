import hashlib

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
        with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
            TorchCineosFlowModel()


def _trained_runner():
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
    return config, runner, before, after, loss


def test_torch_flow_training_updates_parameters_when_available(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")

    import torch

    config, runner, before, after, loss = _trained_runner()

    assert loss >= 0.0
    assert runner.step == 1
    assert not torch.equal(before, after)

    checkpoint = runner.save_checkpoint(tmp_path / "flow.pt")
    sidecar = checkpoint.with_name(checkpoint.name + ".sha256")
    assert sidecar.is_file()
    assert (
        sidecar.read_text(encoding="ascii").strip()
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )

    resumed = TorchFlowTrainingRunner.load_checkpoint(checkpoint)
    assert resumed.step == 1
    assert resumed.model.config == config


def test_torch_flow_checkpoint_rejects_tampered_bytes(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")

    _, runner, _, _, _ = _trained_runner()
    checkpoint = runner.save_checkpoint(tmp_path / "flow.pt")

    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        TorchFlowTrainingRunner.load_checkpoint(checkpoint)


def test_torch_flow_checkpoint_requires_integrity_sidecar(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")

    _, runner, _, _, _ = _trained_runner()
    checkpoint = runner.save_checkpoint(tmp_path / "flow.pt")
    checkpoint.with_name(checkpoint.name + ".sha256").unlink()

    with pytest.raises(ValueError, match="integrity sidecar is missing"):
        TorchFlowTrainingRunner.load_checkpoint(checkpoint)
