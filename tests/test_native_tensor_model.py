import pytest

from cineos.native_image.tensor_model import CineosTensorModel, Tensor


def _features(offset: float = 0.0) -> Tensor:
    return Tensor(tuple((index / 10.0) + offset for index in range(8)), (8,))


def test_tensor_model_has_separate_identity_scene_and_latent_paths():
    model = CineosTensorModel.initialized()
    identity = model.encode_identity_tensor(_features())
    scene = model.encode_scene_tensor(_features(0.1))
    latent = model.predict_latent_tensor(identity, scene)

    assert identity.shape == (8,)
    assert scene.shape == (8,)
    assert latent.shape == (16,)
    assert latent.device == "cpu"


def test_tensor_model_forward_is_deterministic():
    first = CineosTensorModel.initialized().forward(_features(), _features(0.2))
    second = CineosTensorModel.initialized().forward(_features(), _features(0.2))
    assert first == second


def test_tensor_mse_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="identical shapes"):
        Tensor((0.0, 1.0), (2,)).mse(Tensor((0.0,), (1,)))
