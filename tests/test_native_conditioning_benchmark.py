import pytest

from cineos.native_image.conditioning_benchmark import _save_pixels_ppm
from cineos.native_image.neural_backend import _load_torch, torch_available


def test_benchmark_pixel_export_validates_dimensions(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    with pytest.raises(ValueError, match="does not match"):
        _save_pixels_ppm(torch.zeros(10), tmp_path / "bad.ppm", 2, 2)


def test_benchmark_pixel_export_writes_real_ppm(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    path = tmp_path / "cell.ppm"
    _save_pixels_ppm(torch.linspace(0.0, 1.0, steps=12), path, 2, 2)
    payload = path.read_bytes()
    assert payload.startswith(b"P6\n2 2\n255\n")
    assert len(payload) == len(b"P6\n2 2\n255\n") + 12
