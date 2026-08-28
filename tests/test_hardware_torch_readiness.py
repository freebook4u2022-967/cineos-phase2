from types import SimpleNamespace

import pytest

from cineos.hardware.torch_readiness import probe_torch_device

GIB = 1024**3


class FakeCuda:
    def __init__(self, *, memory=24 * GIB, bf16=True, count=1):
        self.memory = memory
        self.bf16 = bf16
        self.count = count

    def is_available(self):
        return self.count > 0

    def device_count(self):
        return self.count

    def current_device(self):
        return 0

    def get_device_properties(self, index):
        if index >= self.count:
            raise RuntimeError("missing GPU")
        return SimpleNamespace(name=f"Fake GPU {index}", total_memory=self.memory)

    def is_bf16_supported(self):
        return self.bf16


class FakeTorch:
    float32 = object()
    bfloat16 = object()

    def __init__(self, cuda):
        self.cuda = cuda
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))


def test_probe_reports_cuda_capacity_and_bf16_support():
    report = probe_torch_device(
        "auto",
        dtype="bfloat16",
        minimum_cuda_vram_bytes=20 * GIB,
        torch_module=FakeTorch(FakeCuda(memory=24 * GIB)),
    )

    assert report.ready is True
    assert report.device == "cuda"
    assert report.accelerator == "cuda"
    assert report.device_index == 0
    assert report.device_name == "Fake GPU 0"
    assert report.total_memory_bytes == 24 * GIB
    assert report.bf16_supported is True
    assert report.failures == ()


def test_probe_fails_closed_when_model_vram_requirement_is_not_met():
    report = probe_torch_device(
        "cuda:0",
        minimum_cuda_vram_bytes=24 * GIB,
        torch_module=FakeTorch(FakeCuda(memory=16 * GIB)),
    )

    assert report.ready is False
    assert report.failures == ("cuda-vram-below-minimum",)
    with pytest.raises(RuntimeError, match="cuda-vram-below-minimum"):
        report.require_ready()


def test_probe_rejects_bfloat16_when_cuda_cannot_execute_it():
    report = probe_torch_device(
        "cuda",
        dtype="bfloat16",
        torch_module=FakeTorch(FakeCuda(bf16=False)),
    )

    assert report.ready is False
    assert report.failures == ("cuda-bfloat16-unsupported",)


def test_probe_reports_invalid_explicit_cuda_index_without_fallback():
    report = probe_torch_device(
        "cuda:2",
        torch_module=FakeTorch(FakeCuda(count=1)),
    )

    assert report.ready is False
    assert report.device is None
    assert report.failures == ("requested-device-unavailable",)


def test_probe_validates_declared_dtype_before_model_load():
    report = probe_torch_device(
        "cpu",
        dtype="float8_e4m3fn",
        torch_module=FakeTorch(FakeCuda(count=0)),
    )

    assert report.ready is False
    assert report.failures == ("requested-dtype-unavailable",)


def test_probe_rejects_negative_memory_requirement():
    with pytest.raises(ValueError, match="non-negative"):
        probe_torch_device(
            "cpu",
            minimum_cuda_vram_bytes=-1,
            torch_module=FakeTorch(FakeCuda(count=0)),
        )
