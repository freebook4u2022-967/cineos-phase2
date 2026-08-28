from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_preflight import (
    GPUDeviceProfile,
    GPUPreflightError,
    inspect_cuda_environment,
    plan_gpu_execution,
)

_GIB = 1024**3


class FakeCuda:
    def __init__(self, *, available=True, total_gb=(24.0,), free_gb=18.0, bf16=True):
        self.available = available
        self.total_gb = total_gb
        self.free_gb = free_gb
        self.bf16 = bf16

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.total_gb)

    def get_device_properties(self, index):
        return SimpleNamespace(
            name=f"Fake GPU {index}", total_memory=int(self.total_gb[index] * _GIB)
        )

    def get_device_capability(self, index):
        return (8, index)

    def is_bf16_supported(self):
        return self.bf16

    def current_device(self):
        return 0

    def mem_get_info(self):
        return int(self.free_gb * _GIB), int(self.total_gb[0] * _GIB)


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda


def _profile(total_gb, *, bf16=True):
    return GPUDeviceProfile(
        index=0,
        name="Test GPU",
        compute_capability=(8, 0),
        total_vram_gb=total_gb,
        free_vram_gb=total_gb - 2,
        supports_bfloat16=bf16,
    )


def test_inspect_cuda_environment_records_observed_device_properties():
    profiles = inspect_cuda_environment(
        FakeTorch(FakeCuda(total_gb=(24.0, 48.0), free_gb=17.5))
    )

    assert len(profiles) == 2
    assert profiles[0].name == "Fake GPU 0"
    assert profiles[0].total_vram_gb == pytest.approx(24.0)
    assert profiles[0].free_vram_gb == pytest.approx(17.5)
    assert profiles[0].supports_bfloat16 is True
    assert profiles[1].free_vram_gb is None


def test_inspect_cuda_environment_fails_explicitly_without_cuda():
    with pytest.raises(GPUPreflightError, match="CUDA is unavailable"):
        inspect_cuda_environment(FakeTorch(FakeCuda(available=False)))


def test_plan_uses_resident_execution_when_gpu_has_safe_headroom():
    plan = plan_gpu_execution(_profile(48.0), estimated_model_vram_gb=32.0)

    assert plan.memory_strategy == "resident"
    assert plan.dtype == "bfloat16"
    assert plan.enable_vae_tiling is False
    assert plan.renderer_options()["device"] == "cuda:0"


def test_plan_uses_model_offload_for_constrained_gpu():
    plan = plan_gpu_execution(_profile(24.0), estimated_model_vram_gb=32.0)

    assert plan.memory_strategy == "model_cpu_offload"
    assert plan.enable_vae_tiling is True
    assert plan.enable_vae_slicing is True
    assert plan.enable_attention_slicing is False


def test_plan_uses_sequential_offload_for_tight_gpu():
    plan = plan_gpu_execution(
        _profile(12.0, bf16=False), estimated_model_vram_gb=32.0
    )

    assert plan.memory_strategy == "sequential_cpu_offload"
    assert plan.dtype == "float16"
    assert plan.enable_attention_slicing is True


def test_plan_rejects_gpu_that_is_too_small_for_declared_model():
    with pytest.raises(GPUPreflightError, match="below the conservative minimum"):
        plan_gpu_execution(_profile(8.0), estimated_model_vram_gb=32.0)


def test_plan_rejects_non_positive_model_memory_estimate():
    with pytest.raises(ValueError, match="must be positive"):
        plan_gpu_execution(_profile(24.0), estimated_model_vram_gb=0)
