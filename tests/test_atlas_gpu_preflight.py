from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_preflight import (
    GPUDeviceProfile,
    GPUPreflightError,
    inspect_cuda_environment,
    plan_gpu_execution,
    select_gpu_execution,
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


class FakeDeviceAwareCuda(FakeCuda):
    def __init__(self, *, total_gb, free_gb, bf16=True):
        super().__init__(total_gb=total_gb, free_gb=free_gb[0], bf16=bf16)
        self.free_by_device_gb = free_gb

    def mem_get_info(self, index=None):
        if index is None:
            index = self.current_device()
        return (
            int(self.free_by_device_gb[index] * _GIB),
            int(self.total_gb[index] * _GIB),
        )


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda


def _profile(total_gb, *, free_gb=None, bf16=True, index=0):
    if free_gb is None:
        free_gb = total_gb - 2
    return GPUDeviceProfile(
        index=index,
        name=f"Test GPU {index}",
        compute_capability=(8, 0),
        total_vram_gb=total_gb,
        free_vram_gb=free_gb,
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


def test_inspect_cuda_environment_reads_per_device_free_vram_when_supported():
    profiles = inspect_cuda_environment(
        FakeTorch(
            FakeDeviceAwareCuda(
                total_gb=(24.0, 48.0),
                free_gb=(7.0, 42.0),
            )
        )
    )

    assert profiles[0].free_vram_gb == pytest.approx(7.0)
    assert profiles[1].free_vram_gb == pytest.approx(42.0)


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
    plan = plan_gpu_execution(_profile(12.0, bf16=False), estimated_model_vram_gb=32.0)

    assert plan.memory_strategy == "sequential_cpu_offload"
    assert plan.dtype == "float16"
    assert plan.enable_attention_slicing is True


def test_plan_rejects_gpu_that_is_too_small_for_declared_model():
    with pytest.raises(GPUPreflightError, match="below the conservative minimum"):
        plan_gpu_execution(_profile(8.0), estimated_model_vram_gb=32.0)


def test_plan_uses_live_free_vram_instead_of_idle_total_capacity():
    profile = _profile(48.0, free_gb=18.0)

    plan = plan_gpu_execution(profile, estimated_model_vram_gb=32.0)

    assert plan.memory_strategy == "model_cpu_offload"
    assert plan.fit_margin_gb == pytest.approx(-14.0)
    assert plan.observed_total_vram_gb == pytest.approx(48.0)
    assert plan.observed_free_vram_gb == pytest.approx(18.0)


def test_plan_rejects_busy_gpu_when_free_vram_is_below_working_set_floor():
    with pytest.raises(GPUPreflightError, match="free GPU VRAM"):
        plan_gpu_execution(
            _profile(48.0, free_gb=6.0),
            estimated_model_vram_gb=32.0,
        )


def test_plan_falls_back_to_total_vram_when_live_free_vram_is_unknown():
    profile = GPUDeviceProfile(
        index=1,
        name="Telemetry-limited GPU",
        compute_capability=(8, 0),
        total_vram_gb=48.0,
        free_vram_gb=None,
        supports_bfloat16=True,
    )

    plan = plan_gpu_execution(profile, estimated_model_vram_gb=32.0)

    assert plan.memory_strategy == "resident"
    assert plan.fit_margin_gb == pytest.approx(16.0)


def test_plan_rejects_inconsistent_or_empty_free_vram_telemetry():
    with pytest.raises(GPUPreflightError, match="no free VRAM"):
        plan_gpu_execution(_profile(24.0, free_gb=0.0), estimated_model_vram_gb=16.0)

    with pytest.raises(GPUPreflightError, match="exceeds reported total"):
        plan_gpu_execution(_profile(24.0, free_gb=25.0), estimated_model_vram_gb=16.0)


def test_plan_rejects_non_positive_model_memory_estimate():
    with pytest.raises(ValueError, match="must be positive"):
        plan_gpu_execution(_profile(24.0), estimated_model_vram_gb=0)


def test_select_gpu_execution_prefers_safer_resident_device_over_busy_gpu_zero():
    profiles = (
        _profile(48.0, free_gb=18.0, index=0),
        _profile(48.0, free_gb=44.0, index=1),
    )

    plan = select_gpu_execution(profiles, estimated_model_vram_gb=32.0)

    assert plan.device == "cuda:1"
    assert plan.memory_strategy == "resident"
    assert plan.observed_free_vram_gb == pytest.approx(44.0)


def test_select_gpu_execution_skips_unusable_devices():
    profiles = (
        _profile(48.0, free_gb=5.0, index=0),
        _profile(24.0, free_gb=20.0, index=1),
    )

    plan = select_gpu_execution(profiles, estimated_model_vram_gb=32.0)

    assert plan.device == "cuda:1"
    assert plan.memory_strategy == "model_cpu_offload"


def test_select_gpu_execution_fails_when_every_device_is_below_floor():
    profiles = (
        _profile(24.0, free_gb=4.0, index=0),
        _profile(48.0, free_gb=6.0, index=1),
    )

    with pytest.raises(GPUPreflightError, match="no GPU can safely execute"):
        select_gpu_execution(profiles, estimated_model_vram_gb=32.0)
