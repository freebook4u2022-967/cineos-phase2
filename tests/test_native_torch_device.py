"""Neural device selection must be deterministic and fail closed."""

import pytest

from cineos.hardware.torch_device import resolve_torch_device


class _Flag:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class _Cuda(_Flag):
    def __init__(self, available: bool, count: int = 0) -> None:
        super().__init__(available)
        self.count = count

    def device_count(self) -> int:
        return self.count


class _Backends:
    def __init__(self, mps_available: bool) -> None:
        self.mps = _Flag(mps_available)


class _Torch:
    def __init__(self, *, cuda: bool, cuda_count: int = 0, mps: bool = False) -> None:
        self.cuda = _Cuda(cuda, cuda_count)
        self.backends = _Backends(mps)


def test_auto_prefers_cuda_then_mps_then_cpu() -> None:
    assert (
        resolve_torch_device(
            "auto", torch_module=_Torch(cuda=True, cuda_count=1, mps=True)
        )
        == "cuda"
    )
    assert (
        resolve_torch_device("auto", torch_module=_Torch(cuda=False, mps=True)) == "mps"
    )
    assert (
        resolve_torch_device("auto", torch_module=_Torch(cuda=False, mps=False))
        == "cpu"
    )


def test_explicit_cuda_never_silently_falls_back() -> None:
    with pytest.raises(RuntimeError, match="CUDA device is unavailable"):
        resolve_torch_device("cuda", torch_module=_Torch(cuda=False))


def test_explicit_cuda_index_is_validated() -> None:
    torch = _Torch(cuda=True, cuda_count=2)
    assert resolve_torch_device("cuda:1", torch_module=torch) == "cuda:1"
    with pytest.raises(RuntimeError, match=r"reports 2 device\(s\)"):
        resolve_torch_device("cuda:2", torch_module=torch)


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported torch device"):
        resolve_torch_device("gpu", torch_module=_Torch(cuda=True, cuda_count=1))
