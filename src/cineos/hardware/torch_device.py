"""Deterministic PyTorch accelerator selection for CINEOS neural runtimes.

The base hardware probe deliberately does not require PyTorch.  Neural deployment,
however, needs a single fail-closed policy for turning a requested device into the
concrete accelerator that will execute model weights.  ``auto`` prefers CUDA,
then Apple MPS, and finally CPU.  Explicit accelerator requests never silently
fall back to CPU.
"""

from __future__ import annotations

from typing import Any


def _load_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PyTorch is required for neural device selection; install cineos[neural]"
        ) from exc
    return torch


def _mps_available(torch: Any) -> bool:
    try:
        mps = torch.backends.mps
    except AttributeError:
        return False
    try:
        return bool(mps.is_available())
    except (AttributeError, RuntimeError):
        return False


def _cuda_available(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


def resolve_torch_device(
    requested: str = "auto", *, torch_module: Any | None = None
) -> str:
    """Resolve a neural runtime device without hiding accelerator failures.

    ``auto`` is the only mode permitted to fall back between device classes.  An
    explicit ``cuda``/``cuda:N`` or ``mps`` request raises when that accelerator
    cannot actually execute PyTorch workloads.  This prevents production jobs
    from appearing healthy while unexpectedly running expensive inference on CPU.
    """

    normalized = requested.strip().lower()
    if not normalized:
        raise ValueError("torch device must not be empty")

    torch = torch_module or _load_torch()

    if normalized == "auto":
        if _cuda_available(torch):
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"

    if normalized == "cpu":
        return "cpu"

    if normalized == "mps":
        if not _mps_available(torch):
            raise RuntimeError("requested MPS device is unavailable to PyTorch")
        return "mps"

    if normalized == "cuda" or normalized.startswith("cuda:"):
        if not _cuda_available(torch):
            raise RuntimeError("requested CUDA device is unavailable to PyTorch")
        if normalized == "cuda":
            return "cuda"
        index_text = normalized.removeprefix("cuda:")
        if not index_text.isdigit():
            raise ValueError(f"invalid CUDA device: {requested!r}")
        index = int(index_text)
        try:
            device_count = int(torch.cuda.device_count())
        except (AttributeError, RuntimeError) as exc:
            raise RuntimeError("unable to enumerate CUDA devices") from exc
        if index >= device_count:
            raise RuntimeError(
                f"requested CUDA device {index} is unavailable; PyTorch reports "
                f"{device_count} device(s)"
            )
        return f"cuda:{index}"

    raise ValueError(
        f"unsupported torch device {requested!r}; expected auto, cpu, mps, cuda, or cuda:N"
    )
