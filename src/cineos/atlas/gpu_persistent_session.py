"""Persistent foundation-video GPU session for connected-shot inference.

The legacy single-shot executor intentionally owns a complete renderer lifecycle.
That is ideal for isolated proof, but expensive for a 5-10 shot film sequence.
This module keeps one selected foundation model resident across multiple renders
while preserving the same request, artifact, provenance, and MP4 integrity gates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any

from .foundation_profiles import FoundationExecutionProfile
from .gpu_foundation_smoke import (
    GPUFoundationExecutionError,
    GPUFoundationExecutionReceipt,
    _expected_artifact_path,
    _remove_stale_expected_artifact,
    _runtime_provenance,
    _validate_result_identity,
    _validate_video_artifact,
)
from .gpu_preflight import inspect_cuda_environment, select_gpu_execution
from .native_request import NativeShotRequest
from .production_continuity_identity_runtime import bind_continuity_identity_runtime
from .production_multi_reference import bind_production_multi_reference_runtime
from .production_references import bind_production_reference_runtime


class PersistentGPUSessionError(GPUFoundationExecutionError):
    """Raised when a persistent GPU session is misused or cannot render safely."""


class PersistentGPUFoundationExecutor:
    """Keep one foundation renderer loaded for a connected sequence.

    The object is callable with the normal ``ShotExecutor`` signature, so existing
    benchmark/retry orchestration can use it without duplicating rendering logic.
    It must be opened with a context manager; shutdown is guaranteed on exit.

    Session receipts also expose measured model-load amortization and, when the
    selected CUDA runtime provides the standard PyTorch memory-stat APIs, per-shot
    peak allocated/reserved VRAM. These are observations only: absence of an API is
    never replaced with invented measurements.
    """

    def __init__(
        self,
        profile: FoundationExecutionProfile,
        *,
        output_dir: str | Path,
        estimated_model_vram_gb: float | None = None,
        prefer_bfloat16: bool = True,
        torch_module: Any | None = None,
        reference_loader: Any | None = None,
        multi_reference_adapter: Any | None = None,
        continuity_identity_adapter: Any | None = None,
        pipeline_factory: Any | None = None,
        video_exporter: Any | None = None,
    ) -> None:
        self.profile = profile
        self.output_dir = Path(output_dir)
        self.estimated_model_vram_gb = estimated_model_vram_gb
        self.prefer_bfloat16 = prefer_bfloat16
        self.torch_module = torch_module
        self.reference_loader = reference_loader
        self.multi_reference_adapter = multi_reference_adapter
        self.continuity_identity_adapter = continuity_identity_adapter
        self.pipeline_factory = pipeline_factory
        self.video_exporter = video_exporter
        self._renderer: Any | None = None
        self._plan: Any | None = None
        self._runtime: dict[str, Any] | None = None
        self._model_load_seconds: float | None = None
        self._render_count = 0
        self._cumulative_render_seconds = 0.0

    @property
    def is_open(self) -> bool:
        return self._renderer is not None

    def open(self) -> PersistentGPUFoundationExecutor:
        if self.is_open:
            raise PersistentGPUSessionError("persistent GPU session is already open")
        estimated_vram = (
            self.profile.minimum_gpu_vram_gb
            if self.estimated_model_vram_gb is None
            else float(self.estimated_model_vram_gb)
        )
        devices = inspect_cuda_environment(torch_module=self.torch_module)
        plan = select_gpu_execution(
            devices,
            estimated_model_vram_gb=estimated_vram,
            prefer_bfloat16=self.prefer_bfloat16,
        )
        renderer_kwargs: dict[str, Any] = {
            "output_dir": self.output_dir,
            "reference_loader": self.reference_loader,
            "multi_reference_adapter": self.multi_reference_adapter,
            "pipeline_factory": self.pipeline_factory,
            "video_exporter": self.video_exporter,
        }
        if self.continuity_identity_adapter is not None:
            renderer_kwargs["continuity_identity_adapter"] = (
                self.continuity_identity_adapter
            )
        renderer = self.profile.renderer(**renderer_kwargs)
        renderer.initialize()
        load_started = perf_counter()
        try:
            renderer.load_model(**plan.renderer_options())
            renderer.warmup()
        except Exception:
            renderer.shutdown()
            raise
        model_load_seconds = perf_counter() - load_started

        runtime = _runtime_provenance(
            plan,
            torch_module=self.torch_module,
            reference_loader=self.reference_loader,
            pipeline_factory=self.pipeline_factory,
            video_exporter=self.video_exporter,
        )
        runtime = bind_production_reference_runtime(runtime, self.reference_loader)
        runtime = bind_production_multi_reference_runtime(
            runtime, self.multi_reference_adapter
        )
        runtime = bind_continuity_identity_runtime(
            runtime, self.continuity_identity_adapter
        )
        runtime["persistent_model_session"] = True
        runtime["session_model_load_seconds"] = model_load_seconds
        self._renderer = renderer
        self._plan = plan
        self._runtime = runtime
        self._model_load_seconds = model_load_seconds
        self._render_count = 0
        self._cumulative_render_seconds = 0.0
        return self

    def close(self) -> None:
        renderer = self._renderer
        self._renderer = None
        self._plan = None
        self._runtime = None
        self._model_load_seconds = None
        self._render_count = 0
        self._cumulative_render_seconds = 0.0
        if renderer is not None:
            renderer.shutdown()

    def __enter__(self) -> PersistentGPUFoundationExecutor:
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __call__(
        self,
        request: NativeShotRequest,
        profile: FoundationExecutionProfile,
        *,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> GPUFoundationExecutionReceipt:
        if kwargs:
            raise PersistentGPUSessionError(
                "persistent GPU executor does not accept per-shot runtime overrides"
            )
        if profile != self.profile:
            raise PersistentGPUSessionError(
                "persistent GPU executor profile changed within one model session"
            )
        try:
            requested_root = Path(output_dir).resolve(strict=False)
            session_root = self.output_dir.resolve(strict=False)
        except OSError as exc:
            raise PersistentGPUSessionError(
                "cannot resolve persistent output directory"
            ) from exc
        if requested_root != session_root:
            raise PersistentGPUSessionError(
                "persistent GPU executor output directory changed within one model session"
            )
        return self.render(request)

    def discard_quality_rejected_result(
        self, receipt: GPUFoundationExecutionReceipt
    ) -> None:
        """Ensure rejected GPU output cannot become a successor continuity anchor."""

        renderer = self._renderer
        if renderer is None:
            raise PersistentGPUSessionError(
                "persistent GPU session must be opened before rejecting a render"
            )
        discard = getattr(renderer, "discard_quality_rejected_result", None)
        if discard is None:
            return
        if not callable(discard):
            raise PersistentGPUSessionError(
                "renderer exposes a non-callable quality rejection hook"
            )
        try:
            discard(receipt)
        except Exception as exc:
            raise PersistentGPUSessionError(
                "renderer could not remove rejected continuity state"
            ) from exc

    def render(self, request: NativeShotRequest) -> GPUFoundationExecutionReceipt:
        renderer = self._renderer
        plan = self._plan
        runtime = self._runtime
        model_load_seconds = self._model_load_seconds
        if (
            renderer is None
            or plan is None
            or runtime is None
            or model_load_seconds is None
        ):
            raise PersistentGPUSessionError(
                "persistent GPU session must be opened before rendering"
            )

        expected_artifact = _expected_artifact_path(request, self.output_dir)
        _remove_stale_expected_artifact(expected_artifact)
        self._reset_cuda_peak_memory_stats(renderer)
        started = perf_counter()
        result = renderer.render(request)
        elapsed = perf_counter() - started
        cuda_memory = self._read_cuda_peak_memory_stats(renderer)
        artifact = _validate_result_identity(
            request,
            self.profile,
            result,
            expected_artifact,
        )
        try:
            output_bytes = artifact.stat().st_size
        except OSError as exc:
            raise PersistentGPUSessionError(
                f"renderer reported {artifact} but no readable video artifact exists"
            ) from exc
        if output_bytes <= 0:
            raise PersistentGPUSessionError(
                f"renderer produced an empty video artifact at {artifact}"
            )
        media_payload_bytes = _validate_video_artifact(artifact)

        digest = hashlib.sha256()
        try:
            with artifact.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise PersistentGPUSessionError(
                f"rendered video artifact cannot be hashed: {artifact}"
            ) from exc

        self._render_count += 1
        self._cumulative_render_seconds += elapsed
        receipt_runtime = dict(runtime)
        receipt_runtime.update(
            {
                "session_render_index": self._render_count,
                "session_cumulative_render_seconds": self._cumulative_render_seconds,
                "session_amortized_model_load_seconds_per_render": (
                    model_load_seconds / self._render_count
                ),
                "session_total_measured_execution_seconds": (
                    model_load_seconds + self._cumulative_render_seconds
                ),
            }
        )
        receipt_runtime.update(cuda_memory)

        return GPUFoundationExecutionReceipt(
            result=result,
            execution_plan=plan,
            profile_id=self.profile.profile_id,
            origin=self.profile.origin,
            output_bytes=output_bytes,
            output_sha256=digest.hexdigest(),
            elapsed_seconds=elapsed,
            media_payload_bytes=media_payload_bytes,
            runtime_provenance=receipt_runtime,
        )

    def _torch_runtime(self, renderer: Any) -> Any | None:
        """Return the actual torch runtime behind the renderer, when observable."""
        return getattr(renderer, "_torch", None) or self.torch_module

    @staticmethod
    def _cuda_device_index(renderer: Any) -> int:
        device = str(getattr(renderer, "_device", "cuda"))
        if device == "cuda":
            return 0
        prefix, separator, suffix = device.partition(":")
        if prefix != "cuda" or not separator:
            return 0
        try:
            return max(0, int(suffix))
        except ValueError:
            return 0

    def _reset_cuda_peak_memory_stats(self, renderer: Any) -> None:
        """Reset peak counters when PyTorch exposes them; otherwise record nothing."""
        torch_runtime = self._torch_runtime(renderer)
        cuda = getattr(torch_runtime, "cuda", None)
        reset = getattr(cuda, "reset_peak_memory_stats", None)
        if not callable(reset):
            return
        try:
            reset(self._cuda_device_index(renderer))
        except (RuntimeError, TypeError, ValueError):
            # Telemetry must never break an otherwise valid render on older/custom
            # PyTorch-compatible runtimes. Missing evidence remains visibly missing.
            return

    def _read_cuda_peak_memory_stats(self, renderer: Any) -> dict[str, int]:
        """Measure per-shot CUDA peaks without synthesizing unavailable evidence."""
        torch_runtime = self._torch_runtime(renderer)
        cuda = getattr(torch_runtime, "cuda", None)
        allocated = getattr(cuda, "max_memory_allocated", None)
        reserved = getattr(cuda, "max_memory_reserved", None)
        if not callable(allocated) or not callable(reserved):
            return {}
        device_index = self._cuda_device_index(renderer)
        try:
            peak_allocated = int(allocated(device_index))
            peak_reserved = int(reserved(device_index))
        except (RuntimeError, TypeError, ValueError):
            return {}
        if peak_allocated < 0 or peak_reserved < 0:
            return {}
        return {
            "cuda_peak_memory_allocated_bytes": peak_allocated,
            "cuda_peak_memory_reserved_bytes": peak_reserved,
        }


__all__ = ["PersistentGPUFoundationExecutor", "PersistentGPUSessionError"]
