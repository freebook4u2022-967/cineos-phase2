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
from .production_references import bind_production_reference_runtime


class PersistentGPUSessionError(GPUFoundationExecutionError):
    """Raised when a persistent GPU session is misused or cannot render safely."""


class PersistentGPUFoundationExecutor:
    """Keep one foundation renderer loaded for a connected sequence.

    The object is callable with the normal ``ShotExecutor`` signature, so existing
    benchmark/retry orchestration can use it without duplicating rendering logic.
    It must be opened with a context manager; shutdown is guaranteed on exit.
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
        pipeline_factory: Any | None = None,
        video_exporter: Any | None = None,
    ) -> None:
        self.profile = profile
        self.output_dir = Path(output_dir)
        self.estimated_model_vram_gb = estimated_model_vram_gb
        self.prefer_bfloat16 = prefer_bfloat16
        self.torch_module = torch_module
        self.reference_loader = reference_loader
        self.pipeline_factory = pipeline_factory
        self.video_exporter = video_exporter
        self._renderer: Any | None = None
        self._plan: Any | None = None
        self._runtime: dict[str, Any] | None = None

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
        renderer = self.profile.renderer(
            output_dir=self.output_dir,
            reference_loader=self.reference_loader,
            pipeline_factory=self.pipeline_factory,
            video_exporter=self.video_exporter,
        )
        renderer.initialize()
        try:
            renderer.load_model(**plan.renderer_options())
            renderer.warmup()
        except Exception:
            renderer.shutdown()
            raise

        runtime = _runtime_provenance(
            plan,
            torch_module=self.torch_module,
            reference_loader=self.reference_loader,
            pipeline_factory=self.pipeline_factory,
            video_exporter=self.video_exporter,
        )
        runtime = bind_production_reference_runtime(runtime, self.reference_loader)
        runtime["persistent_model_session"] = True
        self._renderer = renderer
        self._plan = plan
        self._runtime = runtime
        return self

    def close(self) -> None:
        renderer = self._renderer
        self._renderer = None
        self._plan = None
        self._runtime = None
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

    def render(self, request: NativeShotRequest) -> GPUFoundationExecutionReceipt:
        renderer = self._renderer
        plan = self._plan
        runtime = self._runtime
        if renderer is None or plan is None or runtime is None:
            raise PersistentGPUSessionError(
                "persistent GPU session must be opened before rendering"
            )

        expected_artifact = _expected_artifact_path(request, self.output_dir)
        _remove_stale_expected_artifact(expected_artifact)
        started = perf_counter()
        result = renderer.render(request)
        elapsed = perf_counter() - started
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

        return GPUFoundationExecutionReceipt(
            result=result,
            execution_plan=plan,
            profile_id=self.profile.profile_id,
            origin=self.profile.origin,
            output_bytes=output_bytes,
            output_sha256=digest.hexdigest(),
            elapsed_seconds=elapsed,
            media_payload_bytes=media_payload_bytes,
            runtime_provenance=dict(runtime),
        )


__all__ = ["PersistentGPUFoundationExecutor", "PersistentGPUSessionError"]
