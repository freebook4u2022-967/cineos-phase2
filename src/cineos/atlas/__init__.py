"""Public Atlas renderer SDK."""

from .adapter import RendererAdapter, RendererLifecycleError, RendererState
from .base_renderer import BaseRenderer
from .capabilities import (
    CapabilityError,
    NegotiatedCapabilities,
    Range,
    RendererCapabilities,
    Resolution,
)
from .connected_continuity_evidence import (
    ConnectedContinuityEvidenceError,
    production_visual_continuity_evidence,
    validate_connected_visual_continuity,
)
from .diffusers_video import (
    DiffusersVideoError,
    DiffusersVideoRenderer,
    DiffusersVideoResult,
    FoundationProvenance,
)
from .gpu_connected_benchmark import (
    GPUConnectedBenchmarkError,
    GPUConnectedBenchmarkReceipt,
    run_connected_gpu_benchmark,
)
from .gpu_foundation_smoke import (
    GPUFoundationExecutionError,
    GPUFoundationExecutionReceipt,
    execute_foundation_gpu_shot,
)
from .gpu_preflight import (
    GPUDeviceProfile,
    GPUExecutionPlan,
    GPUPreflightError,
    inspect_cuda_environment,
    plan_gpu_execution,
    select_gpu_execution,
)
from .gpu_quality_benchmark import (
    GPUQualityBenchmarkError,
    QualityGatedShotExecutor,
    run_quality_gated_connected_gpu_benchmark,
)
from .native_ingest import (
    NativeRenderReceipt,
    NativeRequestError,
    ingest_native_request,
    validate_native_request,
)
from .native_request import (
    NATIVE_SHOT_SCHEMA,
    NativeShotRequest,
    compile_native_shot_request,
)
from .production_connected_evidence import (
    ProductionConnectedEvidence,
    ProductionConnectedEvidenceError,
    production_connected_evidence,
    validate_production_connected_evidence,
)
from .production_reference import (
    PRODUCTION_REFERENCE_MANIFEST_SCHEMA,
    ProductionReferenceError,
    ProductionReferenceManifestLoader,
    execute_production_reference_gpu_shot,
)
from .quality_retry import (
    QualityRetryError,
    QualityRetryPolicy,
    build_quality_retry_request,
)
from .registry import RendererFactory, RendererRegistry
from .runtime import (
    AtlasRuntime,
    RuntimeJob,
    RuntimeState,
    RuntimeStateError,
    RuntimeTask,
    TaskHandler,
)
from .semantic_video_scorer import (
    LearnedIdentityMotionScorer,
    SemanticVideoScorerError,
)
from .session import RendererSession

__all__ = [
    "BaseRenderer",
    "AtlasRuntime",
    "CapabilityError",
    "ConnectedContinuityEvidenceError",
    "DiffusersVideoError",
    "DiffusersVideoRenderer",
    "DiffusersVideoResult",
    "FoundationProvenance",
    "GPUConnectedBenchmarkError",
    "GPUConnectedBenchmarkReceipt",
    "GPUDeviceProfile",
    "GPUExecutionPlan",
    "GPUFoundationExecutionError",
    "GPUFoundationExecutionReceipt",
    "GPUPreflightError",
    "GPUQualityBenchmarkError",
    "LearnedIdentityMotionScorer",
    "NATIVE_SHOT_SCHEMA",
    "NativeRenderReceipt",
    "NativeRequestError",
    "NativeShotRequest",
    "NegotiatedCapabilities",
    "PRODUCTION_REFERENCE_MANIFEST_SCHEMA",
    "ProductionConnectedEvidence",
    "ProductionConnectedEvidenceError",
    "ProductionReferenceError",
    "ProductionReferenceManifestLoader",
    "QualityGatedShotExecutor",
    "QualityRetryError",
    "QualityRetryPolicy",
    "Range",
    "RendererAdapter",
    "RendererCapabilities",
    "RendererFactory",
    "RendererLifecycleError",
    "RendererRegistry",
    "RendererSession",
    "RendererState",
    "Resolution",
    "RuntimeJob",
    "RuntimeState",
    "RuntimeStateError",
    "RuntimeTask",
    "SemanticVideoScorerError",
    "TaskHandler",
    "build_quality_retry_request",
    "compile_native_shot_request",
    "execute_foundation_gpu_shot",
    "execute_production_reference_gpu_shot",
    "ingest_native_request",
    "inspect_cuda_environment",
    "plan_gpu_execution",
    "production_connected_evidence",
    "production_visual_continuity_evidence",
    "run_connected_gpu_benchmark",
    "run_quality_gated_connected_gpu_benchmark",
    "select_gpu_execution",
    "validate_connected_visual_continuity",
    "validate_native_request",
    "validate_production_connected_evidence",
]
