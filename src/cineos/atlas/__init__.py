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
from .registry import RendererFactory, RendererRegistry
from .runtime import (
    AtlasRuntime,
    RuntimeJob,
    RuntimeState,
    RuntimeStateError,
    RuntimeTask,
    TaskHandler,
)
from .session import RendererSession

__all__ = [
    "BaseRenderer",
    "AtlasRuntime",
    "CapabilityError",
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
    "NATIVE_SHOT_SCHEMA",
    "NativeRenderReceipt",
    "NativeRequestError",
    "NativeShotRequest",
    "NegotiatedCapabilities",
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
    "TaskHandler",
    "compile_native_shot_request",
    "execute_foundation_gpu_shot",
    "ingest_native_request",
    "inspect_cuda_environment",
    "plan_gpu_execution",
    "run_connected_gpu_benchmark",
    "select_gpu_execution",
    "validate_native_request",
]
