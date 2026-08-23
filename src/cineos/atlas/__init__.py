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
    "ingest_native_request",
    "validate_native_request",
]
