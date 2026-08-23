"""CINEOS native image research API."""

from .backend import NativeImageResearchBackend, NativeImageResearchResult
from .conditioning import (
    NATIVE_IMAGE_PLAN_SCHEMA,
    NativeImageConditioningPlan,
    compile_native_image_plan,
)
from .frame_runtime import (
    NativeFrameAttempt,
    NativeFrameGenerationResult,
    NativeFrameObserver,
    NativeFrameRuntime,
)
from .latent_model import (
    CineosLatentFrameModel,
    NativePixelFrame,
    ProceduralLatentComponents,
    TrainableLatentComponents,
)
from .rerender import AutomaticRerenderController, RerenderDecision, correction_payload
from .temporal_identity import (
    IdentityObservation,
    IdentityObservationError,
    IdentityQCReport,
    IdentityVisualQCGate,
    TemporalIdentityMemory,
    apply_temporal_identity_memory,
)
from .visual_qc import (
    VISUAL_QC_AXES,
    MultiAxisVisualQCGate,
    VisualContinuityMemory,
    VisualContinuityObservation,
    VisualQCReport,
    build_rerender_directives,
)

__all__ = [
    "NATIVE_IMAGE_PLAN_SCHEMA",
    "VISUAL_QC_AXES",
    "AutomaticRerenderController",
    "CineosLatentFrameModel",
    "IdentityObservation",
    "IdentityObservationError",
    "IdentityQCReport",
    "IdentityVisualQCGate",
    "MultiAxisVisualQCGate",
    "NativeFrameAttempt",
    "NativeFrameGenerationResult",
    "NativeFrameObserver",
    "NativeFrameRuntime",
    "NativeImageConditioningPlan",
    "NativeImageResearchBackend",
    "NativeImageResearchResult",
    "NativePixelFrame",
    "ProceduralLatentComponents",
    "RerenderDecision",
    "TemporalIdentityMemory",
    "TrainableLatentComponents",
    "VisualContinuityMemory",
    "VisualContinuityObservation",
    "VisualQCReport",
    "apply_temporal_identity_memory",
    "build_rerender_directives",
    "compile_native_image_plan",
    "correction_payload",
]
