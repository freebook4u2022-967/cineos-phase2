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
from .measured_observer import (
    MEASURED_NATIVE_FRAME_OBSERVER_SCHEMA,
    MeasuredNativeFrameObserver,
    VisualContinuitySource,
    merge_visual_observations,
)
from .model_release import (
    MODEL_RELEASE_RECORD_SCHEMA,
    NativeModelReleaseController,
    NativeModelReleaseDecision,
)
from .neural_backend import (
    NeuralModelConfig,
    TorchCineosFlowModel,
    TorchFlowTrainingRunner,
    torch_available,
)
from .rerender import AutomaticRerenderController, RerenderDecision, correction_payload
from .spatial_evidence import (
    SPATIAL_FRAME_EVIDENCE_SCHEMA,
    MeasuredSpatialContinuityObserver,
    SpatialContinuityMemory,
    SpatialFrameDescriptor,
    describe_spatial_rgb_frame,
    spatial_similarity,
)
from .temporal_identity import (
    IdentityObservation,
    IdentityObservationError,
    IdentityQCReport,
    IdentityVisualQCGate,
    TemporalIdentityMemory,
    apply_temporal_identity_memory,
)
from .tensor_checkpoint import (
    TENSOR_CHECKPOINT_SCHEMA,
    TensorCheckpointError,
    TensorTrainingCheckpoint,
)
from .tensor_model import CineosTensorModel, LinearTensorLayer, Tensor
from .tensor_training import (
    FlowMatchingBatch,
    FlowMatchingResult,
    TensorBatchTrainer,
    TensorSGDOptimizer,
    flow_matching_objective,
    move_tensor,
)
from .trainable_model import (
    NativeTrainableModel,
    NativeTrainingLoop,
    SGDOptimizer,
    TrainableParameterSet,
    TrainingStepResult,
)
from .training import (
    CHECKPOINT_MANIFEST_SCHEMA,
    DATASET_MANIFEST_SCHEMA,
    LearnedIdentityEncoder,
    LearnedLatentSampler,
    LearnedRGBDecoder,
    LearnedSceneEncoder,
    NativeCheckpointManifest,
    NativeDatasetManifest,
    NativeTrainingSample,
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
    "CHECKPOINT_MANIFEST_SCHEMA",
    "DATASET_MANIFEST_SCHEMA",
    "MEASURED_NATIVE_FRAME_OBSERVER_SCHEMA",
    "MODEL_RELEASE_RECORD_SCHEMA",
    "NATIVE_IMAGE_PLAN_SCHEMA",
    "SPATIAL_FRAME_EVIDENCE_SCHEMA",
    "TENSOR_CHECKPOINT_SCHEMA",
    "VISUAL_QC_AXES",
    "AutomaticRerenderController",
    "CineosLatentFrameModel",
    "CineosTensorModel",
    "FlowMatchingBatch",
    "FlowMatchingResult",
    "IdentityObservation",
    "IdentityObservationError",
    "IdentityQCReport",
    "IdentityVisualQCGate",
    "LearnedIdentityEncoder",
    "LearnedLatentSampler",
    "LearnedRGBDecoder",
    "LearnedSceneEncoder",
    "LinearTensorLayer",
    "MeasuredNativeFrameObserver",
    "MeasuredSpatialContinuityObserver",
    "MultiAxisVisualQCGate",
    "NativeCheckpointManifest",
    "NativeDatasetManifest",
    "NativeFrameAttempt",
    "NativeFrameGenerationResult",
    "NativeFrameObserver",
    "NativeFrameRuntime",
    "NativeImageConditioningPlan",
    "NativeImageResearchBackend",
    "NativeImageResearchResult",
    "NativeModelReleaseController",
    "NativeModelReleaseDecision",
    "NativePixelFrame",
    "NativeTrainableModel",
    "NativeTrainingLoop",
    "NativeTrainingSample",
    "NeuralModelConfig",
    "ProceduralLatentComponents",
    "RerenderDecision",
    "SGDOptimizer",
    "SpatialContinuityMemory",
    "SpatialFrameDescriptor",
    "TemporalIdentityMemory",
    "Tensor",
    "TensorBatchTrainer",
    "TensorCheckpointError",
    "TensorSGDOptimizer",
    "TensorTrainingCheckpoint",
    "TorchCineosFlowModel",
    "TorchFlowTrainingRunner",
    "TrainableLatentComponents",
    "TrainableParameterSet",
    "TrainingStepResult",
    "VisualContinuityMemory",
    "VisualContinuityObservation",
    "VisualContinuitySource",
    "VisualQCReport",
    "apply_temporal_identity_memory",
    "build_rerender_directives",
    "compile_native_image_plan",
    "correction_payload",
    "describe_spatial_rgb_frame",
    "flow_matching_objective",
    "merge_visual_observations",
    "move_tensor",
    "spatial_similarity",
    "torch_available",
]
