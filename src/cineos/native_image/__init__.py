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
from .neural_backend import (
    NeuralModelConfig,
    TorchCineosFlowModel,
    TorchFlowTrainingRunner,
    torch_available,
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
    "NATIVE_IMAGE_PLAN_SCHEMA",
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
    "NativePixelFrame",
    "NativeTrainableModel",
    "NativeTrainingLoop",
    "NativeTrainingSample",
    "NeuralModelConfig",
    "ProceduralLatentComponents",
    "RerenderDecision",
    "SGDOptimizer",
    "TemporalIdentityMemory",
    "Tensor",
    "TensorBatchTrainer",
    "TensorSGDOptimizer",
    "TorchCineosFlowModel",
    "TorchFlowTrainingRunner",
    "TrainableLatentComponents",
    "TrainableParameterSet",
    "TrainingStepResult",
    "VisualContinuityMemory",
    "VisualContinuityObservation",
    "VisualQCReport",
    "apply_temporal_identity_memory",
    "build_rerender_directives",
    "compile_native_image_plan",
    "correction_payload",
    "flow_matching_objective",
    "move_tensor",
    "torch_available",
]
