from __future__ import annotations

import pytest

from cineos.native_image.tensor_model import Tensor
from cineos.native_video.runtime import (
    MotionDampingRetryPolicy,
    NativeTemporalRuntime,
    TemporalGenerationError,
)
from cineos.native_video.temporal_model import NativeTemporalModel, TemporalFrameInput
from cineos.native_video.temporal_qc import TemporalContinuityGate, TemporalQCPolicy


def _frame(index: int, motion: float = 0.05) -> TemporalFrameInput:
    return TemporalFrameInput(
        shot_id="shot-001",
        frame_index=index,
        identity=Tensor((0.1,) * 8, (8,)),
        scene=Tensor((0.2,) * 8, (8,)),
        motion=Tensor((motion,) * 4, (4,)),
    )


def test_runtime_commits_only_qc_accepted_candidate() -> None:
    model = NativeTemporalModel.initialized()
    runtime = NativeTemporalRuntime.default(model=model)
    state = model.initial_state("shot-001")

    result = runtime.generate_frame(_frame(0), state)

    assert result.report.accepted is True
    assert result.attempts == 1
    assert state.last_frame_index == 0
    assert state.last_latent == result.candidate.latent
    assert state.metadata["temporal_attempts"] == 1
    assert state.metadata["temporal_retries"] == 0


def test_runtime_exhausted_retries_preserve_last_accepted_frame() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    NativeTemporalRuntime.default(model=model).generate_frame(_frame(0), state)
    accepted_snapshot = state.snapshot()

    strict_gate = TemporalContinuityGate(
        TemporalQCPolicy(warn_delta=0.0, reject_delta=0.0)
    )
    runtime = NativeTemporalRuntime(
        model=model,
        gate=strict_gate,
        retry_policy=MotionDampingRetryPolicy(),
        max_retries=2,
    )

    with pytest.raises(TemporalGenerationError) as exc_info:
        runtime.generate_frame(_frame(1, motion=1.0), state)

    assert exc_info.value.attempts == 3
    assert state.last_frame_index == accepted_snapshot["last_frame_index"]
    assert state.last_latent is not None
    assert list(state.last_latent.values) == accepted_snapshot["last_latent"]
    assert state.metadata["temporal_failed_candidates"] == 3


def test_motion_damping_retry_policy_preserves_identity_scene_and_device() -> None:
    frame = _frame(1, motion=0.8)
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    first = model.propose(_frame(0), state)
    model.commit(first, state)
    candidate = model.propose(frame, state)
    report = TemporalContinuityGate(
        TemporalQCPolicy(warn_delta=0.0, reject_delta=0.0)
    ).evaluate(candidate, state)

    adapted = MotionDampingRetryPolicy(damping=0.5).adapt(
        frame,
        report,
        attempt=1,
    )

    assert adapted.identity is frame.identity
    assert adapted.scene is frame.scene
    assert adapted.motion.device == frame.motion.device
    assert adapted.motion.values == (0.4,) * 4


def test_runtime_rejects_negative_retry_budget() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        NativeTemporalRuntime(
            model=NativeTemporalModel.initialized(),
            gate=TemporalContinuityGate(),
            retry_policy=MotionDampingRetryPolicy(),
            max_retries=-1,
        )
