from cineos.native_image.tensor_model import Tensor
from cineos.native_video.temporal_model import (
    TemporalFrameOutput,
    TemporalSequenceState,
)
from cineos.native_video.temporal_qc import TemporalContinuityGate, TemporalQCPolicy


def _state() -> TemporalSequenceState:
    return TemporalSequenceState(
        shot_id="shot-1",
        hidden=Tensor((0.0, 0.0), (2,), "cpu"),
        last_frame_index=3,
        last_latent=Tensor((0.1, 0.2), (2,), "cpu"),
    )


def _candidate(delta: float) -> TemporalFrameOutput:
    return TemporalFrameOutput(
        shot_id="shot-1",
        frame_index=4,
        latent=Tensor((0.2, 0.3), (2,), "cpu"),
        hidden=Tensor((0.3, 0.4), (2,), "cpu"),
        continuity_delta=delta,
    )


def test_temporal_qc_accepts_small_continuity_delta_without_mutating_state() -> None:
    state = _state()
    snapshot = state.snapshot()
    report = TemporalContinuityGate().evaluate(_candidate(0.1), state)

    assert report.decision == "accept"
    assert report.accepted is True
    assert report.should_retry is False
    assert state.snapshot() == snapshot


def test_temporal_qc_warns_on_moderate_drift() -> None:
    report = TemporalContinuityGate().evaluate(_candidate(0.5), _state())

    assert report.decision == "warn"
    assert report.accepted is True
    assert report.directives == ("monitor temporal drift on the next accepted frame",)


def test_temporal_qc_retries_large_jump_and_preserves_state() -> None:
    state = _state()
    snapshot = state.snapshot()
    report = TemporalContinuityGate().evaluate(_candidate(0.9), state)

    assert report.decision == "retry"
    assert report.accepted is False
    assert report.should_retry is True
    assert "do not commit rejected recurrent state" in report.directives
    assert state.snapshot() == snapshot


def test_temporal_qc_policy_rejects_invalid_threshold_order() -> None:
    try:
        TemporalQCPolicy(warn_delta=0.8, reject_delta=0.5)
    except ValueError as exc:
        assert "warn_delta cannot exceed reject_delta" in str(exc)
    else:
        raise AssertionError("invalid temporal QC policy must fail")


def test_temporal_qc_rejects_candidate_from_wrong_shot() -> None:
    candidate = TemporalFrameOutput(
        shot_id="other-shot",
        frame_index=4,
        latent=Tensor((0.2, 0.3), (2,), "cpu"),
        hidden=Tensor((0.3, 0.4), (2,), "cpu"),
        continuity_delta=0.1,
    )
    try:
        TemporalContinuityGate().evaluate(candidate, _state())
    except ValueError as exc:
        assert "same shot" in str(exc)
    else:
        raise AssertionError("cross-shot temporal QC must fail")
