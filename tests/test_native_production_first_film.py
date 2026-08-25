from __future__ import annotations

from pathlib import Path

import pytest

from cineos.native_video.film_bridge import (
    NativeFilmContinuityBridge,
    temporal_model_fingerprint,
)
from cineos.native_video.final_eval import FFmpegTemporalFilmEvaluator, TemporalFilmEvalPolicy
from cineos.native_video.final_gate import MeasuredFinalFilmGate
from cineos.native_video.production_first_film import (
    PRODUCTION_FIRST_FILM_RUNTIME_KIND,
    build_production_first_film_runtime,
    final_gate_policy_fingerprint,
)
from cineos.native_video.renderer_binding import NativeFilmRendererBinding


class _NativeRenderer:
    def render(self, planned, target: str | Path, *, temporal_state):
        raise AssertionError("composition tests must not render video")


def _runtime_state(runtime) -> dict[str, object]:
    provider = runtime.runner.orchestrator.checkpoint_state_provider
    assert provider is not None
    state = provider()
    assert state is not None
    return state


def test_production_runtime_wires_native_continuity_and_required_final_qc() -> None:
    runtime = build_production_first_film_runtime(_NativeRenderer())

    assert isinstance(runtime.renderer_binding, NativeFilmRendererBinding)
    assert runtime.renderer_binding.continuity is runtime.continuity
    assert runtime.runner.renderer is runtime.renderer_binding
    assert runtime.runner.final_film_evaluator is runtime.final_gate
    assert runtime.runner.require_final_film_evaluation is True
    assert runtime.final_gate.require_audio is True
    assert runtime.final_gate.audio_evaluator is not None

    assert runtime.manifest.renderer_id == "cineos-native"
    assert runtime.manifest.device == "cpu"
    assert runtime.manifest.max_recovery_attempts == 2
    assert runtime.manifest.require_final_film_evaluation is True
    assert runtime.manifest.require_audio is True
    assert runtime.manifest.temporal_model_fingerprint == temporal_model_fingerprint(
        runtime.continuity.model
    )
    assert runtime.manifest.final_gate_policy_fingerprint == final_gate_policy_fingerprint(
        runtime.final_gate
    )

    orchestrator = runtime.runner.orchestrator
    assert orchestrator.checkpoint_state_provider is not None
    assert orchestrator.checkpoint_state_restorer is not None
    assert orchestrator.checkpoint_state_resetter == runtime.continuity.reset
    assert orchestrator.shot_attempt_start == runtime.continuity.start_attempt
    assert orchestrator.shot_attempt_accepted == runtime.continuity.accept_attempt
    assert orchestrator.shot_attempt_rejected == runtime.continuity.reject_attempt

    state = _runtime_state(runtime)
    assert state["kind"] == PRODUCTION_FIRST_FILM_RUNTIME_KIND
    assert state["runtime_manifest"] == runtime.manifest.snapshot()
    assert state["continuity"] == runtime.continuity.snapshot()


def test_production_runtime_preserves_explicit_policy_dependencies() -> None:
    continuity = NativeFilmContinuityBridge.default(device="cpu")
    gate = MeasuredFinalFilmGate(require_audio=True)

    runtime = build_production_first_film_runtime(
        _NativeRenderer(),
        continuity=continuity,
        final_gate=gate,
        renderer_id="cineos-native-v1",
        max_recovery_attempts=4,
        device="cpu",
    )

    assert runtime.continuity is continuity
    assert runtime.final_gate is gate
    assert runtime.runner.renderer_id == "cineos-native-v1"
    assert runtime.runner.orchestrator.max_recovery_attempts == 4
    assert runtime.manifest.renderer_id == "cineos-native-v1"
    assert runtime.manifest.max_recovery_attempts == 4


def test_production_resume_rejects_changed_renderer_before_continuity_restore() -> None:
    runtime = build_production_first_film_runtime(_NativeRenderer())
    payload = _runtime_state(runtime)
    saved_manifest = dict(payload["runtime_manifest"])  # type: ignore[arg-type]
    saved_manifest["renderer_id"] = "different-renderer"
    payload["runtime_manifest"] = saved_manifest

    restorer = runtime.runner.orchestrator.checkpoint_state_restorer
    assert restorer is not None
    before = runtime.continuity.snapshot()

    with pytest.raises(ValueError, match="renderer_id"):
        restorer(payload)

    assert runtime.continuity.snapshot() == before


def test_production_resume_rejects_changed_temporal_model_fingerprint() -> None:
    runtime = build_production_first_film_runtime(_NativeRenderer())
    payload = _runtime_state(runtime)
    saved_manifest = dict(payload["runtime_manifest"])  # type: ignore[arg-type]
    saved_manifest["temporal_model_fingerprint"] = "not-the-active-model"
    payload["runtime_manifest"] = saved_manifest

    restorer = runtime.runner.orchestrator.checkpoint_state_restorer
    assert restorer is not None
    with pytest.raises(ValueError, match="temporal_model_fingerprint"):
        restorer(payload)


def test_production_resume_rejects_changed_final_gate_thresholds() -> None:
    saved = build_production_first_film_runtime(_NativeRenderer())
    stricter_gate = MeasuredFinalFilmGate(
        temporal_evaluator=FFmpegTemporalFilmEvaluator(
            policy=TemporalFilmEvalPolicy(max_black_ratio=0.01)
        ),
        require_audio=True,
    )
    current = build_production_first_film_runtime(
        _NativeRenderer(), final_gate=stricter_gate
    )

    assert (
        saved.manifest.final_gate_policy_fingerprint
        != current.manifest.final_gate_policy_fingerprint
    )
    restorer = current.runner.orchestrator.checkpoint_state_restorer
    assert restorer is not None
    before = current.continuity.snapshot()

    with pytest.raises(ValueError, match="final_gate_policy_fingerprint"):
        restorer(_runtime_state(saved))

    assert current.continuity.snapshot() == before


def test_production_resume_rejects_legacy_unbound_final_gate_policy() -> None:
    runtime = build_production_first_film_runtime(_NativeRenderer())
    payload = _runtime_state(runtime)
    saved_manifest = dict(payload["runtime_manifest"])  # type: ignore[arg-type]
    saved_manifest.pop("final_gate_policy_fingerprint")
    payload["runtime_manifest"] = saved_manifest

    restorer = runtime.runner.orchestrator.checkpoint_state_restorer
    assert restorer is not None
    with pytest.raises(ValueError, match="final_gate_policy_fingerprint"):
        restorer(payload)


def test_production_resume_allows_retry_budget_change() -> None:
    saved = build_production_first_film_runtime(
        _NativeRenderer(), max_recovery_attempts=1
    )
    current = build_production_first_film_runtime(
        _NativeRenderer(), max_recovery_attempts=5
    )

    restorer = current.runner.orchestrator.checkpoint_state_restorer
    assert restorer is not None
    restorer(_runtime_state(saved))

    assert current.continuity.snapshot() == saved.continuity.snapshot()


def test_production_resume_rejects_unbound_legacy_runtime_state() -> None:
    runtime = build_production_first_film_runtime(_NativeRenderer())
    restorer = runtime.runner.orchestrator.checkpoint_state_restorer
    assert restorer is not None

    with pytest.raises(ValueError, match="unsupported production FIRST FILM"):
        restorer(runtime.continuity.snapshot())


def test_production_resume_rejects_missing_manifest_or_continuity() -> None:
    runtime = build_production_first_film_runtime(_NativeRenderer())
    restorer = runtime.runner.orchestrator.checkpoint_state_restorer
    assert restorer is not None

    with pytest.raises(ValueError, match="runtime_manifest"):
        restorer({"kind": PRODUCTION_FIRST_FILM_RUNTIME_KIND})

    with pytest.raises(ValueError, match="continuity"):
        restorer(
            {
                "kind": PRODUCTION_FIRST_FILM_RUNTIME_KIND,
                "runtime_manifest": runtime.manifest.snapshot(),
            }
        )


def test_production_runtime_rejects_device_mismatch() -> None:
    continuity = NativeFilmContinuityBridge.default(device="cpu")

    with pytest.raises(ValueError, match="continuity device"):
        build_production_first_film_runtime(
            _NativeRenderer(), continuity=continuity, device="cuda"
        )


def test_production_runtime_rejects_negative_recovery_budget() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_production_first_film_runtime(_NativeRenderer(), max_recovery_attempts=-1)


def test_production_runtime_rejects_empty_runtime_identity() -> None:
    with pytest.raises(ValueError, match="renderer_id"):
        build_production_first_film_runtime(_NativeRenderer(), renderer_id=" ")

    with pytest.raises(ValueError, match="device"):
        build_production_first_film_runtime(_NativeRenderer(), device=" ")
