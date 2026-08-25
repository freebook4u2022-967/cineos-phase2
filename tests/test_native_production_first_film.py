from __future__ import annotations

from pathlib import Path

import pytest

from cineos.native_video.film_bridge import NativeFilmContinuityBridge
from cineos.native_video.final_gate import MeasuredFinalFilmGate
from cineos.native_video.production_first_film import (
    build_production_first_film_runtime,
)
from cineos.native_video.renderer_binding import NativeFilmRendererBinding


class _NativeRenderer:
    def render(self, planned, target: str | Path, *, temporal_state):
        raise AssertionError("composition tests must not render video")


def test_production_runtime_wires_native_continuity_and_required_final_qc() -> None:
    runtime = build_production_first_film_runtime(_NativeRenderer())

    assert isinstance(runtime.renderer_binding, NativeFilmRendererBinding)
    assert runtime.renderer_binding.continuity is runtime.continuity
    assert runtime.runner.renderer is runtime.renderer_binding
    assert runtime.runner.final_film_evaluator is runtime.final_gate
    assert runtime.runner.require_final_film_evaluation is True
    assert runtime.final_gate.require_audio is True
    assert runtime.final_gate.audio_evaluator is not None

    orchestrator = runtime.runner.orchestrator
    assert orchestrator.checkpoint_state_provider == runtime.continuity.snapshot
    assert orchestrator.checkpoint_state_restorer == runtime.continuity.restore
    assert orchestrator.checkpoint_state_resetter == runtime.continuity.reset
    assert orchestrator.shot_attempt_start == runtime.continuity.start_attempt
    assert orchestrator.shot_attempt_accepted == runtime.continuity.accept_attempt
    assert orchestrator.shot_attempt_rejected == runtime.continuity.reject_attempt


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


def test_production_runtime_rejects_device_mismatch() -> None:
    continuity = NativeFilmContinuityBridge.default(device="cpu")

    with pytest.raises(ValueError, match="continuity device"):
        build_production_first_film_runtime(
            _NativeRenderer(), continuity=continuity, device="cuda"
        )


def test_production_runtime_rejects_negative_recovery_budget() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_production_first_film_runtime(
            _NativeRenderer(), max_recovery_attempts=-1
        )
