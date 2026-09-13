from pathlib import Path

import pytest

from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.sequence_quality import ArtifactMeasuredSequenceQualityEvaluator


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-production-quality-retry",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={"action": "walk"},
        approved_reference_ids=["lead-approved-reference"],
        deterministic_seed=5100 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
        metadata={"prompt": f"lead continues through shot {index}"},
    )
    request.refresh_hash()
    return request


def _synthetic_quality_evaluator(*_args, **_kwargs):
    return {
        "accepted": True,
        "identity_similarity": 0.95,
        "temporal_consistency": 0.94,
        "artifact_integrity": 0.99,
        "motion_quality": 0.92,
    }


class _UnusedAttestedObserver:
    """Valid observer identity for gates that must reject before measurement."""

    production_measurement_evidence = True
    observer_id = "test-unused-production-observer/0.1"

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("metric extractor must not run before render")


def _measured_quality_evaluator():
    return ArtifactMeasuredSequenceQualityEvaluator(_UnusedAttestedObserver())


def _injected_executor(*_args, **_kwargs):
    raise AssertionError("injected executor must be rejected before rendering")


def test_production_quality_retry_rejects_injected_executor_before_render(tmp_path):
    benchmark_id = "reject-injected-production-retry"

    with pytest.raises(
        ProductionGPUQualityRetryError,
        match="unmodified default shot executor",
    ):
        run_production_quality_retry_connected_gpu_benchmark(
            benchmark_id,
            [_request(index) for index in range(5)],
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=_measured_quality_evaluator(),
            shot_executor=_injected_executor,
        )

    assert list(Path(tmp_path).iterdir()) == []


def test_production_quality_retry_rejects_synthetic_quality_before_render(tmp_path):
    with pytest.raises(
        ProductionGPUQualityRetryError,
        match="artifact-measured sequence quality evidence",
    ):
        run_production_quality_retry_connected_gpu_benchmark(
            "reject-synthetic-production-quality",
            [_request(index) for index in range(5)],
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=_synthetic_quality_evaluator,
        )

    assert list(Path(tmp_path).iterdir()) == []


def test_production_quality_retry_requires_connected_five_to_ten_shot_contract(
    tmp_path,
):
    with pytest.raises(Exception, match="between 5 and 10 shots"):
        run_production_quality_retry_connected_gpu_benchmark(
            "too-short-production-retry",
            [_request(index) for index in range(4)],
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=_measured_quality_evaluator(),
        )
