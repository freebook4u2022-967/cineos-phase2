from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_foundation_smoke import execute_foundation_gpu_shot
from cineos.atlas.gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.sequence_quality import ArtifactMeasuredSequenceQualityEvaluator


class AttestedNoopObserver:
    production_measurement_evidence = True
    observer_id = "test-persistent-observer/0.1"

    def __call__(self, *_args, **_kwargs):
        return {}


def _quality_evaluator():
    return ArtifactMeasuredSequenceQualityEvaluator(AttestedNoopObserver())


def _production_receipt(tmp_path):
    return SimpleNamespace(
        manifest_path=str(tmp_path / "evidence.json"),
        production_gpu_evidence=True,
        production_quality_evidence=True,
        evidence_tier="production-gpu-quality-gated",
    )


def test_production_quality_retry_reuses_one_persistent_model_session(
    monkeypatch, tmp_path
):
    lifecycle = []
    captured = {}
    receipt = _production_receipt(tmp_path)

    class FakePersistentExecutor:
        def __init__(self, profile, *, output_dir, **kwargs):
            self.profile = profile
            self.output_dir = output_dir
            self.kwargs = kwargs
            lifecycle.append(("init", self))

        def __enter__(self):
            lifecycle.append(("enter", self))
            return self

        def __exit__(self, exc_type, exc, traceback):
            lifecycle.append(("exit", self))

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return receipt

    monkeypatch.setattr(
        "cineos.atlas.gpu_production_quality_retry.PersistentGPUFoundationExecutor",
        FakePersistentExecutor,
    )
    monkeypatch.setattr(
        "cineos.atlas.gpu_production_quality_retry.run_quality_retry_connected_gpu_benchmark",
        fake_run,
    )

    profile = object()
    result = run_production_quality_retry_connected_gpu_benchmark(
        "connected-quality",
        (),
        profile,
        output_dir=tmp_path,
        quality_evaluator=_quality_evaluator(),
        shot_executor_kwargs={
            "estimated_model_vram_gb": 21.5,
            "prefer_bfloat16": False,
        },
    )

    assert result is receipt
    assert [event[0] for event in lifecycle] == ["init", "enter", "exit"]
    executor = lifecycle[0][1]
    assert executor.profile is profile
    assert executor.kwargs == {
        "estimated_model_vram_gb": 21.5,
        "prefer_bfloat16": False,
    }
    assert captured["kwargs"]["shot_executor"] is executor
    assert captured["kwargs"]["shot_executor_kwargs"] is None


def test_production_quality_retry_rejects_injected_runtime_before_session(
    monkeypatch, tmp_path
):
    opened = []

    class ShouldNotOpen:
        def __init__(self, *args, **kwargs):
            opened.append(True)

    monkeypatch.setattr(
        "cineos.atlas.gpu_production_quality_retry.PersistentGPUFoundationExecutor",
        ShouldNotOpen,
    )

    with pytest.raises(
        ProductionGPUQualityRetryError,
        match="forbids injected runtime boundary kwargs: pipeline_factory",
    ):
        run_production_quality_retry_connected_gpu_benchmark(
            "connected-quality",
            (),
            object(),
            output_dir=tmp_path,
            quality_evaluator=_quality_evaluator(),
            shot_executor=execute_foundation_gpu_shot,
            shot_executor_kwargs={"pipeline_factory": object()},
        )

    assert opened == []


def test_production_quality_retry_rejects_unknown_runtime_options(tmp_path):
    with pytest.raises(
        ProductionGPUQualityRetryError,
        match="unsupported runtime kwargs: mystery_option",
    ):
        run_production_quality_retry_connected_gpu_benchmark(
            "connected-quality",
            (),
            object(),
            output_dir=tmp_path,
            quality_evaluator=_quality_evaluator(),
            shot_executor_kwargs={"mystery_option": True},
        )
