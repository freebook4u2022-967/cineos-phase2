from types import SimpleNamespace

import pytest

import cineos.atlas.gpu_production_quality_retry as production_retry
from cineos.atlas.gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.sequence_quality import ArtifactMeasuredSequenceQualityEvaluator


class _UnusedAttestedObserver:
    """Valid production-observer shape for wrapper-only preflight tests."""

    production_measurement_evidence = True
    observer_id = "test-unused-production-observer/0.1"

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("observer must not run in wrapper preflight tests")


def _evaluator():
    return ArtifactMeasuredSequenceQualityEvaluator(_UnusedAttestedObserver())


def _receipt(manifest_path, *, gpu=True, quality=True, tier=None):
    if tier is None:
        tier = (
            "production-gpu-quality-gated"
            if gpu and quality
            else "production-gpu-execution" if gpu else "non-production-or-injected"
        )
    return SimpleNamespace(
        manifest_path=str(manifest_path),
        production_gpu_evidence=gpu,
        production_quality_evidence=quality,
        evidence_tier=tier,
    )


def test_production_wrapper_fails_closed_without_artifact_bound_quality(
    monkeypatch, tmp_path
):
    manifest = tmp_path / "benchmark.gpu-benchmark.json"
    manifest.write_text("stale evidence\n", encoding="utf-8")
    fake = _receipt(manifest, gpu=True, quality=False)

    monkeypatch.setattr(
        production_retry,
        "run_quality_retry_connected_gpu_benchmark",
        lambda *_args, **_kwargs: fake,
    )

    with pytest.raises(
        ProductionGPUQualityRetryError,
        match="artifact-bound measured QC evidence",
    ):
        run_production_quality_retry_connected_gpu_benchmark(
            "production-qc",
            (),
            SimpleNamespace(),
            output_dir=tmp_path,
            quality_evaluator=_evaluator(),
        )

    assert not manifest.exists()


def test_production_wrapper_requires_exact_quality_gated_evidence_tier(
    monkeypatch, tmp_path
):
    manifest = tmp_path / "benchmark.gpu-benchmark.json"
    manifest.write_text("ambiguous evidence\n", encoding="utf-8")
    fake = _receipt(
        manifest,
        gpu=True,
        quality=True,
        tier="unexpected-production-tier",
    )

    monkeypatch.setattr(
        production_retry,
        "run_quality_retry_connected_gpu_benchmark",
        lambda *_args, **_kwargs: fake,
    )

    with pytest.raises(
        ProductionGPUQualityRetryError,
        match="production-gpu-quality-gated evidence tier",
    ):
        run_production_quality_retry_connected_gpu_benchmark(
            "production-tier",
            (),
            SimpleNamespace(),
            output_dir=tmp_path,
            quality_evaluator=_evaluator(),
        )

    assert not manifest.exists()


def test_production_wrapper_returns_only_fully_quality_gated_receipt(
    monkeypatch, tmp_path
):
    manifest = tmp_path / "benchmark.gpu-benchmark.json"
    manifest.write_text("verified evidence\n", encoding="utf-8")
    fake = _receipt(manifest, gpu=True, quality=True)

    monkeypatch.setattr(
        production_retry,
        "run_quality_retry_connected_gpu_benchmark",
        lambda *_args, **_kwargs: fake,
    )

    result = run_production_quality_retry_connected_gpu_benchmark(
        "production-pass",
        (),
        SimpleNamespace(),
        output_dir=tmp_path,
        quality_evaluator=_evaluator(),
    )

    assert result is fake
    assert manifest.exists()


@pytest.mark.parametrize(
    "boundary",
    ["torch_module", "reference_loader", "pipeline_factory", "video_exporter"],
)
def test_production_wrapper_rejects_injected_runtime_kwargs_before_execution(
    monkeypatch, tmp_path, boundary
):
    called = False

    def unexpected_execution(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("benchmark must fail before rendering")

    monkeypatch.setattr(
        production_retry,
        "run_quality_retry_connected_gpu_benchmark",
        unexpected_execution,
    )

    with pytest.raises(
        ProductionGPUQualityRetryError,
        match="forbids injected runtime boundary kwargs",
    ):
        run_production_quality_retry_connected_gpu_benchmark(
            "production-injected",
            (),
            SimpleNamespace(),
            output_dir=tmp_path,
            quality_evaluator=_evaluator(),
            shot_executor_kwargs={boundary: object()},
        )

    assert called is False


def test_production_wrapper_allows_real_runtime_tuning_kwargs(monkeypatch, tmp_path):
    manifest = tmp_path / "benchmark.gpu-benchmark.json"
    manifest.write_text("verified evidence\n", encoding="utf-8")
    fake = _receipt(manifest, gpu=True, quality=True)
    captured = {}

    def fake_benchmark(*_args, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(
        production_retry,
        "run_quality_retry_connected_gpu_benchmark",
        fake_benchmark,
    )

    result = run_production_quality_retry_connected_gpu_benchmark(
        "production-tuned",
        (),
        SimpleNamespace(),
        output_dir=tmp_path,
        quality_evaluator=_evaluator(),
        shot_executor_kwargs={
            "estimated_model_vram_gb": 18.0,
            "prefer_bfloat16": False,
        },
    )

    assert result is fake
    assert captured["shot_executor_kwargs"] == {
        "estimated_model_vram_gb": 18.0,
        "prefer_bfloat16": False,
    }
