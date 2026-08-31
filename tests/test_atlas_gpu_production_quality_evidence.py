from types import SimpleNamespace

import pytest

import cineos.atlas.gpu_production_quality_retry as production_retry
from cineos.atlas.gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.sequence_quality import ArtifactMeasuredSequenceQualityEvaluator


def _evaluator():
    return ArtifactMeasuredSequenceQualityEvaluator(lambda *_args, **_kwargs: {})


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
