from types import SimpleNamespace

import pytest

import cineos.atlas.gpu_production_quality_retry as production_retry
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.sequence_quality import ArtifactMeasuredSequenceQualityEvaluator


class _UnusedAttestedObserver:
    """Valid production-observer shape for wrapper-only preflight tests."""

    production_measurement_evidence = True
    observer_id = "test-unused-production-observer/0.1"

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("observer must not run in wrapper preflight tests")


def _evaluator():
    return ArtifactMeasuredSequenceQualityEvaluator(_UnusedAttestedObserver())


def _requests():
    requests = []
    for index in range(5):
        request = NativeShotRequest(
            shot_id=f"shot-{index}",
            scene_id="scene-production-wrapper",
            camera={"movement": "tracking"},
            characters=[{"character_id": "lead"}],
            environment={"location": "street"},
            wardrobe=[],
            props=[],
            continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
            performance={"action": "walk"},
            approved_reference_ids=["lead-approved-reference"],
            deterministic_seed=6100 + index,
            renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
            metadata={"prompt": f"lead continues through shot {index}"},
        )
        request.refresh_hash()
        requests.append(request)
    return requests


def _patch_persistent(monkeypatch, capture=None):
    class FakePersistentExecutor:
        def __init__(self, *args, **kwargs):
            if capture is not None:
                capture["args"] = args
                capture["kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("fake persistent executor must not render")

    monkeypatch.setattr(
        production_retry,
        "PersistentGPUFoundationExecutor",
        FakePersistentExecutor,
    )


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
    _patch_persistent(monkeypatch)

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
            _requests(),
            WAN22_TI2V_5B_PROFILE,
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
    _patch_persistent(monkeypatch)

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
            _requests(),
            WAN22_TI2V_5B_PROFILE,
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
    _patch_persistent(monkeypatch)

    monkeypatch.setattr(
        production_retry,
        "run_quality_retry_connected_gpu_benchmark",
        lambda *_args, **_kwargs: fake,
    )

    result = run_production_quality_retry_connected_gpu_benchmark(
        "production-pass",
        _requests(),
        WAN22_TI2V_5B_PROFILE,
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
            _requests(),
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=_evaluator(),
            shot_executor_kwargs={boundary: object()},
        )

    assert called is False


def test_production_wrapper_allows_real_runtime_tuning_kwargs(monkeypatch, tmp_path):
    manifest = tmp_path / "benchmark.gpu-benchmark.json"
    manifest.write_text("verified evidence\n", encoding="utf-8")
    fake = _receipt(manifest, gpu=True, quality=True)
    inner = {}
    session = {}
    _patch_persistent(monkeypatch, session)

    def fake_benchmark(*_args, **kwargs):
        inner.update(kwargs)
        return fake

    monkeypatch.setattr(
        production_retry,
        "run_quality_retry_connected_gpu_benchmark",
        fake_benchmark,
    )

    result = run_production_quality_retry_connected_gpu_benchmark(
        "production-tuned",
        _requests(),
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        quality_evaluator=_evaluator(),
        shot_executor_kwargs={
            "estimated_model_vram_gb": 18.0,
            "prefer_bfloat16": False,
        },
    )

    assert result is fake
    assert session["kwargs"]["estimated_model_vram_gb"] == 18.0
    assert session["kwargs"]["prefer_bfloat16"] is False
    assert inner["shot_executor_kwargs"] is None
