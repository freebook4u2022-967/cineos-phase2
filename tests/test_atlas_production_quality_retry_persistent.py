import hashlib
import json
from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_foundation_smoke import execute_foundation_gpu_shot
from cineos.atlas.gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_continuity_identity import compose_continuity_identity_board
from cineos.atlas.production_multi_reference import ProductionReferenceBoardAdapter
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


def _connected_requests(reference_ids=("lead-approved-reference",)):
    requests = []
    for index in range(5):
        request = NativeShotRequest(
            shot_id=f"shot-{index}",
            scene_id="persistent-session-scene",
            camera={"movement": "tracking"},
            characters=[{"character_id": "lead"}],
            environment={"location": "street"},
            wardrobe=[],
            props=[],
            continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
            performance={"action": "walk"},
            approved_reference_ids=list(reference_ids),
            deterministic_seed=7100 + index,
            renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
            metadata={"prompt": f"lead continues through shot {index}"},
        )
        request.refresh_hash()
        requests.append(request)
    return requests


def _reference_manifest(tmp_path, reference_ids=("lead-approved-reference",)):
    references = []
    for index, reference_id in enumerate(reference_ids):
        asset = tmp_path / f"reference-{index}.png"
        asset.write_bytes(f"approved-reference-{index}".encode())
        references.append(
            {
                "reference_id": reference_id,
                "path": asset.name,
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "references.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "cineos-approved-reference-manifest/0.1",
                "references": references,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _install_fake_executor(monkeypatch, lifecycle):
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

    monkeypatch.setattr(
        "cineos.atlas.gpu_production_quality_retry.PersistentGPUFoundationExecutor",
        FakePersistentExecutor,
    )


def test_production_quality_retry_reuses_one_persistent_model_session(
    monkeypatch, tmp_path
):
    lifecycle = []
    captured = {}
    receipt = _production_receipt(tmp_path)
    _install_fake_executor(monkeypatch, lifecycle)

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return receipt

    monkeypatch.setattr(
        "cineos.atlas.gpu_production_quality_retry.run_quality_retry_connected_gpu_benchmark",
        fake_run,
    )

    profile = object()
    result = run_production_quality_retry_connected_gpu_benchmark(
        "connected-quality",
        _connected_requests(),
        profile,
        output_dir=tmp_path,
        quality_evaluator=_quality_evaluator(),
        reference_manifest=_reference_manifest(tmp_path),
        shot_executor_kwargs={
            "estimated_model_vram_gb": 21.5,
            "prefer_bfloat16": False,
        },
    )

    assert result is receipt
    assert [event[0] for event in lifecycle] == ["init", "enter", "exit"]
    executor = lifecycle[0][1]
    assert executor.profile is profile
    assert executor.kwargs["estimated_model_vram_gb"] == 21.5
    assert executor.kwargs["prefer_bfloat16"] is False
    assert executor.kwargs["reference_loader"].reference_ids == (
        "lead-approved-reference",
    )
    assert "multi_reference_adapter" not in executor.kwargs
    assert "continuity_identity_adapter" not in executor.kwargs
    assert captured["kwargs"]["shot_executor"] is executor
    assert captured["kwargs"]["shot_executor_kwargs"] is None


def test_production_quality_retry_can_select_first_party_identity_refresh(
    monkeypatch, tmp_path
):
    lifecycle = []
    receipt = _production_receipt(tmp_path)
    _install_fake_executor(monkeypatch, lifecycle)
    monkeypatch.setattr(
        "cineos.atlas.gpu_production_quality_retry.run_quality_retry_connected_gpu_benchmark",
        lambda *args, **kwargs: receipt,
    )

    run_production_quality_retry_connected_gpu_benchmark(
        "identity-refresh-candidate",
        _connected_requests(),
        object(),
        output_dir=tmp_path,
        quality_evaluator=_quality_evaluator(),
        reference_manifest=_reference_manifest(tmp_path),
        continuity_identity_refresh=True,
    )

    executor = lifecycle[0][1]
    assert executor.kwargs["continuity_identity_adapter"] is (
        compose_continuity_identity_board
    )


def test_production_quality_retry_constructs_audited_multi_reference_adapter(
    monkeypatch, tmp_path
):
    lifecycle = []
    receipt = _production_receipt(tmp_path)
    _install_fake_executor(monkeypatch, lifecycle)
    monkeypatch.setattr(
        "cineos.atlas.gpu_production_quality_retry.run_quality_retry_connected_gpu_benchmark",
        lambda *args, **kwargs: receipt,
    )
    reference_ids = ("lead-approved-reference", "partner-approved-reference")

    run_production_quality_retry_connected_gpu_benchmark(
        "multi-character-quality",
        _connected_requests(reference_ids),
        object(),
        output_dir=tmp_path,
        quality_evaluator=_quality_evaluator(),
        reference_manifest=_reference_manifest(tmp_path, reference_ids),
    )

    executor = lifecycle[0][1]
    assert isinstance(
        executor.kwargs["multi_reference_adapter"],
        ProductionReferenceBoardAdapter,
    )


def test_production_quality_retry_requires_reference_manifest_before_session(
    monkeypatch, tmp_path
):
    lifecycle = []
    _install_fake_executor(monkeypatch, lifecycle)

    with pytest.raises(
        ProductionGPUQualityRetryError,
        match="hash-pinned reference manifest",
    ):
        run_production_quality_retry_connected_gpu_benchmark(
            "connected-quality",
            _connected_requests(),
            object(),
            output_dir=tmp_path,
            quality_evaluator=_quality_evaluator(),
        )

    assert lifecycle == []


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
        match="forbids injected runtime boundary kwargs: continuity_identity_adapter",
    ):
        run_production_quality_retry_connected_gpu_benchmark(
            "connected-quality",
            (),
            object(),
            output_dir=tmp_path,
            quality_evaluator=_quality_evaluator(),
            shot_executor=execute_foundation_gpu_shot,
            shot_executor_kwargs={"continuity_identity_adapter": object()},
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
